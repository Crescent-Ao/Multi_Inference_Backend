from .inference_engine import InferenceEngine
import numpy as np
import cv2
import torch
from torch.cuda import amp
from typing import Dict, Any
from pathlib import Path
import pandas as pd
from collections import namedtuple, OrderedDict
from utils.seg_util import generate_segment_colors
from loguru import logger
import torch.nn.functional as F
import re
import tensorrt as trt
from utils.util import *
class SemanticSegmentation(InferenceEngine):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.config = config
        self.load_model()
        
    def onnx_export(self) -> None:
        if self.pt:
            from utils.seg_util import load_checkpoint
            import onnx
            from io import BytesIO
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = load_checkpoint(self.config.get("model_path"), num_classes=self.config.get("num_classes"), map_location=device)
            model.to(device)
            model.eval()
            dummy_input = torch.randn(self.config.get('onnx_export')['batch_size'], 3, self.config.get('img_size')[0], self.config.get('img_size')[1])
            if self.config.get('precision') == "FP16":
                dummy_input = dummy_input.half()
                model = model.half()
            else:
                dummy_input = dummy_input.float()
                model = model.float()
            dummy_input = dummy_input.to(device)
            dummy_out = model(dummy_input)
            dynamic_axes = None
            import ipdb
            ipdb.set_trace()
            if self.config.get('onnx_export')['dynamic_batch']:
                dynamic_axes = {
                    self.config.get('onnx_export')['input_names'][0] :{
                        0:'batch',
                    },}
                output_axes = {
                        self.config.get('onnx_export')['output_names'][0]: {0: 'batch'},
                    }
                dynamic_axes.update(output_axes)
            try:
                logger.info('\nStarting to export ONNX...')
                import ipdb
                ipdb.set_trace()
                export_file = self.config.get("model_path").replace('.pt', '.onnx')
                with BytesIO() as f:
                    torch.onnx.export(model, dummy_input, f, verbose=False, opset_version=18,
                              training=torch.onnx.TrainingMode.EVAL,
                              do_constant_folding=True,
                              input_names=self.config.get('onnx_export')["input_names"],
                              output_names=self.config.get('onnx_export')["output_names"],
                              dynamic_axes=dynamic_axes)
                    f.seek(0)
                    onnx_model = onnx.load(f)
                    onnx.checker.check_model(onnx_model)
                if self.config.get('onnx_export')['simplify']:
                    try:
                        import onnxsim
                        logger.info('\nStarting to simplify ONNX...')
                        onnx_model, check = onnxsim.simplify(onnx_model)
                        assert check, 'assert check failed'
                    except Exception as e:
                        logger.info(f'Simplifier failure: {e}')
                onnx.save(onnx_model, export_file)
                logger.info(f'ONNX export success, saved as {export_file}')
            except Exception as e:
                logger.error(f'Export ONNX failed: {e}')   
        else:
            logger.info("The ONNX export must use pt model")
            raise ValueError("Invalid model format") 
    def allocate_buffers(self,dynamic_shapes=[]):
        inputs = []
        outputs = []
        bindings = []

        for binding in self.model:
            dims = self.model.get_binding_shape(binding)
            print(dims)
            if dims[0] == -1:
                assert(len(dynamic_shapes) > 0)
                dims[0] = dynamic_shapes[0]
            size = trt.volume(dims) * 1
            print(dims)
            print(size)
            dtype = trt.nptype(self.model.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if self.model.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs,bindings

    def load_model(self):
        path = self.config.get("model_path")
        pt, jt, onnx, opvino, trt, aclite = self._model_type(path)
        self.fp16 = pt or jt or onnx or opvino or trt 
        self.fp16 = False if self.config.get("precision") == "FP32" else True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if pt:
            from utils.seg_util import load_checkpoint
            model = load_checkpoint(path, num_classes=self.config.get("num_classes"), map_location=self.device)
            if self.device.type != "cpu":
                img_size = self.config.get("img_size")
                model(torch.zeros(1, 3, img_size[0], img_size[1]).to(self.device).type_as(next(model.parameters())))
            model.half() if self.fp16 else model.float()
            
        elif jt:
            extra_files = {"config.txt": ""}
            model = torch.jit.load(path, _extra_files=extra_files, map_location=self.device)
            model.half() if self.fp16 else model.float()
            
        elif onnx:
            import onnxruntime
            providers = ['CPUExecutionProvider']
            if self.device.type == "cuda":
                providers.append('CUDAExecutionProvider')
            model = onnxruntime.InferenceSession(path, providers=providers)
            output_names = [output.name for output in model.get_outputs()]
            
        elif trt:
            import tensorrt as trt  # noqa https://developer.nvidia.com/nvidia-tensorrt-download
            check_version(trt.__version__, ">=7.0.0", hard=True)
            check_version(trt.__version__, "!=10.1.0", msg="https://github.com/ultralytics/ultralytics/pull/14239")
            if self.device.type == "cpu":
                self.device = torch.device("cuda:0")
            Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
            logger = trt.Logger(trt.Logger.INFO)
            trt.init_libnvinfer_plugins(logger, '')
            # Read file
            path = self.config.get("model_path")
            with open(path, "rb") as f, trt.Runtime(logger) as runtime:
                model = runtime.deserialize_cuda_engine(f.read())  # read engine
                context = model.create_execution_context()
            

            bindings = OrderedDict()
            output_names = []
            fp16 = False  # default updated below
            dynamic = False
            is_trt10 = not hasattr(model, "num_bindings")
            num = range(model.num_io_tensors) if is_trt10 else range(model.num_bindings)
            from loguru import logger
            for i in num:
                if is_trt10:
                    name = model.get_tensor_name(i)
                    dtype = trt.nptype(model.get_tensor_dtype(name))
                    is_input = model.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                    if is_input:
                        if -1 in tuple(model.get_tensor_shape(name)):
                            dynamic = True
                            context.set_input_shape(name, tuple(model.get_tensor_profile_shape(name, 0)[1]))
                        if dtype == np.float16:
                            fp16 = True
                    else:
                        output_names.append(name)
                    import ipdb
                    ipdb.set_trace()
                    shape = tuple(context.get_tensor_shape(name))
                    logger.info(f"name: {name}, shape: {shape}")
                else:  # TensorRT < 10.0
                    name = model.get_binding_name(i)
                    dtype = trt.nptype(model.get_binding_dtype(i))
                    is_input = model.binding_is_input(i)
                    if model.binding_is_input(i):
                        if -1 in tuple(model.get_binding_shape(i)):  # dynamic
                            dynamic = True
                            context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[1]))
                        if dtype == np.float16:
                            fp16 = True
                    else:
                        output_names.append(name)
                    shape = tuple(context.get_binding_shape(i))
                im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(self.device)
                bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
            binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
            batch_size = bindings["images"].shape[0]  # if dynamic, this is inst
            self.dynamic = False
            self.context = context
            self.model = model
            self.bindings = bindings
            self.output_names = output_names
            self.binding_addrs = binding_addrs

            
           



           
        elif opvino:
            import openvino as ov
            model = ov.Core().read_model(path)
            compiled_model = ov.Core().compile_model(model, self.device)
            output_names = compiled_model.outputs
            
        elif aclite:
            from utils.acl_util import AclLiteResource
            from acllite_model import AclLiteModel
            acl_resource = AclLiteResource()
            acl_resource.init()
            model = AclLiteModel(path)
            
        else:
            raise ValueError("Invalid model format")
            
        self.__dict__.update(locals())

    @staticmethod
    def export_formats():
        x = [
            ["PyTorch", "-", ".pt", True, True],
            ["TorchScript", "torchscript", ".torchscript", True, True],
            ["ONNX", "onnx", ".onnx", True, True],
            ["OpenVINO", "openvino", "_openvino_model", True, False],
            ["TensorRT", "engine", ".engine", False, True],
            ["ACLite", "cann", ".om", False, False]
        ]
        return pd.DataFrame(x, columns=["Format", "Argument", "Suffix", "CPU", "GPU"])

    def _model_type(self, path):
        sf = list(self.export_formats()["Suffix"])
        types = [s in Path(path).name for s in sf]
        return types

    def preprocess(self, input_data: Any) -> Any:
        """前处理: 图像预处理和标准化"""
        if isinstance(input_data, str):
            img = cv2.imread(input_data)
        elif isinstance(input_data, np.ndarray):
            img = input_data.copy()
        else:
            img = np.array(input_data)
            
        img_src = img.copy()
        
        # Resize
        if img.shape[:2] != self.config.get("img_size"):
            img = cv2.resize(img, self.config.get("img_size")[::-1], interpolation=cv2.INTER_LINEAR)
        
        # Convert to float32 and BGR to RGB
        img = img.astype(np.float32)[:, :, ::-1]
        
        # Normalize
        img = img / 255.0
        img -= np.array([0.485, 0.456, 0.406])
        img /= np.array([0.229, 0.224, 0.225])
        
        # HWC to CHW
        img = img.transpose((2, 0, 1)).copy()
        img = torch.from_numpy(img).unsqueeze(0)
        
        return img.to(self.device), img_src
    
    def inference(self, preprocessed_data: Any) -> Any:
        if self.fp16 and preprocessed_data.dtype != torch.float16:
            preprocessed_data = preprocessed_data.half()
            
        if self.pt:
            if self.device == torch.device("cpu"):
                preprocessed_data = preprocessed_data.cpu()
            y = self.model(preprocessed_data)
            import ipdb
            ipdb.set_trace()
        elif self.trt:
            try:
                if self.dynamic and preprocessed_data.shape != self.bindings["images"].shape:
                    if self.is_trt10:
                        self.context.set_input_shape("images", preprocessed_data.shape)
                        self.bindings["images"] = self.bindings["images"]._replace(shape=preprocessed_data.shape)
                        for name in self.output_names:
                            self.bindings[name].data.resize_(tuple(self.context.get_tensor_shape(name)))
                    else:
                        i = self.model.get_binding_index("images")
                        self.context.set_binding_shape(i, preprocessed_data.shape)
                        self.bindings["images"] = self.bindings["images"]._replace(shape=preprocessed_data.shape)
                        for name in self.output_names:
                            i = self.model.get_binding_index(name)
                            self.bindings[name].data.resize_(tuple(self.context.get_binding_shape(i)))

                s = self.bindings["images"].shape
                assert preprocessed_data.shape == s, f"input size {preprocessed_data.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
                self.binding_addrs["images"] = int(preprocessed_data.data_ptr())
                self.context.execute_v2(list(self.binding_addrs.values()))
                y = [self.bindings[x].data for x in sorted(self.output_names)]
            finally:
                    print("end")
        elif self.jt:
            if self.fp16:
                preprocessed_data = preprocessed_data.half()
            y = self.model(preprocessed_data)
            
        elif self.onnx:
            print(preprocessed_data.shape)
            preprocessed_data = preprocessed_data.cpu().numpy()
            if self.fp16:
                preprocessed_data = preprocessed_data.astype(np.float16)
            import ipdb
            ipdb.set_trace()
            y = self.model.run(self.output_names, {self.model.get_inputs()[0].name: preprocessed_data})
        elif self.opvino:
            if self.fp16:
                preprocessed_data = preprocessed_data.astype(np.float16)
            y = self.compiled_model(preprocessed_data)[self.output_names[0]]
            
        elif self.aclite:
            preprocessed_data = np.array(preprocessed_data)
            y = self.model.execute([preprocessed_data,])
            
        else:
            raise ValueError("Invalid model format")
            
        if isinstance(y, (list, tuple)):
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)

    def from_numpy(self, x):
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x
    def to_rgb(self, mask):
        h, w = mask.shape[0], mask.shape[1]
        mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
        mask_convert = mask[np.newaxis, :, :]

        import seaborn as sns
        palette = sns.color_palette("hls", len(cfg.CLASSES))
        colors = []
        for i in range(len(cfg.CLASSES)):
            color = [int(palette[i][2] * 255), int(palette[i][1] * 255), int(palette[i][0] * 255)]
            colors.append(color)
        # Black: Background
        mask_rgb[np.all(mask_convert == 4, axis=0)] = [0, 0, 0]
        # Blue: OxyPond
        mask_rgb[np.all(mask_convert == 0, axis=0)] = colors[0]#[255, 204, 0]
        # Red: Pond
        mask_rgb[np.all(mask_convert == 1, axis=0)] = colors[1]#[0, 0, 255]
        # Green: DryPand
        mask_rgb[np.all(mask_convert == 2, axis=0)] = colors[2]#[0, 255, 64]
        # Yellow: CagePond
        mask_rgb[np.all(mask_convert == 3, axis=0)] = colors[3]#[0, 204, 255]
        return mask_rgb
    def postprocess(self, inference_output: Any) -> Any:
        """后处理: 将预测结果转换为分割图像"""
        # Resize to original input size
        pred = F.interpolate(inference_output, 
                            size=self.config.get("img_size"),
                            mode='bilinear', 
                            align_corners=True)
        
        # Get class predictions
        pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()
        
        # Convert to RGB mask
        import seaborn as sns
        palette = sns.color_palette("hls", self.config.get("num_classes"))
        h, w = pred.shape
        mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
        colors = []
        for i in range(self.config.get("num_classes")):
            color = [int(palette[i][2] * 255), int(palette[i][1] * 255), int(palette[i][0] * 255)]
            colors.append(color)
        
        # Apply color map
        for i, color in enumerate(colors):
            mask_rgb[pred == i] = color
        
        return mask_rgb


    def run(self, input_data: Any) -> Any:
        # 完整的推理流程
        img, img_src = self.preprocess(input_data)
        pred = self.inference(img)
        import ipdb
        ipdb.set_trace()
        mask = self.postprocess(pred)
        
        # Resize back to original image size
        if mask.shape[:2] != img_src.shape[:2]:
            mask = cv2.resize(mask, 
                             (img_src.shape[1], img_src.shape[0]), 
                             interpolation=cv2.INTER_NEAREST)
        
        return mask
