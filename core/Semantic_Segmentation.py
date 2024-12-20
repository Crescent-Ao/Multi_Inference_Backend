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
                    torch.onnx.export(model, dummy_input, f, verbose=False, opset_version=13,
                              training=torch.onnx.TrainingMode.EVAL,
                              input_names=["images"],
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
     
    def load_model(self):
        path = self.config.get("model_path")
        pt, jt, onnx, opvino, trt, aclite = self._model_type(path)
        self.fp16 = pt or jt or onnx or opvino or trt 
        self.fp16 = False if self.config.get("precision") == "FP32" else True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        import ipdb; ipdb.set_trace()
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
            import tensorrt as trt
            if self.device == "cpu":
               self.device = torch.device("cuda:0")
            Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
            logger = trt.Logger(trt.Logger.WARNING)
            trt.init_libnvinfer_plugins(logger, "")
            with open(path, "rb") as f, trt.Runtime(logger) as runtime:
                model = runtime.deserialize_cuda_engine(f.read())
            context = model.create_execution_context()
            bindings = OrderedDict()
            output_names = []
            dynamic = False
            fp16 = False
            for i in range(model.num_bindings):
                name = model.get_binding_name(i)
                dtype = trt.nptype(model.get_binding_dtype(i))
                if model.binding_is_input(i):
                    if -1 in tuple(model.get_binding_shape(i)):
                        dynamic = True
                        context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[2]))
                    if dtype == np.float16:
                        fp16 = True
                else:
                    output_names.append(name)
                shape = tuple(context.get_binding_shape(i))
                im = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(self.device)
                bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
            binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
            
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
        if isinstance(input_data, str):
            img = cv2.imread(input_data)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(input_data, np.ndarray):
            if input_data.shape[-1] == 3:
                img = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
            else:
                img = input_data
        else:
            img = np.array(input_data)
            
        img_src = img.copy()
        
        # resize到指定大小
        if img.shape[:2] != self.config.get("img_size"):
            img = cv2.resize(img, self.config.get("img_size"), interpolation=cv2.INTER_LINEAR)
            
        # 标准化
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img.astype(np.float32) / 255.0
        img = (img - mean) / std
        #cv2.normalize(img, img, 0, 255, cv2.NORM_L2)#cv2.NORM_MINMAX

        # 转换为tensor并添加batch维度
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        img = img.unsqueeze(0)
        return img.to(self.device), img_src
    
    def inference(self, preprocessed_data: Any) -> Any:
        if self.fp16 and preprocessed_data.dtype != torch.float16:
            preprocessed_data = preprocessed_data.half()
            
        if self.pt:
            if self.device == torch.device("cpu"):
                preprocessed_data = preprocessed_data.cpu()
            y = self.model(preprocessed_data)
            
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
        elif self.trt:
            if self.dynamic and preprocessed_data.shape != self.bindings["images"].shape:
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
        """后处理:将预测结果转换为可视化的RGB图像
        
        Args:
            inference_output: 模型输出的预测结果
            
        Returns:
            mask_rgb: 可视化的RGB分割图,shape为(H,W,3)
        """
        # 获取预测类别
        preds = torch.nn.Softmax(dim=1)(inference_output).argmax(dim=1)
        pred = preds[0].cpu().numpy()
        
        # 转换为RGB掩码
        h, w = pred.shape
        mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
        pred = pred[np.newaxis, :, :]  # 添加通道维度 [1,H,W]
        
        # 获取类别名称和颜色映射
        class_names = self.config.get("names", [])
        import seaborn as sns
        palette = sns.color_palette("hls", len(class_names))
        colors = []
        for i in range(len(class_names)):
            colors.append([
                int(palette[i][2] * 255),  # R
                int(palette[i][1] * 255),  # G
                int(palette[i][0] * 255)   # B
            ])
        
        # 为每个类别赋予对应颜色
        for i, color in enumerate(colors):
            mask_rgb[np.all(pred == i, axis=0)] = color
            
        return mask_rgb
    
    def run(self, input_data: Any) -> Any:
        img, img_src = self.preprocess(input_data)
        pred = self.inference(img)
        mask = self.postprocess(pred)
        
        # Resize回原图大小
        if mask.shape[:2] != img_src.shape[:2]:
            mask = cv2.resize(mask, (img_src.shape[1], img_src.shape[0]), interpolation=cv2.INTER_NEAREST)
            
        return mask
