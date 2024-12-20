import json
import platform
from collections import OrderedDict, namedtuple
# from copy import copy
from pathlib import Path
from urllib.parse import urlparse
import os
import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda import amp
from yolov6.utils.general import download_ckpt
from yolov6.utils.checkpoint import load_checkpoint

#NOTE 返回的是Tensor 信息

class DetectMultiBackend(nn.Module):
    # YOLOv5 MultiBackend class for python inference on various backends
    def __init__(self, weights="yolov5s.pt", task = 'eval',device=torch.device("cuda:0"), dnn=False, data=None, fp16=False, fuse=True):
        # Usage:
        #   PyTorch:              weights = *.pt
        #   TorchScript:                    *.torchscript
        #   ONNX Runtime:                   *.onnx
        #   ONNX OpenCV DNN:                *.onnx --dnn
        #   OpenVINO:                       *.xml
        #   CoreML:                         *.mlmodel
        #   TensorRT:                       *.engine
        #   TensorFlow SavedModel:          *_saved_model
        #   TensorFlow GraphDef:            *.pb
        #   TensorFlow Lite:                *.tflite
        #   TensorFlow Edge TPU:            *_edgetpu.tflite
        #   PaddlePaddle:                   *_paddle_model
        # from models.experimental import (  # scoped to avoid circular import
        #     attempt_download, attempt_load)

        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        # pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle, triton = self._model_type(w)
        self.fp16_flag = fp16
        pt, jit, onnx, engine = self._model_type(w)
        fp16 &= pt or jit or onnx or engine  # FP16
        # nhwc = coreml or saved_model or pb or tflite or edgetpu  # BHWC formats (vs torch BCWH)
        stride = 32  # default stride
        cuda = torch.cuda.is_available() and device.type != "cpu"  # use CUDA
        # if not (pt or triton):
        #     w = attempt_download(w)  # download if not local
        if pt:  # PyTorch
            
            # model = attempt_load(weights if isinstance(weights, list) else w, device=device, inplace=True, fuse=fuse)
            # stride = max(int(model.stride.max()), 32)  # model stride
            # names = model.module.names if hasattr(model, "module") else model.names  # get class names
            # model = torch.load()
            # NOTE 切换到推理的模式下
            if task != "train":
                if not os.path.exists(weights):
                    download_ckpt(weights)
            model = load_checkpoint(weights, map_location=device)
            self.stride = int(model.stride.max())
            if device.type != "cpu":
                img_size = 800
                model(torch.zeros(1, 3, img_size, img_size).to(device).type_as(next(model.parameters())))

            # switch to deploy
            from yolov6.layers.common import RepVGGBlock

            for layer in model.modules():
                if isinstance(layer, RepVGGBlock):
                    layer.switch_to_deploy()
            model.half() if self.fp16_flag else model.float()

            # ckpt = torch.load(w, map_location="cpu")
            # print(ckpt["best_fitness"])
            # ckpt = (ckpt.get('ema') or ckpt['model']).to(device).float()
            # model = ckpt.eval()
            # model.half() if fp16 else model.float()
            # self.model = model  # explicitly assign for to(), cpu(), cuda(), half()
        elif jit:  # TorchScript
            # LOGGER.info(f'Loading {w} for TorchScript inference...')
            extra_files = {"config.txt": ""}  # model metadata
            model = torch.jit.load(w, _extra_files=extra_files, map_location=device)
            model.half() if fp16 else model.float()
            if extra_files["config.txt"]:  # load metadata dict
                d = json.loads(
                    extra_files["config.txt"],
                    object_hook=lambda d: {int(k) if k.isdigit() else k: v for k, v in d.items()},
                )
                stride, names = int(d["stride"]), d["names"]
        # elif dnn:  # ONNX OpenCV DNN
        #     LOGGER.info(f'Loading {w} for ONNX OpenCV DNN inference...')
        #     check_requirements('opencv-python>=4.5.4')
        #     net = cv2.dnn.readNetFromONNX(w)
        elif onnx:  # ONNX Runtime
            # LOGGER.info(f'Loading {w} for ONNX Runtime inference...')
            # check_requirements(('onnx', 'onnxruntime-gpu' if cuda else 'onnxruntime'))
            import onnxruntime

            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if cuda else ["CPUExecutionProvider"]
            session = onnxruntime.InferenceSession(w, providers=providers)
            output_names = [x.name for x in session.get_outputs()]
            meta = session.get_modelmeta().custom_metadata_map  # metadata
            if "stride" in meta:
                stride, names = int(meta["stride"]), eval(meta["names"])
        # elif xml:  # OpenVINO
        #     LOGGER.info(f'Loading {w} for OpenVINO inference...')
        #     check_requirements('openvino')  # requires openvino-dev: https://pypi.org/project/openvino-dev/
        #     from openvino.runtime import Core, Layout, get_batch
        #     ie = Core()
        #     if not Path(w).is_file():  # if not *.xml
        #         w = next(Path(w).glob('*.xml'))  # get *.xml file from *_openvino_model dir
        #     network = ie.read_model(model=w, weights=Path(w).with_suffix('.bin'))
        #     if network.get_parameters()[0].get_layout().empty:
        #         network.get_parameters()[0].set_layout(Layout("NCHW"))
        #     batch_dim = get_batch(network)
        #     if batch_dim.is_static:
        #         batch_size = batch_dim.get_length()
        #     executable_network = ie.compile_model(network, device_name="CPU")  # device_name="MYRIAD" for Intel NCS2
        #     stride, names = self._load_metadata(Path(w).with_suffix('.yaml'))  # load metadata
        elif engine:  # TensorRT
            # LOGGER.info(f'Loading {w} for TensorRT inference...')
            import tensorrt as trt  # https://developer.nvidia.com/nvidia-tensorrt-download

            # check_version(trt.__version__, '7.0.0', hard=True)  # require tensorrt>=7.0.0
            if device.type == "cpu":
                device = torch.device("cuda:0")
            Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
            logger = trt.Logger(trt.Logger.INFO)
            trt.init_libnvinfer_plugins(logger, namespace="")
            with open(w, "rb") as f, trt.Runtime(logger) as runtime:
                model = runtime.deserialize_cuda_engine(f.read())
            context = model.create_execution_context()
            bindings = OrderedDict()
            output_names = []
            fp16 = False  # default updated below
            # import ipdb
            # ipdb.set_trace()
            dynamic = False
            for i in range(model.num_bindings):
                name = model.get_binding_name(i)
                print(name)
                dtype = trt.nptype(model.get_binding_dtype(i))
                # dtype = np.float16
                if model.binding_is_input(i):
                    if -1 in tuple(model.get_binding_shape(i)):  # dynamic
                        dynamic = True
                        context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[2]))
                    if dtype == np.float16:
                        fp16 = True
                     
                else:  # output
                    output_names.append(name)
                shape = tuple(context.get_binding_shape(i))
                
                im = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(device)
                bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))

            # print(fp16)

            binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
            # NOTE 需要对应
            batch_size = bindings["images"].shape[0]  # if dynamic, this is instead max batch size
            print("finished")

        self.__dict__.update(locals())  # assign all variables to self
    @staticmethod
    def torch_dtype_from_trt(dtype):
        import tensorrt as trt
        if dtype == trt.bool:
            return torch.bool
        elif dtype == trt.int8:
            return torch.int8
        elif dtype == trt.int32:
            return torch.int32
        elif dtype == trt.float16:
            return torch.float16
        elif dtype == trt.float32:
            return torch.float32
        else:
            raise TypeError('%s is not supported by torch' % dtype)
    def forward(self, im, augment=False, visualize=False):
        # YOLOv5 MultiBackend inference
        b, ch, h, w = im.shape  # batch, channel, height, width

        if self.fp16 and im.dtype != torch.float16:
            im = im.half()  # to FP16
            

        # if self.nhwc:
        #     im = im.permute(0, 2, 3, 1)  # torch BCHW to numpy BHWC shape(1,320,192,3)

        if self.pt:  # PyTorch
            # y = self.model(im, augment=augment, visualize=visualize) if augment or visualize else self.model(im)
            if self.device == torch.device("cpu"):
                im = im.cpu()
            y = self.model(im)[0]
        elif self.jit:  # TorchScript
            
            im = im.half()
            y = self.model(im)
        # elif self.dnn:  # ONNX OpenCV DNN
        #     im = im.cpu().numpy()  # torch to numpy
        #     self.net.setInput(im)
        #     y = self.net.forward()
        elif self.onnx:  # ONNX Runtime
            
            # torch to numpy
            if self.fp16_flag:
                im = im.astype(np.float16)
            y = self.session.run(self.output_names, {self.session.get_inputs()[0].name: im})
        # elif self.xml:  # OpenVINO
        #     im = im.cpu().numpy()  # FP32
        #     y = list(self.executable_network([im]).values())
        elif self.engine:  # TensorRT
            if self.dynamic and im.shape != self.bindings["images"].shape:
                i = self.model.get_binding_index("images")
                self.context.set_binding_shape(i, im.shape)  # reshape if dynamic
                self.bindings["images"] = self.bindings["images"]._replace(shape=im.shape)
                for name in self.output_names:
                    i = self.model.get_binding_index(name)
                    self.bindings[name].data.resize_(tuple(self.context.get_binding_shape(i)))
            s = self.bindings["images"].shape
            assert im.shape == s, f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"           
            self.binding_addrs["images"] = int(im.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            # y = [self.bindings[x].data for x in sorted(self.output_names)]
            y = self.bindings["outputs"].data
           
        if isinstance(y, (list, tuple)):
            # import ipdb
            # ipdb.set_trace()
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)

    def from_numpy(self, x):
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warmup(self, imgsz=(1, 3, 800, 800)):
        # Warmup model by running inference once
        warmup_types = self.pt, self.jit, self.onnx, self.engine, self.saved_model, self.pb, self.triton
        if any(warmup_types) and (self.device.type != "cpu" or self.triton):
            im = torch.empty(*imgsz, dtype=torch.half if self.fp16 else torch.float, device=self.device)  # input
            for _ in range(2 if self.jit else 1):  #
                self.forward(im)  # warmup

    @staticmethod
    def export_formats():
        # YOLOv5 export formats
        x = [
            ['PyTorch', '-', '.pt', True, True],
            ['TorchScript', 'torchscript', '.torchscript', True, True],
            ['ONNX', 'onnx', '.onnx', True, True],
            # ['OpenVINO', 'openvino', '_openvino_model', True, False],
            ['TensorRT', 'engine', '.engine', False, True],
            # ['CoreML', 'coreml', '.mlmodel', True, False],
            # ['TensorFlow SavedModel', 'saved_model', '_saved_model', True, True],
            # ['TensorFlow GraphDef', 'pb', '.pb', True, True],
            # ['TensorFlow Lite', 'tflite', '.tflite', True, False],
            # ['TensorFlow Edge TPU', 'edgetpu', '_edgetpu.tflite', False, False],
            # ['TensorFlow.js', 'tfjs', '_web_model', False, False],
            # ['PaddlePaddle', 'paddle', '_paddle_model', True, True],]
        ]
        return pd.DataFrame(x, columns=['Format', 'Argument', 'Suffix', 'CPU', 'GPU'])

    # @staticmethod
    def _model_type(self, p="path/to/model.pt"):
        # return Path(p).suffix
        # Return model type from model path, i.e. path='path/to/model.onnx' -> type=onnx
        # types = [pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle]
        # from export import export_formats
        # from utils.downloads import is_url

        sf = list(self.export_formats().Suffix)  # export suffixes
        # if not is_url(p, check=False):
        #     check_suffix(p, sf)  # checks
        # url = urlparse(p)  # if url may be Triton inference server
        types = [s in Path(p).name for s in sf]
        # types[8] &= not types[9]  # tflite &= not edgetpu
        # triton = not any(types) and all([any(s in url.scheme for s in ["http", "grpc"]), url.netloc])
        return types

    @staticmethod
    def _load_metadata(f=Path("path/to/meta.yaml")):
        # Load metadata from meta.yaml if it exists
        if f.exists():
            d = yaml_load(f)
            return d["stride"], d["names"]  # assign stride, names
        return None, None



