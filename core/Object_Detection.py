# -*- coding: utf-8 -*-

import ipdb.stdout
from .inference_engine import InferenceEngine
import numpy as np
import cv2 as cv2
import torch
from PIL import Image
from torch.cuda import amp
from typing import Dict, Any
from pathlib import Path
import pandas as pd
import json
from collections import namedtuple,OrderedDict
from utils.obb_util import letterbox,plot_box_and_label,generate_colors,rescale,dist2bbox,generate_anchors
from loguru import logger
from io import BytesIO



class ObjectDetection(InferenceEngine):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.config = config
        self.load_model()
    def onnx_export(self) -> None:
        if self.pt:
            from utils.obb_util import load_checkpoint
            import onnx 
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = load_checkpoint(self.config.get("model_path"), map_location=device)
            from utils.obb_layers import RepVGGBlock
            for layer in model.modules():
                if isinstance(layer,RepVGGBlock):
                    layer.switch_to_deploy()  
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

    def load_model(self):
        path = self.config.get("model_path")
        pt,jt,onnx,opvino,trt,aclite = self._model_type(path)
        self.fp16 = pt or jt or onnx or opvino or trt 
        self.fp16 = False if self.config.get("precision") == "FP32" else True
        # 我想定义一个device 作为整个类的成员
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if pt:
            from utils.obb_util import load_checkpoint
            model = load_checkpoint(path, map_location=self.device)
            if self.device.type != "cpu":
                img_size = self.config.get("img_size")
                print(self.device)
                if isinstance(img_size, (list, tuple)):
                    h, w = img_size
                    model(torch.zeros(1, 3, h, w).to(self.device).type_as(next(model.parameters())))
                else:
                    model(torch.zeros(1, 3, img_size, img_size).to(self.device).type_as(next(model.parameters())))
            from utils.obb_layers import RepVGGBlock
            for layer in model.modules():
                if isinstance(layer,RepVGGBlock):
                    layer.switch_to_deploy()
            model.half() if self.fp16 else model.float()
        elif jt:
            extra_files = {"config.txt": ""}  # model metadata
            model = torch.jit.load(path, _extra_files=extra_files, map_location=self.device)
            model.half() if self.fp16 else model.float()
            if extra_files["config.txt"]:  # load metadata dict
                d = json.loads(
                    extra_files["config.txt"],
                    object_hook=lambda d: {int(k) if k.isdigit() else k: v for k, v in d.items()},
                )
        elif onnx:
            import onnxruntime
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.device.type == "cuda" else ["CPUExecutionProvider"]
            model = onnxruntime.InferenceSession(path,providers=providers)
            output_names = [output.name for output in model.get_outputs()]
            print("onnx model loaded successfully")
        elif trt:
            import tensorrt as trt
            if self.device == "cpu":
               self.device = torch.device("cuda:0")
            Binding = namedtuple("Binding",("name","dtype","shape","data","ptr"))
            logger = trt.Logger(trt.Logger.WARNING)
            trt.init_libnvinfer_plugins(logger,"")
            with open(path,"rb") as f, trt.Runtime(logger) as runtime:
                model = runtime.deserialize_cuda_engine(f.read())
            context = model.create_execution_context()
            bindings = OrderedDict()
            output_names = []
            dynamic=False
            fp16=False
            for i in range(model.num_bindings):
                name = model.get_binding_name(i)
                dtype = trt.nptype(model.get_binding_dtype(i))
                if model.binding_is_input(i):
                    if -1 in tuple(model.get_binding_shape(i)):  # dynamic
                        dynamic = True
                        context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[2]))
                    if dtype == np.float16:
                        fp16 = True
                else:
                    output_names.append(name)
                shape = tuple(context.get_binding_shape(i))
                im = torch.from_numpy(np.empty(shape,dtype=np.dtype(dtype))).to(self.device)
                bindings[name] = Binding(name,dtype,shape,im,int(im.data_ptr()))
            binding_addrs = OrderedDict((n,d.ptr) for n,d in bindings.items())
        elif opvino:
            import openvino as ov
            model = ov.Core().read_model(path)
            compiled_model = ov.Core().compile_model(model, self.device)
            output_names = compiled_model.outputs
        elif aclite:
            from utils.acl_util import AclLiteResource
            from acllite_model import AclLiteModel
            from acllite_resource import _ResourceList

            
            # 由于aclite后处理分离了一部分需要单独做一部分
            anchor_points, stride_tensor = generate_anchors(None,[8, 16, 32], device=self.device,is_eval=True)
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
            ["ACLite","cann",".om",False,False]
        ]
        return pd.DataFrame(x, columns=["Format","Argument","Suffix","CPU","GPU"])
    def _model_type(self,path):
        
        sf = list(self.export_formats()["Suffix"])
        
        types = [s in Path(path).name for s in sf]
        # types[8] &= not types[9]  # tflite &= not edgetpu
        # triton = not any(types) and all([any(s in url.scheme for s in ["http", "grpc"]), url.netloc])
        return types
    
    def inference(self, preprocessed_data: Any) -> Any:
        
        b,ch,h,w = preprocessed_data.shape
        if self.fp16 and preprocessed_data.dtype != torch.float16:
            preprocessed_data = preprocessed_data.half()
        if self.pt:
            if self.device==torch.device("cpu"):
                preprocessed_data = preprocessed_data.cpu()
            y = self.model(preprocessed_data)[0]
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
                I = self.model.get_binding_index("images")
                self.context.set_binding_shape(I, preprocessed_data.shape)
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
            y = self.model.run(self.output_names, {self.model.get_inputs()[0].name: preprocessed_data})
        elif self.aclite:
            import ipdb
            ipdb.set_trace()
            preprocessed_data = np.array(preprocessed_data)
        
            preds = self.model.execute([preprocessed_data,])
            preds = torch.from_numpy(preds[0])
            preds[:,:,:4] = dist2bbox(preds[:,:,:4], self.anchor_points,box_format='xywh')*self.stride_tensor 
            y = preds
        else:
            raise ValueError("Invalid model format")
        if isinstance(y, (list, tuple)):
            return self.from_numpy(y[0]) if len(y) == 1 else [self.from_numpy(x) for x in y]
        else:
            return self.from_numpy(y)
    def from_numpy(self, x):
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x
            
    def postprocess(self, inference_output: Any,conf_thres,iou_thres) -> Any:
        num_classes = inference_output.shape[2] - 6
        multi_label = self.config.get("multi_label")
        classes = eval(self.config.get("classes"))
        agnostic = self.config.get("agnostic")
        pred_candidates = torch.logical_and(inference_output[..., 5] > conf_thres, torch.max(inference_output[..., 6:], axis=-1)[0] > conf_thres)  # candidates
        assert 0 <= conf_thres <= 1, f'conf_thresh must be in 0.0 to 1.0, however {conf_thres} is provided.'
        assert 0 <= iou_thres <= 1, f'iou_thres must be in 0.0 to 1.0, however {iou_thres} is provided.'
        max_wh = 4096
        max_nms = 30000
        multi_label &= num_classes > 1

        output = [torch.zeros((0, 7), device=inference_output.device)] * inference_output.shape[0]
        for img_idx, x in enumerate(inference_output):
            x = x[pred_candidates[img_idx]]
            if not x.shape[0]:
                continue
            x[:, 6:] *= x[:, 5:6]
            box = x[:, :4]
            angle = x[:, 4:5].clone()
            import ipdb
            ipdb.set_trace()
            if multi_label:
                box_idx, class_idx = (x[:, 6:] > conf_thres).nonzero(as_tuple=False).T
                x = torch.cat((box[box_idx], angle[box_idx], x[box_idx, class_idx + 6, None], class_idx[:, None].float()), 1)
            else:
                conf, class_idx = x[:, 6:].max(1, keepdim=True)
                x = torch.cat((box, angle, conf, class_idx.float()), 1)[conf.view(-1) > conf_thres]
            if classes is not None:
                x = x[(x[:, 6:7] == torch.tensor(classes, device=x.device)).any(1)]
            num_box = x.shape[0]
            if not num_box:
                continue
            elif num_box > max_nms:
                x = x[x[:, 5].argsort(descending=True)[:max_nms]]
            class_offset = x[:, 6:7] * (0 if agnostic else max_wh)
            boxes, scores = x[:, :4], x[:, 5]
            boxes_xywh = boxes.clone()
            boxes_xy = (boxes_xywh[:, :2].clone() + class_offset).cpu().numpy()
            boxes_wh = boxes_xywh[:, 2:4].clone().cpu().numpy()
            boxes_angle = x[:, 4].clone().cpu().numpy()
            scores_for_cv2_nms = scores.cpu().numpy()
            boxes_for_cv2_nms = []
            for box_inds, _ in enumerate(boxes_xy):
                boxes_for_cv2_nms.append((boxes_xy[box_inds],boxes_wh[box_inds],boxes_angle[box_inds]))
            keep_box_idx = cv2.dnn.NMSBoxesRotated(boxes_for_cv2_nms,scores_for_cv2_nms,conf_thres,iou_thres)
            keep_box_idx = torch.from_numpy(keep_box_idx).type(torch.LongTensor)
            keep_box_idx = keep_box_idx.squeeze(axis=-1)
            output[img_idx] = x[keep_box_idx]
        return output
        
        
        
    def preprocess(self, input_data: Any) -> Any:
        input_data_src = input_data.astype(np.float32)
        input_data = input_data.astype(np.float32)
        input_data= letterbox(input_data, self.config["img_size"],self.config["stride"],auto=False)[0]
        input_data = input_data.transpose((2, 0, 1))[::-1]
        input_data /= 255
        input_data = torch.from_numpy(np.ascontiguousarray(np.expand_dims(input_data,axis=0)))
        return input_data,input_data_src
    
    def visualize(self,det,img,img_src):
        img_ori = img_src.copy()
        names = self.config["names"]
        import ipdb
        ipdb.set_trace()
        if len(det):
            det[:, :4] = rescale(img.shape[2:], det[:, :4], img_src.shape).round()
            for *obb, conf, cls in reversed(det):
                class_num = int(cls)
                label = f"{names[class_num]} {conf:.2f}"
                plot_box_and_label(
                    img_ori,
                    round(sum(img_ori.shape) / 2 * 0.0015),
                    obb,
                    label,
                    color=generate_colors(class_num, True),
                )   
        img_src = np.asarray(img_ori)
        return img_src        

    def run(self,input_data:Any,conf_thres,iou_thres):
        img,img_src = self.preprocess(input_data)
        img = img.to(self.device)
        pred_res = self.inference(img)
        class_names = self.config["names"]
        det = self.postprocess(pred_res,conf_thres,iou_thres)
        img_vis= self.visualize(det[0],img,img_src)
        img_vis = img_vis.astype(np.uint8)
        return img_vis
    def pipeline_run(self, input_data_list, conf_thres, iou_thres):
        """
        流水线推理模式
        """
        from queue import Queue
        from threading import Thread, Event
        
        # 创建有限大小的队列
        preprocess_queue = Queue(maxsize=3)
        inference_queue = Queue(maxsize=3)
        result_queue = Queue()  # 新增结果队列
        
        # 创建停止事件
        stop_event = Event()
        
        def preprocess_worker():
            """预处理线程"""
            try:
                while input_data_list and not stop_event.is_set():
                    input_data = input_data_list.pop(0)
                    img, img_src = self.preprocess(input_data)
                    img = img.to(self.device, non_blocking=True)
                    preprocess_queue.put((img, img_src))
            finally:
                preprocess_queue.put(None)
                    
        def inference_worker():
            """推理线程"""
            try:
                while not stop_event.is_set():
                    data = preprocess_queue.get()
                    if data is None:
                        break
                        
                    img, img_src = data
                    if img.device != self.device:
                        img = img.to(self.device, non_blocking=True)
                    with torch.cuda.amp.autocast(enabled=self.fp16):
                        pred_res = self.inference(img)
                    inference_queue.put((pred_res.cpu(), img.cpu(), img_src))
                    preprocess_queue.task_done()
            finally:
                inference_queue.put(None)
                    
        def postprocess_worker():
            """后处理线程"""
            results = []
            try:
                while not stop_event.is_set():
                    data = inference_queue.get()
                    if data is None:
                        break
                        
                    pred_res, img, img_src = data
                    det = self.postprocess(pred_res.to(self.device), conf_thres, iou_thres)
                    img_vis = self.visualize(det[0], img, img_src)
                    results.append(img_vis.astype(np.uint8))
                    inference_queue.task_done()
                
                # 将结果放入结果队列
                result_queue.put(results)
            except Exception as e:
                print(f"后处理错误: {e}")
                stop_event.set()
                result_queue.put([])  # 发生错误时返回空列表
                
        try:
            # 创建线程
            threads = []
            preprocess_thread = Thread(target=preprocess_worker)
            inference_thread = Thread(target=inference_worker)
            postprocess_thread = Thread(target=postprocess_worker)
            
            threads.extend([preprocess_thread, inference_thread, postprocess_thread])
            
            # 启动所有线程
            for t in threads:
                t.start()
                
            # 等待所有线程完成
            for t in threads:
                t.join()
                
            # 从结果队列获取结果
            results = result_queue.get()
            return results
                
        except Exception as e:
                print(f"Pipeline error: {e}")
                stop_event.set()
                raise
        finally:
            stop_event.set()
    def run_batch(self, input_data_list, conf_thres, iou_thres):
        """批处理入口函数"""
        return self.pipeline_run(input_data_list, conf_thres, iou_thres)