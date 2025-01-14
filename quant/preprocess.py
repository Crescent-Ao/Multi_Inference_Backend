import cv2
import numpy as np
import torch

def process_detection(img: np.ndarray, input_size: list, stride: int = 32) -> np.ndarray:
    from utils.obb_util import letterbox
    img = letterbox(img, input_size, stride=stride, auto=False)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
    img = img.astype(np.float32) / 255.0
    img = np.ascontiguousarray(img, dtype=np.float32)
    return img
    


def process_classification(img: np.ndarray, input_size: list, stride: int = 32) -> np.ndarray:
    """分类任务的预处理函数"""
    # 调整大小
    img = cv2.resize(img, (input_size[1], input_size[0]))
    
    # BGR to RGB and normalize
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
    img = img.astype(np.float32) / 255.0
    
    return img

def process_segmentation(img: np.ndarray, input_size: list, stride: int = 32) -> np.ndarray:
    """分割任务的预处理函数"""
    # 调整大小
    img = cv2.resize(img, (input_size[1], input_size[0]))
    img = img.astype(np.float32)[:, :, ::-1]

    img = img.astype(np.float32) / 255.0
    img -= np.array([0.485, 0.456, 0.406])
    img /= np.array([0.229, 0.224, 0.225])
    img = img.transpose(2, 0, 1)  # HWC to CHW
    img = torch.from_numpy(img).unsqueeze(0).cpu().numpy()
    return img 
