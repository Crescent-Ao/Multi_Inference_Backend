import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from yolov6.utils.nms_R import obb_box_iou_cuda
from mmcv.ops import box_iou_rotated
if __name__=="__main__":
    demo = torch.linspace(0,180,1000).reshape(1000,1)
    hbb = torch.Tensor([20.0,20.0,20.0,60.0]).unsqueeze(0).repeat(1,1000).reshape(1000,4)
    obb_1 = torch.cat([hbb,demo])
    obb_zero = torch.Tensor([20.0,20.0,20.0,60.0]).unsqueeze(0)
    obb_iou = obb_box_iou_cuda(obb_1, obb_zero)
    print(obb_iou)    
