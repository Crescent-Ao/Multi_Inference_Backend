import torch

import cv2 as cv2
import yaml
from core.Object_Detection import ObjectDetection
import os


with open('config/obb_det.yaml', 'r', encoding="utf-8") as f:
    config = yaml.safe_load(f)

current_dir = os.path.dirname(os.path.abspath(__file__))


config["model_path"] = r"C:\Users\87758\vue3_fastAPI\backend\deploy_model\obb_model\best_ckpt.pt"
config["platform"] = "Pytorch"
config["precision"] = "FP32"

infer_model = ObjectDetection(config)
input_path = r"C:\Users\87758\vue3_fastAPI\ors_trt_deployment\YOLOv6-R\00001.jpg"
img = cv2.imread(input_path)

output = infer_model.run(img,0.5,0.5)


import ipdb
ipdb.set_trace()

cv2.imwrite("output.jpg",output)


print("hello")
