import os
import argparse
from loguru import logger
import cv2
import numpy as np
from typing import Dict, Any
import yaml

IMG_FORMATS = [".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".dng", ".webp", ".mpo"]
IMG_FORMATS.extend([f.upper() for f in IMG_FORMATS])

class Calibrator:
    def __init__(self, config_path: str):
        # 读取配置文件
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        calib_config = config['calibration']
        self.task_type = calib_config['task_type']
        self.batch_size = calib_config['batch_size']
        self.length = calib_config['batch_num']
        self.input_w = calib_config['input_size']['width']
        self.input_h = calib_config['input_size']['height']
        
        # 根据任务类型选择预处理函数
        self.process_func = self._get_process_func()
        
        self.img_list = [os.path.join(calib_config['calib_img_dir'], x) 
                        for x in os.listdir(calib_config['calib_img_dir']) 
                        if os.path.splitext(x)[-1] in IMG_FORMATS]
        
        self.calibration_data = np.zeros((self.batch_size, 3, self.input_h, self.input_w), 
                                       dtype=np.float32)
        
        # 初始化索引
        self.index = 0

    def _get_process_func(self):
        """根据任务类型返回对应的预处理函数"""
        if self.task_type == "detection":
            from quant.preprocess import process_detection
            return process_detection
        elif self.task_type == "classification":
            from quant.preprocess import process_classification
            return process_classification
        elif self.task_type == "segmentation":
            from quant.preprocess import process_segmentation
            return process_segmentation
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")
    def reset(self):
        self.index = 0
    def next_batch(self):
        if self.index < self.length:
            for i in range(self.batch_size):
                assert os.path.exists(self.img_list[i + self.index * self.batch_size]), f'{self.img_list[i + self.index * self.batch_size]} not found!!'
                img = cv2.imread(self.img_list[i + self.index * self.batch_size])
                # import ipdb
                # ipdb.set_trace()
                img = self.process_func(img, [self.input_h, self.input_w], 32)
                # import ipdb
                # ipdb.set_trace()
                self.calibration_data[i] = img

            self.index += 1
            return np.ascontiguousarray(self.calibration_data, dtype=np.float32)
        else:
            return np.array([])

    def __len__(self):
        return self.length
    
