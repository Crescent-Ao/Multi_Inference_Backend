import os
import argparse
from loguru import logger
import cv2
import numpy as np
from typing import Dict, Any
import yaml
import os
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import glob

import ctypes
import logging
logger = logging.getLogger(__name__)
ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_char_p
ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]


"""
There are 4 types calibrator in TensorRT.
trt.IInt8LegacyCalibrator
trt.IInt8EntropyCalibrator
trt.IInt8EntropyCalibrator2
trt.IInt8MinMaxCalibrator
"""


IMG_FORMATS = [".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".dng", ".webp", ".mpo"]
IMG_FORMATS.extend([f.upper() for f in IMG_FORMATS])

class TensorRTCalibrator(trt.IInt8MinMaxCalibrator):
    def __init__(self, stream, cache_file=""):
        trt.IInt8MinMaxCalibrator.__init__(self)
        self.stream = stream
        self.d_input = cuda.mem_alloc(self.stream.calibration_data.nbytes)
        self.cache_file = cache_file
        stream.reset()

    def get_batch_size(self):
        return self.stream.batch_size

    def get_batch(self, names):
        print("######################")
        print(names)
        print("######################")
        batch = self.stream.next_batch()
        if not batch.size:
            return None

        cuda.memcpy_htod(self.d_input, batch)
        return [int(self.d_input)]

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                logger.info("Using calibration cache to save time: {:}".format(self.cache_file))
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            logger.info("Caching calibration data for future use: {:}".format(self.cache_file))
            f.write(cache)


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
            from preprocess import process_detection
            return process_detection
        elif self.task_type == "classification":
            from preprocess import process_classification
            return process_classification
        elif self.task_type == "segmentation":
            from preprocess import process_segmentation
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
    
