import os
import argparse
import subprocess
from loguru import logger
import cv2
import numpy as np
import tensorrt as trt
from calibrator import Calibrator
from pathlib import Path
from datetime import datetime
import sys
import yaml
import traceback


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='TensorRT Model Quantization Tool')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to the configuration YAML file')
    parser.add_argument('--verbose', action='store_true',
                      help='Enable verbose logging')
    return parser.parse_args()


class TrtQuantizer:
    def __init__(self, config_path: str):
        """初始化TensorRT量化器"""
        self.setup_logger()
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.setup_paths()
        if self.need_calibration():
            self.calibrator = Calibrator(
                batch_size=self.config['calibration']['batch_size'],
                num_batches=self.config['calibration']['batch_num'],
                img_dir=self.config['calibration']['calib_img_dir'],
                img_size=[
                    self.config['calibration']['input_size']['height'],
                    self.config['calibration']['input_size']['width']
                ]
            )

    def setup_logger(self):
        """设置日志"""
        logger.remove()
        logger.add(
            sink=sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level="INFO"
        )

    def setup_paths(self):
        """设置相关路径"""
        self.tmp_base = Path(self.config['paths']['tmp_base'])
        self.model_output_dir = Path(self.config['paths']['model_output'])
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.tmp_dir = self.tmp_base / timestamp
        
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.model_output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Temporary directory: {self.tmp_dir}")
        logger.info(f"Model output directory: {self.model_output_dir}")

    def need_calibration(self):
        """判断是否需要校准"""
        return (
            self.config['calibration']['precision'].lower() == 'int8' and 
            not self.config['calibration'].get('qat', False)
        )

    def build_engine(self):
        """构建TensorRT引擎"""
        calib_config = self.config['calibration']
        dtype = calib_config['precision'].lower()
        model_path = calib_config['model_path']
        
        logger.info(f"Building TensorRT engine with {dtype} precision...")
        
        # 设置TensorRT日志级别
        TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE if self.config.get('verbose') else trt.Logger.INFO)
        
        # 创建builder和network
        builder = trt.Builder(TRT_LOGGER)
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        
        # 如果是QAT的INT8模型，添加显式精度标志
        if dtype == "int8" and calib_config.get('qat', False):
            network_flags = network_flags | (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_PRECISION))
        
        network = builder.create_network(flags=network_flags)
        parser = trt.OnnxParser(network, TRT_LOGGER)

        # 解析ONNX模型
        with open(model_path, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    logger.error(f"ONNX parsing error: {parser.get_error(error)}")
                raise RuntimeError("Failed to parse ONNX model")

        # 配置builder
        config = builder.create_builder_config()
        config.max_workspace_size = calib_config['max_workspace_size'] << 30  # 转换为GB
        
        # 设置精度模式
        if dtype == "fp16":
            config.flags |= 1 << int(trt.BuilderFlag.FP16)
        elif dtype == "int8":
            config.flags |= 1 << int(trt.BuilderFlag.INT8)
            config.flags |= 1 << int(trt.BuilderFlag.FP16)
            
            # 如果需要INT8校准
            if not calib_config.get('qat', False):
                calib_cache = os.path.join(
                    self.tmp_dir, 
                    f"{Path(model_path).stem}_calibration.cache"
                )
                config.int8_calibrator = self.calibrator
                logger.info('Int8 calibration is enabled.')

        # 构建引擎
        engine = builder.build_engine(network, config)
        if engine is None:
            raise RuntimeError("Failed to build TensorRT engine")

        return engine

    def validate_engine(self, engine_path: str):
        """验证TensorRT引擎"""
        logger.info("Validating TensorRT engine...")
        try:
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            with open(engine_path, 'rb') as f:
                engine = runtime.deserialize_cuda_engine(f.read())
            if engine is None:
                raise RuntimeError("Failed to load TensorRT engine")
            logger.info("Engine validation successful")
            return True
        except Exception as e:
            logger.error(f"Engine validation failed: {e}")
            return False

    def run(self):
        """执行完整的量化流程"""
        try:
            logger.info("Starting TensorRT quantization process...")
            
            # 检查ONNX模型路径
            model_path = self.config['calibration']['model_path']
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"ONNX model not found: {model_path}")

            # 构建引擎
            engine = self.build_engine()
            
            # 设置输出路径
            dtype = self.config['calibration']['precision'].lower()
            engine_path = os.path.join(
                self.model_output_dir, 
                f"{Path(model_path).stem}_{dtype}.engine"
            )
            
            # 保存引擎
            with open(engine_path, 'wb') as f:
                f.write(engine.serialize())
            logger.info(f"Engine saved to: {engine_path}")

            # 验证引擎
            if self.validate_engine(engine_path):
                logger.info("Quantization completed successfully")
                return True
            else:
                logger.error("Engine validation failed")
                return False

        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            traceback.print_exc()
            return False


def main():
    try:
        args = parse_args()
        if args.verbose:
            logger.level("DEBUG")
        logger.info(f"Starting quantization with config: {args.config}")
        quantizer = TrtQuantizer(args.config)
        if quantizer.run():
            logger.info("Quantization completed successfully")
            return 0
        else:
            logger.error("Quantization failed")
            return 1
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    main()



