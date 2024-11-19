import os
import argparse
import subprocess
from loguru import logger
import cv2
import numpy as np
import onnxruntime as ort
import yaml
import amct_onnx as amct
from calibrator import Calibrator
from pathlib import Path
from datetime import datetime
import sys

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Ascend Model Quantization Tool')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to the configuration YAML file')
    parser.add_argument('--verbose', action='store_true',
                      help='Enable verbose logging')
    return parser.parse_args()

class AscendQuantizer:
    def __init__(self, config_path: str):
        """初始化Ascend量化器"""
        self.setup_logger()
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.setup_paths()
        self.calibrator = Calibrator(config_path)
    def setup_logger(self):
        logger.remove()
        logger.add(
            sink=sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level="INFO"
        )
    def setup_paths(self):
        """设置相关路径"""
        # 从配置文件获取基础路径
        self.tmp_base = Path(self.config['paths']['tmp_base'])
        self.model_output_dir = Path(self.config['paths']['model_output'])
        
        # 创建带时间戳的临时目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.tmp_dir = self.tmp_base / timestamp
        
        # 创建必要的目录
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.model_output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Temporary directory: {self.tmp_dir}")
        logger.info(f"Model output directory: {self.model_output_dir}")
    def create_quant_config(self):
        """创建量化配置"""
        config = self.config['calibration']
        config_json_file = os.path.join(self.tmp_dir, 'Ascent_Quant_config.json')
        
        amct.create_quant_config(
            config_file=config_json_file,
            model_file=config['model_path'],
            skip_layers=config.get('skip_layers', []),
            batch_num=config['batch_num'],
            activation_offset=True,
            config_defination=config.get('nuq_config', None)
        )
        return config_json_file
    def quantize_model(self, config_json_file: str):
        """量化模型"""
        config = self.config['calibration']
        scale_offset_record_file = os.path.join(self.tmp_dir, 'record.txt')
        modified_model_file = os.path.join(self.tmp_dir, 'model_quant.onnx')
        return modified_model_file, scale_offset_record_file
    def validate_quantized_model(self, modified_model_file: str):
        """验证量化后的模型"""
        logger.info("Validating quantized model...")
        try:
            ort_session = ort.InferenceSession(
                modified_model_file,
                sess_options=amct.AMCT_SO,
                providers=['CPUExecutionProvider']
            )
            input_name = ort_session.get_inputs()[0].name
            output_names = [x.name for x in ort_session.get_outputs()]
            
            self.calibrator.reset()
            for batch_idx in range(len(self.calibrator)):
                batch = self.calibrator.next_batch()
                if batch.size == 0:
                    break
                result = ort_session.run(output_names, {input_name: batch})
                logger.info(f"Processed batch {batch_idx + 1}/{len(self.calibrator)}")
                
            logger.info("Model validation completed successfully")
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            raise
    def convert_to_om(self, quant_model_path: str) -> bool:
        """使用ATC工具将量化后的ONNX模型转换为OM模型"""
        try:
            config = self.config['atc']
            output_om = os.path.join(self.model_output_dir, config['output_name'])
            
            cmd = [
                'atc',
                f'--soc_version={config["soc_version"]}',
                f'--framework={config["framework"]}',
                f'--model={quant_model_path}',
                f'--output={output_om}'
            ]
            
      
            
            logger.info(f"Running ATC command: {' '.join(cmd)}")
            
            # 执行ATC命令
            process = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            if process.returncode == 0:
                logger.info(f"Successfully converted to OM model: {output_om}")
                return True
            else:
                logger.error(f"ATC conversion failed: {process.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error during ATC conversion: {e}")
            return False
    def run(self):
        """执行完整的量化流程"""
        try:
            logger.info("Starting model quantization process...")
            
            # 1. 创建量化配置
            config_json_file = self.create_quant_config()
            logger.info("Quantization config created")
            
            # 2. 量化模型
            modified_model_file, scale_offset_record_file = self.quantize_model(config_json_file)
            amct.quantize_model(
            config_file=config_json_file,
            model_file=self.config['calibration']['model_path'],
            modified_onnx_file=modified_model_file,
            record_file=scale_offset_record_file
            )
            logger.info("Model quantized successfully")
            
            # 3. 验证量化后的模型
            self.validate_quantized_model(modified_model_file)
            
            # 4. 保存量化模型
            quant_model_path = os.path.join(self.model_output_dir, 'model_quant_final.onnx')
            amct.save_model(modified_model_file, scale_offset_record_file, quant_model_path)
            logger.info(f"Quantized model saved to {quant_model_path}")
            
            # 5. 转换为OM模型
            if self.convert_to_om(quant_model_path):
                logger.info("ATC conversion completed successfully")
            else:
                logger.error("ATC conversion failed")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            return False

def main():
    try:
        args = parse_args()
        if args.verbose:
            logger.level("DEBUG")
        logger.info(f"Starting quantization with config: {args.config}")
        quantizer = AscendQuantizer(args.config)
        if quantizer.run():
            logger.info("Quantization completed successfully")
            return 0
        else:
            logger.error("Quantization failed")
            return 1
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
