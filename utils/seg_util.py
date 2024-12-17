import torch
import torch.nn as nn
import cv2
import numpy as np
from loguru import logger as LOGGER



def load_checkpoint(weights, num_classes, map_location=None, inplace=True, fuse=True):
    """加载权重文件(仅权重)
    
    Args:
        weights: 模型权重文件路径
        num_classes: 类别数量
        map_location: 设备映射位置
        inplace: 是否就地修改模型
        fuse: 是否融合模型层
    Returns:
        加载了权重的模型
    """
    LOGGER.info("Loading weights from {}".format(weights))
    ckpt = torch.load(weights, map_location=map_location)
    
    # 获取模型
    from UNetformer.UNetFormer_lsk_qarepvgg import UNetFormer_lsk_s
    model = UNetFormer_lsk_s(num_classes=num_classes).to(map_location)
    model.load_state_dict(ckpt)
        
    if fuse:
        LOGGER.info("\nFusing model...")
        model = fuse_model(model).eval()
    else:
        model = model.eval()
        
    return model

def load_model_with_structure(weights, map_location=None, inplace=True, fuse=True):
    """加载完整模型(包含结构)
    
    Args:
        weights: 模型文件路径
        map_location: 设备映射位置
        inplace: 是否就地修改模型
        fuse: 是否融合模型层
    Returns:
        加载了权重的模型
    """
    LOGGER.info("Loading complete model from {}".format(weights))
    
    # 加载完整模型(包含结构)
    ckpt = torch.load(weights, map_location=map_location)
    
    # 处理不同的保存格式
    if isinstance(ckpt, dict):
        if 'model' in ckpt:
            model = ckpt['model']
        elif 'state_dict' in ckpt:
            from UNetformer.UNetFormer_lsk_qarepvgg import UNetFormer_lsk_s
            model = UNetFormer_lsk_s(num_classes=ckpt.get('num_classes', 4))
            model.load_state_dict(ckpt['state_dict'])
        else:
            LOGGER.warning("Unknown checkpoint format, trying to load as complete model")
            model = ckpt
    else:
        model = ckpt
    
    # 确保模型在正确的设备上
    model = model.to(map_location)
    
    if fuse:
        LOGGER.info("\nFusing model...")
        model = fuse_model(model).eval()
    else:
        model = model.eval()
        
    # 打印模型信息
    LOGGER.info(f"Model Info:")
    LOGGER.info(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    if hasattr(model, 'num_classes'):
        LOGGER.info(f"Number of classes: {model.num_classes}")
        
    return model

def fuse_model(model):
    """融合模型的卷积和BN层"""
    from UNetformer.UNetFormer_lsk_qarepvgg import QARepVGGBlockV2, RepVGGBlock
    
    for m in model.modules():
        if isinstance(m, (QARepVGGBlockV2, RepVGGBlock)) and not m.deploy:
            m.switch_to_deploy()
    return model

def generate_segment_colors(num_classes, bgr=False):
    """为语义分割生成类别颜色映射
    
    Args:
        num_classes: 类别数量
        bgr: 是否返回BGR格式的颜色
        
    Returns:
        colors: RGB或BGR格式的颜色列表
    """
    # 预定义的hex颜色列表
    hex_colors = [
        'FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86',
        '1A9334', '00D4BB', '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF',
        '520085', 'CB38FF', 'FF95C8', 'FF37C7', '33B2FF', 'B9E1FF', '70E2FF', '0DFFFF',
        'FFE793', 'FFB46B', 'FF6B6B', 'FF2400', '00FF00', '32CD32', '90EE90', '98FB98'
    ]
    
    # 转换hex为RGB
    colors = []
    for hex_color in hex_colors:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        colors.append((r, g, b) if not bgr else (b, g, r))
    
    # 如果预定义颜色不够,随机生成
    if len(colors) < num_classes:
        np.random.seed(42)  # 固定随机种子确保颜色一致性
        for _ in range(num_classes - len(colors)):
            color = tuple(np.random.randint(0, 255, 3))
            colors.append(color if not bgr else color[::-1])
            
    return colors[:num_classes]