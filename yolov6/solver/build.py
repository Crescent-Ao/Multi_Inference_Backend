#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import math
import os

import torch
import torch.nn as nn

from yolov6.utils.events import LOGGER


def build_optimizer(cfg, model,distiller=None):
    """ Build optimizer from cfg file."""
    g_bnw, g_w, g_b = [], [], []
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            g_b.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):
            g_bnw.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            g_w.append(v.weight)
    if distiller!=None:
        op = hasattr(distiller, "get_learnable_parameter")
        print(op)
        if op:
            distill_para = distiller.get_learnable_parameters()
            for j in distill_para:
                if hasattr(j, 'bias') and isinstance(j.bias, nn.Parameter):
                    g_b.append(j.bias)
                if isinstance(j, nn.BatchNorm2d):
                    g_bnw.append(j.weight)
                elif hasattr(j, 'weight') and isinstance(j.weight, nn.Parameter):
                    g_w.append(j.weight)
                    
    assert cfg.solver.optim == 'SGD' or 'Adam', 'ERROR: unknown optimizer, use SGD defaulted'
    if cfg.solver.optim == 'SGD':
        optimizer = torch.optim.SGD(g_bnw, lr=cfg.solver.lr0, momentum=cfg.solver.momentum, nesterov=True)
    elif cfg.solver.optim == 'Adam':
        optimizer = torch.optim.Adam(g_bnw, lr=cfg.solver.lr0, betas=(cfg.solver.momentum, 0.999))
    elif cfg.solver.optim == 'AdamW':
        optimizer = torch.optim.AdamW(g_bnw, lr=cfg.solver.lr0, betas=(cfg.solver.momentum, 0.999), eps=1e-3)

    optimizer.add_param_group({'params': g_w, 'weight_decay': cfg.solver.weight_decay})
    optimizer.add_param_group({'params': g_b})

    del g_bnw, g_w, g_b
    return optimizer


def build_lr_scheduler(cfg, optimizer, epochs):
    """Build learning rate scheduler from cfg file."""
    if cfg.solver.lr_scheduler == 'Cosine':
        lf = lambda x: ((1 - math.cos(x * math.pi / epochs)) / 2) * (cfg.solver.lrf - 1) + 1
    elif cfg.solver.lr_scheduler == 'Constant':
        lf = lambda x: 1.0
    else:
        LOGGER.error('unknown lr scheduler, use Cosine defaulted')

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    return scheduler, lf
