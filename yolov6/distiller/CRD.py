import torch
from torch import nn
import torch.nn.functional as F
# CRD CFG
from yacs.config import CfgNode as CN
CFG = CN()

CFG.CRD = CN()
CFG.CRD.MODE = "exact"  # ("exact", "relax")
CFG.CRD.FEAT = CN()
CFG.CRD.FEAT.DIM = 128
CFG.CRD.FEAT.STUDENT_DIM = 256
CFG.CRD.FEAT.TEACHER_DIM = 256
CFG.CRD.LOSS = CN()
CFG.CRD.LOSS.CE_WEIGHT = 1.0
CFG.CRD.LOSS.FEAT_WEIGHT = 0.8
CFG.CRD.NCE = CN()
CFG.CRD.NCE.K = 16384
CFG.CRD.NCE.MOMENTUM = 0.5
CFG.CRD.NCE.TEMPERATURE = 0.07
