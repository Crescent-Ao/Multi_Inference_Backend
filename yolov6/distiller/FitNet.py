import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvReg(nn.Module):
    """Convolutional regression"""

    def __init__(self, s_shape, t_shape, use_relu=True):
        super(ConvReg, self).__init__()
        self.use_relu = use_relu
        s_N, s_C, s_H, s_W = s_shape
        t_N, t_C, t_H, t_W = t_shape
        if s_H == 2 * t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=3, stride=2, padding=1)
        elif s_H * 2 == t_H:
            self.conv = nn.ConvTranspose2d(s_C, t_C, kernel_size=4, stride=2, padding=1)
        elif s_H >= t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=(1 + s_H - t_H, 1 + s_W - t_W))
        else:
            raise NotImplemented("student size {}, teacher size {}".format(s_H, t_H))
        self.bn = nn.BatchNorm2d(t_C)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.use_relu:
            return self.relu(self.bn(x))
        else:
            return self.bn(x)


class FitNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.feat_loss_weight = cfg.FITNET.LOSS.FEAT_WEIGHT
        self.hint_layer = cfg.FITNET.HINT_LAYER
        feat_s_shapes, feat_t_shapes = cfg.FITNET.SHAPES
        self.conv_reg = ConvReg(
            feat_s_shapes[self.hint_layer], feat_t_shapes[self.hint_layer]
        )

    def get_learnable_parameters(self):
        return self.conv_reg.modules()

    def forward(self, student_feature_maps, teacher_feature_maps):
        f_s = self.conv_reg(student_feature_maps[self.hint_layer])
        f_t = teacher_feature_maps[self.hint_layer]
        loss_feat = F.mse_loss(f_s, f_t)
        return loss_feat
