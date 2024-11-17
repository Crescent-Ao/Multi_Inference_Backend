import torch
import torch.nn as nn
import torch.nn.functional as F



def nst_loss(g_s, g_t):
    return sum([single_stage_nst_loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t)])


def single_stage_nst_loss(f_s, f_t):
    s_H, t_H = f_s.shape[2], f_t.shape[2]
    if s_H > t_H:
        f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
    elif s_H < t_H:
        f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))

    f_s = f_s.view(f_s.shape[0], f_s.shape[1], -1)
    f_s = F.normalize(f_s, dim=2)
    f_t = f_t.view(f_t.shape[0], f_t.shape[1], -1)
    f_t = F.normalize(f_t, dim=2)

    return (
        poly_kernel(f_t, f_t).mean().detach()
        + poly_kernel(f_s, f_s).mean()
        - 2 * poly_kernel(f_s, f_t).mean()
    )


def poly_kernel(a, b):
    a = a.unsqueeze(1)
    b = b.unsqueeze(2)
    res = (a * b).sum(-1).pow(2)
    return res

class NST(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg = cfg
    def forward(self, student_feature_maps, teacher_feature_maps):
        loss_feat = nst_loss(student_feature_maps, teacher_feature_maps)
        return loss_feat