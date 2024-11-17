import torch
import torch.nn as nn
import torch.nn.functional as F  
#NOTE AT Has been implemented
def single_stage_at_loss(f_s, f_t, p):
    def _at(feat, p):
        return F.normalize(feat.pow(p).mean(1).reshape(feat.size(0), -1))

    s_H, t_H = f_s.shape[2], f_t.shape[2]
    if s_H > t_H:
        f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
    elif s_H < t_H:
        f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))
    return (_at(f_s, p) - _at(f_t, p)).pow(2).mean()


def at_loss(g_s, g_t, p):
    return sum([single_stage_at_loss(f_s, f_t, p) for f_s, f_t in zip(g_s, g_t)])


class AT(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.feat_loss_weight = cfg.AT.LOSS.FEAT_WEIGHT
        self.p = cfg.AT.P
    def forward(self, student_feature_maps, teacher_feature_maps):
        loss_feat = self.feat_loss_weight* at_loss(student_feature_maps, teacher_feature_maps,self.p)
        return loss_feat
        