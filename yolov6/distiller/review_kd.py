import torch
from torch import nn
import torch.nn.functional as F
from yacs.config import CfgNode as CN

# torchSize [12,64,128,128]
# torchSize [12,128,64,64]
# torchSize [12,256,32,32]
# torchSize []



class ABF(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, fuse):
        super(ABF, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channel),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                mid_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channel),
        )
        if fuse:
            self.att_conv = nn.Sequential(
                nn.Conv2d(mid_channel * 2, 2, kernel_size=1),
                nn.Sigmoid(),
            )
        else:
            self.att_conv = None
        nn.init.kaiming_uniform_(self.conv1[0].weight, a=1)  # pyre-ignore
        nn.init.kaiming_uniform_(self.conv2[0].weight, a=1)  # pyre-ignore

    def forward(self, x, y=None, shape=None, out_shape=None):
        n, _, h, w = x.shape
        # transform student features
        print("first abf")
        x = self.conv1(x)
                
        if self.att_conv is not None:
            # upsample residual features
            y = F.interpolate(y, (shape, shape), mode="nearest")
            # fusion
            z = torch.cat([x, y], dim=1)
            z = self.att_conv(z)
            x = x * z[:, 0].view(n, 1, h, w) + y * z[:, 1].view(n, 1, h, w)
        # output
        if x.shape[-1] != out_shape:
            x = F.interpolate(x, (out_shape, out_shape), mode="nearest")
        y = self.conv2(x)
        return y, x

class ReviewKD(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.shapes = cfg.REVIEWKD.SHAPES
        self.out_shapes = cfg.REVIEWKD.OUT_SHAPES
        in_channels = cfg.REVIEWKD.IN_CHANNELS
        out_channels = cfg.REVIEWKD.OUT_CHANNELS
        self.ce_loss_weight = cfg.REVIEWKD.CE_WEIGHT
        self.reviewkd_loss_weight = cfg.REVIEWKD.REVIEWKD_WEIGHT
        self.warmup_epochs = cfg.REVIEWKD.WARMUP_EPOCHS
        self.stu_preact = cfg.REVIEWKD.STU_PREACT
        self.max_mid_channel = cfg.REVIEWKD.MAX_MID_CHANNEL

        abfs = nn.ModuleList()
        mid_channel = min(512, in_channels[-1])
        mid_channel = min(512, in_channels[-1])
        for idx, in_channel in enumerate(in_channels):
            abfs.append(
                ABF(
                    in_channel,
                    mid_channel,
                    out_channels[idx],
                    idx < len(in_channels) - 1,
                )
            )
        abfs = abfs.cuda()
        self.abfs = abfs[::-1]
    def get_learnable_parameters(self):
        return self.abfs.modules()
    @staticmethod
    def hcl_loss(fstudent, fteacher):
        loss_all = 0.0
        for fs, ft in zip(fstudent, fteacher):
            n, c, h, w = fs.shape
            loss = F.mse_loss(fs, ft, reduction="mean")
            cnt = 1.0
            tot = 1.0
            for l in [4, 2, 1]:
                if l >= h:
                    continue
                tmpfs = F.adaptive_avg_pool2d(fs, (l, l))
                tmpft = F.adaptive_avg_pool2d(ft, (l, l))
                cnt /= 2.0
                loss += F.mse_loss(tmpfs, tmpft, reduction="mean") * cnt
                tot += cnt
            loss = loss / tot
            loss_all = loss_all + loss
        return loss_all

    def forward(self, student_features, teacher_features):
        results = []
        x = student_features[::-1]
        for i in student_features:
            print(i.shape)
        print("student")
        for j in teacher_features:
            print(j.shape)
        print("teacher")
        out_features, res_features = self.abfs[0](x[0], out_shape=self.out_shapes[0])
        
        results.append(out_features)
        for features, abf, shape, out_shape in zip(
            x, self.abfs, self.shapes, self.out_shapes
        ):
            
            out_features, res_features = abf(features, res_features, shape, out_shape)
            results.insert(0, out_features)
        loss_review_kd = ReviewKD.hcl_loss(results, teacher_features)
        return loss_review_kd
