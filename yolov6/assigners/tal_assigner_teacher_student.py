import torch
import torch.nn as nn
import torch.nn.functional as F
from yolov6.utils.nms_R import obb_box_iou_cuda
from yolov6.assigners.assigner_utils import (dist_calculator, iou_calculator,
                                             select_candidates_in_gts,select_candidates_in_gts_R,
                                             select_highest_overlaps)

from yolov6.utils.nms_R import xyxy2xywh
class TaskAlignedAssignerMutual(nn.Module):
    def __init__(self, topk=13, num_classes=80, alpha=1.0, beta=5.0, gamma = 3.0,eps=1e-9, max_epoch=100):
        super(TaskAlignedAssignerMutual, self).__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps
        self.max_epoch = max_epoch
        self.lut_angle, self.lut_index, self.lut_iou = self.initialize_lut()
    @torch.no_grad()
    def forward(self, pd_scores_s, pd_bboxes_s, pd_angles_s, pd_scores_t, pd_bboxes_t, pd_angles_t, anc_points, gt_labels, gt_bboxes, gt_angles, mask_gt, epoch_num):
        """This code referenced to
           https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py

        Args:
            pd_scores_s (Tensor): shape(bs, num_total_anchors, num_classes)
            pd_bboxes_s (Tensor): shape(bs, num_total_anchors, 4)
            pd_angles_s (Tensor): shape(bs, num_total_anchors, num_angles)
            pd_scores_t (Tensor): shape(bs, num_total_anchors, num_classes)
            pd_bboxes_t (Tensor): shape(bs, num_total_anchors, 4)
            pd_angles_t (Tensor): shape(bs, num_total_anchors, 1)
            #NOTE 传递都是角度解码的相关问题
            anc_points (Tensor): shape(num_total_anchors, 2)
            gt_labels (Tensor): shape(bs, n_max_boxes, 1)
            gt_bboxes (Tensor): shape(bs, n_max_boxes, 4)
            gt_angles (Tensor): shape(bs, n_max_boxes, 1)
            mask_gt (Tensor): shape(bs, n_max_boxes, 1)
        Returns:
            target_labels (Tensor): shape(bs, num_total_anchors)
            target_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            target_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            fg_mask (Tensor): shape(bs, num_total_anchors)
        """
        self.bs = pd_scores_s.size(0)
        self.n_max_boxes = gt_bboxes.size(1)
        self.num_total_anchors = pd_scores_s.size(1)
        if self.n_max_boxes == 0:
            device = gt_bboxes.device
            return (
                torch.full_like(pd_scores_s[..., 0], self.bg_idx).to(device),
                torch.zeros_like(pd_bboxes_s).to(device),
                torch.zeros_like(pd_angles_s).to(device),
                torch.zeros_like(pd_scores_s).to(device),
                torch.zeros_like(pd_scores_s[..., 0]).to(device),
            )

        cycle, step, self.bs = (1, self.bs, self.bs) if self.n_max_boxes <=50 else (self.bs, 1, 1)
        target_labels_lst, target_bboxes_lst, target_scores_lst, fg_mask_lst = [], [], [], []
        # loop batch dim in case of numerous object box
        align_metric_t_list = []
        align_metric_s_list = []
        mask_pos_in_gt_list = []
        for i in range(cycle):  # TODO change this
            start, end = i * step, (i + 1) * step
            pd_scores_s_ = pd_scores_s[start:end, ...]
            pd_bboxes_s_ = pd_bboxes_s[start:end, ...]
            pd_angles_s_ = pd_angles_s[start:end, ...]
            pd_scores_t_ = pd_scores_t[start:end, ...]
            pd_bboxes_t_ = pd_bboxes_t[start:end, ...]
            pd_angles_t_ = pd_angles_t[start:end, ...]
            gt_labels_ = gt_labels[start:end, ...]
            gt_bboxes_ = gt_bboxes[start:end, ...]
            gt_angles_ = gt_angles[start:end, ...]
            mask_gt_ = mask_gt[start:end, ...]
            mask_pos_mutal, mask_assign_in_gt, align_metric_mutual, align_metric_t, align_metric_s, overlaps_mutual, overlaps_t, overlaps_s = self.get_pos_mask_mutual(
                pd_scores_s_,pd_bboxes_s_,pd_angles_s_, pd_scores_t_, pd_bboxes_t_, pd_angles_t_, gt_labels_, gt_bboxes_, gt_angles_, anc_points, mask_gt_, epoch_num
            )

            # mask_pos, align_metric, overlaps = self.get_pos_mask(
            #     pd_scores_, pd_bboxes_, pd_angles_, gt_labels_, gt_bboxes_, gt_angles_,anc_points, mask_gt_
            # )

            target_gt_idx, fg_mask, mask_pos_mutal = select_highest_overlaps(mask_pos_mutal, overlaps_s, self.n_max_boxes)

            # assigned target
            target_labels, target_bboxes, target_angles, target_scores = self.get_targets(
                gt_labels_, gt_bboxes_, gt_angles_, target_gt_idx, fg_mask
            )
            # NOTE Mutual 分配
            align_metric_s *= mask_pos_mutal
            pos_align_metrics = align_metric_s.max(axis=-1, keepdim=True)[0]
            pos_overlaps = (overlaps_s * mask_pos_mutal).max(axis=-1, keepdim=True)[0]
            norm_align_metric = (align_metric_s * pos_overlaps / (pos_align_metrics + self.eps)).max(-2)[0].unsqueeze(-1) 
            target_scores = target_scores * norm_align_metric
    
            # align_metric_mutual *= mask_pos_mutal
            # pos_align_metrics = align_metric_mutual.max(axis=-1, keepdim=True)[0]
            # pos_overlaps = (overlaps_mutual * mask_pos_mutal).max(axis=-1, keepdim=True)[0]
            # norm_align_metric = (align_metric_mutual * pos_overlaps / (pos_align_metrics + self.eps)).max(-2)[0].unsqueeze(-1) 
            # target_scores = target_scores * norm_align_metric
            # NOTE 学生模型对齐
            align_metric_s *= mask_assign_in_gt
            pos_align_metrics_s = align_metric_s.max(axis=-1, keepdim=True)[0]
            pos_overlaps_s = (overlaps_s*mask_assign_in_gt).max(axis=-1, keepdim=True)[0]
            norm_align_metric_s = (align_metric_s * pos_overlaps_s / (pos_align_metrics_s + self.eps))
            # NOTE 教师模型对应
            align_metric_t *= mask_assign_in_gt
            pos_align_metrics_t = align_metric_t.max(axis=-1, keepdim=True)[0]
            pos_overlaps_t = (overlaps_t*mask_assign_in_gt).max(axis=-1, keepdim=True)[0]
            norm_align_metric_t = (align_metric_t * pos_overlaps_t / (pos_align_metrics_t + self.eps))
            # # normalize
            # align_metric *= mask_pos
            # pos_align_metrics = align_metric.max(axis=-1, keepdim=True)[0]
            # pos_overlaps = (overlaps * mask_pos).max(axis=-1, keepdim=True)[0]
            # norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).max(-2)[0].unsqueeze(-1)
            # target_scores = target_scores * norm_align_metric

            # append
            target_labels_lst.append(target_labels)
            target_bboxes_lst.append(target_bboxes)
            target_scores_lst.append(target_scores)
            fg_mask_lst.append(fg_mask)
            align_metric_s_list.append(norm_align_metric_s)
            align_metric_t_list.append(norm_align_metric_t)
            mask_pos_in_gt_list.append(mask_assign_in_gt)
            
            

        # concat
        target_labels = torch.cat(target_labels_lst, 0)
        target_bboxes = torch.cat(target_bboxes_lst, 0)
        target_scores = torch.cat(target_scores_lst, 0)
        fg_mask = torch.cat(fg_mask_lst, 0)
        align_metric_t = torch.cat(align_metric_t_list, 0)
        align_metric_s = torch.cat(align_metric_s_list, 0)
        mask_pos_in_gt = torch.cat(mask_pos_in_gt_list)

        return target_labels, target_bboxes, target_angles, target_scores, fg_mask.bool(), align_metric_t, align_metric_s, mask_pos_in_gt
    def get_pos_mask_mutual(self, pd_scores_s, pd_bboxes_s, pd_angles_s, pd_scores_t, pd_bboxes_t, pd_angles_t, gt_labels, gt_bboxes, gt_angles, anc_points, mask_gt, epoch_num):
        # get student anchor_align metric
        align_metric_s, overlaps_s, lut_angle_s = self.get_box_metrics(pd_scores_s, pd_bboxes_s, pd_angles_s, gt_labels, gt_bboxes, gt_angles)
        align_metric_t, overlaps_t, lut_angle_t = self.get_box_metrics(pd_scores_t, pd_bboxes_t, pd_angles_t, gt_labels, gt_bboxes, gt_angles)
        # mask_in_gts = select_candidates_in_gts(anc_points, gt_bboxes)
        # overlaps_s = overlaps_s * lut_angle_s
        # overlaps_t = overlaps_t * lut_angle_t
        #NOTE 更换成旋转检测
        mask_in_gts = select_candidates_in_gts_R(anc_points, gt_bboxes, gt_angles)
        distill_decay = ((1 - torch.cos(epoch_num * torch.pi / torch.tensor(self.max_epoch))) / 2) * (
            0.01 - 1)+1
        align_metric_mutual = (align_metric_t + 0.0*distill_decay*align_metric_t)/(0.0*distill_decay+1)
        overlaps_mutual = (overlaps_t + 0.0*distill_decay*overlaps_t)/(0.0*distill_decay+1)
        # NOTE 需要进行一下取舍的计算
        # mask_topk_s =  self.select_topk_candidates(
        #     align_metric_s* mask_in_gts, topk_mask=mask_gt.repeat([1, 1, self.topk]).bool()
        # )
        # mask_topk_t =  self.select_topk_candidates(
        #     align_metric_t * mask_in_gts, topk_mtsk=mask_gt.repeat([1, 1, self.topk]).bool()
        # )
        mask_topk_mutual =  self.select_topk_candidates(
            align_metric_mutual * mask_in_gts, topk_mask=mask_gt.repeat([1, 1, self.topk]).bool()
        )
        mask_pos_mutal = mask_topk_mutual * mask_in_gts * mask_gt
        mask_assign_ingt = mask_in_gts * mask_gt

        
        return mask_pos_mutal, mask_assign_ingt, align_metric_mutual, align_metric_t, align_metric_s, overlaps_mutual, overlaps_t, overlaps_s
        
    def get_pos_mask(self, pd_scores, pd_bboxes, pd_angles, gt_labels, gt_bboxes, gt_angles, anc_points, mask_gt):

        # get anchor_align metric
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, pd_angles, gt_labels, gt_bboxes, gt_angles)
        # get in_gts mask
        mask_in_gts = select_candidates_in_gts(anc_points, gt_bboxes)
        # get topk_metric mask
        mask_topk = self.select_topk_candidates(
            align_metric * mask_in_gts, topk_mask=mask_gt.repeat([1, 1, self.topk]).bool()
        )
        # merge all mask to a final mask
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps
    def get_box_metrics(self, pd_scores, pd_bboxes, pd_angles, gt_labels, gt_bboxes, gt_angles):
        
        pd_scores = pd_scores.permute(0, 2, 1)
        gt_labels = gt_labels.to(torch.long)
        # print("pd_angels", pd_angles.shape)
        # gt_angles = gt_angles.to(torch.long)
        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)
        ind[0] = torch.arange(end=self.bs).view(-1, 1).repeat(1, self.n_max_boxes)
        ind[1] = gt_labels.squeeze(-1)
        bbox_scores = pd_scores[ind[0], ind[1]]
        # ind_angle = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)
        # ind_angle[0] = torch.arange(end=self.bs).view(-1, 1).repeat(1, self.n_max_boxes)
        # ind_angle[1] = gt_angles.squeeze(-1)
        # angle_scores = pd_angles[ind_angle[0], ind_angle[1]]
        # self.ipdb_test(bbox_scores,10,"bbox_scores")
        # self.ipdb_test(angle_scores,100,"angle_scores")
        
        overlaps = iou_calculator(gt_bboxes, pd_bboxes)
        # angle_ratio = self.angle_measure(pd_angles, gt_angles, gt_bboxes)
        lut_angle_weight = self.lut_angle_measure(pd_angles, gt_angles, gt_bboxes)
        # self.ipdb_test(angle_ratio, 10, "angle_distance")
        # self.aps, 10, "overlaps")
        # print(overlaps.shape)
        # print(angle_ratio.shape,"angle_ratio")
        align_metric = bbox_scores.pow(self.alpha) * (overlaps).pow(self.beta)* lut_angle_weight.pow(self.gamma)
        # self.ipdb_test(align_metric, 10, "align metric")
        return align_metric, overlaps, lut_angle_weight
    def lut_angle_measure(self, pd_angles, gt_angles, gt_bboxes):
        """
            pd_angles (Tensor): shape(bs, num_total_anchors, 1)
            gt_bboxes (Tensor): shape(bs, n_max_boxes, 4)
            gt_angles (Tensor): shape(bs, n_max_boxes, 1)
            IoU (Tensor): shape(bs, n_max_boxes, num_total_anchors)
        """
       
        gt_bboxes_xywh = xyxy2xywh(gt_bboxes)
        # print(gt_bboxes_xywh.shape,"gt_bboxes_xywh")
        theta_diff = torch.abs(gt_angles.unsqueeze(2)-pd_angles.unsqueeze(1))
        # print(theta_diff.shape,"theta diff ")
        gt_area_ratio = gt_bboxes_xywh[...,-2:-1]/(gt_bboxes_xywh[...,-1:]+1e-5)

        lut_area_index = torch.abs(gt_area_ratio -self.lut_index.reshape(1,1,-1)).argmin(dim=-1,keepdim=True).unsqueeze(-1)
        # theta_diff = torch.abs(theta_diff-self.lut_angle.reshape(1,1,1,-1)).argmin(dim=-1,keepdim=True)
        
        # lut_theta_index = torch.abs(theta_diff-self.lut_angle.reshape(1,1,1,-1)).argmin(dim=-1,keepdim=True)
        lut_theta_index = torch.clip(theta_diff,0.0,180.0)*400.0/180
        lut_index = torch.cat([lut_area_index.repeat(1,1,self.num_total_anchors,1),lut_theta_index],dim=-1).long()
        
        lut_angle_weight = self.lut_iou[lut_index[...,0], lut_index[...,1]]
        return lut_angle_weight
        # NOTE LUT 近似计算
    def angle_measure(self, pd_angles, gt_angles, gt_bboxes):
        """calculate the angle measurement

        Args:
            pd_angles (Tensor): 
            gt_angles (_type_): _description_
            gt_bboxes (_type_): _description_

        Returns:
            _type_: _description_
        """
        gt_bboxes_xywh = xyxy2xywh(gt_bboxes)
        theta_diff = torch.abs(gt_angles.unsqueeze(2)-pd_angles.unsqueeze(1))
        # print(theta_diff.shape,"theta angles shape")
        # NOTE 求解出角度偏离的差向 [bs,n_max_boxes,num_total_anchors]

        gt_area_ratio = gt_bboxes_xywh[...,-2:-1]/(gt_bboxes_xywh[...,-1:]+1e-5)
        gt_area_ratio = gt_area_ratio.unsqueeze(2)
        # print(gt_area_ratio.shape, "gt area shape")
        # # gt_area_ratio = gt_area_ratio.repeat(1,1,self.num_total_anchors)
        # print(gt_area_ratio.shape, "gt area shape")
        # NOTE 求解出对应的area ratio值
        gt_area_value = 1.0/gt_area_ratio/(2-1.0/gt_area_ratio)
    
        # NOTE 求解出公式的value
        
        # gt_area_value = gt_area_value.repeat(1,1,self.num_total_anchors)
        # NOTE 当前的shape式 [bs,n_max_boxes, num_total_anchors]

        max_value = TaskAlignedAssigner.cos_function(0, gt_area_ratio, beta=2)
        min_value = TaskAlignedAssigner.cos_function(90, gt_area_ratio, beta=2)
        # print(max_value.shape,"max_value")
        # print(min_value.shape, "min_value")
        # print(theta_diff.shape,"theta diff")
        # print(gt_area_ratio.shape,"gt_area ratio")
        # print(gt_area_value.shape, "gt_value shape")
        angle_ratio = (TaskAlignedAssigner.cos_function(theta_diff,gt_area_ratio,beta=2)-min_value)/(max_value-min_value)*(1-gt_area_value)+gt_area_value
      
        angle_ratio = torch.where(torch.isnan(angle_ratio),torch.zeros_like(angle_ratio),angle_ratio)
        # print(torch.sum(angle_ratio==torch.nan),"nan number")
        # print(torch.max(angle_ratio),"angle ratio max")
        return angle_ratio.squeeze(-1)
        
        
    @staticmethod
    def cos_function(x, area_ratio, beta):
        # NOTE Area_ratio 
        y = 0.5+0.5*torch.cos(beta/area_ratio*x/180.0*torch.pi+torch.pi*(1-beta/(2.0*area_ratio)))
        return y
    def ipdb_test(self,input_scores, topk=500, log_str = "align metric"):
        demo = input_scores.clone().reshape(-1,1)
        vals, index = torch.topk(demo,k=topk,dim=0)
        mean_value = torch.mean(vals)
        print(log_str+"top k mean value", mean_value)
        print(log_str+"max value", torch.max(vals))
        
    def select_topk_candidates(self, metrics, largest=True, topk_mask=None):

        num_anchors = metrics.shape[-1]
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, axis=-1, largest=largest)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(axis=-1, keepdim=True) > self.eps).tile([1, 1, self.topk])
        topk_idxs = torch.where(topk_mask, topk_idxs, torch.zeros_like(topk_idxs))
        is_in_topk = F.one_hot(topk_idxs, num_anchors).sum(axis=-2)
        is_in_topk = torch.where(is_in_topk > 1, torch.zeros_like(is_in_topk), is_in_topk)
        return is_in_topk.to(metrics.dtype)

    def get_targets(self, gt_labels, gt_bboxes, gt_angles, target_gt_idx, fg_mask):

        # assigned target labels
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes
        target_labels = gt_labels.long().flatten()[target_gt_idx]

        # assigned target boxes
        target_bboxes = gt_bboxes.reshape([-1, 4])[target_gt_idx]

        # assigned target scores
        target_labels[target_labels < 0] = 0
        target_scores = F.one_hot(target_labels, self.num_classes)
        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, torch.full_like(target_scores, 0))

        # assigned angle boxes
        target_angles = gt_angles.reshape([-1, 1])[target_gt_idx]

        return target_labels, target_bboxes, target_angles, target_scores
    def initialize_lut(self, base_size = 10, point_nums = 400):
        angle_num = torch.linspace(0,180,point_nums).cuda()
        area_ratio = torch.linspace(1.0,7.0,point_nums).cuda()
        shift_area, shift_angle = torch.meshgrid(area_ratio, angle_num)
    # 偏移的角度以及偏移的面积，
        proposed_value = torch.stack([shift_angle,shift_area], axis=-1).reshape(-1,2)
        proposed_obb = torch.ones((proposed_value.size(0),5))
        proposed_obb[...,2:3] =(proposed_value[...,1:2] * base_size).reshape(-1,1)
        proposed_obb[...,3:4] = proposed_obb[...,3:4]*base_size
        proposed_obb[...,4:5] = proposed_value[...,0:1]
        gt_obb = torch.ones(point_nums,5)
        gt_obb[...,4:5] = 0.0
        gt_obb[...,2:3] = (area_ratio*base_size).reshape(-1,1)
        gt_obb[...,3:4] = base_size
        lut_iou = torch.zeros((proposed_obb.size(0),1))
        for i,start_idex in enumerate(range(0,len(proposed_obb),point_nums)):
            lut_iou[start_idex:start_idex+point_nums] = obb_box_iou_cuda(proposed_obb[start_idex:start_idex+point_nums],gt_obb[i].unsqueeze(0))
        lut_iou = lut_iou.reshape(point_nums,point_nums).cuda()
        return angle_num, area_ratio, lut_iou