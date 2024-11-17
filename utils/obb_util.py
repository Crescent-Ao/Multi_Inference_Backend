import torch
from loguru import logger as LOGGER
import torch.nn as nn
import cv2 as cv2
import numpy as np
def generate_anchors(feats, fpn_strides, grid_cell_size=5.0, grid_cell_offset=0.5, device='cpu', is_eval=False, mode='af'):
    anchors = []
    anchor_points = []
    stride_tensor = []
    num_anchors_list = []
    feats = [[100,100],[50,50],[25,25]]
    if is_eval:
        for i, stride in enumerate(fpn_strides):
            h, w = feats[i]
            shift_x = torch.arange(end=w, device=device) + grid_cell_offset
            shift_y = torch.arange(end=h, device=device) + grid_cell_offset
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
            anchor_point = torch.stack([shift_x, shift_y], axis=-1).to(torch.float)
            if mode == 'af': # anchor-free
                anchor_points.append(anchor_point.reshape([-1, 2]))
                stride_tensor.append(torch.full((h * w, 1), stride, dtype=torch.float, device=device))
            elif mode == 'ab': # anchor-based
                anchor_points.append(anchor_point.reshape([-1, 2]).repeat(3,1))
                stride_tensor.append(torch.full((h * w, 1), stride, dtype=torch.float, device=device).repeat(3,1))
        anchor_points = torch.cat(anchor_points)
        stride_tensor = torch.cat(stride_tensor)
        return anchor_points, stride_tensor
    else:
        for i, stride in enumerate(fpn_strides):
            _, _, h, w = feats[i].shape
            cell_half_size = grid_cell_size * stride * 0.5
            shift_x = (torch.arange(end=w, device=device) + grid_cell_offset) * stride
            shift_y = (torch.arange(end=h, device=device) + grid_cell_offset) * stride
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
            anchor = torch.stack([shift_x - cell_half_size, shift_y - cell_half_size,shift_x + cell_half_size, shift_y + cell_half_size], axis=-1).clone().to(feats[0].dtype)
            anchor_point = torch.stack([shift_x, shift_y], axis=-1).clone().to(feats[0].dtype)
            if mode == 'af': # anchor-free
                anchors.append(anchor.reshape([-1, 4]))
                anchor_points.append(anchor_point.reshape([-1, 2]))
            elif mode == 'ab': # anchor-based
                anchors.append(anchor.reshape([-1, 4]).repeat(3,1))
                anchor_points.append(anchor_point.reshape([-1, 2]).repeat(3,1))
            num_anchors_list.append(len(anchors[-1]))
            stride_tensor.append(torch.full([num_anchors_list[-1], 1], stride, dtype=feats[0].dtype))
        anchors = torch.cat(anchors)
        anchor_points = torch.cat(anchor_points).to(device)
        stride_tensor = torch.cat(stride_tensor).to(device)
        return anchors, anchor_points, num_anchors_list, stride_tensor
def dist2bbox(distance, anchor_points, box_format='xyxy'):
    lt, rb = torch.split(distance, 2, -1)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if box_format == 'xyxy':
        bbox = torch.cat([x1y1, x2y2], -1)
    elif box_format == 'xywh':
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        bbox = torch.cat([c_xy, wh], -1)
    return bbox
def fuse_conv_and_bn(conv, bn):
    '''Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/.'''
    fusedconv = (
        nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            groups=conv.groups,
            bias=True,
        )
        .requires_grad_(False)
        .to(conv.weight.device)
    )

    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # prepare spatial bias
    b_conv = (
        torch.zeros(conv.weight.size(0), device=conv.weight.device)
        if conv.bias is None
        else conv.bias
    )
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(
        torch.sqrt(bn.running_var + bn.eps)
    )
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv

def fuse_model(model):
    '''Fuse convolution and batchnorm layers of the model.'''
    from .obb_layers import Conv,SimConv, Conv_C3

    for m in model.modules():
        if (type(m) is Conv or type(m) is SimConv or type(m) is Conv_C3) and hasattr(m, "bn"):
            m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
            delattr(m, "bn")  # remove batchnorm
            m.forward = m.forward_fuse  # update forward
    return model


def load_state_dict(weights, model, map_location=None):
    """Load weights from checkpoint file, only assign weights those layers' name and shape are match."""
    # TODO model 不匹配时候的weights重新加载会不会有bug, 有bug的话需要修改
    ckpt = torch.load(weights, map_location=map_location)
    state_dict = ckpt["model"].float().state_dict()
    model_state_dict = model.state_dict()
    state_dict = {
        k: v
        for k, v in state_dict.items()
        if k in model_state_dict and v.shape == model_state_dict[k].shape
    }
    model.load_state_dict(state_dict, strict=False)
    del ckpt, state_dict, model_state_dict
    return model


def load_checkpoint(weights, map_location=None, inplace=True, fuse=True):
    """Load model from checkpoint file."""
    LOGGER.info("Loading checkpoint from {}".format(weights))
    ckpt = torch.load(weights, map_location=map_location)  # load
    model = ckpt["ema" if ckpt.get("ema") else "model"].float()
    if fuse:
        LOGGER.info("\nFusing model...")
        model = fuse_model(model).eval()
    else:
        model = model.eval()
    return model


def letterbox(
    im,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=True,
    scaleup=False,
    stride=32,
    return_int=False,
):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    if not return_int:
        return im, r, (dw, dh)
    else:
        return im, r, (left, top)


def generate_colors(i, bgr=False):
    hex = (
        "FF3838",
        "FF9D97",
        "FF701F",
        "FFB21D",
        "CFD231",
        "48F90A",
        "92CC17",
        "3DDB86",
        "1A9334",
        "00D4BB",
        "2C99A8",
        "00C2FF",
        "344593",
        "6473FF",
        "0018EC",
        "8438FF",
        "520085",
        "CB38FF",
        "FF95C8",
        "FF37C7",
    )
    palette = []
    for iter in hex:
        h = "#" + iter
        palette.append(tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4)))
    num = len(palette)
    color = palette[int(i) % num]
    return (color[2], color[1], color[0]) if bgr else color
def plot_box_and_label(
        image, lw, box, label="", color=(128, 128, 128), txt_color=(255, 255, 255), font=cv2.FONT_HERSHEY_COMPLEX
    ):
        # Add one xyxy box to image with label
        cx = int(box[0])
        cy = int(box[1])
        w = int(box[2])
        h = int(box[3])
        angle = int(box[4])
        rect = ((cx, cy), (w, h), angle)
        poly = cv2.boxPoints(longSideFormat2minAreaRect(rect))
        poly = np.int0(poly)
        cv2.drawContours(
            image,
            contours=[poly],
            contourIdx=-1,
            color=color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )
        if label:
            tf = lw-1  # font thickness
            w, h = cv2.getTextSize(label, 0, fontScale=lw / 2, thickness=tf)[0]  # text width, height
            cv2.putText(
                image,
                label,
                (cx, cy),
                font,
                lw / 2,
                txt_color,
                thickness=tf,
                lineType=cv2.LINE_AA,
            )
        
def longSideFormat2minAreaRect(longSide_inf):
    longSide = longSide_inf[1][0]
    shortSide = longSide_inf[1][1]
    theta = longSide_inf[-1]
    width = longSide
    height = shortSide
    if theta == 0:
        width = shortSide
        height = longSide
        theta = 90  
    else:
        if np.around(longSide, 2) == np.around(shortSide, 2):
            width = longSide
            height = shortSide
            pass
        if theta > 90:
            width = shortSide
            height = longSide
            theta -= 90
        else:
            pass

    if theta >= 180:
        raise ValueError("theta >= 180")

    return (longSide_inf[0], (width, height), theta)

def rescale(ori_shape, boxes, target_shape):
    """Rescale the output to the original image shape"""
    ratio = min(ori_shape[0] / target_shape[0], ori_shape[1] / target_shape[1])
    padding = (ori_shape[1] - target_shape[1] * ratio) / 2, (ori_shape[0] - target_shape[0] * ratio) / 2

    boxes[:, [0]] -= padding[0]
    boxes[:, [1]] -= padding[1]
    boxes[:, :4] /= ratio

    return boxes    


def generate_anchors(feats, fpn_strides, grid_cell_size=5.0, grid_cell_offset=0.5, device='cpu', is_eval=False, mode='af'):
    anchors = []
    anchor_points = []
    stride_tensor = []
    num_anchors_list = []
    feats = [[100,100],[50,50],[25,25]]
    if is_eval:
        for i, stride in enumerate(fpn_strides):
            h, w = feats[i]
            shift_x = torch.arange(end=w, device=device) + grid_cell_offset
            shift_y = torch.arange(end=h, device=device) + grid_cell_offset
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
            anchor_point = torch.stack([shift_x, shift_y], axis=-1).to(torch.float)
            if mode == 'af': # anchor-free
                anchor_points.append(anchor_point.reshape([-1, 2]))
                stride_tensor.append(torch.full((h * w, 1), stride, dtype=torch.float, device=device))
            elif mode == 'ab': # anchor-based
                anchor_points.append(anchor_point.reshape([-1, 2]).repeat(3,1))
                stride_tensor.append(torch.full((h * w, 1), stride, dtype=torch.float, device=device).repeat(3,1))
        anchor_points = torch.cat(anchor_points)
        stride_tensor = torch.cat(stride_tensor)
        return anchor_points, stride_tensor
    else:
        for i, stride in enumerate(fpn_strides):
            _, _, h, w = feats[i].shape
            cell_half_size = grid_cell_size * stride * 0.5
            shift_x = (torch.arange(end=w, device=device) + grid_cell_offset) * stride
            shift_y = (torch.arange(end=h, device=device) + grid_cell_offset) * stride
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
            anchor = torch.stack([shift_x - cell_half_size, shift_y - cell_half_size,shift_x + cell_half_size, shift_y + cell_half_size], axis=-1).clone().to(feats[0].dtype)
            anchor_point = torch.stack([shift_x, shift_y], axis=-1).clone().to(feats[0].dtype)
            if mode == 'af': # anchor-free
                anchors.append(anchor.reshape([-1, 4]))
                anchor_points.append(anchor_point.reshape([-1, 2]))
            elif mode == 'ab': # anchor-based
                anchors.append(anchor.reshape([-1, 4]).repeat(3,1))
                anchor_points.append(anchor_point.reshape([-1, 2]).repeat(3,1))
            num_anchors_list.append(len(anchors[-1]))
            stride_tensor.append(torch.full([num_anchors_list[-1], 1], stride, dtype=feats[0].dtype))
        anchors = torch.cat(anchors)
        anchor_points = torch.cat(anchor_points).to(device)
        stride_tensor = torch.cat(stride_tensor).to(device)
        return anchors, anchor_points, num_anchors_list, stride_tensor


def dist2bbox(distance, anchor_points, box_format='xyxy'):
    lt, rb = torch.split(distance, 2, -1)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if box_format == 'xyxy':
        bbox = torch.cat([x1y1, x2y2], -1)
    elif box_format == 'xywh':
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        bbox = torch.cat([c_xy, wh], -1)
    return bbox