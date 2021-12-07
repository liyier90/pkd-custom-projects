import numpy as np
import torch
from torchvision.ops import nms


def bbox_iou(box1, box2, x1y1x2y2=False):
    """
    Returns the IoU of two bounding boxes
    """
    N, M = len(box1), len(box2)
    if x1y1x2y2:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    else:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    # get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1.unsqueeze(1), b2_x1)
    inter_rect_y1 = torch.max(b1_y1.unsqueeze(1), b2_y1)
    inter_rect_x2 = torch.min(b1_x2.unsqueeze(1), b2_x2)
    inter_rect_y2 = torch.min(b1_y2.unsqueeze(1), b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, 0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1, 0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b1_area = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1)).view(-1, 1).expand(N, M)
    b2_area = ((b2_x2 - b2_x1) * (b2_y2 - b2_y1)).view(1, -1).expand(N, M)

    return inter_area / (b1_area + b2_area - inter_area + 1e-16)


def build_targets_max(target, anchor_wh, nA, nC, nGh, nGw):
    """
    returns nT, nCorrect, tx, ty, tw, th, tconf, tcls
    """
    nB = len(target)  # number of images in batch

    txy = torch.zeros(nB, nA, nGh, nGw, 2).cuda()  # batch size, anchors, grid size
    twh = torch.zeros(nB, nA, nGh, nGw, 2).cuda()
    tconf = torch.LongTensor(nB, nA, nGh, nGw).fill_(0).cuda()
    tcls = (
        torch.ByteTensor(nB, nA, nGh, nGw, nC).fill_(0).cuda()
    )  # nC = number of classes
    tid = torch.LongTensor(nB, nA, nGh, nGw, 1).fill_(-1).cuda()
    for b in range(nB):
        t = target[b]
        t_id = t[:, 1].clone().long().cuda()
        t = t[:, [0, 2, 3, 4, 5]]
        nTb = len(t)  # number of targets
        if nTb == 0:
            continue

        # gxy, gwh = t[:, 1:3] * nG, t[:, 3:5] * nG
        gxy, gwh = t[:, 1:3].clone(), t[:, 3:5].clone()
        gxy[:, 0] = gxy[:, 0] * nGw
        gxy[:, 1] = gxy[:, 1] * nGh
        gwh[:, 0] = gwh[:, 0] * nGw
        gwh[:, 1] = gwh[:, 1] * nGh
        gi = torch.clamp(gxy[:, 0], min=0, max=nGw - 1).long()
        gj = torch.clamp(gxy[:, 1], min=0, max=nGh - 1).long()

        # Get grid box indices and prevent overflows (i.e. 13.01 on 13 anchors)
        # gi, gj = torch.clamp(gxy.long(), min=0, max=nG - 1).t()
        # gi, gj = gxy.long().t()

        # iou of targets-anchors (using wh only)
        box1 = gwh
        box2 = anchor_wh.unsqueeze(1)
        inter_area = torch.min(box1, box2).prod(2)
        iou = inter_area / (box1.prod(1) + box2.prod(2) - inter_area + 1e-16)

        # Select best iou_pred and anchor
        iou_best, a = iou.max(0)  # best anchor [0-2] for each target

        # Select best unique target-anchor combinations
        if nTb > 1:
            _, iou_order = torch.sort(-iou_best)  # best to worst

            # Unique anchor selection
            u = torch.stack((gi, gj, a), 0)[:, iou_order]
            # _, first_unique = np.unique(u, axis=1, return_index=True)  # first unique indices
            first_unique = return_torch_unique_index(
                u, torch.unique(u, dim=1)
            )  # torch alternative
            i = iou_order[first_unique]
            # best anchor must share significant commonality (iou) with target
            i = i[iou_best[i] > 0.60]  # TODO: examine arbitrary threshold
            if len(i) == 0:
                continue

            a, gj, gi, t = a[i], gj[i], gi[i], t[i]
            t_id = t_id[i]
            if len(t.shape) == 1:
                t = t.view(1, 5)
        else:
            if iou_best < 0.60:
                continue

        tc, gxy, gwh = t[:, 0].long(), t[:, 1:3].clone(), t[:, 3:5].clone()
        gxy[:, 0] = gxy[:, 0] * nGw
        gxy[:, 1] = gxy[:, 1] * nGh
        gwh[:, 0] = gwh[:, 0] * nGw
        gwh[:, 1] = gwh[:, 1] * nGh

        # XY coordinates
        txy[b, a, gj, gi] = gxy - gxy.floor()

        # Width and height
        twh[b, a, gj, gi] = torch.log(gwh / anchor_wh[a])  # yolo method
        # twh[b, a, gj, gi] = torch.sqrt(gwh / anchor_wh[a]) / 2 # power method

        # One-hot encoding of label
        tcls[b, a, gj, gi, tc] = 1
        tconf[b, a, gj, gi] = 1
        tid[b, a, gj, gi] = t_id.unsqueeze(1)
    tbox = torch.cat([txy, twh], -1)
    return tconf, tbox, tid


def build_targets_thres(target, anchor_wh, nA, nC, nGh, nGw):
    ID_THRESH = 0.5
    FG_THRESH = 0.5
    BG_THRESH = 0.4
    nB = len(target)  # number of images in batch
    assert len(anchor_wh) == nA

    tbox = torch.zeros(nB, nA, nGh, nGw, 4).cuda()  # batch size, anchors, grid size
    tconf = torch.LongTensor(nB, nA, nGh, nGw).fill_(0).cuda()
    tid = torch.LongTensor(nB, nA, nGh, nGw, 1).fill_(-1).cuda()
    for b in range(nB):
        t = target[b]
        t_id = t[:, 1].clone().long().cuda()
        t = t[:, [0, 2, 3, 4, 5]]
        nTb = len(t)  # number of targets
        if nTb == 0:
            continue

        gxy, gwh = t[:, 1:3].clone(), t[:, 3:5].clone()
        gxy[:, 0] = gxy[:, 0] * nGw
        gxy[:, 1] = gxy[:, 1] * nGh
        gwh[:, 0] = gwh[:, 0] * nGw
        gwh[:, 1] = gwh[:, 1] * nGh
        gxy[:, 0] = torch.clamp(gxy[:, 0], min=0, max=nGw - 1)
        gxy[:, 1] = torch.clamp(gxy[:, 1], min=0, max=nGh - 1)

        gt_boxes = torch.cat([gxy, gwh], dim=1)  # Shape Ngx4 (xc, yc, w, h)

        anchor_mesh = generate_anchor(nGh, nGw, anchor_wh)
        anchor_list = (
            anchor_mesh.permute(0, 2, 3, 1).contiguous().view(-1, 4)
        )  # Shpae (nA x nGh x nGw) x 4
        # print(anchor_list.shape, gt_boxes.shape)
        iou_pdist = bbox_iou(anchor_list, gt_boxes)  # Shape (nA x nGh x nGw) x Ng
        iou_max, max_gt_index = torch.max(
            iou_pdist, dim=1
        )  # Shape (nA x nGh x nGw), both

        iou_map = iou_max.view(nA, nGh, nGw)
        gt_index_map = max_gt_index.view(nA, nGh, nGw)

        # nms_map = pooling_nms(iou_map, 3)

        id_index = iou_map > ID_THRESH
        fg_index = iou_map > FG_THRESH
        bg_index = iou_map < BG_THRESH
        ign_index = (iou_map < FG_THRESH) * (iou_map > BG_THRESH)
        tconf[b][fg_index] = 1
        tconf[b][bg_index] = 0
        tconf[b][ign_index] = -1

        gt_index = gt_index_map[fg_index]
        gt_box_list = gt_boxes[gt_index]
        gt_id_list = t_id[gt_index_map[id_index]]
        # print(gt_index.shape, gt_index_map[id_index].shape, gt_boxes.shape)
        if torch.sum(fg_index) > 0:
            tid[b][id_index] = gt_id_list.unsqueeze(1)
            fg_anchor_list = anchor_list.view(nA, nGh, nGw, 4)[fg_index]
            delta_target = encode_delta(gt_box_list, fg_anchor_list)
            tbox[b][fg_index] = delta_target
    return tconf, tbox, tid


def decode_delta(delta, fg_anchor_list):
    px, py, pw, ph = (
        fg_anchor_list[:, 0],
        fg_anchor_list[:, 1],
        fg_anchor_list[:, 2],
        fg_anchor_list[:, 3],
    )
    dx, dy, dw, dh = delta[:, 0], delta[:, 1], delta[:, 2], delta[:, 3]
    gx = pw * dx + px
    gy = ph * dy + py
    gw = pw * torch.exp(dw)
    gh = ph * torch.exp(dh)
    return torch.stack([gx, gy, gw, gh], dim=1)


def decode_delta_map(delta_map, anchors):
    """
    :param: delta_map, shape (nB, nA, nGh, nGw, 4)
    :param: anchors, shape (nA,4)
    """
    nB, nA, nGh, nGw, _ = delta_map.shape
    anchor_mesh = generate_anchor(nGh, nGw, anchors)
    anchor_mesh = anchor_mesh.permute(
        0, 2, 3, 1
    ).contiguous()  # Shpae (nA x nGh x nGw) x 4
    anchor_mesh = anchor_mesh.unsqueeze(0).repeat(nB, 1, 1, 1, 1)
    pred_list = decode_delta(delta_map.view(-1, 4), anchor_mesh.view(-1, 4))
    pred_map = pred_list.view(nB, nA, nGh, nGw, 4)
    return pred_map


def encode_delta(gt_box_list, fg_anchor_list):
    px, py, pw, ph = (
        fg_anchor_list[:, 0],
        fg_anchor_list[:, 1],
        fg_anchor_list[:, 2],
        fg_anchor_list[:, 3],
    )
    gx, gy, gw, gh = (
        gt_box_list[:, 0],
        gt_box_list[:, 1],
        gt_box_list[:, 2],
        gt_box_list[:, 3],
    )
    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = torch.log(gw / pw)
    dh = torch.log(gh / ph)
    return torch.stack([dx, dy, dw, dh], dim=1)


def fast_nms(
    boxes,
    scores,
    iou_thres: float = 0.5,
    top_k: int = 200,
    second_threshold: bool = False,
    conf_thres: float = 0.5,
):
    """
    Vectorized, approximated, fast NMS, adopted from YOLACT:
    https://github.com/dbolya/yolact/blob/master/layers/functions/detection.py
    The original version is for multi-class NMS, here we simplify the code for single-class NMS
    """
    scores, idx = scores.sort(0, descending=True)

    idx = idx[:top_k].contiguous()
    scores = scores[:top_k]
    num_dets = idx.size()

    boxes = boxes[idx, :]

    iou = jaccard(boxes, boxes)
    iou.triu_(diagonal=1)
    iou_max, _ = iou.max(dim=0)

    keep = iou_max <= iou_thres

    if second_threshold:
        keep *= scores > conf_thres

    return idx[keep]


def generate_anchor(nGh, nGw, anchor_wh):
    nA = len(anchor_wh)
    yy, xx = torch.meshgrid(torch.arange(nGh), torch.arange(nGw))
    xx, yy = xx.cuda(), yy.cuda()

    mesh = torch.stack([xx, yy], dim=0)  # Shape 2, nGh, nGw
    mesh = mesh.unsqueeze(0).repeat(nA, 1, 1, 1).float()  # Shape nA x 2 x nGh x nGw
    anchor_offset_mesh = (
        anchor_wh.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, nGh, nGw)
    )  # Shape nA x 2 x nGh x nGw
    anchor_mesh = torch.cat(
        [mesh, anchor_offset_mesh], dim=1
    )  # Shape nA x 4 x nGh x nGw
    return anchor_mesh


@torch.jit.script
def intersect(box_a, box_b):
    """We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [n,A,4].
      box_b: (tensor) bounding boxes, Shape: [n,B,4].
    Return:
      (tensor) intersection area, Shape: [n,A,B].
    """
    n = box_a.size(0)
    A = box_a.size(1)
    B = box_b.size(1)
    max_xy = torch.min(
        box_a[:, :, 2:].unsqueeze(2).expand(n, A, B, 2),
        box_b[:, :, 2:].unsqueeze(1).expand(n, A, B, 2),
    )
    min_xy = torch.max(
        box_a[:, :, :2].unsqueeze(2).expand(n, A, B, 2),
        box_b[:, :, :2].unsqueeze(1).expand(n, A, B, 2),
    )
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, :, 0] * inter[:, :, :, 1]


def jaccard(box_a, box_b, iscrowd: bool = False):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes. If iscrowd=True, put the crowd in box_b.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    use_batch = True
    if box_a.dim() == 2:
        use_batch = False
        box_a = box_a[None, ...]
        box_b = box_b[None, ...]

    inter = intersect(box_a, box_b)
    area_a = (
        ((box_a[:, :, 2] - box_a[:, :, 0]) * (box_a[:, :, 3] - box_a[:, :, 1]))
        .unsqueeze(2)
        .expand_as(inter)
    )  # [A,B]
    area_b = (
        ((box_b[:, :, 2] - box_b[:, :, 0]) * (box_b[:, :, 3] - box_b[:, :, 1]))
        .unsqueeze(1)
        .expand_as(inter)
    )  # [A,B]
    union = area_a + area_b - inter

    out = inter / area_a if iscrowd else inter / union
    return out if use_batch else out.squeeze(0)


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4, method="standard"):
    """
    Removes detections with lower object confidence score than 'conf_thres'
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    Args:
        prediction,
        conf_thres,
        nms_thres,
        method = 'standard' or 'fast'
    """

    output = [None for _ in range(len(prediction))]
    for image_i, pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        # Get score and class with highest confidence

        v = pred[:, 4] > conf_thres
        v = v.nonzero().squeeze()
        if len(v.shape) == 0:
            v = v.unsqueeze(0)

        pred = pred[v]

        # If none are remaining => process next image
        if not pred.shape[0]:
            continue
        # From (center x, center y, width, height) to (x1, y1, x2, y2)
        pred[:, :4] = xywh2xyxy(pred[:, :4])

        # Non-maximum suppression
        if method == "standard":
            nms_indices = nms(pred[:, :4], pred[:, 4], nms_thres)
        elif method == "fast":
            nms_indices = fast_nms(
                pred[:, :4], pred[:, 4], iou_thres=nms_thres, conf_thres=conf_thres
            )
        else:
            raise ValueError("Invalid NMS type!")
        det_max = pred[nms_indices]

        if len(det_max) > 0:
            # Add max detections to outputs
            output[image_i] = (
                det_max
                if output[image_i] is None
                else torch.cat((output[image_i], det_max))
            )

    return output


def return_torch_unique_index(u, uv):
    n = uv.shape[1]  # number of columns
    first_unique = torch.zeros(n, device=u.device).long()
    for j in range(n):
        first_unique[j] = (uv[:, j : j + 1] == u).all(0).nonzero()[0]

    return first_unique


def scale_coords(img_size, coords, img0_shape):
    # Rescale x1, y1, x2, y2 from 416 to image size
    gain_w = float(img_size[0]) / img0_shape[1]  # gain  = old / new
    gain_h = float(img_size[1]) / img0_shape[0]
    gain = min(gain_w, gain_h)
    pad_x = (img_size[0] - img0_shape[1] * gain) / 2  # width padding
    pad_y = (img_size[1] - img0_shape[0] * gain) / 2  # height padding
    coords[:, [0, 2]] -= pad_x
    coords[:, [1, 3]] -= pad_y
    coords[:, 0:4] /= gain
    coords[:, :4] = torch.clamp(coords[:, :4], min=0)
    return coords


def xywh2xyxy(x):
    # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    # x, y are coordinates of center
    # (x1, y1) and (x2, y2) are coordinates of bottom left and top right respectively.
    y = torch.zeros_like(x) if x.dtype is torch.float32 else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # Bottom left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # Bottom left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # Top right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # Top right y
    return y


def xyxyn2xyxy(inputs: np.ndarray, height: float, width: float) -> np.ndarray:
    """Converts from [x1, y1, x2, y2] to normalised [x1, y1, x2, y2].
    (x1, y1) is the top left corner and (x2, y2) is the bottom right corner.
    Normalised coordinates are w.r.t. original image size.
    """
    outputs = np.empty_like(inputs)
    outputs[:, [0, 2]] = inputs[:, [0, 2]] * width
    outputs[:, [1, 3]] = inputs[:, [1, 3]] * height

    return outputs


def xyxyn2tlwh(x, height, width):
    y = np.empty_like(x)
    y[:, 0] = x[:, 0] * width  # Bottom left x
    y[:, 1] = x[:, 1] * height  # Bottom left y
    y[:, 2] = (x[:, 2] - x[:, 0]) * width  # Top right x
    y[:, 3] = (x[:, 3] - x[:, 1]) * height  # Top right y
    return y


def tlwh2xyxyn(x, height, width):
    # Convert bounding box format from [t, l, w, h] to [x1, y1, x2, y2]
    # x, y are coordinates of center
    # (x1, y1) and (x2, y2) are coordinates of bottom left and top right respectively.
    y = np.empty_like(x)
    y[:, 0] = x[:, 0] / width  # Bottom left x
    y[:, 1] = x[:, 1] / height  # Bottom left y
    y[:, 2] = (x[:, 0] + x[:, 2]) / width  # Top right x
    y[:, 3] = (x[:, 1] + x[:, 3]) / height  # Top right y
    return y
