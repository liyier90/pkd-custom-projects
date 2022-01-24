import torch
import torch.nn as nn

from custom_nodes.model.fairmotv1.fairmot_files.utils import (
    gather_feat,
    transpose_and_gather_feat,
)


def mot_decode(heatmap, wh, reg, K):
    batch, _, _, _ = heatmap.size()

    # perform nms on heatmap
    heatmap = _nms(heatmap)

    scores, inds, clses, ys, xs = _topk(heatmap, K)
    reg = transpose_and_gather_feat(reg, inds)
    reg = reg.view(batch, K, 2)
    xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
    ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    wh = transpose_and_gather_feat(wh, inds)
    wh = wh.view(batch, K, 4)
    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    bboxes = torch.cat(
        [xs - wh[..., 0:1], ys - wh[..., 1:2], xs + wh[..., 2:3], ys + wh[..., 3:4]],
        dim=2,
    )
    detections = torch.cat([bboxes, scores, clses], dim=2)

    return detections, inds


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _topk(scores, K):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = torch.true_divide(topk_inds, width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = torch.true_divide(topk_ind, K).int()
    topk_inds = gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs
