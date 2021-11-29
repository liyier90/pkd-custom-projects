import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import build_targets_max, build_targets_thres, decode_delta_map


class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()

    def forward(self, x):
        return x


class Upsample(nn.Module):
    # Custom Upsample layer (nn.Upsample gives deprecated warning message)

    def __init__(self, scale_factor=1, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)


class YOLOLayer(nn.Module):
    def __init__(self, anchors, nC, nID, nE, img_size, yolo_layer):
        super(YOLOLayer, self).__init__()
        self.layer = yolo_layer
        nA = len(anchors)
        self.anchors = torch.FloatTensor(anchors)
        self.nA = nA  # number of anchors (3)
        self.nC = nC  # number of classes (80)
        self.nID = nID  # number of identities
        self.img_size = 0
        self.emb_dim = nE
        self.shift = [1, 3, 5]

        self.SmoothL1Loss = nn.SmoothL1Loss()
        self.SoftmaxLoss = nn.CrossEntropyLoss(ignore_index=-1)
        self.CrossEntropyLoss = nn.CrossEntropyLoss()
        self.IDLoss = nn.CrossEntropyLoss(ignore_index=-1)
        self.s_c = nn.Parameter(-4.15 * torch.ones(1))  # -4.15
        self.s_r = nn.Parameter(-4.85 * torch.ones(1))  # -4.85
        self.s_id = nn.Parameter(-2.3 * torch.ones(1))  # -2.3

        self.emb_scale = math.sqrt(2) * math.log(self.nID - 1) if self.nID > 1 else 1

    def forward(self, p_cat, img_size, targets=None, classifier=None, test_emb=False):
        p, p_emb = p_cat[:, :24, ...], p_cat[:, 24:, ...]
        nB, nGh, nGw = p.shape[0], p.shape[-2], p.shape[-1]

        if self.img_size != img_size:
            create_grids(self, img_size, nGh, nGw)

            if p.is_cuda:
                self.grid_xy = self.grid_xy.cuda()
                self.anchor_wh = self.anchor_wh.cuda()

        p = (
            p.view(nB, self.nA, self.nC + 5, nGh, nGw)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )  # prediction

        p_emb = p_emb.permute(0, 2, 3, 1).contiguous()
        p_box = p[..., :4]
        p_conf = p[..., 4:6].permute(0, 4, 1, 2, 3)  # Conf

        # Training
        if targets is not None:
            if test_emb:
                tconf, tbox, tids = build_targets_max(
                    targets, self.anchor_vec.cuda(), self.nA, self.nC, nGh, nGw
                )
            else:
                tconf, tbox, tids = build_targets_thres(
                    targets, self.anchor_vec.cuda(), self.nA, self.nC, nGh, nGw
                )
            tconf, tbox, tids = tconf.cuda(), tbox.cuda(), tids.cuda()
            mask = tconf > 0

            # Compute losses
            nT = sum([len(x) for x in targets])  # number of targets
            nM = mask.sum().float()  # number of anchors (assigned to targets)
            nP = torch.ones_like(mask).sum().float()
            if nM > 0:
                lbox = self.SmoothL1Loss(p_box[mask], tbox[mask])
            else:
                FT = torch.cuda.FloatTensor if p_conf.is_cuda else torch.FloatTensor
                lbox, lconf = FT([0]), FT([0])
            lconf = self.SoftmaxLoss(p_conf, tconf)
            lid = torch.Tensor(1).fill_(0).squeeze().cuda()
            emb_mask, _ = mask.max(1)

            # For convenience we use max(1) to decide the id, TODO: more reseanable strategy
            tids, _ = tids.max(1)
            tids = tids[emb_mask]
            embedding = p_emb[emb_mask].contiguous()
            embedding = self.emb_scale * F.normalize(embedding)
            nI = emb_mask.sum().float()

            if test_emb:
                if np.prod(embedding.shape) == 0 or np.prod(tids.shape) == 0:
                    return torch.zeros(0, self.emb_dim + 1).cuda()
                emb_and_gt = torch.cat([embedding, tids.float()], dim=1)
                return emb_and_gt

            if len(embedding) > 1:
                logits = classifier(embedding).contiguous()
                lid = self.IDLoss(logits, tids.squeeze())

            # Sum loss components
            loss = (
                torch.exp(-self.s_r) * lbox
                + torch.exp(-self.s_c) * lconf
                + torch.exp(-self.s_id) * lid
                + (self.s_r + self.s_c + self.s_id)
            )
            loss *= 0.5

            return loss, loss.item(), lbox.item(), lconf.item(), lid.item(), nT

        else:
            p_conf = torch.softmax(p_conf, dim=1)[:, 1, ...].unsqueeze(-1)
            p_emb = F.normalize(
                p_emb.unsqueeze(1).repeat(1, self.nA, 1, 1, 1).contiguous(), dim=-1
            )
            # p_emb_up = F.normalize(shift_tensor_vertically(p_emb, -self.shift[self.layer]), dim=-1)
            # p_emb_down = F.normalize(shift_tensor_vertically(p_emb, self.shift[self.layer]), dim=-1)
            p_cls = torch.zeros(nB, self.nA, nGh, nGw, 1).cuda()  # Temp
            p = torch.cat([p_box, p_conf, p_cls, p_emb], dim=-1)
            # p = torch.cat([p_box, p_conf, p_cls, p_emb, p_emb_up, p_emb_down], dim=-1)
            p[..., :4] = decode_delta_map(p[..., :4], self.anchor_vec.to(p))
            p[..., :4] *= self.stride

            return p.view(nB, -1, p.shape[-1])


def create_grids(self, img_size, nGh, nGw):
    self.stride = img_size[0] / nGw
    assert self.stride == img_size[1] / nGh, "{} v.s. {}/{}".format(
        self.stride, img_size[1], nGh
    )

    # build xy offsets
    grid_x = torch.arange(nGw).repeat((nGh, 1)).view((1, 1, nGh, nGw)).float()
    grid_y = (
        torch.arange(nGh)
        .repeat((nGw, 1))
        .transpose(0, 1)
        .view((1, 1, nGh, nGw))
        .float()
    )
    # grid_y = grid_x.permute(0, 1, 3, 2)
    self.grid_xy = torch.stack((grid_x, grid_y), 4)

    # build wh gains
    self.anchor_vec = self.anchors / self.stride
    self.anchor_wh = self.anchor_vec.view(1, self.nA, 1, 1, 2)
