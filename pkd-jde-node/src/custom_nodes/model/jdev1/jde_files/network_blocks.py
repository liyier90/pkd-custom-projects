"""Network blocks for constructing the Darknet-53 backbone of the JDE model.

Modifications include:
- Removed custom Upsample module
- Removed training related code in YOLOLayer.forward()
- Removed loss related member variables
"""

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from custom_nodes.model.jdev1.jde_files.utils import decode_delta_map


class EmptyLayer(nn.Module):
    """Placeholder for `route` and `shortcut` layers."""

    @staticmethod
    def forward(inputs: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call.

        Args:
            inputs (torch.Tensor): Input from the previous layer.

        Returns:
            (torch.Tensor): The input, unmodified.
        """
        return inputs


class YOLOLayer(nn.Module):
    def __init__(
        self,
        anchors: List[Tuple[float, float]],
        num_classes: int,
        num_identities: int,
        embedding_dim: int,
        img_size: Tuple[int, int],
        yolo_layer: int,
    ):
        super().__init__()
        self.layer = yolo_layer
        self.anchors = torch.FloatTensor(anchors)
        self.num_anchors = len(anchors)  # number of anchors (4)
        self.num_classes = num_classes  # number of classes (80)
        self.num_identities = num_identities
        self.img_size = 0  # TODO: why is img_size unused
        self.emb_dim = embedding_dim
        self.shift = [1, 3, 5]

        self.anchor_vec: torch.Tensor
        self.anchor_wh: torch.Tensor
        self.grid_xy: torch.Tensor
        self.stride: float

        self.emb_scale = (
            math.sqrt(2) * math.log(self.num_identities - 1)
            if self.num_identities > 1
            else 1
        )

    def forward(self, inputs, img_size):
        """Defines the computation performed at every call.

        Args:
            inputs (torch.Tensor): Feature maps at various scales.
            img_size (Tuple[int, int]): Image size as specified by backbone
                configuration.

        Returns:
            (torch.Tensor): TODO
        """
        # From arxiv article:
        # Prediction map has dimension B * (6A + D) * H * W where A is number
        # of anchor templates, D is embedding dimension. B, H, and W are
        # batch size, height, and width (of the feature maps) respectively.
        pred_anchor, pred_embedding = inputs[:, :24, ...], inputs[:, 24:, ...]
        batch_size, grid_height, grid_width = (
            pred_anchor.shape[0],
            pred_anchor.shape[-2],
            pred_anchor.shape[-1],
        )

        if self.img_size != img_size:
            self._create_grids(img_size, grid_height, grid_width)

            if pred_anchor.is_cuda:
                self.grid_xy = self.grid_xy.cuda()
                self.anchor_wh = self.anchor_wh.cuda()

        # prediction
        pred_anchor = (
            pred_anchor.view(
                batch_size,
                self.num_anchors,
                self.num_classes + 5,
                grid_height,
                grid_width,
            )
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

        pred_embedding = pred_embedding.permute(0, 2, 3, 1).contiguous()
        pred_box = pred_anchor[..., :4]
        pred_conf = pred_anchor[..., 4:6].permute(0, 4, 1, 2, 3)

        pred_conf = torch.softmax(pred_conf, dim=1)[:, 1, ...].unsqueeze(-1)
        pred_embedding = F.normalize(
            pred_embedding.unsqueeze(1)
            .repeat(1, self.num_anchors, 1, 1, 1)
            .contiguous(),
            dim=-1,
        )
        pred_cls = torch.zeros(
            batch_size, self.num_anchors, grid_height, grid_width, 1
        ).cuda()
        pred_anchor = torch.cat([pred_box, pred_conf, pred_cls, pred_embedding], dim=-1)
        pred_anchor[..., :4] = decode_delta_map(
            pred_anchor[..., :4], self.anchor_vec.to(pred_anchor)
        )
        pred_anchor[..., :4] *= self.stride

        return pred_anchor.view(batch_size, -1, pred_anchor.shape[-1])

    def _create_grids(self, img_size, grid_height, grid_width):
        self.stride = img_size[0] / grid_width
        assert (
            self.stride == img_size[1] / grid_height
        ), f"Inconsistent stride size: {self.stride} v.s. {img_size[1]} / {grid_height}"

        # build xy offsets
        grid_x = (
            torch.arange(grid_width)
            .repeat((grid_height, 1))
            .view((1, 1, grid_height, grid_width))
            .float()
        )
        grid_y = (
            torch.arange(grid_height)
            .repeat((grid_width, 1))
            .transpose(0, 1)
            .view((1, 1, grid_height, grid_width))
            .float()
        )
        self.grid_xy = torch.stack((grid_x, grid_y), 4)

        # build wh gains
        self.anchor_vec = self.anchors / self.stride
        self.anchor_wh = self.anchor_vec.view(1, self.num_anchors, 1, 1, 2)
