"""Efficient Decoupled Head.

Modifications:
- Removed nc, anchors instance attributes
- Renamed no, nl, na instance attributes
- Hard coded "inplace" behaviour
- Remove training related code
"""

import math
from typing import List

import torch
import torch.nn as nn

from ..layers.common import Conv


class EfficientDecoupledHead(nn.Module):  # pylint: disable=too-many-instance-attributes
    """Efficient Decoupled Head
    With hardware-aware design, the decoupled head is optimized with
    hybridchannels methods.
    """

    # Initial probability, used for initializing biases
    prior_prob = 1e-2

    def __init__(
        self,
        num_classes: int = 80,
        anchors: int = 1,
        num_layers: int = 3,
        head_layers: nn.Sequential = None,
    ) -> None:
        # detection layer
        super().__init__()
        assert head_layers is not None
        self.num_outputs = num_classes + 5  # number of outputs per anchor
        self.num_layers = num_layers  # number of detection layers
        self.num_anchors = anchors
        self.grid = [torch.zeros(1)] * num_layers

        stride = [8, 16, 32]  # strides computed during build
        self.stride = torch.tensor(stride)

        # Init decouple head
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()

        # Efficient decoupled head layers
        for i in range(num_layers):
            idx = i * 6
            self.stems.append(head_layers[idx])
            self.cls_convs.append(head_layers[idx + 1])
            self.reg_convs.append(head_layers[idx + 2])
            self.cls_preds.append(head_layers[idx + 3])
            self.reg_preds.append(head_layers[idx + 4])
            self.obj_preds.append(head_layers[idx + 5])

    def initialize_biases(self):
        """Initialize biases for classification and regression layers."""
        for conv in self.cls_preds:
            bias = conv.bias.view(self.num_anchors, -1)
            bias.data.fill_(-math.log((1 - self.prior_prob) / self.prior_prob))
            conv.bias = torch.nn.Parameter(bias.view(-1), requires_grad=True)
        for conv in self.obj_preds:
            bias = conv.bias.view(self.num_anchors, -1)
            bias.data.fill_(-math.log((1 - self.prior_prob) / self.prior_prob))
            conv.bias = torch.nn.Parameter(bias.view(-1), requires_grad=True)

    def forward(  # pylint: disable=too-many-locals
        self, xin: List[torch.Tensor]
    ) -> torch.Tensor:
        """Defines the computation performed at every call.

        Args:
            xin (List[torch.Tensor]): Inputs from `RepPANNeck`.

        Returns:
            (torch.Tensor): The decoded output with the shape (B,D,85) where
            B is the batch size, D is the number of detections. The 85 columns
            consist of the following values:
            [x, y, w, h, conf, (cls_conf of the 80 COCO classes)].
        """
        outputs = []
        for i in range(self.num_layers):
            xin[i] = self.stems[i](xin[i])

            cls_feat = self.cls_convs[i](xin[i])
            cls_output = self.cls_preds[i](cls_feat)

            reg_feat = self.reg_convs[i](xin[i])
            reg_output = self.reg_preds[i](reg_feat)
            obj_output = self.obj_preds[i](reg_feat)

            output = torch.cat(
                [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
            )
            batch_size, _, hsize, wsize = output.shape
            output = (
                output.view(
                    batch_size, self.num_anchors, self.num_outputs, hsize, wsize
                )
                .permute(0, 1, 3, 4, 2)
                .contiguous()
            )
            if self.grid[i].shape[2:4] != output.shape[2:4]:
                y_grid, x_grid = torch.meshgrid(
                    [
                        torch.arange(hsize).to(self.stride.device),
                        torch.arange(wsize).to(self.stride.device),
                    ]
                )
                self.grid[i] = (
                    torch.stack((x_grid, y_grid), 2)
                    .view(1, self.num_anchors, hsize, wsize, 2)
                    .float()
                )
            output[..., 0:2] = (output[..., 0:2] + self.grid[i]) * self.stride[i]  # xy
            output[..., 2:4] = torch.exp(output[..., 2:4]) * self.stride[i]  # wh
            outputs.append(output.view(batch_size, -1, self.num_outputs))
        return torch.cat(outputs, 1)


def build_efficient_decoupled_head_layers(
    channels_list: List[int], num_anchors: int, num_classes: int
) -> nn.Sequential:
    """Creates the network layers for the Efficient Decoupled Head.

    Args:
        channels_list (List[int]): Number of input and output channels for the
            various Conv layers.
        num_anchors (int): Number of anchors.
        num_classes (int): Number of detectable classes.

    Returns:
        (nn.Sequential): A container of the network layers.
    """
    head_layers = nn.Sequential(
        # stem0
        Conv(
            in_channels=channels_list[6],
            out_channels=channels_list[6],
            kernel_size=1,
            stride=1,
        ),
        # cls_conv0
        Conv(
            in_channels=channels_list[6],
            out_channels=channels_list[6],
            kernel_size=3,
            stride=1,
        ),
        # reg_conv0
        Conv(
            in_channels=channels_list[6],
            out_channels=channels_list[6],
            kernel_size=3,
            stride=1,
        ),
        # cls_pred0
        nn.Conv2d(
            in_channels=channels_list[6],
            out_channels=num_classes * num_anchors,
            kernel_size=1,
        ),
        # reg_pred0
        nn.Conv2d(
            in_channels=channels_list[6], out_channels=4 * num_anchors, kernel_size=1
        ),
        # obj_pred0
        nn.Conv2d(
            in_channels=channels_list[6], out_channels=1 * num_anchors, kernel_size=1
        ),
        # stem1
        Conv(
            in_channels=channels_list[8],
            out_channels=channels_list[8],
            kernel_size=1,
            stride=1,
        ),
        # cls_conv1
        Conv(
            in_channels=channels_list[8],
            out_channels=channels_list[8],
            kernel_size=3,
            stride=1,
        ),
        # reg_conv1
        Conv(
            in_channels=channels_list[8],
            out_channels=channels_list[8],
            kernel_size=3,
            stride=1,
        ),
        # cls_pred1
        nn.Conv2d(
            in_channels=channels_list[8],
            out_channels=num_classes * num_anchors,
            kernel_size=1,
        ),
        # reg_pred1
        nn.Conv2d(
            in_channels=channels_list[8], out_channels=4 * num_anchors, kernel_size=1
        ),
        # obj_pred1
        nn.Conv2d(
            in_channels=channels_list[8], out_channels=1 * num_anchors, kernel_size=1
        ),
        # stem2
        Conv(
            in_channels=channels_list[10],
            out_channels=channels_list[10],
            kernel_size=1,
            stride=1,
        ),
        # cls_conv2
        Conv(
            in_channels=channels_list[10],
            out_channels=channels_list[10],
            kernel_size=3,
            stride=1,
        ),
        # reg_conv2
        Conv(
            in_channels=channels_list[10],
            out_channels=channels_list[10],
            kernel_size=3,
            stride=1,
        ),
        # cls_pred2
        nn.Conv2d(
            in_channels=channels_list[10],
            out_channels=num_classes * num_anchors,
            kernel_size=1,
        ),
        # reg_pred2
        nn.Conv2d(
            in_channels=channels_list[10], out_channels=4 * num_anchors, kernel_size=1
        ),
        # obj_pred2
        nn.Conv2d(
            in_channels=channels_list[10], out_channels=1 * num_anchors, kernel_size=1
        ),
    )
    return head_layers
