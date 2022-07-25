"""YOLOv6 model.

Modifications include:
- Removed `training_mode`, hard code to use RepVGGBlock only
- Removed `build_model()`
"""

import math
from typing import Any, Callable, Dict, Tuple

import torch
import torch.nn as nn

from ..utils.torch_utils import initialize_weights
from .efficient_decoupled_head import (
    EfficientDecoupledHead,
    build_efficient_decoupled_head_layers,
)
from .efficientrep import EfficientRep
from .rep_pan_neck import RepPANNeck


class YOLOv6(nn.Module):
    """YOLOv6 model with backbone, neck and head.

    The default parts are EfficientRep Backbone, Rep-PAN and
    Efficient Decoupled Head.
    """

    def __init__(self, config, channels=3, num_classes=None, anchors=None) -> None:
        super().__init__()
        # Build network
        self.backbone, self.neck, self.detect = build_network(
            config,
            channels,
            num_classes,
            anchors,
            config["model"]["head"]["num_layers"],
        )

        # Init Detect head
        self.stride = self.detect.stride
        self.detect.i = config["model"]["head"]["begin_indices"]
        self.detect.f = config["model"]["head"]["out_indices"]
        self.detect.initialize_biases()

        # Init weights
        initialize_weights(self)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call.

        Args:
            inputs: Input images.

        Returns:
            (torch.Tensor): The decoded output with the shape (B,D,85) where
            B is the batch size, D is the number of detections. The 85 columns
            consist of the following values:
            [x, y, w, h, conf, (cls_conf of the 80 COCO classes)].
        """
        inputs = self.backbone(inputs)
        inputs = self.neck(inputs)
        output = self.detect(inputs)
        return output

    def _apply(self, fn: Callable) -> nn.Module:
        """Applies the generic function to all the modules and submodules.

        For YOLOv6, this ensures that stride and grid are moved to the same
        device as the model.
        """
        self = super()._apply(fn)  # pylint: disable=self-cls-assignment
        self.detect.stride = fn(self.detect.stride)
        self.detect.grid = list(map(fn, self.detect.grid))
        return self


def make_divisible(value: int, divisor: int) -> int:
    """Upward revision of ``value`` to make it evenly divisible by the
    ``divisor``.
    """
    return math.ceil(value / divisor) * divisor


def build_network(
    config: Dict[str, Any],
    channels: int,
    num_classes: int,
    anchors: int,
    num_layers: int,
) -> Tuple[EfficientRep, RepPANNeck, EfficientDecoupledHead]:
    """Creates the backbone, neck and head networks for the YOLOv6 model.

    Args:
        config (Dict[str, Any]): Network configuration options.
        channels (int): Number of input channels.
        num_classes (int): Number of detectable classes.
        anchors (int): Number of anchors.
        num_layers (int): Number of detection layers in the head network.

    Returns:
        (Tuple[EfficientRep, RepPANNeck, EfficientDecoupledHead]): The backbone,
        neck, and head networks.
    """
    model = config["model"]  # model configs
    num_repeat = [
        (max(round(i * model["depth_multiple"]), 1) if i > 1 else i)
        for i in (model["backbone"]["num_repeats"] + model["neck"]["num_repeats"])
    ]
    channels_list = [
        make_divisible(i * model["width_multiple"], 8)
        for i in (model["backbone"]["out_channels"] + model["neck"]["out_channels"])
    ]

    backbone = EfficientRep(channels, channels_list, num_repeat)
    neck = RepPANNeck(channels_list, num_repeat)
    head_layers = build_efficient_decoupled_head_layers(
        channels_list, model["head"]["anchors"], num_classes
    )

    head = EfficientDecoupledHead(
        num_classes, anchors, num_layers, head_layers=head_layers
    )

    return backbone, neck, head
