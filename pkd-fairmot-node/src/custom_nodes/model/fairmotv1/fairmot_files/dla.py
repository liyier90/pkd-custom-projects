"""
Modifications include:
- Remove fill_fc_weights() and other fc weights initialisation
- Remove fill_up_weights()
- Remove code branch when head_conv <= 0
- Avoid using loading model zoo weights to create fc layer in DLA
- DLASeg:
  - Remove base_name, pretrained, and out_channel arguments
  - Add default value for final_kernel, last_level, and head_conv
  - Hardcode final_kernel to 1
  - Remove load_base_weights
- DLA:
  - Use BasicBlock only
  - Remove linear_root argument
  - Omit creating num_classes member variable
  - Remove _make_level()
  - Remove dilation argument from _make_conv_level()
"""

from typing import Dict, List

import numpy as np
import torch
from torch import nn

from custom_nodes.model.fairmotv1.fairmot_files.network_blocks import (
    BN_MOMENTUM,
    BasicBlock,
    DeformConv,
    Tree,
)


class DLASeg(nn.Module):
    """The encoder-decoder network comprising of:
    - ResNet-34 with an enhanced Deep Layer Aggregation (DLA) as backbone
    - CenterNet as the detection branch
    - FairMOT EmbeddingHead as the Re-ID branch

    Args:
        heads (Dict[str, int]): Configuration for the output channels for the
            various heads of the model.
        down_ratio (int): The downscale ratio from images to heatmap. In the
            case of FairMOT, this is 4.
        last_level (int): The last level of input feature fed into the
            upsampling block. Default is 5.
        head_conv (int): Number of channels in all heads. Default is 256.
    """

    def __init__(
        self,
        heads: Dict[str, int],
        down_ratio: int,
        last_level: int = 5,
        head_conv: int = 256,
    ) -> None:
        super().__init__()
        self.first_level = int(np.log2(down_ratio))
        self.last_level = last_level
        self.base = DLA([1, 1, 1, 2, 2, 1], [16, 32, 64, 128, 256, 512])
        channels = self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level :]))]
        self.dla_up = DLAUp(self.first_level, channels[self.first_level :], scales)

        self.ida_up = IDAUp(
            channels[self.first_level],
            channels[self.first_level : self.last_level],
            [2 ** i for i in range(self.last_level - self.first_level)],
        )

        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            head_network = nn.Sequential(
                nn.Conv2d(
                    channels[self.first_level],
                    head_conv,
                    kernel_size=3,
                    padding=1,
                    bias=True,
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    head_conv, classes, kernel_size=1, stride=1, padding=0, bias=True
                ),
            )
            # Sets the dict key as the member variable
            self.__setattr__(head, head_network)

    def forward(  # pylint: disable=invalid-name
        self, x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): Input from the previous layer.

        Returns:
            (Dict[str, torch.Tensor]): A dictionary of tensors with keys
            corresponding to `self.heads`.
        """
        layers = self.base(x)
        layers = self.dla_up(layers)

        y = [layers[i].clone() for i in range(self.last_level - self.first_level)]
        self.ida_up(y, 0, len(y))

        outputs = {}
        for head in self.heads:
            outputs[head] = getattr(self, head)(y[-1])
        return outputs


class DLA(nn.Module):
    """Deep Layer Aggregation to be used as the backbone of FairMOT's
    encoder-decoder network.

    Args:
        levels (List[int]): List of aggregation depths at various stages.
        channels (List[int]): List of number of channels at various stages.
        num_classes (int): Number of classes for classification. NOTE: Not used
            in FairMOT is needed to properly load the model weights.
        residual_root (bool): Flag to indicate if a residual layer should be
            used in the root block. Default is False.
    """

    def __init__(
        self,
        levels: List[int],
        channels: List[int],
        num_classes: int = 1000,
        residual_root: bool = False,
    ) -> None:
        super().__init__()
        # DLA-34 (used by FairMOT) uses basic blocks as the residual block
        block = BasicBlock
        self.channels = channels
        self.base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
        )
        self.level0 = self._make_conv_level(channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2
        )
        self.level2 = Tree(
            levels[2],
            block,
            channels[1],
            channels[2],
            2,
            level_root=False,
            root_residual=residual_root,
        )
        self.level3 = Tree(
            levels[3],
            block,
            channels[2],
            channels[3],
            2,
            level_root=True,
            root_residual=residual_root,
        )
        self.level4 = Tree(
            levels[4],
            block,
            channels[3],
            channels[4],
            2,
            level_root=True,
            root_residual=residual_root,
        )
        self.level5 = Tree(
            levels[5],
            block,
            channels[4],
            channels[5],
            2,
            level_root=True,
            root_residual=residual_root,
        )
        # Apparently not needed
        self.fc = nn.Conv2d(  # pylint: disable=invalid-name
            self.channels[-1],
            num_classes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    def forward(  # pylint: disable=invalid-name
        self, x: torch.Tensor
    ) -> List[torch.Tensor]:
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): Input from the previous layer.

        Returns:
            (List[torch.Tensor]): A list of tensors containing the output at
            every stage.
        """
        outputs = []
        x = self.base_layer(x)
        for i in range(6):
            x = getattr(self, f"level{i}")(x)
            outputs.append(x)
        return outputs

    @staticmethod
    def _make_conv_level(
        in_channels: int, out_channels: int, num_convs: int, stride: int = 1
    ) -> nn.Sequential:
        """Creates the networks for one of the earlier stages in DLA.

        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels producted by the
                convolution.
            num_convs (int): Number of convolution layers in the stage.
            stride (int): Stride of the convolution. Default is 1.

        Returns:
            (nn.Sequential): A sequential container of all the networks in the
            stage.
        """
        modules = []
        for i in range(num_convs):
            modules.extend(
                [
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        stride=stride if i == 0 else 1,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True),
                ]
            )
            in_channels = out_channels
        return nn.Sequential(*modules)


class DLAUp(nn.Module):
    """DLA upsample network.

    Args:
        start_level (int): The starting stage of this upsample network.
        channels (List[int]): List of number of channels at various stages.
        scales (List[int]): Scale factors at various stages.
        in_channels (List[int]): Number of channels in the input image at
            various stages.
    """

    def __init__(
        self,
        start_level: int,
        channels: List[int],
        scales: List[int],
        in_channels: List[int] = None,
    ) -> None:
        super().__init__()
        self.start_level = start_level
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        scales_arr = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(
                self,
                f"ida_{i}",
                IDAUp(channels[j], in_channels[j:], scales_arr[j:] // scales_arr[j]),
            )
            scales_arr[j + 1 :] = scales_arr[j]
            in_channels[j + 1 :] = [channels[j] for _ in channels[j + 1 :]]

    def forward(self, layers: List[torch.Tensor]) -> List[torch.Tensor]:
        """Defines the computation performed at every call.

        Args:
            layers (List[torch.Tensor]): Inputs from the various stages.

        Returns:
            (List[torch.Tensor]): A list of tensors containing the output at
            every stage.
        """
        out = [layers[-1]]  # start with 32
        for i in range(len(layers) - self.start_level - 1):
            ida = getattr(self, f"ida_{i}")
            ida(layers, len(layers) - i - 2, len(layers))
            out.insert(0, layers[-1])
        return out


class IDAUp(nn.Module):
    def __init__(self, o, channels, up_f):
        super().__init__()
        for i in range(1, len(channels)):
            c = channels[i]
            f = int(up_f[i])
            proj = DeformConv(c, o)
            node = DeformConv(o, o)

            up = nn.ConvTranspose2d(
                o,
                o,
                f * 2,
                stride=f,
                padding=f // 2,
                output_padding=0,
                groups=o,
                bias=False,
            )

            setattr(self, "proj_" + str(i), proj)
            setattr(self, "up_" + str(i), up)
            setattr(self, "node_" + str(i), node)

    def forward(self, layers, start_level, endp):
        for i in range(start_level + 1, endp):
            upsample = getattr(self, "up_" + str(i - start_level))
            project = getattr(self, "proj_" + str(i - start_level))
            layers[i] = upsample(project(layers[i]))
            node = getattr(self, "node_" + str(i - start_level))
            layers[i] = node(layers[i] + layers[i - 1])
