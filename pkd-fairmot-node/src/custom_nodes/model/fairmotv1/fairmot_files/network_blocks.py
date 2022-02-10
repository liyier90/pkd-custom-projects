"""Modifications:
- Rename planes to channels to be consistent to Conv2d argument naming
- Rearrange argument order in Tree
"""

from typing import List

import torch
from torch import nn

from custom_nodes.model.fairmotv1.fairmot_files.dcn import DCNv2

BN_MOMENTUM = 0.1


class BasicBlock(nn.Module):
    """Basic residual block structure used by DLA.

    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the block.
        stride (int): Stride of the convolution. Default is 1.
        dilation (int): Spacing between kernel elements. Default is 1.
    """

    def __init__(
        self, in_channels: int, out_channels: int, stride: int = 1, dilation: int = 1
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            bias=False,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=dilation,
            bias=False,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM)
        self.stride = stride

    def forward(
        self, inputs: torch.Tensor, residual: torch.Tensor = None
    ) -> torch.Tensor:
        """Defines the computation performed at every call.

        Args:
            inputs (torch.Tensor): Input from the previous layer.

        Returns:
            (torch.Tensor): The output tensor of the block.
        """
        if residual is None:
            residual = inputs

        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class DeformConv(nn.Module):
    """Performs deformable convolution followed by batch norm.

    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.actf = nn.Sequential(
            nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM), nn.ReLU(inplace=True)
        )
        self.conv = DCNv2(
            in_channels,
            out_channels,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            dilation=1,
            deformable_groups=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=invalid-name
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): Input from the previous layer.

        Returns:
            (torch.Tensor): The output tensor of the block.
        """
        x = self.conv(x)
        x = self.actf(x)
        return x


class Root(nn.Module):
    """Root node in Hierarchical Deep Aggregation.

    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the block.
        kernel_size (int): Size of the convolving kernel, used for calculating
            padding size only.
        residual (bool): Flag to indicate if a residual layer should be used.
    """

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, residual: bool
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            stride=1,
            bias=False,
            padding=(kernel_size - 1) // 2,
        )
        self.bn = nn.BatchNorm2d(  # pylint: disable=invalid-name
            out_channels, momentum=BN_MOMENTUM
        )
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """Defines the computation performed at every call.

        Args:
            inputs (List[torch.Tensor]): Inputs from the previous layers.

        Returns:
            (torch.Tensor): The output tensor of the block.
        """
        children = inputs
        out = self.conv(torch.cat(inputs, 1))
        out = self.bn(out)
        if self.residual:
            out += children[0]
        out = self.relu(out)

        return out


class Tree(nn.Module):
    """Tree node used to construct the structure of Hierarchical Deep
    Aggregation (HDA).

    Args;
        level (int): The stage number of this node in HDA.
        block (BasicBlock): The type of residual block used.
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the block.
        stride (int): Stride of the convolution.
        level_root (bool): Flag to indicate is this is the root node for the
            stage.
        dilation (int): Spacing between kernel elements. Default is 1.
        root_dim (int): Number of channels in the input image to the Root node.
        root_kernel_size (int): Size of the convolving kernel in the Root node.
        residual_root (bool): Flag to indicate if a residual layer should be
            used in the Root node. Default is False.
    """

    def __init__(
        self,
        level: int,
        block: "BasicBlock",
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        level_root: bool = False,
        dilation: int = 1,
        root_dim: int = 0,
        root_kernel_size: int = 1,
        root_residual: bool = False,
    ) -> None:
        super().__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if level == 1:
            self.tree1 = block(in_channels, out_channels, stride, dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1, dilation=dilation)
        else:
            self.tree1 = Tree(
                level - 1,
                block,
                in_channels,
                out_channels,
                stride,
                dilation=dilation,
                root_dim=0,
                root_kernel_size=root_kernel_size,
                root_residual=root_residual,
            )
            self.tree2 = Tree(
                level - 1,
                block,
                out_channels,
                out_channels,
                dilation=dilation,
                root_dim=root_dim + out_channels,
                root_kernel_size=root_kernel_size,
                root_residual=root_residual,
            )
        if level == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size, root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = level
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, bias=False
                ),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
            )

    def forward(
        self,
        inputs: torch.Tensor,
        residual: torch.Tensor = None,
        children: List[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Defines the computation performed at every call.

        Args:
            inputs (torch.Tensor): Input from the previous layer.
            residual (torch.Tensor): Residual from the connected node.
            children (List[torch.Tensor]): Inputs from child nodes of the
                current node.

        Returns:
            (torch.Tensor): The output tensor of the block.
        """
        children = [] if children is None else children
        bottom = self.downsample(inputs) if self.downsample else inputs
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        out_1 = self.tree1(inputs, residual)
        if self.levels == 1:
            out_2 = self.tree2(out_1)
            out = self.root([out_2, out_1] + children)
        else:
            children.append(out_1)
            out = self.tree2(out_1, children=children)

        return out
