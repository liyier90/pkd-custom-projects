"""Common network blocks.

Modifications:
- Remove unused classes: SiLU, Concat, RealVGGBlock, ScaleLayer, LinearAddBlock,
  IdentityBasedConv1x1, BNAndPadLayer, DBBBlock, DiverseBranchBlock, and
  DetectBackend
- Remove unused function: conv_bn_v2
- Remove dbb_transforms import
- In RepVGGBlock:
  - Hardcode use_se=False,
"""

import warnings
from typing import Any, Callable, Optional, Tuple, Union, no_type_check

import numpy as np
import torch
import torch.nn as nn


class Conv(nn.Module):
    """Normal Conv with SiLU activation"""

    def __init__(  # pylint: disable=invalid-name,too-many-arguments
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call."""
        return self.act(self.bn(self.conv(inputs)))

    def forward_fuse(self, inputs: torch.Tensor) -> torch.Tensor:
        """The computation performed at every call when conv and batch norm
        layers are fused.
        """
        return self.act(self.conv(inputs))


class SimConv(nn.Module):
    """Normal Conv with ReLU activation"""

    def __init__(  # pylint: disable=invalid-name,too-many-arguments
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call."""
        return self.act(self.bn(self.conv(inputs)))

    def forward_fuse(self, inputs: torch.Tensor) -> torch.Tensor:
        """The computation performed at every call when conv and batch norm
        layers are fused.
        """
        return self.act(self.conv(inputs))


class SimSPPF(nn.Module):
    """Simplified SPPF with ReLU activation"""

    def __init__(  # pylint: disable=invalid-name
        self, in_channels: int, out_channels: int, kernel_size: int = 5
    ) -> None:
        super().__init__()
        hidden_channels = in_channels // 2
        self.cv1 = SimConv(in_channels, hidden_channels, 1, 1)
        self.cv2 = SimConv(hidden_channels * 4, out_channels, 1, 1)
        self.m = nn.MaxPool2d(
            kernel_size=kernel_size, stride=1, padding=kernel_size // 2
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call."""
        inputs = self.cv1(inputs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            output_1 = self.m(inputs)
            output_2 = self.m(output_1)
            return self.cv2(
                torch.cat([inputs, output_1, output_2, self.m(output_2)], 1)
            )


class Transpose(nn.Module):
    """Normal Transpose, default for upsampling"""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 2, stride: int = 2
    ) -> None:
        super().__init__()
        self.upsample_transpose = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=True,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call."""
        return self.upsample_transpose(inputs)


class RepVGGBlock(nn.Module):  # pylint: disable=too-many-instance-attributes
    """RepVGGBlock is a basic rep-style block, including training and deploy status
    This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    def __init__(  # pylint: disable=invalid-name,too-many-arguments
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        padding_mode: str = "zeros",
        deploy: bool = False,
    ) -> None:
        """Initializes the class.

        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution.
            kernel_size (int): Size of the convolving kernel. Default: 3
            stride (int): Stride of the convolution. Default: 1
            padding (int): Zero-padding added to both sides of the input.
                Default: 1
            dilation (int): Spacing between kernel elements. Default: 1
            groups (int): Number of blocked connections from input channels to
                output channels. Default: 1
            padding_mode (string): Default: "zeros"
            deploy (bool): Whether to be deploy status or training status.
                Default: False
        """
        super().__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()
        self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True,
                padding_mode=padding_mode,
            )
        else:
            self.rbr_identity = (
                nn.BatchNorm2d(num_features=in_channels)
                if out_channels == in_channels and stride == 1
                else None
            )
            # Type hinting doesn't work well for accessing modules in a
            # nn.Sequential object. Cast to Any to suppress errors in
            # switch_to_deploy()
            self.rbr_dense: Any = conv_bn(
                in_channels, out_channels, kernel_size, stride, padding, groups
            )
            self.rbr_1x1 = conv_bn(
                in_channels, out_channels, 1, stride, padding_11, groups
            )
        self.id_tensor: torch.Tensor

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward process"""
        if hasattr(self, "rbr_reparam"):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(
            self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)
        )

    def get_equivalent_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the equivalent kernel and bias of `rbr_dense`, `rbr_1x1`, and
        `rbr_identity` to `rbr_reparam`. Used when switching to deploy-mode.
        """
        kernel_3x3, bias_3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel_1x1, bias_1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernel_id, bias_id = self._fuse_bn_tensor(self.rbr_identity)
        return (
            kernel_3x3 + self.pad_1x1_to_3x3_tensor(kernel_1x1) + kernel_id,
            bias_3x3 + bias_1x1 + bias_id,
        )

    def switch_to_deploy(self) -> None:
        """Switches block to deploy mode. Removes `rbr_identity`, `rbr_dense`,
        `rbr_1x1`, and `id_tensor`. Replace with `rbr_reparam`.
        """
        if hasattr(self, "rbr_reparam"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            in_channels=self.rbr_dense.conv.in_channels,
            out_channels=self.rbr_dense.conv.out_channels,
            kernel_size=self.rbr_dense.conv.kernel_size,
            stride=self.rbr_dense.conv.stride,
            padding=self.rbr_dense.conv.padding,
            dilation=self.rbr_dense.conv.dilation,
            groups=self.rbr_dense.conv.groups,
            bias=True,
        )
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias  # type: ignore
        for para in self.parameters():
            para.detach_()
        self.__delattr__("rbr_dense")
        self.__delattr__("rbr_1x1")
        if hasattr(self, "rbr_identity"):
            self.__delattr__("rbr_identity")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")
        self.deploy = True

    # Type hinting does not play well with accessing potentially NoneType
    # attributes
    @no_type_check
    def _fuse_bn_tensor(
        self, branch: Optional[Union[nn.Sequential, nn.BatchNorm2d]]
    ) -> Union[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]]:
        """Fuses the conv and batchnorm layers."""
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros(
                    (self.in_channels, input_dim, 3, 3), dtype=np.float32
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        # Similar form as w_bn from https://nenadmarkus.com/p/fusing-batchnorm-and-conv/
        kernel_bn = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * kernel_bn, beta - running_mean * gamma / std

    @staticmethod
    def pad_1x1_to_3x3_tensor(
        kernel_1x1: Optional[torch.Tensor],
    ) -> Union[int, torch.Tensor]:
        """Pads a 1x1 kernel to 3x3. Returns zero if kernel_1x1 is None."""
        if kernel_1x1 is None:
            return 0
        return torch.nn.functional.pad(kernel_1x1, [1, 1, 1, 1])


class RepBlock(nn.Module):
    """
    RepBlock is a stage block with rep-style basic block
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n: int = 1,
        block: Callable = RepVGGBlock,
    ) -> None:
        super().__init__()
        self.conv1 = block(in_channels, out_channels)
        self.block = (
            nn.Sequential(*(block(out_channels, out_channels) for _ in range(n - 1)))
            if n > 1
            else None
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call."""
        inputs = self.conv1(inputs)
        if self.block is not None:
            inputs = self.block(inputs)
        return inputs


def conv_bn(  # pylint: disable=too-many-arguments
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    padding: int,
    groups: int = 1,
) -> nn.Sequential:
    """Basic cell for rep-style block, including conv and bn."""
    result = nn.Sequential()
    result.add_module(
        "conv",
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        ),
    )
    result.add_module("bn", nn.BatchNorm2d(num_features=out_channels))
    return result
