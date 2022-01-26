"""
Modifications include:
- Remove fill_fc_weights() and other fc weights initialisation
- Remove fill_up_weights()
- Remove code branch when head_conv <= 0
- Avoid using loading model zoo weights to create fc layer in DLA
"""

import numpy as np
from torch import nn

from custom_nodes.model.fairmotv1.fairmot_files.network_blocks import (
    BN_MOMENTUM,
    BasicBlock,
    DeformConv,
    Tree,
)


class DLASeg(nn.Module):
    def __init__(
        self,
        base_name,
        heads,
        pretrained,
        down_ratio,
        final_kernel,
        last_level,
        head_conv,
        out_channel=0,
    ):
        super().__init__()
        assert down_ratio in [2, 4, 8, 16]
        self.first_level = int(np.log2(down_ratio))
        self.last_level = last_level
        self.base = DLA(
            [1, 1, 1, 2, 2, 1], [16, 32, 64, 128, 256, 512], block=BasicBlock
        )
        channels = self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level :]))]
        self.dla_up = DLAUp(self.first_level, channels[self.first_level :], scales)

        if out_channel == 0:
            out_channel = channels[self.first_level]

        self.ida_up = IDAUp(
            out_channel,
            channels[self.first_level : self.last_level],
            [2 ** i for i in range(self.last_level - self.first_level)],
        )

        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            fc = nn.Sequential(
                nn.Conv2d(
                    channels[self.first_level],
                    head_conv,
                    kernel_size=3,
                    padding=1,
                    bias=True,
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    head_conv,
                    classes,
                    kernel_size=final_kernel,
                    stride=1,
                    padding=final_kernel // 2,
                    bias=True,
                ),
            )
            self.__setattr__(head, fc)

    def forward(self, x):
        x = self.base(x)
        x = self.dla_up(x)

        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))

        z = {}
        for head in self.heads:
            z[head] = self.__getattr__(head)(y[-1])
        return z

    def load_base_weights(self, base_weights_path):
        self.base.load_pretrained_model(base_weights_path)


class DLA(nn.Module):
    def __init__(
        self,
        levels,
        channels,
        num_classes=1000,
        block=BasicBlock,
        residual_root=False,
        linear_root=False,
    ):
        super().__init__()
        self.channels = channels
        self.num_classes = num_classes
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
        self.fc = nn.Conv2d(
            self.channels[-1],
            self.num_classes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

    def _make_level(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.MaxPool2d(stride, stride=stride),
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample))
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend(
                [
                    nn.Conv2d(
                        inplanes,
                        planes,
                        kernel_size=3,
                        stride=stride if i == 0 else 1,
                        padding=dilation,
                        bias=False,
                        dilation=dilation,
                    ),
                    nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True),
                ]
            )
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        y = []
        x = self.base_layer(x)
        for i in range(6):
            x = getattr(self, "level{}".format(i))(x)
            y.append(x)
        return y


class DLAUp(nn.Module):
    def __init__(self, startp, channels, scales, in_channels=None):
        super().__init__()
        self.startp = startp
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(
                self,
                "ida_{}".format(i),
                IDAUp(channels[j], in_channels[j:], scales[j:] // scales[j]),
            )
            scales[j + 1 :] = scales[j]
            in_channels[j + 1 :] = [channels[j] for _ in channels[j + 1 :]]

    def forward(self, layers):
        out = [layers[-1]]  # start with 32
        for i in range(len(layers) - self.startp - 1):
            ida = getattr(self, "ida_{}".format(i))
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

    def forward(self, layers, startp, endp):
        for i in range(startp + 1, endp):
            upsample = getattr(self, "up_" + str(i - startp))
            project = getattr(self, "proj_" + str(i - startp))
            layers[i] = upsample(project(layers[i]))
            node = getattr(self, "node_" + str(i - startp))
            layers[i] = node(layers[i] + layers[i - 1])
