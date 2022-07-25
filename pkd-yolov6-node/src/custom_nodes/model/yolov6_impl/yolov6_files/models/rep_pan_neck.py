"""Rep-PAN Neck network.

Modifications:
- Hardcode RepVGGBlock
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from ..layers.common import RepBlock, RepVGGBlock, SimConv, Transpose


class RepPANNeck(nn.Module):  # pylint: disable=too-many-instance-attributes
    """RepPANNeck Module
    EfficientRep is the default backbone of this model.
    RepPANNeck has the balance of feature fusion ability and hardware efficiency.
    """

    def __init__(  # pylint: disable=invalid-name
        self,
        channels_list=None,
        num_repeats=None,
    ) -> None:
        super().__init__()

        assert channels_list is not None
        assert num_repeats is not None

        self.Rep_p4 = RepBlock(
            in_channels=channels_list[3] + channels_list[5],
            out_channels=channels_list[5],
            n=num_repeats[5],
            block=RepVGGBlock,
        )

        self.Rep_p3 = RepBlock(
            in_channels=channels_list[2] + channels_list[6],
            out_channels=channels_list[6],
            n=num_repeats[6],
            block=RepVGGBlock,
        )

        self.Rep_n3 = RepBlock(
            in_channels=channels_list[6] + channels_list[7],
            out_channels=channels_list[8],
            n=num_repeats[7],
            block=RepVGGBlock,
        )

        self.Rep_n4 = RepBlock(
            in_channels=channels_list[5] + channels_list[9],
            out_channels=channels_list[10],
            n=num_repeats[8],
            block=RepVGGBlock,
        )

        self.reduce_layer0 = SimConv(
            in_channels=channels_list[4],
            out_channels=channels_list[5],
            kernel_size=1,
            stride=1,
        )

        self.upsample0 = Transpose(
            in_channels=channels_list[5],
            out_channels=channels_list[5],
        )

        self.reduce_layer1 = SimConv(
            in_channels=channels_list[5],
            out_channels=channels_list[6],
            kernel_size=1,
            stride=1,
        )

        self.upsample1 = Transpose(
            in_channels=channels_list[6], out_channels=channels_list[6]
        )

        self.downsample2 = SimConv(
            in_channels=channels_list[6],
            out_channels=channels_list[7],
            kernel_size=3,
            stride=2,
        )

        self.downsample1 = SimConv(
            in_channels=channels_list[8],
            out_channels=channels_list[9],
            kernel_size=3,
            stride=2,
        )

    def forward(  # pylint: disable=too-many-locals
        self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> List[torch.Tensor]:  # pylint: disable=too-many-locals
        """Defines the computation performed at every call.

        Args:
            inputs (List[torch.Tensor]): Inputs from `EfficientRep`.

        Returns:
            (List[torch.Tensor]):
        """
        x_2, x_1, x_0 = inputs

        fpn_out_0 = self.reduce_layer0(x_0)
        upsample_feat_0 = self.upsample0(fpn_out_0)
        f_concat_layer_0 = torch.cat([upsample_feat_0, x_1], 1)
        f_out_0 = self.Rep_p4(f_concat_layer_0)

        fpn_out_1 = self.reduce_layer1(f_out_0)
        upsample_feat_1 = self.upsample1(fpn_out_1)
        f_concat_layer_1 = torch.cat([upsample_feat_1, x_2], 1)
        pan_out_2 = self.Rep_p3(f_concat_layer_1)

        down_feat1 = self.downsample2(pan_out_2)
        p_concat_layer1 = torch.cat([down_feat1, fpn_out_1], 1)
        pan_out_1 = self.Rep_n3(p_concat_layer1)

        down_feat0 = self.downsample1(pan_out_1)
        p_concat_layer2 = torch.cat([down_feat0, fpn_out_0], 1)
        pan_out_0 = self.Rep_n4(p_concat_layer2)

        outputs = [pan_out_2, pan_out_1, pan_out_0]

        return outputs
