"""Darknet-53 backbone for JDE model.

Modifications include:
- Remove training related code such as:
    - classifier
    - loss_names
    - losses
    - test_emb
    - uniform initialisation of batch norm
- Refactor to remove unused code
    - enumerate in forward()
    - "maxpool" in _create_nodes
- Refactor for proper type hinting
    - renamed one of layer_i to layer_indices in forward()
- Refactor in _create_nodes to reduce the number of local variables
- Use the nn.Upsample instead of the custom one since it no longer gives
    deprecated warning
- Removed yolo_layer_count since layer member variable has been removed in
    YOLOLayer as it's not used
"""

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from custom_nodes.model.jdev1.jde_files.network_blocks import (  # Upsample,
    EmptyLayer,
    YOLOLayer,
)


class Darknet(nn.Module):
    """YOLOv3 object detection model.

    Args:
        cfg_dict (List[Dict[str, Any]]): Model architecture
            configurations.
        num_identities (int): TODO
    """

    def __init__(self, cfg_dict: List[Dict[str, Any]], num_identities: int) -> None:
        super().__init__()
        self.module_defs = cfg_dict
        self.module_defs[0]["nID"] = num_identities
        self.img_size = [
            int(self.module_defs[0]["width"]),
            int(self.module_defs[0]["height"]),
        ]
        self.emb_dim = int(self.module_defs[0]["embedding_dim"])
        self.hyperparams, self.module_list = _create_modules(self.module_defs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pylint: disable=invalid-name
        """Defines the computation performed at every call.

        Args:
            inputs (torch.Tensor): Input from the previous layer.

        Returns:
            (torch.Tensor): A dictionary of tensors with keys corresponding to
                `self.out_features`.
        """
        layer_outputs: List[torch.Tensor] = []
        outputs = []

        for module_def, module in zip(self.module_defs, self.module_list):
            module_type = module_def["type"]
            if module_type in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif module_type == "route":
                layer_indices = list(map(int, module_def["layers"].split(",")))
                if len(layer_indices) == 1:
                    x = layer_outputs[layer_indices[0]]
                else:
                    x = torch.cat([layer_outputs[i] for i in layer_indices], 1)
            elif module_type == "shortcut":
                x = layer_outputs[-1] + layer_outputs[int(module_def["from"])]
            elif module_type == "yolo":
                x = module[0](x, self.img_size)
                outputs.append(x)
            layer_outputs.append(x)

        return torch.cat(outputs, 1)


def _create_modules(
    module_defs: List[Dict[str, Any]]
) -> Tuple[Dict[str, Any], nn.ModuleList]:
    """Constructs module list of layer blocks from module configuration in
    `module_defs`

    NOTE: Each `module_def` in `module_defs` is parsed as a dictionary
    containing string values. As a result, "1" can sometimes represent True
    instead of the number of the key. We try to do == "1" instead of implicit
    boolean by converting it to int.

    Args:
        module_defs (List[Dict[str, Any]]): A list of module definitions.

    Returns:
        (Tuple[Dict[str, Any], nn.ModuleList]): A tuple containing a dictionary
            of model hyperparameters and a ModuleList containing the modules
            in the model.
    """
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams["channels"])]
    module_list = nn.ModuleList()
    for i, module_def in enumerate(module_defs):
        module_type = module_def["type"]
        modules = nn.Sequential()
        if module_type == "convolutional":
            has_batch_norm = module_def["batch_normalize"] == "1"
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            modules.add_module(
                f"conv_{i}",
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=(kernel_size - 1) // 2 if module_def["pad"] == "1" else 0,
                    bias=not has_batch_norm,
                ),
            )
            if has_batch_norm:
                modules.add_module(f"batch_norm_{i}", nn.BatchNorm2d(filters))
            if module_def["activation"] == "leaky":
                modules.add_module(f"leaky_{i}", nn.LeakyReLU(0.1))
        elif module_type == "upsample":
            modules.add_module(
                f"upsample_{i}",
                nn.Upsample(scale_factor=int(module_def["stride"]), mode="nearest"),
            )
        elif module_type == "route":
            filters = sum(
                [
                    output_filters[i + 1 if i > 0 else i]
                    for i in map(int, module_def["layers"].split(","))
                ]
            )
            modules.add_module(f"route_{i}", EmptyLayer())
        elif module_type == "shortcut":
            filters = output_filters[int(module_def["from"])]
            modules.add_module(f"shortcut_{i}", EmptyLayer())
        elif module_type == "yolo":
            # Extract anchors
            anchor_dims = iter(map(float, module_def["anchors"].split(",")))
            # This lets us do pairwise() with no overlaps
            anchors = list(zip(anchor_dims, anchor_dims))
            anchors = [anchors[i] for i in map(int, module_def["mask"].split(","))]
            # Define detection layer
            modules.add_module(
                f"yolo_{i}",
                YOLOLayer(
                    anchors,
                    int(module_def["classes"]),
                    int(hyperparams["nID"]),
                    int(hyperparams["embedding_dim"]),
                ),
            )

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list
