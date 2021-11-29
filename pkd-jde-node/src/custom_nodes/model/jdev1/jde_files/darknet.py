from collections import OrderedDict

import torch
import torch.nn as nn

from .network_blocks import EmptyLayer, Upsample, YOLOLayer

# try:
#     from utils.syncbn import SyncBN

#     batch_norm = SyncBN
# except ImportError:
#     batch_norm = nn.BatchNorm2d

batch_norm = nn.BatchNorm2d


class Darknet(nn.Module):
    """YOLOv3 object detection model"""

    def __init__(self, cfg_dict, nID=0, test_emb=False):
        super(Darknet, self).__init__()
        # if isinstance(cfg_dict, str):
        #     cfg_dict = parse_model_cfg(cfg_dict)
        self.module_defs = cfg_dict
        self.module_defs[0]["nID"] = nID
        self.img_size = [
            int(self.module_defs[0]["width"]),
            int(self.module_defs[0]["height"]),
        ]
        self.emb_dim = int(self.module_defs[0]["embedding_dim"])
        self.hyperparams, self.module_list = create_modules(self.module_defs)
        self.loss_names = ["loss", "box", "conf", "id", "nT"]
        self.losses = OrderedDict()
        for ln in self.loss_names:
            self.losses[ln] = 0
        self.test_emb = test_emb

        self.classifier = nn.Linear(self.emb_dim, nID) if nID > 0 else None

    def forward(self, x, targets=None, targets_len=None):
        self.losses = OrderedDict()
        for ln in self.loss_names:
            self.losses[ln] = 0
        is_training = (targets is not None) and (not self.test_emb)
        # img_size = x.shape[-1]
        layer_outputs = []
        output = []

        for i, (module_def, module) in enumerate(
            zip(self.module_defs, self.module_list)
        ):
            mtype = module_def["type"]
            if mtype in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif mtype == "route":
                layer_i = [int(x) for x in module_def["layers"].split(",")]
                if len(layer_i) == 1:
                    x = layer_outputs[layer_i[0]]
                else:
                    x = torch.cat([layer_outputs[i] for i in layer_i], 1)
            elif mtype == "shortcut":
                layer_i = int(module_def["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif mtype == "yolo":
                if is_training:  # get loss
                    targets = [targets[i][: int(l)] for i, l in enumerate(targets_len)]
                    x, *losses = module[0](x, self.img_size, targets, self.classifier)
                    for name, loss in zip(self.loss_names, losses):
                        self.losses[name] += loss
                elif self.test_emb:
                    if targets is not None:
                        targets = [
                            targets[i][: int(l)] for i, l in enumerate(targets_len)
                        ]
                    x = module[0](
                        x, self.img_size, targets, self.classifier, self.test_emb
                    )
                else:  # get detections
                    x = module[0](x, self.img_size)
                output.append(x)
            layer_outputs.append(x)

        if is_training:
            self.losses["nT"] /= 3
            output = [o.squeeze() for o in output]
            return sum(output), torch.Tensor(list(self.losses.values())).cuda()
        elif self.test_emb:
            return torch.cat(output, 0)
        return torch.cat(output, 1)


def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams["channels"])]
    module_list = nn.ModuleList()
    yolo_layer_count = 0
    for i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def["type"] == "convolutional":
            bn = int(module_def["batch_normalize"])
            filters = int(module_def["filters"])
            kernel_size = int(module_def["size"])
            pad = (kernel_size - 1) // 2 if int(module_def["pad"]) else 0
            modules.add_module(
                "conv_%d" % i,
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def["stride"]),
                    padding=pad,
                    bias=not bn,
                ),
            )
            if bn:
                after_bn = batch_norm(filters)
                modules.add_module("batch_norm_%d" % i, after_bn)
                # BN is uniformly initialized by default in pytorch 1.0.1.
                # In pytorch>1.2.0, BN weights are initialized with constant 1,
                # but we find with the uniform initialization the model converges faster.
                nn.init.uniform_(after_bn.weight)
                nn.init.zeros_(after_bn.bias)
            if module_def["activation"] == "leaky":
                modules.add_module("leaky_%d" % i, nn.LeakyReLU(0.1))

        elif module_def["type"] == "maxpool":
            kernel_size = int(module_def["size"])
            stride = int(module_def["stride"])
            if kernel_size == 2 and stride == 1:
                modules.add_module("_debug_padding_%d" % i, nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(
                kernel_size=kernel_size,
                stride=stride,
                padding=int((kernel_size - 1) // 2),
            )
            modules.add_module("maxpool_%d" % i, maxpool)

        elif module_def["type"] == "upsample":
            upsample = Upsample(scale_factor=int(module_def["stride"]))
            modules.add_module("upsample_%d" % i, upsample)

        elif module_def["type"] == "route":
            layers = [int(x) for x in module_def["layers"].split(",")]
            filters = sum([output_filters[i + 1 if i > 0 else i] for i in layers])
            modules.add_module("route_%d" % i, EmptyLayer())

        elif module_def["type"] == "shortcut":
            filters = output_filters[int(module_def["from"])]
            modules.add_module("shortcut_%d" % i, EmptyLayer())

        elif module_def["type"] == "yolo":
            anchor_idxs = [int(x) for x in module_def["mask"].split(",")]
            # Extract anchors
            anchors = [float(x) for x in module_def["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            nC = int(module_def["classes"])  # number of classes
            img_size = (int(hyperparams["width"]), int(hyperparams["height"]))
            # Define detection layer
            yolo_layer = YOLOLayer(
                anchors,
                nC,
                int(hyperparams["nID"]),
                int(hyperparams["embedding_dim"]),
                img_size,
                yolo_layer_count,
            )
            modules.add_module("yolo_%d" % i, yolo_layer)
            yolo_layer_count += 1

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list
