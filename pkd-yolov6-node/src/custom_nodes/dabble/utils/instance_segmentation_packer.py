# Copyright 2022 AI Singapore
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Pack instance segmentation results into COCO format
"""

import logging
from typing import Any, Dict, List

from peekingduck.pipeline.nodes.draw.utils.general import (
    project_points_onto_original_image,
)

from .constants import COCO_CATEGORY_DICTIONARY, COCO_LABEL_DICTIONARY
from .core import encode_mask_results
from .detection_packer import DetectionPacker


class InstanceSegmentationPacker:  # pylint: disable=too-few-public-methods
    """Packs the outputs from PeekingDuck's Instance Segmentation models into COCO's
    evaluation format.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self.bbox_labels_list = sorted(list(COCO_LABEL_DICTIONARY.keys()))

    def pack(
        self,
        model_predictions: List[Dict[str, Any]],
        inputs: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Appends new model prediction to the provided `model_predictions`."""
        encoded_masks = encode_mask_results(inputs["masks"])

        for bbox, bbox_label, mask, mask_score in zip(
            inputs["bboxes"],
            inputs["bbox_labels"],
            encoded_masks,
            inputs["bbox_scores"],
        ):
            # segms must be run length encoded
            if isinstance(mask["counts"], bytes):
                mask["counts"] = mask["counts"].decode()
            bbox_label_index = COCO_CATEGORY_DICTIONARY[bbox_label]
            bbox = project_points_onto_original_image(bbox, inputs["img_size"])
            model_predictions.append(
                {
                    "image_id": int(inputs["img_id"]),
                    "category_id": int(bbox_label_index),
                    "score": float(mask_score),
                    "segmentation": mask,
                    "bbox": DetectionPacker.xyxy2tlwh(bbox),
                }
            )

        return model_predictions
