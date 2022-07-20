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
Pack object detection results into COCO format
"""

import logging
from typing import Any, Dict, List

import numpy as np
from peekingduck.pipeline.nodes.draw.utils.general import (
    project_points_onto_original_image,
)

from .constants import COCO_CATEGORY_DICTIONARY


class DetectionPacker:  # pylint: disable=too-few-public-methods
    """Packs the outputs from PeekingDuck's object detection models into COCO's
    evaluation format.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def pack(
        self, model_predictions: List[Dict[str, Any]], inputs: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Appends new model prediction to the provided `model_predictions`."""
        img_id = inputs["img_id"]
        img_size = inputs["img_size"]
        bboxes = inputs["bboxes"]
        bbox_scores = inputs["bbox_scores"]
        bbox_labels = inputs["bbox_labels"]

        for bbox, bbox_label, bbox_score in zip(bboxes, bbox_labels, bbox_scores):
            bbox_label_index = COCO_CATEGORY_DICTIONARY[bbox_label]
            bbox = project_points_onto_original_image(bbox, img_size)

            model_predictions.append(
                {
                    "image_id": int(img_id),
                    "category_id": int(bbox_label_index),
                    "bbox": self.xyxy2tlwh(bbox),
                    "score": bbox_score,
                }
            )

        return model_predictions

    @staticmethod
    def xyxy2tlwh(xyxy: np.ndarray) -> List[int]:
        """Converts bounding box from (x1, y1, x2, y2), where (x1, y1) is top
        left and (x2, y2) is bottom right, to (t, l, w, h), where (t, l) is top
        left and (w, h) is width and height of the bbox."""
        return [
            xyxy[0][0],
            xyxy[0][1],
            xyxy[1][0] - xyxy[0][0],
            xyxy[1][1] - xyxy[0][1],
        ]
