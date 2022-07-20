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
Pack pose estimation results into COCO format
"""

import logging
from typing import Any, Dict, List

import numpy as np
from peekingduck.pipeline.nodes.draw.utils.general import (
    project_points_onto_original_image,
)


class KeypointPacker:  # pylint: disable=too-few-public-methods
    """
    Pack the outputs from PeekingDuck's pose estimation models into COCO's
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
        keypoints = inputs["keypoints"]
        keypoint_scores = inputs["keypoint_scores"]

        for keypoint, score in zip(keypoints, keypoint_scores):
            keypoint = project_points_onto_original_image(keypoint, img_size)
            pred = np.append(keypoint, np.ones((len(keypoint), 1)), axis=1)
            pred = list(pred.flat)

            model_predictions.append(
                {
                    "image_id": int(img_id),
                    "category_id": 1,
                    "keypoints": pred,
                    "score": sum(score) / len(score),
                }
            )

        return model_predictions
