# Copyright 2021 AI Singapore
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
Mosaics area bounded by bounding boxes over detected object
"""

from typing import Any, Dict, List

import cv2
import numpy as np
from peekingduck.pipeline.nodes.node import AbstractNode


class Node(AbstractNode):  # pylint: disable=too-few-public-methods
    """Mosaics areas bounded by bounding boxes on image.

    The mosaic bbox node helps to anonymize detected objects by pixelating the
    areas bounded by bounding boxes in an image.

    Inputs:

        |img|

        |bboxes|

    Outputs:
        |img|

    Configs:
        mosaic_level (:obj:`int`): **default = 7**
            defines the resolution of a mosaic filter (width x height). The
            number corresponds to the number of rows and columns used to create
            a mosaic. For example, the default setting (mosaic_level: 7) creates
            a 7 x 7 mosaic filter. Increasing the number increases the intensity
            of pixelation over an area.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)

        self.mosaic_level = self.config["mosaic_level"]

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        mosaic_img = self.mosaic_bbox(inputs["img"], inputs["bboxes"])
        outputs = {"img": mosaic_img}
        return outputs

    def mosaic_bbox(self, image: np.ndarray, bboxes: List[np.ndarray]) -> np.ndarray:
        """Mosaics areas bounded by bounding boxes on image

        Args:
            image (np.ndarray): image in numpy array
            bboxes (np.ndarray): numpy array of detected bboxes

        Returns:
            image (np.ndarray): image in numpy array
        """
        height, width = image.shape[:2]
        orig_image = image.copy()

        for bbox in bboxes:
            rows = slice(int(max(0, bbox[1]) * height), int(min(1, bbox[3]) * height))
            cols = slice(int(max(0, bbox[0]) * width), int(min(1, bbox[2]) * width))

            # slice the image to get the area bounded by bbox
            image[rows, cols] = self.mosaic(orig_image[rows, cols])

        return image

    def mosaic(self, image: np.ndarray) -> np.ndarray:
        """Mosaics a given input image

        Args:
            image (np.ndarray): image in numpy array

        Returns:
            image (np.ndarray): image in numpy array
        """
        height, width = image.shape[:2]
        image = cv2.resize(
            image,
            (self.mosaic_level, self.mosaic_level),
            interpolation=cv2.INTER_LANCZOS4,
        )
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)
        return image
