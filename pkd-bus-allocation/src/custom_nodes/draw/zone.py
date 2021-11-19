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
Draws the 2D boundaries of a zone
"""

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from peekingduck.pipeline.nodes.draw.utils.constants import (
    PRIMARY_PALETTE,
    PRIMARY_PALETTE_LENGTH,
    VERY_THICK,
)
from peekingduck.pipeline.nodes.draw.utils.general import (
    get_image_size,
    project_points_onto_original_image,
)
from peekingduck.pipeline.nodes.node import AbstractNode


class Node(AbstractNode):
    """Draws the boundaries of each specified zone onto the image.

    The draw zones node uses the zones from the zone_count dabble node to
    draw a bounding box that represents the zone boundaries onto the image.

    Inputs:

        |img|

        |zones|

    Outputs:
        |none|

    Configs:
        None.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self.zones = np.asarray(self.zones)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Draws the boundaries of each specified zone onto the image.

        Args:
            inputs (dict): Dict with keys "zones", "img".

        Returns:
            outputs (dict): Dict with keys "none".
        """
        image_size = get_image_size(inputs["img"])
        for i, zone_pts in enumerate(self.zones):
            zone_pts = list(
                map(
                    tuple,
                    project_points_onto_original_image(zone_pts, image_size).astype(
                        int
                    ),
                )
            )
            draw_zone(inputs["img"], zone_pts, i)

        return {}


def draw_zone(frame: np.array, points: List[Tuple[int]], zone_index: int) -> None:
    num_points = len(points)
    for i in range(num_points):
        cv2.line(
            frame,
            points[i],
            points[(i + 1) % num_points],
            PRIMARY_PALETTE[(zone_index + 1) % PRIMARY_PALETTE_LENGTH],
            VERY_THICK,
        )
