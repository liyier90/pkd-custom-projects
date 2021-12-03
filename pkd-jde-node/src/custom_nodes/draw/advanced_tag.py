"""
Node template for creating custom nodes.
"""

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from peekingduck.pipeline.nodes.draw.utils.constants import (
    NORMAL_FONTSCALE,
    THICK,
    TOMATO,
    VERY_THICK,
)
from peekingduck.pipeline.nodes.draw.utils.general import (
    get_image_size,
    project_points_onto_original_image,
)
from peekingduck.pipeline.nodes.node import AbstractNode


class Node(AbstractNode):
    """This is a template class of how to write a node for PeekingDuck.

    Args:
        config (:obj:`Dict[str, Any]` | :obj:`None`): Node configuration.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)

        # initialize/load any configs and models here
        # configs can be called by self.<config_name> e.g. self.filepath
        # self.logger.info(f"model loaded with configs: config")
        self.location: List[float]

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
        """This node does ___.

        Args:
            inputs (dict): Dictionary with keys "__", "__".

        Returns:
            outputs (dict): Dictionary with keys "__".
        """
        _draw_advanced_tags(
            inputs["img"], inputs["bboxes"], inputs["obj_tags"], TOMATO, self.location
        )
        return {}


def _draw_advanced_tags(
    frame: np.ndarray,
    bboxes: List[np.ndarray],
    tags: List[str],
    colour: Tuple[int, int, int],
    location: List[float],
) -> None:
    """Draw annotations above bboxes.
    Args:
        frame (np.array): image of current frame
        bboxes (List[List[float]]): bounding box coordinates
        annotations (List[string]): tag associated with bounding box
        color (Tuple[int, int, int]): color of text
    """
    image_size = get_image_size(frame)
    for idx, bbox in enumerate(bboxes):
        _draw_advanced_tag(frame, bbox, tags[idx], image_size, colour, location)


def _draw_advanced_tag(
    frame: np.ndarray,
    bbox: np.ndarray,
    tag: str,
    image_size: Tuple[int, int],
    colour: Tuple[int, int, int],
    location: List[float],
) -> None:
    """Draw a annotation on single bounding box."""
    top_left, btm_right = project_points_onto_original_image(bbox, image_size)

    # Find offset to centralize text
    (w_text, h_text), _ = cv2.getTextSize(
        tag, cv2.FONT_HERSHEY_SIMPLEX, NORMAL_FONTSCALE, THICK
    )
    bbox_w_offset = int((btm_right[0] - top_left[0]) * location[0])
    # bbox_h_offset = int((btm_right[1] - top_left[1]) * location[1])
    w_offset = int(bbox_w_offset - w_text / 2)
    # h_offset = int(bbox_h_offset - h_text / 2)
    h_offset = int(h_text * location[1])
    position = (int(top_left[0] + w_offset), int(top_left[1] + h_offset))
    cv2.putText(
        frame,
        tag,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        NORMAL_FONTSCALE,
        colour,
        VERY_THICK,
    )
