from typing import List, Tuple

import cv2
import numpy as np
from cv2 import FONT_HERSHEY_SIMPLEX

from peekingduck.pipeline.nodes.draw.utils.constants import (
    THICK,
    VERY_THICK,
    NORMAL_FONTSCALE,
)
from peekingduck.pipeline.nodes.draw.utils.general import (
    get_image_size,
    project_points_onto_original_image,
)


def draw_annotations(
    frame: np.array,
    bboxes: List[List[float]],
    annotations: List[str],
    colour: Tuple[int, int, int],
    location: Tuple[float, float],
    key: str,
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
        _draw_annotation(frame, bbox, annotations[idx][key], image_size, colour, location)


def _draw_annotation(
    frame: np.array,
    bbox: np.array,
    annotation: str,
    image_size: Tuple[int, int],
    colour: Tuple[int, int, int],
    location: Tuple[float, float],
) -> None:
    """Draw a annotation on single bounding box."""
    top_left, btm_right = project_points_onto_original_image(bbox, image_size)

    # Find offset to centralize text
    (w_text, h_text), _ = cv2.getTextSize(
        annotation, FONT_HERSHEY_SIMPLEX, NORMAL_FONTSCALE, THICK
    )
    bbox_w_offset = int((btm_right[0] - top_left[0]) * location[0])
    bbox_h_offset = int((btm_right[1] - top_left[1]) * location[1])
    w_offset = int(bbox_w_offset - w_text / 2)
    h_offset = int(bbox_h_offset - h_text / 2)
    position = (int(top_left[0] + w_offset), int(top_left[1] + h_offset))
    cv2.putText(
        frame, annotation, position, FONT_HERSHEY_SIMPLEX, NORMAL_FONTSCALE, colour, VERY_THICK
    )
