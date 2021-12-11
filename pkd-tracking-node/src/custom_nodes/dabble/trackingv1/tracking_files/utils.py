"""Utility functions used by tracking-by-detection trackers."""

import numpy as np


def iou_candidates(bbox: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Computes Intersection-over-Union of `bbox` and each of the candidate
    bbox in `candidates`.

    Args:
        bbox (np.ndarray): A bounding box in format `(top left x, top left y,
            width, height)`.
        candidates (np.ndarray): A matrix of candidate bounding boxes
            (one per row) in the same format as `bbox`.
    Returns:
        np.ndarray: The IoU in [0, 1] between the `bbox` and each candidate. A
            higher score means a larger fraction of the `bbox` is occluded by
            the candidate.
    """
    bbox_tl = bbox[:2]
    bbox_br = bbox[:2] + bbox[2:]
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, :2] + candidates[:, 2:]

    top_left = np.c_[
        np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
        np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis],
    ]
    bottom_right = np.c_[
        np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
        np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis],
    ]
    width_height = np.maximum(0.0, bottom_right - top_left)

    area_intersection = width_height.prod(axis=1)
    area_bbox = bbox[2:].prod()
    area_candidates = candidates[:, 2:].prod(axis=1)

    return area_intersection / (area_bbox + area_candidates - area_intersection)


def iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """Calculates the Intersection-over-Union (IoU) of two bounding boxes. Each
    bounding box has the format (x1, y1, x2, y2) where (x1, y1) is the top-left
    corner and (x2, y2) is the bottom right corner.

    Args:
        bbox1 (np.ndarray): The first bounding box.
        bbox2 (np.ndarray): The other bounding box.

    Returns:
        float: IoU of bbox1, bbox2.

    References:
        https://github.com/bochinski/iou-tracker/blob/master/util.py
    """
    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]

    (x0_1, y0_1, x1_1, y1_1), (x0_2, y0_2, x1_2, y1_2) = bbox1, bbox2

    # get the overlap rectangle
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)

    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0.0

    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection

    iou_value = size_intersection / size_union

    return iou_value


def iou_tlwh(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """Calculates the Intersection-over-Union (IoU) of two bounding boxes. Each
    bounding box as the format (t, l, w, h) where (t, l) is the top-left
    corner, w is the width, and h is the height.

    Args:
        bbox1 (np.ndarray): The first bounding box.
        bbox2 (np.ndarray): The other bounding box.

    Returns:
        (float): IoU of bbox1, bbox2.

    References:
        https://github.com/bochinski/iou-tracker/blob/master/util.py
    """
    # Convert to (x1, y1, x2, y2) format
    bbox1 = bbox1[0], bbox1[1], bbox1[0] + bbox1[2], bbox1[1] + bbox1[3]
    bbox2 = bbox2[0], bbox2[1], bbox2[0] + bbox2[2], bbox2[1] + bbox2[3]
    # bbox1[2:] = bbox1[:2] + bbox1[2:]
    # bbox2[2:] = bbox2[:2] + bbox2[2:]

    iou_value = iou(bbox1, bbox2)

    return iou_value


def xyxyn2tlwh(inputs: np.ndarray, height: int, width: int) -> np.ndarray:
    """Converts bounding boxes format from (x1, y2, x2, y2) to (t, l, w, h).
    (x1, y1) is the normalised coordinates of the top-left corner, (x2, y2) is
    the normalised coordinates of the bottom-right corner. (t, l) is the
    original coordinates of the top-left corner, (w, h) is the original width
    and height of the bounding box.

    Args:
        inputs (np.ndarray): Bounding box coordinates with (x1, y1, x2, y2)
            format.
        height (int): Original height of bounding box.
        width (int): Original width of bounding box.

    Returns:
        (np.ndarray): Converted bounding box coordinates with (t, l, w, h)
            format.
    """
    outputs = np.empty_like(inputs)
    outputs[:, 0] = inputs[:, 0] * width  # Bottom left x
    outputs[:, 1] = inputs[:, 1] * height  # Bottom left y
    outputs[:, 2] = (inputs[:, 2] - inputs[:, 0]) * width  # Top right x
    outputs[:, 3] = (inputs[:, 3] - inputs[:, 1]) * height  # Top right y
    return outputs
