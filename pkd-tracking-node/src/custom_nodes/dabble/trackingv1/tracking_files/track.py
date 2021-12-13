"""Data class to store information of a single tracked detection."""

import numpy as np


class Track:
    """Stores information for each tracked detection.

    Args:
        track_id (int): Tracking ID of the detection.
        bbox (np.ndarray): Bounding box coordinates with (t, l, w, h) format
            where (t, l) is the top-left corner, w is the width, and h is the
            height.
        score (float): Detection confidence score.

    Attributes:
        bbox (np.ndarray): Bounding box coordinates with (t, l, w, h) format
            where (t, l) is the top-left corner, w is the width, and h is the
            height.
        iou_score (float): The Intersection-over-Union value between the
            current `bbox` and the immediate previous `bbox`.
        lost (int): The number of consecutive frames where this detection is
            not found in the video frame.
        score (float): Detection confidence score.
        track_id (int): Tracking ID of the detection.
    """

    def __init__(self, track_id: int, bbox: np.ndarray, score: float) -> None:
        self.track_id = track_id
        self.lost = 0
        self.update(bbox, score)

    def update(self, bbox: np.ndarray, score: float, iou_score: float = 0.0) -> None:
        """Updates the tracking result with information from the latest frame.

        Args:
            bbox (np.ndarray): Bounding box with format (t, l, w, h) where
                (t, l) is the top-left corner, w is the width, and h is the
                height.
            score (float): Detection confidence score.
            iou_score (float): Intersection-over-Union between the current
                detection bounding box and its last detected bounding box.
        """
        self.bbox = bbox
        self.score = score
        self.iou_score = iou_score
        self.lost = 0
