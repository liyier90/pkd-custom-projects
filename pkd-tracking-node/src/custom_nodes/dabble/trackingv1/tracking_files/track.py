from typing import Tuple

import numpy as np


class Track:
    """Stores information for each tracked detection."""

    def __init__(
        self,
        track_id: int,
        frame_id: int,
        bbox: np.ndarray,
        class_id: int,
        score: float,
        iou_score: float = 0.0,
    ) -> None:
        self.track_id = track_id

        self.max_score = 0.0
        self.lost = 0
        self.age = 0

        self.update(frame_id, bbox, class_id, score, iou_score)

    def output(
        self,
    ) -> Tuple[int, int, float, float, float, float, float, int, int, int]:
        """Outputs tracking result in MOT Challenge format."""
        return (
            self.frame_id,
            self.track_id,
            self.bbox[0],
            self.bbox[1],
            self.bbox[2],
            self.bbox[3],
            self.score,
            -1,
            -1,
            -1,
        )

    def update(
        self,
        frame_id: int,
        bbox: np.ndarray,
        class_id: int,
        score: float,
        iou_score: float,
        lost: int = 0,
    ) -> None:
        """Updates the tracking result with information from the latest frame.

        Args:
            frame_id (int): Camera frame ID.
            bbox (np.ndarray): Bounding box with format (t, l, w, h) where
                (t, l) is the top-left corner, w is the width, and h is the
                height.
            class_id (int): Detection class ID.
            score (float): Detection confidence score.
            iou_score (float): Intersection-over-Union between the current
                detection bounding box and its last detected bounding box.
            lost (int): Number of consecutive frames where this detections is
                not detected.
        """
        self.frame_id = frame_id
        self.bbox = bbox
        self.class_id = class_id
        self.score = score
        self.iou_score = iou_score

        if lost == 0:
            self.lost = 0
        else:
            self.lost += lost

        self.max_score = max(self.max_score, score)
        self.age += 1
