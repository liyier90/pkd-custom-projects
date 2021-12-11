"""Tracking-by-detection trackers."""

import logging
from typing import Any, Dict, List

from custom_nodes.dabble.trackingv1.tracking_files.iou_tracker import IOUTracker
from custom_nodes.dabble.trackingv1.tracking_files.opencv_tracker import OpenCVTracker


class DetectionTracker:
    """Tracks detection bounding boxes using the chosen algorithm.

    Args:
        tracker_type (str): Type of tracking algorithm to be used, one of
            ["iou", "mosse"].

    Raises:
        ValueError: `tracker_type` is not one of ["iou", "mosse"].
    """

    trackers = {"iou": IOUTracker(), "mosse": OpenCVTracker()}

    def __init__(self, tracker_type: str) -> None:
        self.logger = logging.getLogger(__name__)

        try:
            self.tracker = self.trackers[tracker_type]
        except KeyError as error:
            raise ValueError("tracker_type must be one of ['iou', 'mosse']") from error

    def track_detections(self, inputs: Dict[str, Any]) -> List[str]:
        """Tracks detections using the selected algorithm.

        Args:
            inputs (Dict[str, Any]): Dictionary with keys "__", "__".

        Returns:
            (List[str]): Tracking IDs of the detection bounding boxes.
        """
        track_ids = self.tracker.track_detections(inputs)
        return track_ids
