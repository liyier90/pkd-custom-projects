"""Tracking-by-detection using OpenCV's MOSSE Tracker."""

from typing import Any, Dict, List, NamedTuple

import cv2
import numpy as np

from custom_nodes.dabble.trackingv1.tracking_files.utils import iou, xyxyn2tlwh


class OpenCVTracker:
    """OpenCV's MOSSE tracker.

    Attributes:
        is_initialised (bool): Flag to determine this is the first run of the
            tracker.
        iou_threshold (float): Minimum IoU value to be used with the matching
            logic.
        next_track_id (int): ID for the next (unmatched) object detection to be
            tracked.
        tracking_dict (Dict[int, Track]): Maps track IDs to their respective
            tracker and bbox coordinates.
    """

    def __init__(self) -> None:
        self.is_initialised = False
        self.iou_threshold = 0.1
        self.next_track_id = 0
        self.tracking_dict: Dict[int, Track] = {}

    def track_detections(self, inputs: Dict[str, Any]) -> List[str]:
        """Initialises and updates the tracker on each frame.

        Args:
            inputs (Dict[str, Any]): Dictionary with keys "img" and "boxes".

        Returns:
            (List[str]): Tracking IDs of the detection bounding boxes.
        """
        frame = inputs["img"]
        frame_size = frame.shape[:2]
        tlwhs = xyxyn2tlwh(inputs["bboxes"], *frame_size)

        if self.is_initialised:
            obj_track_ids = self._match_and_track(frame, tlwhs)
        else:
            for tlwh in tlwhs:
                self._initialise_tracker(frame, tlwh)
            obj_track_ids = list(map(str, self.tracking_dict.keys()))
            self.is_initialised = True
        self._update_tracker_bboxes(frame)

        return obj_track_ids

    def _initialise_tracker(self, frame: np.ndarray, bbox: np.ndarray) -> None:
        """Starts a tracker for each bbox.

        Args:
            frame (np.ndarray): Image frame parsed from video.
            bbox (np.ndarray): Single detected bounding box.
        """
        tracker = cv2.TrackerMOSSE_create()
        tracker.init(frame, tuple(bbox))
        self.next_track_id += 1
        self.tracking_dict[self.next_track_id] = Track(tracker, bbox)

    def _match_and_track(self, frame: np.ndarray, bboxes: np.ndarray) -> List[str]:
        """Matches detections to tracked bboxes, creates a new track if no
        match is found.

        Args:
            frame (np.ndarray): Input video frame.
            bboxes (np.ndarray): Detection bounding boxes with (t, l, w, h)
                format where (t, l) is the coordinate of the top-left corner,
                w is the width, and h is the height of the bounding box
                respectively.

        Returns:
            (List[str]): A list of track IDs for the detections in the current
                frame.
        """
        obj_track_ids = [""] * len(bboxes)

        prev_tracked_bboxes = [track.bbox for _, track in self.tracking_dict.items()]
        matching_dict = {}

        for bbox in bboxes:
            ious = iou(bbox, np.array(prev_tracked_bboxes))
            matching_dict[tuple(bbox)] = (
                ious.argmax() if max(ious) >= self.iou_threshold else None
            )

        track_ids = []
        for bbox, track_id in matching_dict.items():
            if track_id is not None:
                track_ids.append(str(list(self.tracking_dict)[track_id]))
            else:
                self._initialise_tracker(frame, np.array(bbox))

        for i, track_id in enumerate(track_ids):
            if track_id not in obj_track_ids:
                obj_track_ids[i] = track_id

        return obj_track_ids

    def _update_tracker_bboxes(self, frame: np.ndarray) -> None:
        """Updates location of previously tracked detections in subsequent
        frames. Removes the track if its tracker fails to update.
        """
        failures = []
        for track_id, track in self.tracking_dict.items():
            success, bbox = track.tracker.update(frame)
            if success:
                self.tracking_dict[track_id] = Track(track.tracker, np.array(bbox))
            else:
                failures.append(track_id)
        for track_id in failures:
            del self.tracking_dict[track_id]


class Track(NamedTuple):
    """Stores the individual OpenCV MOSSE tracker and its bounding box
    coordinates.

    Attributes:
        tracker (cv2.TrackerMOSSE): OpenCV MOSSE tracker.
        bbox (np.ndarray): Bounding box coordinates in (t, l, w, h) format where
            (t, l) is the top-left coordinate, w is the width, and h is the
            height of the bounding box.
    """

    tracker: cv2.TrackerMOSSE
    bbox: np.ndarray
