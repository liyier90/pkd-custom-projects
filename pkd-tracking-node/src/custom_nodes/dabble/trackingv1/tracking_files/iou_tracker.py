"""Tracking-by-detection using IoU Tracker."""

from collections import OrderedDict
from typing import Any, Dict, List, Tuple

import numpy as np

from custom_nodes.dabble.trackingv1.tracking_files.track import Track
from custom_nodes.dabble.trackingv1.tracking_files.utils import iou_tlwh, xyxyn2tlwh


class IOUTracker:
    """Simple tracking class based on Intersection over Union (IoU) of bounding
    boxes.

    This method is based on the assumption that the detector produces a
    detection per frame for every object to be tracked. Furthermore, it is
    assumed that detections of an object in consecutive frames have an
    unmistakably high overlap IoU which is commonly the case when using
    sufficiently high frame rates.

    References:
        High-Speed Tracking-by-Detection Without Using Image Information:
        http://elvera.nue.tu-berlin.de/files/1517Bochinski2017.pdf

        Inference code adapted from
        https://github.com/adipandas/multi-object-tracker
    """

    def __init__(self) -> None:
        self.iou_threshold = 0.1
        self.max_lost = 10
        self.next_track_id = 0

        self.tracks: "OrderedDict[int, Track]" = OrderedDict()

    def track_detections(self, inputs: Dict[str, Any]) -> List[str]:
        """Initialises and updates tracker on each frame.

        Args:
            inputs (Dict[str, Any]): Dictionary with keys "img", "bboxes", and
                "bbox_scores".

        Returns:
            (List[str]): List of track IDs.
        """
        frame = inputs["img"]
        frame_size = frame.shape[:2]
        tlwhs = xyxyn2tlwh(inputs["bboxes"], *frame_size)

        detections = list(zip(tlwhs, inputs["bbox_scores"]))
        tracks = self.update(detections)
        track_ids = self._order_track_ids_by_bbox(tlwhs, tracks)

        return track_ids

    def update(self, detections: List[Tuple[np.ndarray, np.ndarray]]) -> List[Track]:
        """Updates the tracker. Creates new tracks for untracked objects,
        updates tracked objects with the new class ID and bounding box
        coordinates. Removes tracks which have not been detected for longer
        than the `max_lost` threshold.

        Args:
            detections (List[Tuple[np.ndarray, np.ndarray]]): List of tuples
                containing the bounding box coordinates and detection
                confidence score for each of the detection. The bounding box
                has the format (t, l, w, h) where (t, l) is the top-left
                corner, w is the width, and h is the height.

        Returns:
            (List[Track]): All tracked detections in the current frame.
        """
        track_ids = list(self.tracks.keys())

        updated_tracks = []
        for track_id in track_ids:
            if detections:
                tracked_bbox = self.tracks[track_id].bbox
                # Find best match by IoU
                idx, best_match = max(
                    enumerate(detections), key=lambda x: iou_tlwh(tracked_bbox, x[1][0])
                )
                tlwh, score = best_match
                iou_value = iou_tlwh(tracked_bbox, tlwh)
                if iou_value >= self.iou_threshold:
                    self._update_track(track_id, tlwh, score, iou_value)
                    updated_tracks.append(track_id)
                    del detections[idx]
            if not updated_tracks or track_id != updated_tracks[-1]:
                self.tracks[track_id].lost += 1
                if self.tracks[track_id].lost > self.max_lost:
                    self._remove_track(track_id)
        for tlwh, score in detections:
            self._add_track(tlwh, score)
        outputs = self._get_tracks()
        return outputs

    def _add_track(self, bbox: np.ndarray, score: float) -> None:
        """Adds a newly detected object to the list of tracked detections.

        Args:
            bbox (np.ndarray): Bounding box with format (t, l, w, h) where
                (t, l) is the top-left corner, w is the width, and h is the
                height.
            score (float): Detection confidence score.
        """
        self.tracks[self.next_track_id] = Track(self.next_track_id, bbox, score)
        self.next_track_id += 1

    def _get_tracks(self) -> List[Track]:
        """All tracked detections in the current frame."""
        return [track for _, track in self.tracks.items() if track.lost == 0]

    def _remove_track(self, track_id: int) -> None:
        """Removes the specified track. Typically called when the track has not
        been detected in the frame for longer than `max_lost` consecutive
        frames.

        Args:
            track_id (int): The track to delete.
        """
        del self.tracks[track_id]

    def _update_track(
        self, track_id: int, bbox: np.ndarray, score: float, iou_score: float
    ) -> None:
        """Updates the specified tracked detection.

        Args:
            track_id (int): ID of the tracked detection.
            bbox (np.ndarray): Bounding box coordinates with (t, l, w, h)
                format where (t, l) is the top-left corner, w is the width, and
                h is the height.
            score (float): Detection confidence score.
            iou_score (float): Intersection-over-Union between the current
                detection bounding box and its last detected bounding box.
        """
        self.tracks[track_id].update(bbox, score, iou_score)

    @staticmethod
    def _order_track_ids_by_bbox(bboxes: np.ndarray, tracks: List[Track]) -> List[str]:
        """Extracts the track IDs and orders them by their respective bounding
        boxes.

        Args:
            bboxes (np.ndarray): Detection bounding boxes with (t, l, w, h)
                format where (t, l) is the top-left corner, w is the width, and
                h is the height.
            tracks (List[Track]): List of tracked detections.

        Returns:
            (List[str]): Track IDs of the detections in the current frame.
        """
        bbox_to_track_id = {tuple(track.bbox): track.track_id for track in tracks}
        track_ids = [str(bbox_to_track_id[bbox]) for bbox in map(tuple, bboxes)]
        return track_ids
