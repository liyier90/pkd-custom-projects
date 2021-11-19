# AISG Boilerplate
from typing import Any, Dict

import numpy as np

from peekingduck.pipeline.nodes.draw.utils.general import get_image_size
from peekingduck.pipeline.nodes.node import AbstractNode


class Node(AbstractNode):
    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)

        self.stationarity_tracker = StationarityTracker(iou_threshold=0.96)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        bus_statuses = [""] * len(inputs["bbox_labels"])

        bus_mask = inputs["bbox_labels"] == "bus"
        statuses = iter(
            self.stationarity_tracker.update_and_get_statuses(
                inputs["img"],
                inputs["bboxes"][bus_mask],
                inputs["obj_track_ids"][bus_mask],
            )
        )
        for i, label in enumerate(inputs["bbox_labels"]):
            if label == "bus":
                bus_statuses[i] = next(statuses)

        return {"bus_statuses": bus_statuses}


class StationarityTracker:
    def __init__(self, n_init=3, max_age=10, iou_threshold=0.95):
        self.n_init = n_init
        self.max_age = max_age
        self.iou_threshold = iou_threshold

        self.tracks = {}
        self.frame_count = 0

    def update_and_get_statuses(self, frame, bboxes, track_ids):
        frame_size = get_image_size(frame)
        bboxes_xywh = Utils.xyxyn2xywh(bboxes, *frame_size)

        ret_statuses = [""] * len(track_ids)
        updated_tracks = []
        for i, (track_id, bbox) in enumerate(zip(track_ids, bboxes_xywh)):
            if track_id in self.tracks:
                track = self.tracks[track_id]
            else:
                track = Track(bbox)
                self.tracks[track_id] = track
            track.update(bbox, self.iou_threshold)
            updated_tracks.append(track_id)
            if track.hits >= self.n_init:
                ret_statuses[i] = "Stationary"

        # Idle tracks not used this frame and delete those older than max_age
        deletions = []
        for track_id in self.tracks:
            if track_id not in updated_tracks:
                self.tracks[track_id].idle()
                if self.tracks[track_id].time_since_update >= self.max_age:
                    deletions.append(track_id)
        for track_id in deletions:
            del self.tracks[track_id]

        return ret_statuses


class Track:
    def __init__(self, bbox):
        self.bbox = bbox

        self.time_since_update = 0
        self.hits = 0
        self.hit_streak = 0

    def update(self, bbox, iou_threshold):
        if Utils.iou(self.bbox, bbox) >= iou_threshold:
            self.time_since_update = 0
            self.hits += 1
            self.hit_streak += 1
        else:
            # One frame buffer
            if self.hit_streak == 0:
                self.hits = 0
            self.hit_streak = 0

        self.bbox = bbox

    def idle(self):
        self.time_since_update += 1
        self.hit_streak = 0


class Utils:
    @staticmethod
    def xyxyn2xywh(x, w, h):
        y = np.copy(x)
        y[:, 0] = x[:, 0] * w
        y[:, 1] = x[:, 1] * h
        y[:, 2] = (x[:, 2] - x[:, 0]) * w
        y[:, 3] = (x[:, 3] - x[:, 1]) * h

        return y

    @staticmethod
    def iou(bbox_1, bbox_2):
        xx1 = np.maximum(bbox_1[0], bbox_2[0])
        yy1 = np.maximum(bbox_1[1], bbox_2[1])
        xx2 = np.minimum(bbox_1[0] + bbox_1[2], bbox_2[0] + bbox_2[2])
        yy2 = np.minimum(bbox_1[1] + bbox_1[3], bbox_2[1] + bbox_2[3])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        wh = w * h
        o = wh / (bbox_1[2] * bbox_1[3] + bbox_2[2] * bbox_2[3] - wh)
        return o
