from typing import Any, Dict, List

import cv2
import numpy as np
from peekingduck.pipeline.nodes.draw.utils.general import get_image_size


class OpenCvTracker:
    def __init__(self):
        self.is_initialized = False
        self.iou_threshold = 0.1
        self.next_obj_id = 0
        self.track_dict: Dict[int, Dict[str, Any]] = {}

    def update_and_get_tracks(self, frame, bboxes):
        frame_size = get_image_size(frame)
        bboxes_xywh = Utils.xyxyn2xywh(bboxes, *frame_size)

        if not self.is_initialized:
            for bbox in bboxes_xywh:
                self.initialize_tracker(frame, bbox)
            obj_track_ids = list(map(str, self.track_dict.keys()))
        else:
            obj_track_ids = self._update_and_track(frame, bboxes_xywh)

        return obj_track_ids

    def initialize_tracker(self, frame: np.ndarray, bbox: List[float]) -> None:
        tracker = cv2.TrackerCSRT_create()
        tracker.init(frame, tuple(bbox))
        self.track_dict[self.next_obj_id] = {"tracker": tracker, "bbox": bbox}
        self.next_obj_id += 1

    def _update_and_track(self, frame, bboxes):
        obj_track_ids = [""] * len(bboxes)
        prev_tracked_bboxes = [
            np.array(value["bbox"]) for _, value in self.track_dict.items()
        ]
        matching_dict = {}

        for bbox in bboxes:
            ious = Utils.iou(np.array(bbox), np.array(prev_tracked_bboxes))
            matching_dict[tuple(bbox)] = (
                ious.argmax() if max(ious) >= self.iou_thres else None
            )

        track_ids = []
        for bbox, id in matching_dict.items():
            if id is not None:
                track_ids.append(str(list(self.track_dict)[id]))
            else:
                self.initialize_tracker(frame, bbox)

        for i, id in enumerate(track_ids):
            if id not in obj_track_ids:
                obj_track_ids[i] = id

        return obj_track_ids


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
    def xyxy2xywh(x):
        y = np.copy(x)
        y[:, 0] = x[:, 0]
        y[:, 1] = x[:, 1]
        y[:, 2] = x[:, 2] - x[:, 0]
        y[:, 3] = x[:, 3] - x[:, 1]

        return y

    @staticmethod
    def xywh2xyxyn(x, w, h):
        y = np.copy(x)
        if isinstance(x, tuple):
            y[0] = x[0] / w
            y[1] = x[1] / h
            y[2] = (x[0] + x[2]) / w
            y[3] = (x[1] + x[3]) / h
        else:
            y[:, 0] = x[:, 0] / w
            y[:, 1] = x[:, 1] / h
            y[:, 2] = (x[:, 0] + x[:, 2]) / w
            y[:, 3] = (x[:, 1] + x[:, 3]) / h

        return y

    @staticmethod
    def xywh2xyxy(x):
        y = np.copy(x)
        if isinstance(x, tuple):
            y[0] = x[0]
            y[1] = x[1]
            y[2] = x[0] + x[2]
            y[3] = x[1] + x[3]
        else:
            y[:, 0] = x[:, 0]
            y[:, 1] = x[:, 1]
            y[:, 2] = x[:, 0] + x[:, 2]
            y[:, 3] = x[:, 1] + x[:, 3]

        return y

    @staticmethod
    def iou(bbox, candidates):
        bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
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
