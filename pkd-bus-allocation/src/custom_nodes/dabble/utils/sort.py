import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from peekingduck.pipeline.nodes.draw.utils.general import get_image_size


class Sort:
    _NUM_COORDS = 4
    _OPENCV_TRACKER = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "mosse": cv2.TrackerMOSSE_create,
    }

    def __init__(
        self,
        tracker_type,
        skip_predict=False,
        max_age=30,
        min_hits=1,
        iou_threshold=0.5,
        time_since_last_update=200,
    ):
        self.tracker = self._OPENCV_TRACKER[tracker_type]
        self.skip_predict = skip_predict
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.time_since_last_update = time_since_last_update

        self.tracks = []
        self.frame_count = 0
        self._next_id = 0

    def update_and_get_tracks(self, frame, bboxes):
        frame_size = get_image_size(frame)
        bboxes_xywh = Utils.xyxyn2xywh(bboxes, *frame_size)

        curr_preds = np.zeros((len(self.tracks), self._NUM_COORDS))
        failed_to_predict = []
        for p, _ in enumerate(curr_preds):
            if self.skip_predict:
                curr_preds[p] = np.array(self.tracks[p].bbox)
            else:
                success, pos = self.tracks[p].predict(frame)
                if success:
                    curr_preds[p] = np.array(pos)
                else:
                    failed_to_predict.append(p)

        for p in reversed(failed_to_predict):
            curr_preds = np.delete(curr_preds, p, axis=0)
            self.tracks.pop(p)

        matched, unmatched_bboxes, unmatched_tracks = Utils.match_bboxes_to_tracks(
            bboxes_xywh, curr_preds, self.iou_threshold
        )

        failed_to_update = []
        for t, track in enumerate(self.tracks):
            if t not in unmatched_tracks:
                b = matched[np.where(matched[:, 1] == t)[0], 0][0]
                success = track.update(frame, bboxes_xywh[b])
                if not success:
                    failed_to_update.append(t)

        for p in reversed(failed_to_update):
            self.tracks.pop(p)

        unmatched = []
        for i in unmatched_bboxes:
            new_track = Track(self._next_id, self.tracker)

            success = new_track.update(frame, bboxes_xywh[i])
            if success:
                unmatched.append((i, len(self.tracks)))
                self.tracks.append(new_track)
                self._next_id += 1

        ret_track_ids = [""] * len(bboxes)
        for match in matched.tolist() + unmatched:
            if (
                self.tracks[match[1]].time_since_update < self.time_since_last_update
            ) and (
                self.tracks[match[1]].hits >= self.min_hits
                or self.frame_count <= self.min_hits
            ):
                ret_track_ids[match[0]] = str(self.tracks[match[1]].track_id)

        i = len(self.tracks)
        for track in reversed(self.tracks):
            i -= 1
            # success, _ = track.get_state(frame)
            success = True
            if not success or track.time_since_update > self.max_age:
                self.tracks.pop(i)
                continue

        return np.array(ret_track_ids)


class Track:
    def __init__(self, track_id, tracker_constructor):
        self.track_id = track_id
        self._tracker_constructor = tracker_constructor
        self._tracker = None

        self.bbox = None
        self.time_since_update = 0
        self.hits = 0
        self.hit_streak = 0

    def update(self, frame, bbox):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

        self.bbox = tuple(bbox)
        # Can't update python opencv tracker with image + bbox, have to re-initialize
        # https://stackoverflow.com/questions/49755892/
        self._tracker = self._tracker_constructor()
        return self._tracker.init(frame, self.bbox)

    def get_state(self, frame):
        return self._tracker.update(frame)

    def predict(self, frame):
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        return self.get_state(frame)


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

    @staticmethod
    def match_bboxes_to_tracks(bboxes, preds, iou_threshold):
        if len(preds) == 0:
            return np.array([]), np.arange(len(bboxes)), np.array([])

        iou_matrix = np.zeros((len(bboxes), len(preds)), dtype=np.float32)

        for b, bbox in enumerate(bboxes):
            for t, trk in enumerate(preds):
                iou_matrix[b, t] = Utils.iou(bbox, trk)
        matched_bbox_indices, matched_pred_indices = linear_sum_assignment(-iou_matrix)

        unmatched_bboxes = []
        for b, bbox in enumerate(bboxes):
            if b not in matched_bbox_indices:
                unmatched_bboxes.append(b)

        unmatched_tracks = []
        for t, trk in enumerate(preds):
            if t not in matched_pred_indices:
                unmatched_tracks.append(t)

        matches = []
        for b, p in zip(matched_bbox_indices, matched_pred_indices):
            if iou_matrix[b, p] < iou_threshold:
                unmatched_bboxes.append(b)
                unmatched_tracks.append(p)
            else:
                matches.append((b, p))

        return (
            np.array(matches),
            np.array(unmatched_bboxes),
            np.array(unmatched_tracks),
        )
