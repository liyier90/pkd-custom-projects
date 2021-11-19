import cv2
import numpy as np
from peekingduck.pipeline.nodes.draw.utils.general import get_image_size

from track import Track
from utils import Utils


class Sort:
    _NUM_COORDS = 4
    _OPENCV_TRACKER = {"csrt": cv2.TrackerCSRT_create}

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

        updated_ids = []

        curr_preds = np.zeros((len(self.tracks), self._NUM_COORDS))
        failed_to_predict = []
        for p, _ in enumerate(curr_preds):
            if self.skip_predict:
                curr_preds[p] = np.array(self.tracks[p].bbox)
            else:
                success, pos = self.tracks[p].predict(frame)
                updated_ids.append(self.tracks[p].track_id)
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
                updated_ids.append(track.track_id)
                if not success:
                    failed_to_update.append(t)

        for p in reversed(failed_to_update):
            self.tracks.pop(p)

        unmatched = []
        for i in unmatched_bboxes:
            new_track = Track(self._next_id, self.tracker)

            success = new_track.update(frame, bboxes_xywh[i])
            updated_ids.append(new_track.track_id)
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
        print(updated_ids)

        return ret_track_ids
