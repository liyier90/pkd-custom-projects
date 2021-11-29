import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch

from . import matching
from .darknet import Darknet
from .kalman_filter import KalmanFilter
from .track import STrack, TrackState
from .utils import non_max_suppression, scale_coords


class Tracker:
    def __init__(
        self, config: Dict[str, Any], model_dir: Path, frame_rate: float = 30
    ) -> None:
        self.logger = logging.getLogger(__name__)

        self.config = config
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._create_darknet_model()

        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []
        self.removed_stracks: List[STrack] = []

        self.frame_id = 0
        self.det_thresh = config["score_threshold"]
        self.buffer_size = int(frame_rate / 30.0 * config["track_buffer"])
        self.max_time_lost = self.buffer_size

        self.kalman_filter = KalmanFilter()

    def track_objects_from_image(
        self, image: np.ndarray
    ) -> Tuple[List[np.ndarray], List[str], List[float]]:
        padded_image, resized_image = self._preprocess(image)
        padded_image = torch.from_numpy(padded_image).to(self.device).unsqueeze(0)

        online_targets = self.update(padded_image, resized_image)
        online_tlwhs = []
        online_ids = []
        scores = []
        for target in online_targets:
            tlwh = target.tlwh
            vertical = tlwh[2] / tlwh[3] > 1.6
            if not vertical and tlwh[2] * tlwh[3] > self.config["min_box_area"]:
                online_tlwhs.append(tlwh)
                online_ids.append(target.track_id)
                scores.append(target.score)
        # Postprocess here
        bboxes = self._postprocess(online_tlwhs)

        return bboxes, online_ids, scores

    @torch.no_grad()
    def update(self, padded_image, resized_image):
        """Processes the image frame and finds bounding box(detections).

        Associates the detection with corresponding tracklets and also handles
        lost, removed, refound and active tracklets

        Parameters
        ----------
        padded_image : torch.float32
            Tensor of shape depending upon the size of image. By default, shape
            of this tensor is [1, 3, 608, 1088]
        resized_image : ndarray
            ndarray of shape depending on the input image sequence. By default,
            shape is [608, 1080, 3]

        Returns
        -------
        output_stracks : list of Strack(instances)
            The list contains information regarding the online_tracklets for
            the recieved image tensor.
        """
        self.frame_id += 1
        # for storing active tracks, for the current frame
        activated_stracks = []
        # Lost Tracks whose detections are obtained in the current frame
        refind_stracks = []
        # The tracks which are not obtained in the current frame but are not
        # removed.(Lost for some time lesser than the threshold for removing)
        lost_stracks = []
        removed_stracks = []

        # Step 1: Network forward, get detections & embeddings
        pred = self.model(padded_image)
        # pred is tensor of all the proposals (default number of proposals:
        # 54264). Proposals have information associated with the bounding box
        # and embeddings
        pred = pred[pred[..., 4] > self.config["score_threshold"]]
        # pred now has lesser number of proposals. Proposals rejected on basis
        # of object confidence score
        if len(pred) > 0:
            dets = non_max_suppression(
                pred.unsqueeze(0),
                self.config["score_threshold"],
                self.config["nms_threshold"],
            )[0].cpu()
            # Final proposals are obtained in dets. Information of bounding box
            # and embeddings also included
            # Next step changes the detection scales
            scale_coords(self.input_size, dets[:, :4], resized_image.shape).round()
            # Detections is list of (x1, y1, x2, y2, object_conf, class_score,
            # class_pred) class_pred is the embeddings.

            detections = [
                STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f.numpy(), 30)
                for (tlbrs, f) in zip(dets[:, :5], dets[:, 6:])
            ]
        else:
            detections = []

        # Add newly detected tracklets to tracked_stracks
        unconfirmed = []
        tracked_stracks: List[STrack] = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                # previous tracks which are not active in the current frame are
                # added in unconfirmed list
                unconfirmed.append(track)
                # print("Should not be here, in unconfirmed")
            else:
                # Active tracks are added to the local list 'tracked_stracks'
                tracked_stracks.append(track)

        # Step 2: First association, with embedding
        # Combining currently tracked_stracks and lost_stracks
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool, self.kalman_filter)

        dists = matching.embedding_distance(strack_pool, detections)
        # dists = matching.gate_cost_matrix(
        #     self.kalman_filter, dists, strack_pool, detections
        # )
        dists = matching.fuse_motion(self.kalman_filter, dists, strack_pool, detections)
        # The dists is the list of distances of the detection with the tracks
        # in strack_pool
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.7)
        # The matches is the array for corresponding matches of the detection
        # with the corresponding strack_pool

        for itracked, idet in matches:
            # itracked is the id of the track and idet is the detection
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                # If the track is active, add the detection to the track
                track.update(detections[idet], self.frame_id)
                activated_stracks.append(track)
            else:
                # We have obtained a detection from a track which is not
                # active, hence put the track in refind_stracks list
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # None of the steps below happen if there are no undetected tracks.
        # Step 3: Second association, with IOU
        detections = [detections[i] for i in u_detection]
        # detections is now a list of the unmatched detections
        # This is container for stracks which were tracked till the
        r_tracked_stracks = []
        # previous frame but no detection was found for it in the current frame
        for i in u_track:
            if strack_pool[i].state == TrackState.Tracked:
                r_tracked_stracks.append(strack_pool[i])
        dists = matching.iou_distance(r_tracked_stracks, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)
        # matches is the list of detections which matched with corresponding
        # tracks by IOU distance method
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        # Same process done for some unmatched detections, but now considering
        # IOU_distance as measure

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)
        # If no detections are obtained for tracks (u_track), the tracks are
        # added to lost_tracks list and are marked lost

        # Deal with unconfirmed tracks, usually tracks with only one beginning
        # frame
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(
            dists, thresh=0.7
        )
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])

        # The tracks which are yet not matched
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        # after all these confirmation steps, if a new detection is found, it
        # is initialized for a new track
        # Step 4: Init new stracks
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)

        # Step 5: Update state
        # If the tracks are lost for more frames than the threshold number, the
        # tracks are removed.
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)
        # print('Remained match {} s'.format(t4-t3))

        # Update the self.tracked_stracks and self.lost_stracks using the
        # updates in this step.
        self.tracked_stracks = [
            t for t in self.tracked_stracks if t.state == TrackState.Tracked
        ]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        # self.lost_stracks: List[STrack] = [
        #     t for t in self.lost_stracks if t.state == TrackState.Lost
        # ]
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks
        )

        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        # logger.debug(f"===========Frame {self.frame_id}==========")
        # logger.debug(f"Activated: {[track.track_id for track in activated_stracks]}")
        # logger.debug(f"Refind: {[track.track_id for track in refind_stracks]}")
        # logger.debug(f"Lost: {[track.track_id for track in lost_stracks]}")
        # logger.debug(f"Removed: {[track.track_id for track in removed_stracks]}")
        # print(f"Final {t5-t4} s")
        return output_stracks

    def _create_darknet_model(self):
        model_type = self.config["model_type"]
        model_path = self.model_dir / self.config["weights"]["model_file"][model_type]
        model_settings = self._parse_model_config(
            self.model_dir / self.config["weights"]["config_file"][model_type]
        )
        self.input_size = [
            int(model_settings[0]["width"]),
            int(model_settings[0]["height"]),
        ]
        return self._load_darknet_weights(model_path, model_settings)

    def _load_darknet_weights(self, model_path, model_settings):
        ckpt = torch.load(str(model_path), map_location="cpu")
        model = Darknet(model_settings, nID=14455)
        model.load_state_dict(ckpt["model"])
        model.to(self.device).eval()
        return model

    def _postprocess(self, tlwhs: List[np.ndarray]) -> List[np.ndarray]:
        return tlwhs

    def _preprocess(self, image: np.ndarray):
        # Resizing input frame
        video_h, video_w = image.shape[:2]
        ratio_w, ratio_h = (
            float(self.input_size[0]) / video_w,
            float(self.input_size[1]) / video_h,
        )
        ratio = min(ratio_w, ratio_h)
        width, height = int(video_w * ratio), int(video_h * ratio)
        resized_image = cv2.resize(image, (width, height))
        # Padded resize
        padded_image, _, _, _ = self._letterbox(
            resized_image, height=self.input_size[1], width=self.input_size[0]
        )
        # Normalize RGB
        padded_image = padded_image[..., ::-1].transpose(2, 0, 1)
        padded_image = np.ascontiguousarray(padded_image, dtype=np.float32)
        padded_image /= 255.0

        return padded_image, resized_image

    @staticmethod
    def _letterbox(img, height=608, width=1088, color=(127.5, 127.5, 127.5)):
        """Resizes a rectangular image to a padded rectangular."""
        shape = img.shape[:2]  # shape = [height, width]
        ratio = min(float(height) / shape[0], float(width) / shape[1])
        # new_shape = [width, height]
        new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))
        dw = (width - new_shape[0]) / 2  # width padding
        dh = (height - new_shape[1]) / 2  # height padding
        top, bottom = round(dh - 0.1), round(dh + 0.1)
        left, right = round(dw - 0.1), round(dw + 0.1)
        img = cv2.resize(
            img, new_shape, interpolation=cv2.INTER_AREA
        )  # resized, no border
        # padded rectangular
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )
        return img, ratio, dw, dh

    @staticmethod
    def _parse_model_config(config_path: Path) -> List[Dict[str, Any]]:
        """Parse model configuration with context manager"""
        with open(config_path) as infile:
            lines = [
                line
                for line in map(str.strip, infile.readlines())
                if line and not line.startswith("#")
            ]
        module_defs: List[Dict[str, Any]] = []
        for line in lines:
            if line.startswith("["):
                module_defs.append({})
                module_defs[-1]["type"] = line[1:-1].rstrip()
                if module_defs[-1]["type"] == "convolutional":
                    module_defs[-1]["batch_normalize"] = 0
            else:
                key, value = tuple(map(str.strip, line.split("=")))
                if value.startswith("$"):
                    value = module_defs[0].get(value.strip("$"), None)
                module_defs[-1][key] = value
        return module_defs


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())
