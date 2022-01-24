import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from custom_nodes.model.fairmotv1.fairmot_files import matching
from custom_nodes.model.fairmotv1.fairmot_files.decode import mot_decode
from custom_nodes.model.fairmotv1.fairmot_files.dla import DLASeg
from custom_nodes.model.fairmotv1.fairmot_files.kalman_filter import KalmanFilter
from custom_nodes.model.fairmotv1.fairmot_files.track import STrack, TrackState
from custom_nodes.model.fairmotv1.fairmot_files.utils import (
    ctdet_post_process,
    letterbox,
    tlwh2xyxyn,
    transpose_and_gather_feat,
)


class Tracker:
    heads = {"hm": 1, "wh": 4, "id": 128, "reg": 2}
    down_ratio = 4
    final_kernel = 1
    head_conv = 256
    last_level = 5
    num_classes = 1

    mean = np.array([0.408, 0.447, 0.470], dtype=np.float32).reshape((1, 1, 3))
    std = np.array([0.289, 0.274, 0.278], dtype=np.float32).reshape((1, 1, 3))

    def __init__(
        self, config: Dict[str, Any], model_dir: Path, frame_rate: float
    ) -> None:
        self.logger = logging.getLogger(__name__)

        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._create_model(model_dir)

        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []
        self.removed_stracks: List[STrack] = []

        self.frame_id = 0
        self.max_time_lost = int(frame_rate / 30.0 * config["track_buffer"])
        self.max_per_image = self.config["K"]

        self.kalman_filter = KalmanFilter()

    def merge_outputs(self, detections):
        results = {}
        for j in range(1, self.num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0
            ).astype(np.float32)

        scores = np.hstack([results[j][:, 4] for j in range(1, self.num_classes + 1)])
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.num_classes + 1):
                keep_inds = results[j][:, 4] >= thresh
                results[j] = results[j][keep_inds]
        return results

    def prepare_for_tracking(self, dets, meta):
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = ctdet_post_process(
            dets.copy(),
            [meta["c"]],
            [meta["s"]],
            meta["out_height"],
            meta["out_width"],
            self.num_classes,
        )
        for j in range(1, self.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
        return dets[0]

    def track_objects_from_image(
        self, image: np.ndarray
    ) -> Tuple[List[np.ndarray], List[str], List[float]]:
        image_size = image.shape[:2]
        padded_image = self._preprocess(image)
        padded_image = torch.from_numpy(padded_image).to(self.device).unsqueeze(0)

        online_targets = self.update(padded_image, image)
        online_tlwhs = []
        online_ids = []
        scores = []
        for target in online_targets:
            tlwh = target.tlwh
            vertical = tlwh[2] / tlwh[3] > 1.6
            if not vertical and tlwh[2] * tlwh[3] > self.config["min_box_area"]:
                online_tlwhs.append(tlwh)
                online_ids.append(str(target.track_id))
                scores.append(target.score.item())
        if not online_tlwhs:
            return online_tlwhs, online_ids, scores

        bboxes = self._postprocess(np.asarray(online_tlwhs), image_size)

        return bboxes, online_ids, scores

    @torch.no_grad()
    def update(  # pylint: disable=too-many-branches,too-many-locals,too-many-statements
        self, padded_image: torch.Tensor, image: np.ndarray
    ) -> List[STrack]:
        """Processes the image frame and finds bounding box (detections).
        Associates the detection with corresponding tracklets and also handles
        lost, removed, re-found and active tracklets
        Args:
            padded_image (torch.Tensor): Preprocessed image with letterbox
                resizing and colour normalisation.
            image (np.ndarray): The original video frame.
        Returns:
            (List[STrack]): The list contains information regarding the
            online tracklets for the received image tensor.
        """
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        width = image.shape[1]
        height = image.shape[0]
        inp_height = padded_image.shape[2]
        inp_width = padded_image.shape[3]
        c = np.array([width / 2.0, height / 2.0], dtype=np.float32)
        s = max(float(inp_width) / float(inp_height) * height, width) * 1.0
        meta = {
            "c": c,
            "s": s,
            "out_height": inp_height // self.down_ratio,
            "out_width": inp_width // self.down_ratio,
        }

        """ Step 1: Network forward, get detections & embeddings"""
        with torch.no_grad():
            output = self.model(padded_image)[-1]
            hm = output["hm"].sigmoid_()
            wh = output["wh"]
            id_feature = output["id"]
            id_feature = F.normalize(id_feature, dim=1)

            # reg = output["reg"] if self.opt.reg_offset else None
            reg = output["reg"]
            # dets, inds = mot_decode(hm, wh, reg=reg, ltrb=self.opt.ltrb, K=self.opt.K)
            dets, inds = mot_decode(hm, wh, reg, self.config["K"])
            id_feature = transpose_and_gather_feat(id_feature, inds)
            id_feature = id_feature.squeeze(0)
            id_feature = id_feature.cpu().numpy()

        dets = self.prepare_for_tracking(dets, meta)
        dets = self.merge_outputs([dets])[1]

        remain_inds = dets[:, 4] > self.config["score_threshold"]
        dets = dets[remain_inds]
        id_feature = id_feature[remain_inds]

        if len(dets) > 0:
            # Detections is list of (x1, y1, x2, y2, object_conf, class_score,
            # class_pred) class_pred is the embeddings.
            detections = [
                STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f, 30)
                for (tlbrs, f) in zip(dets[:, :5], id_feature)
            ]
        else:
            detections = []

        # Add newly detected tracklets to tracked_stracks
        unconfirmed = []
        tracked_stracks: List[STrack] = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                # Active tracks are added to the local list 'tracked_stracks'
                unconfirmed.append(track)
            else:
                # previous tracks which are not active in the current frame
                # are added in unconfirmed list
                tracked_stracks.append(track)

        # Step 2: First association, with embedding
        # Combining currently tracked_stracks and lost_stracks
        strack_pool = _combine_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)

        # The dists is a matrix of distances of the detection with the tracks
        # in strack_pool
        dists = matching.embedding_distance(strack_pool, detections)
        dists = matching.fuse_motion(self.kalman_filter, dists, strack_pool, detections)
        # matches is the array for corresponding matches of the detection
        # with the corresponding strack_pool
        (
            matches,
            unmatched_track_indices,
            unmatched_det_indices,
        ) = matching.linear_assignment(dists, thresh=0.4)

        for tracked_idx, det_idx in matches:
            # tracked_idx is the id of the track and det_idx is the detection
            track = strack_pool[tracked_idx]
            det = detections[det_idx]
            if track.state == TrackState.Tracked:
                # If the track is active, add the detection to the track
                track.update(detections[det_idx], self.frame_id)
                activated_stracks.append(track)
            else:
                # We have obtained a detection from a track which is not
                # active, hence put the track in refind_stracks list
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # None of the steps below happen if there are no undetected tracks.
        # Step 3: Second association, with IOU
        # detections is now a list of the unmatched detections
        detections = [detections[i] for i in unmatched_det_indices]
        # This is container for stracks which were tracked till the
        # previous frame but no detection was found for it in the current frame
        r_tracked_stracks = [
            strack_pool[i]
            for i in unmatched_track_indices
            if strack_pool[i].state == TrackState.Tracked
        ]
        dists = matching.iou_distance(r_tracked_stracks, detections)
        # matches is the list of detections which matched with corresponding
        # tracks by IOU distance method
        (
            matches,
            unmatched_track_indices,
            unmatched_det_indices,
        ) = matching.linear_assignment(dists, thresh=0.5)
        # Same process done for some unmatched detections, but now considering
        # IOU_distance as measure
        for tracked_idx, det_idx in matches:
            track = r_tracked_stracks[tracked_idx]
            det = detections[det_idx]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:  # pragma: no cover
                # This shouldn't be reached, r_tracked_stracks only takes in
                # tracks with TrackState.TRACKED from above
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        # If no detections are obtained for tracks (unmatched_track_indices),
        # the tracks are added to lost_tracks and are marked lost
        for i in unmatched_track_indices:
            track = r_tracked_stracks[i]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # Deal with unconfirmed tracks, usually tracks with only one beginning
        # frame
        detections = [detections[i] for i in unmatched_det_indices]
        dists = matching.iou_distance(unconfirmed, detections)
        (
            matches,
            unconfirmed_track_indices,
            unmatched_det_indices,
        ) = matching.linear_assignment(dists, thresh=0.7)
        for tracked_idx, det_idx in matches:
            unconfirmed[tracked_idx].update(detections[det_idx], self.frame_id)
            activated_stracks.append(unconfirmed[tracked_idx])
        # The tracks which are yet not matched
        for i in unconfirmed_track_indices:
            track = unconfirmed[i]
            track.mark_removed()
            removed_stracks.append(track)

        # after all these confirmation steps, if a new detection is found, it
        # is initialised for a new track
        # Step 4: Init new stracks
        for inew in unmatched_det_indices:
            track = detections[inew]
            if track.score < self.config["score_threshold"]:
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

        # Update the self.tracked_stracks and self.lost_stracks using the
        # updates in this step.
        self.tracked_stracks = [
            t for t in self.tracked_stracks if t.state == TrackState.Tracked
        ]
        self.tracked_stracks = _combine_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = _combine_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = _substract_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = _substract_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = _remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks
        )

        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        return output_stracks

    def _create_model(self, model_dir: Path):
        model_type = self.config["model_type"]
        model_path = model_dir / self.config["weights"]["model_file"][model_type]
        self.logger.info(
            "FairMOT model loaded with the following config:\n\t"
            f"Model type: {model_type}\n\t"
            f"Score threshold: {self.config['score_threshold']}\n\t"
            f"Max number of output objects: {self.config['K']}\n\t"
            f"Min bounding box area: {self.config['min_box_area']}\n\t"
            f"Track buffer: {self.config['track_buffer']}\n\t"
            f"Input size: {self.config['input_size']}"
        )
        return self._load_model_weights(model_path)

    def _load_model_weights(self, model_path: Path):
        if not model_path.is_file():
            raise ValueError(
                f"Model file does not exist. Please check that {model_path} exists."
            )

        ckpt = torch.load(str(model_path), map_location="cpu")
        model = DLASeg(
            "dla34",
            self.heads,
            pretrained=True,
            down_ratio=self.down_ratio,
            final_kernel=self.final_kernel,
            last_level=self.last_level,
            head_conv=self.head_conv,
        )
        model.load_state_dict(ckpt["state_dict"], strict=False)
        model.to(self.device).eval()
        return model

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocesses the input image by padded resizing with letterbox and
        normalising RGB values.
        Args:
            image (np.ndarray): Input video frame.
        Returns:
            (np.ndarray): Preprocessed image.
        """
        # Padded resize
        padded_image = letterbox(
            image,
            height=self.config["input_size"][1],
            width=self.config["input_size"][0],
        )
        # Normalise RGB
        padded_image = padded_image[..., ::-1].transpose(2, 0, 1)
        padded_image = np.ascontiguousarray(padded_image, dtype=np.float32)
        padded_image /= 255.0

        return padded_image

    @staticmethod
    def _postprocess(tlwhs: np.ndarray, image_shape: Tuple[int, ...]) -> np.ndarray:
        """Post-processes detection bounding boxes by converting them from
        [t, l, w, h] to normalised [x1, y1, x2, y2] format which is required by
        other PeekingDuck draw nodes. (t, l) is the top-left corner, w is
        width, and h is height. (x1, y1) is the top-left corner and (x2, y2) is
        the bottom-right corner.
        Args:
            tlwhs (np.ndarray): Bounding boxes in [t, l, w, h] format.
            image_shape (Tuple[int, ...]): Dimensions of the original video
                frame.
        Returns:
            (np.ndarray): Bounding boxes in normalised [x1, y1, x2, y2] format.
        """
        return tlwh2xyxyn(tlwhs, *image_shape)


def _combine_stracks(stracks_1: List[STrack], stracks_2: List[STrack]) -> List[STrack]:
    """Combines two list of STrack together.
    Args:
        stracks_1 (List[STrack]): List of STrack.
        stracks_2 (List[STrack]): List of STrack.
    Returns:
        (List[STrack]): Combined list of STrack.
    """
    exists = {}
    res = []
    for track in stracks_1:
        exists[track.track_id] = True
        res.append(track)
    for track in stracks_2:
        tid = track.track_id
        # Only add to the list of the track ID has not added before
        if not exists.get(tid, False):
            exists[tid] = True
            res.append(track)
    return res


def _remove_duplicate_stracks(
    stracks_1: List[STrack], stracks_2: List[STrack]
) -> Tuple[List[STrack], List[STrack]]:
    """Remove duplicate STrack based on costs computed using
    Intersection-over-Union (IoU) values. Duplicates are identified by
    cost<0.15, the STrack that is more recently created is marked as
    the duplicate.
    Args:
        stracks_1 (List[STrack]): List of STrack.
        stracks_2 (List[STrack]): List of STrack.
    Returns:
        (Tuple[List[STrack], List[STrack]]): Lists of STrack with duplicates
            removed.
    """
    distances = matching.iou_distance(stracks_1, stracks_2)
    pairs = np.where(distances < 0.15)
    duplicates_1 = []
    duplicates_2 = []
    for idx_1, idx_2 in zip(*pairs):
        age_1 = stracks_1[idx_1].frame_id - stracks_1[idx_1].start_frame
        age_2 = stracks_2[idx_2].frame_id - stracks_2[idx_2].start_frame
        if age_1 > age_2:
            duplicates_2.append(idx_2)
        else:
            duplicates_1.append(idx_1)
    return (
        [t for i, t in enumerate(stracks_1) if i not in duplicates_1],
        [t for i, t in enumerate(stracks_2) if i not in duplicates_2],
    )


def _substract_stracks(
    stracks_1: List[STrack], stracks_2: List[STrack]
) -> List[STrack]:
    """Removes stracks_2 from stracks_1.
    Args:
        stracks_1 (List[STrack]): List of STrack.
        stracks_2 (List[STrack]): List of STrack.
    Returns:
        (List[STrack]): List of STrack.
    """
    stracks = {track.track_id: track for track in stracks_1}
    for track in stracks_2:
        tid = track.track_id
        if tid in stracks:
            del stracks[tid]
    return list(stracks.values())
