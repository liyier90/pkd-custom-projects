"""MOT metrics evaluator."""

import copy
from pathlib import Path

import motmetrics as mm
import numpy as np

mm.lap.default_solver = "lap"


class Evaluator:
    """Evaluates MOT tracking results."""

    def __init__(self, seq_dir: Path) -> None:
        self.seq_dir = seq_dir

        self._load_annotations()
        self._reset_accumulator()

    def eval_file(self, results_path: Path) -> mm.MOTAccumulator:
        """Evaluates a tracking results file.

        Args:
            results_path (Path): Path to results file for a video sequence.

        Returns:
            (mm.MOTAccumulator): Accumulator with objects/detections for each
                frame in the video sequence.
        """
        self._reset_accumulator()

        result_frame_dict = read_results(results_path, is_gt=False)
        frames = sorted(
            list(set(self.gt_frame_dict.keys()) | set(result_frame_dict.keys()))
        )
        for frame_id in frames:
            trk_objs = result_frame_dict.get(frame_id, [])
            trk_tlwhs, trk_ids = unzip_objs(trk_objs)[:2]
            self.eval_frame(frame_id, trk_tlwhs, trk_ids)

        return self.acc

    def eval_frame(self, frame_id, trk_tlwhs, trk_ids) -> None:
        """Evaluates one frame.

        Args:
            frame_id (int): Frame index.
        """
        # results
        trk_tlwhs = np.copy(trk_tlwhs)
        trk_ids = np.copy(trk_ids)

        # gts
        gt_objs = self.gt_frame_dict.get(frame_id, [])
        gt_tlwhs, gt_ids = unzip_objs(gt_objs)[:2]

        # ignore boxes
        ignore_objs = self.gt_ignore_frame_dict.get(frame_id, [])
        ignore_tlwhs = unzip_objs(ignore_objs)[0]

        # remove ignored results
        keep = np.ones(len(trk_tlwhs), dtype=bool)
        iou_distance = mm.distances.iou_matrix(ignore_tlwhs, trk_tlwhs, max_iou=0.5)
        if len(iou_distance) > 0:
            match_is, match_js = mm.lap.linear_sum_assignment(iou_distance)
            match_is, match_js = map(
                lambda a: np.asarray(a, dtype=int), [match_is, match_js]
            )
            match_ious = iou_distance[match_is, match_js]

            match_js = np.asarray(match_js, dtype=int)
            match_js = match_js[np.logical_not(np.isnan(match_ious))]
            keep[match_js] = False
            trk_tlwhs = trk_tlwhs[keep]
            trk_ids = trk_ids[keep]

        # get distance matrix
        iou_distance = mm.distances.iou_matrix(gt_tlwhs, trk_tlwhs, max_iou=0.5)

        # acc
        self.acc.update(gt_ids, trk_ids, iou_distance)

    def _load_annotations(self) -> None:
        """Loads ground truth annotations."""
        gt_path = self.seq_dir / "gt" / "gt.txt"
        self.gt_frame_dict = read_results(gt_path, is_gt=True)
        self.gt_ignore_frame_dict = read_results(gt_path, is_ignore=True)

    def _reset_accumulator(self):
        """Resets the accumulator."""
        self.acc = mm.MOTAccumulator(auto_id=True)

    @staticmethod
    def get_summary(
        accs,
        names,
        metrics=("mota", "num_switches", "idp", "idr", "idf1", "precision", "recall"),
    ):
        names = copy.deepcopy(names)
        if metrics is None:
            metrics = mm.metrics.motchallenge_metrics
        metrics = copy.deepcopy(metrics)

        metrics_host = mm.metrics.create()
        summary = metrics_host.compute_many(
            accs, metrics=metrics, names=names, generate_overall=True
        )

        return summary

    @staticmethod
    def save_summary(summary, filename):
        raise NotImplementedError
        # import pandas as pd

        # writer = pd.ExcelWriter(filename)
        # summary.to_excel(writer)
        # writer.save()

