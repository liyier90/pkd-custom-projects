import copy
import os

import motmetrics as mm
import numpy as np

mm.lap.default_solver = "lap"


class Evaluator:
    def __init__(self, seq_dir):
        self.seq_dir = seq_dir

        self.load_annotations()
        self.reset_accumulator()

    def load_annotations(self):
        gt_path = self.seq_dir / "gt" / "gt.txt"
        self.gt_frame_dict = read_results(gt_path, is_gt=True)
        self.gt_ignore_frame_dict = read_results(gt_path, is_ignore=True)

    def reset_accumulator(self):
        self.acc = mm.MOTAccumulator(auto_id=True)

    def eval_frame(self, frame_id, trk_tlwhs, trk_ids, rtn_events=False):
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

        if (
            rtn_events
            and iou_distance.size > 0
            and hasattr(self.acc, "last_mot_events")
        ):
            # only supported by https://github.com/longcw/py-motmetrics
            events = self.acc.last_mot_events
        else:
            events = None
        return events

    def eval_file(self, results_path):
        self.reset_accumulator()

        result_frame_dict = read_results(results_path, is_gt=False)
        frames = sorted(
            list(set(self.gt_frame_dict.keys()) | set(result_frame_dict.keys()))
        )
        for frame_id in frames:
            trk_objs = result_frame_dict.get(frame_id, [])
            trk_tlwhs, trk_ids = unzip_objs(trk_objs)[:2]
            self.eval_frame(frame_id, trk_tlwhs, trk_ids, rtn_events=False)

        return self.acc

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

        mh = mm.metrics.create()
        summary = mh.compute_many(
            accs, metrics=metrics, names=names, generate_overall=True
        )

        return summary

    @staticmethod
    def save_summary(summary, filename):
        import pandas as pd

        writer = pd.ExcelWriter(filename)
        summary.to_excel(writer)
        writer.save()


def read_results(results_path, is_gt=False, is_ignore=False):
    return read_mot_results(results_path, is_gt, is_ignore)


def read_mot_results(results_path, is_gt, is_ignore):
    valid_labels = {1}
    ignore_labels = {2, 7, 8, 12}
    results_dict = dict()
    if results_path.is_file():
        with open(results_path, "r") as f:
            for line in f.readlines():
                linelist = line.split(",")
                if len(linelist) < 7:
                    continue
                fid = int(linelist[0])
                if fid < 1:
                    continue
                results_dict.setdefault(fid, list())

                if is_gt:
                    label = int(float(linelist[7]))
                    mark = int(float(linelist[6]))
                    if mark == 0 or label not in valid_labels:
                        continue
                    score = 1
                elif is_ignore:
                    label = int(float(linelist[7]))
                    vis_ratio = float(linelist[8])
                    if label not in ignore_labels and vis_ratio >= 0:
                        continue
                    score = 1
                else:
                    score = float(linelist[6])

                tlwh = tuple(map(float, linelist[2:6]))
                target_id = int(linelist[1])

                results_dict[fid].append((tlwh, target_id, score))

    return results_dict


def unzip_objs(objs):
    if len(objs) > 0:
        tlwhs, ids, scores = zip(*objs)
    else:
        tlwhs, ids, scores = [], [], []
    tlwhs = np.asarray(tlwhs, dtype=float).reshape(-1, 4)

    return tlwhs, ids, scores
