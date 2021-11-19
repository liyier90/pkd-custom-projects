import numpy as np
from scipy.optimize import linear_sum_assignment


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
    def iou(bbox, preds):
        top_left = np.c_[
            np.maximum(bbox[0], preds[:, 0])[:, np.newaxis],
            np.maximum(bbox[1], preds[:, 1])[:, np.newaxis],
        ]
        lower_right = np.c_[
            np.minimum(bbox[0] + bbox[2], preds[:, 0] + preds[:, 2])[:, np.newaxis],
            np.minimum(bbox[1] + bbox[3], preds[:, 1] + preds[:, 3])[:, np.newaxis],
        ]
        width_height = np.maximum(0.0, lower_right - top_left)

        area_intersection = width_height.prod(axis=1)
        area_bbox = bbox[2:].prod()
        area_preds = preds[:, 2:].prod(axis=1)

        return area_intersection / (area_bbox + area_preds - area_intersection)

    @staticmethod
    def match_bboxes_to_tracks(bboxes, preds, iou_threshold):
        if len(preds) == 0:
            return np.array([]), np.arange(len(bboxes)), np.array([])

        iou_matrix = np.zeros((len(bboxes), len(preds)), dtype=np.float32)

        for b, bbox in enumerate(bboxes):
            iou_matrix[b] = Utils.iou(bbox, preds)
        matched_bbox_indices, matched_pred_indices = linear_sum_assignment(-iou_matrix)

        unmatched_bboxes = []
        for b, bbox in enumerate(bboxes):
            if b not in matched_bbox_indices:
                unmatched_bboxes.append(b)

        unmatched_tracks = []
        for t, _ in enumerate(preds):
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
