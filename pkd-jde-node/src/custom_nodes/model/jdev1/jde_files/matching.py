import lap
import numpy as np
from scipy.spatial.distance import cdist

from custom_nodes.model.jdev1.jde_files import kalman_filter


def bbox_ious(bboxes_1, bboxes_2):
    """Calculates the Intersection-over-Union (IoU) between bounding boxes.
    Bounding boxes have the format (x1, y1, x2, y2), where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner. The algorithm is
    adapted from simple-faster-rcnn-pytorch with modifications such as adding
    1 to area calculations to match the equations used in cython_bbox.

    Args:
        bboxes_1 (np.ndarray): An array with shape (N, 4) where N is the number
            of bounding boxes.
        bboxes_2 (np.ndarray): An array similar to `bboxes_2` with shape (K, 4)
            where K is the number of bounding boxes.

    Returns:
        (np.ndarray): An array with shape (N, K). The element at index (n, k)
        contains the IoU between the n-th bounding box in `bboxes_1` and the
        k-th bounding box in `bboxes_2`.

    Reference:
        simple-faster-rcnn-pytorch
        https://github.com/chenyuntc/simple-faster-rcnn-pytorch/blob/master/model/utils/bbox_tools.py#L145

        cython_bbox:
        https://github.com/samson-wang/cython_bbox/blob/master/src/cython_bbox.pyx
    """
    # top left
    intersect_tl = np.maximum(bboxes_1[:, np.newaxis, :2], bboxes_2[:, :2])
    # bottom right
    intersect_br = np.minimum(bboxes_1[:, np.newaxis, 2:], bboxes_2[:, 2:]) + 1

    intersect_area = np.prod(intersect_br - intersect_tl, axis=2) * (
        intersect_tl < intersect_br
    ).all(axis=2)
    area_1 = np.prod(bboxes_1[:, 2:] - bboxes_1[:, :2] + 1, axis=1)
    area_2 = np.prod(bboxes_2[:, 2:] - bboxes_2[:, :2] + 1, axis=1)
    iou_values = intersect_area / (area_1[:, np.newaxis] + area_2 - intersect_area)

    return iou_values


def embedding_distance(tracks, detections):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float)
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float)
    # Nomalized features
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features))

    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.xyah for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric="maha"
        )
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
        len(btracks) > 0 and isinstance(btracks[0], np.ndarray)
    ):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.xyxy for track in atracks]
        btlbrs = [track.xyxy for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    _ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if _ious.size == 0:
        return _ious

    _ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float),
        np.ascontiguousarray(btlbrs, dtype=np.float),
    )

    return _ious


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            tuple(range(cost_matrix.shape[0])),
            tuple(range(cost_matrix.shape[1])),
        )
    matches, unmatched_a, unmatched_b = [], [], []
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b
