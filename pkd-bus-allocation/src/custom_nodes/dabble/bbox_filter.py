# AISG boilerplate
from operator import itemgetter
from typing import Any, Dict, List, Tuple

import numpy as np
from shapely.geometry import Polygon

from peekingduck.pipeline.nodes.node import AbstractNode


class Node(AbstractNode):
    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        bboxes, bbox_labels, bbox_scores = itemgetter(
            "bboxes", "bbox_labels", "bbox_scores"
        )(inputs)
        bboxes, bbox_labels, bbox_scores = self.filter_score_thres(
            bboxes, bbox_labels, bbox_scores
        )
        bboxes, bbox_labels, bbox_scores = self.filter_zone_overlap(
            bboxes, bbox_labels, bbox_scores
        )

        return {
            "bboxes": bboxes,
            "bbox_labels": bbox_labels,
            "bbox_scores": bbox_scores,
        }

    def filter_score_thres(
        self, bboxes: List[np.array], bbox_labels: List[str], bbox_scores: List[float]
    ) -> Tuple[List[np.array], List[str], List[float]]:
        score_thres = {"bus": 0.8}
        mask = np.ones(bbox_labels.shape, dtype=bool)
        for cls, thres in score_thres.items():
            mask &= (bbox_labels == cls) & (bbox_scores >= thres) | (bbox_labels != cls)

        return bboxes[mask], bbox_labels[mask], bbox_scores[mask]

    def filter_zone_overlap(
        self, bboxes: List[np.array], bbox_labels: List[str], bbox_scores: List[float]
    ) -> Tuple[List[np.array], List[str], List[float]]:
        overlap_thres = {
            "bus": {
                "thres": 0.25,
                "zone": [
                    [1000 / 1920, 300 / 1080],
                    [1225 / 1920, 300 / 1080],
                    [1100 / 1920, 1],
                    [300 / 1920, 1],
                ],
            }
        }
        mask = np.ones(bbox_labels.shape, dtype=bool)
        for cls, thres_d in overlap_thres.items():
            zone_poly = Polygon(thres_d["zone"])
            for i, (det_bbox, det_cls) in enumerate(zip(bboxes, bbox_labels)):
                if cls == det_cls:
                    bbox_poly = convert_bbox_to_polygon(det_bbox)
                    if zone_poly.intersects(bbox_poly):
                        mask[i] = (
                            zone_poly.intersection(bbox_poly).area / bbox_poly.area
                            >= thres_d["thres"]
                        )
                    else:
                        mask[i] = False

        return bboxes[mask], bbox_labels[mask], bbox_scores[mask]


def convert_bbox_to_polygon(bbox: np.array) -> Polygon:
    return Polygon(
        [(bbox[0], bbox[1]), (bbox[2], bbox[1]), (bbox[2], bbox[3]), (bbox[0], bbox[3])]
    )
