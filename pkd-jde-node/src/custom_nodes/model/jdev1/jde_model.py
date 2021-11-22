from typing import Any, Dict, List, Tuple

import numpy as np

from .jde_files.tracker import Tracker


class JDEModel:
    def __init__(self, config: Dict[str, Any]) -> None:
        # Check threshold values
        if not 0 <= config["iou_threshold"] <= 1:
            raise ValueError("iou_threshold must be in [0, 1]")
        if not 0 <= config["nms_threshold"] <= 1:
            raise ValueError("nms_threshold must be in [0, 1]")
        if not 0 <= config["score_threshold"] <= 1:
            raise ValueError("score_threshold must be in [0, 1]")

        # Check for weights

        self.tracker = Tracker(config)

    def predict(
        self, image: np.ndarray
    ) -> Tuple[List[np.ndarray], List[str], List[float], List[str]]:
        return self.tracker.track_objects_from_image(image)
