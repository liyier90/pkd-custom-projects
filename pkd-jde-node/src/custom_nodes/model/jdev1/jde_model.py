from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from .jde_files.tracker import Tracker


class JDEModel:
    def __init__(self, config: Dict[str, Any], frame_rate: float) -> None:
        # Check threshold values
        if not 0 <= config["iou_threshold"] <= 1:
            raise ValueError("iou_threshold must be in [0, 1]")
        if not 0 <= config["nms_threshold"] <= 1:
            raise ValueError("nms_threshold must be in [0, 1]")
        if not 0 <= config["score_threshold"] <= 1:
            raise ValueError("score_threshold must be in [0, 1]")

        # Check for weights
        # weights_dir, model_dir = finder.find_paths(
        #     config["root"], config["weights"], config["weights_parent_dir"]
        # )
        print()
        weights_dir = (
            Path(config["weights_parent_dir"]).expanduser() / "peekingduck_weights"
        )
        model_dir = weights_dir / config["weights"]["model_subdir"]

        self.tracker = Tracker(config, model_dir, frame_rate)

    def predict(
        self, image: np.ndarray
    ) -> Tuple[List[np.ndarray], List[str], List[float]]:
        return self.tracker.track_objects_from_image(image)
