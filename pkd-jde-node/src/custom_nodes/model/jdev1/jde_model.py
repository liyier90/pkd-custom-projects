"""JDE model for human detection and tracking."""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from custom_nodes.model.jdev1.jde_files.tracker import Tracker

# from peekingduck.weights_utils import finder


class JDEModel:
    """JDE Model with model types: 576x320, 865x480, and 1088x608.

    Args:
        config (Dict[str, Any]): Model configuration options.
        frame_rate (float): The frame rate of the current video sequence,
            used for computing the size of track buffer.

    Raises:
        ValueError: `iou_threshold` is beyond [0, 1].
        ValueError: `nms_threshold` is beyond [0, 1].
        ValueError: `score_threshold` is beyond [0, 1].
    """

    def __init__(self, config: Dict[str, Any], frame_rate: float) -> None:
        # Check threshold values
        if not 0 <= config["iou_threshold"] <= 1:
            raise ValueError("iou_threshold must be in [0, 1]")
        if not 0 <= config["nms_threshold"] <= 1:
            raise ValueError("nms_threshold must be in [0, 1]")
        if not 0 <= config["score_threshold"] <= 1:
            raise ValueError("score_threshold must be in [0, 1]")

        # # Check for weights
        # weights_dir, model_dir = finder.find_paths(
        #     config["root"], config["weights"], config["weights_parent_dir"]
        # )
        weights_dir = (
            Path(config["weights_parent_dir"]).expanduser() / "peekingduck_weights"
        )
        model_dir = weights_dir / config["weights"]["model_subdir"]

        self.tracker = Tracker(config, model_dir, frame_rate)

    def predict(
        self, image: np.ndarray
    ) -> Tuple[List[np.ndarray], List[str], List[float]]:
        """Track objects from image.

        Args:
            image (np.ndarray): Image in numpy array.

        Returns:
            (Tuple[List[np.ndarray], List[str], List[float]]): A tuple of
            - Numpy array of detected bounding boxes.
            - List of track IDs.
            - List of detection confidence scores.
        """
        return self.tracker.track_objects_from_image(image)
