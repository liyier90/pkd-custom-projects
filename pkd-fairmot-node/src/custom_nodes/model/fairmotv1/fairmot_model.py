"""FairMOT model for human detection and tracking."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from custom_nodes.model.fairmotv1.fairmot_files.tracker import Tracker


class FairMOTModel:
    """FairMOT Model for person tracking.
    Args:
        config (Dict[str, Any]): Model configuration options.
        frame_rate (float): The frame rate of the current video sequence,
            used for computing the size of track buffer.
    Raises:
        ValueError: `score_threshold` is beyond [0, 1].
        ValueError: `K` is less than 1.
        ValueError: `min_box_area` is less than 1.
        ValueError: `track_buffer` is less than 1.
    """

    def __init__(self, config: Dict[str, Any], frame_rate: float) -> None:

        self.logger = logging.getLogger(__name__)
        # Check threshold values
        if not 0 < config["score_threshold"] <= 1:
            raise ValueError("iou_threshold must be in [0, 1]")
        ensure_more_than_zero(config, ["K", "min_box_area", "track_buffer"])

        # Check for weights
        # TODO: need to change this when pushing to PKD
        model_dir = (
            Path(config["weights_parent_dir"]).expanduser()
            / config["weights"]["model_subdir"]
        )

        self.tracker = Tracker(config, model_dir, frame_rate)

    def predict(
        self, image: np.ndarray
    ) -> Tuple[List[np.ndarray], List[str], List[float], List[str]]:
        """Track objects from image.

        Args:
            image (np.ndarray): Image in numpy array.

        Returns:
            (Tuple[List[np.ndarray], List[str], List[float]]): A tuple of
            - Numpy array of detected bounding boxes.
            - List of detection class labels (person).
            - List of detection confidence scores.
            - List of track IDs.

        Raises:
            TypeError: The provided `image` is not a numpy array.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a np.ndarray")
        bboxes, track_ids, bbox_scores = self.tracker.track_objects_from_image(image)
        bbox_labels = ["person"] * len(bboxes)
        return bboxes, bbox_labels, bbox_scores, track_ids


def ensure_more_than_zero(config, key):
    if isinstance(key, str):
        if config[key] < 1:
            raise ValueError(f"{key} must be more than 0")
    elif isinstance(key, list):
        for k in key:
            ensure_more_than_zero(config, k)
    else:
        raise TypeError("'key' must be either 'str' or 'list'")
