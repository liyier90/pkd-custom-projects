"""FairMOT model for human detection and tracking."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from custom_nodes.model.fairmotv1.fairmot_files.tracker import Tracker


class FairMOTModel:
    def __init__(self, config: Dict[str, Any], frame_rate: float) -> None:
        self.logger = logging.getLogger(__name__)
        # Check threshold values

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
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a np.ndarray")
        bboxes, track_ids, bbox_scores = self.tracker.track_objects_from_image(image)
        bbox_labels = ["person"] * len(bboxes)
        return bboxes, bbox_labels, bbox_scores, track_ids
