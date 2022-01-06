import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch


class Tracker:
    def __init__(
        self, config: Dict[str, Any], model_dir: Path, frame_rate: float
    ) -> None:
        self.logger = logging.getLogger(__name__)

        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._create_model(model_dir)

    def track_objects_from_image(
        self, image: np.ndarray
    ) -> Tuple[List[np.ndarray], List[str], List[float]]:
        return [], [], []

    def _create_model(self, model_dir: Path):
        model_type = self.config["model_type"]
        model_paths = {
            "model": model_dir / self.config["weights"]["model_file"][model_type],
            "base": model_dir / self.config["weights"]["model_file"]["base"],
        }
        self.logger.info(
            "FairMOT model loaded with the following config:\n\t"
            f"Model type: {model_type}\n\t"
            f"Score threshold: {self.config['score_threshold']}\n\t"
            f"Max number of output objects: {self.config['K']}\n\t"
            f"Min bounding box area: {self.config['min_box_area']}\n\t"
            f"Track buffer: {self.config['track_buffer']}"
        )
        return self._load_model_weights(model_paths)

    def _load_model_weights(self, model_paths: Dict[str, Path]):
        for key in model_paths:
            if not model_paths[key].is_file():
                raise ValueError(
                    "Model file does not exist. Please check that "
                    f"{model_paths[key]} exists."
                )

        print(model_paths)
