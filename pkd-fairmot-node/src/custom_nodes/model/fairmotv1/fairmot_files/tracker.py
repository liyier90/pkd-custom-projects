import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from custom_nodes.model.fairmotv1.fairmot_files.dla import DLASeg
from custom_nodes.model.fairmotv1.fairmot_files.kalman_filter import KalmanFilter
from custom_nodes.model.fairmotv1.fairmot_files.track import STrack


class Tracker:
    heads = {"hm": 1, "wh": 4, "id": 128, "reg": 2}
    down_ratio = 4
    final_kernel = 1
    head_conv = 256
    last_level = 5

    mean = np.array([0.408, 0.447, 0.470], dtype=np.float32).reshape((1, 1, 3))
    std = np.array([0.289, 0.274, 0.278], dtype=np.float32).reshape((1, 1, 3))

    def __init__(
        self, config: Dict[str, Any], model_dir: Path, frame_rate: float
    ) -> None:
        self.logger = logging.getLogger(__name__)

        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._create_model(model_dir)

        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []
        self.removed_stracks: List[STrack] = []

        self.frame_id = 0
        self.max_time_lost = int(frame_rate / 30.0 * config["track_buffer"])
        self.max_per_image = self.config["K"]

        self.kalman_filter = KalmanFilter()

    def track_objects_from_image(
        self, image: np.ndarray
    ) -> Tuple[List[np.ndarray], List[str], List[float]]:
        return [], [], []

    def _create_model(self, model_dir: Path):
        model_type = self.config["model_type"]
        model_path = model_dir / self.config["weights"]["model_file"][model_type]
        self.logger.info(
            "FairMOT model loaded with the following config:\n\t"
            f"Model type: {model_type}\n\t"
            f"Score threshold: {self.config['score_threshold']}\n\t"
            f"Max number of output objects: {self.config['K']}\n\t"
            f"Min bounding box area: {self.config['min_box_area']}\n\t"
            f"Track buffer: {self.config['track_buffer']}"
        )
        return self._load_model_weights(model_path)

    def _load_model_weights(self, model_path: Path):
        if not model_path.is_file():
            raise ValueError(
                f"Model file does not exist. Please check that {model_path} exists."
            )

        ckpt = torch.load(str(model_path), map_location="cpu")
        model = DLASeg(
            "dla34",
            self.heads,
            pretrained=True,
            down_ratio=self.down_ratio,
            final_kernel=self.final_kernel,
            last_level=self.last_level,
            head_conv=self.head_conv,
        )
        model.load_state_dict(ckpt["state_dict"], strict=False)
        model.to(self.device).eval()
        return model
