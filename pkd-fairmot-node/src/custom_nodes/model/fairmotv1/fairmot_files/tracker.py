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

    def track_objects_from_image(
        self, image: np.ndarray
    ) -> Tuple[List[np.ndarray], List[str], List[float]]:
        return [], [], []
