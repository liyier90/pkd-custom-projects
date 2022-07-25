import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from .yolov6_files.detector import Detector


class YOLOv6Model:  # pylint: disable=too-few-public-methods
    """Validates configuration, loads YOLOv6 model, and performs inference.

    Configuration options are validated to ensure they have valid types and
    values. Model weights files are downloaded if not found in the location
    indicated by the `weights_dir` configuration option.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.weights = self.config["weights"][self.config["model_format"]]
        model_dir = Path(__file__).resolve().parents[6] / "YOLOv6" / "weights"
        with open(model_dir / self.weights["classes_file"]) as infile:
            class_names = [line.strip() for line in infile.readlines()]

        self.detect_ids = self.config["detect"]
        self.detector = Detector(
            model_dir,
            class_names,
            self.detect_ids,
            self.config["model_type"],
            self.config["num_classes"],
            self.weights["model_file"],
            self.config["max_detections"],
            self.config["agnostic_nms"],
            self.config["fuse"],
            self.config["half"],
            self.config["multi_label"],
            self.config["input_size"],
            self.config["iou_threshold"],
            self.config["score_threshold"],
        )

    def predict(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predicts bboxes from image.

        Args:
            image (np.ndarray): Input image frame.

        Returns:
            (Tuple[np.ndarray, np.ndarray, np.ndarray]): Returned tuple
            contains:
            - An array of detection bboxes
            - An array of human-friendly detection class names
            - An array of detection scores

        Raises:
            TypeError: The provided `image` is not a numpy array.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a np.ndarray")
        return self.detector.predict_object_bbox_from_image(image)
