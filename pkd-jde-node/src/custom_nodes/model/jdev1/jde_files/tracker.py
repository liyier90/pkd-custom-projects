import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch


class Tracker:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.logger = logging.getLogger(__name__)

        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._create_darknet_model()

    def track_objects_from_image(
        self, image: np.ndarray
    ) -> Tuple[List[np.ndarray], List[str], List[float], List[str]]:
        padded_image, resized_image = self._preprocess(image)
        padded_image = torch.from_numpy(padded_image).to(self.device).unsqueeze(0)

        online_targets = self.model.update(padded_image, resized_image)
        online_tlwhs = []
        online_ids = []
        for target in online_targets:
            tlwh = target.tlwh
            target_id = target.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if not vertical and tlwh[2] * tlwh[3] > self.config["min_box_area"]:
                online_tlwhs.append(tlwh)
                online_ids.append(target_id)
        # Postprocess here

        return

    def _create_darknet_model(self):
        model_type = self.config["model_type"]
        model_path = self.config["root"] / self.config["model_files"][model_type]
        model_settings = self._parse_model_config(
            self.config["config_files"][model_type]
        )
        self.input_size = [
            int(model_settings[0]["width"]),
            int(model_settings[0]["height"]),
        ]
        return self._load_darknet_weights(model_path, model_settings)

    def _load_darknet_weights(self, model_path, model_settings):
        ckpt = torch.load(str(model_path), map_location="cpu")
        model = Darknet(model_settings, nID=14455)
        model.load_state_dict(ckpt["model"])
        model.to(self.device).eval()
        return model

    def _preprocess(self, image: np.ndarray):
        # Resizing input frame
        video_h, video_w = image.shape[:2]
        ratio_w, ratio_h = (
            float(self.input_size[0]) / video_w,
            float(self.input_size[1]) / video_h,
        )
        ratio = min(ratio_w, ratio_h)
        width, height = int(video_w * ratio), int(video_h * ratio)
        resized_image = cv2.resize(image, (width, height))
        # Padded resize
        padded_image, _, _, _ = self._letterbox(
            resized_image, height=self.input_size[1], width=self.input_size[0]
        )
        # Normalize RGB
        padded_image = padded_image[..., ::-1].transpose(2, 0, 1)
        padded_image = np.ascontiguousarray(padded_image, dtype=np.float32)
        padded_image /= 255.0

        return padded_image, resized_image

    @staticmethod
    def _letterbox(img, height=608, width=1088, color=(127.5, 127.5, 127.5)):
        """Resizes a rectangular image to a padded rectangular."""
        shape = img.shape[:2]  # shape = [height, width]
        ratio = min(float(height) / shape[0], float(width) / shape[1])
        # new_shape = [width, height]
        new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))
        dw = (width - new_shape[0]) / 2  # width padding
        dh = (height - new_shape[1]) / 2  # height padding
        top, bottom = round(dh - 0.1), round(dh + 0.1)
        left, right = round(dw - 0.1), round(dw + 0.1)
        img = cv2.resize(
            img, new_shape, interpolation=cv2.INTER_AREA
        )  # resized, no border
        # padded rectangular
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )
        return img, ratio, dw, dh

    @staticmethod
    def _parse_model_config(config_path: Path) -> List[Dict[str, Any]]:
        """Parse model configuration with context manager"""
        with open(config_path) as infile:
            lines = [
                line
                for line in map(str.strip, infile.readlines())
                if line and not line.startswith("#")
            ]
        module_defs: List[Dict[str, Any]] = []
        for line in lines:
            if line.startswith("["):
                module_defs.append({})
                module_defs[-1]["type"] = line[1:-1].rstrip()
                if module_defs[-1]["type"] == "convolutional":
                    module_defs[-1]["batch_normalize"] = 0
            else:
                key, value = tuple(map(str.strip, line.split("=")))
                if value.startswith("$"):
                    value = module_defs[0].get(value.strip("$"), None)
                module_defs[-1][key] = value
        return module_defs
