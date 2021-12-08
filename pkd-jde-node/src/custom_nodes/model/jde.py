# Copyright 2021 AI Singapore
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Node template for creating custom nodes.
"""

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
from pathlib import Path
from typing import Any, Dict

from peekingduck.pipeline.nodes.node import AbstractNode

from custom_nodes.model.jdev1 import jde_model


class Node(AbstractNode):
    """JDE tracking model.

    Args:
        config (:obj:`Dict[str, Any]`): Node configuration.
    """

    def __init__(self, config: Dict[str, Any], **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self._frame_rate = 30.0
        self.config = config
        self.config["root"] = Path(__file__).resolve().parents[4]

        self.model = jde_model.JDEModel(self.config, self._frame_rate)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Tracks objects from image.

        Specifically for use with MOT evaluation, will attempt to get optional
        input `mot_metadata` and recreate `JDEModel` with the appropriate
        frame rate when necessary.

        Args:
            inputs (Dict[str, Any]): Dictionary with keys "img". When running
                under MOT evaluation, contains "mot_metadata" key as well.

        Returns:
            outputs (dict): Dictionary containing:
            - bboxes (List[np.ndarray]): Bounding boxes for tracked targets.
            - bbox_labels (List[str]): Tracking IDs, for compatibility with
                draw nodes.
            - bbox_scores (List[float]): Detection confidence scores.
            - obj_track_ids (List[str]): Tracking IDs, specifically for use
                with `mot_evaluator`.
        """
        metadata = inputs.get(
            "mot_metadata", {"frame_rate": self._frame_rate, "reset_model": False}
        )
        frame_rate = metadata["frame_rate"]
        reset_model = metadata["reset_model"]

        if frame_rate != self._frame_rate or reset_model:
            self._frame_rate = frame_rate
            self._reset_model()

        bboxes, track_ids, scores = self.model.predict(inputs["img"])
        return {
            "bboxes": bboxes,
            "bbox_labels": track_ids,
            "bbox_scores": scores,
            "obj_track_ids": track_ids,
        }

    def _reset_model(self):
        self.logger.info(
            f"Creating new model with frame rate: {self._frame_rate:.2f}..."
        )
        self.model = jde_model.JDEModel(self.config, self._frame_rate)
