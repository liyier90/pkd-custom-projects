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

"""Human detection and tracking model."""

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
from pathlib import Path
from typing import Any, Dict

from peekingduck.pipeline.nodes.node import AbstractNode

from custom_nodes.model.jdev1 import jde_model


class Node(AbstractNode):
    """Initialises and uses JDE tracking model to detect and track people from
    the supplied image frame.

    JDE is a fast and high-performance multiple-object tracker that learns the
    object detection task and appearance embedding task simultaneously in a
    shared neural network.

    Inputs:
        |img|

    Outputs:
        |bboxes|

        |bbox_labels|

        |bbox_scores|

        |obj_tags|

    Configs:
        model_type (:obj:`str`): **{"576x320", "865x480", "1088x608},
            default="576x320"**. |br|
            Defines the type of JDE model to be used.
        weights_parent_dir (:obj:`Optional[str]`): **default = null**. |br|
            Change the parent directory where weights will be stored by
            replacing ``null`` with an absolute path to the desired directory.
        iou_threshold (:obj:`float`): **default = 0.5**. |br|
            Threshold value for Intersecton-over-Union of detections.
        nms_threshold (:obj:`float`): **default = 0.4**. |br|
            Threshold values for non-max suppression.
        score_threshold (:obj:`float`): **default = 0.5**. |br|
            Object confidence score threshold.
        min_box_area (:obj:`int`): **default = 200**. |br|
            Minimum value for area of detected bounding box. Calculated by
            width * height.
        track_buffer (:obj:`int`): **default = 30**. |br|
            Threshold to remove track if track is lost for more frames
            than value.

    References:
        Towards Real-Time Multi-Object Tracking:
        https://arxiv.org/abs/1909.12605v2

        Model weights trained by:
        https://github.com/Zhongdao/Towards-Realtime-MOT
    """

    def __init__(self, config: Dict[str, Any], **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self._frame_rate = 30.0
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
            - obj_tags (List[str]): Tracking IDs, specifically for use
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
            "obj_tags": track_ids,
        }

    def _reset_model(self) -> None:
        """Creates a new instance of the JDE model with the frame rate
        supplied by `mot_metadata`.
        """
        self.logger.info(
            f"Creating new model with frame rate: {self._frame_rate:.2f}..."
        )
        self.model = jde_model.JDEModel(self.config, self._frame_rate)
