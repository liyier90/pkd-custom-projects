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
    """This is a template class of how to write a node for PeekingDuck.

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
        """This node does ___.

        Args:
            inputs (dict): Dictionary with keys "__", "__".

        Returns:
            outputs (dict): Dictionary with keys "__".
        """
        frame_rate = inputs.get("frame_rate", self._frame_rate)
        reset_model = inputs.get("reset_model", False)

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
