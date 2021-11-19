from typing import Any, Dict

import numpy as np

from peekingduck.pipeline.nodes.node import AbstractNode


class Node(AbstractNode):
    def __init__(self, config: Dict[str, Any] = None, **kwargs) -> None:
        super().__init__(config, node_path=__name__, **kwargs)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        bboxes = inputs["bboxes"]
        expanded_bboxes = np.array(list(map(self._expand_bbox, bboxes)))
        return {"bboxes": expanded_bboxes}

    def _expand_bbox(self, bbox: np.ndarray) -> np.ndarray:
        h = bbox[3] - bbox[1]
        w = bbox[2] - bbox[0]
        bbox[0] = max(0, bbox[0] - w * self.scale_factor)
        bbox[1] = max(0, bbox[1] - h * self.scale_factor)
        bbox[2] = min(1, bbox[2] + w * self.scale_factor)
        bbox[3] = min(1, bbox[3] + h * self.scale_factor)

        return bbox
