"""YOLOv6 custom node."""

from typing import Any, Dict

import numpy as np
from peekingduck.pipeline.nodes.abstract_node import AbstractNode

from .yolov6_impl import yolov6_model


class Node(AbstractNode):
    """This is a template class of how to write a node for PeekingDuck.

    Args:
        config (:obj:`Dict[str, Any]` | :obj:`None`): Node configuration.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self.config["detect"] = list(range(self.config["num_classes"]))
        self.model = yolov6_model.YOLOv6Model(self.config)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
        """This node does ___.

        Args:
            inputs (dict): Dictionary with keys "__", "__".

        Returns:
            outputs (dict): Dictionary with keys "__".
        """
        bboxes, labels, scores = self.model.predict(inputs["img"])
        bboxes = np.clip(bboxes, 0, 1)
        # outputs = {
        #     "bboxes": np.zeros((1, 4), dtype=np.float32),
        #     # "bbox_labels": np.empty((0)),
        #     "bbox_labels": ["person"],
        #     "bbox_scores": np.zeros((1), dtype=np.float32),
        # }
        outputs = {"bboxes": bboxes, "bbox_labels": labels, "bbox_scores": scores}
        return outputs
