# AISG boilerplate
from typing import Any, Dict

from peekingduck.pipeline.nodes.draw.utils.constants import TOMATO
from peekingduck.pipeline.nodes.node import AbstractNode

from .utils.bbox import draw_annotations


class Node(AbstractNode):
    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        for i, key in enumerate(self.keys):
            draw_annotations(
                inputs["img"],
                inputs["bboxes"],
                inputs["obj_tags"],
                TOMATO,
                self.location[i],
                key,
            )

        return {}
