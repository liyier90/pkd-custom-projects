# AISG Boilerplate
from typing import Any, Dict

from peekingduck.pipeline.nodes.node import AbstractNode

from .utils.pkd_opencv_tracker import OpenCvTracker
from .utils.sort import Sort


class Node(AbstractNode):
    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)

        self.tracker = Sort("csrt", True)
        # self.tracker = Sort("csrt")
        # self.tracker = OpenCvTracker()

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        obj_track_ids = self.tracker.update_and_get_tracks(
            inputs["img"], inputs["bboxes"]
        )
        return {"obj_track_ids": obj_track_ids}
