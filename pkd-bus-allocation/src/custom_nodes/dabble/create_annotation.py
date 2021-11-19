# AISG boilerplate
from typing import Any, Dict

from peekingduck.pipeline.nodes.node import AbstractNode


class Node(AbstractNode):
    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        obj_annotations = [""] * len(inputs["bbox_labels"])
        for i, (label, score, track_id, bus_status) in enumerate(
            zip(
                inputs["bbox_labels"],
                inputs["bbox_scores"],
                inputs["obj_track_ids"],
                inputs["bus_statuses"],
            )
        ):
            obj_annotations[i] = {
                "label": f"{label} {score:.2f}",
                "track_id": track_id,
                "bus_status": bus_status,
            }

        return {"obj_annotations": obj_annotations}
