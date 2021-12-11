"""Simple multiple object tracking for detected bounding boxes."""

from typing import Any, Dict

from peekingduck.pipeline.nodes.node import AbstractNode

from custom_nodes.dabble.trackingv1.detection_tracker import DetectionTracker


class Node(AbstractNode):
    """Uses bounding boxes detected by an object detector model to track multiple
    objects.

    Currently, two types of tracking algorithms can be selected: MOSSE, IOU.

    Please view each tracker's script, or the "Multi Object Tracking" use case
    documentation for more details.

    Inputs:
        |bboxes|

        |bbox_scores|

        |bbox_labels|

    Outputs:
        |obj_tags|

    Configs:
        tracking_type (:obj:`str`): **{"iou", "mosse"}, default="iou"**. |br|
            Type of tracking algorithm to be used. For more information about
            the trackers, please view the use case documentation.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self.tracker = DetectionTracker(self.tracking_type)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Tracks detection bounding boxes.

        Args:
            inputs (Dict[str, Any]): Dictionary with keys "__", "__".

        Returns:
            outputs (Dict[str, Any]): Tracking IDs of bounding boxes.
                "obj_tags" key is used for compatibility with draw nodes.

        """

        track_ids = self.tracker.track_detections(inputs)

        return {"obj_tags": track_ids}
