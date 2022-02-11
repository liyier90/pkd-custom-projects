"""Human detection and tracking model that balances the importance between
detection and re-ID tasks.
"""

from pathlib import Path
from typing import Any, Dict

from peekingduck.pipeline.nodes.node import AbstractNode

from custom_nodes.model.fairmotv1 import fairmot_model


class Node(AbstractNode):  # pylint: disable=too-few-public-methods
    """This is a template class of how to write a node for PeekingDuck.

    Args:
        config (:obj:`Dict[str, Any]` | :obj:`None`): Node configuration.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self._frame_rate = 30.0
        self.config["root"] = Path(__file__).resolve().parents[4]

        self.model = fairmot_model.FairMOTModel(self.config, self._frame_rate)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """This node does ___.

        Args:
            inputs (dict): Dictionary with keys "__", "__".

        Returns:
            outputs (dict): Dictionary with keys "__".
        """
        metadata = inputs.get(
            "mot_metadata", {"frame_rate": self._frame_rate, "reset_model": False}
        )
        frame_rate = metadata["frame_rate"]
        reset_model = metadata["reset_model"]

        if frame_rate != self._frame_rate or reset_model:
            self._frame_rate = frame_rate
            self._reset_model()

        bboxes, bbox_labels, bbox_scores, track_ids = self.model.predict(inputs["img"])
        outputs = {
            "bboxes": bboxes,
            "bbox_labels": bbox_labels,
            "bbox_scores": bbox_scores,
            "obj_tags": track_ids,
        }
        return outputs

    def _reset_model(self) -> None:
        """Creates a new instance of the FairMOT model with the frame rate
        supplied by `mot_metadata`.
        """
        self.logger.info(
            f"Creating new model with frame rate: {self._frame_rate:.2f}..."
        )
        self.model = fairmot_model.FairMOTModel(self.config, self._frame_rate)
