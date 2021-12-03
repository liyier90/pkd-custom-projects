"""
Node template for creating custom nodes.
"""
from pathlib import Path
from typing import Any, Dict, Union

from peekingduck.pipeline.nodes.node import AbstractNode


class Node(AbstractNode):
    """This is a template class of how to write a node for PeekingDuck.

    Args:
        config (:obj:`Dict[str, Any]` | :obj:`None`): Node configuration.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)

        self.output_dir: Union[Path, str]
        self.output_dir = Path(self.output_dir).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
        """This node does ___.

        Args:
            inputs (dict): Dictionary with keys "__", "__".

        Returns:
            outputs (dict): Dictionary with keys "__".
        """
        print(inputs["seq_name"], inputs["frame_idx"])
        return {}
