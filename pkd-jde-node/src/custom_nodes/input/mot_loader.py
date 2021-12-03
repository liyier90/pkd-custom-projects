"""
Node template for creating custom nodes.
"""

import configparser
from pathlib import Path
from typing import Any, Dict

import cv2
from peekingduck.pipeline.nodes.node import AbstractNode


class Node(AbstractNode):
    """This is a template class of how to write a node for PeekingDuck.

    Args:
        config (:obj:`Dict[str, Any]` | :obj:`None`): Node configuration.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)

        self.input_dir = Path(self.input_dir).expanduser()  # type: ignore
        self.sequences = list(self.input_dir.iterdir())
        self.num_sequences = len(self.sequences)
        self.seq_idx = 0
        self.image_loader = iter(ImageLoader(self.current_sequence))
        self.frame_rate = self._get_seq_frame_rate()

    @property
    def current_sequence(self):
        """Path to the current sequence directory of images."""
        return self.sequences[self.seq_idx]

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """This node does ___.

        Args:
            inputs (dict): Dictionary with keys "__", "__".

        Returns:
            outputs (dict): Dictionary with keys "__".
        """
        try:
            img_path = next(self.image_loader)
        except StopIteration:
            self.seq_idx += 1
            if self.seq_idx >= self.num_sequences:
                # Run out of input sequence
                return {"img": None, "seq_name": None, "pipeline_end": True}
            self.image_loader = iter(ImageLoader(self.current_sequence))
            self.frame_rate = self._get_seq_frame_rate()
            img_path = next(self.image_loader)

        img = cv2.imread(str(img_path))

        print(img_path)

        return {
            "img": img,
            "seq_name": img_path.parent.name,
            "frame_rate": self.frame_rate,
            "pipeline_end": False,
        }

    def _get_seq_frame_rate(self):
        seq_config = configparser.ConfigParser()
        seq_config.read(self.current_sequence / "seqinfo.ini")
        return float(seq_config["Sequence"]["frameRate"])


class ImageLoader:
    def __init__(self, path):
        self.count: int
        self.files = sorted(list((path / "img1").glob("*.jpg")))
        self.num_files = len(self.files)

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if self.count >= self.num_files:
            raise StopIteration
        return self.files[self.count]
