"""
Node template for creating custom nodes.
"""

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Union

import cv2
from peekingduck.pipeline.nodes.node import AbstractNode

from custom_nodes.input.mot_loader_files.loader import ImageLoader, SequenceLoader


class Node(AbstractNode):
    """Loads MOT dataset for evaluation.

    Args:
        config (:obj:`Dict[str, Any]`): Node configuration.

    Attributes:
        input_dir (Union[Path, str]): Path to where MOT dataset (subset) is
            located, e.g. /path/to/MOT16/train/
        sequences (List[Path]): List of video sequences in the dataset.
        num_sequences (int): Total number of video sequences.
        seq_idx (int): Index of the current sequence being loaded.
        image_loader (ImageLoader): Loader to load JPG images from the `img1`
            subdirectories in each sequence directory.
        frame_rate (float): Frame rate of the video sequence, used to determine
            buffer size in JDE tracker.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)

        self.input_dir: Union[Path, str]
        if self.input_dir is None:
            raise ValueError("input_dir cannot be unset")
        self.input_dir = Path(self.input_dir).expanduser()
        self.seq_loader: SequenceLoader = iter(SequenceLoader(self.input_dir))
        self.img_loader = iter(ImageLoader(next(self.seq_loader)))

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Loads the next image in the current sequence.

        Args:
            inputs (Dict[str, Any]): Not used.

        Returns:
            outputs (Dict[str, Any]): A dictionary containing:
            - img (np.ndarray): Image data.
            - filename (str): Name of image file, for compatibility with output
                nodes.
            - mot_metadata (Dict[str, Any]): MOT related metadata such as:
                - frame_idx (int): Index of video frame, for recording tracking
                    results.
                - frame_rate (float): Frame rate of current video sequence, for
                    deciding buffer size in JDE.
                - reset_model (bool): Flag to signal whether the tracking model
                    should be recreated, so clear past tracks when processing a
                    new video sequence.
                - seq_dir: (Path): Path of video sequence subdirectory, used
                    for storing results by video sequence.
            - pipeline_end (bool): Flag to signal when all video sequences have
                been processed and to start evaluating results.
        """
        reset_model = False
        try:
            img_path = next(self.img_loader)
        except StopIteration:  # Ran out of images in current sequence
            try:
                self.img_loader = iter(ImageLoader(next(self.seq_loader)))
                reset_model = True
                img_path = next(self.img_loader)
            except StopIteration:  # Ran out of video sequences in dataset
                return {
                    "img": None,
                    "filename": None,
                    "mot_metadata": defaultdict(lambda: None),
                    "pipeline_end": True,
                }

        img = cv2.imread(str(img_path))

        return {
            "img": img,
            "filename": img_path.name,
            "mot_metadata": {
                "frame_idx": int(img_path.stem),
                "frame_rate": self.seq_loader.current_frame_rate,
                "frame_size": img.shape[:2],
                "reset_model": reset_model,
                "seq_dir": img_path.parents[1],
            },
            "pipeline_end": False,
        }
