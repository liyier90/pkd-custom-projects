"""Loader classes to load either video sequences or images from MOT dataset."""

import configparser
from pathlib import Path


class ImageLoader:
    """Loads images from the 'img1' subdirectory in MOT dataset."""

    def __init__(self, path: Path) -> None:
        self.count: int
        self.files = sorted(list((path / "img1").glob("*.jpg")))
        self.num_files = len(self.files)

    def __iter__(self) -> "ImageLoader":
        self.count = -1
        return self

    def __next__(self) -> Path:
        self.count += 1
        if self.count >= self.num_files:
            raise StopIteration
        return self.files[self.count]


class SequenceLoader:
    """Loads video sequence from MOT dataset."""

    def __init__(self, data_dir: Path) -> None:
        self.count: int
        self.sequences = sorted(list(data_dir.iterdir()))
        self.num_sequences = len(self.sequences)

    def __iter__(self) -> "SequenceLoader":
        self.count = -1
        return self

    def __next__(self) -> Path:
        self.count += 1
        if self.count >= self.num_sequences:
            raise StopIteration
        return self.sequences[self.count]

    @property
    def current_frame_rate(self) -> float:
        """Frame rate of the current video sequence."""
        seq_config = configparser.ConfigParser()
        seq_config.read(self.current_sequence / "seqinfo.ini")
        return float(seq_config["Sequence"]["frameRate"])

    @property
    def current_sequence(self) -> Path:
        """Path to the current sequence directory of images."""
        return self.sequences[self.count]
