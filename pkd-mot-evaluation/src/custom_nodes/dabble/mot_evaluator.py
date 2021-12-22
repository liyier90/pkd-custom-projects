"""Saves tracking result by video sequence and summarise evaluation results."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import motmetrics as mm
import numpy as np
from peekingduck.pipeline.nodes.node import AbstractNode

from custom_nodes.dabble.mot_evaluator_files.evaluator import Evaluator
from custom_nodes.dabble.mot_evaluator_files.utils import xyxyn2tlwh


class Node(AbstractNode):
    """Saves tracking results per video sequence and summarises evaluation
    result at the end.

    Args:
        config (:obj:`Dict[str, Any]`): Node configuration.

    Attributes:
        output_dir (Union[Path, str]): Path to directory for storing per video
            sequence tracking results and evaluation summary.
        seq_dir ()
    """

    # pylint: disable=too-few-public-methods

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)

        try:
            self.output_dir = Path(self.output_dir).expanduser()  # type: ignore
        except TypeError as error:
            raise ValueError("output_dir cannot be unset") from error
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.seq_dir: Optional[Path] = None
        self.is_initialised = False
        self.sequences: List[str] = []
        self.results: List[str] = []
        self.accumulators: List[mm.MOTAccumulator] = []

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Appends tracking results per frame, saves tracking results to a
        result file per video sequence, evaluate against ground truth after
        each video sequences, and summarise results after all video sequences
        are processed.

        Args:
            inputs (Dict[str, Any]): Dictionary with keys "bboxes", "obj_tags",
                "mot_metadata", and "pipeline_end".

        Returns:
            (Dict[str, Any]): Empty dictionary.
        """
        metadata = inputs["mot_metadata"]
        if self.is_initialised:
            self._save_results(metadata["seq_dir"])
        else:
            self.seq_dir = metadata["seq_dir"]
            self.is_initialised = True

        if inputs["pipeline_end"]:
            self.logger.info("Evaluating...")
            self._summarise_results()
        elif inputs["obj_tags"]:
            self._append_single_frame_results(
                inputs["bboxes"], metadata, inputs["obj_tags"]
            )

        return {}

    def _append_single_frame_results(
        self, bboxes: np.ndarray, metadata: Dict[str, Any], track_ids: List[str]
    ) -> None:
        """Appends a new line for each tracking result of the current frame to
        the results file of the current video sequence.

        The format of each line follows the requirements of MOT Challenge:
        frame, id, bb_left, bb_top, bb_width, bb_height, score, x, y, z

        The world coordinates (x, y, z) are ignored for this evaluation and is
        filled as -1. The score of each detection is filled as 1 following the
        format of the original repo.

        Args:
            bboxes (np.ndarray): Bounding boxes for each of the tracked
                detections.
            metadata (Dict[str, Any]): Metadata required for MOT evaluation.
                "frame_size" and "frame_idx" are used.
            track_ids (List[str]): List of IDs for each of the tracked
                detections.
        """
        tlwhs = xyxyn2tlwh(bboxes, *metadata["frame_size"])
        for tlwh, track_id in zip(tlwhs, track_ids):
            if not track_id or int(track_id) < 0:
                continue
            self.results.append(
                f"{metadata['frame_idx']},{track_id},"
                f"{','.join(np.char.mod('%f', tlwh))},1,-1,-1,-1\n"
            )

    def _save_results(self, seq_dir: Optional[Path]) -> None:
        """Saves the tracking results of a video sequence to a text file for
        evaluation later.

        Args:
            seq_dir (Optional[Path]): Path to the directory of video sequence
                for evaluation. This is `None` when the last video sequence has
                been completely processed.
        """
        # Only save when the entire video sequence has been processed
        if self.seq_dir == seq_dir:
            return
        assert isinstance(self.seq_dir, Path)
        self.logger.info(f"Saving {self.seq_dir.name} results...")
        result_path = self.output_dir / f"{self.seq_dir.name}.txt"
        with open(result_path, "w") as outfile:
            outfile.writelines(self.results)
        evaluator = Evaluator(self.seq_dir)
        self.accumulators.append(evaluator.eval_file(result_path))
        self.sequences.append(self.seq_dir.name)
        # Update and reset the relevant variables after writing to file
        self.seq_dir = seq_dir
        self.results = []

    def _summarise_results(self) -> None:
        """Summarises tracking evaluation results and prints in the format
        required by MOT Challenge.
        """
        metrics = mm.metrics.motchallenge_metrics
        metrics_host = mm.metrics.create()
        summary = Evaluator.get_summary(self.accumulators, self.sequences, metrics)
        summary_string = mm.io.render_summary(
            summary,
            formatters=metrics_host.formatters,
            namemap=mm.io.motchallenge_metric_names,
        )
        print(summary_string)
