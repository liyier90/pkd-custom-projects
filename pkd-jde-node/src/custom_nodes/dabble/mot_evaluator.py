"""
Node template for creating custom nodes.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import motmetrics as mm
import numpy as np
from peekingduck.pipeline.nodes.node import AbstractNode

from custom_nodes.dabble.utils.evaluator import Evaluator
from custom_nodes.model.jdev1.jde_files.utils import xyxyn2tlwh


class Node(AbstractNode):
    """This is a template class of how to write a node for PeekingDuck.

    Args:
        config (:obj:`Dict[str, Any]`): Node configuration.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)

        self.output_dir: Union[Path, str]
        if self.output_dir is None:
            raise ValueError("input_dir cannot be unset")
        self.output_dir = Path(self.output_dir).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.seq_dir: Optional[Path] = None
        self.sequences: List[str] = []
        self.results: List[str] = []
        self.accumulators: List[mm.MOTAccumulator] = []

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Append tracking results per frame, saves tracking results to a
        result file per video sequence, evaluate against ground truth after
        each video sequences, and summarise results after all video sequences
        are processed.

        Args:
            inputs (dict): Dictionary with keys "bboxes", "obj_track_ids",
                "mot_metadata", and "pipeline_end".

        Returns:
            outputs (dict): Dictionary with keys "__".
        """
        metadata = inputs["mot_metadata"]
        self._save_results(metadata["seq_dir"])

        if inputs["pipeline_end"]:
            self.logger.info("Evaluating...")
            self._summarise_results()
        elif inputs["obj_track_ids"]:
            self._append_single_frame_results(
                inputs["bboxes"], metadata, inputs["obj_track_ids"]
            )

        return {}

    def _append_single_frame_results(
        self, bboxes: np.ndarray, metadata: Dict[str, Any], track_ids: List[str]
    ) -> None:
        tlwhs = xyxyn2tlwh(bboxes, *metadata["frame_size"])
        for tlwh, track_id in zip(tlwhs, track_ids):
            if int(track_id) < 0:
                continue
            self.results.append(
                f"{metadata['frame_idx']},{track_id},"
                f"{','.join(np.char.mod('%f', tlwh))},1,-1,-1,-1\n"
            )

    def _save_results(self, seq_dir):
        if self.seq_dir is None:
            self.seq_dir = seq_dir
        if self.seq_dir != seq_dir:
            self.logger.info(f"Saving {self.seq_dir.name} results...")
            result_path = self.output_dir / f"{self.seq_dir.name}.txt"
            with open(result_path, "w") as outfile:
                outfile.writelines(self.results)
            evaluator = Evaluator(self.seq_dir)
            self.accumulators.append(evaluator.eval_file(result_path))
            self.sequences.append(self.seq_dir.name)
            self.seq_dir = seq_dir
            self.results = []

    def _summarise_results(self):
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
