"""
Node template for creating custom nodes.
"""

# pylint: disable=import-error
from pathlib import Path
from typing import Any, Dict, List, Optional

import motmetrics as mm
import numpy as np
from peekingduck.pipeline.nodes.node import AbstractNode

from custom_nodes.dabble.utils.evaluator import Evaluator
from custom_nodes.model.jdev1.jde_files.utils import xyxyn2tlwh


class Node(AbstractNode):
    """This is a template class of how to write a node for PeekingDuck.

    Args:
        config (:obj:`Dict[str, Any]` | :obj:`None`): Node configuration.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)

        self.output_dir = Path(self.output_dir).expanduser()  # type: ignore
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.seq_dir: Optional[Path] = None
        self.seqs: List[str] = []
        self.results: List[str] = []
        self.accs: List[mm.MOTAccumulator] = []

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
        """This node does ___.

        Args:
            inputs (dict): Dictionary with keys "__", "__".

        Returns:
            outputs (dict): Dictionary with keys "__".
        """
        metadata = inputs["mot_metadata"]
        self._save_results(metadata["seq_dir"])

        if inputs["pipeline_end"]:
            self.logger.info("Evaluating...")
            # get summary
            metrics = mm.metrics.motchallenge_metrics
            metrics_host = mm.metrics.create()
            summary = Evaluator.get_summary(self.accs, self.seqs, metrics)
            strsummary = mm.io.render_summary(
                summary,
                formatters=metrics_host.formatters,
                namemap=mm.io.motchallenge_metric_names,
            )
            print(strsummary)
        elif inputs["obj_track_ids"]:
            tlwhs = xyxyn2tlwh(inputs["bboxes"], *metadata["frame_size"])
            for tlwh, track_id in zip(tlwhs, inputs["obj_track_ids"]):
                if int(track_id) < 0:
                    continue
                self.results.append(
                    f"{metadata['frame_idx']},{track_id},"
                    f"{','.join(np.char.mod('%f', tlwh))},1,-1,-1,-1\n"
                )

        return {}

    def _save_results(self, seq_dir):
        if self.seq_dir is None:
            self.seq_dir = seq_dir
        if self.seq_dir != seq_dir:
            self.logger.info(f"Saving {self.seq_dir.name} results...")
            result_path = self.output_dir / f"{self.seq_dir.name}.txt"
            with open(result_path, "w") as outfile:
                outfile.writelines(self.results)
            evaluator = Evaluator(self.seq_dir)
            self.accs.append(evaluator.eval_file(result_path))
            self.seqs.append(self.seq_dir.name)
            self.seq_dir = seq_dir
            self.results = []
