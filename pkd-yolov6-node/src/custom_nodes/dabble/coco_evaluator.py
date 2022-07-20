# Copyright 2022 AI Singapore
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluates model performance using COCO dataset and API."""

import datetime
import sys
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List

from peekingduck.pipeline.nodes.node import AbstractNode
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .utils.detection_packer import DetectionPacker
from .utils.instance_segmentation_packer import InstanceSegmentationPacker
from .utils.keypoint_packer import KeypointPacker


class Node(AbstractNode):  # pylint: disable=too-few-public-methods
    """Packs results from model into COCO evaluation format and perform
    evaluation using COCO API.

    Inputs:
        |pipeline_end|

        cat_ids (List[int]): List of category IDs corresponding to the object
            detection classes.

        img_id (int): Image ID of the COCO dataset image.

        img_size (Tuple[int, int]): Size (width, height) of the image.

    Outputs:
        |none|

    Optional_inputs:
        |bboxes|

        |bbox_labels|

        |bbox_scores|

        |keypoints|

        |keypoint_scores|

        |keypoint_conns|

    Configs:
        evaluation_task (:obj: `str`): **default = "object_detection"**. |br|
            Evaluate model based on the specified task.
        instances_dir (:obj:`str`):
            **default = "coco/annotations/instances_val2017.json"**. |br|
            The path to look for ground truths for object detection.
        keypoints_dir (:obj:`str`):
            **default = "coco/annotations/person_keypoints_val2017.json"**. |br|
            The path to look for ground truths for pose estimation.
    """

    eval_types = {
        "object_detection": ["bbox"],
        "pose_estimation": ["keypoints"],
        "instance_segmentation": ["bbox", "segm"],
    }

    # pylint: disable=access-member-before-definition
    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        assert config is not None

        if self.output_dir is None:  # type: ignore
            self.output_dir = Path(__file__).resolve().parents[3] / "results"
        else:
            self.output_dir = Path(self.output_dir).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.evaluation_task = config["evaluation_task"]
        self.model_predictions: List[Dict[str, Any]] = []
        self.img_ids: List[int] = []
        if self.evaluation_task == "pose_estimation":
            self.coco = COCO(config["keypoints_dir"])
            self.packer = KeypointPacker()
        elif self.evaluation_task == "object_detection":
            self.coco = COCO(config["instances_dir"])
            self.packer = DetectionPacker()
        elif self.evaluation_task == "instance_segmentation":
            self.coco = COCO(config["instances_dir"])
            self.packer = InstanceSegmentationPacker()
        else:
            raise ValueError(
                "evaluation_task can only be pose_estimation, object_detection "
                "or instance segmentation"
            )

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Packs model predictions based on COCO format and perform evaluate.

        Args:
             inputs (Dict[str, Any]): Dictionary of inputs with key
                "pipeline_end", "img_id", and "img_size"

        Returns:
            (Dict[str, Any]): Empty dictionary.
        """
        self.model_predictions = self.packer.pack(self.model_predictions, inputs)

        if inputs["cat_ids"]:
            self.img_ids.append(inputs["img_id"])

        if inputs["pipeline_end"]:
            self.logger.info("Evaluating model results...")
            self._evaluate_predictions(inputs["cat_ids"])

        return {}

    def _evaluate_predictions(self, cat_ids: List[int]) -> None:
        """Evaluates detections against COCO groundtruths. Also saves results
        based on node configuration.
        """
        coco_eval_list = []
        string_stdout_list = []
        coco_dt = self.coco.loadRes(self.model_predictions)
        for eval_type in self.eval_types[self.evaluation_task]:
            coco_eval_list.append(COCOeval(self.coco, coco_dt, eval_type))

        print(coco_eval_list)
        for coco_eval in coco_eval_list:
            if cat_ids:
                coco_eval.params.catIds = [cat_ids]
                coco_eval.params.imgIds = self.img_ids

            coco_eval.evaluate()
            coco_eval.accumulate()

            # Pipe stdout to StringIO so we can save it to file
            original_stdout = sys.stdout
            string_stdout = StringIO()
            sys.stdout = string_stdout
            coco_eval.summarize()
            sys.stdout = original_stdout
            string_stdout_list.append(string_stdout)
            print(string_stdout.getvalue())

        if self.save_summary:
            current_time = datetime.datetime.now().strftime("%y%m%d-%H-%M-%S")
            summary_path = self.output_dir / f"map_evaluation_{current_time}.txt"
            with open(summary_path, "w") as outfile:
                for string_stdout, eval_type in zip(
                    string_stdout_list, self.eval_types[self.evaluation_task]
                ):
                    outfile.write(eval_type + "\n")
                    outfile.write(string_stdout.getvalue() + "\n")
            self.logger.info(f"Summary saved to {summary_path}.")
