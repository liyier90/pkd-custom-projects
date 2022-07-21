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

"""
Load images into the pipeline for model evaluation.
"""

import logging
from pathlib import Path
from queue import Queue
from typing import Any, Dict, List

from peekingduck.pipeline.nodes.input.utils.read import VideoNoThread
from peekingduck.pipeline.nodes.node import AbstractNode
from pycocotools.coco import COCO


class Node(AbstractNode):  # pylint: disable=too-many-instance-attributes
    """Use images from COCO validation dataset for model evaluation

    Inputs:
        |none|

    Outputs:
        |img|

        |pipeline_end|

        |filename|

        |saved_video_fps|

        cat_ids (List[int]): List of category IDs corresponding to the object
            detection classes.

        img_id (int): Image ID of the COCO dataset image.

        img_size (Tuple[int, int]): Size (width, height) of the image.

    Configs:
        input_dir (:obj:`str`): **default = "coco/val2017"**. |br|
            The directory to look for images for evaluation.
        evaluation_class (:obj:`list`): **default = ["all"]**. |br|
            Extract images containing the specified classes for evaluation.
        instances_dir (:obj:`str`):
            **default = "coco/annotations/instances_val2017.json"**. |br|
            The path to look for metadata of the images.
    """

    allowed_extensions = ["jpg", "jpeg", "png"]

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        assert config is not None

        self.file_end = False
        self.frame_counter = -1
        self.progress_ckpt = 10
        self.image_metadata: Dict[str, Any] = {}
        self.cat_ids: List[int] = []
        self._get_files(
            Path(self.input_dir),
            Path(config["instances_dir"]),
            config["evaluation_class"],
        )
        self._get_next_input()

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Load images into the pipeline for model evaluation.

        Args:
             inputs (Dict[str, Any]): Empty dictionary.

        Returns:
            (Dict[str, Any]): Dictionary of outputs with key "img",
            "pipeline_end", "filename", "saved_video_fps", "cat_ids", "img_id",
            and "img_size"
        """
        outputs = self._run_single_file()

        approx_processed = round((self.frame_counter / len(self.image_metadata)) * 100)
        self.frame_counter += 1

        if self.file_end:
            if approx_processed > self.progress_ckpt:
                self.logger.info(f"Approximately processed: {self.progress_ckpt}% ...")
                self.progress_ckpt += 10
            pct_complete = round(100 * self.frame_counter / self.videocap.frame_count)
            self.logger.debug(f"#frames={self.frame_counter}, done={pct_complete}%")
            self._get_next_input()
            outputs = self._run_single_file()

        return outputs

    def _run_single_file(self) -> Dict[str, Any]:
        success, img = self.videocap.read_frame()

        self.file_end = True
        outputs = {
            "img": None,
            "pipeline_end": True,
            "filename": self._file_name,
            "saved_video_fps": self._fps,
            "img_id": self.image_metadata[self._file_name]["id"],
            "img_size": self.image_metadata[self._file_name]["image_size"],
            "cat_ids": self.cat_ids,
        }
        if success:
            self.file_end = False
            outputs["img"] = img
            outputs["pipeline_end"] = False

        return outputs

    def _get_files(self, path: Path, metadata_dir: Path, eval_class: List[str]) -> None:
        coco_instance = COCO(metadata_dir)

        if eval_class[0] == "all":
            self.logger.info("Using images from all the categories for evaluation.")
            img_ids = sorted(coco_instance.getImgIds())
        else:
            self.logger.info(f"Using images from: {eval_class}")
            self.cat_ids = coco_instance.getCatIds(catNms=eval_class)
            img_ids = coco_instance.getImgIds(catIds=self.cat_ids)

        self._filepaths = Queue(maxsize=0)
        for img_id in img_ids:
            img = coco_instance.loadImgs(img_id)[0]
            image_path = Path(Path.cwd(), path, img["file_name"])
            self._filepaths.put(image_path)

            self.image_metadata[img["file_name"]] = {
                "id": img_id,
                "image_size": (img["width"], img["height"]),
            }

    def _get_next_input(self) -> None:
        if not self._filepaths.empty():
            file_path = self._filepaths.get_nowait()
            self._file_name = file_path.name

            if self._is_valid_file_type(file_path):
                self.videocap = VideoNoThread(str(file_path), False)  # type: ignore
                videocap_logger = logging.getLogger("VideoNoThread")
                videocap_logger.setLevel(logging.WARNING)
                self._fps = self.videocap.fps
            else:
                self.logger.warning(
                    f"Skipping '{file_path}' as it is not an accepted "
                    f"file format {str(self.allowed_extensions)}"
                )
                self._get_next_input()

    def _is_valid_file_type(self, filepath: Path) -> bool:
        return filepath.suffix[1:] in self.allowed_extensions

    def release_resources(self) -> None:
        """Overrides base class method to free video resource."""
        self.videocap.shutdown()
