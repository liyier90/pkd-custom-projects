# Copyright 2021 AI Singapore

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Write the output image/video to file
"""

import datetime
import os
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
from peekingduck.pipeline.nodes.node import AbstractNode

# from peekingduck.pipeline.nodes.output.utils.csvlogger import CSVLogger

# from .utils.csv_logger import CSVLogger

# role of this node is to be able to take in multiple frames, stitch them together and output them.
# to do: need to have 'live' kind of data when there is no filename
# to do: it will be good to have the accepted file format as a configuration
# to do: somewhere so that input and output can use this config for media related issues


class Node(AbstractNode):
    """Node that outputs the processed image or video to a file.

    Inputs:
        |img|

        |filename|

        |saved_video_fps|

        |pipeline_end|

    Outputs:
        None

    Configs:
        output_dir (:obj:`str`): **default = 'PeekingDuck/data/output'**

            Output directory for files to be written locally.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)

        self._file_name = None
        self.output_dir = Path(self.output_dir).expanduser()
        self._prepare_directory()
        self._file_path_with_timestamp = None
        self.logger.info("Output directory used is: %s", self.output_dir)

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Writes media information to filepath"""

        # reset and terminate when there are no more data
        if inputs["pipeline_end"]:
            return {}

        if self._file_name is None:
            self._prepare_writer(inputs["filename"])

        if inputs["filename"] != self._file_name:
            self._prepare_writer(inputs["filename"])

        cv2.imwrite(self._file_path_with_timestamp, inputs["img"])

        return {}

    def _prepare_writer(self, filename: str) -> None:
        self._file_path_with_timestamp = self._append_datetime_filename(filename)  # type: ignore

    def _prepare_directory(self) -> None:  # type: ignore
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _append_datetime_filename(self, filename: str) -> str:
        self._file_name = filename  # type: ignore
        current_time = datetime.datetime.now()
        # output as '240621-15-09-13'
        time_str = current_time.strftime("%d%m%y-%H-%M-%S")

        # append timestamp to filename before extension Format: filename_timestamp.extension
        filename_with_timestamp = (
            filename.split(".")[-2] + "_" + time_str + "." + filename.split(".")[-1]
        )
        file_path_with_timestamp = os.path.join(
            self.output_dir,
            filename_with_timestamp,
        )

        return file_path_with_timestamp

    @staticmethod
    def _append_datetime_csv_filepath(filepath: str) -> str:
        """
        Append time stamp to the filename
        """
        current_time = datetime.datetime.now()  # type: ignore
        # output as '240621-15-09-13'
        time_str = current_time.strftime("%d%m%y-%H-%M-%S")

        file_name = filepath.split(".")[-2]
        file_ext = filepath.split(".")[-1]

        # append timestamp to filename before extension
        # Format: filename_timestamp.extension
        filepath_with_timestamp = f"{file_name}_{time_str}.{file_ext}"

        return filepath_with_timestamp
