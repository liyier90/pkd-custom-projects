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
Core function to perform RLE mask encoding
"""

import numpy as np
import pycocotools.mask as mask_util


def encode_mask_results(mask_results):
    """Encode bitmap mask to RLE code.
    Args:
        mask_results (list): A list of shape (num_of_instances, height, width). bitmap mask results.
    Returns:
        list: RLE encoded mask.
    """
    encoded_mask_results = [
        mask_util.encode(np.array(mask[:, :, np.newaxis], order="F", dtype="uint8"))[0]
        for mask in mask_results
    ]
    for mask in mask_results:
        encoded_mask_results.append(
            mask_util.encode(
                np.array(mask[:, :, np.newaxis], order="F", dtype="uint8")
            )[0]
        )  # encoded with RLE

    return encoded_mask_results
