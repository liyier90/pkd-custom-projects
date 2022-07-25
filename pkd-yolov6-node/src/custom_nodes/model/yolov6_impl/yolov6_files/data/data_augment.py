"""Image processing functions.
Reference:
https://github.com/ultralytics/yolov5/blob/master/utils/dataloaders.py
"""

from typing import Tuple

import cv2
import numpy as np


def letterbox(  # pylint: disable=too-many-arguments
    image: np.ndarray,
    new_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
    auto: bool = True,
    scaleup: bool = True,
    stride: int = 32,
) -> np.ndarray:
    """Resizes a rectangular image to a padded rectangular image.

    Args:
        image (np.ndarray): Image frame.
        new_shape (Tuple[int, int]): Height and width of padded image.
        color (Tuple[float, float, float]): Color used for padding around
            the image. (114, 114, 114) is chosen as it is used by the
            original project during model training.
        auto (bool): When True, add padding such that the final shape will be
            divisible by ``stride``.
        scaleup (bool): Flag to determine if we allow scaling up the image.
        stride (int): Stride size of the model.

    Returns:
        (np.ndarray): Padded rectangular image.
    """
    # Resize and pad image while meeting stride-multiple constraints
    shape = image.shape[:2]  # current shape [height, width]

    # Scale ratio (new / old)
    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        ratio = min(ratio, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * ratio)), int(round(shape[0] * ratio))
    # wh padding
    width_padding, height_padding = (
        float(new_shape[1] - new_unpad[0]),
        float(new_shape[0] - new_unpad[1]),
    )

    if auto:  # minimum rectangle
        width_padding, height_padding = (
            np.mod(width_padding, stride),
            np.mod(height_padding, stride),
        )

    width_padding /= 2  # divide padding into 2 sides
    height_padding /= 2

    if shape[::-1] != new_unpad:  # resize
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(height_padding - 0.1)), int(round(height_padding + 0.1))
    left, right = int(round(width_padding - 0.1)), int(round(width_padding + 0.1))
    # add border
    image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return image
