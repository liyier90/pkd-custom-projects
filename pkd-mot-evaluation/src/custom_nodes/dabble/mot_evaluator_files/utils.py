"""Utility functions for MOT Challenge Evaluator."""

import numpy as np


def xyxyn2tlwh(inputs: np.ndarray, height: int, width: int) -> np.ndarray:
    """Converts from normalised [x1, y1, x2, y2] to [t, l, w, h] format.

    (x1, y1) and (x2, y2) are coordinates of top left and bottom right
    respectively. (t, l) is the coordinates of the top left corner, w is the
    width, and h is the height.
    """
    outputs = np.empty_like(inputs)
    outputs[:, 0] = inputs[:, 0] * width
    outputs[:, 1] = inputs[:, 1] * height
    outputs[:, 2] = (inputs[:, 2] - inputs[:, 0]) * width
    outputs[:, 3] = (inputs[:, 3] - inputs[:, 1]) * height
    return outputs
