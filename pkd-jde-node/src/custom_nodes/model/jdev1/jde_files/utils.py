"""Utility functions for JDE model."""

from typing import List, Optional, Tuple

import numpy as np
import torch
from torchvision.ops import nms


def decode_delta(delta: torch.Tensor, anchor_mesh: torch.Tensor) -> torch.Tensor:
    """Converts raw output to (x, y, w, h) format where (x, y) is the center,
    w is the width, and h is the height of the bounding box.

    Args:
        delta (torch.Tensor): Raw output from the YOLOLayer.
        anchor_mesh (torch.Tensor): Tensor containing the grid points and their
            respective anchor offsets.

    Returns:
        (torch.Tensor): Decoded bounding box tensor.
    """
    delta[..., :2] = delta[..., :2] * anchor_mesh[..., 2:] + anchor_mesh[..., :2]
    delta[..., 2:] = torch.exp(delta[..., 2:]) * anchor_mesh[..., 2:]
    return delta


def decode_delta_map(
    delta_map: torch.Tensor, anchors: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Decodes raw bounding box output in to (x, y, w, h) format where
    (x, y) is the center, w is the width, and h is the height.

    Args:
        delta_map (torch.Tensor): A tensor with the shape
            (batch_size, num_anchors, grid_height, grid_width, 4) containing
            raw bounding box predictions.
        anchors (torch.Tensor): A tensor with the shape (num_anchors, 4)
            containing the anchors used for the `YOLOLayer`.
        device (torch.device): The device which a `torch.Tensor` is on or
            will be allocated.

    Returns:
        (torch.Tensor): Tensor containing the decoded bounding boxes.
    """
    batch_size, num_anchors, grid_height, grid_width, _ = delta_map.shape
    anchor_mesh = generate_anchor(grid_height, grid_width, anchors, device)
    # Shape (num_anchors x grid_height x grid_width) x 4
    anchor_mesh = anchor_mesh.permute(0, 2, 3, 1).contiguous()
    anchor_mesh = anchor_mesh.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
    pred_list = decode_delta(delta_map.view(-1, 4), anchor_mesh.view(-1, 4))
    pred_map = pred_list.view(batch_size, num_anchors, grid_height, grid_width, 4)
    return pred_map


def generate_anchor(
    grid_height: int, grid_width: int, anchor_wh: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Generates grid anchors for a single level.

    Args:
        grid_height (int): Height of feature map.
        grid_width (int): Width of feature map.
        anchor_wh (torch.Tensor): Width and height of the anchor boxes.
        device (torch.device): The device which a `torch.Tensor` is on or
            will be allocated.

    Returns:
        (torch.Tensor): Anchors of a feature map in a single level.
    """
    num_anchors = len(anchor_wh)
    y_vec, x_vec = torch.meshgrid(torch.arange(grid_height), torch.arange(grid_width))
    x_vec, y_vec = x_vec.to(device), y_vec.to(device)

    # Shape 2 x grid_height x grid_width
    mesh = torch.stack([x_vec, y_vec], dim=0)
    # Shape num_anchors x 2 x grid_height x grid_width
    mesh = mesh.unsqueeze(0).repeat(num_anchors, 1, 1, 1).float()
    # Shape num_anchors x 2 x grid_height x grid_width
    anchor_offset_mesh = (
        anchor_wh.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, grid_height, grid_width)
    )
    # Shape num_anchors x 4 x grid_height x grid_width
    anchor_mesh = torch.cat([mesh, anchor_offset_mesh], dim=1)
    return anchor_mesh


def non_max_suppression(
    prediction: torch.Tensor, score_threshold: float, nms_threshold: float
) -> List[Optional[torch.Tensor]]:
    """Removes detections with lower object confidence score than
    `score_threshold`. Non-Maximum Suppression to further filter detections.

    Args:
        prediction (torch.Tensor): Predicted bounding boxes.
        score_threshold (float): Threshold for detection confidence score.
        nms_threshold (float): Threshold for Intersection-over-Union values of
            the bounding boxes.

    Returns:
        (List[Optional[torch.Tensor]]): List of detections with shape
            (x1, y1, x2, y2, object_conf, class_score, class_pred). For
            detections which have all bounding boxes filtered by `nms`, the
            element will be `None` instead.
    """
    output: List[Optional[torch.Tensor]] = [None for _ in range(len(prediction))]
    for i, pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        mask = pred[:, 4] > score_threshold
        mask = mask.nonzero().squeeze()
        if not mask.shape:
            mask = mask.unsqueeze(0)

        pred = pred[mask]
        # If none are remaining => process next image
        if pred.shape[0] == 0:
            continue

        # From (center x, center y, width, height) to (x1, y1, x2, y2)
        pred[:, :4] = xywh2xyxy(pred[:, :4])
        # Non-maximum suppression
        nms_indices = nms(pred[:, :4], pred[:, 4], nms_threshold)
        det_max = pred[nms_indices]

        if len(det_max) == 0:
            continue
        # Add max detections to outputs
        output[i] = (
            det_max if output[i] is None else torch.cat((output[i], det_max))  # type: ignore
        )

    return output


def scale_coords(
    img_size: List[int], coords: torch.Tensor, img0_size: Tuple[int, int]
) -> torch.Tensor:
    """Rescales bounding box coordinates (x1, y1, x2, y2) from `img_size` to
    `img0_size`.

    Args:
        img_size (List[int]): Model input size (w x h).
        coords (torch.Tensor): Bounding box coordinates.
        img0_size (Tuple[int, int]): Size of original video frame (h x w).

    Returns:
        (torch.Tensor): Bounding boxes with resized coordinates.
    """
    # gain = old / new
    gain = min(float(img_size[0]) / img0_size[1], float(img_size[1]) / img0_size[0])
    pad_x = (img_size[0] - img0_size[1] * gain) / 2  # width padding
    pad_y = (img_size[1] - img0_size[0] * gain) / 2  # height padding
    coords[:, [0, 2]] -= pad_x
    coords[:, [1, 3]] -= pad_y
    coords[:, :4] /= gain
    coords[:, :4] = torch.clamp(coords[:, :4], min=0)
    return coords


def xywh2xyxy(inputs: torch.Tensor) -> torch.Tensor:
    """Converts from [x, y, w, h] to [x1, y1, x2, y2] format.

    (x, y) are coordinates of center. (x1, y1) and (x2, y2) are coordinates of
    top left and bottom right respectively.
    """
    outputs = torch.empty_like(inputs)
    outputs[:, 0] = inputs[:, 0] - inputs[:, 2] / 2
    outputs[:, 1] = inputs[:, 1] - inputs[:, 3] / 2
    outputs[:, 2] = inputs[:, 0] + inputs[:, 2] / 2
    outputs[:, 3] = inputs[:, 1] + inputs[:, 3] / 2
    return outputs


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


def tlwh2xyxyn(inputs: np.ndarray, height: int, width: int) -> np.ndarray:
    """Converts from [t, l, w, h] to [x1, y1, x2, y2] format.

    (x1, y1) and (x2, y2) are coordinates of top left and bottom right
    respectively. (t, l) is the coordinates of the top left corner, w is the
    width, and h is the height.
    """
    outputs = np.empty_like(inputs)
    outputs[:, 0] = inputs[:, 0] / width
    outputs[:, 1] = inputs[:, 1] / height
    outputs[:, 2] = (inputs[:, 0] + inputs[:, 2]) / width
    outputs[:, 3] = (inputs[:, 1] + inputs[:, 3]) / height
    return outputs
