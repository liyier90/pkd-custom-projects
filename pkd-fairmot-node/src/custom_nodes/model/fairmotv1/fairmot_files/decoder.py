"""Decodes model output to bbox and indices.

Modifications:
- Refactor mot_decode() to a class instead
- Removed unnecessary creation of lists since batch size 1 is hardcoded
"""
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from custom_nodes.model.fairmotv1.fairmot_files.utils import (
    gather_feat,
    transform_coords,
    transpose_and_gather_feat,
)


class Decoder:
    """Decodes model output to bounding box coordinates and indices following
    the approach adopted by CenterNet.
    """

    def __init__(self, max_per_image: int, down_ratio: int, num_classes: int) -> None:
        self.max_per_image = max_per_image
        self.down_ratio = down_ratio
        self.num_classes = num_classes

    def __call__(
        self,
        heatmap: torch.Tensor,
        size: torch.Tensor,
        offset: torch.Tensor,
        orig_shape: Tuple[int, ...],
        input_shape: torch.Size,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decodes model outputs to bounding box coordinates and indices.

        Args:
            heatmap (torch.Tensor): A heatmap predicting where the object
                center will be.
            size (torch.Tensor): Size of the bounding boxes w.r.t. the object
                centers.
            offset (torch.Tensor): A continuous offset relative to the object
                centers to localise objects more precisely.
            orig_shape (Tuple[int, ...]): Shape of the original image.
            input_shape (torch.Size): Shape of the image fed to the model.

        Returns:
            (Tuple[torch.Tensor, torch.Tensor]): A tuple containing detections
            and their respective indices. Indices are used to filter the Re-ID
            feature tensor.
        """
        k = self.max_per_image
        batch, _, _, _ = heatmap.size()

        heatmap = self._nms(heatmap)

        scores, indices, classes, y_coords, x_coords = self._topk(heatmap)
        offset = transpose_and_gather_feat(offset, indices)
        offset = offset.view(batch, k, 2)
        x_coords = x_coords.view(batch, k, 1) + offset[:, :, 0:1]
        y_coords = y_coords.view(batch, k, 1) + offset[:, :, 1:2]
        size = transpose_and_gather_feat(size, indices)
        size = size.view(batch, k, 4)
        classes = classes.view(batch, k, 1)
        scores = scores.view(batch, k, 1)
        bboxes = torch.cat(
            [
                x_coords - size[..., 0:1],
                y_coords - size[..., 1:2],
                x_coords + size[..., 2:3],
                y_coords + size[..., 3:4],
            ],
            dim=2,
        )
        detections = torch.cat([bboxes, scores, classes], dim=2)
        detections = self._post_process(detections, orig_shape, input_shape)

        return detections, indices

    def _post_process(
        self,
        detections: torch.Tensor,
        orig_shape: Tuple[int, ...],
        input_shape: torch.Size,
    ) -> np.ndarray:
        """Post processes the detections following the approach by CenterNet.
        Translates/scales detections w.r.t. original image shape.

        Args:
            detections (torch.Tensor): Detections with the format
                [x1, y1, x2, y2, score, class] where (x1, y1) is top left and
                (x2, y2) is bottom right.
            orig_shape (Tuple[int, ...]): Shape of the original image.
            input_shape (torch.Size): Shape of the image fed to the model.

        Returns:
            (np.ndarray): Transformed detections w.r.t. the original image
            shape.
        """
        orig_h = float(orig_shape[0])
        orig_w = float(orig_shape[1])
        input_h = float(input_shape[2])
        input_w = float(input_shape[3])

        dets = detections.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        # Detection batch size has been hardcoded to 1 in FairMOT
        dets = _ctdet_post_process(
            dets[0].copy(),
            np.array([orig_w / 2.0, orig_h / 2.0], dtype=np.float32),
            max(input_w / input_h * orig_h, orig_w),
            (input_w // self.down_ratio, input_h // self.down_ratio),
            self.num_classes,
        )
        for j in range(1, self.num_classes + 1):
            dets[j] = np.array(dets[j], dtype=np.float32).reshape(-1, 5)
        return dets

    def _topk(
        self, scores: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Selects top k scores and decodes to get xy coordinates.

        Args:
            scores (torch.Tensor): In the case of FairMOT, this is a heatmap
                predicting where the object center will be.

        Returns:
            (Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
            torch.Tensor]): Tuple containing top k detection scores and their
            respective indices, classes, y-, and x- coordinates.
        """
        k = self.max_per_image
        batch, cat, height, width = scores.size()

        topk_scores, topk_indices = torch.topk(scores.view(batch, cat, -1), k)

        topk_indices = (topk_indices % (height * width)).view(batch, -1, 1)
        topk_y_coords = torch.div(topk_indices, width, rounding_mode="floor")
        topk_x_coords = topk_indices % width

        topk_score, topk_index = torch.topk(topk_scores.view(batch, -1), k)
        topk_classes = torch.div(topk_index, k, rounding_mode="trunc")

        topk_indices = gather_feat(topk_indices, topk_index).view(batch, k)
        topk_y_coords = gather_feat(topk_y_coords, topk_index).view(batch, k)
        topk_x_coords = gather_feat(topk_x_coords, topk_index).view(batch, k)

        return topk_score, topk_indices, topk_classes, topk_y_coords, topk_x_coords

    @staticmethod
    def _nms(heatmap: torch.Tensor, kernel: int = 3) -> torch.Tensor:
        """Uses maxpool to filter the max score and get local peaks.

        Args:
            heatmap (torch.Tensor): A heatmap predicting where the object
                center will be.
            kernel (int): Size of the window to take a max over.

        Returns:
            (torch.Tensor): Heatmap with only local peaks remaining.
        """
        pad = (kernel - 1) // 2

        hmax = F.max_pool2d(heatmap, (kernel, kernel), stride=1, padding=pad)
        keep = (hmax == heatmap).float()
        return heatmap * keep


def _ctdet_post_process(
    detections: np.ndarray,
    center: np.ndarray,
    scale: float,
    output_size: Tuple[float, float],
    num_classes: int,
) -> Dict[int, List[List[float]]]:
    """Post-processes detections and translate/scale it back to the original
    image.

    Args:
        detections (np.ndarray): An array of detections each having the format
            [x1, y1, x2, y2, score, class] where (x1, y1) is top left and
            (x2, y2) is bottom right.
        center (np.ndarray): Coordinate of the center of the original image.
        scale (float): Scale between original image and input image fed to the
            model.
        output_size (Tuple[float, float]): Size of output by the model.
        num_classes (int): Number of classes. In the case of FairMOT, it's 1.

    Returns:
        (Dict[int, List[List[float]]]): A list of dicts containing detections
        using 1-based classes as its keys.
    """
    detections[:, :2] = transform_coords(detections[:, :2], center, scale, output_size)
    detections[:, 2:4] = transform_coords(
        detections[:, 2:4], center, scale, output_size
    )
    classes = detections[:, -1]

    top_preds = {}
    for j in range(num_classes):
        mask = classes == j
        top_preds[j + 1] = (
            np.concatenate([detections[mask, :4], detections[mask, 4:5]], axis=1)
            .astype(np.float32)
            .tolist()
        )
    return top_preds
