"""Non Maximum Suppression."""

import os
import time
from typing import List

import cv2
import numpy as np
import torch
import torchvision
from peekingduck.pipeline.utils.bbox.transforms import xywh2xyxy

# Settings
torch.set_printoptions(linewidth=320, precision=5, profile="long")
# format short g, %precision=5
np.set_printoptions(linewidth=320, formatter={"float_kind": "{:11.5g}".format})
# prevent OpenCV from multithreading (incompatible with PyTorch DataLoader)
cv2.setNumThreads(0)
os.environ["NUMEXPR_MAX_THREADS"] = str(min(os.cpu_count(), 8))  # NumExpr max threads

MAX_WH = 4096  # maximum box width and height
MAX_NMS = 30000  # maximum number of boxes put into torchvision.ops.nms()
TIME_LIMIT = 10.0  # quit the function when nms cost time exceed the limit time.


def non_max_suppression(  # pylint: disable=too-many-arguments, too-many-locals
    prediction: torch.Tensor,
    score_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    classes: torch.Tensor = None,
    agnostic: bool = False,
    multi_label=False,
    max_det=300,
) -> List[torch.Tensor]:
    """Runs Non-Maximum Suppression (NMS) on inference results.
    This code is borrowed from:
    https://github.com/ultralytics/yolov5/blob/47233e1698b89fc437a4fb9463c815e9171be955/utils/general.py#L775

    Args:
        prediction (torch.Tensor): Prediction tensor with shape
            [N, 5 + num_classes], N is the number of bboxes.
        score_threshold (float):  Confidence threshold.
        iou_threshold (float): IoU threshold.
        classes (Optional[torch.Tensor]): NMS will only keep the classes you
            provide.
        agnostic (bool): When True, we do class-independent NMS, otherwise,
            different class would do NMS respectively.
        multi_label (bool): When True, one box can have multi labels,
            otherwise, one box only have one label.
        max_det (int): Max number of output bboxes.

    Returns:
        (List[torch.Tensor]): A list of detections, each item is one tensor with
        shape (num_boxes, 6), 6 is for [xyxy, conf, cls].
    """
    num_classes = prediction.shape[2] - 5  # number of classes
    pred_candidates = prediction[..., 4] > score_threshold  # candidates

    # Check the parameters.
    assert (
        0 <= score_threshold <= 1
    ), f"score_threshold must be in 0.0 to 1.0, however {score_threshold} is provided."
    assert (
        0 <= iou_threshold <= 1
    ), f"iou_threshold must be in 0.0 to 1.0, however {iou_threshold} is provided."

    # Function settings.
    multi_label &= num_classes > 1  # multiple labels per box

    tik = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for img_idx, pred in enumerate(prediction):  # image index, image inference
        pred = pred[pred_candidates[img_idx]]  # confidence
        # If no box remains, skip the next process.
        if not pred.shape[0]:
            continue
        # confidence multiply the objectness
        pred[:, 5:] *= pred[:, 4:5]  # conf = obj_conf * cls_conf
        # (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(pred[:, :4])

        # Detections matrix's shape is  (n,6), each row represents (xyxy, conf, cls)
        if multi_label:
            box_idx, class_idx = (
                (pred[:, 5:] > score_threshold).nonzero(as_tuple=False).T
            )
            pred = torch.cat(
                (
                    box[box_idx],
                    pred[box_idx, class_idx + 5, None],
                    class_idx[:, None].float(),
                ),
                1,
            )
        else:  # Only keep the class with highest scores.
            conf, class_idx = pred[:, 5:].max(1, keepdim=True)
            pred = torch.cat((box, conf, class_idx.float()), 1)[
                conf.view(-1) > score_threshold
            ]

        # Filter by class, only keep boxes whose category is in classes.
        if classes is not None:
            pred = pred[(pred[:, 5:6] == classes).any(1)]

        # Check shape
        num_box = pred.shape[0]  # number of boxes
        if num_box == 0:  # no boxes kept.
            continue
        if num_box > MAX_NMS:  # excess max boxes' number.
            # sort by confidence
            pred = pred[pred[:, 4].argsort(descending=True)[:MAX_NMS]]

        # Batched NMS
        class_offset = pred[:, 5:6] * (0 if agnostic else MAX_WH)  # classes
        # boxes (offset by class), scores
        boxes, scores = pred[:, :4] + class_offset, pred[:, 4]
        keep_box_idx = torchvision.ops.nms(boxes, scores, iou_threshold)  # NMS
        if keep_box_idx.shape[0] > max_det:  # limit detections
            keep_box_idx = keep_box_idx[:max_det]

        output[img_idx] = pred[keep_box_idx]
        if (time.time() - tik) > TIME_LIMIT:
            print(f"WARNING: NMS cost time exceed the limited {TIME_LIMIT}s.")
            break  # time limit exceeded

    return output
