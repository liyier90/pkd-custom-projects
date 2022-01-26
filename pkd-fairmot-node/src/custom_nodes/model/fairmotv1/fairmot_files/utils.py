"""Utility functions used by FairMOT.

Modifications include:
- Hardcode ctdet_post_process to handle detections of batch size 1 since this
    assumptions is already made when calling the function
"""

from typing import Tuple

import cv2
import numpy as np

# def ctdet_post_process(
#     detections: np.ndarray,
#     center: np.ndarray,
#     scale: float,
#     output_size: Tuple[float, float],
#     num_classes: int,
# ) -> Dict[int, List[List[float]]]:
#     """Post-processes detections and translate/scale it back to the original
#     image.

#     Args:
#         detections (np.ndarray): An array of detections each having the format
#             [x1, y1, x2, y2, score, class] where (x1, y1) is top left and
#             (x2, y2) is bottom right.
#         center (np.ndarray): Coordinate of the center of the original image.
#         scale (float): Scale between original image and input image fed to the
#             model.
#         output_size (Tuple[float, float]): Size of output by the model.
#         num_classes (int): Number of classes. In the case of FairMOT, it's 1.

#     Returns:
#         (Dict[int, List[List[float]]]): A list of dicts containing detections
#         using 1-based classes as its keys.
#     """
#     detections[:, :2] = _transform_coords(detections[:, :2], center, scale, output_size)
#     detections[:, 2:4] = _transform_coords(
#         detections[:, 2:4], center, scale, output_size
#     )
#     classes = detections[:, -1]

#     top_preds = {}
#     for j in range(num_classes):
#         mask = classes == j
#         top_preds[j + 1] = (
#             np.concatenate([detections[mask, :4], detections[mask, 4:5]], axis=1)
#             .astype(np.float32)
#             .tolist()
#         )
#     return top_preds


def gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def letterbox(
    image: np.ndarray,
    height: int,
    width: int,
    colour: Tuple[float, float, float] = (127.5, 127.5, 127.5),
) -> np.ndarray:
    """Resizes a rectangular image to a padded rectangular image.

    Args:
        image (np.ndarray): Image frame.
        height (int): Height of padded image.
        width (int): Width of padded image.
        colour (Tuple[float, float, float]): Colour used for padding around
            the image. (127.5, 127.5, 127.5) is chosen as it is used by the
            original project during model training.

    Returns:
        (np.ndarray): Padded rectangular image.
    """
    shape = image.shape[:2]  # shape = [height, width]
    ratio = min(float(height) / shape[0], float(width) / shape[1])
    # new_shape = [width, height]
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))
    width_padding = (width - new_shape[0]) / 2
    height_padding = (height - new_shape[1]) / 2
    top = round(height_padding - 0.1)
    bottom = round(height_padding + 0.1)
    left = round(width_padding - 0.1)
    right = round(width_padding + 0.1)
    # resized, no border
    image = cv2.resize(image, new_shape, interpolation=cv2.INTER_AREA)
    # padded rectangular
    image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=colour
    )
    return image


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


def transform_coords(
    coords: np.ndarray,
    center: np.ndarray,
    scale: float,
    output_size: Tuple[float, float],
) -> np.ndarray:
    """Translates and scales the coordinate w.r.t. the original image size.

    Args:
        coords (np.ndarray): Coordinates of a bbox corner.
        center (np.ndarray): Coordinates of the center of the original image.
        scale (float): Scale between the input image fed to the model and the
            original image.
        output_size(Tuple[float, float]): Output image size by the model.

    Returns:
        (np.ndarray): Transformed coordinates.
    """
    # target_coords = np.zeros(coords.shape)
    target_coords = np.zeros_like(coords)
    trans = _get_affine_transform(center, scale, 0, output_size, inv=True)
    for i in range(coords.shape[0]):
        target_coords[i, :2] = _affine_transform(coords[i, :2], trans)
    return target_coords


def transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = gather_feat(feat, ind)
    return feat


def _affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.0], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def _get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def _get_affine_transform(
    center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=False
):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = _get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = _get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def _get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result
