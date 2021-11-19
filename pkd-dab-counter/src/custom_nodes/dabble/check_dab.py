from typing import Any, Dict, Tuple

import numpy as np

from peekingduck.pipeline.nodes.node import AbstractNode
from peekingduck.pipeline.nodes.draw.utils.general import (
    get_image_size,
    project_points_onto_original_image,
)

from .utils.pose import is_accute, is_bent, is_straight


L_SHOULDER = 5
R_SHOULDER = 6
L_ELBOW = 7
R_ELBOW = 8
L_WRIST = 9
R_WRIST = 10
L_HIP = 11
R_HIP = 12

L_ARM_KEYPOINTS = [L_SHOULDER, L_ELBOW, L_WRIST]
R_ARM_KEYPOINTS = [R_SHOULDER, R_ELBOW, R_WRIST]
L_SIDE_KEYPOINTS = [L_HIP, L_SHOULDER, L_ELBOW]
R_SIDE_KEYPOINTS = [R_HIP, R_SHOULDER, R_ELBOW]


class Node(AbstractNode):
    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)

        self.dab_count = 0
        self.is_dabbing = False
        self.toggled_status = False
        self.curr_status = self.negative_tag
        self.last_status = self.negative_tag
        self.frame_count = 0

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        image_size = get_image_size(inputs["img"])
        obj_tags = [""] * len(inputs["keypoints"])
        for i, keypoints in enumerate(inputs["keypoints"]):
            (
                l_arm_kpts,
                r_arm_kpts,
                l_side_kpts,
                r_side_kpts,
            ) = self._get_keypoint_groups(keypoints)

            l_arm_angle = self._get_angle(l_arm_kpts, image_size)
            r_arm_angle = self._get_angle(r_arm_kpts, image_size)
            l_side_angle = self._get_angle(l_side_kpts, image_size)
            r_side_angle = self._get_angle(r_side_kpts, image_size)
            dab_result = self._detect_dab(
                l_arm_angle, r_arm_angle, l_side_angle, r_side_angle
            )
            tmp_status = self.positive_tag if dab_result else self.negative_tag
            self._handle_status(tmp_status)

            obj_tags[i] = {
                "dab_count": str(self.dab_count),
                "curr_status": self.curr_status,
            }

        return {"is_dab": dab_result, "obj_tags": obj_tags}

    def _detect_dab(
        self,
        l_arm_angle: float,
        r_arm_angle: float,
        l_side_angle: float,
        r_side_angle: float,
    ) -> bool:
        return (
            is_accute(l_arm_angle, self.max_accute_angle)
            and is_straight(r_arm_angle, self.min_straight_angle)
            and is_bent(r_side_angle, self.min_bent_angle, self.max_bent_angle)
        ) or (
            is_straight(l_arm_angle, self.min_straight_angle)
            and is_accute(r_arm_angle, self.max_accute_angle)
            and is_bent(l_side_angle, self.min_bent_angle, self.max_bent_angle)
        )

    def _handle_status(self, tmp_status: str) -> None:
        if tmp_status == self.last_status:
            self.frame_count += 1
            if self.frame_count >= self.buffer_frames:
                self.curr_status = self.last_status
        else:
            self.last_status = tmp_status
            self.frame_count = 1

        if self.curr_status == self.positive_tag:
            if not self.is_dabbing:
                self.dab_count += 1
                self.is_dabbing = True
        else:
            self.is_dabbing = False

    @staticmethod
    def _get_keypoint_groups(
        keypoints: np.array,
    ) -> Tuple[np.array, np.array, np.array, np.array]:
        l_arm_keypoints = keypoints[L_ARM_KEYPOINTS, :]
        r_arm_keypoints = keypoints[R_ARM_KEYPOINTS, :]
        l_side_keypoints = keypoints[L_SIDE_KEYPOINTS, :]
        r_side_keypoints = keypoints[R_SIDE_KEYPOINTS, :]

        return (
            l_arm_keypoints,
            r_arm_keypoints,
            l_side_keypoints,
            r_side_keypoints,
        )

    @staticmethod
    def _get_angle(keypoints: np.array, image_size: Tuple[int, int]) -> float:
        projected_points = project_points_onto_original_image(keypoints, image_size)
        ba = projected_points[1] - projected_points[0]
        bc = projected_points[1] - projected_points[2]
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        if np.isclose(norm_ba, 0) or np.isclose(norm_bc, 0):
            angle = 0
        else:
            angle = np.arccos(np.clip(np.dot(ba, bc) / norm_ba / norm_bc, -1, 1))

        return np.degrees(angle)
