def is_accute(angle: float, max_angle: float) -> bool:
    return angle < max_angle


def is_bent(angle: float, min_angle: float, max_angle: float) -> bool:
    return angle >= min_angle and angle < max_angle


def is_straight(angle: float, min_angle: float) -> bool:
    return angle >= min_angle
