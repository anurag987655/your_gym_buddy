import math
from typing import Dict, List, Sequence, Tuple

import numpy as np


# MediaPipe Pose landmark indices used by feature engineering and quality gates.
LANDMARKS = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}

COMMON_REQUIRED = (
    LANDMARKS["left_shoulder"],
    LANDMARKS["right_shoulder"],
    LANDMARKS["left_hip"],
    LANDMARKS["right_hip"],
    LANDMARKS["left_knee"],
    LANDMARKS["right_knee"],
    LANDMARKS["left_ankle"],
    LANDMARKS["right_ankle"],
)

POSE_REQUIRED = {
    "squat": COMMON_REQUIRED,
    "plank": COMMON_REQUIRED + (LANDMARKS["left_elbow"], LANDMARKS["right_elbow"]),
    "downdog": COMMON_REQUIRED + (LANDMARKS["left_wrist"], LANDMARKS["right_wrist"]),
    "tree": COMMON_REQUIRED,
    "warrior2": COMMON_REQUIRED,
    "goddess": COMMON_REQUIRED,
}

FEATURE_NAMES = [
    "left_knee_angle",
    "right_knee_angle",
    "left_hip_angle",
    "right_hip_angle",
    "left_ankle_angle",
    "right_ankle_angle",
    "torso_to_horizontal",
    "shoulder_width",
    "hip_width",
    "knee_width",
    "ankle_width",
    "left_femur_length",
    "right_femur_length",
    "left_tibia_length",
    "right_tibia_length",
    "left_torso_length",
    "right_torso_length",
    "left_arm_length",
    "right_arm_length",
    "hip_to_ankle_center",
    "knee_to_ankle_ratio_left",
    "knee_to_ankle_ratio_right",
    "knee_forward_left",
    "knee_forward_right",
]


def landmarks_to_array(landmarks: Sequence) -> np.ndarray:
    """Convert MediaPipe landmarks to [33, 4] ndarray of x, y, z, visibility."""
    arr = np.zeros((33, 4), dtype=np.float32)
    for i, lm in enumerate(landmarks):
        arr[i] = (lm.x, lm.y, lm.z, lm.visibility)
    return arr


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-8
    cos_theta = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return math.degrees(math.acos(cos_theta))


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _normalize_xy(arr: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Normalize points by translation + in-plane rotation + scale for camera/size robustness.
    Returns normalized xy points and scale.
    """
    xy = arr[:, :2].astype(np.float32).copy()

    l_hip = xy[LANDMARKS["left_hip"]]
    r_hip = xy[LANDMARKS["right_hip"]]
    hip_center = (l_hip + r_hip) / 2.0
    xy -= hip_center

    l_sh = xy[LANDMARKS["left_shoulder"]]
    r_sh = xy[LANDMARKS["right_shoulder"]]
    shoulder_vec = r_sh - l_sh
    angle = math.atan2(shoulder_vec[1], shoulder_vec[0])
    c = math.cos(-angle)
    s = math.sin(-angle)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    xy = xy @ rot.T

    shoulder_width = _distance(xy[LANDMARKS["left_shoulder"]], xy[LANDMARKS["right_shoulder"]])
    hip_width = _distance(xy[LANDMARKS["left_hip"]], xy[LANDMARKS["right_hip"]])
    scale = max((shoulder_width + hip_width) / 2.0, 1e-3)
    xy /= scale
    return xy, scale


def pick_side(landmarks: Sequence) -> str:
    arr = landmarks_to_array(landmarks)
    left_idxs = [
        LANDMARKS["left_shoulder"],
        LANDMARKS["left_hip"],
        LANDMARKS["left_knee"],
        LANDMARKS["left_ankle"],
    ]
    right_idxs = [
        LANDMARKS["right_shoulder"],
        LANDMARKS["right_hip"],
        LANDMARKS["right_knee"],
        LANDMARKS["right_ankle"],
    ]
    left_vis = float(np.mean(arr[left_idxs, 3]))
    right_vis = float(np.mean(arr[right_idxs, 3]))
    return "left" if left_vis >= right_vis else "right"


def has_required_visibility(
    landmarks: Sequence,
    required_indices: Sequence[int],
    visibility_threshold: float,
) -> Tuple[bool, List[int], float]:
    arr = landmarks_to_array(landmarks)
    vis = arr[list(required_indices), 3]
    missing = [idx for idx in required_indices if arr[idx, 3] < visibility_threshold]
    return len(missing) == 0, missing, float(np.mean(vis))


def required_indices_for_pose(pose_name: str) -> Tuple[int, ...]:
    pose_lower = (pose_name or "").lower()
    for key, indices in POSE_REQUIRED.items():
        if key in pose_lower:
            return indices
    return COMMON_REQUIRED


def compute_engineered_features(landmarks: Sequence) -> Tuple[Dict[str, float], Dict[str, float]]:
    arr = landmarks_to_array(landmarks)
    xy, _ = _normalize_xy(arr)

    pts = {
        "ls": xy[LANDMARKS["left_shoulder"]],
        "rs": xy[LANDMARKS["right_shoulder"]],
        "le": xy[LANDMARKS["left_elbow"]],
        "re": xy[LANDMARKS["right_elbow"]],
        "lw": xy[LANDMARKS["left_wrist"]],
        "rw": xy[LANDMARKS["right_wrist"]],
        "lh": xy[LANDMARKS["left_hip"]],
        "rh": xy[LANDMARKS["right_hip"]],
        "lk": xy[LANDMARKS["left_knee"]],
        "rk": xy[LANDMARKS["right_knee"]],
        "la": xy[LANDMARKS["left_ankle"]],
        "ra": xy[LANDMARKS["right_ankle"]],
    }

    shoulder_mid = (pts["ls"] + pts["rs"]) / 2.0
    hip_mid = (pts["lh"] + pts["rh"]) / 2.0
    ankle_mid = (pts["la"] + pts["ra"]) / 2.0
    torso_vec = shoulder_mid - hip_mid

    features = {
        "left_knee_angle": _angle(pts["lh"], pts["lk"], pts["la"]),
        "right_knee_angle": _angle(pts["rh"], pts["rk"], pts["ra"]),
        "left_hip_angle": _angle(pts["ls"], pts["lh"], pts["lk"]),
        "right_hip_angle": _angle(pts["rs"], pts["rh"], pts["rk"]),
        "left_ankle_angle": _angle(pts["lk"], pts["la"], pts["lh"]),
        "right_ankle_angle": _angle(pts["rk"], pts["ra"], pts["rh"]),
        "torso_to_horizontal": abs(math.degrees(math.atan2(torso_vec[1], torso_vec[0]))),
        "shoulder_width": _distance(pts["ls"], pts["rs"]),
        "hip_width": _distance(pts["lh"], pts["rh"]),
        "knee_width": _distance(pts["lk"], pts["rk"]),
        "ankle_width": _distance(pts["la"], pts["ra"]),
        "left_femur_length": _distance(pts["lh"], pts["lk"]),
        "right_femur_length": _distance(pts["rh"], pts["rk"]),
        "left_tibia_length": _distance(pts["lk"], pts["la"]),
        "right_tibia_length": _distance(pts["rk"], pts["ra"]),
        "left_torso_length": _distance(pts["ls"], pts["lh"]),
        "right_torso_length": _distance(pts["rs"], pts["rh"]),
        "left_arm_length": _distance(pts["ls"], pts["lw"]),
        "right_arm_length": _distance(pts["rs"], pts["rw"]),
        "hip_to_ankle_center": _distance(hip_mid, ankle_mid),
        "knee_to_ankle_ratio_left": _distance(pts["lk"], pts["la"]) / (_distance(pts["lh"], pts["lk"]) + 1e-6),
        "knee_to_ankle_ratio_right": _distance(pts["rk"], pts["ra"]) / (_distance(pts["rh"], pts["rk"]) + 1e-6),
        "knee_forward_left": abs(pts["lk"][0] - pts["la"][0]),
        "knee_forward_right": abs(pts["rk"][0] - pts["ra"][0]),
    }

    side = pick_side(landmarks)
    if side == "left":
        knee_angle = features["left_knee_angle"]
        hip_angle = features["left_hip_angle"]
        torso_lean = abs(pts["ls"][0] - pts["lh"][0])
        hip_height_delta = float(pts["lh"][1] - ((pts["ls"][1] + pts["la"][1]) / 2.0))
        hip_y = float(pts["lh"][1])
        knee_forward = features["knee_forward_left"]
    else:
        knee_angle = features["right_knee_angle"]
        hip_angle = features["right_hip_angle"]
        torso_lean = abs(pts["rs"][0] - pts["rh"][0])
        hip_height_delta = float(pts["rh"][1] - ((pts["rs"][1] + pts["ra"][1]) / 2.0))
        hip_y = float(pts["rh"][1])
        knee_forward = features["knee_forward_right"]

    metrics = {
        "side": side,
        "knee_angle": float(knee_angle),
        "hip_angle": float(hip_angle),
        "torso_lean": float(torso_lean),
        "hip_height_delta": hip_height_delta,
        "hip_y": hip_y,
        "knee_forward": float(knee_forward),
    }

    return features, metrics


def feature_vector(features: Dict[str, float]) -> List[float]:
    return [float(features[name]) for name in FEATURE_NAMES]
