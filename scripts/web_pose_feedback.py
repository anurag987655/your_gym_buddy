import argparse
import json
import sys

import cv2
import mediapipe as mp

try:
    from scripts.pose_features import (
        COMMON_REQUIRED,
        compute_engineered_features,
        has_required_visibility,
        required_indices_for_pose,
    )
except ModuleNotFoundError:
    from pose_features import (  # type: ignore
        COMMON_REQUIRED,
        compute_engineered_features,
        has_required_visibility,
        required_indices_for_pose,
    )

mp_pose = mp.solutions.pose


def _names_from_indices(indices):
    names = [mp_pose.PoseLandmark(i).name for i in indices]
    return [name.replace("LEFT_", "").replace("RIGHT_", "") for name in names]


def _squat_phase(knee_angle):
    if knee_angle < 100:
        return "bottom"
    if knee_angle < 155:
        return "descent"
    return "standing"


def build_pose_hint(selected_pose, metrics):
    pose = (selected_pose or "").lower()
    knee_angle = metrics["knee_angle"]
    hip_angle = metrics["hip_angle"]
    torso_lean = metrics["torso_lean"]
    hip_height_delta = metrics["hip_height_delta"]

    if pose == "squat":
        phase = _squat_phase(knee_angle)
        if phase in {"descent", "bottom"} and torso_lean > 0.23:
            return "needs_adjustment", "Keep chest up and brace your core to avoid folding forward."
        if phase == "bottom" and knee_angle > 112:
            return "needs_adjustment", "Go a little deeper while keeping heels grounded and spine neutral."
        if phase == "standing":
            return "good", "Great setup. Start the squat by sending hips back and down with control."
        return "good", "Solid squat pattern. Keep knees tracking over toes and push through mid-foot."

    if pose == "plank":
        if hip_height_delta > 0.07:
            return "needs_adjustment", "Lift your hips slightly by squeezing glutes and core."
        if hip_height_delta < -0.07:
            return "needs_adjustment", "Lower hips a bit to align shoulders, hips, and ankles."
        return "good", "Good plank line. Keep neck neutral and core tight."

    if pose == "downdog":
        if hip_height_delta > -0.05:
            return "needs_adjustment", "Send hips up and back to lengthen your spine."
        return "good", "Nice down dog shape. Press through palms and lengthen your back."

    if pose == "tree":
        if torso_lean > 0.10:
            return "needs_adjustment", "Stack ribs over hips and fix your gaze for balance."
        return "good", "Good tree pose balance. Keep hips level and breathe steadily."

    if pose in {"warrior2", "goddess"}:
        if knee_angle > 125:
            return "needs_adjustment", "Bend your knee more and keep it tracking over toes."
        if hip_angle < 145:
            return "needs_adjustment", "Lift your torso taller and keep your chest open."
        return "good", "Strong stance. Stay grounded through both feet."

    return "good", "Good posture."


def analyze_image(image_path, selected_pose, visibility_threshold=0.6):
    image = cv2.imread(image_path)
    if image is None:
        return {
            "success": False,
            "error": "Unable to read image"
        }

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        results = pose.process(image_rgb)

    if not results.pose_landmarks:
        return {
            "success": False,
            "error": "No person detected",
            "feedback": "Move your whole body into frame and try again."
        }

    landmarks = results.pose_landmarks.landmark
    common_ok, common_missing, common_conf = has_required_visibility(
        landmarks,
        COMMON_REQUIRED,
        visibility_threshold,
    )

    if not common_ok:
        return {
            "success": False,
            "error": "Low landmark confidence",
            "feedback": "I need a clearer full-body view before giving feedback.",
            "visibility": {
                "score": round(common_conf, 3),
                "missing": _names_from_indices(common_missing)[:6],
            },
        }

    pose_required = required_indices_for_pose(selected_pose)
    pose_ok, pose_missing, pose_conf = has_required_visibility(
        landmarks,
        pose_required,
        visibility_threshold,
    )

    if not pose_ok:
        return {
            "success": False,
            "error": "Selected pose landmarks not clear",
            "feedback": "Adjust camera angle so key joints for this pose are visible.",
            "visibility": {
                "score": round(pose_conf, 3),
                "missing": _names_from_indices(pose_missing)[:6],
            },
        }

    _, metrics = compute_engineered_features(landmarks)
    status, cue = build_pose_hint(selected_pose, metrics)

    return {
        "success": True,
        "pose": selected_pose,
        "status": status,
        "feedback": cue,
        "metrics": {
            "knee_angle": round(metrics["knee_angle"], 1),
            "hip_angle": round(metrics["hip_angle"], 1),
            "torso_lean": round(metrics["torso_lean"], 3),
            "hip_height_delta": round(metrics["hip_height_delta"], 3),
            "side": metrics["side"],
        },
        "visibility": {
            "score": round(pose_conf, 3),
            "missing": [],
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--pose", required=True)
    args = parser.parse_args()

    result = analyze_image(args.image, args.pose)
    sys.stdout.write(json.dumps(result))


if __name__ == "__main__":
    main()
