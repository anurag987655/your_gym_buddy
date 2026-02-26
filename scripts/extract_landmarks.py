import glob
import os

import cv2
import mediapipe as mp
import pandas as pd

try:
    from scripts.pose_features import (
        COMMON_REQUIRED,
        FEATURE_NAMES,
        compute_engineered_features,
        feature_vector,
        has_required_visibility,
    )
except ModuleNotFoundError:
    from pose_features import (  # type: ignore
        COMMON_REQUIRED,
        FEATURE_NAMES,
        compute_engineered_features,
        feature_vector,
        has_required_visibility,
    )

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)


def extract_features_from_image(image_path, visibility_threshold=0.2):
    image = cv2.imread(image_path)
    if image is None:
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        return None

    landmarks = results.pose_landmarks.landmark
    ok, _, _ = has_required_visibility(landmarks, COMMON_REQUIRED, visibility_threshold)
    if not ok:
        return None

    features, _ = compute_engineered_features(landmarks)
    return feature_vector(features)


def process_dataset():
    data = []

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    base_raw_path = os.path.join(project_root, "data", "raw")

    datasets = {
        "squat_good": os.path.join(base_raw_path, "squat_good"),
        "squat_bad_back": os.path.join(base_raw_path, "squat_bad_back"),
        "squat_bad_heel": os.path.join(base_raw_path, "squat_bad_heel"),
        "downdog": os.path.join(base_raw_path, "downdog"),
        "tree": os.path.join(base_raw_path, "tree"),
        "goddess": os.path.join(base_raw_path, "goddess"),
        "plank": os.path.join(base_raw_path, "plank"),
        "warrior2": os.path.join(base_raw_path, "warrior2"),
    }

    for label, folder_path in datasets.items():
        kept = 0
        seen = 0
        print(f"Processing {label}...")
        if not os.path.exists(folder_path):
            print(f"Directory {folder_path} not found. Skipping.")
            continue

        for img_path in glob.glob(os.path.join(folder_path, "*")):
            if img_path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                seen += 1
                features = extract_features_from_image(img_path)
                if features:
                    kept += 1
                    data.append([label] + features)

        print(f"  Kept {kept}/{seen} samples after confidence gating.")

    if not data:
        print("No features extracted. Check image paths and visibility thresholds.")
        return

    df = pd.DataFrame(data, columns=["label"] + FEATURE_NAMES)
    output_dir = os.path.join(project_root, "data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "pose_data.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} samples to {output_path}")


if __name__ == "__main__":
    process_dataset()
