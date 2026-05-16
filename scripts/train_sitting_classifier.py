"""
Train a RandomForest classifier on labeled sitting posture images.

Usage:
    python scripts/train_sitting_classifier.py

Reads images from  data/raw/sitting_posture/<class_name>/*.jpeg
Saves model to     models/sitting_posture_clf.pkl
"""
import glob
import os
import sys

import cv2
import joblib
import mediapipe as mp
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# ── Landmark indices ────────────────────────────────────────────────────────
NOSE        = 0
LEFT_EAR    = 7
RIGHT_EAR   = 8
LEFT_SHOULDER  = 11
RIGHT_SHOULDER = 12
LEFT_HIP    = 23
RIGHT_HIP   = 24

# ── Class map ───────────────────────────────────────────────────────────────
# folder name → canonical label
CLASS_MAP = {
    "goodpose":    "goodpose",
    "titled_head": "titled_head",
    "round_head":  "round_head",
    "round_back":  "round_back",
}


def _dist(a, b):
    return float(np.linalg.norm(a - b))


def extract_features(lm):
    """
    Given a 33×4 numpy array of (x, y, z, visibility), compute
    10 scale-invariant geometric features for sitting posture.
    """
    l_ear = lm[LEFT_EAR, :2]
    r_ear = lm[RIGHT_EAR, :2]
    l_sh  = lm[LEFT_SHOULDER, :2]
    r_sh  = lm[RIGHT_SHOULDER, :2]
    l_hip = lm[LEFT_HIP, :2]
    r_hip = lm[RIGHT_HIP, :2]
    nose  = lm[NOSE, :2]

    avg_ear = (l_ear + r_ear) / 2.0
    avg_sh  = (l_sh + r_sh) / 2.0
    avg_hip = (l_hip + r_hip) / 2.0

    ear_width = max(_dist(l_ear, r_ear), 1e-6)
    sh_width  = max(_dist(l_sh, r_sh), 1e-6)

    # All features normalized by shoulder width for scale invariance
    feats = [
        # Head tilt (positive = left ear lower)
        (l_ear[1] - r_ear[1]) / sh_width,
        # Shoulder tilt (positive = left shoulder lower)
        (l_sh[1] - r_sh[1]) / sh_width,
        # Vertical distance from ear to shoulder (tall = high value)
        (avg_sh[1] - avg_ear[1]) / sh_width,
        # Horizontal torso lean (shoulder vs hip)
        (avg_sh[0] - avg_hip[0]) / sh_width,
        # Ear span relative to shoulder span
        ear_width / sh_width,
        # Nose horizontal offset from shoulder midpoint
        (nose[0] - avg_sh[0]) / sh_width,
        # Nose vertical above shoulder midpoint
        (avg_sh[1] - nose[1]) / sh_width,
        # Left ear-to-shoulder vertical gap (individually)
        (l_sh[1] - l_ear[1]) / sh_width,
        # Right ear-to-shoulder vertical gap
        (r_sh[1] - r_ear[1]) / sh_width,
        # Hip levelness
        abs(l_hip[1] - r_hip[1]) / sh_width,
    ]
    return feats


def process_image(image_path, pose):
    image = cv2.imread(image_path)
    if image is None:
        return None
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    if not results.pose_landmarks:
        return None

    lm = np.array(
        [[p.x, p.y, p.z, p.visibility] for p in results.pose_landmarks.landmark],
        dtype=np.float32,
    )

    # Require key upper-body landmarks to be visible
    required = [LEFT_EAR, RIGHT_EAR, LEFT_SHOULDER, RIGHT_SHOULDER, NOSE]
    if any(lm[i, 3] < 0.3 for i in required):
        return None

    return extract_features(lm)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, "data", "raw", "sitting_posture")
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "sitting_posture_clf.pkl")

    X, y = [], []
    mp_pose = mp.solutions.pose

    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        for folder_name, label in CLASS_MAP.items():
            folder = os.path.join(data_dir, folder_name)
            if not os.path.isdir(folder):
                print(f"[WARN] Folder not found: {folder}")
                continue

            files = glob.glob(os.path.join(folder, "*.jpeg")) + \
                    glob.glob(os.path.join(folder, "*.jpg")) + \
                    glob.glob(os.path.join(folder, "*.png"))

            kept = 0
            for fp in files:
                feats = process_image(fp, pose)
                if feats is not None:
                    X.append(feats)
                    y.append(label)
                    kept += 1

            print(f"  {folder_name}: {kept}/{len(files)} usable images")

    if len(X) < 10:
        print("Not enough usable images to train. Exiting.")
        sys.exit(1)

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\n── Classification Report ──────────────────────────────")
    print(classification_report(y_test, y_pred, zero_division=0))

    joblib.dump(clf, model_path)
    print(f"Model saved → {model_path}")


if __name__ == "__main__":
    main()
