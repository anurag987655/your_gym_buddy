import os
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

try:
    from scripts.pose_features import FEATURE_NAMES
except ModuleNotFoundError:
    from pose_features import FEATURE_NAMES  # type: ignore


def rebalance_dataframe(df, label_col="label", min_ratio_to_max=0.8, random_state=42):
    counts = df[label_col].value_counts()
    max_count = int(counts.max())
    target_min = max(1, int(max_count * min_ratio_to_max))

    balanced_parts = []
    for label, group in df.groupby(label_col):
        if len(group) < target_min:
            balanced_group = resample(
                group,
                replace=True,
                n_samples=target_min,
                random_state=random_state,
            )
            balanced_parts.append(balanced_group)
        else:
            balanced_parts.append(group)

    out = pd.concat(balanced_parts, ignore_index=True)
    return out.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def train_model():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    csv_path = os.path.join(project_root, "data", "processed", "pose_data.csv")

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run extract_landmarks.py first.")
        return

    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print("Class distribution before balancing:")
    print(df["label"].value_counts().to_string())

    X = df.drop("label", axis=1)
    y = df["label"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded,
    )

    train_df = X_train.copy()
    train_df["label"] = le.inverse_transform(y_train)
    train_df = rebalance_dataframe(train_df, label_col="label", min_ratio_to_max=0.8)
    print("\nTrain distribution after balancing:")
    print(train_df["label"].value_counts().to_string())

    X_train = train_df.drop("label", axis=1)
    y_train = le.transform(train_df["label"])

    print("\nTraining Random Forest model...")
    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    model_dir = os.path.join(project_root, "models")
    os.makedirs(model_dir, exist_ok=True)

    with open(os.path.join(model_dir, "pose_classifier.pkl"), "wb") as f:
        pickle.dump(
            {
                "model": clf,
                "feature_names": FEATURE_NAMES,
                "model_format_version": 2,
            },
            f,
        )

    with open(os.path.join(model_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)

    print(f"Model and Label Encoder saved to {model_dir}")


if __name__ == "__main__":
    train_model()
