import os
import pickle

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def _load_model(model_path):
    with open(model_path, "rb") as f:
        blob = pickle.load(f)
    if isinstance(blob, dict):
        return blob["model"]
    return blob


def _draw_confusion_matrix_image(cm, labels, out_path):
    n = len(labels)
    cell = 110
    margin_left = 260
    margin_top = 130
    width = margin_left + (n * cell) + 40
    height = margin_top + (n * cell) + 80

    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    max_val = max(int(cm.max()), 1)

    cv2.putText(canvas, "Confusion Matrix", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(canvas, "True label (rows) vs Predicted label (columns)", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 1, cv2.LINE_AA)

    for i, label in enumerate(labels):
        cv2.putText(
            canvas,
            label[:16],
            (margin_left + i * cell + 4, margin_top - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            label[:22],
            (15, margin_top + i * cell + 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )

    for r in range(n):
        for c in range(n):
            v = int(cm[r, c])
            t = v / max_val
            color = (255, int(255 - 180 * t), int(255 - 220 * t))
            x1 = margin_left + c * cell
            y1 = margin_top + r * cell
            x2 = x1 + cell
            y2 = y1 + cell
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (180, 180, 180), 1)
            cv2.putText(
                canvas,
                str(v),
                (x1 + 32, y1 + 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (20, 20, 20),
                2,
                cv2.LINE_AA,
            )

    cv2.imwrite(out_path, canvas)


def _to_markdown_table(df):
    headers = ["class"] + list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for idx, row in df.iterrows():
        values = [str(idx)] + [str(row[c]) for c in df.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def evaluate():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    csv_path = os.path.join(project_root, "data", "processed", "pose_data.csv")
    model_path = os.path.join(project_root, "models", "pose_classifier.pkl")
    le_path = os.path.join(project_root, "models", "label_encoder.pkl")
    reports_dir = os.path.join(project_root, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    if not (os.path.exists(csv_path) and os.path.exists(model_path) and os.path.exists(le_path)):
        raise SystemExit("Missing dataset/model artifacts. Run extraction and training first.")

    df = pd.read_csv(csv_path)
    X = df.drop("label", axis=1)
    y = df["label"]

    with open(le_path, "rb") as f:
        le = pickle.load(f)
    y_encoded = le.transform(y)

    _, X_test, _, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded,
    )

    clf = _load_model(model_path)
    y_pred = clf.predict(X_test)

    labels = list(le.classes_)
    report = classification_report(y_test, y_pred, target_names=labels, output_dict=True, zero_division=0)
    per_class = pd.DataFrame(report).T.loc[labels, ["precision", "recall", "f1-score", "support"]]
    per_class = per_class.rename(columns={"f1-score": "f1_score"})
    per_class["support"] = per_class["support"].astype(int)
    per_class_rounded = per_class.copy()
    for col in ["precision", "recall", "f1_score"]:
        per_class_rounded[col] = per_class_rounded[col].round(3)

    table_path = os.path.join(reports_dir, "evaluation_table.csv")
    markdown_path = os.path.join(reports_dir, "evaluation_table.md")
    per_class_rounded.to_csv(table_path, index_label="class")
    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write(_to_markdown_table(per_class_rounded))

    cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(labels)))
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_csv_path = os.path.join(reports_dir, "confusion_matrix.csv")
    cm_png_path = os.path.join(reports_dir, "confusion_matrix.png")
    cm_df.to_csv(cm_csv_path, index_label="true_label")
    _draw_confusion_matrix_image(cm, labels, cm_png_path)

    print(f"Saved per-class table CSV: {table_path}")
    print(f"Saved per-class table MD:  {markdown_path}")
    print(f"Saved confusion CSV:       {cm_csv_path}")
    print(f"Saved confusion image:     {cm_png_path}")
    print("\nPer-class summary:")
    print(per_class_rounded.to_string())


if __name__ == "__main__":
    evaluate()
