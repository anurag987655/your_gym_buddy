import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DATASET = ROOT / "data" / "processed" / "pose_data.csv"
EVAL_TABLE = REPORTS / "evaluation_table.csv"
CONFUSION = REPORTS / "confusion_matrix.csv"


PALETTE = {
    "ink": (32, 36, 41),
    "muted": (101, 113, 128),
    "grid": (225, 230, 238),
    "panel": (250, 252, 255),
    "green": (55, 152, 115),
    "blue": (53, 112, 205),
    "amber": (219, 151, 49),
    "red": (207, 83, 73),
    "teal": (73, 170, 163),
}


def text(img, value, xy, scale=0.55, color=None, thickness=1, align="left"):
    color = color or PALETTE["ink"]
    font = cv2.FONT_HERSHEY_SIMPLEX
    if align == "center":
        size = cv2.getTextSize(value, font, scale, thickness)[0]
        xy = (int(xy[0] - size[0] / 2), xy[1])
    cv2.putText(img, value, xy, font, scale, color, thickness, cv2.LINE_AA)


def panel(width, height, title, subtitle=None):
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (width - 1, height - 1), (232, 238, 246), 1)
    text(img, title, (34, 50), scale=0.95, thickness=2)
    if subtitle:
        text(img, subtitle, (34, 82), scale=0.48, color=PALETTE["muted"])
    return img


def save(path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def draw_class_distribution(df):
    counts = df["label"].value_counts().sort_values(ascending=True)
    width, height = 1100, 650
    img = panel(
        width,
        height,
        "Dataset Class Distribution",
        "Processed MediaPipe samples used for model training and evaluation.",
    )
    left, right, top, bottom = 250, 1010, 125, 575
    max_count = int(counts.max())
    bar_h = 34
    gap = 18
    colors = [
        PALETTE["green"],
        PALETTE["blue"],
        PALETTE["teal"],
        PALETTE["amber"],
        (142, 112, 196),
        (88, 151, 190),
        (197, 111, 89),
        (106, 158, 96),
    ]

    for i, (label, count) in enumerate(counts.items()):
        y = top + i * (bar_h + gap)
        x2 = left + int((right - left) * count / max_count)
        text(img, label, (36, y + 23), scale=0.52)
        cv2.rectangle(img, (left, y), (right, y + bar_h), (239, 244, 249), -1)
        cv2.rectangle(img, (left, y), (x2, y + bar_h), colors[i % len(colors)], -1)
        text(img, str(int(count)), (x2 + 12, y + 23), scale=0.5, color=PALETTE["ink"])

    text(img, f"Total processed samples: {len(df):,}", (34, bottom + 35), scale=0.58, thickness=2)
    save(REPORTS / "figure_class_distribution.png", img)


def draw_f1_scores(eval_df):
    df = eval_df.sort_values("f1_score", ascending=True)
    width, height = 1100, 650
    img = panel(
        width,
        height,
        "Per-Class F1 Score",
        "Higher values indicate stronger precision-recall balance for each posture class.",
    )
    left, right, top = 250, 1010, 125
    bar_h = 34
    gap = 18

    for i, row in enumerate(df.itertuples(index=False)):
        label = row.class_
        score = float(row.f1_score)
        y = top + i * (bar_h + gap)
        x2 = left + int((right - left) * score)
        color = PALETTE["green"] if score >= 0.94 else PALETTE["amber"]
        text(img, label, (36, y + 23), scale=0.52)
        cv2.rectangle(img, (left, y), (right, y + bar_h), (239, 244, 249), -1)
        cv2.rectangle(img, (left, y), (x2, y + bar_h), color, -1)
        text(img, f"{score:.3f}", (x2 + 12, y + 23), scale=0.5)

    for tick in [0.8, 0.9, 1.0]:
        x = left + int((right - left) * tick)
        cv2.line(img, (x, top - 10), (x, top + 8 * (bar_h + gap)), PALETTE["grid"], 1)
        text(img, f"{tick:.1f}", (x - 16, 585), scale=0.45, color=PALETTE["muted"])

    macro = eval_df["f1_score"].mean()
    text(img, f"Macro F1: {macro:.3f}", (34, 620), scale=0.58, thickness=2)
    save(REPORTS / "figure_f1_scores.png", img)


def draw_confusion_heatmap(cm_df):
    labels = list(cm_df.index)
    cm = cm_df.to_numpy(dtype=np.float32)
    width, height = 1180, 900
    img = panel(
        width,
        height,
        "Confusion Matrix",
        "Rows are true classes; columns are predicted classes.",
    )
    left, top = 260, 155
    cell = 84
    max_val = max(float(cm.max()), 1.0)

    for i, label in enumerate(labels):
        text(img, label[:18], (28, top + i * cell + 50), scale=0.45)
        text(img, label[:12], (left + i * cell + 8, top - 22), scale=0.4, color=PALETTE["muted"])

    for r in range(len(labels)):
        for c in range(len(labels)):
            v = int(cm[r, c])
            intensity = v / max_val
            base = np.array([244, 249, 253], dtype=np.float32)
            target = np.array(PALETTE["blue"], dtype=np.float32)
            color = tuple(int(x) for x in (base * (1 - intensity) + target * intensity))
            x1 = left + c * cell
            y1 = top + r * cell
            cv2.rectangle(img, (x1, y1), (x1 + cell, y1 + cell), color, -1)
            cv2.rectangle(img, (x1, y1), (x1 + cell, y1 + cell), (215, 223, 233), 1)
            text(img, str(v), (x1 + cell // 2, y1 + 50), scale=0.55, color=PALETTE["ink"], thickness=2, align="center")

    correct = int(np.trace(cm))
    total = int(cm.sum())
    accuracy = correct / total if total else 0.0
    text(img, f"Held-out accuracy from matrix: {accuracy:.3f}", (34, 855), scale=0.58, thickness=2)
    save(REPORTS / "figure_confusion_matrix.png", img)


def draw_feature_pipeline():
    width, height = 1180, 520
    img = panel(
        width,
        height,
        "System Pipeline",
        "The application combines landmark extraction, engineered features, model prediction, and deterministic coaching.",
    )
    steps = [
        ("Camera Frame", "Webcam or image frame"),
        ("MediaPipe Pose", "33 body landmarks"),
        ("Feature Engineering", "Angles, widths, distances"),
        ("Classifier + Rules", "Pose class, issue, phase"),
        ("Coach UI", "Stable text + voice cues"),
    ]
    x0, y0 = 58, 185
    box_w, box_h, gap = 190, 130, 35
    colors = [PALETTE["teal"], PALETTE["blue"], PALETTE["green"], PALETTE["amber"], (126, 96, 183)]
    for i, (title, subtitle) in enumerate(steps):
        x = x0 + i * (box_w + gap)
        cv2.rectangle(img, (x, y0), (x + box_w, y0 + box_h), (246, 249, 253), -1)
        cv2.rectangle(img, (x, y0), (x + box_w, y0 + box_h), colors[i], 3)
        cv2.circle(img, (x + 28, y0 + 34), 17, colors[i], -1)
        text(img, str(i + 1), (x + 22, y0 + 40), scale=0.55, color=(255, 255, 255), thickness=2)
        text(img, title, (x + 18, y0 + 73), scale=0.52, thickness=2)
        text(img, subtitle, (x + 18, y0 + 104), scale=0.42, color=PALETTE["muted"])
        if i < len(steps) - 1:
            ax = x + box_w + 7
            ay = y0 + box_h // 2
            cv2.arrowedLine(img, (ax, ay), (ax + gap - 14, ay), PALETTE["muted"], 2, tipLength=0.28)

    text(img, "Reliability layer: visibility gates + rolling feedback stability before showing or speaking a cue.", (58, 410), scale=0.56, thickness=2)
    save(REPORTS / "figure_system_pipeline.png", img)


def main():
    df = pd.read_csv(DATASET)
    eval_df = pd.read_csv(EVAL_TABLE).rename(columns={"class": "class_"})
    cm_df = pd.read_csv(CONFUSION, index_col=0)
    draw_class_distribution(df)
    draw_f1_scores(eval_df)
    draw_confusion_heatmap(cm_df)
    draw_feature_pipeline()
    print("Generated report figures in reports/")


if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")
    main()
