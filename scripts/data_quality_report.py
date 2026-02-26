import os

import pandas as pd


def report_dataset_quality(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing dataset: {csv_path}")

    df = pd.read_csv(csv_path)
    counts = df["label"].value_counts().sort_values(ascending=False)
    max_count = int(counts.max())

    print("Class counts:")
    print(counts.to_string())

    print("\nClass balance ratios (class_count / max_class_count):")
    ratios = (counts / max_count).round(3)
    print(ratios.to_string())

    weak = ratios[ratios < 0.8]
    if weak.empty:
        print("\nNo severely underrepresented classes found (< 0.8 ratio).")
    else:
        print("\nUnderrepresented classes (< 0.8 ratio):")
        print(weak.to_string())
        print("\nRecommended next data-collection batch:")
        for label, ratio in weak.items():
            target = int(max_count * 0.9)
            needed = max(0, target - int(counts[label]))
            print(f"- {label}: add ~{needed} harder real-world samples (angles, lighting, clothing).")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    csv_path = os.path.join(project_root, "data", "processed", "pose_data.csv")
    report_dataset_quality(csv_path)


if __name__ == "__main__":
    main()
