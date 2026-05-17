# Your Gym Buddy

Your Gym Buddy is a real-time posture coaching system for exercise, yoga, and sitting posture. The current web app uses MediaPipe in the browser to extract body landmarks, sends landmark JSON to a Node/Express API, and routes each pose to Python analysis scripts for deterministic coaching feedback.

The project is designed as a defense-friendly prototype: it includes model training/evaluation scripts, generated report figures, a formal report, voice cues, workout timing, pose hold timing, and squat repetition counting.

## Features

- Browser-side MediaPipe landmark extraction with `@mediapipe/tasks-vision`.
- Landmark-only backend payloads for the web flow; raw webcam frames are not uploaded by the current web app.
- Pose-specific Python analyzers for squat, plank, downdog, tree, goddess, and sitting posture.
- Fallback/general analyzer in `scripts/web_pose_feedback.py`.
- Deterministic-first feedback rules for explainable coaching cues.
- Optional Groq-based cue variation when `GROQ_API_KEY` is configured.
- Rolling feedback stability gate in the frontend to reduce flickering cues.
- Optional browser voice cues for stable corrective feedback.
- Workout timer, pose hold timer, squat rep counter, and reset stats button.
- Dataset processing, model training, evaluation, and report asset generation tools.

## Supported Poses

The web UI currently supports:

- Squat
- Plank
- Downward Dog
- Tree
- Warrior II
- Goddess
- Sitting Posture

Note: `warrior2` currently routes through the general fallback analyzer because there is no dedicated `warrior2_analysis.py` script yet.

## Project Structure

```text
your_gym_buddy/
├── data/
│   ├── raw/                         # Source image datasets
│   └── processed/pose_data.csv      # Engineered feature dataset
├── models/
│   ├── pose_classifier.pkl
│   └── label_encoder.pkl
├── reports/
│   ├── evaluation_table.csv
│   ├── confusion_matrix.csv
│   ├── figure_class_distribution.png
│   ├── figure_confusion_matrix.png
│   ├── figure_f1_scores.png
│   └── figure_system_pipeline.png
├── scripts/
│   ├── pose_features.py             # Shared engineered feature helpers
│   ├── extract_landmarks.py         # Dataset feature extraction
│   ├── train_classifier.py          # Pose classifier training
│   ├── evaluate_model.py            # Evaluation tables and confusion matrix
│   ├── build_report_assets.py       # Report figure generation
│   ├── squat_analysis.py            # Squat specialist analyzer
│   ├── plank_analysis.py            # Plank specialist analyzer
│   ├── downdog_analysis.py          # Downward Dog specialist analyzer
│   ├── tree_analysis.py             # Tree specialist analyzer
│   ├── goddess_analysis.py          # Goddess specialist analyzer
│   ├── sitting_pose.py              # Sitting posture analyzer
│   └── web_pose_feedback.py         # General/fallback landmark analyzer
├── web/
│   ├── client/                      # React + Vite frontend
│   └── server/                      # Express API
├── posture_analyzer.py              # Desktop/OpenCV webcam analyzer
├── report.md                        # Full defense report
├── final_report.md                  # Short final report draft, if present
└── requirements.txt
```

## Setup

Python:

```bash
cd ~/Desktop/your_gym_buddy
source venv/bin/activate
pip install -r requirements.txt
```

Web dependencies:

```bash
cd ~/Desktop/your_gym_buddy/web/server && npm install
cd ~/Desktop/your_gym_buddy/web/client && npm install
```

Optional LLM cue variation:

```bash
cd ~/Desktop/your_gym_buddy
echo "GROQ_API_KEY=your_key_here" > .env
```

If `GROQ_API_KEY` is omitted, the system still works with deterministic feedback.

## Run The Web App

Start backend and frontend together:

```bash
cd ~/Desktop/your_gym_buddy/web
npm run dev
```

Open:

```text
http://localhost:5173/
```

The API runs at:

```text
http://localhost:4000
```

## How The Web Flow Works

1. React opens the webcam.
2. MediaPipe Tasks Vision extracts pose landmarks in the browser.
3. The frontend posts landmark JSON to `/api/analyze`.
4. Express validates the selected pose and landmarks.
5. `web/server/analyzePose.js` routes the request to the matching Python analyzer.
6. The Python script returns structured feedback such as `status`, `issue`, `phase`, `severity`, `feedback`, and `metrics`.
7. The frontend applies stability gating, displays feedback, updates stats, and optionally speaks corrective cues.

## Train And Evaluate

Rebuild the processed dataset and classifier:

```bash
cd ~/Desktop/your_gym_buddy
python3 -m scripts.extract_landmarks
python3 -m scripts.data_quality_report
python3 -m scripts.train_classifier
python3 -m scripts.evaluate_model
```

Generate report figures:

```bash
cd ~/Desktop/your_gym_buddy
venv/bin/python scripts/build_report_assets.py
```

## Current Evaluation Snapshot

From the generated evaluation artifacts:

- Held-out accuracy: `0.929`
- Macro F1: `0.936`
- Weighted F1: `0.929`
- Processed samples: `3,738`
- Held-out test samples: `748`

Useful figures:

```text
reports/figure_class_distribution.png
reports/figure_f1_scores.png
reports/figure_confusion_matrix.png
reports/figure_system_pipeline.png
```

## Command-Line Demo Commands

The current web backend uses landmark JSON. For image-file demos, use specialist scripts that still support `--image`.

Working squat correction example:

```bash
cd ~/Desktop/your_gym_buddy
IMG=data/raw/squat_bad_back/frame_338.jpg; venv/bin/python scripts/squat_analysis.py --image "$IMG" --pose squat
```

Working sitting/image examples require a valid sitting image dataset if present:

```bash
cd ~/Desktop/your_gym_buddy
venv/bin/python scripts/sitting_pose.py --image "path/to/sitting_image.jpg"
```

Safety-gate behavior can also be demonstrated through the web app by moving partly out of frame or hiding important joints.

## Desktop Analyzer

The older desktop webcam analyzer is still available:

```bash
cd ~/Desktop/your_gym_buddy
source venv/bin/activate
python3 posture_analyzer.py
```

It uses OpenCV, MediaPipe, the trained classifier, deterministic cues, and optional Groq cue polishing.

## Reports

Main defense report:

```text
report.md
```

Shorter final report draft, if present:

```text
final_report.md
```

Generated visual assets are stored in:

```text
reports/
```

## Runtime Notes

- The current web app uses browser-side MediaPipe and sends landmarks, not image files.
- Some specialist analyzers are rule-first and do not all use the Random Forest model yet.
- Squat model loading should use the local `models/` directory; avoid hardcoded paths if moving the project.
- Browser voice quality depends on the operating system and browser speech synthesis voices.
- Webcam accuracy depends heavily on camera angle, lighting, full-body visibility, and joint visibility.
- The project is a coaching prototype and should not be presented as medical or injury-prevention software.
