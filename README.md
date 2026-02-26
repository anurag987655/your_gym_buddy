# Your Gym Buddy

Your Gym Buddy is a real-time posture coach using MediaPipe + a classifier, with deterministic coaching cues and optional LLM tone polishing.

## Features
- Rotation-invariant feature engineering: normalized joint angles + relative distances (not raw landmarks only).
- Deterministic-first coaching: rule-based safety/form cues are primary output.
- Optional LLM polish: Groq rewrites deterministic cues for tone only.
- Squat phase state machine: `descent -> bottom -> ascent` plus rep counting.
- Confidence gating: classification and coaching only run when required joints are visible.
- Dataset quality tooling: class-balance report + training-time balancing.

## Project Structure
```text
your_gym_buddy/
├── data/
├── models/
├── scripts/
│   ├── pose_features.py       # Shared engineered feature + confidence logic
│   ├── extract_landmarks.py   # Dataset feature extraction
│   ├── train_classifier.py    # Balanced model training
│   ├── data_quality_report.py # Class imbalance report
│   └── feedback_agent.py      # Deterministic cue polishing (optional LLM)
├── posture_analyzer.py
├── .env
└── requirements.txt
```

## Setup
```bash
cd ~/Desktop/your_gym_buddy
source venv/bin/activate
pip install -r requirements.txt
```

## Prepare Data + Train
```bash
python3 -m scripts.extract_landmarks
python3 -m scripts.data_quality_report
python3 -m scripts.train_classifier
python3 -m scripts.evaluate_model
```

## Run
```bash
# Optional: if omitted, app runs deterministic coaching only
echo "GROQ_API_KEY=your_key_here" > .env
python3 posture_analyzer.py
```

## Runtime Notes
- If key joints are not visible enough, coaching is gated to prevent wrong feedback.
- Squat feedback is phase-specific (descent/bottom/ascent) instead of generic.
- LLM output never replaces rule logic; it only rephrases the deterministic cue.
