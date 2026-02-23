# Your Gym Buddy 🏋️‍♂️

Your Gym Buddy is an AI-powered fitness instructor that uses computer vision to analyze your posture in real-time and provides encouraging, actionable feedback via an LLM (Groq Llama 3.3).

## 🚀 Features
- **Real-time Pose Classification**: Detects multiple poses using MediaPipe + a trained classifier.
- **Pose-Specific Form Analysis**: Uses side-aware joint metrics (knee/hip/torso/hip height signals) for more accurate cues.
- **Non-Blocking AI Coaching**: Groq feedback runs in a background worker so webcam FPS stays smooth.
- **Customizable**: Pipeline for extracting landmarks from your own datasets and training custom models.
- **Performance-Oriented Runtime**: Uses MediaPipe lite model, frame downscaling, and frame skipping for efficiency.

## 📁 Project Structure
```text
your_gym_buddy/
├── data/               # Extracted landmark CSVs
├── models/             # Trained .pkl models and encoders
├── scripts/
│   ├── extract_landmarks.py  # MediaPipe landmark extraction
│   ├── train_classifier.py   # Model training script
│   └── feedback_agent.py     # Groq LLM feedback logic
├── posture_analyzer.py # Main real-time application
├── .env                # GROQ_API_KEY=...
├── requirements.txt    # Project dependencies
└── venv/               # Python virtual environment
```

## 🛠️ Setup Instructions

### 1. Environment Setup
```bash
# Navigate to project
cd ~/Desktop/your_gym_buddy

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare the Model
Ensure your datasets are located at the paths defined in `scripts/extract_landmarks.py`, then run:
```bash
python3 scripts/extract_landmarks.py
python3 scripts/train_classifier.py
```

### 3. Run the Application
Create a `.env` file in project root and launch:
```bash
echo "GROQ_API_KEY=your_groq_api_key_here" > .env
python3 posture_analyzer.py
```

`posture_analyzer.py` and `scripts/feedback_agent.py` both auto-load `.env`.

## ⚙️ Performance Tuning
You can tune real-time performance directly in `posture_analyzer.py`:
- `self.frame_width`: lower value reduces CPU usage.
- `self.process_every_n_frames`: higher value increases FPS but may reduce responsiveness.
- `self.cooldown`: controls how frequently coaching messages appear.

## 🧪 Notes on Runtime Logs
On first run, MediaPipe may download `pose_landmark_lite.tflite`.
Logs like EGL/OpenGL/XNNPACK and TensorFlow Lite feedback-manager warnings are usually informational and not fatal.

## 🔐 Secrets and GitHub
- `.env` is ignored by `.gitignore`, so it should not be pushed by default.
- Before pushing, verify with:
```bash
git ls-files .env
```
This should print nothing.
- Also check history to ensure the key was never committed previously.

## 🤖 AI Coaching System Prompt
The Feedback Agent is programmed with the following persona:
> "You are a professional yoga and fitness instructor. Return one concise coaching cue (max 15 words). Use only the pose label and metrics in the state packet. Prioritize safety-critical corrections first."

## 📜 License
MIT
