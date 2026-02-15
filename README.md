# Your Gym Buddy 🏋️‍♂️

Your Gym Buddy is an AI-powered fitness instructor that uses computer vision to analyze your posture in real-time and provides encouraging, actionable feedback via an LLM (Gemini 1.5 Flash).

## 🚀 Features
- **Real-time Pose Classification**: Detects Squats, Tree Pose, and Downward Dog using MediaPipe.
- **Form Analysis**: Calculates joint angles (e.g., knee and back angles for squats) to detect "Poor Form".
- **AI Coaching**: Integrated with Gemini API to provide professional, concise fitness corrections (under 15 words).
- **Customizable**: Pipeline for extracting landmarks from your own datasets and training custom models.

## 📁 Project Structure
```text
your_gym_buddy/
├── data/               # Extracted landmark CSVs
├── models/             # Trained .pkl models and encoders
├── scripts/
│   ├── extract_landmarks.py  # MediaPipe landmark extraction
│   ├── train_classifier.py   # Model training script
│   └── feedback_agent.py     # Gemini LLM feedback logic
├── posture_analyzer.py # Main real-time application
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
Set your Gemini API key and launch the analyzer:
```bash
export GEMINI_API_KEY='your_gemini_api_key_here'
python3 posture_analyzer.py
```

## 🤖 AI Coaching System Prompt
The Feedback Agent is programmed with the following persona:
> "You are a professional yoga and fitness instructor. Your goal is to provide actionable, encouraging feedback based on the user's current posture state. Keep corrections extremely concise (under 15 words). Focus on immediate improvement and positive reinforcement."

## 📜 License
MIT
