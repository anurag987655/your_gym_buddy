# Your Gym Buddy: Real-Time Posture Coaching System

## Project Defense Report

**Project title:** Your Gym Buddy  
**Domain:** Computer vision, human pose estimation, fitness coaching, real-time web application  
**Core technologies:** MediaPipe Pose, OpenCV, scikit-learn, React, Express, Python  

---

## 1. Executive Summary

Your Gym Buddy is a real-time posture coaching system designed to help users receive immediate feedback while performing exercises and yoga poses. The system uses MediaPipe Pose to extract human body landmarks, converts those landmarks into engineered biomechanical features, classifies the pose using a trained machine learning model, and generates deterministic coaching feedback through posture-specific rules.

The project includes both a Python-based posture analyzer and a web application. The web version captures webcam frames, sends them to a Node/Express backend, runs the Python pose feedback engine, and presents coaching feedback through text, optional voice cues, workout timers, pose hold tracking, and squat repetition counting.

The trained classifier was evaluated on a held-out test set of **748 samples** and achieved:

- **Accuracy:** 0.929
- **Macro Precision:** 0.941
- **Macro Recall:** 0.933
- **Macro F1 Score:** 0.936
- **Weighted F1 Score:** 0.929

These results show that the system is strong enough for a prototype posture assistant, especially because final coaching feedback is not based only on model prediction. It is also protected by landmark visibility checks, deterministic form rules, and a rolling feedback stability gate.

---

## 2. Problem Statement

Many people perform workouts or yoga poses without a trainer watching their form. This can lead to:

- Poor posture habits
- Reduced training effectiveness
- Higher risk of strain or injury
- Lack of confidence during home workouts

The project addresses this problem by building a lightweight AI form coach that can run from a webcam and give immediate feedback such as:

- Whether the selected pose is visible
- Whether the posture looks stable
- Whether the user needs to adjust depth, torso angle, hip position, or knee bend
- How long the user has held a pose
- How many squat repetitions have been completed

The goal is not to replace a human trainer. The goal is to provide accessible, real-time, low-cost posture guidance.

---

## 3. System Overview

The system follows a deterministic-first design. Machine learning is used to recognize posture patterns, but final safety and coaching cues are produced by explicit rules. This makes the feedback more explainable and easier to defend than a black-box text response.

![System Pipeline](reports/figure_system_pipeline.png)

### Main Components

| Component | Purpose |
|---|---|
| React frontend | Captures webcam frames, displays feedback, tracks session stats |
| Express backend | Receives frames and calls the Python analyzer |
| MediaPipe Pose | Extracts 33 human body landmarks |
| Feature engineering module | Converts landmarks into angles, distances, ratios, and posture metrics |
| Random Forest classifier | Predicts the posture class from engineered features |
| Rule-based feedback engine | Generates form-specific coaching cues |
| Stability gate | Prevents random or flickering feedback from single-frame noise |
| Voice cue system | Speaks accepted corrective cues when enabled |

---

## 4. Dataset

The processed dataset contains **3,738 samples** across eight posture classes:

| Class | Samples |
|---|---:|
| downdog | 209 |
| goddess | 245 |
| plank | 318 |
| squat_bad_back | 598 |
| squat_bad_heel | 845 |
| squat_good | 956 |
| tree | 220 |
| warrior2 | 347 |

![Dataset Class Distribution](reports/figure_class_distribution.png)

The dataset includes both correct and incorrect squat examples. This is important because the system is not only trying to recognize poses; it is trying to distinguish good form from common mistakes such as:

- Rounded or folded torso during squat
- Heel-related squat issues
- Shallow squat depth
- Incorrect hip alignment in static poses

---

## 5. Feature Engineering

Raw landmark coordinates alone are sensitive to camera position, body size, and image placement. To improve robustness, the project computes engineered features from MediaPipe landmarks.

Examples of engineered features include:

- Left and right knee angles
- Left and right hip angles
- Left and right ankle angles
- Torso angle relative to horizontal
- Shoulder, hip, knee, and ankle widths
- Femur and tibia length estimates
- Hip-to-ankle center distance
- Knee-to-ankle ratios
- Knee-forward displacement

The feature pipeline also normalizes landmark positions by body scale and orientation. This helps reduce dependency on the user’s distance from the camera or exact position in the frame.

This design makes the model easier to justify during defense because the inputs correspond to meaningful body mechanics rather than opaque pixel values.

---

## 6. Model Training

The classifier is trained using a Random Forest model. Random Forest was selected because:

- It works well on tabular engineered features
- It is less sensitive to feature scaling than many other models
- It supports nonlinear decision boundaries
- It is interpretable enough for a prototype defense
- It trains quickly on the available dataset

The training process includes class balancing to reduce bias toward overrepresented classes. The trained model is saved as:

- `models/pose_classifier.pkl`
- `models/label_encoder.pkl`

The model artifact also stores the feature schema so the runtime analyzer can detect mismatches between training features and inference features.

---

## 7. Evaluation Results

The model was evaluated on a stratified held-out test split containing **748 samples**.

![Per-Class F1 Score](reports/figure_f1_scores.png)

### Per-Class Performance

| Class | Precision | Recall | F1 Score | Support |
|---|---:|---:|---:|---:|
| downdog | 1.000 | 0.952 | 0.976 | 42 |
| goddess | 0.939 | 0.939 | 0.939 | 49 |
| plank | 0.896 | 0.938 | 0.916 | 64 |
| squat_bad_back | 0.991 | 0.950 | 0.970 | 120 |
| squat_bad_heel | 0.966 | 0.846 | 0.902 | 169 |
| squat_good | 0.865 | 0.974 | 0.916 | 191 |
| tree | 0.952 | 0.909 | 0.930 | 44 |
| warrior2 | 0.917 | 0.957 | 0.936 | 69 |

### Summary Metrics

| Metric | Score |
|---|---:|
| Accuracy | 0.929 |
| Macro Precision | 0.941 |
| Macro Recall | 0.933 |
| Macro F1 | 0.936 |
| Weighted F1 | 0.929 |

![Confusion Matrix](reports/figure_confusion_matrix.png)

The strongest classes are `downdog` and `squat_bad_back`, while `squat_bad_heel` has the lowest recall. This suggests that heel-related squat errors may be harder to identify from 2D landmarks alone, especially when the foot or heel is partially hidden or the camera angle is not ideal.

---

## 8. Real-Time Feedback Mechanism

The web app does not blindly display every frame’s result. The feedback mechanism has been improved to make cues more reliable.

### Feedback Reliability Layers

1. **Visibility gating**  
   The system first checks whether key joints are visible enough. If important joints are missing, feedback is blocked and the user is asked to adjust the camera.

2. **Structured issue output**  
   The Python analyzer returns structured fields:

   ```json
   {
     "status": "needs_adjustment",
     "issue": "squat_torso_fold",
     "phase": "bottom",
     "severity": 0.72,
     "feedback": "Keep chest up and brace your core to avoid folding forward."
   }
   ```

3. **Rolling stability gate**  
   The frontend stores the most recent feedback keys and only accepts a corrective cue after the same issue appears repeatedly. This reduces random feedback caused by one noisy frame.

4. **Voice cue cooldown**  
   Voice feedback is optional and only speaks stable corrective cues. It does not speak every frame, and it does not speak normal “good form” messages.

This design makes the system feel calmer and more trainer-like.

---

## 9. User Features

The web app includes several user-facing features beyond basic feedback:

| Feature | Description |
|---|---|
| Target pose selection | User chooses squat, plank, downdog, tree, warrior2, or goddess |
| Real-time coach mode | Continuously analyzes webcam frames |
| One-frame analysis | Allows manual testing of a single captured frame |
| Voice cues | Optional spoken corrective instructions |
| Workout timer | Tracks active real-time coaching duration |
| Pose hold timer | Tracks how long the selected pose is successfully detected |
| Squat rep counter | Counts squat repetitions from bottom-to-standing phase transitions |
| Reset stats | Resets timers and repetition count |

These features make the application more useful as a training assistant rather than only a classifier demo.

---

## 10. Strengths of the Project

- Uses real computer vision landmarks instead of manual input.
- Uses engineered biomechanical features rather than raw pixels.
- Includes both model-based classification and rule-based coaching.
- Provides real-time webcam interaction through a web interface.
- Includes confidence gating for safer feedback.
- Includes stability filtering to reduce flickering or unreliable cues.
- Supports text and voice feedback.
- Tracks workout time, pose time, and squat repetitions.
- Produces evaluation reports and visual metrics.

---

## 11. Limitations

The system is a strong prototype, but it has known limitations:

- It uses 2D landmarks, so depth-related issues can be difficult to detect.
- Camera angle strongly affects squat and foot-related feedback.
- Heel errors are harder to detect reliably when feet are not clearly visible.
- The web version currently starts a Python process for analysis, which adds latency.
- The squat rep counter depends on clear phase transitions from the analyzer.
- The system should not be used as medical or injury-prevention advice.

These limitations are expected for a webcam-based prototype and create clear directions for future improvement.

---

## 12. Future Improvements

The most valuable next steps are:

1. **Persistent analyzer process**  
   Keep MediaPipe loaded instead of starting Python for every frame. This would reduce latency significantly.

2. **Browser-side pose detection**  
   Run MediaPipe directly in the browser and send only features or metrics to the backend.

3. **Vision model second opinion**  
   Add an optional “Detailed Form Review” button that sends a captured frame to a vision model for a richer explanation. This should not be used in the real-time loop because it would be slower and less deterministic.

4. **User calibration**  
   Record a short baseline for each user so thresholds can adapt to body proportions and mobility.

5. **Better rep counting**  
   Move squat phase tracking into a persistent session state so reps are counted from continuous motion rather than separate image requests.

6. **Expanded dataset**  
   Add more camera angles, lighting conditions, body types, and exercise mistakes.

---

## 13. Defense Talking Points

If asked why this approach is suitable:

- The system uses MediaPipe because it provides reliable real-time human landmarks without needing to train a pose detector from scratch.
- The model uses engineered features because posture is better represented by angles and distances than by raw image pixels.
- Random Forest is appropriate because the features are tabular, nonlinear, and interpretable enough for a prototype.
- Feedback is deterministic-first because safety cues should be controlled and explainable.
- Voice feedback is optional because repeated audio can distract users.
- A vision-language model could be added later, but it is better as a slow detailed review tool than as the core real-time feedback engine.

If asked about reliability:

- The system checks landmark visibility before giving feedback.
- The frontend waits for repeated evidence before accepting corrective cues.
- The report includes held-out test results, not only training accuracy.
- The limitations are clearly identified, especially camera angle and 2D landmark constraints.

---

## 14. Conclusion

Your Gym Buddy demonstrates a complete posture coaching pipeline: webcam capture, pose landmark extraction, engineered feature generation, machine learning classification, deterministic feedback, voice cues, and workout tracking.

The evaluation results show strong prototype-level performance with **0.929 accuracy** and **0.936 macro F1** on the held-out test set. More importantly, the final application is designed with practical reliability features such as visibility gating and rolling feedback stability. This makes it more defensible than a simple one-frame classifier because it addresses the actual problem of giving useful real-time coaching feedback to users.

The project is therefore a strong demonstration of applied computer vision and machine learning in a real-world fitness assistance scenario.

