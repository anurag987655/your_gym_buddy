import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import time
from scripts.feedback_agent import FeedbackAgent

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose_detector = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Paths
MODEL_PATH = "/home/anurag/Desktop/your_gym_buddy/models/pose_classifier.pkl"
LE_PATH = "/home/anurag/Desktop/your_gym_buddy/models/label_encoder.pkl"

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    return 360 - angle if angle > 180.0 else angle

class PostureAnalyzer:
    def __init__(self, api_key):
        self.clf = None
        self.le = None
        if os.path.exists(MODEL_PATH) and os.path.exists(LE_PATH):
            with open(MODEL_PATH, 'rb') as f: self.clf = pickle.load(f)
            with open(LE_PATH, 'rb') as f: self.le = pickle.load(f)
        
        self.feedback_agent = FeedbackAgent(api_key)
        self.last_feedback_time = 0
        self.cooldown = 6 # seconds

    def run(self):
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose_detector.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                landmarks = []
                for lm in results.pose_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
                
                if self.clf and self.le:
                    pred = self.clf.predict([landmarks])
                    pose_name = self.le.inverse_transform(pred)[0]
                else:
                    pose_name = "Training required"

                # Angle logic
                l = results.pose_landmarks.landmark
                hip = [l[mp_pose.PoseLandmark.LEFT_HIP.value].x, l[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [l[mp_pose.PoseLandmark.LEFT_KNEE.value].x, l[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [l[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, l[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                shoulder = [l[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, l[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

                knee_angle = calculate_angle(hip, knee, ankle)
                back_angle = calculate_angle(shoulder, hip, knee)

                cv2.putText(image, f"Pose: {pose_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Feedback logic
                if (time.time() - self.last_feedback_time) > self.cooldown:
                    if "bad" in pose_name.lower() or (pose_name == "squat_good" and knee_angle > 110):
                        fb = self.feedback_agent.get_feedback({"pose": pose_name, "knee": int(knee_angle)})
                        print(f"Coach: {fb}")
                        self.last_feedback_time = time.time()

            cv2.imshow('Your Gym Buddy', image)
            if cv2.waitKey(10) & 0xFF == ord('q'): break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_API_KEY")
    PostureAnalyzer(API_KEY).run()
