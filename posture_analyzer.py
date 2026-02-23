import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import time
import queue
import threading
from scripts.feedback_agent import FeedbackAgent

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Get the project root directory
project_root = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(project_root, "models", "pose_classifier.pkl")
LE_PATH = os.path.join(project_root, "models", "label_encoder.pkl")
ENV_PATH = os.path.join(project_root, ".env")


KEY_LANDMARKS = (
    mp_pose.PoseLandmark.LEFT_SHOULDER.value,
    mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
    mp_pose.PoseLandmark.LEFT_HIP.value,
    mp_pose.PoseLandmark.RIGHT_HIP.value,
    mp_pose.PoseLandmark.LEFT_KNEE.value,
    mp_pose.PoseLandmark.RIGHT_KNEE.value,
    mp_pose.PoseLandmark.LEFT_ANKLE.value,
    mp_pose.PoseLandmark.RIGHT_ANKLE.value,
)


def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle


def load_env_file(env_path):
    if not os.path.exists(env_path):
        return

    with open(env_path, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            if key.startswith("export "):
                key = key.replace("export ", "", 1).strip()
            if key and key not in os.environ:
                os.environ[key] = value


class PostureAnalyzer:
    def __init__(self, api_key):
        self.pose_detector = mp_pose.Pose(
            model_complexity=0,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )

        self.clf = None
        self.le = None
        if os.path.exists(MODEL_PATH) and os.path.exists(LE_PATH):
            print(f"Loading model from {MODEL_PATH}...")
            with open(MODEL_PATH, "rb") as f:
                self.clf = pickle.load(f)
            with open(LE_PATH, "rb") as f:
                self.le = pickle.load(f)
        else:
            print("Model files not found. Please train the model first.")

        self.feedback_agent = FeedbackAgent(api_key)
        self.last_feedback_time = 0
        self.cooldown = 6  # seconds
        self.visibility_threshold = 0.6
        self.frame_width = 960
        self.process_every_n_frames = 2

        self.last_results = None
        self.feedback_pending = False
        self.last_feedback_signature = None
        self.latest_feedback = None
        self.feedback_version = 0
        self.printed_feedback_version = 0

        self.feedback_queue = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.feedback_thread = threading.Thread(target=self._feedback_worker, daemon=True)
        self.feedback_thread.start()

    def _feedback_worker(self):
        while not self.stop_event.is_set():
            try:
                packet = self.feedback_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            feedback_text = self.feedback_agent.get_feedback(packet)
            self.latest_feedback = feedback_text
            self.feedback_version += 1
            self.feedback_pending = False
            self.feedback_queue.task_done()

    def _point(self, landmarks, idx):
        return [landmarks[idx].x, landmarks[idx].y]

    def _pick_side(self, landmarks):
        left_idxs = (
            mp_pose.PoseLandmark.LEFT_SHOULDER.value,
            mp_pose.PoseLandmark.LEFT_HIP.value,
            mp_pose.PoseLandmark.LEFT_KNEE.value,
            mp_pose.PoseLandmark.LEFT_ANKLE.value,
        )
        right_idxs = (
            mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
            mp_pose.PoseLandmark.RIGHT_HIP.value,
            mp_pose.PoseLandmark.RIGHT_KNEE.value,
            mp_pose.PoseLandmark.RIGHT_ANKLE.value,
        )

        left_vis = np.mean([landmarks[i].visibility for i in left_idxs])
        right_vis = np.mean([landmarks[i].visibility for i in right_idxs])
        return "left" if left_vis >= right_vis else "right"

    def _compute_metrics(self, landmarks):
        side = self._pick_side(landmarks)
        if side == "left":
            shoulder_idx = mp_pose.PoseLandmark.LEFT_SHOULDER.value
            hip_idx = mp_pose.PoseLandmark.LEFT_HIP.value
            knee_idx = mp_pose.PoseLandmark.LEFT_KNEE.value
            ankle_idx = mp_pose.PoseLandmark.LEFT_ANKLE.value
        else:
            shoulder_idx = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
            hip_idx = mp_pose.PoseLandmark.RIGHT_HIP.value
            knee_idx = mp_pose.PoseLandmark.RIGHT_KNEE.value
            ankle_idx = mp_pose.PoseLandmark.RIGHT_ANKLE.value

        shoulder = self._point(landmarks, shoulder_idx)
        hip = self._point(landmarks, hip_idx)
        knee = self._point(landmarks, knee_idx)
        ankle = self._point(landmarks, ankle_idx)

        knee_angle = calculate_angle(hip, knee, ankle)
        hip_angle = calculate_angle(shoulder, hip, ankle)
        torso_lean = abs(shoulder[0] - hip[0])
        hip_height_delta = hip[1] - ((shoulder[1] + ankle[1]) / 2.0)

        return {
            "side": side,
            "knee_angle": knee_angle,
            "hip_angle": hip_angle,
            "torso_lean": torso_lean,
            "hip_height_delta": hip_height_delta,
        }

    def _build_pose_hint(self, pose_name, metrics):
        pose = pose_name.lower()
        knee_angle = metrics["knee_angle"]
        hip_angle = metrics["hip_angle"]
        torso_lean = metrics["torso_lean"]
        hip_height_delta = metrics["hip_height_delta"]

        if "squat_bad_back" in pose:
            return "rounded_back", "Lift chest and brace core to keep a neutral spine."
        if "squat_bad_heel" in pose:
            return "heels_rising", "Push through mid-foot and heels; keep heels planted."
        if "squat_good" in pose and knee_angle > 115:
            return "shallow_squat", "Sit back and lower hips slightly for a deeper squat."
        if "plank" in pose:
            if hip_height_delta > 0.07:
                return "sagging_hips", "Squeeze glutes and core to lift hips into one straight line."
            if hip_height_delta < -0.07:
                return "piked_hips", "Lower hips slightly and align shoulders, hips, and ankles."
        if "downdog" in pose and hip_height_delta > -0.05:
            return "low_hips", "Send hips up and back; lengthen spine and press through shoulders."
        if "tree" in pose and torso_lean > 0.10:
            return "leaning_torso", "Stack ribs over hips and fix your gaze to steady balance."
        if ("warrior2" in pose or "goddess" in pose) and knee_angle > 125:
            return "insufficient_knee_bend", "Bend front knee more and track it over the toes."
        if ("warrior2" in pose or "goddess" in pose) and hip_angle < 145:
            return "collapsed_torso", "Lift your torso tall and keep your chest open."
        return None, None

    def _request_feedback(self, state_packet, issue):
        now = time.time()
        if (now - self.last_feedback_time) <= self.cooldown:
            return
        if self.feedback_pending:
            return

        signature = f"{state_packet.get('pose')}::{issue}"
        if signature == self.last_feedback_signature and (now - self.last_feedback_time) <= (self.cooldown * 2):
            return

        try:
            self.feedback_queue.put_nowait(state_packet)
            self.feedback_pending = True
            self.last_feedback_signature = signature
            self.last_feedback_time = now
        except queue.Full:
            pass

    def run(self):
        print("Starting camera feed... Press 'q' to quit.")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise SystemExit("Unable to access camera 0. Check camera permissions or index.")

        all_landmarks_visible_previously = False
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame.shape[1] > self.frame_width:
                scale = self.frame_width / frame.shape[1]
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

            frame_count += 1
            should_process = (frame_count % self.process_every_n_frames == 0) or (self.last_results is None)
            if should_process:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.last_results = self.pose_detector.process(rgb)

            image = frame.copy()
            results = self.last_results

            if results and results.pose_landmarks:
                landmarks_obj = results.pose_landmarks.landmark
                invisible_landmarks_indices = [
                    i for i in KEY_LANDMARKS if landmarks_obj[i].visibility < self.visibility_threshold
                ]

                if invisible_landmarks_indices:
                    all_landmarks_visible_previously = False
                    invisible_landmark_names = [mp_pose.PoseLandmark(i).name for i in invisible_landmarks_indices]
                    clean_names = [name.replace("LEFT_", "").replace("RIGHT_", "") for name in invisible_landmark_names]
                    if (time.time() - self.last_feedback_time) > self.cooldown:
                        feedback_message = f"Please make sure the following body parts are visible: {', '.join(invisible_landmark_names)}"
                        print(f"Coach: {feedback_message}")
                        self.last_feedback_time = time.time()

                    cv2.putText(
                        image,
                        f"Adjust to show: {', '.join(clean_names[:4])}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 0, 255),
                        2,
                    )
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                else:
                    if not all_landmarks_visible_previously:
                        print("Coach: Great! All body parts are visible. Let's start the analysis.")
                        all_landmarks_visible_previously = True

                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                    landmarks = [value for lm in landmarks_obj for value in (lm.x, lm.y, lm.z, lm.visibility)]
                    if self.clf and self.le:
                        pred = self.clf.predict([landmarks])
                        pose_name = self.le.inverse_transform(pred)[0]
                    else:
                        pose_name = "Training required"

                    metrics = self._compute_metrics(landmarks_obj)
                    knee_angle = int(metrics["knee_angle"])

                    cv2.putText(image, f"Pose: {pose_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(
                        image,
                        f"Knee({metrics['side']}): {knee_angle}",
                        (10, 58),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )

                    issue, local_hint = self._build_pose_hint(pose_name, metrics)
                    if issue:
                        state_packet = {
                            "pose": pose_name,
                            "issue": issue,
                            "knee_angle": int(metrics["knee_angle"]),
                            "hip_angle": int(metrics["hip_angle"]),
                            "torso_lean": round(metrics["torso_lean"], 3),
                            "hip_height_delta": round(metrics["hip_height_delta"], 3),
                            "side": metrics["side"],
                            "local_hint": local_hint,
                        }
                        self._request_feedback(state_packet, issue)

            else:
                all_landmarks_visible_previously = False
                if (time.time() - self.last_feedback_time) > self.cooldown:
                    print("Coach: I can't see you. Please position yourself in front of the camera.")
                    self.last_feedback_time = time.time()
                cv2.putText(
                    image,
                    "I can't see you. Position yourself in the frame.",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

            if self.feedback_version > self.printed_feedback_version:
                print(f"Coach: {self.latest_feedback}")
                self.printed_feedback_version = self.feedback_version

            if self.latest_feedback:
                cv2.putText(
                    image,
                    f"Tip: {self.latest_feedback[:90]}",
                    (10, image.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )

            cv2.imshow("Your Gym Buddy", image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        self.pose_detector.close()
        self.stop_event.set()
        self.feedback_thread.join(timeout=1.0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    load_env_file(ENV_PATH)
    API_KEY = os.getenv("GROQ_API_KEY")
    if not API_KEY:
        raise SystemExit("GROQ_API_KEY is not set. Add it in .env or export it before running.")
    PostureAnalyzer(API_KEY).run()
