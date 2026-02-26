import os
import queue
import threading
import time

import cv2
import mediapipe as mp
import pickle

from scripts.feedback_agent import FeedbackAgent
from scripts.pose_features import (
    COMMON_REQUIRED,
    FEATURE_NAMES,
    compute_engineered_features,
    feature_vector,
    has_required_visibility,
    required_indices_for_pose,
)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

project_root = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(project_root, "models", "pose_classifier.pkl")
LE_PATH = os.path.join(project_root, "models", "label_encoder.pkl")
ENV_PATH = os.path.join(project_root, ".env")


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


class SquatStateMachine:
    """Tracks squat movement phases and rep count using hip trajectory + knee angle."""

    def __init__(self):
        self.phase = "standing"
        self.rep_count = 0
        self.prev_hip_y = None
        self.hip_y_smooth = None
        self.knee_angle_smooth = None
        self.phase_frames = 0

    def reset(self):
        self.phase = "standing"
        self.prev_hip_y = None
        self.hip_y_smooth = None
        self.knee_angle_smooth = None
        self.phase_frames = 0

    def _smooth(self, value, prev, alpha=0.35):
        if value is None:
            return prev
        if prev is None:
            return float(value)
        return float((alpha * value) + ((1.0 - alpha) * prev))

    def update(self, hip_y, knee_angle):
        if hip_y is None or knee_angle is None:
            return self.phase, False

        self.hip_y_smooth = self._smooth(hip_y, self.hip_y_smooth, alpha=0.35)
        self.knee_angle_smooth = self._smooth(knee_angle, self.knee_angle_smooth, alpha=0.30)

        velocity = 0.0 if self.prev_hip_y is None else (self.hip_y_smooth - self.prev_hip_y)
        self.prev_hip_y = self.hip_y_smooth

        next_phase = self.phase
        self.phase_frames += 1

        min_dwell = 3

        if self.phase == "standing":
            # Enter descent only with clear downward movement + knee bend.
            if self.phase_frames >= min_dwell and velocity > 0.005 and self.knee_angle_smooth < 162:
                next_phase = "descent"

        elif self.phase == "descent":
            # Require deeper bend and low velocity near bottom to avoid bounce flicker.
            if (
                self.phase_frames >= min_dwell
                and self.knee_angle_smooth < 98
                and abs(velocity) < 0.0045
            ):
                next_phase = "bottom"
            # Recovery path in case user aborts squat.
            elif self.phase_frames >= 5 and self.knee_angle_smooth > 166 and velocity < 0.002:
                next_phase = "standing"

        elif self.phase == "bottom":
            # Exit bottom only when ascent is clear and knees extend a bit.
            if self.phase_frames >= min_dwell and velocity < -0.0035 and self.knee_angle_smooth > 96:
                next_phase = "ascent"

        elif self.phase == "ascent":
            # Count rep only after full extension and near-stable top.
            if self.phase_frames >= min_dwell and self.knee_angle_smooth > 166 and abs(velocity) < 0.0035:
                next_phase = "standing"
                self.rep_count += 1

        changed = next_phase != self.phase
        if changed:
            self.phase = next_phase
            self.phase_frames = 0
        return self.phase, changed


class PostureAnalyzer:
    def __init__(self, api_key=None):
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
                model_blob = pickle.load(f)
            with open(LE_PATH, "rb") as f:
                self.le = pickle.load(f)

            if isinstance(model_blob, dict):
                self.clf = model_blob.get("model")
                trained_features = model_blob.get("feature_names")
                if trained_features and list(trained_features) != list(FEATURE_NAMES):
                    raise SystemExit(
                        "Model feature schema mismatch. Re-run training with current feature set."
                    )
            else:
                self.clf = model_blob
                print("Warning: legacy model format detected (no feature schema metadata).")
        else:
            print("Model files not found. Please train the model first.")

        self.feedback_agent = FeedbackAgent(api_key)
        self.llm_enabled = self.feedback_agent.client is not None
        self.last_feedback_time = 0
        self.cooldown = 4
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

        self.squat_machine = SquatStateMachine()
        self.nonsquat_frames = 0
        self.last_rule_signature = None
        self.last_rule_feedback_time = 0.0

    def _feedback_worker(self):
        while not self.stop_event.is_set():
            try:
                packet = self.feedback_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            feedback_text = self.feedback_agent.get_feedback(packet)
            if feedback_text:
                self.latest_feedback = feedback_text
                self.feedback_version += 1
            self.feedback_pending = False
            self.feedback_queue.task_done()

    def _build_pose_hint(self, pose_name, metrics, phase):
        pose = (pose_name or "").lower()
        knee_angle = metrics["knee_angle"]
        hip_angle = metrics["hip_angle"]
        torso_lean = metrics["torso_lean"]
        hip_height_delta = metrics["hip_height_delta"]

        if "squat" in pose:
            if "bad_back" in pose:
                if phase == "descent":
                    return "rounded_back_descent", "Keep chest proud during descent and brace your core."
                if phase == "bottom":
                    return "rounded_back_bottom", "At bottom, lengthen spine and avoid lower-back rounding."
                return "rounded_back_ascent", "Drive up with a neutral spine, not a rounded back."

            if "bad_heel" in pose:
                if phase == "descent":
                    return "heels_rising_descent", "Descend slowly while keeping full foot pressure, especially heels."
                if phase == "bottom":
                    return "heels_rising_bottom", "At bottom, keep heels grounded and knees tracking over toes."
                return "heels_rising_ascent", "Push through heels and mid-foot as you stand."

            if phase == "bottom" and knee_angle > 112:
                return "shallow_bottom", "Sink slightly deeper at bottom while keeping your chest up."
            if phase == "descent" and torso_lean > 0.23:
                return "torso_folding", "Hinge less forward; keep ribs stacked over hips."
            return None, None

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
        if not self.llm_enabled:
            return

        now = time.time()
        if (now - self.last_feedback_time) <= self.cooldown:
            return
        if self.feedback_pending:
            return

        signature = f"{state_packet.get('pose')}::{state_packet.get('phase')}::{issue}"
        if signature == self.last_feedback_signature and (now - self.last_feedback_time) <= (self.cooldown * 2):
            return

        try:
            self.feedback_queue.put_nowait(state_packet)
            self.feedback_pending = True
            self.last_feedback_signature = signature
            self.last_feedback_time = now
        except queue.Full:
            pass

    def _emit_rule_feedback(self, signature, cue):
        now = time.time()
        if signature != self.last_rule_signature or (now - self.last_rule_feedback_time) > self.cooldown:
            print(f"Coach: {cue}")
            self.last_rule_signature = signature
            self.last_rule_feedback_time = now

    def _names_from_indices(self, indices):
        names = [mp_pose.PoseLandmark(i).name for i in indices]
        return [name.replace("LEFT_", "").replace("RIGHT_", "") for name in names]

    def run(self):
        print("Starting camera feed... Press 'q' to quit.")
        if not self.llm_enabled:
            print("GROQ_API_KEY not set. Running deterministic coaching only.")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise SystemExit("Unable to access camera 0. Check camera permissions or index.")

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
                landmarks = results.pose_landmarks.landmark
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                common_ok, common_missing, confidence = has_required_visibility(
                    landmarks,
                    COMMON_REQUIRED,
                    self.visibility_threshold,
                )
                if not common_ok:
                    self.squat_machine.reset()
                    self.nonsquat_frames = 0
                    missing_names = self._names_from_indices(common_missing)
                    cv2.putText(
                        image,
                        f"Low confidence joints: {', '.join(missing_names[:4])}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 0, 255),
                        2,
                    )
                    cv2.putText(
                        image,
                        "Move fully into frame before coaching starts.",
                        (10, 58),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )
                    pose_name = "confidence_low"
                else:
                    features, metrics = compute_engineered_features(landmarks)
                    if self.clf and self.le:
                        pred = self.clf.predict([feature_vector(features)])
                        pose_name = self.le.inverse_transform(pred)[0]
                    else:
                        pose_name = "training_required"

                    pose_required = required_indices_for_pose(pose_name)
                    pose_ok, pose_missing, pose_conf = has_required_visibility(
                        landmarks,
                        pose_required,
                        self.visibility_threshold,
                    )

                    phase = "static"
                    if "squat" in pose_name.lower():
                        self.nonsquat_frames = 0
                        phase, _ = self.squat_machine.update(metrics.get("hip_y"), metrics.get("knee_angle"))
                    else:
                        self.nonsquat_frames += 1
                        if self.nonsquat_frames >= 8:
                            self.squat_machine.reset()

                    cv2.putText(image, f"Pose: {pose_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 0), 2)
                    cv2.putText(
                        image,
                        f"Conf: {confidence:.2f} | Phase: {phase}",
                        (10, 58),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.58,
                        (0, 255, 255),
                        2,
                    )
                    if "squat" in pose_name.lower():
                        cv2.putText(
                            image,
                            f"Reps: {self.squat_machine.rep_count}",
                            (10, 84),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.62,
                            (255, 220, 0),
                            2,
                        )

                    if pose_ok:
                        issue, deterministic_cue = self._build_pose_hint(pose_name, metrics, phase)
                        if issue and deterministic_cue:
                            self.latest_feedback = deterministic_cue
                            self._emit_rule_feedback(f"{pose_name}:{phase}:{issue}", deterministic_cue)
                            state_packet = {
                                "pose": pose_name,
                                "issue": issue,
                                "phase": phase,
                                "knee_angle": int(metrics["knee_angle"]),
                                "hip_angle": int(metrics["hip_angle"]),
                                "torso_lean": round(metrics["torso_lean"], 3),
                                "hip_height_delta": round(metrics["hip_height_delta"], 3),
                                "side": metrics["side"],
                                "deterministic_cue": deterministic_cue,
                            }
                            self._request_feedback(state_packet, issue)
                    else:
                        missing_names = self._names_from_indices(pose_missing)
                        cv2.putText(
                            image,
                            f"Need clearer view: {', '.join(missing_names[:4])}",
                            (10, image.shape[0] - 45),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (0, 100, 255),
                            2,
                        )
                        cv2.putText(
                            image,
                            f"Pose confidence gated ({pose_conf:.2f})",
                            (10, image.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (0, 100, 255),
                            2,
                        )

            else:
                self.squat_machine.reset()
                self.nonsquat_frames = 0
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
                    f"Tip: {self.latest_feedback[:95]}",
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
    api_key = os.getenv("GROQ_API_KEY")
    PostureAnalyzer(api_key).run()
