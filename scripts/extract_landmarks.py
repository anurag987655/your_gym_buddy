import cv2
import mediapipe as mp
import pandas as pd
import os
import glob

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def extract_landmarks_from_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    
    if results.pose_landmarks:
        landmarks = []
        for lm in results.pose_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
        return landmarks
    return None

def process_dataset():
    data = []
    
    # Get the project root directory (Desktop/your_gym_buddy)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    base_raw_path = os.path.join(project_root, "data", "raw")
    
    datasets = {
        "squat_good": os.path.join(base_raw_path, "squat_good"),
        "squat_bad_back": os.path.join(base_raw_path, "squat_bad_back"),
        "squat_bad_heel": os.path.join(base_raw_path, "squat_bad_heel"),
        "downdog": os.path.join(base_raw_path, "downdog"),
        "tree": os.path.join(base_raw_path, "tree"),
        "goddess": os.path.join(base_raw_path, "goddess"),
        "plank": os.path.join(base_raw_path, "plank"),
        "warrior2": os.path.join(base_raw_path, "warrior2"),
    }
    
    for label, folder_path in datasets.items():
        print(f"Processing {label}...")
        if not os.path.exists(folder_path):
            print(f"Directory {folder_path} not found. Skipping.")
            continue
            
        for img_path in glob.glob(os.path.join(folder_path, "*")):
            if img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                landmarks = extract_landmarks_from_image(img_path)
                if landmarks:
                    data.append([label] + landmarks)
    
    if not data:
        print("No landmarks extracted. Check your image paths.")
        return

    # Define column names: label, x0, y0, z0, v0, ...
    columns = ['label']
    for i in range(33):
        columns.extend([f'x{i}', f'y{i}', f'z{i}', f'v{i}'])
        
    df = pd.DataFrame(data, columns=columns)
    output_dir = os.path.join(project_root, "data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "pose_data.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} samples to {output_path}")

if __name__ == "__main__":
    process_dataset()
