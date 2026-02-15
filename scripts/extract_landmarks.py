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
    
    # Define paths based on your environment
    datasets = {
        "squat_good": "/home/anurag/Desktop/Dataset/train/Good",
        "squat_bad_back": "/home/anurag/Desktop/Dataset/train/Bad back",
        "squat_bad_heel": "/home/anurag/Desktop/Dataset/train/Bad heel",
        "downdog": "/home/anurag/Desktop/yoga_posture/DATASET/TRAIN/downdog",
        "tree": "/home/anurag/Desktop/yoga_posture/DATASET/TRAIN/tree"
    }
    
    for label, folder_path in datasets.items():
        print(f"Processing {label}...")
        if not os.path.exists(folder_path):
            print(f"Directory {folder_path} not found. Skipping.")
            continue
            
        for img_path in glob.glob(os.path.join(folder_path, "*")):
            if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
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
    output_path = "/home/anurag/Desktop/your_gym_buddy/data/train_data.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} samples to {output_path}")

if __name__ == "__main__":
    process_dataset()
