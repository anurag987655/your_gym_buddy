import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import pickle
import os

def train_model():
    # Get the project root directory (Desktop/your_gym_buddy)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    csv_path = os.path.join(project_root, "data", "processed", "pose_data.csv")
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run extract_landmarks.py first.")
        return

    # Load data
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    X = df.drop('label', axis=1)
    y = df['label']

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Train Random Forest
    print("Training Random Forest model...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Save model and label encoder
    model_dir = os.path.join(project_root, "models")
    os.makedirs(model_dir, exist_ok=True)
    
    with open(os.path.join(model_dir, "pose_classifier.pkl"), 'wb') as f:
        pickle.dump(clf, f)
    
    with open(os.path.join(model_dir, "label_encoder.pkl"), 'wb') as f:
        pickle.dump(le, f)
        
    print(f"Model and Label Encoder saved to {model_dir}")

if __name__ == "__main__":
    train_model()
