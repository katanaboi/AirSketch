import csv
import os

import numpy as np


def extract_hand_landmark_points(landmarks):
    """
    Extract normalized hand landmark coordinates for model input.

    Args:
        landmarks: MediaPipe hand landmarks object

    Returns:
        List of 42 values (21 landmarks × 2 coordinates: x, y)
    """
    landmark_points = []

    # Extract x, y coordinates for all 21 hand landmarks
    for landmark in landmarks.landmark:
        # Normalize coordinates (MediaPipe already provides normalized coords 0-1)
        landmark_points.append(landmark.x)
        landmark_points.append(landmark.y)
        # Note: We're not using z coordinate to keep it 2D (42 features total)

    return landmark_points


def extract_hand_landmark_points_with_z(landmarks):
    """
    Extract hand landmark coordinates including z-coordinate.

    Args:
        landmarks: MediaPipe hand landmarks object

    Returns:
        List of 63 values (21 landmarks × 3 coordinates: x, y, z)
    """
    landmark_points = []

    # Extract x, y, z coordinates for all 21 hand landmarks
    for landmark in landmarks.landmark:
        landmark_points.append(landmark.x)
        landmark_points.append(landmark.y)
        landmark_points.append(landmark.z)

    return landmark_points


def normalize_landmarks_relative_to_wrist(landmarks):
    """
    Normalize landmarks relative to wrist position to make gestures translation-invariant.

    Args:
        landmarks: MediaPipe hand landmarks object

    Returns:
        List of normalized landmark coordinates
    """
    # Get wrist position (landmark 0)
    wrist_x = landmarks.landmark[0].x
    wrist_y = landmarks.landmark[0].y

    normalized_points = []

    for landmark in landmarks.landmark:
        # Subtract wrist position to make relative
        relative_x = landmark.x - wrist_x
        relative_y = landmark.y - wrist_y

        normalized_points.append(relative_x)
        normalized_points.append(relative_y)

    return normalized_points


def calculate_landmark_distances(landmarks):
    """
    Calculate distances between key landmarks for gesture recognition.

    Args:
        landmarks: MediaPipe hand landmarks object

    Returns:
        List of calculated distances between key points
    """
    # MediaPipe hand landmark indices
    WRIST = 0
    THUMB_TIP = 4
    INDEX_TIP = 8
    MIDDLE_TIP = 12
    RING_TIP = 16
    PINKY_TIP = 20

    distances = []

    # Extract coordinates
    points = [(lm.x, lm.y) for lm in landmarks.landmark]

    # Calculate distances from wrist to fingertips
    for tip_idx in [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]:
        dist = np.sqrt(
            (points[tip_idx][0] - points[WRIST][0]) ** 2
            + (points[tip_idx][1] - points[WRIST][1]) ** 2
        )
        distances.append(dist)

    # Calculate distances between adjacent fingertips
    fingertips = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
    for i in range(len(fingertips) - 1):
        dist = np.sqrt(
            (points[fingertips[i]][0] - points[fingertips[i + 1]][0]) ** 2
            + (points[fingertips[i]][1] - points[fingertips[i + 1]][1]) ** 2
        )
        distances.append(dist)

    return distances


def save_to_csv(landmarks, label, dataset_name="hand_landmarks_dataset"):
    os.makedirs("data", exist_ok=True)
    filename = f"data/{dataset_name}.csv"
    landmark_points = extract_hand_landmark_points(landmarks)

    file_exists = os.path.exists(filename)

    try:
        with open(filename, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)

            if not file_exists:
                header = []
                for i in range(21):
                    header.extend([f"x{i}", f"y{i}"])
                header.append("label")
                writer.writerow(header)

            row_data = landmark_points + [label]
            writer.writerow(row_data)
    except IOError as e:
        print(f"Error saving to CSV: {e}")


def load_dataset_from_csv(filename="data/hand_landmarks_dataset.csv"):
    """
    Load dataset from CSV file for training.

    Args:
        filename: CSV filename to load

    Returns:
        tuple: (features, labels) as numpy arrays
    """
    if not os.path.exists(filename):
        print(f"Dataset file {filename} not found!")
        return None, None

    features = []
    labels = []

    with open(filename, "r") as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            # Extract features (all columns except 'label')
            feature_row = []
            for key in row.keys():
                if key != "label":
                    feature_row.append(float(row[key]))

            features.append(feature_row)
            labels.append(row["label"])

    return np.array(features), np.array(labels)


def preprocess_dataset(features, labels):
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    print(f"Dataset preprocessing complete:")
    print(f"- Features shape: {scaled_features.shape}")
    print(f"- Number of classes: {len(label_encoder.classes_)}")
    print(f"- Classes: {list(label_encoder.classes_)}")

    return scaled_features, encoded_labels, label_encoder, scaler


def create_sample_data():
    import random

    gestures = ["thumbs_up", "peace", "fist", "open_hand", "pointing"]

    for gesture in gestures:
        for i in range(10):
            fake_landmarks = type("FakeLandmarks", (), {})()
            fake_landmarks.landmark = []

            for j in range(21):
                fake_landmark = type("FakeLandmark", (), {})()
                fake_landmark.x = random.random()
                fake_landmark.y = random.random()
                fake_landmark.z = random.random()
                fake_landmarks.landmark.append(fake_landmark)

            save_to_csv(fake_landmarks, gesture, "sample_dataset")

    print("Sample dataset created: sample_dataset.csv")
    print("This is for testing purposes only. Replace with real gesture data!")


# Hand landmark indices for reference
HAND_LANDMARKS = {
    "WRIST": 0,
    "THUMB_CMC": 1,
    "THUMB_MCP": 2,
    "THUMB_IP": 3,
    "THUMB_TIP": 4,
    "INDEX_MCP": 5,
    "INDEX_PIP": 6,
    "INDEX_DIP": 7,
    "INDEX_TIP": 8,
    "MIDDLE_MCP": 9,
    "MIDDLE_PIP": 10,
    "MIDDLE_DIP": 11,
    "MIDDLE_TIP": 12,
    "RING_MCP": 13,
    "RING_PIP": 14,
    "RING_DIP": 15,
    "RING_TIP": 16,
    "PINKY_MCP": 17,
    "PINKY_PIP": 18,
    "PINKY_DIP": 19,
    "PINKY_TIP": 20,
}


def get_landmark_name(index):
    """Get landmark name by index."""
    for name, idx in HAND_LANDMARKS.items():
        if idx == index:
            return name
    return f"LANDMARK_{index}"


if __name__ == "__main__":
    # Test the functions
    print("Testing utils functions...")

    # Create sample data for testing
    create_sample_data()

    # Load and show dataset info
    features, labels = load_dataset_from_csv("sample_dataset.csv")
    if features is not None:
        print(
            f"Loaded dataset: {features.shape[0]} samples, {features.shape[1]} features"
        )
        print(f"Unique labels: {np.unique(labels)}")

        # Preprocess dataset
        processed_features, encoded_labels, label_encoder, scaler = preprocess_dataset(
            features, labels
        )
        print("Dataset preprocessing successful!")
