"""
Yoga Asana Classification — Feature Extraction & Model Training Pipeline
=======================================================================
Provides landmark extraction, 11-feature biomechanical angle calculations,
synthetic & dataset training, evaluation, and model artifact serialization.

Artifacts saved:
    - yoga_pose_model.joblib
    - feature_scaler.joblib
    - label_encoder.joblib
    - feature_columns.joblib
"""

import os
import sys
import argparse
import urllib.request
import numpy as np
import cv2
import mediapipe as mp
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 33 MediaPipe Pose Landmark Indices Mapping
LM = {
    "NOSE": 0,
    "LEFT_EYE_INNER": 1,
    "LEFT_EYE": 2,
    "LEFT_EYE_OUTER": 3,
    "RIGHT_EYE_INNER": 4,
    "RIGHT_EYE": 5,
    "RIGHT_EYE_OUTER": 6,
    "LEFT_EAR": 7,
    "RIGHT_EAR": 8,
    "MOUTH_LEFT": 9,
    "MOUTH_RIGHT": 10,
    "LEFT_SHOULDER": 11,
    "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW": 13,
    "RIGHT_ELBOW": 14,
    "LEFT_WRIST": 15,
    "RIGHT_WRIST": 16,
    "LEFT_PINKY": 17,
    "RIGHT_PINKY": 18,
    "LEFT_INDEX": 19,
    "RIGHT_INDEX": 20,
    "LEFT_THUMB": 21,
    "RIGHT_THUMB": 22,
    "LEFT_HIP": 23,
    "RIGHT_HIP": 24,
    "LEFT_KNEE": 25,
    "RIGHT_KNEE": 26,
    "LEFT_ANKLE": 27,
    "RIGHT_ANKLE": 28,
    "LEFT_HEEL": 29,
    "RIGHT_HEEL": 30,
    "LEFT_FOOT_INDEX": 31,
    "RIGHT_FOOT_INDEX": 32,
}

POSE_CONNECTIONS = [
    # Torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Left Arm
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    # Right Arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    # Left Leg
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    # Right Leg
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
    # Face
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
]

FEATURE_COLUMNS = [
    "left_elbow_angle",
    "right_elbow_angle",
    "left_shoulder_angle",
    "right_shoulder_angle",
    "left_hip_angle",
    "right_hip_angle",
    "left_knee_angle",
    "right_knee_angle",
    "left_ankle_angle",
    "right_ankle_angle",
    "torso_inclination",
]


class PoseDetector:
    """
    Universal MediaPipe Pose Detector supporting MediaPipe Tasks API and legacy solutions.
    """
    def __init__(self, model_path="pose_landmarker.task", min_detection_confidence=0.5):
        self.model_path = model_path
        self.min_confidence = min_detection_confidence
        self.detector = None
        self.use_tasks_api = False

        # Attempt to use modern MediaPipe Tasks API first
        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            self._init_tasks_api()
        except Exception as e:
            try:
                self._init_solutions_api()
            except Exception as e2:
                raise RuntimeError(f"Failed to initialize MediaPipe Pose: Tasks API error ({e}), Solutions API error ({e2})")

    def _init_tasks_api(self):
        if not os.path.exists(self.model_path):
            print(f"[*] Downloading MediaPipe PoseLandmarker model to '{self.model_path}'...")
            url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
            urllib.request.urlretrieve(url, self.model_path)
            print(f"[*] Downloaded PoseLandmarker model successfully.")

        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            min_pose_detection_confidence=self.min_confidence,
            min_pose_presence_confidence=self.min_confidence,
            min_tracking_confidence=self.min_confidence,
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)
        self.use_tasks_api = True

    def _init_solutions_api(self):
        self.detector = mp.solutions.pose.Pose(
            static_image_mode=False,
            min_detection_confidence=self.min_confidence,
            min_tracking_confidence=0.5,
        )
        self.use_tasks_api = False

    def detect(self, frame):
        """
        Extracts landmarks from an OpenCV BGR frame.
        Returns:
            (landmarks_xyz, landmarks_list)
            landmarks_xyz: ndarray of shape (33, 3) or None
            landmarks_list: raw landmark objects or None
        """
        if frame is None:
            return None, None

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.use_tasks_api:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            results = self.detector.detect(mp_image)
            if not results or not results.pose_landmarks or len(results.pose_landmarks) == 0:
                return None, None
            landmarks_list = results.pose_landmarks[0]
            landmarks_xyz = np.array([[lm.x, lm.y, lm.z] for lm in landmarks_list], dtype=np.float32)
            return landmarks_xyz, landmarks_list
        else:
            results = self.detector.process(image_rgb)
            if not results or not results.pose_landmarks:
                return None, None
            landmarks_list = results.pose_landmarks.landmark
            landmarks_xyz = np.array([[lm.x, lm.y, lm.z] for lm in landmarks_list], dtype=np.float32)
            return landmarks_xyz, landmarks_list


def calculate_angle(a, b, c, use_3d=False):
    """
    Calculates the interior angle at vertex b formed by points a-b-c.
    
    Parameters:
        a, b, c: Coordinates (x, y) or (x, y, z) as arrays or lists.
        use_3d: If True, computes angle in 3D Euclidean space.
    
    Returns:
        angle in degrees in the range [0.0, 180.0].
    """
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    c = np.array(c, dtype=np.float64)

    if use_3d and len(a) >= 3 and len(b) >= 3 and len(c) >= 3:
        v1 = a[:3] - b[:3]
        v2 = c[:3] - b[:3]
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-7 or norm2 < 1e-7:
            return 180.0
        cosine = np.dot(v1, v2) / (norm1 * norm2)
        cosine = np.clip(cosine, -1.0, 1.0)
        return float(np.degrees(np.arccos(cosine)))
    else:
        # 2D angle
        rad1 = np.arctan2(a[1] - b[1], a[0] - b[0])
        rad2 = np.arctan2(c[1] - b[1], c[0] - b[0])
        angle = np.abs(np.degrees(rad1 - rad2))
        if angle > 180.0:
            angle = 360.0 - angle
        return float(angle)


def extract_landmarks(frame, pose_model):
    """
    Extracts 33 pose landmarks (x, y, z) from a BGR image frame.
    Supports either PoseDetector instance or legacy MediaPipe Pose model.
    """
    if frame is None:
        return None

    if isinstance(pose_model, PoseDetector):
        xyz, _ = pose_model.detect(frame)
        return xyz

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if hasattr(pose_model, "process"):
        results = pose_model.process(image_rgb)
        if not results or not results.pose_landmarks:
            return None
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark], dtype=np.float32)
        return landmarks

    return None


def extract_features(landmarks_xyz):
    """
    Extracts the 11 biomechanical features (joint angles and torso inclination)
    used across training and live inference.
    
    Parameters:
        landmarks_xyz: numpy array of shape (33, 3) or dict mapping landmark name to [x, y, z].
        
    Returns:
        dict mapping feature column name -> float angle in degrees.
    """
    if isinstance(landmarks_xyz, dict):
        def get_pt(name):
            idx = LM[name] if isinstance(name, str) else name
            return landmarks_xyz[idx]
    else:
        def get_pt(name):
            idx = LM[name] if isinstance(name, str) else name
            return landmarks_xyz[idx]

    # Extract 3D points
    l_shoulder = get_pt("LEFT_SHOULDER")
    r_shoulder = get_pt("RIGHT_SHOULDER")
    l_elbow = get_pt("LEFT_ELBOW")
    r_elbow = get_pt("RIGHT_ELBOW")
    l_wrist = get_pt("LEFT_WRIST")
    r_wrist = get_pt("RIGHT_WRIST")
    l_hip = get_pt("LEFT_HIP")
    r_hip = get_pt("RIGHT_HIP")
    l_knee = get_pt("LEFT_KNEE")
    r_knee = get_pt("RIGHT_KNEE")
    l_ankle = get_pt("LEFT_ANKLE")
    r_ankle = get_pt("RIGHT_ANKLE")
    l_foot = get_pt("LEFT_FOOT_INDEX")
    r_foot = get_pt("RIGHT_FOOT_INDEX")

    # Joint Angles
    left_elbow = calculate_angle(l_shoulder, l_elbow, l_wrist)
    right_elbow = calculate_angle(r_shoulder, r_elbow, r_wrist)

    left_shoulder = calculate_angle(l_hip, l_shoulder, l_elbow)
    right_shoulder = calculate_angle(r_hip, r_shoulder, r_elbow)

    left_hip = calculate_angle(l_shoulder, l_hip, l_knee)
    right_hip = calculate_angle(r_shoulder, r_hip, r_knee)

    left_knee = calculate_angle(l_hip, l_knee, l_ankle)
    right_knee = calculate_angle(r_hip, r_knee, r_ankle)

    left_ankle = calculate_angle(l_knee, l_ankle, l_foot)
    right_ankle = calculate_angle(r_knee, r_ankle, r_foot)

    # Torso inclination: angle between spine (mid-hip to mid-shoulder) and vertical
    mid_shoulder = (np.array(l_shoulder) + np.array(r_shoulder)) / 2.0
    mid_hip = (np.array(l_hip) + np.array(r_hip)) / 2.0
    
    spine = mid_shoulder - mid_hip
    vertical = np.array([0.0, -1.0, 0.0])
    norm_spine = np.linalg.norm(spine[:2])
    if norm_spine > 1e-7:
        cos_inc = np.dot(spine[:2], vertical[:2]) / norm_spine
        torso_inclination = float(np.degrees(np.arccos(np.clip(cos_inc, -1.0, 1.0))))
    else:
        torso_inclination = 0.0

    return {
        "left_elbow_angle": float(left_elbow),
        "right_elbow_angle": float(right_elbow),
        "left_shoulder_angle": float(left_shoulder),
        "right_shoulder_angle": float(right_shoulder),
        "left_hip_angle": float(left_hip),
        "right_hip_angle": float(right_hip),
        "left_knee_angle": float(left_knee),
        "right_knee_angle": float(right_knee),
        "left_ankle_angle": float(left_ankle),
        "right_ankle_angle": float(right_ankle),
        "torso_inclination": float(torso_inclination),
    }


def generate_synthetic_pose_dataset(samples_per_class=350, random_seed=42):
    """
    Generates a realistic, biomechanically grounded synthetic training dataset
    for yoga poses including Warrior II, Tree, Goddess, Downward Dog, Plank, and Mountain.
    """
    np.random.seed(random_seed)

    pose_specs = {
        "Warrior II": [
            {"means": [172, 172, 92, 92, 115, 168, 95, 175, 88, 105, 5], "std": 6},
            {"means": [172, 172, 92, 92, 168, 115, 175, 95, 105, 88, 5], "std": 6},
        ],
        "Tree": [
            {"means": [155, 155, 150, 150, 172, 120, 176, 45, 105, 75, 4], "std": 7},
            {"means": [155, 155, 150, 150, 120, 172, 45, 176, 75, 105, 4], "std": 7},
            {"means": [60, 60, 45, 45, 172, 120, 176, 45, 105, 75, 4], "std": 6},
            {"means": [60, 60, 45, 45, 120, 172, 45, 176, 75, 105, 4], "std": 6},
        ],
        "Goddess": [
            {"means": [90, 90, 92, 92, 110, 110, 100, 100, 90, 90, 6], "std": 7},
            {"means": [95, 95, 88, 88, 118, 118, 90, 90, 88, 88, 5], "std": 6},
            {"means": [85, 85, 95, 95, 102, 102, 110, 110, 92, 92, 6], "std": 7},
        ],
        "Downward Dog": [
            {"means": [175, 175, 162, 162, 68, 68, 174, 174, 85, 85, 48], "std": 6},
        ],
        "Plank": [
            {"means": [175, 175, 88, 88, 172, 172, 175, 175, 88, 88, 82], "std": 5},
        ],
        "Mountain": [
            {"means": [174, 174, 20, 20, 175, 175, 176, 176, 92, 92, 3], "std": 4},
        ],
    }

    X_list = []
    y_list = []

    for pose_name, variants in pose_specs.items():
        n_per_variant = samples_per_class // len(variants)
        for variant in variants:
            means = np.array(variant["means"], dtype=np.float64)
            std = float(variant["std"])
            
            samples = np.random.normal(loc=means, scale=std, size=(n_per_variant, len(FEATURE_COLUMNS)))
            noise = np.random.uniform(-2.5, 2.5, size=samples.shape)
            samples = np.clip(samples + noise, 0.0, 180.0)

            for sample in samples:
                X_list.append(sample)
                y_list.append(pose_name)

    return np.array(X_list, dtype=np.float32), np.array(y_list)


def train_and_save_model(output_dir="models", n_estimators=120, test_size=0.2, random_seed=42):
    """
    Trains a Yoga Pose Classifier and saves all 4 pipeline artifacts to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"[*] Generating synthetic biomechanical training dataset...")
    X, y = generate_synthetic_pose_dataset(samples_per_class=400, random_seed=random_seed)
    
    print(f"[*] Dataset shape: {X.shape}, Classes: {np.unique(y).tolist()}")

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_seed, stratify=y_encoded
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Classifier
    print(f"[*] Training RandomForest Classifier...")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=16,
        min_samples_split=3,
        random_state=random_seed,
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)

    # Evaluation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n================ MODEL EVALUATION ================")
    print(f" 5-Fold Cross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f" Test Set Accuracy: {acc * 100:.2f}%")
    print(f"==================================================\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # Serialize artifacts
    model_path = os.path.join(output_dir, "yoga_pose_model.joblib")
    scaler_path = os.path.join(output_dir, "feature_scaler.joblib")
    encoder_path = os.path.join(output_dir, "label_encoder.joblib")
    columns_path = os.path.join(output_dir, "feature_columns.joblib")

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(label_encoder, encoder_path)
    joblib.dump(FEATURE_COLUMNS, columns_path)

    print(f"[+] Saved artifacts to '{output_dir}':")
    print(f"    - {model_path}")
    print(f"    - {scaler_path}")
    print(f"    - {encoder_path}")
    print(f"    - {columns_path}")

    return model, scaler, label_encoder, FEATURE_COLUMNS


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Yoga Pose Classifier")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models",
        help="Directory to save the trained model and preprocessors",
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=120,
        help="Number of trees in Random Forest",
    )
    args = parser.parse_args()

    train_and_save_model(output_dir=args.output_dir, n_estimators=args.n_estimators)
