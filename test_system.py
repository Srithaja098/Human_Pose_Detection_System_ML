"""
Automated Verification & Diagnostics Suite for Human Pose Detection System
========================================================================
Validates dependencies, model pipeline, feature extraction, pose predictions,
and real-time inference components.
"""

import os
import sys
import numpy as np
import cv2

def run_tests():
    print("=" * 60)
    print(" HUMAN POSE DETECTION SYSTEM — VERIFICATION & DIAGNOSTICS")
    print("=" * 60)

    # 1. Test Dependencies
    print("\n[Step 1/5] Verifying Dependencies...")
    for pkg in ["cv2", "mediapipe", "sklearn", "numpy", "joblib"]:
        mod = __import__(pkg)
        ver = getattr(mod, "__version__", "loaded")
        print(f"  [+] {pkg}: OK ({ver})")

    # 2. Test Model Training Pipeline Import
    print("\n[Step 2/5] Testing train_yoga_pose_classifier.py...")
    from train_yoga_pose_classifier import (
        LM,
        FEATURE_COLUMNS,
        PoseDetector,
        calculate_angle,
        extract_features,
        train_and_save_model,
        generate_synthetic_pose_dataset,
    )
    print(f"  [+] MediaPipe Landmark Map (LM) count: {len(LM)} landmarks")
    print(f"  [+] Feature Columns count: {len(FEATURE_COLUMNS)} features")

    # Angle calculation sanity check
    angle_90 = calculate_angle([0, 1], [0, 0], [1, 0])
    assert abs(angle_90 - 90.0) < 1e-4, f"Angle calculation failed: expected 90.0, got {angle_90}"
    print(f"  [+] calculate_angle 90-deg test: {angle_90:.1f}° (PASSED)")

    # 3. Test Pipeline Loading & Inference
    print("\n[Step 3/5] Testing Model Loading & Inference...")
    from model import load_pipeline, predict_pose, generate_pose_feedback

    model, scaler, label_encoder, feature_cols = load_pipeline("models")
    print(f"  [+] Loaded Model: {type(model).__name__}")
    print(f"  [+] Scaler: {type(scaler).__name__}")
    print(f"  [+] Label Encoder Classes ({len(label_encoder.classes_)}): {list(label_encoder.classes_)}")

    # 4. Test Predictions on Canonical Yoga Pose feature vectors
    print("\n[Step 4/5] Testing Canonical Pose Predictions...")
    test_cases = {
        "Warrior II": {
            "left_elbow_angle": 175.0, "right_elbow_angle": 175.0,
            "left_shoulder_angle": 90.0, "right_shoulder_angle": 90.0,
            "left_hip_angle": 115.0, "right_hip_angle": 170.0,
            "left_knee_angle": 92.0, "right_knee_angle": 175.0,
            "left_ankle_angle": 90.0, "right_ankle_angle": 105.0,
            "torso_inclination": 5.0,
        },
        "Goddess": {
            "left_elbow_angle": 90.0, "right_elbow_angle": 90.0,
            "left_shoulder_angle": 92.0, "right_shoulder_angle": 92.0,
            "left_hip_angle": 110.0, "right_hip_angle": 110.0,
            "left_knee_angle": 98.0, "right_knee_angle": 98.0,
            "left_ankle_angle": 90.0, "right_ankle_angle": 90.0,
            "torso_inclination": 5.0,
        },
        "Plank": {
            "left_elbow_angle": 175.0, "right_elbow_angle": 175.0,
            "left_shoulder_angle": 88.0, "right_shoulder_angle": 88.0,
            "left_hip_angle": 172.0, "right_hip_angle": 172.0,
            "left_knee_angle": 175.0, "right_knee_angle": 175.0,
            "left_ankle_angle": 88.0, "right_ankle_angle": 88.0,
            "torso_inclination": 85.0,
        },
        "Downward Dog": {
            "left_elbow_angle": 175.0, "right_elbow_angle": 175.0,
            "left_shoulder_angle": 165.0, "right_shoulder_angle": 165.0,
            "left_hip_angle": 68.0, "right_hip_angle": 68.0,
            "left_knee_angle": 175.0, "right_knee_angle": 175.0,
            "left_ankle_angle": 85.0, "right_ankle_angle": 85.0,
            "torso_inclination": 48.0,
        },
    }

    for expected_pose, feats in test_cases.items():
        feat_vec = np.array([[feats[c] for c in feature_cols]], dtype=np.float32)
        scaled_vec = scaler.transform(feat_vec)
        pred_idx = model.predict(scaled_vec)[0]
        predicted_label = label_encoder.inverse_transform([pred_idx])[0]
        conf = float(np.max(model.predict_proba(scaled_vec)[0]))
        feedback = generate_pose_feedback(predicted_label, feats)
        
        status = "PASSED" if predicted_label == expected_pose else "MISMATCH"
        print(f"  [+] Expected: '{expected_pose}' -> Predicted: '{predicted_label}' ({conf*100:.1f}%) | Tip: '{feedback}' [{status}]")
        assert predicted_label == expected_pose, f"Failed prediction for {expected_pose}"

    # 5. Test PoseDetector on Synthetic Frame
    print("\n[Step 5/5] Testing PoseDetector on Synthetic Frame...")
    detector = PoseDetector(min_detection_confidence=0.5)
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(test_frame, "Pose Test", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    landmarks_xyz, landmarks_list = detector.detect(test_frame)
    print("  [+] Blank frame detection handled gracefully (no crash, returned None):", landmarks_xyz is None)

    print("\n" + "=" * 60)
    print(" ALL SYSTEM DIAGNOSTICS & TESTS PASSED WITH 0 ERRORS! ")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
