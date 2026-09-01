"""
Yoga Asana Classification — Real-Time Pose Detection & Posture Feedback
=======================================================================
Stage 1 Real-Time Pose Estimation and Machine Learning Classifier

Classes Detected:
    - Warrior II (Virabhadrasana II)
    - Tree Pose (Vrksasana)
    - Goddess Pose (Utkata Konasana)
    - Downward-Facing Dog (Adho Mukha Svanasana)
    - Plank Pose (Phalakasana)
    - Mountain Pose (Tadasana)

Usage:
    # Run live webcam detection:
    python model.py

    # Specify custom model directory or camera:
    python model.py --model_dir models --camera_index 0

    # Run on an image or video file:
    python model.py --image path/to/pose.jpg
    python model.py --video path/to/video.mp4

Press 'q' to quit the window.
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2
import joblib

from train_yoga_pose_classifier import (
    LM,
    POSE_CONNECTIONS,
    FEATURE_COLUMNS,
    PoseDetector,
    extract_landmarks,
    extract_features,
    calculate_angle,
    train_and_save_model,
)


def ensure_models_exist(model_dir):
    """
    Checks if required model artifacts exist in model_dir.
    If missing, automatically trains and saves a new model pipeline.
    """
    required_files = [
        "yoga_pose_model.joblib",
        "feature_scaler.joblib",
        "label_encoder.joblib",
        "feature_columns.joblib",
    ]
    missing = [f for f in required_files if not os.path.exists(os.path.join(model_dir, f))]
    if missing:
        print(f"[!] Model files missing in '{model_dir}': {missing}")
        print(f"[*] Automatically training model pipeline now...")
        train_and_save_model(output_dir=model_dir)


def load_pipeline(model_dir="models"):
    """
    Loads trained classifier, feature scaler, label encoder, and feature column list.
    """
    ensure_models_exist(model_dir)
    model = joblib.load(os.path.join(model_dir, "yoga_pose_model.joblib"))
    scaler = joblib.load(os.path.join(model_dir, "feature_scaler.joblib"))
    label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.joblib"))
    feature_cols = joblib.load(os.path.join(model_dir, "feature_columns.joblib"))
    return model, scaler, label_encoder, feature_cols


def predict_pose(landmarks_xyz, model, scaler, label_encoder, feature_cols):
    """
    Runs trained ML pipeline on a single frame's landmark coordinates.
    Returns:
        (predicted_label, confidence, feats_dict)
    """
    feats_dict = extract_features(landmarks_xyz)
    feature_vector = np.array([[feats_dict[c] for c in feature_cols]], dtype=np.float32)
    feature_vector_scaled = scaler.transform(feature_vector)

    prediction = model.predict(feature_vector_scaled)[0]
    label = label_encoder.inverse_transform([prediction])[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(feature_vector_scaled)[0]
        confidence = float(np.max(proba))

    return label, confidence, feats_dict


def generate_pose_feedback(label, feats_dict):
    """
    Provides real-time posture alignment tips based on extracted joint angles.
    """
    feedback = []
    if label == "Warrior II":
        l_knee = feats_dict.get("left_knee_angle", 180)
        r_knee = feats_dict.get("right_knee_angle", 180)
        min_knee = min(l_knee, r_knee)
        if min_knee > 120:
            feedback.append("Bend front knee closer to 90 degrees")
        l_shld = feats_dict.get("left_shoulder_angle", 0)
        r_shld = feats_dict.get("right_shoulder_angle", 0)
        if abs(l_shld - 90) > 20 or abs(r_shld - 90) > 20:
            feedback.append("Keep arms parallel to ground")

    elif label == "Tree":
        l_knee = feats_dict.get("left_knee_angle", 180)
        r_knee = feats_dict.get("right_knee_angle", 180)
        bent_knee = min(l_knee, r_knee)
        if bent_knee > 85:
            feedback.append("Place foot firmly on inner thigh/calf")
        torso = feats_dict.get("torso_inclination", 0)
        if torso > 15:
            feedback.append("Lengthen spine straight upwards")

    elif label == "Goddess":
        l_knee = feats_dict.get("left_knee_angle", 180)
        r_knee = feats_dict.get("right_knee_angle", 180)
        if l_knee > 120 or r_knee > 120:
            feedback.append("Sink deeper into squat")
        l_elb = feats_dict.get("left_elbow_angle", 180)
        r_elb = feats_dict.get("right_elbow_angle", 180)
        if abs(l_elb - 90) > 25 or abs(r_elb - 90) > 25:
            feedback.append("Bend elbows at 90-degree cactus arms")

    elif label == "Plank":
        torso = feats_dict.get("torso_inclination", 0)
        l_hip = feats_dict.get("left_hip_angle", 180)
        if l_hip < 155:
            feedback.append("Avoid sagging or lifting hips too high")

    elif label == "Downward Dog":
        l_hip = feats_dict.get("left_hip_angle", 180)
        if l_hip > 90:
            feedback.append("Push hips up and back into inverted V")

    if not feedback:
        feedback.append("Great posture alignment!")

    return feedback[0]


def draw_skeleton(frame, landmarks_xyz):
    """
    Renders high-visibility skeleton connections and joint circles on the frame.
    """
    if landmarks_xyz is None or len(landmarks_xyz) < 33:
        return

    h, w, _ = frame.shape

    # Draw connection bones
    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx < len(landmarks_xyz) and end_idx < len(landmarks_xyz):
            p1 = landmarks_xyz[start_idx]
            p2 = landmarks_xyz[end_idx]
            x1, y1 = int(p1[0] * w), int(p1[1] * h)
            x2, y2 = int(p2[0] * w), int(p2[1] * h)
            cv2.line(frame, (x1, y1), (x2, y2), (245, 117, 66), 3, cv2.LINE_AA)

    # Draw joint nodes
    for idx, pt in enumerate(landmarks_xyz):
        cx, cy = int(pt[0] * w), int(pt[1] * h)
        cv2.circle(frame, (cx, cy), 5, (245, 66, 230), -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 7, (255, 255, 255), 1, cv2.LINE_AA)


def draw_hud(frame, label, confidence, feedback, fps, confidence_threshold=0.6):
    """
    Renders a sleek, modern visual interface on top of the video feed.
    """
    h, w, _ = frame.shape
    overlay = frame.copy()

    # Top banner background
    cv2.rectangle(overlay, (0, 0), (w, 85), (20, 24, 33), -1)
    # Bottom banner background
    cv2.rectangle(overlay, (0, h - 50), (w, h), (20, 24, 33), -1)

    # Blend for glassmorphic effect
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Accent color based on status
    status_color = (0, 220, 100) if (confidence and confidence >= confidence_threshold) else (0, 165, 255)
    if label == "No person detected":
        status_color = (80, 80, 220)
    cv2.line(frame, (0, 85), (w, 85), status_color, 2)

    # Title
    cv2.putText(
        frame,
        "AI POSE DETECTOR",
        (20, 30),
        cv2.FONT_HERSHEY_DUPLEX,
        0.65,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    # Pose Label & Confidence
    if label == "No person detected":
        display_text = "No Person Detected"
    elif confidence is not None and confidence < confidence_threshold:
        display_text = f"Uncertain ({label} - {confidence * 100:.0f}%)"
    else:
        conf_str = f" ({confidence * 100:.0f}%)" if confidence is not None else ""
        display_text = f"{label.upper()}{conf_str}"

    cv2.putText(
        frame,
        display_text,
        (20, 68),
        cv2.FONT_HERSHEY_DUPLEX,
        1.0,
        status_color,
        2,
        cv2.LINE_AA,
    )

    # FPS Counter (Top Right)
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(
        frame,
        fps_text,
        (w - 130, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )

    # Posture Feedback (Bottom Bar)
    feedback_text = f"Tip: {feedback}" if feedback else "Press 'q' to exit"
    cv2.putText(
        frame,
        feedback_text,
        (20, h - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    # Exit instruction (Bottom Right)
    cv2.putText(
        frame,
        "Exit: 'q'",
        (w - 100, h - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (150, 150, 150),
        1,
        cv2.LINE_AA,
    )


def process_source(source, model, scaler, label_encoder, feature_cols,
                   confidence_threshold=0.6, is_image=False, save_output=None):
    """
    Main processing loop for camera stream, video file, or static image.
    """
    cap = None
    if is_image:
        frame_orig = cv2.imread(source)
        if frame_orig is None:
            raise FileNotFoundError(f"Could not open image file: {source}")
    else:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source '{source}'. Check camera connection/index.")

    detector = PoseDetector(min_detection_confidence=0.5)
    prev_time = time.time()

    while True:
        if is_image:
            frame = frame_orig.copy()
        else:
            ret, frame = cap.read()
            if not ret:
                print("[*] End of stream or video playback completed.")
                break
            if isinstance(source, int):
                frame = cv2.flip(frame, 1)

        # Calculate FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 30.0
        prev_time = curr_time

        # Pose Detection
        landmarks_xyz, _ = detector.detect(frame)

        label = "No person detected"
        confidence = None
        feedback = None

        if landmarks_xyz is not None:
            # Draw skeleton
            draw_skeleton(frame, landmarks_xyz)

            # ML Classification
            label, confidence, feats_dict = predict_pose(
                landmarks_xyz, model, scaler, label_encoder, feature_cols
            )
            feedback = generate_pose_feedback(label, feats_dict)

        # Draw UI
        draw_hud(frame, label, confidence, feedback, fps, confidence_threshold)

        if save_output:
            cv2.imwrite(save_output, frame)
            print(f"[+] Result saved to '{save_output}'")

        # Display window
        cv2.imshow("Human Pose Detection & Yoga Asana Recognition", frame)
        key = cv2.waitKey(1 if not is_image else 0) & 0xFF
        if key == ord("q") or is_image:
            break

    if cap:
        cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Real-Time Human Pose Detection & Yoga Asana Classification")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models",
        help="Directory containing trained .joblib pipeline files (default: 'models')",
    )
    parser.add_argument(
        "--camera_index",
        type=int,
        default=0,
        help="Camera device index for webcam (default: 0)",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Optional path to video file instead of live webcam",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Optional path to image file for single-image pose classification",
    )
    parser.add_argument(
        "--save_output",
        type=str,
        default=None,
        help="Optional path to save resulting frame/image",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.6,
        help="Minimum confidence threshold before flagging pose as uncertain (default: 0.6)",
    )
    args = parser.parse_args()

    # Load / Auto-train model pipeline
    model, scaler, label_encoder, feature_cols = load_pipeline(args.model_dir)
    print(f"[*] Pipeline loaded successfully.")
    print(f"[*] Recognizable Asana Classes: {list(label_encoder.classes_)}")

    # Determine input source
    if args.image:
        print(f"[*] Processing image: {args.image}")
        process_source(
            args.image,
            model,
            scaler,
            label_encoder,
            feature_cols,
            confidence_threshold=args.confidence_threshold,
            is_image=True,
            save_output=args.save_output,
        )
    elif args.video:
        print(f"[*] Processing video file: {args.video}")
        process_source(
            args.video,
            model,
            scaler,
            label_encoder,
            feature_cols,
            confidence_threshold=args.confidence_threshold,
            is_image=False,
            save_output=args.save_output,
        )
    else:
        print(f"[*] Starting live webcam feed (Camera Index: {args.camera_index})...")
        print("[*] Press 'q' in the display window to quit.")
        process_source(
            args.camera_index,
            model,
            scaler,
            label_encoder,
            feature_cols,
            confidence_threshold=args.confidence_threshold,
            is_image=False,
            save_output=args.save_output,
        )


if __name__ == "__main__":
    main()