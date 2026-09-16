"""
Yoga Asana Classification — Real-Time Pose Detection & Posture Feedback
=======================================================================
Stage 1 Real-Time Pose Estimation and Machine Learning Classifier

Classes Detected:
    - Warrior II (Virabhadrasana II)
    - Tree Pose (Vrksasana)
    - Plank Pose (Phalakasana)
    - Mountain Pose (Tadasana)
    - Sitting / Resting

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
    Runs trained ML pipeline on a single frame's landmark coordinates
    with biomechanical ground-truth verification for Mountain and Tree poses.
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

    # Biomechanical Ground-Truth Verification for Mountain & Tree poses
    if landmarks_xyz is not None and len(landmarks_xyz) >= 33:
        l_knee = feats_dict.get("left_knee_angle", 180)
        r_knee = feats_dict.get("right_knee_angle", 180)
        l_hip = feats_dict.get("left_hip_angle", 180)
        r_hip = feats_dict.get("right_hip_angle", 180)
        torso = feats_dict.get("torso_inclination", 0)

        l_ank_y = landmarks_xyz[LM["LEFT_ANKLE"]][1]
        r_ank_y = landmarks_xyz[LM["RIGHT_ANKLE"]][1]
        l_foot_y = landmarks_xyz[LM["LEFT_FOOT_INDEX"]][1]
        r_foot_y = landmarks_xyz[LM["RIGHT_FOOT_INDEX"]][1]
        mid_hip_y = (landmarks_xyz[LM["LEFT_HIP"]][1] + landmarks_xyz[LM["RIGHT_HIP"]][1]) / 2.0

        min_knee = min(l_knee, r_knee)
        max_knee = max(l_knee, r_knee)
        ankle_diff_y = abs(l_ank_y - r_ank_y)
        foot_diff_y = abs(l_foot_y - r_foot_y)

        # 1. Tree Pose (Vrksasana):
        # Standing upright (mid_hip_y < 0.78, torso <= 22)
        # ONE standing leg is straight (max_knee >= 142)
        # ONE knee is bent (min_knee <= 110)
        # Foot lifted off floor (ankle_diff_y >= 0.035 or foot_diff_y >= 0.035 or min_knee <= 95)
        is_tree = (
            mid_hip_y < 0.78
            and max_knee >= 142
            and min_knee <= 110
            and torso <= 22
            and (ankle_diff_y >= 0.035 or foot_diff_y >= 0.035 or min_knee <= 95)
        )

        # 2. Mountain Pose (Tadasana - Hands Interlocked Overhead & Stretching on Toes):
        l_shld = feats_dict.get("left_shoulder_angle", 0)
        r_shld = feats_dict.get("right_shoulder_angle", 0)
        l_elb = feats_dict.get("left_elbow_angle", 180)
        r_elb = feats_dict.get("right_elbow_angle", 180)
        l_ank = feats_dict.get("left_ankle_angle", 90)
        r_ank = feats_dict.get("right_ankle_angle", 90)

        l_wrist_y = landmarks_xyz[LM["LEFT_WRIST"]][1]
        r_wrist_y = landmarks_xyz[LM["RIGHT_WRIST"]][1]
        l_wrist_x = landmarks_xyz[LM["LEFT_WRIST"]][0]
        r_wrist_x = landmarks_xyz[LM["RIGHT_WRIST"]][0]
        l_shld_y = landmarks_xyz[LM["LEFT_SHOULDER"]][1]
        r_shld_y = landmarks_xyz[LM["RIGHT_SHOULDER"]][1]
        nose_y = landmarks_xyz[LM["NOSE"]][1]

        l_heel_y = landmarks_xyz[LM["LEFT_HEEL"]][1]
        r_heel_y = landmarks_xyz[LM["RIGHT_HEEL"]][1]

        # Hands interlocked and stretched straight above head:
        arms_overhead_interlocked = (
            (l_wrist_y < l_shld_y and r_wrist_y < r_shld_y)
            and (l_wrist_y < nose_y and r_wrist_y < nose_y)
            and (l_shld >= 135 and r_shld >= 135)
            and (l_elb >= 135 and r_elb >= 135)
            and (abs(l_wrist_x - r_wrist_x) < 0.25)
        )

        # Stretching on toes:
        on_toes = (
            (l_heel_y < l_foot_y - 0.012 or r_heel_y < r_foot_y - 0.012)
            or (l_ank >= 104 or r_ank >= 104)
        )

        is_standing_upright = (
            mid_hip_y < 0.78
            and l_knee >= 145 and r_knee >= 145
            and l_hip >= 140 and r_hip >= 140
            and torso <= 20
        )

        is_mountain_correct = (
            is_standing_upright
            and arms_overhead_interlocked
            and on_toes
        )

        if is_tree:
            label = "Tree"
            confidence = max(confidence if confidence is not None else 0.92, 0.95)
        elif is_mountain_correct:
            label = "Mountain"
            confidence = max(confidence if confidence is not None else 0.92, 0.98)
        elif is_standing_upright and label not in ["Warrior II", "Plank"]:
            label = "Standing"
            confidence = 0.92

    return label, confidence, feats_dict


def detect_sitting_or_partial(landmarks_xyz, landmarks_list, feats_dict):
    """
    Detects if the user is truly seated or sitting close to the camera (desk sitting)
    where only the upper body is visible, while ensuring full-body standing poses
    (like Mountain and Tree) are never falsely blocked.
    Returns:
        (is_detected, label, feedback, confidence)
    """
    if landmarks_xyz is None or len(landmarks_xyz) < 33:
        return False, None, None, None

    l_hip_idx = LM["LEFT_HIP"]
    r_hip_idx = LM["RIGHT_HIP"]
    l_knee_idx = LM["LEFT_KNEE"]
    r_knee_idx = LM["RIGHT_KNEE"]

    mid_hip_y = (landmarks_xyz[l_hip_idx][1] + landmarks_xyz[r_hip_idx][1]) / 2.0
    l_knee_y = landmarks_xyz[l_knee_idx][1]
    r_knee_y = landmarks_xyz[r_knee_idx][1]

    l_knee = feats_dict.get("left_knee_angle", 180)
    r_knee = feats_dict.get("right_knee_angle", 180)
    l_hip = feats_dict.get("left_hip_angle", 180)
    r_hip = feats_dict.get("right_hip_angle", 180)
    torso = feats_dict.get("torso_inclination", 0)

    # 1. True Desk Sitting / Upper-Body Only Check:
    # A user is ONLY sitting at a desk if their hips are in the lower portion of the frame
    # (mid_hip_y > 0.75) AND knees are cut off below the screen (both > 0.95 or invisible).
    # When a user is standing, mid_hip_y is <= 0.75, so desk sitting will NEVER trigger!
    knees_below_frame = (l_knee_y > 0.96 and r_knee_y > 0.96)
    low_knee_vis = False
    if landmarks_list is not None and len(landmarks_list) > 28:
        l_knee_v = getattr(landmarks_list[l_knee_idx], "visibility", 1.0)
        r_knee_v = getattr(landmarks_list[r_knee_idx], "visibility", 1.0)
        if l_knee_v < 0.35 and r_knee_v < 0.35:
            low_knee_vis = True

    if mid_hip_y > 0.75 and (knees_below_frame or low_knee_vis):
        return (
            True,
            "Sitting / Step Back",
            "Lower body not in frame. Step back so full body is visible.",
            0.95,
        )

    # 2. Chair or Floor Sitting Check:
    # Full body is visible, but user is seated:
    # Both hips bent (~90°), both knees bent (~90°), torso upright.
    # If either knee is straight (>= 142°), the user is standing, NOT sitting!
    is_chair_sitting = (
        (mid_hip_y > 0.50)
        and (65 <= l_hip <= 130 and 65 <= r_hip <= 130)
        and (55 <= l_knee <= 130 and 55 <= r_knee <= 130)
        and torso < 35
    )

    is_floor_sitting = (
        (45 <= l_hip <= 85 and 45 <= r_hip <= 85)
        and (30 <= l_knee <= 70 and 30 <= r_knee <= 70)
        and torso < 25
    )

    if is_chair_sitting or is_floor_sitting:
        return (
            True,
            "Sitting",
            "Currently sitting. Stand up and step back to perform yoga asanas.",
            0.95,
        )

    return False, None, None, None


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

    elif label == "Plank":
        torso = feats_dict.get("torso_inclination", 0)
        l_hip = feats_dict.get("left_hip_angle", 180)
        if l_hip < 155:
            feedback.append("Avoid sagging or lifting hips too high")

    elif label == "Sitting":
        feedback.append("Currently sitting. Stand up and step back to perform yoga asanas")

    elif label == "Sitting / Step Back":
        feedback.append("Lower body not in frame. Step back so full body is visible")

    elif label == "Standing":
        l_shld = feats_dict.get("left_shoulder_angle", 0)
        r_shld = feats_dict.get("right_shoulder_angle", 0)
        l_elb = feats_dict.get("left_elbow_angle", 180)
        r_elb = feats_dict.get("right_elbow_angle", 180)
        l_ank = feats_dict.get("left_ankle_angle", 90)
        r_ank = feats_dict.get("right_ankle_angle", 90)

        arms_up = (l_shld >= 135 and r_shld >= 135 and l_elb >= 135 and r_elb >= 135)
        on_toes = (l_ank >= 104 or r_ank >= 104)

        if not arms_up and not on_toes:
            feedback.append("Interlock fingers, stretch arms straight above head & lift heels on toes for Mountain Pose")
        elif not arms_up:
            feedback.append("Interlock fingers and stretch arms straight above head")
        elif not on_toes:
            feedback.append("Lift heels and stretch upward on your toes!")
        else:
            feedback.append("Great posture alignment! Full upward stretch on toes")

    elif label == "Mountain":
        feedback.append("Great posture alignment! Full upward stretch on toes")

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
    elif "Sitting" in label or "Step Back" in label:
        status_color = (255, 180, 40)
    elif "Standing" in label:
        status_color = (240, 200, 50)
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
        landmarks_xyz, landmarks_list = detector.detect(frame)

        label = "No person detected"
        confidence = None
        feedback = None

        if landmarks_xyz is not None:
            # Draw skeleton
            draw_skeleton(frame, landmarks_xyz)

            # Feature Extraction
            feats_dict = extract_features(landmarks_xyz)

            # Check if user is sitting (chair/floor) or lower body is cut off
            is_sitting, sit_label, sit_feedback, sit_conf = detect_sitting_or_partial(
                landmarks_xyz, landmarks_list, feats_dict
            )

            if is_sitting:
                label = sit_label
                confidence = sit_conf
                feedback = sit_feedback
            else:
                # ML Classification
                label, confidence, _ = predict_pose(
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