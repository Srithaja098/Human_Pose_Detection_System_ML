"""
Expert Pose Recording Tool
==========================
Records a yoga instructor or expert holding a target asana via live webcam.
Only frames where all scored joint landmarks are visible (visibility >= 0.5)
are retained and written to expert_angles.csv.

Usage:
    python record_expert.py --asana "Warrior II" --samples 150
    python record_expert.py --asana "Tree" --samples 150
    python record_expert.py --asana "Mountain" --samples 150
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2
import pandas as pd

from train_yoga_pose_classifier import PoseDetector, extract_features, LM
from pose_config import (
    SCORED_ASANAS,
    SCORED_JOINTS,
    JOINT_LANDMARK_MAP,
    MIN_LANDMARK_VISIBILITY,
)


def are_all_scored_joints_visible(landmarks_list, min_visibility=MIN_LANDMARK_VISIBILITY):
    """
    Checks if all landmarks involved in the 9 scored joint angles have
    MediaPipe visibility >= min_visibility.
    """
    if landmarks_list is None or len(landmarks_list) < 33:
        return False, ["Full body not detected"]

    missing_joints = []
    # Collect all unique landmark indices needed for scored joints
    needed_landmarks = set()
    for joint_name in SCORED_JOINTS:
        lm_indices = JOINT_LANDMARK_MAP.get(joint_name, [])
        needed_landmarks.update(lm_indices)

    for lm_idx in needed_landmarks:
        vis = getattr(landmarks_list[lm_idx], "visibility", 1.0)
        if vis < min_visibility:
            # Identify which joints are affected
            for joint_name, indices in JOINT_LANDMARK_MAP.items():
                if lm_idx in indices and joint_name not in missing_joints:
                    missing_joints.append(joint_name)

    return (len(missing_joints) == 0), missing_joints


def record_expert_pose(asana_name, output_csv="expert_angles.csv",
                       target_samples=150, camera_index=0):
    """
    Records expert posture frames from webcam and appends to output_csv.
    """
    if asana_name not in SCORED_ASANAS:
        print(f"[!] Error: '{asana_name}' is not in the recognized scored asanas: {SCORED_ASANAS}")
        sys.exit(1)

    print("=" * 65)
    print(f" EXPERT POSE RECORDER — Target Asana: '{asana_name}'")
    print(f" Target Valid Frames: {target_samples} | Output CSV: {output_csv}")
    print("=" * 65)
    print("Instructions:")
    print("  1. Position your camera so your full body is in frame.")
    print("  2. Assume the pose stably.")
    print("  3. Press SPACEBAR when ready to start recording.")
    print("  4. Hold steady. Only frames where all scored joints are visible are saved.")
    print("  5. Press 'q' at any time to cancel.\n")

    detector = PoseDetector(min_detection_confidence=0.5)
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[!] Error: Could not open camera at index {camera_index}")
        sys.exit(1)

    is_recording = False
    captured_data = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[!] Error reading from camera.")
                break

            # Mirror for natural user interaction
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            landmarks_xyz, landmarks_list = detector.detect(frame)
            all_visible = False
            missing = []

            if landmarks_xyz is not None:
                all_visible, missing = are_all_scored_joints_visible(
                    landmarks_list, min_visibility=MIN_LANDMARK_VISIBILITY
                )

                # Draw joint markers
                for pt in landmarks_xyz:
                    cx, cy = int(pt[0] * w), int(pt[1] * h)
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 0) if all_visible else (0, 140, 255), -1)

                if is_recording and all_visible:
                    feats = extract_features(landmarks_xyz)
                    row = {
                        "asana": asana_name,
                        "timestamp": time.time(),
                    }
                    for joint in SCORED_JOINTS:
                        row[joint] = feats.get(joint, 0.0)
                    captured_data.append(row)

            # Draw HUD overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 80), (25, 25, 25), -1)
            cv2.rectangle(overlay, (0, h - 50), (w, h), (25, 25, 25), -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

            # Header text
            cv2.putText(
                frame,
                f"EXPERT RECORDING: {asana_name.upper()}",
                (20, 32),
                cv2.FONT_HERSHEY_DUPLEX,
                0.75,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            status_text = "READY — Press SPACE to start" if not is_recording else f"RECORDING ({len(captured_data)}/{target_samples})"
            status_color = (0, 220, 255) if not is_recording else (0, 255, 100)
            cv2.putText(
                frame,
                status_text,
                (20, 65),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                status_color,
                2,
                cv2.LINE_AA,
            )

            # Bottom status
            if landmarks_xyz is None:
                bottom_tip = "No person detected. Step back into full view."
            elif not all_visible:
                bottom_tip = f"Joints occluded: {', '.join(missing[:2])} (Visibility < {MIN_LANDMARK_VISIBILITY})"
            elif is_recording:
                bottom_tip = "Hold steady... recording clean frames."
            else:
                bottom_tip = "Pose visible and clear. Press SPACE to begin."

            cv2.putText(
                frame,
                bottom_tip,
                (20, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow("Expert Pose Recorder", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):
                is_recording = not is_recording
                print(f"[*] Recording state toggled: {'RECORDING' if is_recording else 'PAUSED'}")
            elif key == ord("q"):
                print("[*] Recording cancelled by user.")
                break

            if len(captured_data) >= target_samples:
                print(f"\n[+] Successfully captured {target_samples} valid expert frames!")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    if captured_data:
        df_new = pd.DataFrame(captured_data)
        if os.path.exists(output_csv):
            df_old = pd.read_csv(output_csv)
            # Remove existing rows for this asana to overwrite cleanly
            df_old = df_old[df_old["asana"] != asana_name]
            df_final = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_final = df_new

        df_final.to_csv(output_csv, index=False)
        print(f"[+] Saved expert frames to '{output_csv}' ({len(captured_data)} frames for '{asana_name}')")
        print(f"[*] Next step: build reference profile via:")
        print(f"    python build_reference.py --csv {output_csv} --asana \"{asana_name}\"\n")
    else:
        print("[!] No frames were saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record Expert Yoga Asana Posture Frames")
    parser.add_argument(
        "--asana",
        type=str,
        required=True,
        choices=SCORED_ASANAS,
        help=f"Target asana name to record ({SCORED_ASANAS})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="expert_angles.csv",
        help="Path to CSV file where expert angles are saved (default: expert_angles.csv)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=150,
        help="Number of valid full-visibility frames to capture (default: 150)",
    )
    parser.add_argument(
        "--camera_index",
        type=int,
        default=0,
        help="Webcam camera index (default: 0)",
    )
    args = parser.parse_args()

    record_expert_pose(
        asana_name=args.asana,
        output_csv=args.output,
        target_samples=args.samples,
        camera_index=args.camera_index,
    )
