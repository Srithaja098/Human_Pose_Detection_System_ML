"""
Reference Pose Data Recorder
============================
Records reference pose videos and per-frame 11 biomechanical angles from a
webcam or an existing video file (--source). Reference data is recorded by
project members performing poses correctly (not external certified experts).

Usage:
    python record_reference.py --pose "Warrior II" --person_name Alice --duration 10 --camera_index 0
    python record_reference.py --pose "Sitting" --person_name Bob --source path/to/video.mp4
"""

import os
import sys
import time
import argparse
from datetime import datetime
from typing import Optional, Tuple, List, Dict

import cv2
import numpy as np
import pandas as pd

from train_yoga_pose_classifier import PoseDetector, extract_features, FEATURE_COLUMNS, LM
from model import detect_sitting_or_partial, draw_skeleton
from mistake_catalog import ACTIVE_CLASSES


TARGET_VALID_FRAMES = 150
LOWER_BODY_INDICES = [
    LM["LEFT_HIP"],
    LM["RIGHT_HIP"],
    LM["LEFT_KNEE"],
    LM["RIGHT_KNEE"],
    LM["LEFT_ANKLE"],
    LM["RIGHT_ANKLE"],
]


def print_setup_tips(pose_name: str) -> None:
    """Prints camera setup and pose-specific recording instructions."""
    print("=" * 70)
    print(f" REFERENCE DATA RECORDER — Pose: '{pose_name}'")
    print(" Note: Recorded by project members as ground-truth reference data.")
    print("=" * 70)
    print("Setup Tips:")
    print("  - Place camera at hip height, 2-3 m away, full body in frame, plain background.")
    if pose_name in ("Mountain", "Warrior II"):
        print(f"  - Face the camera for {pose_name}.")
    elif pose_name == "Tree":
        print("  - Face the camera for Tree pose.")
    if pose_name in ("Warrior II", "Tree"):
        print(f"  - Record BOTH sides for {pose_name} (left leg forward/raised and right leg forward/raised).")
    if pose_name == "Sitting":
        print("  - For Sitting, record chair sitting and floor cross-legged sitting with full body visible.")
    print(f"  - Target: {TARGET_VALID_FRAMES}+ valid frames. Press 'q' at any time to stop early.\n")


def is_valid_reference_frame(
    landmarks_xyz: Optional[np.ndarray],
    landmarks_list: Optional[list],
    feats_dict: Dict[str, float],
    pose_name: str,
) -> Tuple[bool, str]:
    """
    Validates whether a detected frame is suitable for reference data capture.

    Reuses detect_sitting_or_partial() visibility/framing logic:
    - Skips frames where no person is detected or lower body is cut off / low visibility.
    - Exception: when pose_name == 'Sitting', frames flagged as 'Sitting' are accepted
      as long as the full body (hips, knees, ankles) remains visible inside the frame.
    """
    if landmarks_xyz is None or len(landmarks_xyz) < 33:
        return False, "No person detected"

    # Check full lower-body framing and visibility
    l_knee_y = float(landmarks_xyz[LM["LEFT_KNEE"]][1])
    r_knee_y = float(landmarks_xyz[LM["RIGHT_KNEE"]][1])
    l_ank_y = float(landmarks_xyz[LM["LEFT_ANKLE"]][1])
    r_ank_y = float(landmarks_xyz[LM["RIGHT_ANKLE"]][1])

    if l_knee_y > 0.98 or r_knee_y > 0.98 or l_ank_y > 0.99 or r_ank_y > 0.99:
        return False, "Lower body cut off; step back so full body is visible"

    if landmarks_list is not None and len(landmarks_list) >= 33:
        for idx in LOWER_BODY_INDICES:
            vis = getattr(landmarks_list[idx], "visibility", 1.0)
            if vis < 0.35:
                return False, "Low lower-body visibility"

    is_sit, sit_label, sit_tip, _ = detect_sitting_or_partial(
        landmarks_xyz, landmarks_list, feats_dict
    )

    if pose_name == "Sitting":
        if is_sit and sit_label == "Sitting / Step Back":
            return False, sit_tip or "Lower body not in frame"
        return True, "Valid Sitting frame"

    if is_sit:
        return False, sit_tip or f"Rejected ({sit_label})"

    return True, "Valid frame"


def record_reference_session(
    pose_name: str,
    person_name: str,
    duration: float = 10.0,
    camera_index: int = 0,
    source: Optional[str] = None,
    expert_dir: str = "expert_data",
    countdown_sec: Optional[int] = None,
    headless: bool = False,
) -> Tuple[Optional[str], int]:
    """
    Records reference video and appends valid per-frame 11-angle rows to
    `<expert_dir>/expert_angles.csv`.

    Returns:
        (saved_video_path, valid_frame_count)
    """
    if pose_name not in ACTIVE_CLASSES:
        print(
            f"[!] Error: '{pose_name}' is not an active class. "
            f"Active classes are: {ACTIVE_CLASSES}"
        )
        return None, 0

    print_setup_tips(pose_name)

    cap_source = source if source is not None else camera_index
    if source is not None and not os.path.exists(source):
        print(f"[!] Error: Video source not found: '{source}'")
        return None, 0

    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        print(f"[!] Error: Could not open video source '{cap_source}'.")
        return None, 0

    # Default countdown: 5s for live webcam, 0s for pre-recorded video files unless specified
    if countdown_sec is None:
        countdown_sec = 0 if source is not None else 5

    pose_dir = os.path.join(expert_dir, pose_name)
    os.makedirs(pose_dir, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_person = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in person_name.strip()) or "member"
    video_path = os.path.join(pose_dir, f"{safe_person}_{timestamp_str}.mp4")
    csv_path = os.path.join(expert_dir, "expert_angles.csv")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1.0 or fps > 120.0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    detector = PoseDetector(min_detection_confidence=0.5)

    captured_rows: List[Dict[str, object]] = []
    raw_frames_written = 0
    user_stopped = False

    def show_preview(img: np.ndarray) -> bool:
        """Displays preview frame if GUI is enabled; returns True if 'q' pressed."""
        if headless:
            return False
        try:
            cv2.imshow(f"Reference Recorder - {pose_name}", img)
            key = cv2.waitKey(1) & 0xFF
            return key == ord("q")
        except cv2.error:
            return False

    try:
        # Phase 1: Countdown preview
        if countdown_sec > 0:
            cd_start = time.time()
            while True:
                elapsed = time.time() - cd_start
                remaining = countdown_sec - elapsed
                if remaining <= 0:
                    break
                ret, frame = cap.read()
                if not ret:
                    break
                if source is None:
                    frame = cv2.flip(frame, 1)

                display = frame.copy()
                landmarks_xyz, _ = detector.detect(display)
                if landmarks_xyz is not None:
                    draw_skeleton(display, landmarks_xyz)

                cv2.putText(
                    display,
                    f"POSE: {pose_name} | STARTING IN {int(np.ceil(remaining))}...",
                    (20, 45),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.85,
                    (0, 220, 255),
                    2,
                    cv2.LINE_AA,
                )
                if show_preview(display):
                    user_stopped = True
                    break

        # Phase 2: Recording for --duration seconds (or until video ends / 'q')
        if not user_stopped:
            rec_start = time.time()
            while True:
                if source is None and (time.time() - rec_start) >= duration:
                    break
                if source is not None and duration > 0 and (raw_frames_written / fps) >= duration:
                    break

                ret, frame = cap.read()
                if not ret:
                    break
                if source is None:
                    frame = cv2.flip(frame, 1)

                # Save raw frame (no HUD/skeleton overlay)
                writer.write(frame)
                raw_frames_written += 1

                landmarks_xyz, landmarks_list = detector.detect(frame)
                valid = False
                status_msg = "No person detected"

                if landmarks_xyz is not None:
                    feats = extract_features(landmarks_xyz)
                    valid, status_msg = is_valid_reference_frame(
                        landmarks_xyz, landmarks_list, feats, pose_name
                    )
                    if valid:
                        row: Dict[str, object] = {"pose": pose_name, "person": safe_person}
                        for col in FEATURE_COLUMNS:
                            row[col] = round(float(feats[col]), 4)
                        captured_rows.append(row)

                display = frame.copy()
                if landmarks_xyz is not None:
                    draw_skeleton(display, landmarks_xyz)

                color = (0, 220, 100) if valid else (0, 165, 255)
                cv2.putText(
                    display,
                    f"REC {pose_name} | Valid Frames: {len(captured_rows)} (Target: {TARGET_VALID_FRAMES}+)",
                    (20, 40),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.7,
                    color,
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    display,
                    f"Status: {status_msg} | Press 'q' to stop",
                    (20, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                if show_preview(display):
                    break

    finally:
        cap.release()
        writer.release()
        if not headless:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    if raw_frames_written == 0 and os.path.exists(video_path):
        os.remove(video_path)
        video_path = None

    if captured_rows:
        df_new = pd.DataFrame(captured_rows, columns=["pose", "person"] + FEATURE_COLUMNS)
        if os.path.exists(csv_path):
            df_old = pd.read_csv(csv_path)
            df_final = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_final = df_new
        df_final.to_csv(csv_path, index=False)
        print(f"[+] Appended {len(captured_rows)} valid frames to '{csv_path}'")
    else:
        print("[!] Warning: 0 valid frames captured; CSV was not updated.")

    if video_path:
        print(f"[+] Saved raw video ({raw_frames_written} frames) to '{video_path}'")

    return video_path, len(captured_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Record reference pose data (webcam or video)")
    parser.add_argument(
        "--pose",
        type=str,
        required=True,
        help=f"Target pose name. Active classes: {ACTIVE_CLASSES}",
    )
    parser.add_argument(
        "--person_name",
        type=str,
        default="Member1",
        help="Name of the project member performing the pose",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Recording duration in seconds (default: 10)",
    )
    parser.add_argument(
        "--camera_index",
        type=int,
        default=0,
        help="Webcam device index (default: 0)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Optional path to video file instead of webcam",
    )
    parser.add_argument(
        "--expert_dir",
        type=str,
        default="expert_data",
        help="Directory for recorded reference videos and CSV (default: expert_data)",
    )
    parser.add_argument(
        "--countdown",
        type=int,
        default=None,
        help="Countdown seconds before recording (default: 5 for webcam, 0 for --source)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Disable cv2.imshow preview window (useful for automated video tests)",
    )
    args = parser.parse_args()

    if args.pose not in ACTIVE_CLASSES:
        print(
            f"[!] Inactive or unknown pose '{args.pose}'. "
            f"Only active classes are supported: {ACTIVE_CLASSES}"
        )
        sys.exit(0)

    record_reference_session(
        pose_name=args.pose,
        person_name=args.person_name,
        duration=args.duration,
        camera_index=args.camera_index,
        source=args.source,
        expert_dir=args.expert_dir,
        countdown_sec=args.countdown,
        headless=args.headless,
    )


if __name__ == "__main__":
    main()
