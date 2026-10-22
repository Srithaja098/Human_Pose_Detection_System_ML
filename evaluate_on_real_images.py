"""
Evaluation Suite on Real Yoga Pose Images
=========================================
Evaluates the trained Yoga Pose Classifier on real-world labelled images.
Takes a root dataset directory formatted as <root>/<class>/*.jpg, runs
PoseDetector + extract_features + predict_pose, and outputs:
- Per-class precision, recall, F1-score, accuracy, and support
- Images where no person was detected (MediaPipe detection failures)
- Confusion Matrix (with dedicated breakdown for Mountain vs. Standing confusion)
- Support for --no_rules (pure ML classifier vs hybrid ML + geometric rules)
- Explicit reporting of missing classes (e.g. missing Mountain in Kaggle dataset)
  without fabricating any numbers.

Usage:
    # Hybrid evaluation (ML + Heuristic rules):
    python evaluate_on_real_images.py --data_dir path/to/dataset

    # Pure ML evaluation (rules disabled):
    python evaluate_on_real_images.py --data_dir path/to/dataset --no_rules

    # Mountain definition variant:
    python evaluate_on_real_images.py --data_dir path/to/dataset --mountain_def arms_at_sides
"""

import os
import sys
import glob
import argparse
import numpy as np
import cv2
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from train_yoga_pose_classifier import PoseDetector, extract_features
from model import load_pipeline, predict_pose
from pose_config import (
    SCORED_ASANAS,
    UNSCORED_STATES,
    DEFAULT_MOUNTAIN_DEF,
    MOUNTAIN_DEFINITIONS,
)

# Supported image extensions
VALID_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")

# Mapping from canonical class names to common dataset subfolder names
# (e.g. Kaggle niharika41298, Yoga-82, custom recorded folders)
FOLDER_ALIAS_MAP = {
    "Warrior II": [
        "warrior2", "warrior_2", "warrior ii", "warrior_ii",
        "virabhadrasana_ii", "virabhadrasana2", "warrior",
    ],
    "Tree": [
        "tree", "tree_pose", "vrksasana", "vrikshasana",
    ],
    "Mountain": [
        "mountain", "mountain_pose", "tadasana", "mountainpose",
    ],
    "Standing": [
        "standing", "standing_pose", "samasthiti", "neutral_standing",
    ],
    "Sitting": [
        "sitting", "seated", "sukhasana", "chair", "sitting_pose",
    ],
}


def find_class_subfolders(data_dir):
    """
    Scans data_dir and matches directories against canonical class names.
    Returns:
        dict mapping canonical_class -> list of image filepaths.
    """
    found_data = {}
    if not os.path.exists(data_dir):
        return found_data

    # List subdirectories in data_dir
    entries = os.listdir(data_dir)
    subdirs = [d for d in entries if os.path.isdir(os.path.join(data_dir, d))]

    for canonical_name, aliases in FOLDER_ALIAS_MAP.items():
        matched_files = []
        matched_dir_name = None

        # Check exact canonical name match first
        for d in subdirs:
            d_clean = d.strip().lower()
            if d_clean == canonical_name.lower() or d_clean in aliases:
                matched_dir_name = d
                dir_path = os.path.join(data_dir, d)
                for ext in VALID_EXTENSIONS:
                    matched_files.extend(glob.glob(os.path.join(dir_path, ext)))
                    matched_files.extend(glob.glob(os.path.join(dir_path, ext.upper())))
                break

        if matched_files:
            found_data[canonical_name] = {
                "folder": matched_dir_name,
                "files": sorted(list(set(matched_files))),
            }

    return found_data


def evaluate_dataset(data_dir, model_dir="models", no_rules=False,
                     mountain_def=DEFAULT_MOUNTAIN_DEF, confidence_threshold=0.5):
    """
    Runs full evaluation on images in data_dir.
    """
    print("=" * 70)
    print(" EVALUATION ON REAL YOGA POSE IMAGES")
    print(f" Dataset Root:     '{data_dir}'")
    print(f" Classification:   {'PURE ML ONLY (--no_rules active)' if no_rules else 'HYBRID (ML + Biomechanical Rules)'}")
    print(f" Mountain Pose:    '{mountain_def}'")
    print("=" * 70)

    # 1. Discover data
    class_data = find_class_subfolders(data_dir)

    print("\n--- Dataset Discovery & Verification ---")
    all_target_classes = SCORED_ASANAS + ["Standing", "Sitting"]
    available_classes = []

    for c in all_target_classes:
        if c in class_data and len(class_data[c]["files"]) > 0:
            count = len(class_data[c]["files"])
            folder = class_data[c]["folder"]
            print(f"  [+] Class '{c}': FOUND {count} images in '{folder}/'")
            available_classes.append(c)
        else:
            print(f"  [-] Class '{c}': NO REAL IMAGES FOUND (SKIPPED — Not fabricated)")

    if not available_classes:
        print("\n[!] Error: No matching image folders found in data_dir.")
        print(f"    Expected folder names matching: {list(FOLDER_ALIAS_MAP.keys())}")
        print("    Example structure:")
        print(f"      {data_dir}/warrior2/*.jpg")
        print(f"      {data_dir}/tree/*.jpg")
        print(f"      {data_dir}/mountain/*.jpg (Optional - custom or Yoga-82)")
        return

    # 2. Load model pipeline
    print("\n[*] Loading ML pipeline...")
    model, scaler, label_encoder, feature_cols = load_pipeline(model_dir)
    detector = PoseDetector(min_detection_confidence=confidence_threshold)

    y_true = []
    y_pred = []
    no_person_per_class = {c: 0 for c in available_classes}
    total_images = 0

    print("\n[*] Running pose estimation & inference across real images...")

    for cls_name in available_classes:
        file_list = class_data[cls_name]["files"]
        print(f"  -> Processing '{cls_name}' ({len(file_list)} images)...")

        for fpath in file_list:
            total_images += 1
            frame = cv2.imread(fpath)
            if frame is None:
                continue

            landmarks_xyz, landmarks_list = detector.detect(frame)
            if landmarks_xyz is None:
                no_person_per_class[cls_name] += 1
                continue

            pred_label, conf, _ = predict_pose(
                landmarks_xyz, model, scaler, label_encoder, feature_cols,
                no_rules=no_rules, mountain_def=mountain_def
            )

            y_true.append(cls_name)
            y_pred.append(pred_label)

    detected_count = len(y_true)
    total_undetected = sum(no_person_per_class.values())

    print("\n" + "=" * 70)
    print(" EVALUATION RESULTS & HONEST METRICS REPORT")
    print("=" * 70)
    print(f"Total Images Evaluated:       {total_images}")
    print(f"Person Successfully Detected: {detected_count} ({detected_count / total_images * 100:.1f}%)" if total_images else "")
    print(f"No Person Detected (Failed):  {total_undetected} ({total_undetected / total_images * 100:.1f}%)" if total_images else "")
    print("\n--- Failure to Detect Person by Class ---")
    for c in available_classes:
        n_fail = no_person_per_class[c]
        n_total = len(class_data[c]["files"])
        print(f"  {c:<12}: {n_fail:>3} / {n_total:<3} undetected ({n_fail / n_total * 100:.1f}%)")

    if detected_count == 0:
        print("[!] No persons were detected in the dataset. Cannot compute classification metrics.")
        return

    # 3. Accuracy & Classification Report
    acc = accuracy_score(y_true, y_pred)
    print(f"\nOverall Classification Accuracy (on detected images): {acc * 100:.2f}%")
    print(f"Pipeline Mode: {'PURE ML (No Rules)' if no_rules else 'HYBRID (ML + Biomechanical Rules)'}")

    all_possible_labels = list(label_encoder.classes_)
    # Include any predicted labels not in label_encoder classes if any
    unique_labels = sorted(list(set(y_true + y_pred)))

    print("\n--- Detailed Classification Metrics ---")
    print(classification_report(y_true, y_pred, labels=unique_labels, zero_division=0))

    # 4. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    cm_df = pd.DataFrame(cm, index=[f"Actual {l}" for l in unique_labels],
                         columns=[f"Pred {l}" for l in unique_labels])
    print("\n--- Confusion Matrix ---")
    print(cm_df.to_string())

    # 5. Mountain vs. Standing Confusion Focus (Feature 4)
    print("\n--- Mountain vs. Standing Confusion Pair Analysis ---")
    has_mountain = "Mountain" in available_classes
    has_standing = "Standing" in available_classes

    if has_mountain or has_standing:
        # Check actual Mountain predicted as Standing
        mtn_as_standing = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "Mountain" and yp == "Standing")
        total_mtn = sum(1 for yt in y_true if yt == "Mountain")
        # Check actual Standing predicted as Mountain
        standing_as_mtn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "Standing" and yp == "Mountain")
        total_standing = sum(1 for yt in y_true if yt == "Standing")

        if total_mtn > 0:
            print(f"  Actual Mountain -> Predicted Standing: {mtn_as_standing} / {total_mtn} ({mtn_as_standing / total_mtn * 100:.1f}%)")
        else:
            print("  Actual Mountain images: Not present in dataset.")

        if total_standing > 0:
            print(f"  Actual Standing -> Predicted Mountain: {standing_as_mtn} / {total_standing} ({standing_as_mtn / total_standing * 100:.1f}%)")
        else:
            print("  Actual Standing images: Not present in dataset.")
    else:
        print("  Neither Mountain nor Standing classes were present in the provided dataset directory.")
        print("  To evaluate Mountain vs Standing, provide images via:")
        print(f"    {data_dir}/mountain/ and {data_dir}/standing/")

    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Yoga Pose Classifier on Real Images")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to folder containing class subdirectories (e.g. data/warrior2/*.jpg)",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models",
        help="Directory containing trained model artifacts (default: models)",
    )
    parser.add_argument(
        "--no_rules",
        action="store_true",
        help="Run pure ML inference without heuristic overrides",
    )
    parser.add_argument(
        "--mountain_def",
        type=str,
        default=DEFAULT_MOUNTAIN_DEF,
        choices=MOUNTAIN_DEFINITIONS,
        help="Mountain pose definition variant: 'arms_overhead' or 'arms_at_sides'",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.5,
        help="Pose detector confidence threshold (default: 0.5)",
    )
    args = parser.parse_args()

    evaluate_dataset(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        no_rules=args.no_rules,
        mountain_def=args.mountain_def,
        confidence_threshold=args.confidence_threshold,
    )


if __name__ == "__main__":
    main()
