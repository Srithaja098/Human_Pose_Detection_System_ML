"""
Reference Profile Builder & Loader
==================================
Builds ground-truth reference profiles for the 4 active classes
(Warrior II, Tree, Mountain, Sitting) from recorded reference data or
explicitly labelled synthetic placeholder fallbacks.

Input priority per active class:
  1. Images/videos in <expert_dir>/<pose>/ (via PoseDetector + extract_features)
  2. <expert_dir>/expert_angles.csv (columns: pose, person optional, + 11 FEATURE_COLUMNS)
  3. Fallback: canonical mean angles from generate_synthetic_pose_dataset()
     with a clear WARNING and source="synthetic_placeholder".

Saves:
  - <out_dir>/expert_reference.json
  - <out_dir>/expert_reference.csv
"""

import os
import sys
import json
import glob
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import cv2
import numpy as np
import pandas as pd

from train_yoga_pose_classifier import (
    PoseDetector,
    extract_features,
    FEATURE_COLUMNS,
    generate_synthetic_pose_dataset,
)
from mistake_catalog import (
    ACTIVE_CLASSES,
    MIRROR_FEATURE_MAP,
)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def _mirror_angles_dict(angles: Dict[str, float]) -> Dict[str, float]:
    """Returns the left/right-mirrored angle dictionary."""
    return {
        feat: float(angles[MIRROR_FEATURE_MAP.get(feat, feat)])
        for feat in FEATURE_COLUMNS
    }


def _classify_variant(pose_name: str, row: Dict[str, float], filename: str = "") -> str:
    """
    Assigns a variant label to a single sample so left/right mirrored poses
    and chair/floor sitting variants are stored separately.
    """
    fname_lower = filename.lower()
    if pose_name == "Warrior II":
        if "right" in fname_lower:
            return "right_front"
        if "left" in fname_lower:
            return "left_front"
        return (
            "left_front"
            if row["left_knee_angle"] <= row["right_knee_angle"]
            else "right_front"
        )
    if pose_name == "Tree":
        if "right" in fname_lower:
            return "right_standing"
        if "left" in fname_lower:
            return "left_standing"
        return (
            "left_standing"
            if row["left_knee_angle"] >= row["right_knee_angle"]
            else "right_standing"
        )
    if pose_name == "Sitting":
        if "floor" in fname_lower or "sukhasana" in fname_lower:
            return "floor"
        if "chair" in fname_lower:
            return "chair"
        if row["left_knee_angle"] < 70.0 and row["right_knee_angle"] < 70.0:
            return "floor"
        return "chair"
    return "canonical"


def _compute_feature_stats(
    df: pd.DataFrame,
    person_col: str = "person",
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Optional[float]]]:
    """
    Computes mean, std, min, max, n_samples, and between_person_std per feature.
    """
    features_stats: Dict[str, Dict[str, Any]] = {}
    between_person_std: Dict[str, Optional[float]] = {}

    has_multiple_people = (
        person_col in df.columns
        and df[person_col].dropna().astype(str).nunique() > 1
    )

    for feat in FEATURE_COLUMNS:
        vals = df[feat].dropna().astype(float).values
        if len(vals) == 0:
            features_stats[feat] = {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "n_samples": 0,
            }
            between_person_std[feat] = None
            continue

        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        min_val = float(np.min(vals))
        max_val = float(np.max(vals))

        bp_std: Optional[float] = None
        if has_multiple_people:
            person_means = df.groupby(person_col)[feat].mean().dropna().values
            if len(person_means) > 1:
                bp_std = round(float(np.std(person_means, ddof=1)), 4)

        features_stats[feat] = {
            "mean": round(mean_val, 4),
            "std": round(std_val, 4),
            "min": round(min_val, 4),
            "max": round(max_val, 4),
            "n_samples": int(len(vals)),
        }
        between_person_std[feat] = bp_std

    return features_stats, between_person_std


def _extract_from_media_dir(
    pose_dir: str,
    pose_name: str,
    detector: PoseDetector,
    sample_step: int = 5,
) -> pd.DataFrame:
    """
    Extracts 11 angles from all images and videos inside `<expert_dir>/<pose>/`.
    """
    if not os.path.isdir(pose_dir):
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    files = sorted(glob.glob(os.path.join(pose_dir, "*")))

    for fpath in files:
        if not os.path.isfile(fpath):
            continue
        ext = os.path.splitext(fpath)[1].lower()
        base = os.path.basename(fpath)
        person = base.split("_")[0] if "_" in base else "member"

        if ext in IMAGE_EXTENSIONS:
            img = cv2.imread(fpath)
            if img is None:
                continue
            xyz, _ = detector.detect(img)
            if xyz is None:
                continue
            feats = extract_features(xyz)
            variant = _classify_variant(pose_name, feats, base)
            rows.append({"pose": pose_name, "person": person, "variant": variant, **feats})

        elif ext in VIDEO_EXTENSIONS:
            cap = cv2.VideoCapture(fpath)
            if not cap.isOpened():
                continue
            frame_idx = 0
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame_idx % max(1, sample_step) == 0:
                        xyz, _ = detector.detect(frame)
                        if xyz is not None:
                            feats = extract_features(xyz)
                            variant = _classify_variant(pose_name, feats, base)
                            rows.append(
                                {"pose": pose_name, "person": person, "variant": variant, **feats}
                            )
                    frame_idx += 1
            finally:
                cap.release()

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _load_from_angles_csv(csv_path: str, pose_name: str) -> pd.DataFrame:
    """
    Loads recorded samples for `pose_name` from `<expert_dir>/expert_angles.csv`.
    """
    if not os.path.exists(csv_path):
        return pd.DataFrame()

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return pd.DataFrame()

    if df.empty or "pose" not in df.columns:
        return pd.DataFrame()

    missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        print(f"[!] Warning: '{csv_path}' missing columns {missing_cols}; ignoring CSV.")
        return pd.DataFrame()

    df_pose = df[df["pose"].astype(str) == pose_name].copy()
    if df_pose.empty:
        return pd.DataFrame()

    if "person" not in df_pose.columns:
        df_pose["person"] = "member"

    df_pose["variant"] = [
        _classify_variant(pose_name, row) for row in df_pose[FEATURE_COLUMNS].to_dict("records")
    ]
    return df_pose


def _synthetic_placeholder_dataframe(pose_name: str) -> pd.DataFrame:
    """
    Builds placeholder reference samples directly from generate_synthetic_pose_dataset()
    canonical pose distributions, tagged by variant. For bilateral poses (Warrior II,
    Tree), the right-sided variant is generated as the exact bilateral mirror of the
    left-sided samples so left/right variants are strictly symmetric.
    """
    X, y = generate_synthetic_pose_dataset(samples_per_class=300, random_seed=42)
    mask = y == pose_name
    X_pose = X[mask]

    rows: List[Dict[str, Any]] = []
    n_samples = len(X_pose)
    if pose_name in ("Warrior II", "Tree"):
        left_key = "left_front" if pose_name == "Warrior II" else "left_standing"
        right_key = "right_front" if pose_name == "Warrior II" else "right_standing"
        for vec in X_pose[: n_samples // 2]:
            feats = {col: float(vec[j]) for j, col in enumerate(FEATURE_COLUMNS)}
            rows.append({"pose": pose_name, "person": "synthetic_placeholder", "variant": left_key, **feats})
            mirrored = _mirror_angles_dict(feats)
            rows.append({"pose": pose_name, "person": "synthetic_placeholder", "variant": right_key, **mirrored})
        return pd.DataFrame(rows)

    for idx, vec in enumerate(X_pose):
        feats = {col: float(vec[j]) for j, col in enumerate(FEATURE_COLUMNS)}
        if pose_name == "Sitting":
            # Last third of synthetic Sitting specs are floor cross-legged
            variant = "floor" if idx >= (2 * n_samples) // 3 else "chair"
        else:
            variant = "canonical"
        rows.append({"pose": pose_name, "person": "synthetic_placeholder", "variant": variant, **feats})

    return pd.DataFrame(rows)


def _ensure_mirror_variant(pose_name: str, variants_dict: Dict[str, Dict[str, Any]]) -> None:
    """
    For Warrior II and Tree, ensures both left and right mirrored variants exist
    so performing the pose on the opposite side is never penalised as a mistake.
    """
    pairs = {
        "Warrior II": ("left_front", "right_front"),
        "Tree": ("left_standing", "right_standing"),
    }
    if pose_name not in pairs:
        return
    left_key, right_key = pairs[pose_name]
    if left_key in variants_dict and right_key not in variants_dict:
        src = variants_dict[left_key]
        variants_dict[right_key] = {
            "n_samples": src["n_samples"],
            "mirrored_from": left_key,
            "features": {
                feat: dict(src["features"][MIRROR_FEATURE_MAP[feat]])
                for feat in FEATURE_COLUMNS
            },
        }
    elif right_key in variants_dict and left_key not in variants_dict:
        src = variants_dict[right_key]
        variants_dict[left_key] = {
            "n_samples": src["n_samples"],
            "mirrored_from": right_key,
            "features": {
                feat: dict(src["features"][MIRROR_FEATURE_MAP[feat]])
                for feat in FEATURE_COLUMNS
            },
        }


def build_reference_profiles(
    expert_dir: str = "expert_data",
    out_dir: str = "results",
    poses: Optional[List[str]] = None,
    sample_step: int = 5,
    save: bool = True,
) -> Dict[str, Any]:
    """
    Builds reference profiles for the requested active classes following the
    3-tier input priority (media files -> expert_angles.csv -> synthetic placeholder).
    """
    target_poses = poses if poses is not None else list(ACTIVE_CLASSES)
    invalid = [p for p in target_poses if p not in ACTIVE_CLASSES]
    if invalid:
        print(
            f"[!] Ignoring inactive pose(s) {invalid}. "
            f"Active classes are: {ACTIVE_CLASSES}"
        )
        target_poses = [p for p in target_poses if p in ACTIVE_CLASSES]

    if not target_poses:
        return {}

    os.makedirs(out_dir, exist_ok=True)
    csv_angles_path = os.path.join(expert_dir, "expert_angles.csv")
    detector: Optional[PoseDetector] = None

    profiles: Dict[str, Any] = {}
    csv_rows: List[Dict[str, Any]] = []

    for pose_name in target_poses:
        pose_dir = os.path.join(expert_dir, pose_name)
        df_pose = pd.DataFrame()
        source_type = "recorded"

        # Priority 1: Media files in expert_data/<pose>/
        has_media = os.path.isdir(pose_dir) and any(
            os.path.splitext(f)[1].lower() in (IMAGE_EXTENSIONS + VIDEO_EXTENSIONS)
            for f in os.listdir(pose_dir)
            if os.path.isfile(os.path.join(pose_dir, f))
        )
        if has_media:
            if detector is None:
                detector = PoseDetector(min_detection_confidence=0.5)
            df_pose = _extract_from_media_dir(
                pose_dir, pose_name, detector, sample_step=sample_step
            )

        # Priority 2: expert_data/expert_angles.csv
        if df_pose.empty:
            df_pose = _load_from_angles_csv(csv_angles_path, pose_name)

        # Priority 3: Synthetic placeholder fallback
        if df_pose.empty:
            source_type = "synthetic_placeholder"
            print(
                f"[WARNING] No recorded reference data found for '{pose_name}' in "
                f"'{pose_dir}' or '{csv_angles_path}'. Using canonical mean angles from "
                f"generate_synthetic_pose_dataset() as synthetic_placeholder."
            )
            df_pose = _synthetic_placeholder_dataframe(pose_name)

        people = sorted(df_pose["person"].dropna().astype(str).unique().tolist())
        overall_features, between_person_std = _compute_feature_stats(df_pose)

        variants: Dict[str, Any] = {}
        for variant_name, df_var in df_pose.groupby("variant"):
            v_feats, _ = _compute_feature_stats(df_var)
            variants[str(variant_name)] = {
                "n_samples": int(len(df_var)),
                "features": v_feats,
            }

        _ensure_mirror_variant(pose_name, variants)

        profiles[pose_name] = {
            "source": source_type,
            "built_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_samples": int(len(df_pose)),
            "n_people": len(people) if source_type == "recorded" else 0,
            "people": people if source_type == "recorded" else [],
            "between_person_std": between_person_std,
            "features": overall_features,
            "variants": variants,
        }

        for variant_name, v_data in variants.items():
            for feat in FEATURE_COLUMNS:
                fstats = v_data["features"][feat]
                csv_rows.append(
                    {
                        "pose": pose_name,
                        "variant": variant_name,
                        "source": source_type,
                        "feature": feat,
                        "mean": fstats["mean"],
                        "std": fstats["std"],
                        "min": fstats["min"],
                        "max": fstats["max"],
                        "n_samples": fstats["n_samples"],
                        "between_person_std": between_person_std.get(feat),
                    }
                )

    if save:
        json_path = os.path.join(out_dir, "expert_reference.json")
        csv_out_path = os.path.join(out_dir, "expert_reference.csv")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(profiles, f, indent=2)
        pd.DataFrame(csv_rows).to_csv(csv_out_path, index=False)
        print(f"[+] Saved reference profiles to '{json_path}' and '{csv_out_path}'")

    return profiles


def select_best_variant(
    angles: Dict[str, float],
    pose_profile: Dict[str, Any],
) -> Tuple[str, Dict[str, Dict[str, float]]]:
    """
    Selects the stored variant (e.g., left_front vs right_front, or chair vs floor)
    with the lowest total absolute deviation from `angles`.
    """
    variants = pose_profile.get("variants", {})
    if not variants:
        return "canonical", pose_profile["features"]

    best_name = None
    best_total_dev = float("inf")
    best_features = None

    for var_name, var_data in variants.items():
        var_feats = var_data["features"]
        total_dev = sum(
            abs(float(angles[feat]) - float(var_feats[feat]["mean"]))
            for feat in FEATURE_COLUMNS
        )
        if total_dev < best_total_dev:
            best_total_dev = total_dev
            best_name = var_name
            best_features = var_feats

    return str(best_name), best_features


def load_or_build_reference(
    expert_dir: str = "expert_data",
    out_dir: str = "results",
    poses: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Loads `<out_dir>/expert_reference.json` if present and containing all
    requested poses; otherwise builds and saves it.
    """
    json_path = os.path.join(out_dir, "expert_reference.json")
    target_poses = poses if poses is not None else list(ACTIVE_CLASSES)
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if all(p in data for p in target_poses):
                return data
        except Exception:
            pass
    return build_reference_profiles(expert_dir=expert_dir, out_dir=out_dir, poses=target_poses)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Reference Profiles for Active Poses")
    parser.add_argument(
        "--build",
        action="store_true",
        help="Build reference profiles from expert_data/ or synthetic placeholder fallback",
    )
    parser.add_argument(
        "--expert_dir",
        type=str,
        default="expert_data",
        help="Directory containing recorded reference media/CSV (default: expert_data)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results",
        help="Output directory for expert_reference.json/csv (default: results)",
    )
    parser.add_argument(
        "--sample_step",
        type=int,
        default=5,
        help="Sample every Nth frame from reference videos (default: 5)",
    )
    parser.add_argument(
        "--poses",
        nargs="+",
        default=["all"],
        help=f"Poses to build ('all' or subset of {ACTIVE_CLASSES})",
    )
    args = parser.parse_args()

    if not args.build:
        parser.print_help()
        return

    if args.poses == ["all"] or "all" in args.poses:
        selected_poses = list(ACTIVE_CLASSES)
    else:
        selected_poses = args.poses

    invalid = [p for p in selected_poses if p not in ACTIVE_CLASSES]
    if invalid:
        print(
            f"[!] Inactive pose(s) requested: {invalid}. "
            f"Active scope is strictly: {ACTIVE_CLASSES}"
        )
        selected_poses = [p for p in selected_poses if p in ACTIVE_CLASSES]
        if not selected_poses:
            sys.exit(0)

    build_reference_profiles(
        expert_dir=args.expert_dir,
        out_dir=args.out_dir,
        poses=selected_poses,
        sample_step=args.sample_step,
        save=True,
    )


if __name__ == "__main__":
    main()
