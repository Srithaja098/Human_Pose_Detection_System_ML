"""
Deviation Analysis & Sitting Robustness Evaluation Engine (Headless CLI)
======================================================================
Evaluates posture deviations across increasing mistake severities for active
asanas (Warrior II, Tree, Mountain) and runs Sitting robustness + reverse
false-positive checks for Sitting.

Usage:
    python deviation_analysis.py --mode simulate --iterations 10 --poses all --seed 42
    python deviation_analysis.py --mode captured --iterations 10 --poses "Warrior II" Tree Mountain Sitting
"""

import os
import sys
import re
import glob
import math
import time
import warnings
import argparse
from datetime import datetime
from types import SimpleNamespace
from typing import Dict, List, Tuple, Optional, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2

warnings.filterwarnings("ignore", module="sklearn")

from train_yoga_pose_classifier import (
    PoseDetector,
    extract_features,
    FEATURE_COLUMNS,
    LM,
)
from model import (
    load_pipeline,
    predict_pose,
    generate_pose_feedback,
    detect_sitting_or_partial,
    draw_skeleton,
)
from mistake_catalog import (
    ACTIVE_POSES,
    ACTIVE_NON_POSES,
    ACTIVE_CLASSES,
    get_mistakes_for_pose,
    get_sitting_variations,
)
from expert_reference import (
    build_reference_profiles,
    select_best_variant,
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
)


SHORT_JOINT_LABELS: Dict[str, str] = {
    "left_elbow_angle": "L Elbow",
    "right_elbow_angle": "R Elbow",
    "left_shoulder_angle": "L Shldr",
    "right_shoulder_angle": "R Shldr",
    "left_hip_angle": "L Hip",
    "right_hip_angle": "R Hip",
    "left_knee_angle": "L Knee",
    "right_knee_angle": "R Knee",
    "left_ankle_angle": "L Ankle",
    "right_ankle_angle": "R Ankle",
    "torso_inclination": "Torso",
}


def classify_severity_label(abs_dev: float) -> str:
    """Maps an absolute joint deviation in degrees to a severity category."""
    if abs_dev < 10.0:
        return "Good"
    if abs_dev <= 20.0:
        return "Minor"
    if abs_dev <= 35.0:
        return "Moderate"
    return "Major"


def classify_angles_ml(
    angles: Dict[str, float],
    model: Any,
    scaler: Any,
    label_encoder: Any,
    feature_cols: List[str],
) -> Tuple[str, float]:
    """
    Runs the loaded ML classifier pipeline on an 11-angle feature dictionary.
    """
    feat_vec = np.array([[float(angles[c]) for c in feature_cols]], dtype=np.float32)
    feat_scaled = scaler.transform(feat_vec)
    pred_idx = model.predict(feat_scaled)[0]
    label = str(label_encoder.inverse_transform([pred_idx])[0])
    if hasattr(model, "predict_proba"):
        conf = float(np.max(model.predict_proba(feat_scaled)[0]))
    else:
        conf = 1.0
    return label, conf


def compute_deviation_metrics(
    user_angles: Dict[str, float],
    pose_profile: Dict[str, Any],
    score_tau: float = 20.0,
) -> Dict[str, Any]:
    """
    Computes per-joint and overall deviation metrics against the best-matching
    variant in `pose_profile` (so mirrored poses or chair/floor variants are
    never penalised).
    """
    variant_name, ref_feats = select_best_variant(user_angles, pose_profile)

    per_joint: List[Dict[str, Any]] = []
    abs_devs: List[float] = []
    sq_devs: List[float] = []

    for feat in FEATURE_COLUMNS:
        ref_mean = float(ref_feats[feat]["mean"])
        ref_std = float(ref_feats[feat]["std"])
        u_val = float(user_angles[feat])

        signed_dev = u_val - ref_mean
        abs_dev = abs(signed_dev)
        denom_std = max(ref_std, 2.0)
        z_score = abs_dev / denom_std
        pct_dev = (abs_dev / ref_mean * 100.0) if abs(ref_mean) > 1e-6 else 0.0
        sev_label = classify_severity_label(abs_dev)

        abs_devs.append(abs_dev)
        sq_devs.append(signed_dev ** 2)
        per_joint.append(
            {
                "feature": feat,
                "reference_mean": round(ref_mean, 4),
                "reference_std": round(ref_std, 4),
                "user_angle": round(u_val, 4),
                "signed_dev": round(signed_dev, 4),
                "abs_dev": round(abs_dev, 4),
                "z_score": round(z_score, 4),
                "pct_dev": round(pct_dev, 4),
                "severity_label": sev_label,
            }
        )

    mae = float(np.mean(abs_devs))
    rmse = float(np.sqrt(np.mean(sq_devs)))
    max_idx = int(np.argmax(abs_devs))
    max_dev = float(abs_devs[max_idx])
    worst_joint = FEATURE_COLUMNS[max_idx]
    score = float(100.0 * math.exp(-mae / max(score_tau, 1e-6)))

    return {
        "variant_used": variant_name,
        "per_joint": per_joint,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "max_dev": round(max_dev, 4),
        "worst_joint": worst_joint,
        "score": round(score, 4),
    }


def _build_synthetic_landmarks(
    mid_hip_y: float = 0.55,
    knee_y: float = 0.74,
    ankle_y: float = 0.92,
    knee_vis: float = 0.95,
    ankle_vis: float = 0.95,
) -> Tuple[np.ndarray, List[SimpleNamespace]]:
    """
    Creates minimal (33, 3) landmark coordinates and visibility objects for
    calling detect_sitting_or_partial() in simulate mode.
    """
    xyz = np.zeros((33, 3), dtype=np.float32)
    xyz[:, 0] = 0.5
    xyz[:, 1] = 0.35

    xyz[LM["NOSE"]] = [0.5, max(0.08, mid_hip_y - 0.42), 0.0]
    xyz[LM["LEFT_SHOULDER"]] = [0.43, max(0.15, mid_hip_y - 0.28), 0.0]
    xyz[LM["RIGHT_SHOULDER"]] = [0.57, max(0.15, mid_hip_y - 0.28), 0.0]
    xyz[LM["LEFT_HIP"]] = [0.45, mid_hip_y, 0.0]
    xyz[LM["RIGHT_HIP"]] = [0.55, mid_hip_y, 0.0]
    xyz[LM["LEFT_KNEE"]] = [0.45, knee_y, 0.0]
    xyz[LM["RIGHT_KNEE"]] = [0.55, knee_y, 0.0]
    xyz[LM["LEFT_ANKLE"]] = [0.45, ankle_y, 0.0]
    xyz[LM["RIGHT_ANKLE"]] = [0.55, ankle_y, 0.0]
    xyz[LM["LEFT_FOOT_INDEX"]] = [0.45, ankle_y + 0.03, 0.0]
    xyz[LM["RIGHT_FOOT_INDEX"]] = [0.55, ankle_y + 0.03, 0.0]

    lm_list = [SimpleNamespace(visibility=0.98) for _ in range(33)]
    lm_list[LM["LEFT_KNEE"]].visibility = knee_vis
    lm_list[LM["RIGHT_KNEE"]].visibility = knee_vis
    lm_list[LM["LEFT_ANKLE"]].visibility = ankle_vis
    lm_list[LM["RIGHT_ANKLE"]].visibility = ankle_vis
    return xyz, lm_list


def _check_feedback_hit(tip: str, mistake: Dict[str, Any]) -> bool:
    """
    Checks whether the feedback tip addresses the injected mistake using
    simple keyword matching (ignoring generic 'Great posture alignment' praise).
    """
    if not tip:
        return False
    tip_lower = tip.lower()
    if tip_lower.startswith("great posture alignment"):
        return False
    keywords = mistake.get("feedback_keywords", [])
    return any(kw.lower() in tip_lower for kw in keywords)


def _find_captured_files(mistake_dir: str, pose_name: str, iterations: int) -> Dict[int, str]:
    """
    Scans `<mistake_dir>/<pose_name>/` for files matching `iter_XX_*` or
    fallback ordered media files. Missing directories/files are handled gracefully.
    """
    pose_dir = os.path.join(mistake_dir, pose_name)
    if not os.path.isdir(pose_dir):
        print(f"[!] Warning: Missing captured mistake directory '{pose_dir}'.")
        return {}

    valid_exts = IMAGE_EXTENSIONS + VIDEO_EXTENSIONS
    all_files = sorted(
        f
        for f in glob.glob(os.path.join(pose_dir, "*"))
        if os.path.isfile(f) and os.path.splitext(f)[1].lower() in valid_exts
    )
    if not all_files:
        print(f"[!] Warning: No image/video files found in '{pose_dir}'.")
        return {}

    matched: Dict[int, str] = {}
    for fpath in all_files:
        m = re.match(r"iter_(\d+)_", os.path.basename(fpath), re.IGNORECASE)
        if m:
            idx = int(m.group(1))
            if 1 <= idx <= iterations and idx not in matched:
                matched[idx] = fpath

    # If files weren't prefixed with iter_XX_, assign sequentially
    if not matched:
        for i, fpath in enumerate(all_files[:iterations], start=1):
            matched[i] = fpath

    return matched


def _extract_captured_sample(
    fpath: str,
    detector: PoseDetector,
) -> Tuple[Optional[Dict[str, float]], Optional[np.ndarray], Optional[list], Optional[np.ndarray]]:
    """
    Extracts angle dict, landmarks, and representative frame from an image or video.
    Returns (None, None, None, None) if no person is detected.
    """
    ext = os.path.splitext(fpath)[1].lower()
    if ext in IMAGE_EXTENSIONS:
        frame = cv2.imread(fpath)
        if frame is None:
            return None, None, None, None
        xyz, lm_list = detector.detect(frame)
        if xyz is None:
            return None, None, None, None
        return extract_features(xyz), xyz, lm_list, frame

    if ext in VIDEO_EXTENSIONS:
        cap = cv2.VideoCapture(fpath)
        if not cap.isOpened():
            return None, None, None, None
        feats_list: List[Dict[str, float]] = []
        rep_xyz: Optional[np.ndarray] = None
        rep_lm: Optional[list] = None
        rep_frame: Optional[np.ndarray] = None
        idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % 3 == 0:
                    xyz, lm_list = detector.detect(frame)
                    if xyz is not None:
                        feats_list.append(extract_features(xyz))
                        if rep_xyz is None:
                            rep_xyz, rep_lm, rep_frame = xyz, lm_list, frame.copy()
                idx += 1
        finally:
            cap.release()

        if not feats_list:
            return None, None, None, None
        mean_feats = {
            c: float(np.mean([f[c] for f in feats_list]))
            for c in FEATURE_COLUMNS
        }
        return mean_feats, rep_xyz, rep_lm, rep_frame

    return None, None, None, None


def _save_annotated_frame(
    frame: np.ndarray,
    landmarks_xyz: np.ndarray,
    pose_name: str,
    iteration: int,
    mistake_name: str,
    metrics: Dict[str, Any],
    out_dir: str,
) -> None:
    """Draws skeleton and per-joint deviations on a captured frame and saves it."""
    ann_dir = os.path.join(out_dir, "annotated")
    os.makedirs(ann_dir, exist_ok=True)
    canvas = frame.copy()
    draw_skeleton(canvas, landmarks_xyz)

    h, w, _ = canvas.shape
    cv2.rectangle(canvas, (0, 0), (w, 42), (20, 24, 33), -1)
    cv2.putText(
        canvas,
        f"{pose_name} | iter_{iteration:02d} {mistake_name} | MAE={metrics['MAE']:.1f}deg Score={metrics['score']:.1f}",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 230, 150),
        1,
        cv2.LINE_AA,
    )

    y_pos = 64
    for row in metrics["per_joint"]:
        txt = f"{SHORT_JOINT_LABELS[row['feature']]}: {row['user_angle']:.1f} (dev {row['signed_dev']:+.1f}deg)"
        cv2.putText(
            canvas,
            txt,
            (10, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y_pos += 18
        if y_pos > h - 10:
            break

    safe_pose = pose_name.replace(" ", "_").lower()
    safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", mistake_name)
    out_path = os.path.join(ann_dir, f"{safe_pose}_iter_{iteration:02d}_{safe_name}.png")
    cv2.imwrite(out_path, canvas)


def run_deviation_analysis(
    mode: str = "simulate",
    iterations: int = 10,
    poses: Optional[List[str]] = None,
    seed: int = 42,
    expert_dir: str = "expert_data",
    mistake_dir: str = "mistake_data",
    model_dir: str = "models",
    out_dir: str = "results",
    score_tau: float = 20.0,
    noise_std: float = 2.0,
    annotate: bool = False,
) -> Dict[str, Any]:
    """
    Runs the complete deviation analysis and sitting robustness pipeline.
    Returns paths and DataFrames for programmatic inspection and testing.
    """
    requested = list(ACTIVE_CLASSES) if (poses is None or "all" in poses) else list(poses)
    inactive = [p for p in requested if p not in ACTIVE_CLASSES]
    if inactive:
        print(
            f"[!] Inactive pose(s) requested: {inactive}. "
            f"Only active classes are supported: {ACTIVE_CLASSES}. Skipping inactive poses."
        )
    active_selected = [p for p in requested if p in ACTIVE_CLASSES]
    if not active_selected:
        print("[!] No active poses selected. Exiting without generating outputs.")
        return {"status": "no_active_poses"}

    rng = np.random.default_rng(seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_out_dir = os.path.join(out_dir, timestamp)
    os.makedirs(run_out_dir, exist_ok=True)

    # Build/load reference profiles (saved to out_dir and copied into run_out_dir)
    ref_profiles = build_reference_profiles(
        expert_dir=expert_dir,
        out_dir=out_dir,
        poses=active_selected,
        save=True,
    )

    model, scaler, label_encoder, feature_cols = load_pipeline(model_dir)
    detector: Optional[PoseDetector] = PoseDetector(min_detection_confidence=0.5) if mode == "captured" else None

    deviation_rows: List[Dict[str, Any]] = []
    iteration_rows: List[Dict[str, Any]] = []
    sitting_rows: List[Dict[str, Any]] = []

    # =================================================================
    # 1. Asana Mistake-Deviation Iterations (Warrior II, Tree, Mountain)
    # =================================================================
    selected_asanas = [p for p in active_selected if p in ACTIVE_POSES]
    for pose_name in selected_asanas:
        pose_ref = ref_profiles[pose_name]
        mistakes = get_mistakes_for_pose(pose_name, iterations=iterations)
        captured_map = (
            _find_captured_files(mistake_dir, pose_name, iterations)
            if mode == "captured"
            else {}
        )

        # Pick canonical starting variant for simulation (e.g. left_front / left_standing / canonical)
        if "left_front" in pose_ref["variants"]:
            sim_base_variant = "left_front"
        elif "left_standing" in pose_ref["variants"]:
            sim_base_variant = "left_standing"
        else:
            sim_base_variant = sorted(pose_ref["variants"].keys())[0]
        base_feats = pose_ref["variants"][sim_base_variant]["features"]

        for iter_idx, mistake in enumerate(mistakes, start=1):
            severity = round(iter_idx / float(iterations), 4)
            m_name = mistake["name"]

            if mode == "simulate":
                user_angles: Dict[str, float] = {}
                for feat in FEATURE_COLUMNS:
                    base_mean = float(base_feats[feat]["mean"])
                    delta = float(mistake["delta_deg"].get(feat, 0.0)) * severity
                    noise = float(rng.normal(0.0, noise_std)) if noise_std > 0 else 0.0
                    user_angles[feat] = float(np.clip(base_mean + delta + noise, 0.0, 180.0))

                pred_label, conf = classify_angles_ml(
                    user_angles, model, scaler, label_encoder, feature_cols
                )
                xyz_stand, lm_stand = _build_synthetic_landmarks(
                    mid_hip_y=0.55, knee_y=0.74, ankle_y=0.92, knee_vis=0.95, ankle_vis=0.95
                )
                heur_flag, heur_label, _, _ = detect_sitting_or_partial(
                    xyz_stand, lm_stand, user_angles
                )
            else:
                fpath = captured_map.get(iter_idx)
                if not fpath:
                    print(
                        f"[!] Warning: Missing captured file for {pose_name} iter_{iter_idx:02d} ({m_name}). Skipping."
                    )
                    continue
                user_angles, xyz, lm_list, rep_frame = _extract_captured_sample(fpath, detector)
                if user_angles is None or xyz is None:
                    print(
                        f"[!] Warning: No person detected in '{fpath}' ({pose_name} iter_{iter_idx:02d}). Skipping."
                    )
                    continue
                pred_label, conf, _ = predict_pose(
                    xyz, model, scaler, label_encoder, feature_cols
                )
                conf = float(conf) if conf is not None else 0.0
                heur_flag, heur_label, _, _ = detect_sitting_or_partial(
                    xyz, lm_list, user_angles
                )

            metrics = compute_deviation_metrics(user_angles, pose_ref, score_tau=score_tau)

            if mode == "captured" and annotate and rep_frame is not None and xyz is not None:
                _save_annotated_frame(
                    rep_frame, xyz, pose_name, iter_idx, m_name, metrics, run_out_dir
                )

            tip = generate_pose_feedback(pred_label, user_angles)
            feedback_hit = _check_feedback_hit(tip, mistake)
            correct_class = bool(pred_label == pose_name)

            for j_row in metrics["per_joint"]:
                deviation_rows.append(
                    {
                        "class": pose_name,
                        "iteration": iter_idx,
                        "mistake_or_variation_name": m_name,
                        "severity": severity,
                        "variant_used": metrics["variant_used"],
                        **j_row,
                    }
                )

            iteration_rows.append(
                {
                    "class": pose_name,
                    "iteration": iter_idx,
                    "mistake_or_variation_name": m_name,
                    "severity": severity,
                    "variant_used": metrics["variant_used"],
                    "MAE": metrics["MAE"],
                    "RMSE": metrics["RMSE"],
                    "max_dev": metrics["max_dev"],
                    "worst_joint": metrics["worst_joint"],
                    "score": metrics["score"],
                    "predicted_label": pred_label,
                    "confidence": round(conf, 4),
                    "correct_class": correct_class,
                    "tip": tip,
                    "feedback_hit": feedback_hit,
                    "sitting_heuristic_flag": bool(heur_flag),
                    "sitting_heuristic_label": heur_label or "",
                }
            )

    # =================================================================
    # 2. Sitting Robustness Test & Reverse False-Positive Test
    # =================================================================
    if "Sitting" in active_selected:
        sitting_ref = ref_profiles["Sitting"]
        variations = get_sitting_variations(iterations=iterations)
        captured_sit_map = (
            _find_captured_files(mistake_dir, "Sitting", iterations)
            if mode == "captured"
            else {}
        )

        for iter_idx, var in enumerate(variations, start=1):
            v_name = var["name"]
            severity = round(iter_idx / float(iterations), 4)

            if mode == "simulate":
                user_angles = {}
                for feat in FEATURE_COLUMNS:
                    base_val = float(var["angles"][feat])
                    noise = float(rng.normal(0.0, noise_std)) if noise_std > 0 else 0.0
                    user_angles[feat] = float(np.clip(base_val + noise, 0.0, 180.0))

                lp = var["landmark_profile"]
                xyz, lm_list = _build_synthetic_landmarks(
                    mid_hip_y=lp["mid_hip_y"],
                    knee_y=lp["knee_y"],
                    ankle_y=lp["ankle_y"],
                    knee_vis=lp["knee_vis"],
                    ankle_vis=lp["ankle_vis"],
                )
                ml_label, ml_conf = classify_angles_ml(
                    user_angles, model, scaler, label_encoder, feature_cols
                )
            else:
                fpath = captured_sit_map.get(iter_idx)
                if not fpath:
                    print(
                        f"[!] Warning: Missing captured file for Sitting iter_{iter_idx:02d} ({v_name}). Skipping."
                    )
                    continue
                user_angles, xyz, lm_list, rep_frame = _extract_captured_sample(fpath, detector)
                if user_angles is None or xyz is None:
                    print(
                        f"[!] Warning: No person detected in '{fpath}' (Sitting iter_{iter_idx:02d}). Skipping."
                    )
                    continue
                ml_label, ml_conf, _ = predict_pose(
                    xyz, model, scaler, label_encoder, feature_cols, no_rules=True
                )
                ml_conf = float(ml_conf) if ml_conf is not None else 0.0

            heur_flag, heur_label, heur_tip, _ = detect_sitting_or_partial(
                xyz, lm_list, user_angles
            )
            metrics = compute_deviation_metrics(user_angles, sitting_ref, score_tau=score_tau)

            if mode == "captured" and annotate and rep_frame is not None and xyz is not None:
                _save_annotated_frame(
                    rep_frame, xyz, "Sitting", iter_idx, v_name, metrics, run_out_dir
                )

            combined_label = heur_label if heur_flag else ml_label
            ml_is_sitting = ml_label == "Sitting"
            heur_is_sitting = bool(heur_flag)
            combined_is_sitting = bool(heur_is_sitting or ml_is_sitting)
            tip = heur_tip if heur_flag and heur_tip else generate_pose_feedback(ml_label, user_angles)

            for j_row in metrics["per_joint"]:
                deviation_rows.append(
                    {
                        "class": "Sitting",
                        "iteration": iter_idx,
                        "mistake_or_variation_name": v_name,
                        "severity": severity,
                        "variant_used": metrics["variant_used"],
                        **j_row,
                    }
                )

            iteration_rows.append(
                {
                    "class": "Sitting",
                    "iteration": iter_idx,
                    "mistake_or_variation_name": v_name,
                    "severity": severity,
                    "variant_used": metrics["variant_used"],
                    "MAE": metrics["MAE"],
                    "RMSE": metrics["RMSE"],
                    "max_dev": metrics["max_dev"],
                    "worst_joint": metrics["worst_joint"],
                    "score": metrics["score"],
                    "predicted_label": ml_label,
                    "confidence": round(ml_conf, 4),
                    "correct_class": bool(ml_is_sitting),
                    "tip": tip,
                    "feedback_hit": False,
                    "sitting_heuristic_flag": heur_is_sitting,
                    "sitting_heuristic_label": heur_label or "",
                }
            )

            sitting_rows.append(
                {
                    "test_type": "sitting_variation",
                    "intended_class": "Sitting",
                    "iteration": iter_idx,
                    "variation_name": v_name,
                    "variant_used": metrics["variant_used"],
                    "MAE": metrics["MAE"],
                    "score": metrics["score"],
                    "ml_predicted_label": ml_label,
                    "ml_confidence": round(ml_conf, 4),
                    "ml_is_sitting": ml_is_sitting,
                    "heuristic_flagged": heur_is_sitting,
                    "heuristic_label": heur_label or "",
                    "combined_predicted_label": combined_label,
                    "combined_is_sitting": combined_is_sitting,
                }
            )

        # Reverse false-positive test: feed Warrior II, Tree, Mountain reference samples
        for pose_name in ACTIVE_POSES:
            if pose_name not in ref_profiles:
                continue
            for var_name, var_data in ref_profiles[pose_name]["variants"].items():
                ref_angles = {
                    feat: float(var_data["features"][feat]["mean"])
                    for feat in FEATURE_COLUMNS
                }
                xyz_ref, lm_ref = _build_synthetic_landmarks(
                    mid_hip_y=0.55, knee_y=0.74, ankle_y=0.92, knee_vis=0.95, ankle_vis=0.95
                )
                heur_flag, heur_label, _, _ = detect_sitting_or_partial(
                    xyz_ref, lm_ref, ref_angles
                )
                ml_label, ml_conf = classify_angles_ml(
                    ref_angles, model, scaler, label_encoder, feature_cols
                )
                combined_label = heur_label if heur_flag else ml_label
                sitting_rows.append(
                    {
                        "test_type": "standing_reverse_fp",
                        "intended_class": pose_name,
                        "iteration": 0,
                        "variation_name": f"{pose_name}_{var_name}",
                        "variant_used": var_name,
                        "MAE": 0.0,
                        "score": 100.0,
                        "ml_predicted_label": ml_label,
                        "ml_confidence": round(ml_conf, 4),
                        "ml_is_sitting": bool(ml_label == "Sitting"),
                        "heuristic_flagged": bool(heur_flag),
                        "heuristic_label": heur_label or "",
                        "combined_predicted_label": combined_label,
                        "combined_is_sitting": bool(heur_flag or ml_label == "Sitting"),
                    }
                )

    df_dev = pd.DataFrame(deviation_rows)
    df_iter = pd.DataFrame(iteration_rows)
    df_sit = pd.DataFrame(sitting_rows)

    # =================================================================
    # 3. Pose-Level Summary Aggregates (pose_summary.csv)
    # =================================================================
    pose_summary_rows: List[Dict[str, Any]] = []
    for cls_name in active_selected:
        df_c = df_iter[df_iter["class"] == cls_name]
        df_dev_c = df_dev[df_dev["class"] == cls_name]
        source_label = ref_profiles[cls_name]["source"]

        if df_c.empty:
            continue

        mean_mae = float(df_c["MAE"].mean())
        std_mae = float(df_c["MAE"].std(ddof=1)) if len(df_c) > 1 else 0.0
        mean_rmse = float(df_c["RMSE"].mean())
        mean_score = float(df_c["score"].mean())
        clf_acc = float(df_c["correct_class"].mean())

        # Pearson correlation between severity and MAE (asanas only)
        pearson_r: Optional[float] = None
        pearson_p: Optional[float] = None
        if cls_name in ACTIVE_POSES and len(df_c) >= 2:
            sevs = df_c["severity"].astype(float).values
            maes = df_c["MAE"].astype(float).values
            if np.std(sevs) > 1e-9 and np.std(maes) > 1e-9:
                pearson_r = round(float(np.corrcoef(sevs, maes)[0, 1]), 4)
                # Two-sided t-stat approximation for p-value without scipy dependency
                n = len(sevs)
                if n > 2 and abs(pearson_r) < 1.0 - 1e-12:
                    t_stat = abs(pearson_r) * math.sqrt((n - 2) / (1.0 - pearson_r ** 2))
                    pearson_p = round(float(math.erfc(t_stat / math.sqrt(2.0))), 6)
                elif abs(pearson_r) >= 1.0 - 1e-12:
                    pearson_p = 0.0
            else:
                pearson_r = 0.0
                pearson_p = 1.0

        # Most error-sensitive joint
        joint_means = df_dev_c.groupby("feature")["abs_dev"].mean()
        most_sensitive_joint = str(joint_means.idxmax())
        most_sensitive_dev = round(float(joint_means.max()), 4)

        # First severity where classifier stops recognising intended pose
        misclassified = df_c[~df_c["correct_class"]]
        if not misclassified.empty:
            first_row = misclassified.sort_values("iteration").iloc[0]
            first_mis_iter: Optional[int] = int(first_row["iteration"])
            first_mis_sev: Optional[float] = float(first_row["severity"])
        else:
            first_mis_iter = None
            first_mis_sev = None

        hit_rate: Optional[float] = (
            round(float(df_c["feedback_hit"].mean()), 4)
            if cls_name in ACTIVE_POSES
            else None
        )

        pose_summary_rows.append(
            {
                "class": cls_name,
                "source": source_label,
                "n_iterations": int(len(df_c)),
                "mean_MAE": round(mean_mae, 4),
                "std_MAE": round(std_mae, 4),
                "mean_RMSE": round(mean_rmse, 4),
                "mean_score": round(mean_score, 4),
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "most_sensitive_joint": most_sensitive_joint,
                "most_sensitive_joint_dev": most_sensitive_dev,
                "classifier_accuracy": round(clf_acc, 4),
                "first_misclassified_iter": first_mis_iter,
                "first_misclassified_severity": first_mis_sev,
                "feedback_hit_rate": hit_rate,
            }
        )

    df_pose_summary = pd.DataFrame(pose_summary_rows)

    # Save CSV outputs
    dev_csv = os.path.join(run_out_dir, "deviation_results.csv")
    iter_csv = os.path.join(run_out_dir, "iteration_summary.csv")
    pose_csv = os.path.join(run_out_dir, "pose_summary.csv")
    sit_csv = os.path.join(run_out_dir, "sitting_robustness.csv")

    df_dev.to_csv(dev_csv, index=False)
    df_iter.to_csv(iter_csv, index=False)
    df_pose_summary.to_csv(pose_csv, index=False)
    df_sit.to_csv(sit_csv, index=False)

    # =================================================================
    # 4. Generate 300-DPI Report-Ready Plots
    # =================================================================
    plot_paths = _generate_all_plots(
        df_dev=df_dev,
        df_iter=df_iter,
        df_pose_summary=df_pose_summary,
        df_sit=df_sit,
        ref_profiles=ref_profiles,
        selected_asanas=selected_asanas,
        label_encoder=label_encoder,
        out_dir=run_out_dir,
    )

    # =================================================================
    # 5. Generate Academic Summary Report (report.md)
    # =================================================================
    report_path = _generate_report_md(
        run_out_dir=run_out_dir,
        mode=mode,
        iterations=iterations,
        score_tau=score_tau,
        seed=seed,
        ref_profiles=ref_profiles,
        df_dev=df_dev,
        df_iter=df_iter,
        df_pose_summary=df_pose_summary,
        df_sit=df_sit,
        plot_paths=plot_paths,
        model_classes=[str(c) for c in label_encoder.classes_],
    )

    print(f"[+] Analysis complete! Outputs saved in '{run_out_dir}'")
    return {
        "status": "ok",
        "out_dir": run_out_dir,
        "deviation_results_csv": dev_csv,
        "iteration_summary_csv": iter_csv,
        "pose_summary_csv": pose_csv,
        "sitting_robustness_csv": sit_csv,
        "report_md": report_path,
        "plot_paths": plot_paths,
        "df_dev": df_dev,
        "df_iter": df_iter,
        "df_pose_summary": df_pose_summary,
        "df_sit": df_sit,
    }


def _generate_all_plots(
    df_dev: pd.DataFrame,
    df_iter: pd.DataFrame,
    df_pose_summary: pd.DataFrame,
    df_sit: pd.DataFrame,
    ref_profiles: Dict[str, Any],
    selected_asanas: List[str],
    label_encoder: Any,
    out_dir: str,
) -> List[str]:
    """Generates all 8 plot types (300 dpi PNGs) and returns their file paths."""
    saved_plots: List[str] = []
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 9})

    # Plot 1: Heatmaps (1 per active asana)
    for pose_name in selected_asanas:
        sub = df_dev[df_dev["class"] == pose_name]
        if sub.empty:
            continue
        pivot = sub.pivot_table(
            index="iteration",
            columns="feature",
            values="abs_dev",
            aggfunc="mean",
        ).reindex(columns=FEATURE_COLUMNS)

        iter_labels = []
        for it in pivot.index:
            m_name = sub[sub["iteration"] == it]["mistake_or_variation_name"].iloc[0]
            iter_labels.append(f"#{int(it)}: {m_name}")

        fig, ax = plt.subplots(figsize=(12, 5.5))
        im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=max(45.0, float(np.nanmax(pivot.values))))
        ax.set_xticks(range(len(FEATURE_COLUMNS)))
        ax.set_xticklabels([SHORT_JOINT_LABELS[c] for c in FEATURE_COLUMNS], rotation=30, ha="right")
        ax.set_yticks(range(len(iter_labels)))
        ax.set_yticklabels(iter_labels, fontsize=8)
        ax.set_xlabel("Biomechanical Joint Angle")
        ax.set_ylabel("Mistake Iteration")
        ax.set_title(f"{pose_name} — Per-Joint Absolute Deviation Heatmap (deg)")

        for r in range(pivot.shape[0]):
            for c in range(pivot.shape[1]):
                val = pivot.values[r, c]
                if not np.isnan(val):
                    txt_color = "white" if val > 30.0 else "black"
                    ax.text(c, r, f"{val:.1f}", ha="center", va="center", color=txt_color, fontsize=7)

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Absolute Deviation (deg)")
        fig.tight_layout()
        p_path = os.path.join(out_dir, f"heatmap_{pose_name.lower().replace(' ', '_')}.png")
        fig.savefig(p_path, dpi=300)
        plt.close(fig)
        saved_plots.append(p_path)

    # Plot 2: Line chart — MAE and Posture Score vs Iteration (all asanas)
    if selected_asanas and not df_iter.empty:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
        for pose_name in selected_asanas:
            sub = df_iter[df_iter["class"] == pose_name].sort_values("iteration")
            if sub.empty:
                continue
            ax1.plot(sub["iteration"], sub["MAE"], marker="o", linewidth=2, label=pose_name)
            ax2.plot(sub["iteration"], sub["score"], marker="s", linewidth=2, label=pose_name)

        ax1.set_title("Mean Absolute Error (MAE) vs Mistake Iteration")
        ax1.set_xlabel("Iteration (Increasing Mistake Severity)")
        ax1.set_ylabel("MAE (degrees)")
        ax1.grid(True, linestyle="--", alpha=0.4)
        ax1.legend()

        ax2.set_title("Posture Score (0-100) vs Mistake Iteration")
        ax2.set_xlabel("Iteration (Increasing Mistake Severity)")
        ax2.set_ylabel("Posture Score")
        ax2.set_ylim(0, 105)
        ax2.grid(True, linestyle="--", alpha=0.4)
        ax2.legend()

        fig.tight_layout()
        p_path = os.path.join(out_dir, "mae_score_vs_iteration.png")
        fig.savefig(p_path, dpi=300)
        plt.close(fig)
        saved_plots.append(p_path)

    # Plot 3: Radar chart — Reference vs Mild Mistake vs Severe Mistake (per asana)
    angles_rad = np.linspace(0, 2 * np.pi, len(FEATURE_COLUMNS), endpoint=False).tolist()
    angles_rad += angles_rad[:1]
    joint_names_short = [SHORT_JOINT_LABELS[c] for c in FEATURE_COLUMNS]

    for pose_name in selected_asanas:
        sub_dev = df_dev[df_dev["class"] == pose_name]
        sub_iter = df_iter[df_iter["class"] == pose_name].sort_values("iteration")
        if sub_dev.empty or sub_iter.empty:
            continue

        mild_iter = int(sub_iter["iteration"].iloc[min(1, len(sub_iter) - 1)])
        severe_iter = int(sub_iter["iteration"].iloc[-1])

        mild_df = sub_dev[sub_dev["iteration"] == mild_iter].set_index("feature")
        severe_df = sub_dev[sub_dev["iteration"] == severe_iter].set_index("feature")

        ref_vals = [float(severe_df.loc[c, "reference_mean"]) for c in FEATURE_COLUMNS]
        mild_vals = [float(mild_df.loc[c, "user_angle"]) for c in FEATURE_COLUMNS]
        sev_vals = [float(severe_df.loc[c, "user_angle"]) for c in FEATURE_COLUMNS]

        ref_vals += ref_vals[:1]
        mild_vals += mild_vals[:1]
        sev_vals += sev_vals[:1]

        fig, ax = plt.subplots(figsize=(6.8, 6.4), subplot_kw=dict(polar=True))
        ax.plot(angles_rad, ref_vals, linewidth=2, label="Reference Mean", color="#2ca02c")
        ax.fill(angles_rad, ref_vals, alpha=0.10, color="#2ca02c")

        mild_name = sub_iter[sub_iter["iteration"] == mild_iter]["mistake_or_variation_name"].iloc[0]
        sev_name = sub_iter[sub_iter["iteration"] == severe_iter]["mistake_or_variation_name"].iloc[0]

        ax.plot(angles_rad, mild_vals, linewidth=2, linestyle="--", label=f"Mild (#{mild_iter}: {mild_name})", color="#ff7f0e")
        ax.plot(angles_rad, sev_vals, linewidth=2, linestyle=":", label=f"Severe (#{severe_iter}: {sev_name})", color="#d62728")

        ax.set_xticks(angles_rad[:-1])
        ax.set_xticklabels(joint_names_short, fontsize=8)
        ax.set_ylim(0, 190)
        ax.set_title(f"{pose_name} — Joint Profile: Reference vs Mild vs Severe", pad=18)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.12), fontsize=8)
        fig.tight_layout()
        p_path = os.path.join(out_dir, f"radar_{pose_name.lower().replace(' ', '_')}.png")
        fig.savefig(p_path, dpi=300)
        plt.close(fig)
        saved_plots.append(p_path)

    # Plot 4: Grouped bar chart — Mean absolute deviation per joint across iterations
    if selected_asanas and not df_dev.empty:
        sub_asanas = df_dev[df_dev["class"].isin(selected_asanas)]
        if not sub_asanas.empty:
            grouped = (
                sub_asanas.groupby(["feature", "class"])["abs_dev"]
                .mean()
                .unstack(fill_value=0.0)
                .reindex(FEATURE_COLUMNS)
            )
            x = np.arange(len(FEATURE_COLUMNS))
            n_poses = len(selected_asanas)
            width = 0.8 / max(1, n_poses)

            fig, ax = plt.subplots(figsize=(12, 5.2))
            for idx, pose_name in enumerate(selected_asanas):
                vals = grouped[pose_name].values if pose_name in grouped.columns else np.zeros(len(FEATURE_COLUMNS))
                offset = (idx - (n_poses - 1) / 2.0) * width
                ax.bar(x + offset, vals, width=width, label=pose_name)

            ax.set_xticks(x)
            ax.set_xticklabels([SHORT_JOINT_LABELS[c] for c in FEATURE_COLUMNS], rotation=28, ha="right")
            ax.set_xlabel("Biomechanical Joint")
            ax.set_ylabel("Mean Absolute Deviation (deg)")
            ax.set_title("Mean Absolute Deviation per Joint Across All Iterations")
            ax.grid(True, axis="y", linestyle="--", alpha=0.4)
            ax.legend()
            fig.tight_layout()
            p_path = os.path.join(out_dir, "joint_mean_deviation_bar.png")
            fig.savefig(p_path, dpi=300)
            plt.close(fig)
            saved_plots.append(p_path)

    # Plot 5: Scatter — Mistake Severity vs MAE with regression line + Pearson r
    if selected_asanas and not df_iter.empty:
        n_cols = len(selected_asanas)
        fig, axes = plt.subplots(1, n_cols, figsize=(5.2 * n_cols, 4.5), squeeze=False)
        for idx, pose_name in enumerate(selected_asanas):
            ax = axes[0, idx]
            sub = df_iter[df_iter["class"] == pose_name].sort_values("severity")
            if sub.empty:
                continue
            sx = sub["severity"].astype(float).values
            sy = sub["MAE"].astype(float).values
            ax.scatter(sx, sy, color="#1f77b4", s=55, zorder=3, label="Iterations")
            if len(sx) >= 2 and np.std(sx) > 1e-9:
                slope, intercept = np.polyfit(sx, sy, 1)
                x_line = np.linspace(sx.min(), sx.max(), 50)
                ax.plot(x_line, slope * x_line + intercept, color="#d62728", linewidth=2, label="Linear Fit")
                r_val = float(np.corrcoef(sx, sy)[0, 1]) if np.std(sy) > 1e-9 else 0.0
            else:
                r_val = 0.0

            ax.set_title(f"{pose_name} (Pearson r = {r_val:.3f})")
            ax.set_xlabel("Mistake Severity (0.1 - 1.0)")
            ax.set_ylabel("MAE (degrees)")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(fontsize=8)

        fig.suptitle("Mistake Severity vs Mean Absolute Error (MAE)", y=1.02)
        fig.tight_layout()
        p_path = os.path.join(out_dir, "severity_vs_mae_scatter.png")
        fig.savefig(p_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved_plots.append(p_path)

    # Plot 6: Classifier Confidence vs Iteration per asana
    if selected_asanas and not df_iter.empty:
        n_cols = len(selected_asanas)
        fig, axes = plt.subplots(1, n_cols, figsize=(5.2 * n_cols, 4.5), squeeze=False)
        for idx, pose_name in enumerate(selected_asanas):
            ax = axes[0, idx]
            sub = df_iter[df_iter["class"] == pose_name].sort_values("iteration")
            if sub.empty:
                continue
            ax.plot(sub["iteration"], sub["confidence"], color="#4c72b0", linewidth=1.8, zorder=2)

            correct_sub = sub[sub["correct_class"]]
            wrong_sub = sub[~sub["correct_class"]]
            ax.scatter(correct_sub["iteration"], correct_sub["confidence"], color="#2ca02c", s=60, label="Recognised", zorder=3)
            if not wrong_sub.empty:
                ax.scatter(wrong_sub["iteration"], wrong_sub["confidence"], color="#d62728", marker="X", s=85, label="Misclassified", zorder=4)
                for _, row in wrong_sub.iterrows():
                    ax.annotate(
                        str(row["predicted_label"]),
                        (row["iteration"], row["confidence"]),
                        textcoords="offset points",
                        xytext=(0, 7),
                        ha="center",
                        fontsize=7,
                        color="#d62728",
                    )

            ax.set_title(f"{pose_name} — Classifier Confidence")
            ax.set_xlabel("Mistake Iteration")
            ax.set_ylabel("Confidence")
            ax.set_ylim(0, 1.12)
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(fontsize=8, loc="lower left")

        fig.suptitle("Classifier Confidence & Misclassification Threshold vs Iteration", y=1.02)
        fig.tight_layout()
        p_path = os.path.join(out_dir, "classifier_confidence_vs_iteration.png")
        fig.savefig(p_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved_plots.append(p_path)

    # Plot 7: Confusion Matrix (Intended Active Class vs Predicted Model Classes)
    if not df_iter.empty:
        intended_classes = [c for c in ACTIVE_CLASSES if c in df_iter["class"].unique()]
        pred_classes = [str(c) for c in label_encoder.classes_]
        extra_preds = [p for p in df_iter["predicted_label"].unique() if p not in pred_classes]
        pred_classes = pred_classes + extra_preds

        cm = np.zeros((len(intended_classes), len(pred_classes)), dtype=int)
        for r_idx, true_cls in enumerate(intended_classes):
            sub = df_iter[df_iter["class"] == true_cls]
            for c_idx, pred_cls in enumerate(pred_classes):
                cm[r_idx, c_idx] = int((sub["predicted_label"] == pred_cls).sum())

        fig, ax = plt.subplots(figsize=(max(7.5, len(pred_classes) * 1.25), 5.0))
        im = ax.imshow(cm, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(pred_classes)))
        ax.set_xticklabels(pred_classes, rotation=25, ha="right")
        ax.set_yticks(range(len(intended_classes)))
        ax.set_yticklabels(intended_classes)
        ax.set_xlabel("Predicted Classifier Class")
        ax.set_ylabel("Intended Active Class")
        ax.set_title("Confusion Matrix Across All Mistake / Variation Iterations")

        thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                color = "white" if cm[r, c] > thresh else "black"
                ax.text(c, r, str(cm[r, c]), ha="center", va="center", color=color, fontsize=10, fontweight="bold")

        fig.colorbar(im, ax=ax, label="Iteration Count")
        fig.tight_layout()
        p_path = os.path.join(out_dir, "confusion_matrix.png")
        fig.savefig(p_path, dpi=300)
        plt.close(fig)
        saved_plots.append(p_path)

    # Plot 8: Bar chart of Sitting detection rate (ML vs Heuristic vs Combined + Standing FPR)
    if not df_sit.empty:
        sit_vars = df_sit[df_sit["test_type"] == "sitting_variation"]
        rev_fps = df_sit[df_sit["test_type"] == "standing_reverse_fp"]

        ml_rate = float(sit_vars["ml_is_sitting"].mean() * 100.0) if not sit_vars.empty else 0.0
        heur_rate = float(sit_vars["heuristic_flagged"].mean() * 100.0) if not sit_vars.empty else 0.0
        comb_rate = float(sit_vars["combined_is_sitting"].mean() * 100.0) if not sit_vars.empty else 0.0
        fp_rate = float(rev_fps["heuristic_flagged"].mean() * 100.0) if not rev_fps.empty else 0.0

        labels = ["ML Classifier\nAlone", "Heuristic\nAlone", "Combined\n(Heuristic -> ML)", "Standing Reverse\nFalse-Positive Rate"]
        values = [ml_rate, heur_rate, comb_rate, fp_rate]
        colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]

        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        bars = ax.bar(labels, values, color=colors, width=0.55)
        ax.set_ylim(0, 112)
        ax.set_ylabel("Rate (%)")
        ax.set_title("Sitting Detection Rate (10 Variations) & Standing Reverse False-Positive Rate")
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                val + 2.5,
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        fig.tight_layout()
        p_path = os.path.join(out_dir, "sitting_detection_rate.png")
        fig.savefig(p_path, dpi=300)
        plt.close(fig)
        saved_plots.append(p_path)

    return saved_plots


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Formats a DataFrame as a GitHub-flavored markdown table without tabulate."""
    if df.empty:
        return ""
    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in df.iterrows():
        vals = []
        for c in df.columns:
            v = row[c]
            if isinstance(v, float):
                vals.append("N/A" if pd.isna(v) else f"{v:.4f}".rstrip("0").rstrip("."))
            elif pd.isna(v):
                vals.append("N/A")
            else:
                vals.append(str(v))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep] + rows)


def _generate_report_md(
    run_out_dir: str,
    mode: str,
    iterations: int,
    score_tau: float,
    seed: int,
    ref_profiles: Dict[str, Any],
    df_dev: pd.DataFrame,
    df_iter: pd.DataFrame,
    df_pose_summary: pd.DataFrame,
    df_sit: pd.DataFrame,
    plot_paths: List[str],
    model_classes: List[str],
) -> str:
    """Generates an academic-style markdown summary report (`report.md`)."""
    placeholder_classes = [
        c for c, v in ref_profiles.items() if v.get("source") == "synthetic_placeholder"
    ]
    recorded_classes = [
        c for c, v in ref_profiles.items() if v.get("source") == "recorded"
    ]

    lines: List[str] = []
    lines.append("# Pose Deviation & Sitting Robustness Analysis Report\n")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
    lines.append(f"**Mode:** `{mode}` | **Iterations per Class:** `{iterations}` | **Score Tau:** `{score_tau}` | **Seed:** `{seed}`  ")
    lines.append(f"**Active Scope:** `{ACTIVE_CLASSES}` (Asanas: `{ACTIVE_POSES}`; Non-Pose Robustness: `{ACTIVE_NON_POSES}`)  ")
    lines.append(f"**Classifier Model Classes Loaded:** `{model_classes}`\n")

    if placeholder_classes:
        lines.append("> [!WARNING]")
        lines.append(
            f"> **Placeholder Reference Data Notice:** The following classes used `synthetic_placeholder` "
            f"reference distributions derived from `generate_synthetic_pose_dataset()`: "
            f"**{', '.join(placeholder_classes)}**. These preliminary results must be re-run after "
            f"project members record live reference data via `record_reference.py`.\n"
        )

    lines.append("## 1. Methodology & Reference Data Sources\n")
    lines.append(
        "- Ground-truth reference profiles represent poses performed by project members "
        "(called **reference data**; no external certified yoga experts were used).\n"
        "- Both reference profiles and mistake/variation inputs use the identical 11-angle "
        "`extract_features()` representation from `train_yoga_pose_classifier.py`.\n"
        "- Mirror symmetry (left/right variants for Warrior II and Tree) and Sitting variants "
        "(chair vs floor) are stored separately; deviations are evaluated against the lowest-deviation variant.\n"
    )

    lines.append("### Data Source Breakdown\n")
    lines.append("| Class | Source | Samples | People | Variants |")
    lines.append("|---|---|---:|---:|---|")
    for cls_name, pdata in ref_profiles.items():
        vars_str = ", ".join(pdata.get("variants", {}).keys())
        lines.append(
            f"| {cls_name} | `{pdata['source']}` | {pdata['n_samples']} | {pdata['n_people']} | {vars_str} |"
        )
    lines.append("")

    lines.append("## 2. Pose-Level Summary Metrics\n")
    if not df_pose_summary.empty:
        lines.append(_df_to_markdown(df_pose_summary))
        lines.append("")

    lines.append("## 3. Sitting Robustness & Reverse False-Positive Analysis\n")
    if not df_sit.empty:
        sit_vars = df_sit[df_sit["test_type"] == "sitting_variation"]
        rev_fps = df_sit[df_sit["test_type"] == "standing_reverse_fp"]
        if not sit_vars.empty:
            ml_rate = sit_vars["ml_is_sitting"].mean() * 100.0
            heur_rate = sit_vars["heuristic_flagged"].mean() * 100.0
            comb_rate = sit_vars["combined_is_sitting"].mean() * 100.0
            lines.append(f"- **ML Classifier Alone Sitting Detection Rate:** {ml_rate:.1f}% ({int(sit_vars['ml_is_sitting'].sum())}/{len(sit_vars)})")
            lines.append(f"- **Heuristic (`detect_sitting_or_partial`) Alone Detection Rate:** {heur_rate:.1f}% ({int(sit_vars['heuristic_flagged'].sum())}/{len(sit_vars)})")
            lines.append(f"- **Combined Pipeline (Heuristic First -> ML) Detection Rate:** {comb_rate:.1f}% ({int(sit_vars['combined_is_sitting'].sum())}/{len(sit_vars)})")
        if not rev_fps.empty:
            fp_heur = rev_fps["heuristic_flagged"].mean() * 100.0
            fp_comb = rev_fps["combined_is_sitting"].mean() * 100.0
            lines.append(f"- **Reverse Standing Test False-Positive Rate (Heuristic):** {fp_heur:.1f}% ({int(rev_fps['heuristic_flagged'].sum())}/{len(rev_fps)})")
            lines.append(f"- **Reverse Standing Test False-Positive Rate (Combined):** {fp_comb:.1f}% ({int(rev_fps['combined_is_sitting'].sum())}/{len(rev_fps)})\n")
        lines.append(_df_to_markdown(df_sit))
        lines.append("")

    lines.append("## 4. Key Findings\n")
    if not df_pose_summary.empty:
        asana_sum = df_pose_summary[df_pose_summary["class"].isin(ACTIVE_POSES)]
        if not asana_sum.empty:
            top_mae_row = asana_sum.sort_values("mean_MAE", ascending=False).iloc[0]
            lines.append(
                f"- **Highest Overall Sensitivity:** `{top_mae_row['class']}` showed the largest mean MAE "
                f"({top_mae_row['mean_MAE']:.2f} deg), with `{top_mae_row['most_sensitive_joint']}` as its most "
                f"error-sensitive joint (mean dev {top_mae_row['most_sensitive_joint_dev']:.2f} deg)."
            )
            for _, r in asana_sum.iterrows():
                if pd.notna(r["first_misclassified_iter"]):
                    lines.append(
                        f"- **Classifier Breakdown ({r['class']}):** First stopped recognising `{r['class']}` at "
                        f"iteration {int(r['first_misclassified_iter'])} (severity {r['first_misclassified_severity']:.2f}); "
                        f"overall recognition rate = {r['classifier_accuracy']*100:.1f}%, feedback hit rate = {r['feedback_hit_rate']*100:.1f}%."
                    )
                else:
                    lines.append(
                        f"- **Classifier Stability ({r['class']}):** Maintained recognition across all iterations "
                        f"(100.0%); feedback hit rate = {r['feedback_hit_rate']*100:.1f}%."
                    )
    lines.append("")

    lines.append("## 5. Generated Plots\n")
    for p in plot_paths:
        lines.append(f"- `{os.path.basename(p)}`")
    lines.append("")

    lines.append("## 6. Limitations\n")
    lines.append(
        "- Single-camera 2D monocular projection introduces planar foreshortening on limbs angled toward the camera.\n"
        "- Reference profiles currently marked `synthetic_placeholder` reflect synthetic training means rather than human recordings.\n"
        "- The trained classifier is left untouched; `generate_pose_feedback()` uses threshold rules on a subset of joints, which limits feedback hit rates for poses like Mountain.\n"
    )

    report_path = os.path.join(run_out_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Posture Deviation & Sitting Robustness Analysis")
    parser.add_argument(
        "--mode",
        type=str,
        default="simulate",
        choices=["simulate", "captured"],
        help="Evaluation mode: 'simulate' (perturbs reference) or 'captured' (reads mistake_data/)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of mistake/variation iterations per active class (default: 10)",
    )
    parser.add_argument(
        "--poses",
        nargs="+",
        default=["all"],
        help=f"Poses to analyse ('all' or subset of {ACTIVE_CLASSES})",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for simulate mode (default: 42)")
    parser.add_argument("--expert_dir", type=str, default="expert_data", help="Reference data directory")
    parser.add_argument("--mistake_dir", type=str, default="mistake_data", help="Captured mistake media directory")
    parser.add_argument("--model_dir", type=str, default="models", help="Trained model directory")
    parser.add_argument("--out_dir", type=str, default="results", help="Output root directory")
    parser.add_argument("--score_tau", type=float, default=20.0, help="Decay constant tau for posture score (default: 20)")
    parser.add_argument("--noise_std", type=float, default=2.0, help="Gaussian noise std in simulate mode (default: 2.0)")
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="(Captured mode only) Save frames annotated with skeleton and per-joint deviations",
    )
    args = parser.parse_args()

    run_deviation_analysis(
        mode=args.mode,
        iterations=args.iterations,
        poses=args.poses,
        seed=args.seed,
        expert_dir=args.expert_dir,
        mistake_dir=args.mistake_dir,
        model_dir=args.model_dir,
        out_dir=args.out_dir,
        score_tau=args.score_tau,
        noise_std=args.noise_std,
        annotate=args.annotate,
    )


if __name__ == "__main__":
    main()
