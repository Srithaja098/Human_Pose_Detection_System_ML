"""
Expert Reference Profile Builder
================================
Calculates mean joint angles, standard deviations, and tolerances for
each scored asana from recorded expert frames (expert_angles.csv) or
generates clearly-marked provisional reference profiles.

Tolerance calculation rule:
    tolerance = max(MIN_TOLERANCE_DEG, 2.0 * SD)
    (or DEFAULT_TOLERANCE_DEG = 15.0 deg if SD is unknown or single-frame)

Saved to: reference_profiles.json
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd

from pose_config import (
    SCORED_ASANAS,
    SCORED_JOINTS,
    MIN_TOLERANCE_DEG,
    DEFAULT_TOLERANCE_DEG,
    DEFAULT_REFERENCE_FILE,
    DEFAULT_MOUNTAIN_DEF,
)

# Canonical Starter Reference Definitions (Provisional: True)
# Biomechanically grounded from standard yoga anatomy literature
# (Warrior II on left bent knee; Tree balancing on left leg; Mountain full vertical stretch)
PROVISIONAL_STARTER_DATA = {
    "Warrior II": {
        "description": "Provisional starter reference for Warrior II (Virabhadrasana II)",
        "joints": {
            "left_elbow_angle": 172.0,
            "right_elbow_angle": 172.0,
            "left_shoulder_angle": 90.0,
            "right_shoulder_angle": 90.0,
            "left_hip_angle": 115.0,
            "right_hip_angle": 168.0,
            "left_knee_angle": 95.0,
            "right_knee_angle": 174.0,
            "torso_inclination": 4.0,
        },
    },
    "Tree": {
        "description": "Provisional starter reference for Tree Pose (Vrksasana)",
        "joints": {
            "left_elbow_angle": 170.0,
            "right_elbow_angle": 170.0,
            "left_shoulder_angle": 165.0,
            "right_shoulder_angle": 165.0,
            "left_hip_angle": 172.0,
            "right_hip_angle": 128.0,
            "left_knee_angle": 174.0,
            "right_knee_angle": 70.0,
            "torso_inclination": 3.0,
        },
    },
    "Mountain": {
        "description": "Provisional starter reference for Mountain Pose (Tadasana - arms overhead variant)",
        "joints": {
            "left_elbow_angle": 170.0,
            "right_elbow_angle": 170.0,
            "left_shoulder_angle": 165.0,
            "right_shoulder_angle": 165.0,
            "left_hip_angle": 172.0,
            "right_hip_angle": 172.0,
            "left_knee_angle": 175.0,
            "right_knee_angle": 175.0,
            "torso_inclination": 3.0,
        },
        "arms_at_sides_variant": {
            "left_elbow_angle": 168.0,
            "right_elbow_angle": 168.0,
            "left_shoulder_angle": 20.0,
            "right_shoulder_angle": 20.0,
            "left_hip_angle": 172.0,
            "right_hip_angle": 172.0,
            "left_knee_angle": 175.0,
            "right_knee_angle": 175.0,
            "torso_inclination": 3.0,
        },
    },
}


def compute_tolerance(std):
    """
    Computes tolerance for a joint:
    tolerance = max(10.0 deg, 2.0 * std)
    If std is 0.0, None, or unknown, returns DEFAULT_TOLERANCE_DEG (15.0 deg).
    """
    if std is None or np.isnan(std) or std <= 0.0:
        return float(DEFAULT_TOLERANCE_DEG)
    return float(max(MIN_TOLERANCE_DEG, 2.0 * float(std)))


def generate_starter_profiles(mountain_def=DEFAULT_MOUNTAIN_DEF):
    """
    Generates a starter dictionary for all 3 scored asanas marked with provisional: true.
    """
    profiles = {}
    for asana_name in SCORED_ASANAS:
        raw_info = PROVISIONAL_STARTER_DATA[asana_name]
        joints_dict = raw_info["joints"].copy()

        # If Mountain and arms_at_sides variant requested
        if asana_name == "Mountain" and mountain_def == "arms_at_sides":
            joints_dict = raw_info["arms_at_sides_variant"].copy()

        profile = {
            "provisional": True,
            "description": f"{raw_info['description']} (Marked PROVISIONAL: Not yet replaced by expert recording)",
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "sample_count": 1,
            "joints": {},
        }

        for joint_name in SCORED_JOINTS:
            mean_val = float(joints_dict.get(joint_name, 180.0))
            std_val = 0.0
            tol_val = compute_tolerance(std_val)
            profile["joints"][joint_name] = {
                "mean": round(mean_val, 2),
                "std": round(std_val, 2),
                "tolerance": round(tol_val, 2),
            }

        profiles[asana_name] = profile

    return profiles


def build_profile_from_dataframe(df_asana, asana_name):
    """
    Computes mean, std, and tolerance for each scored joint from a DataFrame.
    """
    sample_count = len(df_asana)
    if sample_count < 1:
        raise ValueError(f"No frames found for asana '{asana_name}'")

    profile = {
        "provisional": False,
        "description": f"Expert reference profile for {asana_name} recorded from live instructor",
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sample_count": int(sample_count),
        "joints": {},
    }

    for joint_name in SCORED_JOINTS:
        if joint_name in df_asana.columns:
            vals = df_asana[joint_name].dropna().values
            if len(vals) == 0:
                mean_val = 0.0
                std_val = 0.0
            else:
                mean_val = float(np.mean(vals))
                std_val = float(np.std(vals)) if len(vals) > 1 else 0.0
        else:
            mean_val = 0.0
            std_val = 0.0

        tol_val = compute_tolerance(std_val)
        profile["joints"][joint_name] = {
            "mean": round(mean_val, 2),
            "std": round(std_val, 2),
            "tolerance": round(tol_val, 2),
        }

    return profile


def load_reference_profiles(reference_file=DEFAULT_REFERENCE_FILE):
    """
    Loads reference_profiles.json from disk.
    If file doesn't exist, generates starter provisional profiles and saves them.
    """
    if not os.path.exists(reference_file):
        print(f"[*] Reference file '{reference_file}' not found. Generating starter provisional profiles...")
        profiles = generate_starter_profiles()
        save_reference_profiles(profiles, reference_file)
        return profiles

    with open(reference_file, "r") as f:
        profiles = json.load(f)
    return profiles


def save_reference_profiles(profiles, reference_file=DEFAULT_REFERENCE_FILE):
    """
    Saves reference profiles dictionary to JSON.
    """
    with open(reference_file, "w") as f:
        json.dump(profiles, f, indent=2)
    print(f"[+] Saved reference profiles to '{reference_file}'")


def build_references_from_csv(csv_path, target_asana=None,
                              output_file=DEFAULT_REFERENCE_FILE):
    """
    Processes expert_angles.csv and updates reference_profiles.json.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Expert angles CSV file not found: '{csv_path}'")

    df = pd.read_csv(csv_path)
    if "asana" not in df.columns:
        raise ValueError(f"CSV must contain 'asana' column. Found: {list(df.columns)}")

    # Load existing or create new profiles
    existing_profiles = {}
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            existing_profiles = json.load(f)

    asanas_to_process = [target_asana] if target_asana else df["asana"].unique().tolist()

    for asana in asanas_to_process:
        df_sub = df[df["asana"] == asana]
        if len(df_sub) == 0:
            print(f"[!] Warning: No frames found in CSV for '{asana}'. Skipping.")
            continue

        print(f"[*] Building expert profile for '{asana}' from {len(df_sub)} valid frames...")
        profile = build_profile_from_dataframe(df_sub, asana)
        existing_profiles[asana] = profile

    save_reference_profiles(existing_profiles, output_file)
    print(f"[+] Successfully updated '{output_file}' with {len(asanas_to_process)} profiles.")


def main():
    parser = argparse.ArgumentParser(description="Build Reference Profiles for Yoga Asanas")
    parser.add_argument(
        "--csv",
        type=str,
        default="expert_angles.csv",
        help="Path to recorded expert CSV file (default: expert_angles.csv)",
    )
    parser.add_argument(
        "--asana",
        type=str,
        default=None,
        help="Specific asana to build reference for (default: all found in CSV)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_REFERENCE_FILE,
        help=f"Path to output JSON file (default: {DEFAULT_REFERENCE_FILE})",
    )
    parser.add_argument(
        "--init_starter",
        action="store_true",
        help="Initialize standard starter profiles (marked provisional: true) for all 3 asanas",
    )
    parser.add_argument(
        "--mountain_def",
        type=str,
        default=DEFAULT_MOUNTAIN_DEF,
        choices=["arms_at_sides", "arms_overhead"],
        help="Mountain pose definition variant for starter profile (default: arms_overhead)",
    )
    args = parser.parse_args()

    if args.init_starter or not os.path.exists(args.csv):
        if not args.init_starter:
            print(f"[*] Notice: CSV '{args.csv}' not found. Initializing provisional starter profiles instead.")
        starter = generate_starter_profiles(mountain_def=args.mountain_def)
        save_reference_profiles(starter, args.output)
    else:
        build_references_from_csv(args.csv, target_asana=args.asana, output_file=args.output)


if __name__ == "__main__":
    main()
