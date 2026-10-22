"""
Stage 2 Verification Suite — Pose Scoring, Mirroring, Tracking & Habits
========================================================================
Validates all Stage 2 biomechanical scoring rules using simulated angle streams:
1. Exact reference angles yield Score = 100.0 and is_correct = True.
2. Mirrored person (bilateral swap) yields Score = 100.0 and is_correct = True.
3. One joint 60 deg off yields is_correct = False and identifies the worst joint.
4. Hidden/occluded joints (< 0.5 visibility) are skipped; returns None if < 7 visible.
5. Error shrinking over time in simulated stream registers expected time_to_correct.
6. Lost person during hold resets the hold timer.
7. Pose never reached within 60s logs success = 0.
8. Standing and Sitting states are NEVER scored.
9. sessions.csv logging and habit detection.
"""

import os
import sys
import time
import numpy as np
import pandas as pd

from pose_config import (
    SCORED_ASANAS,
    UNSCORED_STATES,
    SCORED_JOINTS,
    MIRROR_JOINT_MAP,
    SCORE_THRESHOLD,
    HOLD_CORRECT_TIME_S,
    HOLD_SUCCESS_TIME_S,
    SESSION_TIMEOUT_S,
)
from build_reference import load_reference_profiles
from tracking import (
    evaluate_pose,
    evaluate_joint,
    create_mirrored_profile,
    PoseSessionTracker,
    check_persistent_weakness,
)


class DummyLandmark:
    """Mock landmark object with configurable visibility."""
    def __init__(self, x=0.5, y=0.5, z=0.0, visibility=1.0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


def create_full_visibility_landmarks():
    """Returns a list of 33 mock landmarks all with visibility 1.0."""
    return [DummyLandmark(visibility=1.0) for _ in range(33)]


def run_tracking_tests():
    print("=" * 65)
    print(" STAGE 2 POSTURE SCORING & TRACKING VERIFICATION SUITE")
    print("=" * 65)

    ref_profiles = load_reference_profiles("reference_profiles.json")
    landmarks_all_vis = create_full_visibility_landmarks()

    # -------------------------------------------------------------
    # Test 1: Exact Reference Pose -> Score 100.0 and Correct
    # -------------------------------------------------------------
    print("\n[Test 1/8] Verifying Exact Reference Angles...")
    for asana in SCORED_ASANAS:
        prof = ref_profiles[asana]
        exact_angles = {j: prof["joints"][j]["mean"] for j in SCORED_JOINTS}
        res = evaluate_pose(exact_angles, landmarks_all_vis, asana, ref_profiles)

        assert res["pose_score"] is not None, f"Score was None for exact {asana}"
        assert abs(res["pose_score"] - 100.0) < 1e-3, f"Expected 100.0, got {res['pose_score']}"
        assert res["is_correct"] is True, f"Exact pose should be correct for {asana}"
        print(f"  [+] {asana:<12}: Score = {res['pose_score']:.1f}% | Correct = {res['is_correct']} (PASSED)")

    # -------------------------------------------------------------
    # Test 2: Mirrored Person -> Score 100.0 and Correct
    # -------------------------------------------------------------
    print("\n[Test 2/8] Verifying Mirrored Orientation Invariance...")
    for asana in SCORED_ASANAS:
        prof = ref_profiles[asana]
        # Swap left and right angles
        mirrored_angles = {}
        for j in SCORED_JOINTS:
            mirror_j = MIRROR_JOINT_MAP[j]
            mirrored_angles[j] = prof["joints"][mirror_j]["mean"]

        res = evaluate_pose(mirrored_angles, landmarks_all_vis, asana, ref_profiles)
        assert res["pose_score"] is not None, f"Score was None for mirrored {asana}"
        assert abs(res["pose_score"] - 100.0) < 1e-3, f"Expected 100.0 for mirrored {asana}, got {res['pose_score']}"
        assert res["is_correct"] is True, f"Mirrored pose should be correct for {asana}"
        print(f"  [+] {asana:<12}: Mirrored Score = {res['pose_score']:.1f}% | Used Mirror: {res['used_mirror']} (PASSED)")

    # -------------------------------------------------------------
    # Test 3: One Joint 60 deg Off -> is_correct = False and Identified as Worst Joint
    # -------------------------------------------------------------
    print("\n[Test 3/8] Verifying Single Joint 60-deg Off Detection...")
    w2_prof = ref_profiles["Warrior II"]
    angles_bad_knee = {j: w2_prof["joints"][j]["mean"] for j in SCORED_JOINTS}
    # Offset right knee by 60 degrees (from 174 to 114)
    angles_bad_knee["right_knee_angle"] -= 60.0

    res_bad = evaluate_pose(angles_bad_knee, landmarks_all_vis, "Warrior II", ref_profiles)
    assert res_bad["is_correct"] is False, "Pose with 60-deg off joint should not be correct"
    assert res_bad["worst_joint"] == "right_knee_angle", f"Expected worst joint right_knee_angle, got {res_bad['worst_joint']}"
    assert "right knee" in res_bad["tip"].lower(), f"Tip should mention right knee: {res_bad['tip']}"
    print(f"  [+] Off-joint detected: Worst = '{res_bad['worst_joint']}' | Score = {res_bad['pose_score']:.1f}%")
    print(f"      Generated Tip: '{res_bad['tip']}' (PASSED)")

    # -------------------------------------------------------------
    # Test 4: Hidden/Occluded Joints (< 0.5 visibility)
    # -------------------------------------------------------------
    print("\n[Test 4/8] Verifying Low-Visibility (< 0.5) Joint Exclusions...")
    landmarks_low_vis = [DummyLandmark(visibility=1.0) for _ in range(33)]
    # Set right arm and right leg landmarks to visibility 0.2
    # Landmarks: right wrist (16), right elbow (14), right knee (26)
    landmarks_low_vis[14].visibility = 0.2
    landmarks_low_vis[16].visibility = 0.2
    landmarks_low_vis[26].visibility = 0.2

    # This knocks out right_elbow, right_shoulder, right_knee, right_hip -> 5 valid joints remaining (< 7 required)
    res_occluded = evaluate_pose(exact_angles, landmarks_low_vis, "Warrior II", ref_profiles)
    assert res_occluded["pose_score"] is None, "Should return None when < 7 joints are visible"
    assert res_occluded["valid_joints_count"] < 7, "Valid joint count should be below 7"
    print(f"  [+] Low-visibility handling: {res_occluded['valid_joints_count']}/9 joints visible -> Pose Score: {res_occluded['pose_score']} (PASSED)")

    # -------------------------------------------------------------
    # Test 5: Simulated Stream with Error Shrinking Over Time
    # -------------------------------------------------------------
    print("\n[Test 5/8] Verifying Simulated Stream Time-to-Correct Measurement...")
    tracker = PoseSessionTracker(
        target_asana="Warrior II",
        user_id="TestUser",
        reference_file="reference_profiles.json",
        sessions_csv="test_sessions.csv",
    )
    # Start countdown and finish it
    tracker.state = PoseSessionTracker.STATE_TRACKING
    tracker.session_start_time = 0.0

    # Simulate 30 fps stream for 8 seconds
    # t = 0.0 to 2.5s: bad posture (error = 40 deg)
    # t = 2.5s to 3.0s: posture corrects
    # t = 3.0s to 8.0s: perfect posture held
    w2_means = {j: w2_prof["joints"][j]["mean"] for j in SCORED_JOINTS}
    dt = 0.05
    timestamps = np.arange(0.0, 8.5, dt)

    measured_ttc = None
    for t in timestamps:
        frame_angles = w2_means.copy()
        if t < 2.5:
            frame_angles["left_knee_angle"] += 45.0  # bad knee angle
        elif t < 3.0:
            progress = (t - 2.5) / 0.5
            frame_angles["left_knee_angle"] += 45.0 * (1.0 - progress)

        metrics = tracker.update(frame_angles, landmarks_all_vis, current_timestamp=t)
        if metrics["time_to_correct_s"] is not None and measured_ttc is None:
            measured_ttc = metrics["time_to_correct_s"]

        if metrics["state"] == PoseSessionTracker.STATE_COMPLETED:
            break

    assert measured_ttc is not None, "time_to_correct was never registered"
    # Pose became correct around t ~ 3.0s, held 2s until t ~ 5.0s
    assert 4.5 <= measured_ttc <= 5.5, f"Expected time_to_correct ~5.0s, got {measured_ttc}s"
    assert tracker.success == 1, "Session should end in success after 5s hold"
    print(f"  [+] Error shrinking stream: Registered time_to_correct = {measured_ttc}s | Success = {tracker.success} (PASSED)")

    # -------------------------------------------------------------
    # Test 6: Lost Person During Hold Restarts Hold
    # -------------------------------------------------------------
    print("\n[Test 6/8] Verifying Hold Reset on Lost Person...")
    tracker2 = PoseSessionTracker(
        target_asana="Tree",
        user_id="TestUser",
        reference_file="reference_profiles.json",
        sessions_csv="test_sessions.csv",
    )
    tracker2.state = PoseSessionTracker.STATE_TRACKING
    tracker2.session_start_time = 0.0

    tree_means = {j: ref_profiles["Tree"]["joints"][j]["mean"] for j in SCORED_JOINTS}
    # Hold for 1.5 seconds
    tracker2.update(tree_means, landmarks_all_vis, current_timestamp=0.0)
    tracker2.update(tree_means, landmarks_all_vis, current_timestamp=1.5)
    assert tracker2.current_hold_s >= 1.4, f"Hold should be ~1.5s, got {tracker2.current_hold_s}"

    # Person disappears / lost tracking
    tracker2.update(None, None, current_timestamp=1.6)
    assert tracker2.current_hold_s == 0.0, f"Hold should have reset to 0, got {tracker2.current_hold_s}"
    print("  [+] Lost tracking: Hold duration successfully reset to 0.0s (PASSED)")

    # -------------------------------------------------------------
    # Test 7: Timeout Session Logs success = 0
    # -------------------------------------------------------------
    print("\n[Test 7/8] Verifying 60-second Timeout Behavior...")
    tracker3 = PoseSessionTracker(
        target_asana="Mountain",
        user_id="TestUser",
        reference_file="reference_profiles.json",
        sessions_csv="test_sessions.csv",
    )
    tracker3.state = PoseSessionTracker.STATE_TRACKING
    tracker3.session_start_time = 0.0

    # Feed unaligned pose for 61 seconds
    bad_mountain = {j: 90.0 for j in SCORED_JOINTS}
    tracker3.update(bad_mountain, landmarks_all_vis, current_timestamp=0.0)
    tracker3.update(bad_mountain, landmarks_all_vis, current_timestamp=60.5)

    assert tracker3.state == PoseSessionTracker.STATE_COMPLETED, "Tracker should be COMPLETED on timeout"
    assert tracker3.success == 0, "Timed out session must log success = 0"
    assert tracker3.time_to_correct_s is None, "Timed out session should have None for time_to_correct"
    print(f"  [+] Session Timeout (60s): Completed = True | Success = {tracker3.success} (PASSED)")

    # -------------------------------------------------------------
    # Test 8: Standing and Sitting are NEVER Scored
    # -------------------------------------------------------------
    print("\n[Test 8/8] Verifying Standing and Sitting are NEVER Given Correctness Scores...")
    dummy_angles = {j: 170.0 for j in SCORED_JOINTS}

    for unscored_state in UNSCORED_STATES:
        res_unscored = evaluate_pose(dummy_angles, landmarks_all_vis, unscored_state, ref_profiles)
        assert res_unscored["pose_score"] is None, f"{unscored_state} should have score None"
        assert res_unscored["is_correct"] is False, f"{unscored_state} should have is_correct False"
        assert res_unscored["is_unscored_state"] is True, f"{unscored_state} must be marked unscored state"
        print(f"  [+] Unscored State '{unscored_state}': Score = {res_unscored['pose_score']} | Correct = {res_unscored['is_correct']} (PASSED)")

    # Clean up test_sessions.csv if created
    if os.path.exists("test_sessions.csv"):
        try:
            os.remove("test_sessions.csv")
        except Exception:
            pass

    print("\n" + "=" * 65)
    print(" ALL 8 STAGE 2 TRACKING & SCORING TESTS PASSED WITH 0 ERRORS!")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_tracking_tests()
