"""
Stage 2: Pose Correctness Scoring, Correction Tips & Session Tracking
=====================================================================
Evaluates real-time pose correctness against expert reference profiles,
tracks time-to-correct, delivers joint-specific feedback tips, and logs
session performance to sessions.csv.

Mathematical Definitions:
-------------------------
1. Joint Deviation:
   dev_j = |theta_user_j - mu_expert_j|

2. Piecewise Linear Joint Score:
   Score_j = 100.0                          if dev_j <= tol_j
   Score_j = 100.0 * (1.0 - (dev_j - tol_j) / (2.0 * tol_j))
                                            if tol_j < dev_j <= 3.0 * tol_j
   Score_j = 0.0                            if dev_j > 3.0 * tol_j

3. Overall Pose Score:
   PoseScore = (1 / N) * sum(Score_j)  for all N valid joints (N >= 7)

4. Correctness Condition:
   is_correct = (PoseScore >= 80.0) AND (for all j: dev_j <= 1.5 * tol_j)

5. Left/Right Mirror Symmetry:
   Evaluated against both canonical reference and its bilateral reflection;
   the higher-scoring profile is automatically retained.
"""

import os
import sys
import time
import json
import datetime
import numpy as np
import pandas as pd
import cv2

from pose_config import (
    SCORED_ASANAS,
    UNSCORED_STATES,
    SCORED_JOINTS,
    MIRROR_JOINT_MAP,
    JOINT_LANDMARK_MAP,
    SCORE_THRESHOLD,
    MAX_TOLERANCE_MULTIPLIER,
    ZERO_SCORE_MULTIPLIER,
    MIN_TOLERANCE_DEG,
    DEFAULT_TOLERANCE_DEG,
    MIN_LANDMARK_VISIBILITY,
    MIN_SCORED_JOINTS,
    TOTAL_SCORED_JOINTS,
    COUNTDOWN_TIME_S,
    HOLD_CORRECT_TIME_S,
    HOLD_SUCCESS_TIME_S,
    SESSION_TIMEOUT_S,
    EMA_ALPHA,
    DEFAULT_REFERENCE_FILE,
    DEFAULT_SESSIONS_CSV,
)
from build_reference import load_reference_profiles


# =====================================================================
# Biomechanical Correction Advice Generator
# =====================================================================

def generate_signed_joint_tip(joint_name, signed_diff, abs_diff):
    """
    Generates human-readable posture adjustment instruction based on
    signed angle deviation (user_angle - expert_mean).
    
    signed_diff > 0: user angle is larger than expert mean.
    signed_diff < 0: user angle is smaller than expert mean.
    """
    deg_str = f"{abs(signed_diff):.0f}"
    clean_name = joint_name.replace("_angle", "").replace("_", " ").title()

    if "elbow" in joint_name:
        if signed_diff < 0:
            return f"Straighten your {clean_name} more (about {deg_str} deg more)"
        else:
            return f"Bend your {clean_name} slightly (about {deg_str} deg more)"

    elif "shoulder" in joint_name:
        if signed_diff < 0:
            return f"Raise your {clean_name} higher (about {deg_str} deg more)"
        else:
            return f"Lower your {clean_name} slightly (about {deg_str} deg more)"

    elif "knee" in joint_name:
        if signed_diff > 0:
            return f"Bend your {clean_name} more (about {deg_str} deg more)"
        else:
            return f"Straighten your {clean_name} more (about {deg_str} deg more)"

    elif "hip" in joint_name:
        if signed_diff < 0:
            return f"Open your {clean_name} more (about {deg_str} deg more)"
        else:
            return f"Align your {clean_name} more upright (about {deg_str} deg more)"

    elif "torso" in joint_name:
        return f"Align your torso more vertically upright (about {deg_str} deg off)"

    return f"Adjust your {clean_name} by about {deg_str} deg"


def check_persistent_weakness(user_id, asana_name, current_worst_joint,
                              sessions_csv=DEFAULT_SESSIONS_CSV, window=5, threshold=3):
    """
    Checks if current_worst_joint was the worst joint in >= threshold of the last
    window sessions for this user and asana. If so, returns practice suggestion.
    """
    if not os.path.exists(sessions_csv):
        return None

    try:
        df = pd.read_csv(sessions_csv)
        if len(df) == 0 or "user_id" not in df.columns or "asana" not in df.columns:
            return None

        user_df = df[(df["user_id"] == user_id) & (df["asana"] == asana_name)]
        if len(user_df) < 2:
            return None

        recent = user_df.tail(window)
        worst_counts = (recent["worst_joint"] == current_worst_joint).sum()
        
        # Include current session as potential trigger
        if worst_counts >= (threshold - 1):
            clean_joint = current_worst_joint.replace("_angle", "").replace("_", " ").title()
            return f"Persistent issue: {clean_joint} was your weakest joint in {worst_counts + 1} of your recent sessions. Focus on focused alignment drills for this joint."
    except Exception:
        pass

    return None


# =====================================================================
# Pose Evaluation Engine
# =====================================================================

def evaluate_joint(user_angle, expert_mean, tolerance):
    """
    Evaluates a single joint against expert mean and tolerance.
    Returns:
        (joint_score, deviation, signed_diff)
    """
    deviation = abs(user_angle - expert_mean)
    signed_diff = user_angle - expert_mean

    # Piecewise linear score function
    if deviation <= tolerance:
        score = 100.0
    elif deviation <= ZERO_SCORE_MULTIPLIER * tolerance:
        # Linear drop from 100 to 0
        range_span = (ZERO_SCORE_MULTIPLIER - 1.0) * tolerance
        score = 100.0 * (1.0 - ((deviation - tolerance) / range_span))
    else:
        score = 0.0

    return max(0.0, min(100.0, score)), deviation, signed_diff


def evaluate_pose_against_profile(user_angles, landmarks_list, expert_profile):
    """
    Evaluates user joint angles against a single reference profile.
    Filters out joints with landmark visibility < MIN_LANDMARK_VISIBILITY.
    Returns:
        (pose_score, is_correct, worst_joint, joint_details, valid_count)
    """
    joint_details = {}
    valid_scores = []
    has_exceeded_tolerance_limit = False
    worst_joint = None
    max_relative_dev = -1.0

    profile_joints = expert_profile.get("joints", {})

    for joint_name in SCORED_JOINTS:
        # Check landmark visibility
        is_visible = True
        if landmarks_list is not None and len(landmarks_list) >= 33:
            lm_indices = JOINT_LANDMARK_MAP.get(joint_name, [])
            for idx in lm_indices:
                if idx < len(landmarks_list):
                    vis = getattr(landmarks_list[idx], "visibility", 1.0)
                    if vis < MIN_LANDMARK_VISIBILITY:
                        is_visible = False
                        break

        if not is_visible or joint_name not in user_angles:
            joint_details[joint_name] = {
                "visible": False,
                "user_angle": user_angles.get(joint_name, None),
                "score": None,
                "deviation": None,
                "tolerance": None,
            }
            continue

        u_angle = user_angles[joint_name]
        exp_info = profile_joints.get(joint_name, {})
        exp_mean = exp_info.get("mean", 180.0)
        exp_tol = exp_info.get("tolerance", DEFAULT_TOLERANCE_DEG)

        j_score, dev, signed_diff = evaluate_joint(u_angle, exp_mean, exp_tol)
        valid_scores.append(j_score)

        # Check if any joint exceeds 1.5x tolerance
        if dev > MAX_TOLERANCE_MULTIPLIER * exp_tol:
            has_exceeded_tolerance_limit = True

        # Track worst joint based on deviation relative to tolerance
        relative_dev = dev / exp_tol if exp_tol > 0 else dev
        if relative_dev > max_relative_dev:
            max_relative_dev = relative_dev
            worst_joint = joint_name

        joint_details[joint_name] = {
            "visible": True,
            "user_angle": u_angle,
            "expert_mean": exp_mean,
            "score": j_score,
            "deviation": dev,
            "signed_diff": signed_diff,
            "tolerance": exp_tol,
            "relative_dev": relative_dev,
        }

    valid_count = len(valid_scores)
    if valid_count < MIN_SCORED_JOINTS:
        return None, False, None, joint_details, valid_count

    pose_score = float(np.mean(valid_scores))
    is_correct = (pose_score >= SCORE_THRESHOLD) and (not has_exceeded_tolerance_limit)

    return pose_score, is_correct, worst_joint, joint_details, valid_count


def create_mirrored_profile(profile):
    """
    Constructs a mirrored reference profile by swapping left and right joints.
    """
    mirrored = {
        "provisional": profile.get("provisional", True),
        "description": profile.get("description", "") + " (Mirrored)",
        "sample_count": profile.get("sample_count", 1),
        "joints": {},
    }

    orig_joints = profile.get("joints", {})
    for joint_name, mirror_name in MIRROR_JOINT_MAP.items():
        if mirror_name in orig_joints:
            mirrored["joints"][joint_name] = orig_joints[mirror_name]
        elif joint_name in orig_joints:
            mirrored["joints"][joint_name] = orig_joints[joint_name]

    return mirrored


def evaluate_pose(user_angles, landmarks_list, asana_name, reference_profiles):
    """
    Main evaluation routine:
    1. Checks if asana has an expert reference profile (refuses to run if missing).
    2. Evaluates canonical and mirrored reference profiles.
    3. Keeps the better fit.
    4. Generates corrective feedback.
    
    Returns:
        dict containing:
            pose_score: float (0-100) or None
            is_correct: bool
            worst_joint: str or None
            tip: str
            persistent_tip: str or None
            joint_details: dict
            valid_joints_count: int
            used_mirror: bool
            is_unscored_state: bool
    """
    # Guard: Standing and Sitting are NEVER scored
    if asana_name in UNSCORED_STATES:
        return {
            "pose_score": None,
            "is_correct": False,
            "worst_joint": None,
            "tip": f"Guidance state ({asana_name}). Not in a scored pose yet.",
            "persistent_tip": None,
            "joint_details": {},
            "valid_joints_count": 0,
            "used_mirror": False,
            "is_unscored_state": True,
        }

    # Guard: Asana must have a reference profile
    if asana_name not in reference_profiles:
        raise ValueError(
            f"No reference profile found for '{asana_name}'. "
            f"Tracking cannot run. Please record an instructor using:\n"
            f"    python record_expert.py --asana \"{asana_name}\"\n"
            f"followed by:\n"
            f"    python build_reference.py --csv expert_angles.csv --asana \"{asana_name}\""
        )

    base_profile = reference_profiles[asana_name]
    mirror_profile = create_mirrored_profile(base_profile)

    # Evaluate both orientations
    score_orig, correct_orig, worst_orig, details_orig, count_orig = evaluate_pose_against_profile(
        user_angles, landmarks_list, base_profile
    )
    score_mirr, correct_mirr, worst_mirr, details_mirr, count_mirr = evaluate_pose_against_profile(
        user_angles, landmarks_list, mirror_profile
    )

    # Pick better fit
    val_orig = score_orig if score_orig is not None else -1.0
    val_mirr = score_mirr if score_mirr is not None else -1.0

    if val_mirr > val_orig:
        chosen_score = score_mirr
        chosen_correct = correct_mirr
        chosen_worst = worst_mirr
        chosen_details = details_mirr
        chosen_count = count_mirr
        used_mirror = True
    else:
        chosen_score = score_orig
        chosen_correct = correct_orig
        chosen_worst = worst_orig
        chosen_details = details_orig
        chosen_count = count_orig
        used_mirror = False

    # Generate tips
    tip = "Hold steady! Excellent alignment."
    if chosen_worst is not None and chosen_worst in chosen_details:
        w_info = chosen_details[chosen_worst]
        signed_diff = w_info.get("signed_diff", 0.0)
        abs_diff = w_info.get("deviation", 0.0)
        tip = generate_signed_joint_tip(chosen_worst, signed_diff, abs_diff)

    return {
        "pose_score": chosen_score,
        "is_correct": chosen_correct,
        "worst_joint": chosen_worst,
        "tip": tip,
        "persistent_tip": None,
        "joint_details": chosen_details,
        "valid_joints_count": chosen_count,
        "used_mirror": used_mirror,
        "is_unscored_state": False,
    }


# =====================================================================
# Real-Time Session Tracker State Machine
# =====================================================================

class PoseSessionTracker:
    """
    Manages live session tracking, countdown, hold duration, time-to-correct
    measurement, EMA angle smoothing, and CSV logging.
    """

    STATE_IDLE = "IDLE"
    STATE_COUNTDOWN = "COUNTDOWN"
    STATE_TRACKING = "TRACKING"
    STATE_COMPLETED = "COMPLETED"

    def __init__(self, target_asana, user_id="User1",
                 reference_file=DEFAULT_REFERENCE_FILE,
                 sessions_csv=DEFAULT_SESSIONS_CSV):
        self.target_asana = target_asana
        self.user_id = user_id
        self.reference_file = reference_file
        self.sessions_csv = sessions_csv

        # Load reference profiles
        self.reference_profiles = load_reference_profiles(reference_file)
        if target_asana not in self.reference_profiles:
            raise ValueError(
                f"Reference profile missing for '{target_asana}'. Cannot track.\n"
                f"Record expert reference with: python record_expert.py --asana \"{target_asana}\""
            )

        # State machine variables
        self.state = self.STATE_IDLE
        self.countdown_start_time = None
        self.session_start_time = None
        self.current_time = 0.0

        # Angle smoothing buffer
        self.smoothed_angles = {}

        # Timing & hold progress
        self.current_hold_s = 0.0
        self.max_hold_s = 0.0
        self.time_to_correct_s = None
        self.success = 0
        self.session_duration_s = 0.0

        # Score tracking across session
        self.recorded_scores = []
        self.final_score = 0.0
        self.last_worst_joint = None
        self.last_tip = "Press SPACE to start session"
        self.persistent_warning = None

        # Deviations history across session for logging
        self.joint_deviations_history = {j: [] for j in SCORED_JOINTS}

    def start_countdown(self):
        """
        Transitions from IDLE to COUNTDOWN.
        """
        self.state = self.STATE_COUNTDOWN
        self.countdown_start_time = time.time()
        self.current_hold_s = 0.0
        self.time_to_correct_s = None
        self.recorded_scores = []
        self.smoothed_angles = {}
        self.joint_deviations_history = {j: [] for j in SCORED_JOINTS}
        print(f"[*] Started 5-second countdown for {self.target_asana}...")

    def update(self, raw_features, landmarks_list, current_timestamp=None):
        """
        Processes a single frame. Updates EMA angles, evaluates correctness,
        updates hold timers and session state.
        
        Returns:
            dict containing current tracking HUD metrics.
        """
        now = time.time() if current_timestamp is None else current_timestamp

        # 1. State: IDLE
        if self.state == self.STATE_IDLE:
            return {
                "state": self.state,
                "score": None,
                "is_correct": False,
                "hold_s": 0.0,
                "timer_s": 0.0,
                "time_to_correct_s": None,
                "tip": "Press SPACEBAR to start 5s countdown",
                "persistent_tip": None,
                "worst_joint": None,
                "countdown_val": 0,
            }

        # 2. State: COUNTDOWN
        if self.state == self.STATE_COUNTDOWN:
            elapsed = now - self.countdown_start_time
            remaining = COUNTDOWN_TIME_S - elapsed
            if remaining <= 0:
                self.state = self.STATE_TRACKING
                self.session_start_time = now
                print(f"[*] Countdown finished! Timer started for {self.target_asana}.")
                return {
                    "state": self.STATE_TRACKING,
                    "score": None,
                    "is_correct": False,
                    "hold_s": 0.0,
                    "timer_s": 0.0,
                    "time_to_correct_s": None,
                    "tip": "Assume pose! Session started.",
                    "persistent_tip": None,
                    "worst_joint": None,
                    "countdown_val": 0,
                }
            return {
                "state": self.state,
                "score": None,
                "is_correct": False,
                "hold_s": 0.0,
                "timer_s": 0.0,
                "time_to_correct_s": None,
                "tip": f"Get ready! Starting in {int(np.ceil(remaining))}s...",
                "persistent_tip": None,
                "worst_joint": None,
                "countdown_val": int(np.ceil(remaining)),
            }

        # 3. State: COMPLETED
        if self.state == self.STATE_COMPLETED:
            return {
                "state": self.state,
                "score": self.final_score,
                "is_correct": (self.success == 1),
                "hold_s": self.current_hold_s,
                "timer_s": self.session_duration_s,
                "time_to_correct_s": self.time_to_correct_s,
                "tip": "Session Complete!" if self.success == 1 else "Session Timed Out (60s)",
                "persistent_tip": self.persistent_warning,
                "worst_joint": self.last_worst_joint,
                "countdown_val": 0,
            }

        # 4. State: TRACKING
        session_elapsed = now - self.session_start_time
        self.session_duration_s = session_elapsed

        # Apply EMA smoothing to joint angles:
        # angle_smooth = alpha * angle_raw + (1 - alpha) * angle_prev
        for j in SCORED_JOINTS:
            val = raw_features.get(j, None) if raw_features else None
            if val is not None:
                if j in self.smoothed_angles:
                    self.smoothed_angles[j] = (EMA_ALPHA * val) + ((1.0 - EMA_ALPHA) * self.smoothed_angles[j])
                else:
                    self.smoothed_angles[j] = float(val)

        # Lost person / No landmarks: hold resets!
        if raw_features is None or landmarks_list is None:
            self.current_hold_s = 0.0
            return {
                "state": self.state,
                "score": None,
                "is_correct": False,
                "hold_s": 0.0,
                "timer_s": session_elapsed,
                "time_to_correct_s": self.time_to_correct_s,
                "tip": "Step back into camera view — Hold reset!",
                "persistent_tip": None,
                "worst_joint": None,
                "countdown_val": 0,
            }

        # Evaluate pose against expert reference
        eval_result = evaluate_pose(
            self.smoothed_angles, landmarks_list, self.target_asana, self.reference_profiles
        )

        score = eval_result["pose_score"]
        is_correct = eval_result["is_correct"]
        worst_joint = eval_result["worst_joint"]
        tip = eval_result["tip"]

        if score is not None:
            self.recorded_scores.append(score)
            self.final_score = score

        if worst_joint:
            self.last_worst_joint = worst_joint
            # Accumulate joint deviations
            for j, j_data in eval_result["joint_details"].items():
                if j_data.get("deviation") is not None:
                    self.joint_deviations_history[j].append(j_data["deviation"])

        # Check persistent habit
        if self.persistent_warning is None and worst_joint is not None:
            self.persistent_warning = check_persistent_weakness(
                self.user_id, self.target_asana, worst_joint, self.sessions_csv
            )

        full_tip = tip
        if self.persistent_warning:
            full_tip = f"{tip} | {self.persistent_warning}"
        self.last_tip = full_tip

        # Hold & Time-to-Correct logic
        # In live webcam, dt ~ 1/fps. If timestamps passed, delta = now - prev_now.
        dt = 0.033  # default fallback ~30 fps
        if hasattr(self, "_prev_timestamp") and self._prev_timestamp is not None:
            dt = max(0.001, now - self._prev_timestamp)
        self._prev_timestamp = now

        if is_correct:
            self.current_hold_s += dt
            if self.current_hold_s > self.max_hold_s:
                self.max_hold_s = self.current_hold_s

            # First time reached correct AND held for 2.0s:
            if self.time_to_correct_s is None and self.current_hold_s >= HOLD_CORRECT_TIME_S:
                # time_to_correct = time from timer start until 2s hold reached
                self.time_to_correct_s = round(session_elapsed, 2)
                print(f"[+] FIRST CORRECT POSE HELD FOR 2s! time_to_correct: {self.time_to_correct_s}s")

            # Check for Session Success (Held for 5.0 continuous seconds)
            if self.current_hold_s >= HOLD_SUCCESS_TIME_S:
                self.success = 1
                self.state = self.STATE_COMPLETED
                self.session_duration_s = round(session_elapsed, 2)
                print(f"[+] SESSION SUCCESS! Held {self.target_asana} for {HOLD_SUCCESS_TIME_S}s!")
                self.log_session()
        else:
            # Pose was broken: hold resets!
            self.current_hold_s = 0.0

        # Check for Session Timeout (60.0s elapsed without reaching 5s hold)
        if session_elapsed >= SESSION_TIMEOUT_S and self.state != self.STATE_COMPLETED:
            self.success = 0
            self.state = self.STATE_COMPLETED
            self.session_duration_s = round(session_elapsed, 2)
            print(f"[!] SESSION TIMEOUT (60.0s reached without 5s hold). Success = 0.")
            self.log_session()

        return {
            "state": self.state,
            "score": score,
            "is_correct": is_correct,
            "hold_s": round(self.current_hold_s, 1),
            "timer_s": round(session_elapsed, 1),
            "time_to_correct_s": self.time_to_correct_s,
            "tip": full_tip,
            "persistent_tip": self.persistent_warning,
            "worst_joint": worst_joint,
            "countdown_val": 0,
        }

    def log_session(self):
        """
        Appends the completed session metrics to sessions.csv.
        """
        now_dt = datetime.datetime.now()
        timestamp_str = now_dt.strftime("%Y-%m-%d %H:%M:%S")
        date_str = now_dt.strftime("%Y-%m-%d")

        best_score = float(np.max(self.recorded_scores)) if self.recorded_scores else 0.0
        mean_score = float(np.mean(self.recorded_scores)) if self.recorded_scores else 0.0

        row = {
            "timestamp": timestamp_str,
            "date": date_str,
            "user_id": self.user_id,
            "asana": self.target_asana,
            "success": int(self.success),
            "time_to_correct_s": self.time_to_correct_s if self.time_to_correct_s is not None else "",
            "hold_s": round(self.max_hold_s, 2),
            "best_score": round(best_score, 1),
            "mean_score": round(mean_score, 1),
            "final_score": round(self.final_score, 1),
            "duration_s": round(self.session_duration_s, 2),
            "worst_joint": self.last_worst_joint if self.last_worst_joint else "None",
            "tip": self.last_tip.replace(",", ";"),
        }

        # Add mean deviation per joint
        for j in SCORED_JOINTS:
            dev_list = self.joint_deviations_history.get(j, [])
            mean_dev = round(float(np.mean(dev_list)), 2) if dev_list else 0.0
            row[f"{j}_dev"] = mean_dev

        df_row = pd.DataFrame([row])
        if os.path.exists(self.sessions_csv):
            df_row.to_csv(self.sessions_csv, mode="a", header=False, index=False)
        else:
            df_row.to_csv(self.sessions_csv, mode="w", header=True, index=False)

        print(f"[+] Session logged to '{self.sessions_csv}' for {self.user_id} - {self.target_asana}")


# =====================================================================
# Stage 2 HUD Drawing Helper
# =====================================================================

def draw_stage2_tracking_hud(frame, tracking_metrics, asana_name):
    """
    Renders Stage 2 tracking HUD overlay with score meter, session timer,
    5s hold progress bar, worst joint tip, and countdown.
    """
    h, w, _ = frame.shape
    state = tracking_metrics.get("state", "IDLE")
    score = tracking_metrics.get("score")
    is_correct = tracking_metrics.get("is_correct", False)
    hold_s = tracking_metrics.get("hold_s", 0.0)
    timer_s = tracking_metrics.get("timer_s", 0.0)
    ttc = tracking_metrics.get("time_to_correct_s")
    tip = tracking_metrics.get("tip", "")
    countdown_val = tracking_metrics.get("countdown_val", 0)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 110), (20, 24, 33), -1)
    cv2.rectangle(overlay, (0, h - 55), (w, h), (20, 24, 33), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    # Accent line
    status_color = (0, 220, 100) if is_correct else (0, 165, 255)
    cv2.line(frame, (0, 110), (w, 110), status_color, 2)

    # Top title
    cv2.putText(
        frame,
        f"STAGE 2 TRACKING: {asana_name.upper()}",
        (20, 30),
        cv2.FONT_HERSHEY_DUPLEX,
        0.7,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    # State & Score
    if state == PoseSessionTracker.STATE_IDLE:
        display_state = "IDLE — Press SPACE to Begin"
        color = (255, 200, 50)
    elif state == PoseSessionTracker.STATE_COUNTDOWN:
        display_state = f"COUNTDOWN: {countdown_val}s"
        color = (0, 220, 255)
    elif state == PoseSessionTracker.STATE_COMPLETED:
        display_state = "SESSION FINISHED!"
        color = (0, 255, 100) if is_correct else (0, 0, 255)
    else:
        score_str = f"Score: {score:.1f}%" if score is not None else "Calculating..."
        correct_tag = " [CORRECT]" if is_correct else " [ADJUST]"
        display_state = f"{score_str}{correct_tag}"
        color = (0, 220, 100) if is_correct else (0, 140, 255)

    cv2.putText(
        frame,
        display_state,
        (20, 68),
        cv2.FONT_HERSHEY_DUPLEX,
        0.95,
        color,
        2,
        cv2.LINE_AA,
    )

    # Hold bar & Session timer (Top Right)
    timer_text = f"Timer: {timer_s:.1f}s / {SESSION_TIMEOUT_S:.0f}s"
    cv2.putText(
        frame,
        timer_text,
        (w - 260, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    if ttc is not None:
        ttc_text = f"First Correct: {ttc:.1f}s"
        cv2.putText(
            frame,
            ttc_text,
            (w - 260, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 100),
            1,
            cv2.LINE_AA,
        )

    # Hold Progress Bar (Width 220px)
    bar_x = w - 260
    bar_y = 75
    bar_w = 230
    bar_h = 18
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
    hold_ratio = min(1.0, hold_s / HOLD_SUCCESS_TIME_S)
    fill_w = int(bar_w * hold_ratio)
    fill_color = (0, 220, 100) if hold_ratio >= 1.0 else (0, 180, 255)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), fill_color, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (180, 180, 180), 1)

    hold_label = f"Hold: {hold_s:.1f}s / {HOLD_SUCCESS_TIME_S:.0f}s"
    cv2.putText(
        frame,
        hold_label,
        (bar_x + 10, bar_y + 13),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    # Bottom Bar: Tip
    cv2.putText(
        frame,
        f"Tip: {tip}",
        (20, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        "SPACE: Start | Q: Quit",
        (w - 180, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (160, 160, 160),
        1,
        cv2.LINE_AA,
    )
