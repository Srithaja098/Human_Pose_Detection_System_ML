"""
Progress Report & Habit Tracking Generator
==========================================
Analyzes session logs (sessions.csv) for a given user and generates:
1. A 3-panel stacked PNG chart per asana:
   - Panel 1: Pose score per session with daily rolling average.
   - Panel 2: Time-to-correct per session (lower is better; timeouts marked).
   - Panel 3: Trajectory of the 3 joints furthest from expert alignment.
2. A per-session CSV with the score change from the previous session for that asana.
3. A console summary comparing the first 3 and last 3 sessions.
4. A Tip Effectiveness audit comparing consecutive sessions.

Usage:
    python progress_report.py --user User1
    python progress_report.py --user User1 --asana "Warrior II"
    python progress_report.py --user User1 --csv sessions.csv
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pose_config import (
    SCORED_ASANAS,
    SCORED_JOINTS,
    SESSION_TIMEOUT_S,
    DEFAULT_SESSIONS_CSV,
)


def analyze_tip_effectiveness(df_user_asana):
    """
    Evaluates whether tips shown in session k led to a decrease in that
    joint's deviation in session k+1 for the same user and asana.
    """
    if len(df_user_asana) < 2:
        return {
            "total_pairs": 0,
            "improved": 0,
            "worsened": 0,
            "unchanged": 0,
            "improvement_rate": 0.0,
            "message": "Need at least 2 sessions to evaluate tip effectiveness.",
        }

    records = df_user_asana.sort_values("timestamp").reset_index(drop=True)
    improved = 0
    worsened = 0
    unchanged = 0
    total_evaluated = 0

    for i in range(len(records) - 1):
        worst_j = records.loc[i, "worst_joint"]
        if pd.isna(worst_j) or worst_j == "None":
            continue

        dev_col = f"{worst_j}_dev"
        if dev_col in records.columns:
            dev_k = records.loc[i, dev_col]
            dev_k_plus_1 = records.loc[i + 1, dev_col]

            if pd.notna(dev_k) and pd.notna(dev_k_plus_1):
                total_evaluated += 1
                if dev_k_plus_1 < dev_k - 0.5:
                    improved += 1
                elif dev_k_plus_1 > dev_k + 0.5:
                    worsened += 1
                else:
                    unchanged += 1

    rate = (improved / total_evaluated * 100.0) if total_evaluated > 0 else 0.0

    return {
        "total_pairs": total_evaluated,
        "improved": improved,
        "worsened": worsened,
        "unchanged": unchanged,
        "improvement_rate": rate,
    }


def generate_asana_report(df_user_asana, user_id, asana_name, output_dir="."):
    """
    Generates the 3-panel vertical stacked plot and prints diagnostic metrics.
    """
    df = df_user_asana.sort_values("timestamp").reset_index(drop=True)
    df["final_score"] = pd.to_numeric(df["final_score"], errors="coerce")
    df["previous_score"] = df["final_score"].shift(1)
    # Scores are on a 0-100 scale, so this is a percentage-point change.
    df["improvement_points"] = df["final_score"] - df["previous_score"]
    df["daily_avg_score"] = df.groupby("date")["final_score"].transform("mean")
    n_sessions = len(df)

    print("\n" + "=" * 70)
    print(f" PROGRESS REPORT — User: '{user_id}' | Asana: '{asana_name}'")
    print(f" Total Recorded Sessions: {n_sessions}")
    print("=" * 70)

    session_log = pd.DataFrame({
        "session_number": np.arange(1, n_sessions + 1),
        "timestamp": df["timestamp"],
        "date": df["date"],
        "asana": asana_name,
        "score": df["final_score"],
        "previous_session_score": df["previous_score"],
        "improvement_points": df["improvement_points"],
        "daily_average_score": df["daily_avg_score"],
        "success": df["success"],
        "time_to_correct_s": df["time_to_correct_s"],
    })
    safe_user = "_".join(str(user_id).strip().split())
    safe_asana = "_".join(str(asana_name).strip().split()).lower()
    session_log_path = os.path.join(
        output_dir, f"progress_{safe_user}_{safe_asana}_sessions.csv"
    )
    session_log.to_csv(session_log_path, index=False)
    print(f"\n[+] Per-session score history saved to: '{session_log_path}'")
    print("\n--- Session Score Change (vs previous session for this asana) ---")
    display_log = session_log[[
        "session_number", "date", "score", "previous_session_score", "improvement_points"
    ]].copy()
    display_log["score"] = display_log["score"].map(lambda value: f"{value:.1f}" if pd.notna(value) else "N/A")
    display_log["previous_session_score"] = display_log["previous_session_score"].map(
        lambda value: f"{value:.1f}" if pd.notna(value) else "—"
    )
    display_log["improvement_points"] = display_log["improvement_points"].map(
        lambda value: f"{value:+.1f} pp" if pd.notna(value) else "First session"
    )
    print(display_log.to_string(index=False))

    # 1. Terminal Trend Summary
    if n_sessions < 6:
        print(f"\n[!] Notice: Trend is too short to trust (< 6 sessions recorded: {n_sessions}).")
        print("    At least 6 sessions are required to establish reliable statistical trends.\n")
    else:
        first3 = df.head(3)
        last3 = df.tail(3)

        first3_score = first3["final_score"].mean()
        last3_score = last3["final_score"].mean()

        # Handle time to correct (replace missing/NaN with timeout 60s for averaging)
        ttc_first3 = pd.to_numeric(first3["time_to_correct_s"], errors="coerce").fillna(SESSION_TIMEOUT_S).mean()
        ttc_last3 = pd.to_numeric(last3["time_to_correct_s"], errors="coerce").fillna(SESSION_TIMEOUT_S).mean()

        succ_first3 = first3["success"].mean() * 100.0
        succ_last3 = last3["success"].mean() * 100.0

        print("\n--- 3-Session Baseline vs Recent Performance ---")
        print(f"  Pose Score:        First 3: {first3_score:.1f}%  ->  Last 3: {last3_score:.1f}%  "
              f"({last3_score - first3_score:+.1f}%)")
        print(f"  Time-to-Correct:   First 3: {ttc_first3:.1f}s   ->  Last 3: {ttc_last3:.1f}s   "
              f"({ttc_last3 - ttc_first3:+.1f}s, {'Faster' if ttc_last3 < ttc_first3 else 'Slower'})")
        print(f"  Success Rate:      First 3: {succ_first3:.1f}%  ->  Last 3: {succ_last3:.1f}%  "
              f"({succ_last3 - succ_first3:+.1f}%)")

        # Joint comparison
        joint_dev_cols = [f"{j}_dev" for j in SCORED_JOINTS if f"{j}_dev" in df.columns]
        if joint_dev_cols:
            first3_devs = first3[joint_dev_cols].mean()
            last3_devs = last3[joint_dev_cols].mean()
            improvements = first3_devs - last3_devs  # positive means deviation decreased

            most_improved_col = improvements.idxmax()
            needs_work_col = last3_devs.idxmax()

            clean_improved = most_improved_col.replace("_dev", "").replace("_angle", "").replace("_", " ").title()
            clean_needs_work = needs_work_col.replace("_dev", "").replace("_angle", "").replace("_", " ").title()

            print(f"  Most Improved Joint:      {clean_improved} (Deviation reduced by {improvements[most_improved_col]:.1f}°)")
            print(f"  Joint Needing Most Work:  {clean_needs_work} (Recent average deviation: {last3_devs[needs_work_col]:.1f}°)")

    # 2. Tip Effectiveness Analysis (Feature G)
    tip_res = analyze_tip_effectiveness(df)
    print("\n--- Correction Tip Effectiveness (Lag-1 Audit) ---")
    if tip_res["total_pairs"] > 0:
        print(f"  Evaluated Consecutive Session Pairs: {tip_res['total_pairs']}")
        print(f"  Joint Improved Next Session:         {tip_res['improved']} ({tip_res['improvement_rate']:.1f}%)")
        print(f"  Joint Worsened Next Session:         {tip_res['worsened']}")
        print(f"  Joint Remained Unchanged:            {tip_res['unchanged']}")
        if tip_res["improvement_rate"] >= 50.0:
            print("  Conclusion: Tips are having a POSITIVE corrective effect on user posture.")
        else:
            print("  Conclusion: Tips need reinforcement; user may need additional verbal/visual guidance.")
    else:
        print("  Insufficient session pairs to evaluate tip effectiveness.")

    # 3. Generate 3-Panel Stacked PNG Plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 13), sharex=False)
    fig.patch.set_facecolor("#1e222d")

    session_indices = np.arange(1, n_sessions + 1)

    # Panel 1: Pose Score & Daily Rolling Average
    ax1.set_facecolor("#282c37")
    scores = df["final_score"].values
    ax1.plot(session_indices, scores, marker="o", color="#38ef7d", linewidth=2.2, label="Session Score (%)")
    ax1.axhline(80.0, color="#f1c40f", linestyle="--", alpha=0.8, label="Correctness Threshold (80%)")

    # Daily Average trend line
    ax1.plot(session_indices, df["daily_avg_score"].values, color="#00c6ff", linestyle="-.", linewidth=2.0, label="Daily Average")

    ax1.set_title(f"Pose Accuracy Score Progression — {asana_name}", color="#ffffff", fontsize=13, fontweight="bold", pad=10)
    ax1.set_ylabel("Score (0 - 100)", color="#ffffff", fontsize=11)
    ax1.set_ylim(-5, 105)
    ax1.grid(True, linestyle=":", alpha=0.4, color="#ffffff")
    ax1.tick_params(colors="#ffffff")
    ax1.legend(loc="lower right", facecolor="#1e222d", edgecolor="#ffffff", labelcolor="#ffffff")

    # Panel 2: Time-to-Correct (Lower is better; timeouts marked with red 'X')
    ax2.set_facecolor("#282c37")
    raw_ttc = pd.to_numeric(df["time_to_correct_s"], errors="coerce").values
    successes = df["success"].values

    valid_mask = (successes == 1) & (~np.isnan(raw_ttc))
    timeout_mask = ~valid_mask

    # Plot successful time-to-correct
    if np.any(valid_mask):
        ax2.plot(session_indices[valid_mask], raw_ttc[valid_mask], marker="o", color="#00e5ff", linewidth=2.2, label="Time-to-Correct (s)")

    # Plot timeouts distinctly with red X markers at 60s
    if np.any(timeout_mask):
        timeout_y = np.full(np.sum(timeout_mask), SESSION_TIMEOUT_S)
        ax2.scatter(session_indices[timeout_mask], timeout_y, color="#ff4b5c", marker="X", s=90, zorder=5, label=f"Timeout ({SESSION_TIMEOUT_S:.0f}s, Unsuccessful)")

    ax2.set_title("Time Taken to Reach Correct Pose (Lower is Better)", color="#ffffff", fontsize=13, fontweight="bold", pad=10)
    ax2.set_ylabel("Seconds to Correct", color="#ffffff", fontsize=11)
    ax2.set_ylim(0, SESSION_TIMEOUT_S + 5)
    ax2.grid(True, linestyle=":", alpha=0.4, color="#ffffff")
    ax2.tick_params(colors="#ffffff")
    ax2.legend(loc="upper right", facecolor="#1e222d", edgecolor="#ffffff", labelcolor="#ffffff")

    # Panel 3: Top 3 Joints Furthest from Expert Pose
    ax3.set_facecolor("#282c37")
    joint_dev_cols = [f"{j}_dev" for j in SCORED_JOINTS if f"{j}_dev" in df.columns]
    
    if joint_dev_cols:
        mean_devs = df[joint_dev_cols].mean().sort_values(ascending=False)
        top3_cols = mean_devs.head(3).index.tolist()
        palette = ["#ff7675", "#fdcb6e", "#a29bfe"]

        for idx, col in enumerate(top3_cols):
            clean_name = col.replace("_dev", "").replace("_angle", "").replace("_", " ").title()
            ax3.plot(session_indices, df[col].values, marker="s", color=palette[idx % len(palette)],
                     linewidth=1.8, label=f"{clean_name} (Mean Dev: {mean_devs[col]:.1f}°)")

        ax3.set_title("Top 3 Joints Furthest from Expert Pose (Deviation over Time)", color="#ffffff", fontsize=13, fontweight="bold", pad=10)
        ax3.set_ylabel("Deviation (Degrees)", color="#ffffff", fontsize=11)
        ax3.set_xlabel("Session Number", color="#ffffff", fontsize=11)
        ax3.grid(True, linestyle=":", alpha=0.4, color="#ffffff")
        ax3.tick_params(colors="#ffffff")
        ax3.legend(loc="upper right", facecolor="#1e222d", edgecolor="#ffffff", labelcolor="#ffffff")

    plt.tight_layout()

    plot_path = os.path.join(output_dir, f"progress_{safe_user}_{safe_asana}.png")
    plt.savefig(plot_path, dpi=180, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()

    print(f"\n[+] 3-Panel stacked progress chart saved to: '{plot_path}'")
    print("=" * 70 + "\n")


def generate_reports(user_id, target_asana=None, sessions_csv=DEFAULT_SESSIONS_CSV, output_dir="."):
    """
    Loads sessions.csv, filters for user_id and target_asana, and produces reports.
    """
    if not os.path.exists(sessions_csv):
        print(f"[!] Error: Sessions log '{sessions_csv}' does not exist.")
        print("    Run tracking sessions first using: python model.py --track \"Warrior II\" --user User1")
        return

    df = pd.read_csv(sessions_csv)
    if len(df) == 0:
        print(f"[!] Error: Sessions log '{sessions_csv}' is empty.")
        return

    df_user = df[df["user_id"] == user_id]
    if len(df_user) == 0:
        avail_users = df["user_id"].unique().tolist()
        print(f"[!] Error: No sessions found for user '{user_id}'. Available users: {avail_users}")
        return

    asanas_to_run = [target_asana] if target_asana else df_user["asana"].unique().tolist()

    for asana in asanas_to_run:
        df_sub = df_user[df_user["asana"] == asana]
        if len(df_sub) == 0:
            print(f"[!] No sessions found for user '{user_id}' on asana '{asana}'.")
            continue
        generate_asana_report(df_sub, user_id, asana, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Yoga Posture Progress & Habit Reports")
    parser.add_argument(
        "--user",
        type=str,
        default="User1",
        help="User ID to analyze (default: User1)",
    )
    parser.add_argument(
        "--asana",
        type=str,
        default=None,
        choices=SCORED_ASANAS,
        help="Optional asana filter (default: all asanas practiced by user)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=DEFAULT_SESSIONS_CSV,
        help=f"Path to sessions log CSV (default: {DEFAULT_SESSIONS_CSV})",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory to save generated PNG plots (default: current directory)",
    )
    args = parser.parse_args()

    generate_reports(
        user_id=args.user,
        target_asana=args.asana,
        sessions_csv=args.csv,
        output_dir=args.output_dir,
    )
