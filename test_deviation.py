"""
Unit & Integration Tests for the Deviation Analysis Module
==========================================================
Verifies:
  1. Deviation is exactly 0 when input equals the reference mean.
  2. MAE increases monotonically with severity in simulate mode (noise disabled).
  3. Mirrored Warrior II and Tree poses are not penalised.
  4. Output CSVs contain >= 10 iterations for each of the 4 active classes (40 total).
  5. Existing test_system.py still passes.
  6. Fallback path works when expert_data/ is empty.
  7. Requesting an inactive pose (e.g., Plank) produces a clear message, not a crash.
"""

import io
import os
import sys
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout

import pandas as pd

from train_yoga_pose_classifier import FEATURE_COLUMNS
from mistake_catalog import ACTIVE_POSES, ACTIVE_CLASSES, MIRROR_FEATURE_MAP
from expert_reference import build_reference_profiles
from deviation_analysis import compute_deviation_metrics, run_deviation_analysis
import test_system


class TestDeviationAnalysis(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp(prefix="test_dev_")
        self.empty_expert_dir = os.path.join(self.tmp_dir, "empty_expert")
        self.out_dir = os.path.join(self.tmp_dir, "results")
        os.makedirs(self.empty_expert_dir, exist_ok=True)
        os.makedirs(self.out_dir, exist_ok=True)
        self.ref_profiles = build_reference_profiles(
            expert_dir=self.empty_expert_dir,
            out_dir=self.out_dir,
            save=True,
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_1_zero_deviation_at_reference_mean(self) -> None:
        """(1) Deviation is exactly 0 when input equals the reference mean."""
        for cls_name in ACTIVE_CLASSES:
            pose_ref = self.ref_profiles[cls_name]
            first_var = next(iter(pose_ref["variants"].values()))
            mean_angles = {f: first_var["features"][f]["mean"] for f in FEATURE_COLUMNS}
            metrics = compute_deviation_metrics(mean_angles, pose_ref)
            self.assertEqual(metrics["MAE"], 0.0, f"Expected 0 MAE for {cls_name}")
            self.assertEqual(metrics["RMSE"], 0.0, f"Expected 0 RMSE for {cls_name}")
            self.assertEqual(metrics["max_dev"], 0.0, f"Expected 0 max_dev for {cls_name}")
            self.assertEqual(metrics["score"], 100.0, f"Expected 100 score for {cls_name}")

    def test_2_monotonic_mae_in_simulate_mode_no_noise(self) -> None:
        """(2) MAE increases monotonically with severity in simulate mode (noise disabled)."""
        res = run_deviation_analysis(
            mode="simulate",
            iterations=10,
            poses=ACTIVE_CLASSES,
            seed=42,
            expert_dir=self.empty_expert_dir,
            out_dir=self.out_dir,
            noise_std=0.0,
        )
        df_iter = res["df_iter"]
        for pose_name in ACTIVE_POSES:
            sub = df_iter[df_iter["class"] == pose_name].sort_values("iteration")
            maes = sub["MAE"].tolist()
            self.assertEqual(len(maes), 10)
            for i in range(1, len(maes)):
                self.assertGreater(
                    maes[i],
                    maes[i - 1],
                    f"MAE not strictly increasing for {pose_name} at iter {i+1}: {maes}",
                )

    def test_3_mirrored_warrior2_and_tree_not_penalised(self) -> None:
        """(3) Mirrored Warrior II and Tree are not penalised."""
        for pose_name, var_key in [("Warrior II", "left_front"), ("Tree", "left_standing")]:
            pose_ref = self.ref_profiles[pose_name]
            base_feats = pose_ref["variants"][var_key]["features"]
            mirrored_angles = {
                f: base_feats[MIRROR_FEATURE_MAP[f]]["mean"] for f in FEATURE_COLUMNS
            }
            metrics = compute_deviation_metrics(mirrored_angles, pose_ref)
            self.assertEqual(
                metrics["MAE"],
                0.0,
                f"Mirrored {pose_name} was penalised (MAE={metrics['MAE']})",
            )
            self.assertEqual(metrics["score"], 100.0)

    def test_4_output_csvs_have_10_iterations_per_active_class(self) -> None:
        """(4) Output CSVs contain >= 10 iterations for each of the 4 active classes (40 total)."""
        res = run_deviation_analysis(
            mode="simulate",
            iterations=10,
            poses=["all"],
            seed=42,
            expert_dir=self.empty_expert_dir,
            out_dir=self.out_dir,
            noise_std=2.0,
        )
        df_iter = pd.read_csv(res["iteration_summary_csv"])
        df_dev = pd.read_csv(res["deviation_results_csv"])

        self.assertGreaterEqual(len(df_iter), 40)
        for cls_name in ACTIVE_CLASSES:
            cls_iters = df_iter[df_iter["class"] == cls_name]
            self.assertGreaterEqual(len(cls_iters), 10, f"Missing iterations for {cls_name}")
            cls_dev = df_dev[df_dev["class"] == cls_name]
            self.assertGreaterEqual(len(cls_dev), 10 * len(FEATURE_COLUMNS))

    def test_5_existing_test_system_passes(self) -> None:
        """(5) Existing test_system.py still passes."""
        buf = io.StringIO()
        with redirect_stdout(buf):
            test_system.run_tests()
        # Clean up test_sessions.csv generated by test_tracking inside test_system if present
        if os.path.exists("test_sessions.csv"):
            try:
                os.remove("test_sessions.csv")
            except OSError:
                pass
        self.assertIn("ALL SYSTEM DIAGNOSTICS", buf.getvalue())

    def test_6_fallback_when_expert_data_empty(self) -> None:
        """(6) Fallback path works when expert_data/ is empty."""
        for cls_name in ACTIVE_CLASSES:
            self.assertIn(cls_name, self.ref_profiles)
            self.assertEqual(self.ref_profiles[cls_name]["source"], "synthetic_placeholder")

    def test_7_inactive_pose_produces_clear_message_no_crash(self) -> None:
        """(7) Requesting an inactive pose (e.g. Plank) produces a clear message, not a crash."""
        buf = io.StringIO()
        with redirect_stdout(buf):
            res = run_deviation_analysis(
                mode="simulate",
                iterations=10,
                poses=["Plank"],
                seed=42,
                expert_dir=self.empty_expert_dir,
                out_dir=self.out_dir,
            )
        output = buf.getvalue()
        self.assertEqual(res["status"], "no_active_poses")
        self.assertIn("Inactive pose", output)
        self.assertIn("Plank", output)


if __name__ == "__main__":
    unittest.main(verbosity=2)
