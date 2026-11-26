# Pose Deviation & Sitting Robustness Analysis Report

**Generated:** 2026-11-26 19:23:30  
**Mode:** `simulate` | **Iterations per Class:** `10` | **Score Tau:** `20.0` | **Seed:** `42`  
**Active Scope:** `['Warrior II', 'Tree', 'Mountain', 'Sitting']` (Asanas: `['Warrior II', 'Tree', 'Mountain']`; Non-Pose Robustness: `['Sitting']`)  
**Classifier Model Classes Loaded:** `['Mountain', 'Sitting', 'Standing', 'Tree', 'Warrior II']`

> [!WARNING]
> **Placeholder Reference Data Notice:** The following classes used `synthetic_placeholder` reference distributions derived from `generate_synthetic_pose_dataset()`: **Warrior II, Tree, Mountain, Sitting**. These preliminary results must be re-run after project members record live reference data via `record_reference.py`.

## 1. Methodology & Reference Data Sources

- Ground-truth reference profiles represent poses performed by project members (called **reference data**; no external certified yoga experts were used).
- Both reference profiles and mistake/variation inputs use the identical 11-angle `extract_features()` representation from `train_yoga_pose_classifier.py`.
- Mirror symmetry (left/right variants for Warrior II and Tree) and Sitting variants (chair vs floor) are stored separately; deviations are evaluated against the lowest-deviation variant.

### Data Source Breakdown

| Class | Source | Samples | People | Variants |
|---|---|---:|---:|---|
| Warrior II | `synthetic_placeholder` | 300 | 0 | left_front, right_front |
| Tree | `synthetic_placeholder` | 300 | 0 | left_standing, right_standing |
| Mountain | `synthetic_placeholder` | 300 | 0 | canonical |
| Sitting | `synthetic_placeholder` | 300 | 0 | chair, floor |

## 2. Pose-Level Summary Metrics

| class | source | n_iterations | mean_MAE | std_MAE | mean_RMSE | mean_score | pearson_r | pearson_p | most_sensitive_joint | most_sensitive_joint_dev | classifier_accuracy | first_misclassified_iter | first_misclassified_severity | feedback_hit_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Warrior II | synthetic_placeholder | 10 | 6.9919 | 5.3562 | 11.9156 | 72.6314 | 0.9435 | 0 | left_hip_angle | 13.4772 | 0.9 | 10 | 1 | 0.3 |
| Tree | synthetic_placeholder | 10 | 11.0283 | 11.4685 | 19.2651 | 64.3718 | 0.8465 | 0 | left_shoulder_angle | 31.3622 | 0.7 | 7 | 0.7 | 0.6 |
| Mountain | synthetic_placeholder | 10 | 9.9352 | 9.0056 | 18.4016 | 65.8318 | 0.924 | 0 | right_shoulder_angle | 20.9874 | 0.7 | 8 | 0.8 | 0.1 |
| Sitting | synthetic_placeholder | 10 | 11.2136 | 11.2305 | 16.3273 | 63.8952 | N/A | N/A | right_elbow_angle | 18.1107 | 0.9 | 10 | 1 | N/A |

## 3. Sitting Robustness & Reverse False-Positive Analysis

- **ML Classifier Alone Sitting Detection Rate:** 90.0% (9/10)
- **Heuristic (`detect_sitting_or_partial`) Alone Detection Rate:** 90.0% (9/10)
- **Combined Pipeline (Heuristic First -> ML) Detection Rate:** 100.0% (10/10)
- **Reverse Standing Test False-Positive Rate (Heuristic):** 0.0% (0/5)
- **Reverse Standing Test False-Positive Rate (Combined):** 0.0% (0/5)

| test_type | intended_class | iteration | variation_name | variant_used | MAE | score | ml_predicted_label | ml_confidence | ml_is_sitting | heuristic_flagged | heuristic_label | combined_predicted_label | combined_is_sitting |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sitting_variation | Sitting | 1 | upright_chair_sitting | chair | 5.2956 | 76.7375 | Sitting | 1 | True | True | Sitting | Sitting | True |
| sitting_variation | Sitting | 2 | slouched_chair_sitting | chair | 5.0075 | 77.8508 | Sitting | 1 | True | True | Sitting | Sitting | True |
| sitting_variation | Sitting | 3 | leaning_forward | chair | 10.6817 | 58.6207 | Sitting | 1 | True | True | Sitting | Sitting | True |
| sitting_variation | Sitting | 4 | leaning_back | chair | 12.4061 | 53.7781 | Sitting | 0.9083 | True | True | Sitting | Sitting | True |
| sitting_variation | Sitting | 5 | legs_crossed_on_chair | chair | 10.67 | 58.6548 | Sitting | 1 | True | True | Sitting | Sitting | True |
| sitting_variation | Sitting | 6 | floor_cross_legged | floor | 1.2986 | 93.7131 | Sitting | 1 | True | True | Sitting | Sitting | True |
| sitting_variation | Sitting | 7 | floor_straight_legs | chair | 25.6097 | 27.7903 | Sitting | 0.6 | True | False |  | Sitting | True |
| sitting_variation | Sitting | 8 | partial_view_knees_cut_off | chair | 2.4071 | 88.6607 | Sitting | 1 | True | True | Sitting / Step Back | Sitting / Step Back | True |
| sitting_variation | Sitting | 9 | partial_view_ankles_cut_off | chair | 2.9009 | 86.4982 | Sitting | 1 | True | True | Sitting | Sitting | True |
| sitting_variation | Sitting | 10 | sitting_with_arms_raised | chair | 35.8584 | 16.6474 | Tree | 0.7 | False | True | Sitting | Sitting | True |
| standing_reverse_fp | Warrior II | 0 | Warrior II_left_front | left_front | 0 | 100 | Warrior II | 0.9917 | False | False |  | Warrior II | False |
| standing_reverse_fp | Warrior II | 0 | Warrior II_right_front | right_front | 0 | 100 | Warrior II | 1 | False | False |  | Warrior II | False |
| standing_reverse_fp | Tree | 0 | Tree_left_standing | left_standing | 0 | 100 | Tree | 1 | False | False |  | Tree | False |
| standing_reverse_fp | Tree | 0 | Tree_right_standing | right_standing | 0 | 100 | Tree | 1 | False | False |  | Tree | False |
| standing_reverse_fp | Mountain | 0 | Mountain_canonical | canonical | 0 | 100 | Mountain | 1 | False | False |  | Mountain | False |

## 4. Key Findings

- **Highest Overall Sensitivity:** `Tree` showed the largest mean MAE (11.03 deg), with `left_shoulder_angle` as its most error-sensitive joint (mean dev 31.36 deg).
- **Classifier Breakdown (Warrior II):** First stopped recognising `Warrior II` at iteration 10 (severity 1.00); overall recognition rate = 90.0%, feedback hit rate = 30.0%.
- **Classifier Breakdown (Tree):** First stopped recognising `Tree` at iteration 7 (severity 0.70); overall recognition rate = 70.0%, feedback hit rate = 60.0%.
- **Classifier Breakdown (Mountain):** First stopped recognising `Mountain` at iteration 8 (severity 0.80); overall recognition rate = 70.0%, feedback hit rate = 10.0%.

## 5. Generated Plots

- `heatmap_warrior_ii.png`
- `heatmap_tree.png`
- `heatmap_mountain.png`
- `mae_score_vs_iteration.png`
- `radar_warrior_ii.png`
- `radar_tree.png`
- `radar_mountain.png`
- `joint_mean_deviation_bar.png`
- `severity_vs_mae_scatter.png`
- `classifier_confidence_vs_iteration.png`
- `confusion_matrix.png`
- `sitting_detection_rate.png`

## 6. Limitations

- Single-camera 2D monocular projection introduces planar foreshortening on limbs angled toward the camera.
- Reference profiles currently marked `synthetic_placeholder` reflect synthetic training means rather than human recordings.
- The trained classifier is left untouched; `generate_pose_feedback()` uses threshold rules on a subset of joints, which limits feedback hit rates for poses like Mountain.
