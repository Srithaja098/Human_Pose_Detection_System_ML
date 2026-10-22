# AI Human Pose Detection & Yoga Asana Tracking System (ML & Biomechanics)

A real-time Computer Vision and Machine Learning system for Yoga Asana recognition, biomechanical joint angle feature extraction, expert-reference posture correctness scoring, and day-by-day habit tracking. Developed for B.Tech thesis research.

---

## 1. System Architecture & Scope

The system is organized into two primary stages:

- **Stage 1 (Pose Recognition)**:
  - **Final Scored Asanas (3)**:
    1. **Warrior II** (*Virabhadrasana II*)
    2. **Tree Pose** (*Vrksasana*)
    3. **Mountain Pose** (*Tadasana*)
  - **Unscored Guidance States (2)**:
    - **Standing / Neutral** (*Samasthiti*): User is standing upright naturally; prompted to assume a target asana. Never scored.
    - **Sitting / Resting** (*Sukhasana / Chair / Desk Partial View*): User is seated or lower body is cut off; prompted to stand up and step back. Never scored.
- **Stage 2 (Posture Correctness Scoring & Progress Tracking)**:
  - Compares live joint angles against expert reference profiles (Mean $\mu_j$ and Standard Deviation $\sigma_j$).
  - Bilateral left/right mirror invariance (supports left- and right-sided poses).
  - Joint-specific signed correction tips.
  - Measures **Time-to-Correct** (seconds elapsed until pose is first held correctly for 2 continuous seconds).
  - Session hold completion (5.0s continuous hold = Success) or timeout (60.0s).
  - Appends performance metrics to `sessions.csv` and generates 3-panel progress graphs.

---

## 2. Mathematical Formulations (Thesis Definitions)

These formal definitions describe the Stage 2 posture evaluation engine:

### 2.1 Scored Biomechanical Joints ($N = 9$)
Nine interior joint angles and spine orientation are tracked:
- Left & Right Elbow Angles ($\theta_{\text{LE}}$, $\theta_{\text{RE}}$)
- Left & Right Shoulder Angles ($\theta_{\text{LS}}$, $\theta_{\text{RS}}$)
- Left & Right Hip Angles ($\theta_{\text{LH}}$, $\theta_{\text{RH}}$)
- Left & Right Knee Angles ($\theta_{\text{LK}}$, $\theta_{\text{RK}}$)
- Torso Inclination with respect to vertical ($\theta_{\text{Torso}}$)

> **Note on Ankle Angles**: While left and right ankle angles are measured, they are **excluded from correctness scoring** because 2D monocular perspective projection and foot foreshortening introduce significant noise.

### 2.2 Joint Angle Calculation
The interior angle at joint vertex $B$ formed by adjacent landmarks $A$ and $C$ is computed via vector dot products:

$$\theta = \arccos\left(\frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\| \|\vec{v}\|}\right) \times \frac{180^\circ}{\pi}$$

where $\vec{u} = \vec{A} - \vec{B}$ and $\vec{v} = \vec{C} - \vec{B}$.

### 2.3 Joint Tolerance
For joint $j$ with expert mean $\mu_j$ and sample standard deviation $\sigma_j$:

$$\text{tol}_j = \max(10.0^\circ, 2 \times \sigma_j)$$

If $\sigma_j$ is unknown or the profile is provisional/single-frame, default tolerance is $\text{tol}_j = 15.0^\circ$.

### 2.4 Joint Deviation
$$\text{dev}_j = |\theta_{\text{user}, j} - \mu_{\text{expert}, j}|$$

### 2.5 Piecewise Linear Joint Scoring ($0 - 100$)
Each valid joint is scored on a continuous scale:

$$\text{Score}_j = \begin{cases} 
100.0 & \text{if } \text{dev}_j \le \text{tol}_j \\ 
100.0 \times \left(1 - \frac{\text{dev}_j - \text{tol}_j}{2 \times \text{tol}_j}\right) & \text{if } \text{tol}_j < \text{dev}_j \le 3 \times \text{tol}_j \\ 
0.0 & \text{if } \text{dev}_j > 3 \times \text{tol}_j 
\end{cases}$$

### 2.6 Overall Pose Score
For $N$ visible joints meeting the visibility threshold ($\text{visibility} \ge 0.5$, requiring $N \ge 7$ of 9):

$$\text{PoseScore} = \frac{1}{N} \sum_{j=1}^{N} \text{Score}_j$$

### 2.7 Posture Correctness Condition
A user's posture is classified as **Correct** if and only if:

$$\text{is\_correct} = (\text{PoseScore} \ge 80.0) \land \left(\forall j, \text{dev}_j \le 1.5 \times \text{tol}_j\right)$$

### 2.8 Bilateral Mirror Symmetry
Yoga poses can be executed facing left or right (e.g., Tree pose on the left vs. right foot). The user is evaluated against both the canonical reference profile and its left/right mirrored reflection:

$$\text{Profile}_{\text{mirrored}} = \text{Swap}\left(\theta_{\text{Left}}, \theta_{\text{Right}}\right)$$

The orientation producing the higher pose score is retained.

### 2.9 Exponential Moving Average (EMA) Angle Smoothing
To eliminate single-frame coordinate jitter from MediaPipe, angles are smoothed over consecutive frames:

$$\theta_{\text{smooth}}^{(t)} = \alpha \cdot \theta_{\text{raw}}^{(t)} + (1 - \alpha) \cdot \theta_{\text{smooth}}^{(t-1)}$$

where $\alpha = 0.30$.

### 2.10 Time-to-Correct ($t_{\text{correct}}$)
Following a 5-second countdown ($t = 0$), $t_{\text{correct}}$ is defined as the elapsed time in seconds until the user first achieves $\text{is\_correct} = \text{True}$ and sustains it for at least $2.0$ continuous seconds.

---

## 3. Quickstart & Command Guide

### 3.1 Dependencies
Install the required Python packages (Python 3.9+ supported):

```bash
pip install opencv-python mediapipe scikit-learn numpy pandas matplotlib joblib
```

### 3.2 Automated Diagnostics Suite
Verify all components, angle math, model loading, canonical inference, and Stage 2 tracking simulation tests:

```bash
python test_system.py
```

---

## 4. How to Run Each Script

### 4.1 Live Stage 1 Recognition
Run real-time webcam pose detection with the default glassmorphic HUD:

```bash
python model.py
```

Run on a video file or single image:
```bash
python model.py --video path/to/video.mp4
python model.py --image path/to/image.jpg
```

Toggle pure ML classification without geometric overrides:
```bash
python model.py --no_rules
```

### 4.2 Live Stage 2 Posture Tracking & Scoring
To track posture correctness, time-to-correct, and log daily sessions for an asana:

```bash
# Track Warrior II for User1
python model.py --track "Warrior II" --user "User1"

# Track Tree Pose
python model.py --track "Tree" --user "User1"

# Track Mountain Pose
python model.py --track "Mountain" --user "User1"
```

**Controls in Tracking Mode**:
- **SPACEBAR**: Starts the 5-second countdown. Step into position.
- **`q`**: Safely quit and close the window.

### 4.3 Recording an Instructor (`record_expert.py`)
To record a real yoga teacher holding an asana to replace provisional starter reference profiles:

```bash
# Records 150 clean, full-body frames into expert_angles.csv
python record_expert.py --asana "Warrior II" --samples 150
python record_expert.py --asana "Tree" --samples 150
python record_expert.py --asana "Mountain" --samples 150
```
*Frames are only recorded when all scored joints have landmark visibility $\ge 0.5$.*

### 4.4 Building Reference Profiles (`build_reference.py`)
Compile the recorded expert CSV into `reference_profiles.json`:

```bash
# Update reference_profiles.json from recorded CSV
python build_reference.py --csv expert_angles.csv --asana "Warrior II"

# Or initialize standard provisional starter profiles:
python build_reference.py --init_starter
```

### 4.5 Generating Progress Reports (`progress_report.py`)
Analyze a user's session history in `sessions.csv`:

```bash
# Generate report for all asanas practiced by User1
python progress_report.py --user "User1"

# Generate report for a specific asana
python progress_report.py --user "User1" --asana "Warrior II"
```

Outputs:
1. **3-Panel Stacked PNG Plot** (`progress_<user>_<asana>.png`):
   - Panel 1: Pose accuracy score per session with daily rolling average.
   - Panel 2: Time-to-correct across sessions (lower is better; timeouts marked with red 'X').
   - Panel 3: Trajectory of the 3 joints furthest from expert alignment.
2. **Terminal Statistical Summary**:
   - First 3 sessions vs. last 3 sessions comparison (Score, Time-to-Correct, Success Rate).
   - Most improved joint vs. joint needing the most work.
   - Trend reliability guard: flags if $< 6$ sessions are available.
3. **Tip Effectiveness Lag-1 Audit**:
   - Evaluates whether tips displayed in session $k$ led to reduced deviation in session $k+1$.

### 4.6 Honest Evaluation on Real Images (`evaluate_on_real_images.py`)
Evaluate the classifier against real, un-augmented photos:

```bash
# Evaluate with hybrid rules (ML + heuristics):
python evaluate_on_real_images.py --data_dir path/to/dataset

# Evaluate pure ML classifier (no heuristic overrides):
python evaluate_on_real_images.py --data_dir path/to/dataset --no_rules

# Specify Mountain pose definition variant:
python evaluate_on_real_images.py --data_dir path/to/dataset --mountain_def arms_at_sides
```

Expected directory structure:
```
dataset/
├── warrior2/       # or 'warrior_ii', 'virabhadrasana_ii'
│   ├── img01.jpg
│   └── img02.jpg
├── tree/           # or 'vrksasana'
│   └── img01.jpg
└── mountain/       # (Optional: user photos or Yoga-82; skipped if absent)
    └── img01.jpg
```

---

## 5. Honest Scientific Disclosures & Limitations

For academic transparency and thesis defense, the following methodological boundaries are explicitly stated:

### 5.1 Synthetic Training Data & Reported Accuracy
- The baseline Random Forest model in `train_yoga_pose_classifier.py` was trained on synthetic angle vectors sampled from Gaussian distributions.
- **The reported 100% test accuracy and cross-validation accuracy are purely synthetic benchmarks and DO NOT represent real-world generalization.**
- To report genuine scientific accuracy in a thesis, run `evaluate_on_real_images.py` against external datasets (e.g. Kaggle Yoga Poses, Yoga-82, or self-recorded test sets) and report ML-only (`--no_rules`) vs. hybrid accuracy separately.

### 5.2 Missing Mountain Pose in Public Datasets
- The popular Kaggle Yoga Poses Dataset (`niharika41298`) contains `warrior2` and `tree` poses, but **contains no Mountain / Tadasana pose**.
- Mountain pose data must be obtained either through:
  1. Self-recorded photos/videos using `record_expert.py`.
  2. Subsets of the Yoga-82 dataset.
- `evaluate_on_real_images.py` explicitly skips missing classes without fabricating data.

### 5.3 Mountain Pose Definition (Tadasana)
- **Standard Definition (`arms_at_sides`)**: Arms aligned straight along the sides of the torso, spine erect, feet grounded.
- **Overhead Variant (`arms_overhead`)**: Fingers interlocked and stretched vertically above the head, balancing on toes.
- *Limitation*: In a monocular 2D camera, interlocked fingers cannot be verified reliably via 2D wrist distance alone, and heel lift is susceptible to camera pitch. The definition is configurable via `--mountain_def` and is noted as **to be confirmed with instructor**.
- The confusion matrix specifically isolates the **Mountain vs. Standing** pair.

### 5.4 Monocular 2D Camera Constraints
- **Depth Ambiguity**: 2D coordinate projections lose depth information along the optical axis ($z$-axis).
- **Planar Foreshortening**: When limbs point toward or away from the camera lens, their apparent length shortens, skewing 2D interior angle calculations.
- **Occlusions**: If a body part is self-occluded (e.g., back leg in profile Warrior II), landmark visibility drops. The pipeline requires at least 7 of 9 scored joints with visibility $\ge 0.5$ to compute a score.

### 5.5 Provisional Starter Reference Data
- Initial reference profiles in `reference_profiles.json` are marked with `"provisional": true` and have a default tolerance of $15.0^\circ$.
- These must be replaced with live instructor recordings via `record_expert.py` before final thesis submission.

---

## 6. Repository Layout

```
Human_Pose_Detection_System_ML/
├── pose_config.py                  # Central configuration (classes, joints, thresholds)
├── model.py                        # Real-time detection HUD & --track mode
├── tracking.py                     # Stage 2 scoring, tips, time-to-correct & state machine
├── record_expert.py                # Live webcam instructor recording tool
├── build_reference.py              # Compiles expert_angles.csv -> reference_profiles.json
├── reference_profiles.json         # Mean, std, and tolerances per joint for each asana
├── progress_report.py              # 3-panel PNG generator & habit trend analysis
├── evaluate_on_real_images.py      # Real image benchmark suite with --no_rules switch
├── train_yoga_pose_classifier.py   # Training pipeline & feature extraction engine
├── test_system.py                  # Full verification & diagnostics suite (Steps 1-6)
├── test_tracking.py                # 8 simulated angle stream verification tests
├── pose_landmarker.task            # MediaPipe Pose Landmarker Lite model weights
├── models/                         # Serialized ML artifacts
│   ├── yoga_pose_model.joblib
│   ├── feature_scaler.joblib
│   ├── label_encoder.joblib
│   └── feature_columns.joblib
└── README.md                       # Thesis documentation & usage guide
```