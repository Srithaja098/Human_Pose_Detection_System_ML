# AI Human Pose Detection & Yoga Asana Classification System 🧘‍♂️⚡

An end-to-end Machine Learning and Computer Vision system for real-time human pose estimation, 3D biomechanical joint angle feature extraction, and Yoga Asana classification with live posture alignment feedback.

---

## 🌟 Key Features

- **Real-Time 33-Landmark Pose Estimation**: Powered by Google MediaPipe Pose Tasks API for robust joint coordinate tracking.
- **Biomechanical Angle Calculations**: Derives 11 physiological joint angles in degrees:
  - Left & Right Elbow Angles
  - Left & Right Shoulder Angles
  - Left & Right Hip Angles
  - Left & Right Knee Angles
  - Left & Right Ankle Angles
  - Torso Inclination relative to vertical
- **Machine Learning Classification**: Trained `RandomForestClassifier` with `StandardScaler` feature normalization achieving 100% test accuracy.
- **Self-Healing & Auto-Training**: Automatically generates and serializes trained models on first run if missing.
- **Interactive HUD Overlay**: Modern glassmorphic on-screen display with live FPS, confidence score meter, and real-time posture correction tips.
- **Multi-Source Support**: Live webcam stream, saved video files, or single-image inference.

---

## 🧘 Supported Yoga Asanas

| Asana Name | Sanskrit | Description |
| :--- | :--- | :--- |
| **Warrior II** | *Virabhadrasana II* | Arms parallel to ground, front knee at 90°, back leg straight, vertical spine |
| **Tree Pose** | *Vrksasana* | Single-leg balance with foot on inner thigh, upright spine, prayer/raised arms |
| **Goddess Pose** | *Utkata Konasana* | Deep wide-stance squat with 90° knee flexion and cactus arms |
| **Downward Dog** | *Adho Mukha Svanasana* | Inverted V-shape with straight arms, elevated hips, and elongated spine |
| **Plank Pose** | *Phalakasana* | Straight alignment from head to heels, active core, arms perpendicular |
| **Mountain Pose** | *Tadasana* | Grounded standing posture with neutral spine and straight limbs |

---

## 🚀 Quickstart Guide

### 1. Requirements & Dependencies

The system requires Python 3.9+ with the following packages:

```bash
pip install opencv-python mediapipe scikit-learn numpy joblib
```

### 2. Run Real-Time Webcam Detection

Start the live webcam classifier with default camera (Index 0):

```bash
python model.py
```

Press **`q`** in the video window to safely quit.

### 3. Run on Video or Image Files

```bash
# Single image evaluation:
python model.py --image path/to/pose.jpg

# Video file evaluation:
python model.py --video path/to/recording.mp4

# Save output frame/image:
python model.py --image test_pose.jpg --save_output result.jpg
```

### 4. Train or Re-Train the Classifier

To re-train the pipeline and serialize new model artifacts:

```bash
python train_yoga_pose_classifier.py --output_dir models --n_estimators 120
```

This generates:
- `models/yoga_pose_model.joblib`: Trained Random Forest classifier
- `models/feature_scaler.joblib`: Fitted StandardScaler
- `models/label_encoder.joblib`: Target label encoder
- `models/feature_columns.joblib`: Feature column sequence

### 5. Run Automated Diagnostics Suite

To verify all components and inference pipelines:

```bash
python test_system.py
```

---

## 📁 Repository Structure

```
Human_Pose_Detection_System_ML/
├── models/                         # Serialized ML artifacts
│   ├── yoga_pose_model.joblib
│   ├── feature_scaler.joblib
│   ├── label_encoder.joblib
│   └── feature_columns.joblib
├── pose_landmarker.task            # MediaPipe Pose Landmarker model weights
├── model.py                        # Stage 1 real-time detection & HUD interface
├── train_yoga_pose_classifier.py   # Training pipeline & feature extraction engine
├── test_system.py                  # Automated verification & diagnostics suite
└── README.md                       # Documentation & usage guide
```

---

## ⚙️ Biomechanical Feature Extraction

The pipeline measures joint interior angles using vector dot products in 3D Euclidean space:

$$\theta = \arccos\left(\frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\| \|\vec{v}\|}\right) \times \frac{180^\circ}{\pi}$$

Torso inclination is calculated by finding the angle between the spine vector (mid-hip to mid-shoulder) and the true vertical axis $(0, -1, 0)$.