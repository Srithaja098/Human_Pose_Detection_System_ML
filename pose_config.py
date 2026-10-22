"""
Central Configuration for Human Pose Detection & Yoga Asana System
===================================================================
Defines the final set of recognizable asana classes, unscored guidance states,
scored biomechanical joints, landmark symmetry mappings, and Stage 2 tracking
constants.
"""

# =====================================================================
# Pose Classes & Guidance States
# =====================================================================
# Stage 1 Final set of scored asanas confirmed with professor:
SCORED_ASANAS = [
    "Warrior II",
    "Tree",
    "Mountain",
]

# Unscored guidance states: meaning "not in a pose yet".
# These NEVER receive a correctness score or posture tracking score.
UNSCORED_STATES = [
    "Standing",
    "Sitting",
    "Sitting / Step Back",
]

ALL_CLASSES = SCORED_ASANAS + ["Standing", "Sitting"]

# Mountain Pose Definition Variants:
# - "arms_at_sides": Traditional Tadasana with arms aligned at sides of torso.
# - "arms_overhead": Upward stretch variant with arms extended above head.
DEFAULT_MOUNTAIN_DEF = "arms_overhead"
MOUNTAIN_DEFINITIONS = ["arms_at_sides", "arms_overhead"]

# =====================================================================
# Scored Biomechanical Joints (9 joints total)
# =====================================================================
# Ankles are excluded from correctness scoring because 2D planar projection
# and foot foreshortening make ankle angles too noisy for reliable evaluation.
SCORED_JOINTS = [
    "left_elbow_angle",
    "right_elbow_angle",
    "left_shoulder_angle",
    "right_shoulder_angle",
    "left_hip_angle",
    "right_hip_angle",
    "left_knee_angle",
    "right_knee_angle",
    "torso_inclination",
]

# Symmetrical joint pairs used for left/right mirror evaluation
MIRROR_JOINT_MAP = {
    "left_elbow_angle": "right_elbow_angle",
    "right_elbow_angle": "left_elbow_angle",
    "left_shoulder_angle": "right_shoulder_angle",
    "right_shoulder_angle": "left_shoulder_angle",
    "left_hip_angle": "right_hip_angle",
    "right_hip_angle": "left_hip_angle",
    "left_knee_angle": "right_knee_angle",
    "right_knee_angle": "left_knee_angle",
    "torso_inclination": "torso_inclination",
}

# Mapping from feature name to joint landmarks required for visibility checking
# Landmarks are based on MediaPipe 33-landmark indices
JOINT_LANDMARK_MAP = {
    "left_elbow_angle": [11, 13, 15],       # left_shoulder, left_elbow, left_wrist
    "right_elbow_angle": [12, 14, 16],     # right_shoulder, right_elbow, right_wrist
    "left_shoulder_angle": [23, 11, 13],    # left_hip, left_shoulder, left_elbow
    "right_shoulder_angle": [24, 12, 14],   # right_hip, right_shoulder, right_elbow
    "left_hip_angle": [11, 23, 25],         # left_shoulder, left_hip, left_knee
    "right_hip_angle": [12, 24, 26],        # right_shoulder, right_hip, right_knee
    "left_knee_angle": [23, 25, 27],        # left_hip, left_knee, left_ankle
    "right_knee_angle": [24, 26, 28],       # right_hip, right_knee, right_ankle
    "torso_inclination": [11, 12, 23, 24],  # shoulders and hips
}

# =====================================================================
# Scoring & Tracking Named Constants
# =====================================================================
# Minimum overall score required for a pose to be judged "Correct" (out of 100)
SCORE_THRESHOLD = 80.0

# Multiplier for tolerance: no joint may exceed this multiplier for pose to be correct
MAX_TOLERANCE_MULTIPLIER = 1.5

# Multiplier for tolerance beyond which joint score drops to 0
ZERO_SCORE_MULTIPLIER = 3.0

# Minimum allowed tolerance in degrees (even if expert SD is extremely small)
MIN_TOLERANCE_DEG = 10.0

# Default tolerance in degrees if SD is unknown or single-frame provisional profile
DEFAULT_TOLERANCE_DEG = 15.0

# Minimum MediaPipe landmark visibility required to score a joint
MIN_LANDMARK_VISIBILITY = 0.5

# Minimum number of valid joints required to compute a pose score (out of 9)
MIN_SCORED_JOINTS = 7
TOTAL_SCORED_JOINTS = 9

# Time parameters (in seconds)
COUNTDOWN_TIME_S = 5.0      # Countdown before session timer begins
HOLD_CORRECT_TIME_S = 2.0   # Hold duration required to register time_to_correct
HOLD_SUCCESS_TIME_S = 5.0   # Continuous hold duration required for session success
SESSION_TIMEOUT_S = 60.0    # Maximum session duration before timeout

# Exponential Moving Average (EMA) smoothing coefficient for raw joint angles
# angle_smooth = alpha * angle_raw + (1 - alpha) * angle_prev
EMA_ALPHA = 0.3

# Default paths
DEFAULT_REFERENCE_FILE = "reference_profiles.json"
DEFAULT_SESSIONS_CSV = "sessions.csv"
