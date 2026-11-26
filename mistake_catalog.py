"""
Mistake Catalog & Active Scope Configuration for Deviation Analysis
===================================================================
Defines the active pose scope (4 classes: 3 asanas + 1 non-pose state),
the structured mistake catalog (>= 10 ordered mistakes per active asana),
and the 10 Sitting variations for robustness testing.

To expand scope later (e.g., adding Goddess, Downward Dog, or Plank),
update ACTIVE_POSES / ACTIVE_NON_POSES below and add corresponding catalog entries.
"""

from typing import Dict, List, Any

# =====================================================================
# Active Scope Configuration (Single Place to Edit)
# =====================================================================
ACTIVE_POSES: List[str] = ["Warrior II", "Tree", "Mountain"]
ACTIVE_NON_POSES: List[str] = ["Sitting"]
ACTIVE_CLASSES: List[str] = ACTIVE_POSES + ACTIVE_NON_POSES

# Left/right symmetrical feature pairs for mirror invariance
MIRROR_FEATURE_MAP: Dict[str, str] = {
    "left_elbow_angle": "right_elbow_angle",
    "right_elbow_angle": "left_elbow_angle",
    "left_shoulder_angle": "right_shoulder_angle",
    "right_shoulder_angle": "left_shoulder_angle",
    "left_hip_angle": "right_hip_angle",
    "right_hip_angle": "left_hip_angle",
    "left_knee_angle": "right_knee_angle",
    "right_knee_angle": "left_knee_angle",
    "left_ankle_angle": "right_ankle_angle",
    "right_ankle_angle": "left_ankle_angle",
    "torso_inclination": "torso_inclination",
}

# =====================================================================
# Asana Mistake Catalog (10 ordered mistakes per active asana)
# =====================================================================
# Ordered from mildest to most severe so that across iterations 1..N
# with severity = iter / N, total deviation grows monotonically.
MISTAKE_CATALOG: Dict[str, List[Dict[str, Any]]] = {
    "Warrior II": [
        {
            "name": "elbows_bent",
            "description": "Elbows bent instead of arms extended straight",
            "affected_features": ["left_elbow_angle", "right_elbow_angle"],
            "delta_deg": {"left_elbow_angle": -25.0, "right_elbow_angle": -25.0},
            "feedback_keywords": ["arm", "elbow", "straight", "parallel"],
        },
        {
            "name": "narrow_stance",
            "description": "Stance too narrow (front hip and knee angles too open)",
            "affected_features": ["left_hip_angle", "left_knee_angle"],
            "delta_deg": {"left_hip_angle": 25.0, "left_knee_angle": 30.0},
            "feedback_keywords": ["knee", "stance", "bend", "front"],
        },
        {
            "name": "arms_raised_too_high",
            "description": "Arms raised above horizontal shoulder level",
            "affected_features": ["left_shoulder_angle", "right_shoulder_angle"],
            "delta_deg": {"left_shoulder_angle": 35.0, "right_shoulder_angle": 35.0},
            "feedback_keywords": ["arm", "parallel", "shoulder"],
        },
        {
            "name": "torso_leaning_sideways",
            "description": "Torso leaning sideways over front leg instead of staying vertical",
            "affected_features": ["torso_inclination", "left_hip_angle", "left_shoulder_angle"],
            "delta_deg": {
                "torso_inclination": 25.0,
                "left_hip_angle": -25.0,
                "left_shoulder_angle": -25.0,
            },
            "feedback_keywords": ["torso", "spine", "straight", "arm", "parallel"],
        },
        {
            "name": "back_knee_bent",
            "description": "Back knee bent instead of locked straight",
            "affected_features": ["right_knee_angle", "right_hip_angle"],
            "delta_deg": {"right_knee_angle": -60.0, "right_hip_angle": -20.0},
            "feedback_keywords": ["knee", "back", "straight", "leg"],
        },
        {
            "name": "arms_drooping",
            "description": "Arms drooping below horizontal (shoulder angle ~60 deg)",
            "affected_features": ["left_shoulder_angle", "right_shoulder_angle"],
            "delta_deg": {"left_shoulder_angle": -50.0, "right_shoulder_angle": -50.0},
            "feedback_keywords": ["arm", "parallel", "shoulder"],
        },
        {
            "name": "front_knee_over_bent",
            "description": "Front knee over-bent past 90 degrees (<70 deg)",
            "affected_features": ["left_knee_angle", "left_hip_angle", "left_ankle_angle"],
            "delta_deg": {
                "left_knee_angle": -60.0,
                "left_hip_angle": -45.0,
                "left_ankle_angle": -15.0,
            },
            "feedback_keywords": ["knee", "front", "90"],
        },
        {
            "name": "front_knee_under_bent",
            "description": "Front knee under-bent (knee ~140-170 deg instead of ~90 deg)",
            "affected_features": ["left_knee_angle", "left_hip_angle", "left_ankle_angle"],
            "delta_deg": {
                "left_knee_angle": 60.0,
                "left_hip_angle": 45.0,
                "left_ankle_angle": 20.0,
            },
            "feedback_keywords": ["knee", "bend", "front", "90"],
        },
        {
            "name": "torso_leaning_forward",
            "description": "Torso leaning forward aggressively",
            "affected_features": ["torso_inclination", "left_hip_angle", "right_hip_angle"],
            "delta_deg": {
                "torso_inclination": 45.0,
                "left_hip_angle": -45.0,
                "right_hip_angle": -45.0,
            },
            "feedback_keywords": ["torso", "spine", "hip", "upright", "straight"],
        },
        {
            "name": "head_shoulders_collapsed",
            "description": "Head and shoulders collapsed inward with dropped, bent arms",
            "affected_features": [
                "left_shoulder_angle",
                "right_shoulder_angle",
                "left_elbow_angle",
                "right_elbow_angle",
                "torso_inclination",
            ],
            "delta_deg": {
                "left_shoulder_angle": -45.0,
                "right_shoulder_angle": -45.0,
                "left_elbow_angle": -35.0,
                "right_elbow_angle": -35.0,
                "torso_inclination": 35.0,
            },
            "feedback_keywords": ["arm", "parallel", "shoulder", "torso"],
        },
    ],
    "Tree": [
        {
            "name": "shoulders_hunched",
            "description": "Shoulders hunched upward and elbows slightly bent",
            "affected_features": [
                "left_shoulder_angle",
                "right_shoulder_angle",
                "left_elbow_angle",
                "right_elbow_angle",
            ],
            "delta_deg": {
                "left_shoulder_angle": -20.0,
                "right_shoulder_angle": -20.0,
                "left_elbow_angle": -15.0,
                "right_elbow_angle": -15.0,
            },
            "feedback_keywords": ["arm", "palm", "stretch", "shoulder", "overhead"],
        },
        {
            "name": "standing_knee_bent",
            "description": "Standing leg knee bent instead of straight",
            "affected_features": ["left_knee_angle", "left_hip_angle", "left_ankle_angle"],
            "delta_deg": {
                "left_knee_angle": -45.0,
                "left_hip_angle": -25.0,
                "left_ankle_angle": -10.0,
            },
            "feedback_keywords": ["knee", "standing", "leg", "straight"],
        },
        {
            "name": "raised_foot_on_knee_joint",
            "description": "Raised foot resting directly on the knee joint",
            "affected_features": ["right_knee_angle", "right_hip_angle", "right_ankle_angle"],
            "delta_deg": {
                "right_knee_angle": 35.0,
                "right_hip_angle": 25.0,
                "right_ankle_angle": 25.0,
            },
            "feedback_keywords": ["foot", "thigh", "calf", "knee"],
        },
        {
            "name": "torso_leaning_sideways",
            "description": "Torso leaning laterally to compensate for balance",
            "affected_features": ["torso_inclination", "left_hip_angle", "right_hip_angle"],
            "delta_deg": {
                "torso_inclination": 35.0,
                "left_hip_angle": -30.0,
                "right_hip_angle": 25.0,
            },
            "feedback_keywords": ["spine", "torso", "straight", "lengthen"],
        },
        {
            "name": "hip_collapsed",
            "description": "Standing hip collapsed outward",
            "affected_features": ["left_hip_angle", "right_hip_angle", "torso_inclination"],
            "delta_deg": {
                "left_hip_angle": -40.0,
                "right_hip_angle": -35.0,
                "torso_inclination": 25.0,
            },
            "feedback_keywords": ["hip", "spine", "lengthen", "straight"],
        },
        {
            "name": "foot_not_raised_high_enough",
            "description": "Raised foot too low near ankle (bent knee ~100+ deg)",
            "affected_features": ["right_knee_angle", "right_hip_angle", "right_ankle_angle"],
            "delta_deg": {
                "right_knee_angle": 55.0,
                "right_hip_angle": 35.0,
                "right_ankle_angle": 25.0,
            },
            "feedback_keywords": ["foot", "thigh", "calf", "knee"],
        },
        {
            "name": "arms_not_overhead",
            "description": "Arms held out wide instead of joined overhead",
            "affected_features": ["left_shoulder_angle", "right_shoulder_angle"],
            "delta_deg": {"left_shoulder_angle": -75.0, "right_shoulder_angle": -75.0},
            "feedback_keywords": ["arm", "palm", "overhead", "stretch"],
        },
        {
            "name": "loss_of_balance",
            "description": "Large torso sway and hip/arm compensation from balance loss",
            "affected_features": [
                "torso_inclination",
                "left_hip_angle",
                "right_hip_angle",
                "left_shoulder_angle",
                "right_shoulder_angle",
            ],
            "delta_deg": {
                "torso_inclination": 45.0,
                "left_hip_angle": -35.0,
                "right_hip_angle": -30.0,
                "left_shoulder_angle": -30.0,
                "right_shoulder_angle": -30.0,
            },
            "feedback_keywords": ["spine", "lengthen", "arm", "straight"],
        },
        {
            "name": "arms_dropped",
            "description": "Arms dropped completely down toward sides",
            "affected_features": ["left_shoulder_angle", "right_shoulder_angle"],
            "delta_deg": {"left_shoulder_angle": -125.0, "right_shoulder_angle": -125.0},
            "feedback_keywords": ["arm", "palm", "overhead", "stretch"],
        },
        {
            "name": "elbows_bent_prayer",
            "description": "Hands held in chest prayer position instead of overhead stretch",
            "affected_features": [
                "left_elbow_angle",
                "right_elbow_angle",
                "left_shoulder_angle",
                "right_shoulder_angle",
            ],
            "delta_deg": {
                "left_elbow_angle": -95.0,
                "right_elbow_angle": -95.0,
                "left_shoulder_angle": -115.0,
                "right_shoulder_angle": -115.0,
            },
            "feedback_keywords": ["arm", "palm", "overhead", "stretch", "elbow"],
        },
    ],
    "Mountain": [
        {
            "name": "one_knee_bent",
            "description": "One knee bent instead of both legs locked straight",
            "affected_features": ["left_knee_angle"],
            "delta_deg": {"left_knee_angle": -50.0},
            "feedback_keywords": ["knee", "leg", "straight"],
        },
        {
            "name": "shoulders_raised",
            "description": "Shoulders shrugged toward ears with slight arm bend",
            "affected_features": [
                "left_shoulder_angle",
                "right_shoulder_angle",
                "left_elbow_angle",
                "right_elbow_angle",
            ],
            "delta_deg": {
                "left_shoulder_angle": -20.0,
                "right_shoulder_angle": -20.0,
                "left_elbow_angle": -15.0,
                "right_elbow_angle": -15.0,
            },
            "feedback_keywords": ["shoulder", "arm", "stretch"],
        },
        {
            "name": "hips_tilted",
            "description": "Asymmetric hip tilt shifting weight to one side",
            "affected_features": ["left_hip_angle", "right_hip_angle", "torso_inclination"],
            "delta_deg": {
                "left_hip_angle": -35.0,
                "right_hip_angle": -15.0,
                "torso_inclination": 25.0,
            },
            "feedback_keywords": ["hip", "torso", "spine", "straight"],
        },
        {
            "name": "leaning_back",
            "description": "Torso leaning backward off vertical alignment",
            "affected_features": ["torso_inclination", "left_hip_angle", "right_hip_angle"],
            "delta_deg": {
                "torso_inclination": 30.0,
                "left_hip_angle": -25.0,
                "right_hip_angle": -25.0,
            },
            "feedback_keywords": ["torso", "spine", "straight", "upright"],
        },
        {
            "name": "slouching",
            "description": "Forward upper-body slouch (torso inclination > 15 deg)",
            "affected_features": ["torso_inclination", "left_hip_angle", "right_hip_angle"],
            "delta_deg": {
                "torso_inclination": 35.0,
                "left_hip_angle": -30.0,
                "right_hip_angle": -30.0,
            },
            "feedback_keywords": ["torso", "spine", "straight", "upright"],
        },
        {
            "name": "bent_knees",
            "description": "Both knees bent instead of straight upright legs",
            "affected_features": ["left_knee_angle", "right_knee_angle"],
            "delta_deg": {"left_knee_angle": -55.0, "right_knee_angle": -55.0},
            "feedback_keywords": ["knee", "leg", "straight"],
        },
        {
            "name": "arms_bent",
            "description": "Elbows bent instead of straight upward arm extension",
            "affected_features": ["left_elbow_angle", "right_elbow_angle"],
            "delta_deg": {"left_elbow_angle": -70.0, "right_elbow_angle": -70.0},
            "feedback_keywords": ["arm", "elbow", "stretch", "straight"],
        },
        {
            "name": "arms_away_from_body",
            "description": "Arms drifting outward horizontally instead of aligned vertically",
            "affected_features": ["left_shoulder_angle", "right_shoulder_angle"],
            "delta_deg": {"left_shoulder_angle": -95.0, "right_shoulder_angle": -95.0},
            "feedback_keywords": ["arm", "stretch", "overhead", "shoulder"],
        },
        {
            "name": "arms_raised",
            "description": "Arms dropped out of overhead Tadasana position (arm height mismatch)",
            "affected_features": ["left_shoulder_angle", "right_shoulder_angle"],
            "delta_deg": {"left_shoulder_angle": -135.0, "right_shoulder_angle": -135.0},
            "feedback_keywords": ["arm", "stretch", "overhead", "finger"],
        },
        {
            "name": "partial_squat",
            "description": "Deep knee and hip flexion forming a partial squat",
            "affected_features": [
                "left_knee_angle",
                "right_knee_angle",
                "left_hip_angle",
                "right_hip_angle",
                "torso_inclination",
            ],
            "delta_deg": {
                "left_knee_angle": -70.0,
                "right_knee_angle": -70.0,
                "left_hip_angle": -65.0,
                "right_hip_angle": -65.0,
                "torso_inclination": 25.0,
            },
            "feedback_keywords": ["knee", "hip", "straight", "stand"],
        },
    ],
}

# =====================================================================
# Sitting Robustness Variations (10 real-world sitting / partial scenarios)
# =====================================================================
SITTING_VARIATIONS: List[Dict[str, Any]] = [
    {
        "name": "upright_chair_sitting",
        "description": "Upright posture seated on a chair, full body visible",
        "base_variant": "chair",
        "angles": {
            "left_elbow_angle": 105.0, "right_elbow_angle": 105.0,
            "left_shoulder_angle": 25.0, "right_shoulder_angle": 25.0,
            "left_hip_angle": 95.0, "right_hip_angle": 95.0,
            "left_knee_angle": 90.0, "right_knee_angle": 90.0,
            "left_ankle_angle": 90.0, "right_ankle_angle": 90.0,
            "torso_inclination": 5.0,
        },
        "landmark_profile": {"mid_hip_y": 0.62, "knee_y": 0.78, "ankle_y": 0.92, "knee_vis": 0.95, "ankle_vis": 0.95},
    },
    {
        "name": "slouched_chair_sitting",
        "description": "Slouched upper torso seated on a chair",
        "base_variant": "chair",
        "angles": {
            "left_elbow_angle": 100.0, "right_elbow_angle": 100.0,
            "left_shoulder_angle": 30.0, "right_shoulder_angle": 30.0,
            "left_hip_angle": 110.0, "right_hip_angle": 110.0,
            "left_knee_angle": 95.0, "right_knee_angle": 95.0,
            "left_ankle_angle": 90.0, "right_ankle_angle": 90.0,
            "torso_inclination": 18.0,
        },
        "landmark_profile": {"mid_hip_y": 0.64, "knee_y": 0.80, "ankle_y": 0.92, "knee_vis": 0.95, "ankle_vis": 0.95},
    },
    {
        "name": "leaning_forward",
        "description": "Seated on chair and leaning forward toward desk",
        "base_variant": "chair",
        "angles": {
            "left_elbow_angle": 85.0, "right_elbow_angle": 85.0,
            "left_shoulder_angle": 40.0, "right_shoulder_angle": 40.0,
            "left_hip_angle": 72.0, "right_hip_angle": 72.0,
            "left_knee_angle": 92.0, "right_knee_angle": 92.0,
            "left_ankle_angle": 90.0, "right_ankle_angle": 90.0,
            "torso_inclination": 28.0,
        },
        "landmark_profile": {"mid_hip_y": 0.64, "knee_y": 0.80, "ankle_y": 0.92, "knee_vis": 0.95, "ankle_vis": 0.95},
    },
    {
        "name": "leaning_back",
        "description": "Reclined back on a chair",
        "base_variant": "chair",
        "angles": {
            "left_elbow_angle": 115.0, "right_elbow_angle": 115.0,
            "left_shoulder_angle": 28.0, "right_shoulder_angle": 28.0,
            "left_hip_angle": 122.0, "right_hip_angle": 122.0,
            "left_knee_angle": 100.0, "right_knee_angle": 100.0,
            "left_ankle_angle": 95.0, "right_ankle_angle": 95.0,
            "torso_inclination": 20.0,
        },
        "landmark_profile": {"mid_hip_y": 0.62, "knee_y": 0.78, "ankle_y": 0.92, "knee_vis": 0.95, "ankle_vis": 0.95},
    },
    {
        "name": "legs_crossed_on_chair",
        "description": "Seated on chair with one leg crossed over the other",
        "base_variant": "chair",
        "angles": {
            "left_elbow_angle": 105.0, "right_elbow_angle": 105.0,
            "left_shoulder_angle": 25.0, "right_shoulder_angle": 25.0,
            "left_hip_angle": 90.0, "right_hip_angle": 80.0,
            "left_knee_angle": 92.0, "right_knee_angle": 55.0,
            "left_ankle_angle": 90.0, "right_ankle_angle": 80.0,
            "torso_inclination": 6.0,
        },
        "landmark_profile": {"mid_hip_y": 0.62, "knee_y": 0.78, "ankle_y": 0.90, "knee_vis": 0.90, "ankle_vis": 0.85},
    },
    {
        "name": "floor_cross_legged",
        "description": "Cross-legged floor sitting (Sukhasana)",
        "base_variant": "floor",
        "angles": {
            "left_elbow_angle": 135.0, "right_elbow_angle": 135.0,
            "left_shoulder_angle": 25.0, "right_shoulder_angle": 25.0,
            "left_hip_angle": 65.0, "right_hip_angle": 65.0,
            "left_knee_angle": 45.0, "right_knee_angle": 45.0,
            "left_ankle_angle": 80.0, "right_ankle_angle": 80.0,
            "torso_inclination": 4.0,
        },
        "landmark_profile": {"mid_hip_y": 0.76, "knee_y": 0.84, "ankle_y": 0.90, "knee_vis": 0.95, "ankle_vis": 0.95},
    },
    {
        "name": "floor_straight_legs",
        "description": "Seated on floor with legs extended straight forward (Dandasana)",
        "base_variant": "floor",
        "angles": {
            "left_elbow_angle": 140.0, "right_elbow_angle": 140.0,
            "left_shoulder_angle": 25.0, "right_shoulder_angle": 25.0,
            "left_hip_angle": 88.0, "right_hip_angle": 88.0,
            "left_knee_angle": 170.0, "right_knee_angle": 170.0,
            "left_ankle_angle": 90.0, "right_ankle_angle": 90.0,
            "torso_inclination": 6.0,
        },
        "landmark_profile": {"mid_hip_y": 0.78, "knee_y": 0.82, "ankle_y": 0.86, "knee_vis": 0.95, "ankle_vis": 0.95},
    },
    {
        "name": "partial_view_knees_cut_off",
        "description": "Desk sitting close to camera with knees cut off below frame",
        "base_variant": "chair",
        "angles": {
            "left_elbow_angle": 90.0, "right_elbow_angle": 90.0,
            "left_shoulder_angle": 32.0, "right_shoulder_angle": 32.0,
            "left_hip_angle": 98.0, "right_hip_angle": 98.0,
            "left_knee_angle": 90.0, "right_knee_angle": 90.0,
            "left_ankle_angle": 90.0, "right_ankle_angle": 90.0,
            "torso_inclination": 8.0,
        },
        "landmark_profile": {"mid_hip_y": 0.82, "knee_y": 0.99, "ankle_y": 1.10, "knee_vis": 0.20, "ankle_vis": 0.05},
    },
    {
        "name": "partial_view_ankles_cut_off",
        "description": "Seated with knees visible near bottom edge and ankles cut off",
        "base_variant": "chair",
        "angles": {
            "left_elbow_angle": 98.0, "right_elbow_angle": 98.0,
            "left_shoulder_angle": 28.0, "right_shoulder_angle": 28.0,
            "left_hip_angle": 95.0, "right_hip_angle": 95.0,
            "left_knee_angle": 92.0, "right_knee_angle": 92.0,
            "left_ankle_angle": 90.0, "right_ankle_angle": 90.0,
            "torso_inclination": 7.0,
        },
        "landmark_profile": {"mid_hip_y": 0.70, "knee_y": 0.90, "ankle_y": 1.05, "knee_vis": 0.85, "ankle_vis": 0.15},
    },
    {
        "name": "sitting_with_arms_raised",
        "description": "Seated on chair with both arms stretched overhead",
        "base_variant": "chair",
        "angles": {
            "left_elbow_angle": 160.0, "right_elbow_angle": 160.0,
            "left_shoulder_angle": 155.0, "right_shoulder_angle": 155.0,
            "left_hip_angle": 95.0, "right_hip_angle": 95.0,
            "left_knee_angle": 90.0, "right_knee_angle": 90.0,
            "left_ankle_angle": 90.0, "right_ankle_angle": 90.0,
            "torso_inclination": 6.0,
        },
        "landmark_profile": {"mid_hip_y": 0.62, "knee_y": 0.78, "ankle_y": 0.92, "knee_vis": 0.95, "ankle_vis": 0.95},
    },
]


def get_mistakes_for_pose(pose_name: str, iterations: int = 10) -> List[Dict[str, Any]]:
    """
    Returns `iterations` mistake definitions for an active asana, ordered by
    increasing severity. If `iterations` exceeds the catalog length, cycles
    through entries while preserving increasing severity scaling.
    """
    if pose_name not in MISTAKE_CATALOG:
        raise ValueError(
            f"Pose '{pose_name}' has no mistake catalog. Active asanas: {ACTIVE_POSES}"
        )
    catalog = MISTAKE_CATALOG[pose_name]
    if iterations <= len(catalog):
        return catalog[:iterations]
    return [catalog[i % len(catalog)] for i in range(iterations)]


def get_sitting_variations(iterations: int = 10) -> List[Dict[str, Any]]:
    """
    Returns `iterations` sitting variations for robustness evaluation.
    """
    if iterations <= len(SITTING_VARIATIONS):
        return SITTING_VARIATIONS[:iterations]
    return [SITTING_VARIATIONS[i % len(SITTING_VARIATIONS)] for i in range(iterations)]
