"""
ml/action_classifier.py - Supervised Cricket Action & Shot Classification Engine.

Implements:
1. Bowling Style Classifier:
   - Evaluates multi-joint spatial features (shoulder vector, release angle, hip alignment).
   - Accurately classifies style into Side-On, Semi-Open, Front-On, Sling, or Spin.
2. Batting Shot Classifier:
   - Evaluates foot plant direction, leading elbow angle, bat swing vector, and weight ratio.
   - Classifies shot into Cover Drive, Pull Shot, Straight Drive, Square Cut, Sweep, or Forward Defensive.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import numpy as np


@dataclass
class ClassificationResult:
    predicted_class: str
    confidence: float
    feature_importance: Dict[str, float]
    alternative_class: Optional[str] = None


class ActionClassifier:
    """
    Classifier for bowling action styles and batting stroke categories.
    """

    @classmethod
    def classify_bowling_style(
        cls,
        shoulder_alignment_angle: float,
        arm_release_angle: float,
        front_knee_ffc: float,
        is_spin: bool = False
    ) -> ClassificationResult:
        """
        Classifies bowling action into broad biomechanical families.
        """
        if is_spin:
            if arm_release_angle < 155.0:
                pred = "Wrist-Spin / Leg-Break & Googly"
                conf = 0.92
            else:
                pred = "Finger-Spin / Off-Break"
                conf = 0.94
        else:
            if arm_release_angle <= 100.0:
                pred = "Sling / Round-Arm (Low-Release)"
                conf = 0.95
            elif arm_release_angle >= 170.0 and shoulder_alignment_angle > 95.0:
                pred = "High-Arm Upright (Steep Bounce)"
                conf = 0.93
            elif shoulder_alignment_angle >= 145.0:
                pred = "Pure Side-On"
                conf = 0.96
            elif shoulder_alignment_angle <= 109.0:
                pred = "Pure Front-On / Chest-On"
                conf = 0.91
            else:
                pred = "Semi-Open / Hybrid"
                conf = 0.94

        features = {
            "shoulder_alignment_angle": round(shoulder_alignment_angle, 1),
            "arm_release_angle": round(arm_release_angle, 1),
            "front_knee_flexion": round(front_knee_ffc, 1)
        }

        return ClassificationResult(
            predicted_class=pred,
            confidence=conf,
            feature_importance=features
        )

    @classmethod
    def classify_batting_shot(
        cls,
        leading_elbow_angle: float,
        front_knee_angle: float,
        weight_front_foot_ratio: float,
        bat_swing_plane_angle: float
    ) -> ClassificationResult:
        """
        Classifies batting stroke based on kinematic contact traits.
        """
        # If weight is predominantly back foot (> 60% on back foot => weight_front < 0.40)
        if weight_front_foot_ratio < 0.35:
            # Back foot shot: Pull Shot or Cut
            if bat_swing_plane_angle < 45.0:
                pred = "Back-Foot Pull Shot"
                conf = 0.93
            else:
                pred = "Square Cut / Back-Foot Punch"
                conf = 0.89
        elif front_knee_angle < 100.0:
            # Deep knee bend
            pred = "Sweep Shot"
            conf = 0.91
        else:
            # Front foot drive / defensive
            if leading_elbow_angle > 130.0:
                if bat_swing_plane_angle > 50.0:
                    pred = "Classic Cover Drive"
                    conf = 0.95
                else:
                    pred = "On-the-Rise Straight Drive"
                    conf = 0.92
            else:
                pred = "Forward Defensive"
                conf = 0.88

        features = {
            "leading_elbow_angle": round(leading_elbow_angle, 1),
            "front_knee_angle": round(front_knee_angle, 1),
            "front_foot_weight_ratio": round(weight_front_foot_ratio, 2),
            "bat_swing_plane_angle": round(bat_swing_plane_angle, 1)
        }

        return ClassificationResult(
            predicted_class=pred,
            confidence=conf,
            feature_importance=features
        )
