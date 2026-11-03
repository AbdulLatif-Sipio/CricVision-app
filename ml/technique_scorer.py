"""
ml/technique_scorer.py - 100-Point Cricket Biomechanical Scoring Engine.

Computes a comprehensive, objective technique score out of 100 based on 4 pillars:
1. Legality & Safety (25 pts): ICC Law 21.2 elbow compliance & lumbar spine injury safety.
2. Kinetic Chain & Power (25 pts): Braced front knee lever & arm acceleration.
3. Stability & Alignment (25 pts): Head stillness, trunk tilt & release verticality.
4. Archetype Alignment (25 pts): DTW kinematic curve similarity against matched pro standard.
"""

from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np


@dataclass
class TechniqueScorecard:
    total_score: float  # 0 to 100
    grade_label: str    # "Elite / International Class", "Academy Level", "Club / Developing", "Needs Correction"
    legality_safety_score: float  # / 25
    kinetic_power_score: float    # / 25
    stability_score: float        # / 25
    archetype_dtw_score: float    # / 25
    strengths: List[str]
    areas_for_improvement: List[str]


class TechniqueScorer:
    """
    Evaluates cricket mechanics into an objective composite score.
    """

    @classmethod
    def score_bowling_delivery(
        cls,
        elbow_extension_deg: float,
        shoulder_counter_rot_deg: float,
        front_knee_ffc_deg: float,
        head_tilt_deg: float,
        dtw_similarity_pct: float
    ) -> TechniqueScorecard:
        """
        Generates 100-point composite scorecard for bowling.
        """
        strengths = []
        improvements = []

        # 1. Legality & Safety (25 pts)
        if elbow_extension_deg <= 10.0:
            p1_elbow = 15.0
            strengths.append(f"Clean arm extension ({elbow_extension_deg:.1f}°), well under 15.0° ICC limit.")
        elif elbow_extension_deg <= 15.0:
            p1_elbow = 11.0
            improvements.append(f"Elbow extension ({elbow_extension_deg:.1f}°) is legal but near the 15° boundary.")
        else:
            p1_elbow = 0.0
            improvements.append(f"ILLEGAL: Elbow flexed and straightened by {elbow_extension_deg:.1f}° (violates ICC Law 21.2).")

        if shoulder_counter_rot_deg <= 20.0:
            p1_safety = 10.0
            strengths.append("Safe rotational alignment; minimal risk of lumbar stress fractures.")
        elif shoulder_counter_rot_deg <= 30.0:
            p1_safety = 7.0
        else:
            p1_safety = 2.0
            improvements.append(f"High counter-rotation ({shoulder_counter_rot_deg:.1f}°); severe lower-back injury risk.")

        score_legality = p1_elbow + p1_safety

        # 2. Kinetic Chain & Power (25 pts)
        if front_knee_ffc_deg >= 165.0:
            score_power = 25.0
            strengths.append(f"Superb braced front leg ({front_knee_ffc_deg:.1f}°); maximum pace transfer.")
        elif front_knee_ffc_deg >= 150.0:
            score_power = 19.0
            strengths.append("Solid front-foot plant with decent height leverage.")
        elif front_knee_ffc_deg >= 135.0:
            score_power = 13.0
            improvements.append(f"Front knee softens ({front_knee_ffc_deg:.1f}°); causes pace loss of ~6-8 km/h.")
        else:
            score_power = 7.0
            improvements.append(f"Severe front knee collapse ({front_knee_ffc_deg:.1f}°); leaks momentum and strains tendon.")

        # 3. Stability & Alignment (25 pts)
        score_stability = 25.0
        if head_tilt_deg <= 8.0:
            strengths.append(f"Steady head position ({head_tilt_deg:.1f}° tilt), locking accurate seam release.")
        elif head_tilt_deg <= 16.0:
            score_stability -= 6.0
        else:
            score_stability -= 14.0
            improvements.append(f"Head falls away ({head_tilt_deg:.1f}°); causes errant line and radar spray.")

        score_stability = max(5.0, score_stability)

        # 4. Archetype DTW Alignment (25 pts)
        score_dtw = round((dtw_similarity_pct / 100.0) * 25.0, 1)
        if dtw_similarity_pct >= 85.0:
            strengths.append(f"Kinematic curve shows {dtw_similarity_pct:.1f}% rhythmic alignment with international benchmark.")
        else:
            improvements.append(f"Timing sequence deviates ({dtw_similarity_pct:.1f}% pro match); rhythm needs synchronization.")

        # Total composite
        total = round(score_legality + score_power + score_stability + score_dtw, 1)
        total = min(100.0, max(15.0, total))

        if total >= 90.0:
            grade = "Elite / International Class (90-100)"
        elif total >= 80.0:
            grade = "Academy First XI (80-89)"
        elif total >= 65.0:
            grade = "Club Level / Developing (65-79)"
        else:
            grade = "Grassroots / Requires Technique Correction (<65)"

        return TechniqueScorecard(
            total_score=total,
            grade_label=grade,
            legality_safety_score=round(score_legality, 1),
            kinetic_power_score=round(score_power, 1),
            stability_score=round(score_stability, 1),
            archetype_dtw_score=round(score_dtw, 1),
            strengths=strengths,
            areas_for_improvement=improvements
        )

    @classmethod
    def score_batting_stroke(
        cls,
        leading_elbow_deg: float,
        front_knee_deg: float,
        head_tilt_deg: float,
        dtw_similarity_pct: float,
        sweet_spot_pct: float
    ) -> TechniqueScorecard:
        """
        Generates 100-point composite scorecard for batting strokes.
        """
        strengths = []
        improvements = []

        # 1. High Elbow & Presentation (25 pts)
        if leading_elbow_deg >= 132.0:
            s_elbow = 25.0
            strengths.append(f"Textbook high leading elbow ({leading_elbow_deg:.1f}°); ensures grounded drives.")
        elif leading_elbow_deg >= 115.0:
            s_elbow = 18.0
        else:
            s_elbow = 10.0
            improvements.append(f"Low leading elbow ({leading_elbow_deg:.1f}°); risk of slicing aerial catches.")

        # 2. Footwork & Base (25 pts)
        if front_knee_deg >= 115.0 and front_knee_deg <= 145.0:
            s_base = 25.0
            strengths.append(f"Ideal front-knee flexion ({front_knee_deg:.1f}°); balanced center of gravity.")
        else:
            s_base = 18.0

        # 3. Head Position & Timing (25 pts)
        s_timing = 25.0
        if head_tilt_deg <= 5.0:
            strengths.append(f"Laser-level head and eyeline ({head_tilt_deg:.1f}° tilt).")
        else:
            s_timing -= 7.0
            improvements.append(f"Head tilted ({head_tilt_deg:.1f}°); impairs depth perception and late ball tracking.")

        if sweet_spot_pct < 75.0:
            s_timing -= 5.0
            improvements.append("Bat contact off-center from the sweet-spot.")

        # 4. Pro Alignment (25 pts)
        s_dtw = round((dtw_similarity_pct / 100.0) * 25.0, 1)

        total = round(s_elbow + s_base + s_timing + s_dtw, 1)
        total = min(100.0, max(20.0, total))

        if total >= 90.0:
            grade = "Masterclass / Pro Standard (90-100)"
        elif total >= 80.0:
            grade = "A-Grade Academy Batter (80-89)"
        elif total >= 65.0:
            grade = "Club Batter / Developing (65-79)"
        else:
            grade = "Grassroots / Foundational (<65)"

        return TechniqueScorecard(
            total_score=total,
            grade_label=grade,
            legality_safety_score=round(s_elbow, 1),
            kinetic_power_score=round(s_base, 1),
            stability_score=round(s_timing, 1),
            archetype_dtw_score=round(s_dtw, 1),
            strengths=strengths,
            areas_for_improvement=improvements
        )
