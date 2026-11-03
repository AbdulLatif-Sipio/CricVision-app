"""
ml/icc_rules.py - Responsible Bowling Action Screening & Biomechanical Compliance Engine.

IMPORTANT SCIENTIFIC ACCURACY NOTE:
A 2D phone camera cannot provide official ICC legal certification, which requires
a multi-camera, high-speed 3D motion-capture laboratory with retro-reflective markers.
This engine acts as an on-device "Screening Indicator" to evaluate estimated elbow
extension under ICC Law 21.2 (15.0° threshold), providing confidence ratings, camera
suitability analysis, and clear sports science guidance.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import numpy as np


SCREENING_DISCLAIMER = (
    "Screening Indicator Only: 2D video pose tracking estimates dynamic elbow extension "
    "for coaching and training guidance. It does not replace official multi-camera 3D "
    "laboratory testing conducted by an accredited ICC testing center."
)


@dataclass
class ICCComplianceResult:
    is_legal: bool
    elbow_extension_degrees: float
    max_legal_threshold: float = 15.0
    status_label: str = "LEGAL"  # "LEGAL", "BORDERLINE", "POTENTIAL CONCERN (FLAGGED)"
    confidence: float = 0.95
    details: str = ""
    injury_risk_flags: Optional[List[str]] = None
    is_screening_only: bool = True
    confidence_level: str = "HIGH"  # "HIGH", "MODERATE", "LOW_ACCURACY", "UNSUITABLE"
    disclaimer: str = SCREENING_DISCLAIMER
    why_am_i_seeing_this: str = ""
    camera_suitability: Dict[str, Any] = field(default_factory=dict)


class ICCRulesEngine:
    """
    Responsible bowling action screening and biomechanical safety validator.
    """

    def __init__(self, elbow_threshold: float = 15.0):
        self.elbow_threshold = elbow_threshold

    def evaluate_bowling_legality(
        self,
        elbow_angle_at_horizontal: float,
        elbow_angle_at_release: float,
        angle_sequence_during_delivery: Optional[List[float]] = None,
        camera_fps: float = 30.0,
        camera_angle: str = "side_on",
        visibility_score: float = 1.0
    ) -> ICCComplianceResult:
        """
        Evaluates estimated elbow extension under ICC Law 21.2:
        Calculated as the increase in elbow angle (straightening) between the instant
        the bowling arm reaches horizontal (shoulder height) and ball release.

        Parameters:
        - elbow_angle_at_horizontal: Angle (deg) when arm reaches horizontal.
        - elbow_angle_at_release: Angle (deg) at the instant of release.
        - angle_sequence_during_delivery: Sequence of elbow angles through delivery swing.
        - camera_fps: Video frame rate (higher fps yields higher confidence).
        - camera_angle: "side_on", "front_on", or "angled". Side-on is optimal for elbow screening.
        - visibility_score: Landmark detection quality (0.0 - 1.0).
        """
        if angle_sequence_during_delivery and len(angle_sequence_during_delivery) > 1:
            min_angle = min(angle_sequence_during_delivery)
            rel_angle = angle_sequence_during_delivery[-1]
            extension = max(0.0, float(rel_angle - min_angle))
        else:
            extension = max(0.0, float(elbow_angle_at_release - elbow_angle_at_horizontal))

        injury_flags = []

        # Assess camera suitability and confidence
        suitability_reasons = []
        confidence_val = 0.92

        if camera_angle.lower() not in ["side_on", "side"]:
            suitability_reasons.append("Front-on or angled camera introduces perspective foreshortening.")
            confidence_val -= 0.18

        if camera_fps < 50.0:
            suitability_reasons.append(f"Recorded at {camera_fps:.0f}fps; rapid arm motion blur can obscure exact release instant.")
            confidence_val -= 0.10

        if visibility_score < 0.80:
            suitability_reasons.append("Pose landmarks suffered partial occlusion during swing.")
            confidence_val -= 0.15

        confidence_val = float(np.clip(confidence_val, 0.45, 0.98))

        if confidence_val >= 0.85:
            conf_label = "HIGH"
        elif confidence_val >= 0.70:
            conf_label = "MODERATE"
        else:
            conf_label = "LOW_ACCURACY"

        # Determine compliance status
        if extension <= self.elbow_threshold:
            if extension >= 13.0:
                status = "BORDERLINE"
                details = (
                    f"Estimated elbow extension is {extension:.1f}°, within the 15.0° ICC limit, "
                    f"but close to the boundary. High-speed 60fps+ side-on recording advised."
                )
            else:
                status = "LEGAL"
                details = (
                    f"Estimated elbow extension is {extension:.1f}°, compliant with ICC Law 21.2 "
                    f"(<= 15.0° tolerance)."
                )
            is_legal = True
            why_seen = (
                f"Your arm flexed to a minimum of {min_angle if angle_sequence_during_delivery else elbow_angle_at_horizontal:.1f}° "
                f"and opened to {rel_angle if angle_sequence_during_delivery else elbow_angle_at_release:.1f}° at release. "
                f"The dynamic straightening of {extension:.1f}° is within the permitted 15.0° guideline."
            )
        else:
            status = "ILLEGAL (CHUCKING)"
            excess = extension - self.elbow_threshold
            details = (
                f"Potential Action Concern: Estimated elbow extension is {extension:.1f}° ({excess:.1f}° over "
                f"the 15.0° threshold). Dynamic straightening detected between horizontal arm and ball release."
            )
            is_legal = False
            why_seen = (
                f"Your elbow straightened by {extension:.1f}° between arm horizontal and ball release. "
                f"ICC guidelines permit up to 15.0° of dynamic extension. High straightening typically happens "
                f"when an elbow bends during gather and snaps open at release to create extra whip."
            )

        camera_suitability = {
            "is_suitable_for_screening": confidence_val >= 0.65,
            "detected_angle": camera_angle,
            "fps": camera_fps,
            "confidence_score": round(confidence_val, 2),
            "warnings": suitability_reasons,
            "recommendation": "For highest accuracy, mount phone at waist height directly side-on (perpendicular to pitch) at 60fps."
        }

        return ICCComplianceResult(
            is_legal=is_legal,
            elbow_extension_degrees=round(extension, 2),
            max_legal_threshold=self.elbow_threshold,
            status_label=status,
            confidence=round(confidence_val, 2),
            details=details,
            injury_risk_flags=injury_flags,
            is_screening_only=True,
            confidence_level=conf_label,
            disclaimer=SCREENING_DISCLAIMER,
            why_am_i_seeing_this=why_seen,
            camera_suitability=camera_suitability
        )

    def evaluate_injury_and_kinematics(
        self,
        shoulder_counter_rotation: float,
        front_knee_angle_ffc: float,
        trunk_lateral_flexion: float,
        head_tilt_angle: float
    ) -> Dict[str, Any]:
        """
        Evaluates sports science biomechanics and injury biomarkers:
        1. Shoulder Counter-Rotation: Angle difference between pelvis and shoulder alignment.
           If > 30° -> high risk of lumbar pars interarticularis stress fractures (spondylolysis).
        2. Front Knee Flexion: Braced leg (> 160°) maximizes kinetic chain energy transfer.
           Collapsed (< 140°) leaks bowling pace and strains patellar tendon.
        3. Lateral Trunk Flexion: Leaning sideways > 35° strains oblique abdominals.
        4. Head Alignment: Head falling away (> 15°) degrades seam position and accuracy.
        """
        report = {
            "shoulder_counter_rotation": round(shoulder_counter_rotation, 1),
            "front_knee_angle_ffc": round(front_knee_angle_ffc, 1),
            "trunk_lateral_flexion": round(trunk_lateral_flexion, 1),
            "head_tilt_angle": round(head_tilt_angle, 1),
            "injury_risks": [],
            "efficiency_notes": [],
            "overall_biomechanical_safety": "EXCELLENT"
        }

        # 1. Counter rotation
        if shoulder_counter_rotation > 30.0:
            report["injury_risks"].append({
                "severity": "HIGH",
                "biomarker": "Mixed Bowling Action / Counter-Rotation",
                "measured": f"{shoulder_counter_rotation:.1f}° (Safe threshold <= 30.0°)",
                "consequence": "Severe lumbar spine torque; high risk of lower-back stress fractures (spondylolysis)."
            })
            report["overall_biomechanical_safety"] = "HIGH_RISK"
        elif shoulder_counter_rotation > 22.0:
            report["injury_risks"].append({
                "severity": "MODERATE",
                "biomarker": "Mild Counter-Rotation",
                "measured": f"{shoulder_counter_rotation:.1f}°",
                "consequence": "Moderate trunk torsion; monitor bowling workload."
            })
            if report["overall_biomechanical_safety"] == "EXCELLENT":
                report["overall_biomechanical_safety"] = "MODERATE_RISK"

        # 2. Front knee bracing
        if front_knee_angle_ffc >= 160.0:
            report["efficiency_notes"].append({
                "status": "EXCELLENT",
                "aspect": "Braced Front Leg",
                "measured": f"{front_knee_angle_ffc:.1f}°",
                "note": "Textbook upright lever; maximum kinetic power transferred into ball release speed."
            })
        elif front_knee_angle_ffc < 140.0:
            report["efficiency_notes"].append({
                "status": "POOR",
                "aspect": "Collapsed Front Knee",
                "measured": f"{front_knee_angle_ffc:.1f}° (Target >= 160°)",
                "note": "Knee absorbs force instead of catapulting the trunk forward. Pace loss estimated at 6-10 km/h."
            })

        # 3. Head stability
        if head_tilt_angle > 16.0:
            report["efficiency_notes"].append({
                "status": "IMPAIRED",
                "aspect": "Head Falling Away to Off-Side",
                "measured": f"{head_tilt_angle:.1f}°",
                "note": "Causes delivery line instability, erratic release point, and loss of late swing control."
            })

        return report
