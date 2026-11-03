"""
engine/multi_delivery.py - Multi-Delivery Consistency & Repeatability Analyzer.

Enables cricketers and coaches to analyze multiple deliveries from the same session
(e.g., an entire net over of 4-6 balls) to measure biomechanical repeatability,
detect technique inconsistencies, and identify fatigue-induced mechanical drift.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class SingleDeliverySummary:
    delivery_number: int
    elbow_extension_deg: float
    front_knee_angle_deg: float
    trunk_tilt_deg: float
    release_speed_kmh: float
    head_tilt_deg: float = 8.0
    is_legal: bool = True
    pitch_zone: str = "Good Length | Off-Stump"
    timestamp: Optional[str] = None


@dataclass
class MetricConsistency:
    metric_name: str
    mean_value: float
    std_deviation: float
    min_value: float
    max_value: float
    consistency_grade: str  # "EXCELLENT", "GOOD", "INCONSISTENT", "ERRATIC"
    insight: str


@dataclass
class MultiDeliveryAnalysisResult:
    total_deliveries: int
    overall_repeatability_score: float  # 0 to 100
    repeatability_label: str            # "ELITE REPEATABILITY", "CONSISTENT", "VARIABLE", "FATIGUE/DRIFT DETECTED"
    primary_inconsistency: str
    metric_breakdowns: Dict[str, MetricConsistency]
    flagged_outliers: List[Dict[str, Any]]
    fatigue_drift_detected: bool
    drift_explanation: str
    recommended_drill: Dict[str, Any]
    delivery_comparisons: List[Dict[str, Any]]


class MultiDeliveryAnalyzer:
    """
    Evaluates consistency across multiple deliveries within a training session.
    """

    @classmethod
    def analyze_session(cls, deliveries: List[SingleDeliverySummary]) -> MultiDeliveryAnalysisResult:
        if not deliveries or len(deliveries) < 2:
            raise ValueError("Multi-delivery consistency analysis requires at least 2 deliveries.")

        n = len(deliveries)
        elbows = [d.elbow_extension_deg for d in deliveries]
        knees = [d.front_knee_angle_deg for d in deliveries]
        trunks = [d.trunk_tilt_deg for d in deliveries]
        speeds = [d.release_speed_kmh for d in deliveries if d.release_speed_kmh > 0]
        heads = [d.head_tilt_deg for d in deliveries]

        # 1. Compute stats for key pillars
        knee_mean, knee_std = float(np.mean(knees)), float(np.std(knees))
        elbow_mean, elbow_std = float(np.mean(elbows)), float(np.std(elbows))
        trunk_mean, trunk_std = float(np.mean(trunks)), float(np.std(trunks))
        speed_mean, speed_std = (float(np.mean(speeds)), float(np.std(speeds))) if speeds else (0.0, 0.0)
        head_mean, head_std = float(np.mean(heads)), float(np.std(heads))

        # 2. Evaluate repeatability score (0 - 100)
        # Perfect consistency = standard deviations near 0
        score = 100.0
        score -= min(30.0, knee_std * 3.5)      # Knee angle variance penalty
        score -= min(25.0, trunk_std * 3.0)     # Trunk tilt variance penalty
        score -= min(25.0, elbow_std * 4.0)     # Elbow extension variance penalty
        if speeds:
            score -= min(20.0, speed_std * 2.5) # Speed variance penalty
        score = round(float(np.clip(score, 20.0, 100.0)), 1)

        # 3. Classify label
        if score >= 88.0:
            label = "ELITE REPEATABILITY"
        elif score >= 75.0:
            label = "CONSISTENT"
        elif score >= 60.0:
            label = "VARIABLE"
        else:
            label = "INCONSISTENT / DRIFT DETECTED"

        # 4. Metric breakdowns
        def grade_std(std: float, green_thresh: float, yellow_thresh: float) -> str:
            if std <= green_thresh:
                return "EXCELLENT"
            if std <= yellow_thresh:
                return "GOOD"
            return "INCONSISTENT"

        knee_grade = grade_std(knee_std, 4.0, 8.0)
        trunk_grade = grade_std(trunk_std, 3.5, 7.0)
        elbow_grade = grade_std(elbow_std, 2.5, 5.0)

        metric_breakdowns = {
            "front_knee": MetricConsistency(
                metric_name="Front Knee Bracing at FFC",
                mean_value=round(knee_mean, 1),
                std_deviation=round(knee_std, 1),
                min_value=round(float(np.min(knees)), 1),
                max_value=round(float(np.max(knees)), 1),
                consistency_grade=knee_grade,
                insight="Consistent front knee lock maintains identical release height and release velocity across an over."
            ),
            "trunk_tilt": MetricConsistency(
                metric_name="Trunk Lateral Flexion / Alignment",
                mean_value=round(trunk_mean, 1),
                std_deviation=round(trunk_std, 1),
                min_value=round(float(np.min(trunks)), 1),
                max_value=round(float(np.max(trunks)), 1),
                consistency_grade=trunk_grade,
                insight="Varying your trunk lean between deliveries causes inconsistent line (pulling balls down leg side)."
            ),
            "elbow_extension": MetricConsistency(
                metric_name="Elbow Extension (Screening)",
                mean_value=round(elbow_mean, 1),
                std_deviation=round(elbow_std, 1),
                min_value=round(float(np.min(elbows)), 1),
                max_value=round(float(np.max(elbows)), 1),
                consistency_grade=elbow_grade,
                insight="Elbow arm circle mechanics should remain virtually identical on every ball."
            )
        }

        # 5. Detect Flagged Outliers
        outliers = []
        knee_med = float(np.median(knees))
        trunk_med = float(np.median(trunks))
        for d in deliveries:
            # Check knee collapse outlier
            if abs(d.front_knee_angle_deg - knee_med) >= 15.0 or (knee_std > 0 and abs(d.front_knee_angle_deg - knee_mean) >= 1.25 * knee_std):
                outliers.append({
                    "delivery_number": d.delivery_number,
                    "flaw": "Front Knee Buckling Outlier",
                    "measured": f"{d.front_knee_angle_deg:.1f}°",
                    "average": f"{knee_mean:.1f}°",
                    "impact": "Sudden knee collapse dropped delivery height and released ball short of intended length."
                })
            # Check trunk tilt outlier
            if abs(d.trunk_tilt_deg - trunk_med) >= 8.0 or (trunk_std > 0 and abs(d.trunk_tilt_deg - trunk_mean) >= 1.3 * trunk_std):
                outliers.append({
                    "delivery_number": d.delivery_number,
                    "flaw": "Lateral Trunk Tilt Deviation",
                    "measured": f"{d.trunk_tilt_deg:.1f}°",
                    "average": f"{trunk_mean:.1f}°",
                    "impact": "Excessive sideways lean altered release point, causing line deviation."
                })

        # 6. Fatigue drift check (compare first half vs second half)
        half = n // 2
        first_half_knees = [d.front_knee_angle_deg for d in deliveries[:half]]
        second_half_knees = [d.front_knee_angle_deg for d in deliveries[half:]]
        fatigue_detected = False
        drift_msg = "No significant mechanical degradation detected across the session."

        if len(first_half_knees) > 0 and len(second_half_knees) > 0:
            knee_drop = float(np.mean(first_half_knees) - np.mean(second_half_knees))
            if knee_drop >= 9.0:
                fatigue_detected = True
                drift_msg = (
                    f"Noticeable fatigue drift: Your front knee braced at an average of {np.mean(first_half_knees):.1f}° "
                    f"in early deliveries, but collapsed to {np.mean(second_half_knees):.1f}° in later balls (-{knee_drop:.1f}°). "
                    f"This indicates quadriceps fatigue during longer spells."
                )

        # 7. Identify Primary Inconsistency
        if knee_std >= trunk_std and knee_std >= elbow_std:
            primary_inconsistency = (
                f"Front-side bracing inconsistency: Your front knee varied by {float(np.max(knees) - np.min(knees)):.1f}° "
                f"across these deliveries, directly affecting your release height and length control."
            )
            rec_drill = {
                "title": "Fixed Crease Mark Stride-Lock Drill",
                "equipment": "Chalk line + 1 tennis ball",
                "goal": "Replicate identical landing spot and rigid knee angle for 6 consecutive deliveries.",
                "duration": "4 sets of 6 balls off 3 paces"
            }
        elif trunk_std >= elbow_std:
            primary_inconsistency = (
                f"Trunk alignment inconsistency: Your lateral trunk angle shifted by {float(np.max(trunks) - np.min(trunks)):.1f}° "
                f"between balls, affecting your release channel."
            )
            rec_drill = {
                "title": "Two-Parallel-Rope Runway Drill",
                "equipment": "Two lengths of skipping rope / chalk lines",
                "goal": "Lock upper body in a strict straight corridor from run-up gather to follow-through.",
                "duration": "3 sets of 8 deliveries"
            }
        else:
            primary_inconsistency = (
                f"Elbow extension variance: Your dynamic elbow angle fluctuated by {float(np.max(elbows) - np.min(elbows)):.1f}° "
                f"across balls."
            )
            rec_drill = {
                "title": "Rigid Wall-Slide Rotation Drill",
                "equipment": "Flat wall + tennis ball",
                "goal": "Engrain consistent full-arm extension without flexing the elbow.",
                "duration": "3 sets of 20 revolutions"
            }

        # 8. Delivery-by-delivery comparison table
        comparisons = []
        for d in deliveries:
            comparisons.append({
                "delivery_number": d.delivery_number,
                "elbow_deg": round(d.elbow_extension_deg, 1),
                "knee_deg": round(d.front_knee_angle_deg, 1),
                "trunk_tilt_deg": round(d.trunk_tilt_deg, 1),
                "speed_kmh": round(d.release_speed_kmh, 1) if d.release_speed_kmh > 0 else "N/A",
                "pitch_zone": d.pitch_zone,
                "is_legal": d.is_legal
            })

        return MultiDeliveryAnalysisResult(
            total_deliveries=n,
            overall_repeatability_score=score,
            repeatability_label=label,
            primary_inconsistency=primary_inconsistency,
            metric_breakdowns=metric_breakdowns,
            flagged_outliers=outliers,
            fatigue_drift_detected=fatigue_detected,
            drift_explanation=drift_msg,
            recommended_drill=rec_drill,
            delivery_comparisons=comparisons
        )


# Backward-compatible alias
MultiDeliveryConsistency = MultiDeliveryAnalyzer

