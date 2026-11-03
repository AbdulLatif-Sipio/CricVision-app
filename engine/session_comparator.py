"""
engine/session_comparator.py - Session-to-Session (Before vs. After) Biomechanical Comparator.

Computes exact quantitative deltas across training sessions to answer the #1 question
grassroots players and coaches care about: "Am I actually improving?"
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime


@dataclass
class MetricDelta:
    name: str
    baseline_value: float
    current_value: float
    delta_value: float
    unit: str
    status: str       # "IMPROVED", "STABLE", "NEEDS_WORK", "REGRESSED"
    significance: str  # "MAJOR", "MODERATE", "NEUTRAL"
    insight: str


@dataclass
class SessionComparisonResult:
    session_a_id: str
    session_b_id: str
    baseline_date: str
    current_date: str
    days_between: int
    overall_progress_status: str  # "EXCELLENT_PROGRESS", "SOLID_IMPROVEMENT", "CONSISTENT", "REGRESSION"
    scorecard_delta: float
    key_improvements: List[str]
    areas_still_needing_work: List[str]
    metric_deltas: Dict[str, MetricDelta]
    coach_verdict: str
    next_session_focus: str


class SessionComparator:
    """
    Compares two analytical sessions to extract biomechanical progress and technique changes.
    """

    @classmethod
    def compare(
        cls,
        session_a: Dict[str, Any],
        session_b: Dict[str, Any]
    ) -> SessionComparisonResult:
        """
        Compares baseline Session A (earlier) with current Session B (later).
        """
        # Determine chronological order
        date_a_str = session_a.get("created_at") or "2026-01-01T00:00:00"
        date_b_str = session_b.get("created_at") or "2026-01-02T00:00:00"

        try:
            d_a = datetime.fromisoformat(date_a_str.replace("Z", ""))
            d_b = datetime.fromisoformat(date_b_str.replace("Z", ""))
            if d_a > d_b:
                # Swap so a is baseline, b is current
                session_a, session_b = session_b, session_a
                d_a, d_b = d_b, d_a
            days_between = max(0, (d_b - d_a).days)
        except Exception:
            days_between = 0

        # Extract values
        knee_a = float(session_a.get("front_knee_angle_deg") or 150.0)
        knee_b = float(session_b.get("front_knee_angle_deg") or 150.0)

        elbow_a = float(session_a.get("elbow_extension_deg") or 10.0)
        elbow_b = float(session_b.get("elbow_extension_deg") or 10.0)

        score_a = float(session_a.get("accuracy_score") or 70.0)
        score_b = float(session_b.get("accuracy_score") or 70.0)

        speed_a = float(session_a.get("release_speed_kmh") or 0.0)
        speed_b = float(session_b.get("release_speed_kmh") or 0.0)

        improvements = []
        needs_work = []
        metric_deltas = {}

        # 1. Front Knee Bracing Delta (Higher angle >= 165° is better)
        knee_delta = round(knee_b - knee_a, 1)
        if knee_delta >= 8.0:
            knee_status = "IMPROVED"
            knee_sig = "MAJOR"
            improvements.append(f"Front knee bracing strengthened by +{knee_delta}° (from {knee_a}° to {knee_b}°), enhancing energy catapult leverage.")
        elif knee_delta <= -8.0:
            knee_status = "REGRESSED"
            knee_sig = "MAJOR"
            needs_work.append(f"Front knee collapsed by {abs(knee_delta)}° compared to your baseline session.")
        else:
            knee_status = "STABLE"
            knee_sig = "NEUTRAL"

        metric_deltas["front_knee"] = MetricDelta(
            name="Front Knee Bracing at Impact",
            baseline_value=knee_a,
            current_value=knee_b,
            delta_value=knee_delta,
            unit="deg",
            status=knee_status,
            significance=knee_sig,
            insight="A firmer front leg converts run-up momentum directly into release velocity."
        )

        # 2. Elbow Extension Delta (Lower angle is safer/cleaner)
        elbow_delta = round(elbow_b - elbow_a, 1)
        if elbow_delta <= -3.0:
            elbow_status = "IMPROVED"
            elbow_sig = "MAJOR"
            improvements.append(f"Dynamic elbow straightening reduced by {abs(elbow_delta)}° (down to {elbow_b}°), establishing a much cleaner arm circle.")
        elif elbow_delta >= 4.0:
            elbow_status = "REGRESSED"
            elbow_sig = "MAJOR"
            needs_work.append(f"Elbow straightening increased by +{elbow_delta}° (now {elbow_b}°). Beware of bending during the gather.")
        else:
            elbow_status = "STABLE"
            elbow_sig = "NEUTRAL"

        metric_deltas["elbow_extension"] = MetricDelta(
            name="Elbow Extension (Screening)",
            baseline_value=elbow_a,
            current_value=elbow_b,
            delta_value=elbow_delta,
            unit="deg",
            status=elbow_status,
            significance=elbow_sig,
            insight="Maintaining under 15° of extension keeps your action within ICC compliance guidelines."
        )

        # 3. Overall Score Delta
        score_delta = round(score_b - score_a, 1)
        if score_delta >= 10.0:
            progress_status = "EXCELLENT_PROGRESS"
            coach_verdict = f"Outstanding progress! Your technique scorecard surged by +{score_delta} points since {session_a.get('created_at', 'last session')[:10]}. Drills are clearly taking effect."
        elif score_delta >= 3.0:
            progress_status = "SOLID_IMPROVEMENT"
            coach_verdict = f"Positive development. Technique score improved by +{score_delta} points with noticeable mechanical refinement."
        elif score_delta >= -3.0:
            progress_status = "CONSISTENT"
            coach_verdict = "Technique remained stable across both sessions. Consistency is established; now focus on targeted drill reps to unlock higher efficiency."
        else:
            progress_status = "REGRESSION"
            coach_verdict = f"Technique dipped by {abs(score_delta)} points. Review your run-up balance and return to foundational wall-slide and crease-stride drills."

        metric_deltas["overall_score"] = MetricDelta(
            name="Biomechanical Technique Score",
            baseline_value=score_a,
            current_value=score_b,
            delta_value=score_delta,
            unit="pts",
            status="IMPROVED" if score_delta > 0 else ("REGRESSED" if score_delta < 0 else "STABLE"),
            significance="MAJOR" if abs(score_delta) >= 8.0 else "MODERATE",
            insight="Weighted composite of arm circle legality, knee leverage, torso alignment, and archetype similarity."
        )

        # Speed delta if available
        if speed_a > 0 and speed_b > 0:
            speed_delta = round(speed_b - speed_a, 1)
            metric_deltas["release_speed"] = MetricDelta(
                name="Delivery Release Speed",
                baseline_value=speed_a,
                current_value=speed_b,
                delta_value=speed_delta,
                unit="km/h",
                status="IMPROVED" if speed_delta >= 2.0 else ("REGRESSED" if speed_delta <= -2.0 else "STABLE"),
                significance="MODERATE",
                insight="Speed gains are usually unlocked when the front leg remains rigidly braced at release."
            )
            if speed_delta >= 3.0:
                improvements.append(f"Ball release speed gained +{speed_delta} km/h (now {speed_b} km/h).")

        if not improvements:
            improvements.append("Action mechanics remained repeatable and steady between sessions.")
        if not needs_work:
            needs_work.append("Continue current drill regimen to lock in these mechanical gains under match intensity.")

        # Next session focus
        if knee_b < 155.0:
            next_focus = f"Prioritize front knee lock (currently {knee_b}°). Target reaching >= 165° in your next recorded session."
        elif elbow_b > 12.0:
            next_focus = f"Focus on rigid arm circle rotation (currently {elbow_b}° extension). Target keeping arm extension below 10°."
        else:
            next_focus = "Focus on repeatable run-up rhythm and holding your follow-through balanced for 2 seconds."

        return SessionComparisonResult(
            session_a_id=str(session_a.get("session_id") or session_a.get("id")),
            session_b_id=str(session_b.get("session_id") or session_b.get("id")),
            baseline_date=str(session_a.get("created_at") or "")[:10],
            current_date=str(session_b.get("created_at") or "")[:10],
            days_between=days_between,
            overall_progress_status=progress_status,
            scorecard_delta=score_delta,
            key_improvements=improvements,
            areas_still_needing_work=needs_work,
            metric_deltas=metric_deltas,
            coach_verdict=coach_verdict,
            next_session_focus=next_focus
        )
