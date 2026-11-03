"""
tests/test_features_upgrade.py - Tests for Upgraded & New CricVision Features.

Validates:
1. ICC Rules Screening mode with confidence & disclaimers.
2. Honest Speed Gun (zero fabrication when trajectory < 3).
3. Multi-Delivery consistency & repeatability analysis.
4. Session-to-Session before/after comparison delta engine.
5. 7-Day Weekly Training Planner.
6. Camera suitability guidance and video quality auditing.
7. Structured coaching observation hierarchy.
"""

import pytest
from ml.icc_rules import ICCRulesEngine, SCREENING_DISCLAIMER
from vision.speed_gun import CricketSpeedGun
from engine.multi_delivery import MultiDeliveryAnalyzer, SingleDeliverySummary
from engine.session_comparator import SessionComparator
from coaching.weekly_planner import WeeklyPlanner
from coaching.drill_matrix import DrillEngine
from vision.camera_guide import CameraSuitabilityGuide


def test_icc_screening_mode_and_confidence():
    engine = ICCRulesEngine(elbow_threshold=15.0)

    # Clean legal action side-on at 60fps
    res = engine.evaluate_bowling_legality(
        elbow_angle_at_horizontal=165.0,
        elbow_angle_at_release=172.0,
        camera_fps=60.0,
        camera_angle="side_on"
    )
    assert res.is_legal is True
    assert res.is_screening_only is True
    assert res.confidence_level == "HIGH"
    assert res.disclaimer == SCREENING_DISCLAIMER
    assert res.camera_suitability["is_suitable_for_screening"] is True
    assert "15.0°" in res.why_am_i_seeing_this

    # Front-on low fps reduces confidence
    res_front = engine.evaluate_bowling_legality(
        elbow_angle_at_horizontal=165.0,
        elbow_angle_at_release=172.0,
        camera_fps=24.0,
        camera_angle="front_on"
    )
    assert res_front.confidence_level in ["MODERATE", "LOW_ACCURACY"]
    assert len(res_front.camera_suitability["warnings"]) > 0


def test_honest_speed_gun_zero_fabrication():
    speed_gun = CricketSpeedGun()

    # Trajectory with fewer than 3 points MUST NOT fabricate speed
    short_trajectory = [(100, 200), (105, 205)]
    metrics = speed_gun.compute_speed_from_trajectory(short_trajectory, fps=30.0)

    assert metrics.release_speed_kmh == 0.0
    assert metrics.release_speed_mph == 0.0
    assert metrics.is_calibrated is False
    assert metrics.method_used == "UNAVAILABLE"
    assert "Insufficient ball trajectory" in metrics.status_note


def test_multi_delivery_consistency_analysis():
    deliveries = [
        SingleDeliverySummary(delivery_number=1, elbow_extension_deg=7.0, front_knee_angle_deg=168.0, trunk_tilt_deg=12.0, release_speed_kmh=122.0),
        SingleDeliverySummary(delivery_number=2, elbow_extension_deg=8.0, front_knee_angle_deg=166.0, trunk_tilt_deg=13.0, release_speed_kmh=123.0),
        SingleDeliverySummary(delivery_number=3, elbow_extension_deg=7.5, front_knee_angle_deg=167.0, trunk_tilt_deg=12.5, release_speed_kmh=121.5),
        SingleDeliverySummary(delivery_number=4, elbow_extension_deg=8.2, front_knee_angle_deg=135.0, trunk_tilt_deg=13.0, release_speed_kmh=114.0) # Outlier knee collapse
    ]

    res = MultiDeliveryAnalyzer.analyze_session(deliveries)
    assert res.total_deliveries == 4
    assert 0 <= res.overall_repeatability_score <= 100
    assert "front_knee" in res.metric_breakdowns
    assert len(res.flagged_outliers) > 0
    # Check that Delivery 4 was flagged
    flagged_nums = [o["delivery_number"] for o in res.flagged_outliers]
    assert 4 in flagged_nums


def test_session_comparator_delta_engine():
    session_a = {
        "session_id": "sess-001",
        "accuracy_score": 68.0,
        "front_knee_angle_deg": 142.0,
        "elbow_extension_deg": 16.5,
        "release_speed_kmh": 115.0,
        "created_at": "2026-10-01T10:00:00"
    }
    session_b = {
        "session_id": "sess-002",
        "accuracy_score": 84.0,
        "front_knee_angle_deg": 166.0,  # +24 deg improvement
        "elbow_extension_deg": 9.2,    # -7.3 deg improvement
        "release_speed_kmh": 122.0,
        "created_at": "2026-10-15T10:00:00"
    }

    comp = SessionComparator.compare(session_a, session_b)
    assert comp.days_between == 14
    assert comp.scorecard_delta == 16.0
    assert comp.overall_progress_status == "EXCELLENT_PROGRESS"
    assert comp.metric_deltas["front_knee"].status == "IMPROVED"
    assert comp.metric_deltas["elbow_extension"].status == "IMPROVED"
    assert len(comp.key_improvements) >= 2


def test_weekly_training_planner():
    plan = WeeklyPlanner.create_plan(
        player_name="Haider",
        primary_flaw="collapsed_front_knee",
        skill_level="Intermediate"
    )
    assert plan.player_name == "Haider"
    assert len(plan.daily_schedule) == 7
    # Verify Monday is active and Tuesday is recovery
    assert plan.daily_schedule[0].is_rest_day is False
    assert plan.daily_schedule[1].is_rest_day is True
    assert "Front Leg Bracing" in plan.daily_schedule[0].focus_theme


def test_structured_coaching_observation_builder():
    report = DrillEngine.build_structured_bowling_report(
        is_legal=False,
        elbow_extension=17.5,
        shoulder_counter_rot=15.0,
        front_knee_ffc=165.0,
        head_tilt=6.0,
        confidence_val=0.92
    )
    assert "Elbow dynamic extension" in report["main_observation"]
    assert "Your bowling arm flexed" in report["what_this_means"]
    assert report["drill"]["name"] == "The Rigid Arm Wall-Slide Drill"
    assert "Safety guidance" in report["safety_guidance"]
    assert report["can_dismiss_or_correct"] is True


def test_camera_suitability_specs():
    specs = CameraSuitabilityGuide.GUIDANCE_SPECS
    assert "bowling_side_on" in specs
    assert "bowling_front_on" in specs
    assert "Side-On" in specs["bowling_side_on"]["name"]
