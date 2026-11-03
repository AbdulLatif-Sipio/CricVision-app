"""
tests/test_biomechanics.py - Automated Unit & Integration Tests.
"""

import pytest
from ml.icc_rules import ICCRulesEngine
from ml.archetypes import ArchetypeMatcher
from ml.dtw_matcher import DynamicTimeWarpingMatcher
from engine.kinematics import BiomechanicalKinematics
from backend.database import SessionLocal, AnalysisSessionModel


def test_icc_legality_legal_action():
    engine = ICCRulesEngine(elbow_threshold=15.0)
    res = engine.evaluate_bowling_legality(elbow_angle_at_horizontal=165.0, elbow_angle_at_release=171.0)
    assert res.is_legal is True
    assert res.status_label == "LEGAL"
    assert res.elbow_extension_degrees == 6.0


def test_icc_legality_chucking_action():
    engine = ICCRulesEngine(elbow_threshold=15.0)
    res = engine.evaluate_bowling_legality(elbow_angle_at_horizontal=130.0, elbow_angle_at_release=170.0)
    assert res.is_legal is False
    assert res.status_label == "ILLEGAL (CHUCKING)"
    assert res.elbow_extension_degrees == 40.0


def test_archetype_matching_slinger():
    match = ArchetypeMatcher.match_bowler_style(
        shoulder_alignment_angle=120.0,
        arm_release_angle=70.0,  # Round-arm low release
        front_knee_ffc=150.0,
        is_spin=False
    )
    assert match.archetype_id == "sling_round_arm"
    assert "Malinga" in match.matched_player or "Pathirana" in match.matched_player


def test_3d_angle_calculation():
    # Right angle (90 deg)
    a = [0.0, 1.0, 0.0]
    b = [0.0, 0.0, 0.0]
    c = [1.0, 0.0, 0.0]
    angle = BiomechanicalKinematics.calculate_3d_angle(a, b, c)
    assert abs(angle - 90.0) < 1e-4


def test_dtw_distance_identical_curves():
    curve = [10.0, 20.0, 30.0, 40.0]
    dist, path = DynamicTimeWarpingMatcher.compute_dtw_distance(curve, curve)
    assert dist == 0.0
    assert len(path) == 4


def test_database_session_persistence():
    db = SessionLocal()
    session = AnalysisSessionModel(
        id="test-session-123",
        player_name="Test Bowler",
        activity_type="Bowling",
        media_type="video",
        accuracy_score=92.5,
        release_speed_kmh=135.0,
        is_legal=True,
        elbow_extension_deg=7.5,
        front_knee_angle_deg=168.0,
        matched_legend="Pat Cummins",
        style_category="Semi-Open Pace"
    )
    db.add(session)
    db.commit()

    retrieved = db.query(AnalysisSessionModel).filter_by(id="test-session-123").first()
    assert retrieved is not None
    assert retrieved.matched_legend == "Pat Cummins"
    assert retrieved.accuracy_score == 92.5

    # Cleanup
    db.delete(retrieved)
    db.commit()
    db.close()
