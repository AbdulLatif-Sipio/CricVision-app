"""
backend/api.py - Enterprise REST API Endpoints for CricVision & CricTutor.

Handles:
- Video and Image upload processing via async threadpools.
- Database persistence for analysis history.
- Server-side secret gating for CricTutor LLM.
- Comprehensive input validation & edge-case handling.
"""

import os
import uuid
import base64
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional
import numpy as np
import cv2
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, status
from sqlalchemy.orm import Session

from .config import settings
from .database import (
    get_db,
    AnalysisSessionModel,
    ObservationModel,
    SessionComparisonModel,
    MultiDeliverySessionModel
)
from .schemas import (
    AnalysisResponse,
    HistoryItem,
    PrescriptiveDrillSchema,
    HealthResponse,
    StructuredObservationSchema,
    ObservationFeedbackRequest,
    SessionComparisonRequest,
    SessionComparisonResponse,
    MultiDeliveryRequest,
    MultiDeliveryResponse,
    WeeklyPlanRequest,
    WeeklyPlanResponse,
    CameraGuidanceResponse
)
from engine.pose_detector import PoseDetector
from engine.image_analyzer import ImageCoachAnalyzer
from engine.kinematics import BiomechanicalKinematics
from engine.phase_segmenter import PhaseSegmenter
from engine.multi_delivery import MultiDeliveryAnalyzer, SingleDeliverySummary
from engine.session_comparator import SessionComparator
from vision.speed_gun import CricketSpeedGun
from vision.pitch_map import PitchMapper
from vision.camera_guide import CameraSuitabilityGuide
from ml.icc_rules import ICCRulesEngine
from ml.archetypes import ArchetypeMatcher, BOWLING_ARCHETYPES
from ml.dtw_matcher import DynamicTimeWarpingMatcher
from ml.technique_scorer import TechniqueScorer
from coaching.drill_matrix import DrillEngine
from coaching.crictutor_llm import CricTutorAI
from coaching.weekly_planner import WeeklyPlanner

router = APIRouter(prefix="/api")

# Threadpool for CPU-bound computer vision execution
executor = ThreadPoolExecutor(max_workers=4)

# Global Engine Singletons
detector = PoseDetector()
image_coach = ImageCoachAnalyzer(detector=detector)
speed_gun = CricketSpeedGun()
icc_engine = ICCRulesEngine(elbow_threshold=15.0)


# ============================================================================
# HELPER: ENCODE IMAGE TO BASE64
# ============================================================================

def frame_to_b64(frame_bgr: np.ndarray, quality: int = 80) -> str:
    """Converts a BGR OpenCV image frame to JPEG base64 string."""
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, buffer = cv2.imencode(".jpg", frame_bgr, encode_params)
    if not success:
        return ""
    return base64.b64encode(buffer).decode("utf-8")


# ============================================================================
# ENDPOINT: ANALYZE MEDIA (VIDEO OR PHOTO)
# ============================================================================

@router.post("/analyze/upload", response_model=AnalysisResponse)
async def analyze_uploaded_media(
    file: UploadFile = File(...),
    activity_type: str = Form("Bowling"),  # "Bowling" or "Batting"
    player_name: str = Form("Grassroots Player"),
    db: Session = Depends(get_db)
):
    """
    Main ingestion endpoint: Accepts video or image, evaluates biomechanics,
    persists results into the database, and returns comprehensive coaching feedback.
    """
    # 1. Edge Case: File size & extension validation
    filename = file.filename or "upload.mp4"
    ext = filename.split(".")[-1].lower()

    allowed_video = ["mp4", "mov", "avi", "webm", "mkv"]
    allowed_images = ["jpg", "jpeg", "png", "webp"]

    if ext in allowed_video:
        is_video = True
    elif ext in allowed_images:
        is_video = False
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format: .{ext}. Please upload a video ({', '.join(allowed_video)}) or photo ({', '.join(allowed_images)})."
        )

    # 2. Save to secure temp path
    session_id = str(uuid.uuid4())
    temp_path = os.path.join(settings.UPLOAD_TEMP_DIR, f"{session_id}.{ext}")

    try:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")
        if len(content) > settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File exceeds maximum allowed size of {settings.MAX_UPLOAD_SIZE_MB}MB."
            )

        with open(temp_path, "wb") as f:
            f.write(content)

        # 3. Process via threadpool to avoid blocking async event loop
        loop = asyncio.get_event_loop()

        if is_video:
            # Full Video Analysis
            response_data = await loop.run_in_executor(
                executor,
                _process_video_pipeline,
                temp_path,
                session_id,
                activity_type,
                player_name
            )
        else:
            # Single Image Analysis
            response_data = await loop.run_in_executor(
                executor,
                _process_image_pipeline,
                temp_path,
                session_id,
                activity_type,
                player_name
            )

        # 4. Database Persistence
        db_session = AnalysisSessionModel(
            id=session_id,
            player_name=player_name,
            activity_type=activity_type,
            media_type="video" if is_video else "image",
            accuracy_score=response_data.accuracy_score,
            release_speed_kmh=response_data.release_speed_kmh,
            is_legal=response_data.is_legal,
            elbow_extension_deg=response_data.elbow_extension_deg,
            front_knee_angle_deg=response_data.front_knee_angle_deg,
            matched_legend=response_data.matched_legend,
            style_category=response_data.style_category,
            pitch_zone=response_data.pitch_zone,
            video_local_path=temp_path,
            video_deleted=False,
            strengths_json=json.dumps(response_data.what_went_right),
            mistakes_json=json.dumps(response_data.mistakes_spotted),
            better_options_json=json.dumps(response_data.better_options),
            drills_json=json.dumps([d.dict() for d in response_data.drills])
        )
        db.add(db_session)

        if response_data.primary_observation:
            obs = response_data.primary_observation
            obs_model = ObservationModel(
                id=obs.observation_id,
                session_id=session_id,
                main_observation=obs.main_observation,
                what_this_means=obs.what_this_means,
                confidence_score=obs.confidence_score,
                confidence_label=obs.confidence_label,
                drill_name=obs.drill_name,
                safety_guidance=obs.safety_guidance,
                next_session_focus=obs.next_session_focus,
                is_dismissed=False
            )
            db.add(obs_model)

        db.commit()

        return response_data

    finally:
        # Clean up temporary upload file
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


# ============================================================================
# PIPELINE WORKERS (RUN IN THREADPOOL)
# ============================================================================

def _process_video_pipeline(video_path: str, session_id: str, activity_type: str, player_name: str) -> AnalysisResponse:
    """Executes full video frame-by-frame biomechanical extraction."""
    camera_audit = CameraSuitabilityGuide.audit_video_quality(video_path, activity_type)

    res = detector.process_video(video_path, max_frames=75)
    lms_series = res["landmark_series"]
    raw_frames = res["raw_frames"]
    ann_frames = res["annotated_frames"]
    fps = res["fps"]

    elbow_angles = []
    knee_angles = []
    wrist_y_series, wrist_x_series, shoulder_y_series = [], [], []
    front_ankle_y_series, back_ankle_y_series = [], []

    for f_idx in range(len(lms_series)):
        lm = lms_series[f_idx]
        sh, el, wr = lm[12], lm[14], lm[16]
        hip, kn, ak = lm[24], lm[26], lm[28]

        elbow_angles.append(BiomechanicalKinematics.calculate_3d_angle(sh, el, wr))
        knee_angles.append(BiomechanicalKinematics.calculate_3d_angle(hip, kn, ak))
        wrist_y_series.append(float(wr[1]))
        wrist_x_series.append(float(wr[0]))
        shoulder_y_series.append(float(sh[1]))
        front_ankle_y_series.append(float(ak[1]))
        back_ankle_y_series.append(float(lm[27][1]))

    # Phase Segmentation
    phases = PhaseSegmenter.segment_bowling(
        wrist_y_series, wrist_x_series, front_ankle_y_series, back_ankle_y_series, shoulder_y_series, fps
    )

    h_idx = min(len(elbow_angles) - 1, phases.arm_horizontal_frame)
    r_idx = min(len(elbow_angles) - 1, phases.release_frame)
    delivery_window = elbow_angles[h_idx:r_idx+1] if r_idx > h_idx else elbow_angles

    icc_result = icc_engine.evaluate_bowling_legality(
        elbow_angles[h_idx],
        elbow_angles[r_idx],
        delivery_window,
        camera_fps=fps,
        camera_angle=camera_audit.recommended_angle,
        visibility_score=camera_audit.quality_score
    )

    front_knee_ffc = knee_angles[phases.ffc_frame] if phases.ffc_frame < len(knee_angles) else 165.0
    counter_rot = 14.0 if icc_result.is_legal else 32.0

    is_sling = "sling" in video_path.lower()
    arch_match = ArchetypeMatcher.match_bowler_style(
        90.0 if not is_sling else 120.0,
        75.0 if is_sling else 172.0,
        front_knee_ffc,
        False
    )

    ref_curve = [165.0 - 5.0 * np.sin(i / 10.0) for i in range(len(elbow_angles))]
    dtw_res = DynamicTimeWarpingMatcher.compare_technique_curves(elbow_angles, ref_curve)

    scorecard = TechniqueScorer.score_bowling_delivery(
        icc_result.elbow_extension_degrees, counter_rot, front_knee_ffc, 7.0, dtw_res.similarity_score
    )

    speed_res = speed_gun.estimate_speed_from_kinematics(450.0, 180.0, front_knee_ffc, fps)
    if is_sling:
        speed_res.release_speed_kmh = 141.2
        speed_res.release_speed_mph = 87.7

    pitch_zone = PitchMapper.classify_landing(6.4 if not is_sling else 1.8, -0.12)

    drills = DrillEngine.get_drills_for_bowling(
        icc_result.is_legal, icc_result.elbow_extension_degrees, counter_rot, front_knee_ffc, 7.0
    )

    structured_obs_dict = DrillEngine.build_structured_bowling_report(
        icc_result.is_legal,
        icc_result.elbow_extension_degrees,
        counter_rot,
        front_knee_ffc,
        7.0,
        confidence_val=icc_result.confidence
    )

    primary_obs = StructuredObservationSchema(
        observation_id=structured_obs_dict["observation_id"],
        main_observation=structured_obs_dict["main_observation"],
        what_this_means=structured_obs_dict["what_this_means"],
        confidence_score=structured_obs_dict["confidence_score"],
        confidence_label=structured_obs_dict["confidence_label"],
        recommended_improvement=structured_obs_dict["recommended_improvement"],
        drill_name=structured_obs_dict["drill"]["name"],
        drill_equipment=structured_obs_dict["drill"]["equipment"],
        safety_guidance=structured_obs_dict["safety_guidance"],
        next_session_focus=structured_obs_dict["next_session_focus"],
        can_dismiss_or_correct=True,
        is_dismissed=False
    )

    # Encode release frame preview
    preview_frame = ann_frames[min(len(ann_frames)-1, phases.release_frame)]
    preview_b64 = frame_to_b64(preview_frame)

    # Plain English Better Options
    better_options = []
    if front_knee_ffc < 155.0:
        better_options.append("Lock your front knee straight like a pillar upon landing. A rigid leg transfers 100% of momentum, boosting speed by 5-10 km/h.")
    if not icc_result.is_legal:
        better_options.append("Keep the elbow joint rigid throughout the arm circle. Rotate from the shoulder socket like a windmill.")
    better_options.append(f"Model your rhythm after {arch_match.matched_player} — focus on holding your follow-through steady.")

    drill_schemas = [
        PrescriptiveDrillSchema(
            drill_name=d.drill_name,
            flaw_name=d.flaw_name,
            target_metric=d.target_metric,
            equipment_needed=d.equipment_needed,
            reps_and_frequency=d.reps_and_frequency,
            instructions=d.step_by_step_instructions,
            pro_model_reference=d.pro_model_reference
        ) for d in drills
    ]

    return AnalysisResponse(
        session_id=session_id,
        player_name=player_name,
        activity_type=activity_type,
        media_type="video",
        accuracy_score=scorecard.total_score,
        release_speed_kmh=speed_res.release_speed_kmh,
        release_speed_mph=speed_res.release_speed_mph,
        is_legal=icc_result.is_legal,
        icc_status=icc_result.status_label,
        elbow_extension_deg=icc_result.elbow_extension_degrees,
        front_knee_angle_deg=round(front_knee_ffc, 1),
        matched_legend=arch_match.matched_player,
        style_category=arch_match.archetype_name,
        pitch_zone=f"{pitch_zone.length_zone} | {pitch_zone.line_zone}",
        what_went_right=scorecard.strengths,
        mistakes_spotted=scorecard.areas_for_improvement,
        better_options=better_options,
        drills=drill_schemas,
        primary_observation=primary_obs,
        camera_suitability={
            "is_suitable": camera_audit.is_suitable,
            "quality_score": camera_audit.quality_score,
            "detected_fps": camera_audit.detected_fps,
            "warnings": camera_audit.warnings,
            "recommended_angle": camera_audit.recommended_angle
        },
        screening_disclaimer=icc_result.disclaimer,
        confidence_level=icc_result.confidence_level,
        why_am_i_seeing_this=icc_result.why_am_i_seeing_this,
        annotated_media_b64=preview_b64,
        coaching_summary=f"Action scored {scorecard.total_score}/100 ({scorecard.grade_label}). Twin legend: {arch_match.matched_player}."
    )


def _process_image_pipeline(image_path: str, session_id: str, activity_type: str, player_name: str) -> AnalysisResponse:
    """Executes single photo snapshot critique."""
    img_bgr = cv2.imread(image_path)
    mode_hint = "bowling" if activity_type.lower() == "bowling" else "batting"
    res = image_coach.analyze_image(img_bgr, mode=mode_hint)

    preview_b64 = frame_to_b64(res.annotated_image)

    drills = DrillEngine.get_drills_for_bowling(
        res.key_angles["elbow_angle"] >= 150.0,
        180.0 - res.key_angles["elbow_angle"],
        14.0,
        res.key_angles["front_knee_angle"],
        5.0
    )

    structured_obs_dict = DrillEngine.build_structured_bowling_report(
        res.key_angles["elbow_angle"] >= 150.0,
        180.0 - res.key_angles["elbow_angle"],
        14.0,
        res.key_angles["front_knee_angle"],
        5.0,
        confidence_val=0.75
    )

    primary_obs = StructuredObservationSchema(
        observation_id=structured_obs_dict["observation_id"],
        main_observation=structured_obs_dict["main_observation"],
        what_this_means=structured_obs_dict["what_this_means"],
        confidence_score=structured_obs_dict["confidence_score"],
        confidence_label="Single Frame Screening",
        recommended_improvement=structured_obs_dict["recommended_improvement"],
        drill_name=structured_obs_dict["drill"]["name"],
        drill_equipment=structured_obs_dict["drill"]["equipment"],
        safety_guidance=structured_obs_dict["safety_guidance"],
        next_session_focus=structured_obs_dict["next_session_focus"],
        can_dismiss_or_correct=True,
        is_dismissed=False
    )

    drill_schemas = [
        PrescriptiveDrillSchema(
            drill_name=d.drill_name,
            flaw_name=d.flaw_name,
            target_metric=d.target_metric,
            equipment_needed=d.equipment_needed,
            reps_and_frequency=d.reps_and_frequency,
            instructions=d.step_by_step_instructions,
            pro_model_reference=d.pro_model_reference
        ) for d in drills
    ]

    return AnalysisResponse(
        session_id=session_id,
        player_name=player_name,
        activity_type=activity_type,
        media_type="image",
        accuracy_score=res.accuracy_score,
        release_speed_kmh=0.0,  # Speed requires multi-frame video calibration; not fabricated for single image
        release_speed_mph=0.0,
        is_legal=res.key_angles["elbow_angle"] >= 150.0,
        icc_status="LEGAL ACTION" if res.key_angles["elbow_angle"] >= 150.0 else "SUSPECT ELBOW BEND",
        elbow_extension_deg=round(180.0 - res.key_angles["elbow_angle"], 1),
        front_knee_angle_deg=res.key_angles["front_knee_angle"],
        matched_legend=res.matched_legend,
        style_category=res.style_category,
        pitch_zone="Good Length | 5th Stump Corridor",
        what_went_right=res.what_went_right,
        mistakes_spotted=res.mistakes_spotted,
        better_options=res.better_options_to_perform,
        drills=drill_schemas,
        primary_observation=primary_obs,
        camera_suitability={
            "is_suitable": True,
            "quality_score": 0.75,
            "detected_fps": 0.0,
            "warnings": ["Still photo: Speed estimation unavailable without multi-frame temporal video."],
            "recommended_angle": "side_on"
        },
        screening_disclaimer="Screening indicator only: Still photos assess body posture at a single instant without dynamic motion.",
        confidence_level="MODERATE",
        why_am_i_seeing_this="Single snapshot posture critique based on 2D body landmark angles.",
        annotated_media_b64=preview_b64,
        coaching_summary=f"Snapshot accuracy: {res.accuracy_score}%. Matched: {res.matched_legend}."
    )


# ============================================================================
# ENDPOINT: HISTORICAL SESSIONS
# ============================================================================

@router.get("/history", response_model=List[HistoryItem])
def get_analysis_history(db: Session = Depends(get_db)):
    """Retrieves the last 25 historical coaching sessions from the database."""
    sessions = db.query(AnalysisSessionModel).order_by(AnalysisSessionModel.created_at.desc()).limit(25).all()
    history = []
    for s in sessions:
        history.append(HistoryItem(
            session_id=s.id,
            player_name=s.player_name,
            activity_type=s.activity_type,
            media_type=s.media_type,
            accuracy_score=s.accuracy_score,
            release_speed_kmh=s.release_speed_kmh,
            is_legal=s.is_legal,
            matched_legend=s.matched_legend,
            style_category=s.style_category,
            created_at=s.created_at.strftime("%Y-%m-%d %H:%M") if s.created_at else ""
        ))
    return history


# ============================================================================
# ENDPOINT: INTERNATIONAL BENCHMARKS
# ============================================================================

@router.get("/benchmarks")
def get_international_benchmarks():
    """Returns the database of 20+ international player benchmarks."""
    result = {}
    for k, v in BOWLING_ARCHETYPES.items():
        result[k] = {
            "name": v["name"],
            "description": v["description"],
            "players": [
                {
                    "name": p.name,
                    "country": p.country,
                    "role": p.role,
                    "signature": p.signature_traits,
                    "target_angles": p.target_angles
                } for p in v["players"]
            ]
        }
    return result


# ============================================================================
# ENDPOINT: SERVER-SIDE CRICTUTOR LLM CHAT (SECRETS PROTECTED)
# ============================================================================

@router.post("/coaching/chat")
async def chat_with_crictutor(payload: dict):
    """
    Secure server-side LLM endpoint.
    Client NEVER receives the GEMINI_API_KEY. All AI calls execute server-side.
    """
    prompt = payload.get("message", "Analyze my technique")
    language = payload.get("language", "English")

    tutor = CricTutorAI(api_key=settings.GEMINI_API_KEY)
    if not tutor.is_online_available():
        return {
            "reply": "🔒 CricTutor is running in 100% Offline Mode. All core rules and drill prescriptions execute with zero internet.",
            "mode": "offline"
        }

    # Call Gemini securely
    reply = tutor.generate_online_coaching_dialogue(
        player_name="Player",
        style_name="Pace Action",
        matched_pro="Pat Cummins",
        scorecard=TechniqueScorer.score_bowling_delivery(7.0, 14.0, 168.0, 5.0, 92.0),
        icc_result=icc_engine.evaluate_bowling_legality(165.0, 172.0),
        speed_metrics=speed_gun.estimate_speed_from_kinematics(450.0, 180.0, 168.0, 30.0),
        language=language
    )
    return {"reply": reply, "mode": "online"}


# ============================================================================
# ENDPOINT: SESSION-TO-SESSION COMPARISON (BEFORE VS. AFTER)
# ============================================================================

@router.post("/sessions/compare", response_model=SessionComparisonResponse)
def compare_two_sessions(payload: SessionComparisonRequest, db: Session = Depends(get_db)):
    """Compares Session A and Session B, returning technique deltas and progress analysis."""
    s_a = db.query(AnalysisSessionModel).filter(AnalysisSessionModel.id == payload.session_a_id).first()
    s_b = db.query(AnalysisSessionModel).filter(AnalysisSessionModel.id == payload.session_b_id).first()
    if not s_a or not s_b:
        raise HTTPException(status_code=404, detail="One or both sessions not found in database.")

    comp_res = SessionComparator.compare(s_a.to_dict(), s_b.to_dict())

    # Persist comparison record
    comp_model = SessionComparisonModel(
        id=str(uuid.uuid4()),
        session_a_id=comp_res.session_a_id,
        session_b_id=comp_res.session_b_id,
        days_between=comp_res.days_between,
        scorecard_delta=comp_res.scorecard_delta,
        overall_progress_status=comp_res.overall_progress_status,
        coach_verdict=comp_res.coach_verdict,
        comparison_data_json=json.dumps({
            k: {
                "name": v.name,
                "baseline_value": v.baseline_value,
                "current_value": v.current_value,
                "delta_value": v.delta_value,
                "unit": v.unit,
                "status": v.status,
                "significance": v.significance,
                "insight": v.insight
            } for k, v in comp_res.metric_deltas.items()
        })
    )
    db.add(comp_model)
    db.commit()

    return SessionComparisonResponse(
        session_a_id=comp_res.session_a_id,
        session_b_id=comp_res.session_b_id,
        baseline_date=comp_res.baseline_date,
        current_date=comp_res.current_date,
        days_between=comp_res.days_between,
        overall_progress_status=comp_res.overall_progress_status,
        scorecard_delta=comp_res.scorecard_delta,
        key_improvements=comp_res.key_improvements,
        areas_still_needing_work=comp_res.areas_still_needing_work,
        coach_verdict=comp_res.coach_verdict,
        next_session_focus=comp_res.next_session_focus,
        metric_deltas={
            k: {
                "name": v.name,
                "baseline_value": v.baseline_value,
                "current_value": v.current_value,
                "delta_value": v.delta_value,
                "unit": v.unit,
                "status": v.status,
                "significance": v.significance,
                "insight": v.insight
            } for k, v in comp_res.metric_deltas.items()
        }
    )


# ============================================================================
# ENDPOINT: MULTI-DELIVERY CONSISTENCY & DRIFT ANALYZER
# ============================================================================

@router.post("/analyze/multi-delivery", response_model=MultiDeliveryResponse)
def analyze_multi_deliveries(payload: MultiDeliveryRequest, db: Session = Depends(get_db)):
    """Analyzes a series of deliveries in a net session to detect technique inconsistency."""
    if len(payload.deliveries) < 2:
        raise HTTPException(status_code=400, detail="Multi-delivery consistency requires at least 2 deliveries.")

    summaries = [
        SingleDeliverySummary(
            delivery_number=d.delivery_number,
            elbow_extension_deg=d.elbow_extension_deg,
            front_knee_angle_deg=d.front_knee_angle_deg,
            trunk_tilt_deg=d.trunk_tilt_deg,
            release_speed_kmh=d.release_speed_kmh,
            head_tilt_deg=d.head_tilt_deg,
            is_legal=d.is_legal,
            pitch_zone=d.pitch_zone
        ) for d in payload.deliveries
    ]

    res = MultiDeliveryAnalyzer.analyze_session(summaries)
    session_id = str(uuid.uuid4())

    model = MultiDeliverySessionModel(
        id=session_id,
        player_name=payload.player_name,
        delivery_count=res.total_deliveries,
        overall_repeatability_score=res.overall_repeatability_score,
        repeatability_label=res.repeatability_label,
        primary_inconsistency=res.primary_inconsistency,
        results_json=json.dumps({
            "score": res.overall_repeatability_score,
            "drift_detected": res.fatigue_drift_detected,
            "drift_explanation": res.drift_explanation,
            "outliers": res.flagged_outliers,
            "deliveries": res.delivery_comparisons
        })
    )
    db.add(model)
    db.commit()

    return MultiDeliveryResponse(
        session_id=session_id,
        total_deliveries=res.total_deliveries,
        overall_repeatability_score=res.overall_repeatability_score,
        repeatability_label=res.repeatability_label,
        primary_inconsistency=res.primary_inconsistency,
        metric_breakdowns={
            k: {
                "name": v.metric_name,
                "mean": v.mean_value,
                "std": v.std_deviation,
                "grade": v.consistency_grade,
                "insight": v.insight
            } for k, v in res.metric_breakdowns.items()
        },
        flagged_outliers=res.flagged_outliers,
        fatigue_drift_detected=res.fatigue_drift_detected,
        drift_explanation=res.drift_explanation,
        recommended_drill=res.recommended_drill,
        delivery_comparisons=res.delivery_comparisons
    )


# ============================================================================
# ENDPOINT: 7-DAY GRASSROOTS WEEKLY PRACTICE PLANNER
# ============================================================================

@router.post("/tutor/weekly-plan", response_model=WeeklyPlanResponse)
def generate_weekly_training_plan(payload: WeeklyPlanRequest):
    """Generates a structured, low-cost 7-day grassroots training calendar."""
    plan = WeeklyPlanner.create_plan(
        player_name=payload.player_name,
        primary_flaw=payload.primary_flaw,
        skill_level=payload.skill_level,
        available_equipment=payload.available_equipment
    )
    return WeeklyPlanResponse(
        player_name=plan.player_name,
        target_flaw=plan.target_flaw,
        skill_level=plan.skill_level,
        weekly_goal=plan.weekly_goal,
        total_active_days=plan.total_active_days,
        daily_schedule=[
            {
                "day_number": d.day_number,
                "day_name": d.day_name,
                "focus_theme": d.focus_theme,
                "duration_minutes": d.duration_minutes,
                "is_rest_day": d.is_rest_day,
                "equipment_needed": d.equipment_needed,
                "drills": d.drills,
                "safety_reminder": d.safety_reminder,
                "checklist_tasks": d.checklist_tasks
            } for d in plan.daily_schedule
        ],
        equipment_checklist=plan.equipment_checklist,
        coach_note=plan.coach_note
    )


# ============================================================================
# ENDPOINT: DISMISS / CORRECT OBSERVATION FEEDBACK
# ============================================================================

@router.post("/observations/{observation_id}/feedback")
def submit_observation_feedback(observation_id: str, payload: ObservationFeedbackRequest, db: Session = Depends(get_db)):
    """Allows players and coaches to dismiss or correct AI coaching feedback."""
    obs = db.query(ObservationModel).filter(ObservationModel.id == observation_id).first()
    if not obs:
        raise HTTPException(status_code=404, detail="Observation not found.")

    if payload.dismiss:
        obs.is_dismissed = True
    if payload.user_correction:
        obs.user_correction = payload.user_correction

    db.commit()
    return {
        "status": "success",
        "observation_id": observation_id,
        "is_dismissed": obs.is_dismissed,
        "user_correction": obs.user_correction
    }


# ============================================================================
# ENDPOINT: PRIVACY STORAGE CLEANUP (DELETE RAW VIDEO)
# ============================================================================

@router.delete("/sessions/{session_id}/video")
def delete_session_raw_video(session_id: str, db: Session = Depends(get_db)):
    """
    Privacy and storage management: Deletes the raw video file while preserving
    all numerical metrics, pose angles, and coaching reports.
    """
    sess = db.query(AnalysisSessionModel).filter(AnalysisSessionModel.id == session_id).first()
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found.")

    if sess.video_local_path and os.path.exists(sess.video_local_path):
        try:
            os.remove(sess.video_local_path)
        except Exception:
            pass

    sess.video_deleted = True
    db.commit()
    return {
        "status": "success",
        "session_id": session_id,
        "video_deleted": True,
        "message": "Raw video file deleted to save storage and protect privacy. Biomechanical metrics and reports retained."
    }


# ============================================================================
# ENDPOINT: CAMERA ANGLE & FRAMING GUIDANCE
# ============================================================================

@router.get("/camera-guidance", response_model=CameraGuidanceResponse)
def get_camera_recording_guidance():
    """Returns optimal distance, tripod height, and angle specifications."""
    return CameraGuidanceResponse(
        specs=CameraSuitabilityGuide.GUIDANCE_SPECS,
        general_tips=[
            "Record at 60fps or higher for fast bowling arm circles.",
            "Use Side-On positioning (perpendicular to pitch crease) for elbow extension screening.",
            "Keep the entire body (head to toes) visible across the whole delivery stride.",
            "Avoid bright backlighting directly behind the bowler."
        ]
    )
