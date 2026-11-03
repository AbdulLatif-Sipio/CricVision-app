"""
backend/schemas.py - Production Pydantic v2 API Schemas.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    status_code: int = Field(..., description="HTTP status code")
    error: str = Field(..., description="Error classification")
    message: str = Field(..., description="Human-readable explanation")
    details: Optional[Any] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class PrescriptiveDrillSchema(BaseModel):
    drill_name: str
    flaw_name: str
    target_metric: str
    equipment_needed: str
    reps_and_frequency: str
    instructions: List[str]
    pro_model_reference: str


class StructuredObservationSchema(BaseModel):
    observation_id: str
    main_observation: str
    what_this_means: str
    confidence_score: float
    confidence_label: str
    recommended_improvement: str
    drill_name: str
    drill_equipment: str
    safety_guidance: str
    next_session_focus: str
    can_dismiss_or_correct: bool = True
    is_dismissed: bool = False
    user_correction: Optional[str] = None


class AnalysisResponse(BaseModel):
    session_id: str
    player_name: str
    activity_type: str
    media_type: str
    accuracy_score: float
    release_speed_kmh: float
    release_speed_mph: float
    is_legal: bool
    icc_status: str
    elbow_extension_deg: float
    front_knee_angle_deg: float
    matched_legend: str
    style_category: str
    pitch_zone: str = "Good Length | 5th Stump Channel"
    what_went_right: List[str]
    mistakes_spotted: List[str]
    better_options: List[str]
    drills: List[PrescriptiveDrillSchema]
    primary_observation: Optional[StructuredObservationSchema] = None
    camera_suitability: Optional[Dict[str, Any]] = None
    screening_disclaimer: str = (
        "Screening indicator only: 2D video pose tracking estimates dynamic elbow extension for "
        "coaching guidance and does not replace official 3D laboratory testing."
    )
    confidence_level: str = "HIGH"
    why_am_i_seeing_this: str = ""
    annotated_media_b64: Optional[str] = None
    coaching_summary: str
    processed_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class ObservationFeedbackRequest(BaseModel):
    dismiss: bool = False
    user_correction: Optional[str] = None


class SessionComparisonRequest(BaseModel):
    session_a_id: str
    session_b_id: str


class SessionComparisonResponse(BaseModel):
    session_a_id: str
    session_b_id: str
    baseline_date: str
    current_date: str
    days_between: int
    overall_progress_status: str
    scorecard_delta: float
    key_improvements: List[str]
    areas_still_needing_work: List[str]
    coach_verdict: str
    next_session_focus: str
    metric_deltas: Dict[str, Any]


class MultiDeliveryItem(BaseModel):
    delivery_number: int
    elbow_extension_deg: float
    front_knee_angle_deg: float
    trunk_tilt_deg: float
    release_speed_kmh: float = 0.0
    head_tilt_deg: float = 8.0
    is_legal: bool = True
    pitch_zone: str = "Good Length | Off-Stump"


class MultiDeliveryRequest(BaseModel):
    player_name: str = "Grassroots Player"
    deliveries: List[MultiDeliveryItem]


class MultiDeliveryResponse(BaseModel):
    session_id: str
    total_deliveries: int
    overall_repeatability_score: float
    repeatability_label: str
    primary_inconsistency: str
    metric_breakdowns: Dict[str, Any]
    flagged_outliers: List[Dict[str, Any]]
    fatigue_drift_detected: bool
    drift_explanation: str
    recommended_drill: Dict[str, Any]
    delivery_comparisons: List[Dict[str, Any]]


class WeeklyPlanRequest(BaseModel):
    player_name: str = "Grassroots Player"
    primary_flaw: str = "collapsed_front_knee"
    skill_level: str = "Intermediate"
    available_equipment: Optional[List[str]] = None


class WeeklyPlanResponse(BaseModel):
    player_name: str
    target_flaw: str
    skill_level: str
    weekly_goal: str
    total_active_days: int
    daily_schedule: List[Dict[str, Any]]
    equipment_checklist: List[str]
    coach_note: str


class CameraGuidanceResponse(BaseModel):
    specs: Dict[str, Any]
    general_tips: List[str]


class HistoryItem(BaseModel):
    session_id: str
    player_name: str
    activity_type: str
    media_type: str
    accuracy_score: float
    release_speed_kmh: float
    is_legal: bool
    matched_legend: str
    style_category: str
    created_at: str


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    environment: str
    database: str
    mediapipe_model: str
    uptime_seconds: float
