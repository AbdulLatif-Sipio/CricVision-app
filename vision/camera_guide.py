"""
vision/camera_guide.py - Camera Angle Guidance & Pre-Analysis Video Quality Auditor.

Ensures users record video from proper perspectives (side-on for elbow screening,
front-on for alignment/seam) with full body framing, preventing inaccurate or
hallucinated measurements caused by bad camera placement.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class QualityCheckResult:
    is_suitable: bool
    quality_score: float  # 0.0 to 1.0
    detected_fps: float
    resolution: Tuple[int, int]
    frame_count: int
    duration_seconds: float
    recommended_angle: str
    checklist_status: Dict[str, bool]
    warnings: List[str]
    tips_for_recording: List[str]


class CameraSuitabilityGuide:
    """
    Validates video recording standards prior to heavy biomechanical processing.
    """

    GUIDANCE_SPECS = {
        "bowling_side_on": {
            "name": "Side-On (Perpendicular to Pitch Crease)",
            "primary_purpose": "Elbow extension screening (ICC Law 21.2), front-knee bracing, release height",
            "optimal_distance_m": "5.0 - 7.0 meters from popping crease",
            "tripod_height_m": "1.0 - 1.2 meters (waist height)",
            "recommended_fps": "60 fps or 120 fps slow-motion",
            "framing": "Full body from run-up stride through release and follow-through"
        },
        "bowling_front_on": {
            "name": "Front-On (Behind Stumps / Facing Bowler)",
            "primary_purpose": "Lateral trunk tilt, head alignment, seam presentation, pitch line/length",
            "optimal_distance_m": "6.0 - 8.0 meters behind target or bowling stumps",
            "tripod_height_m": "1.2 - 1.5 meters (chest height)",
            "recommended_fps": "30 fps or 60 fps",
            "framing": "Full bowling corridor showing approach alignment and pitch landing"
        },
        "batting_side_on": {
            "name": "Side-On Batting (Square Leg / Point)",
            "primary_purpose": "Leading elbow elevation, head over ball, weight transfer, bat backlift arc",
            "optimal_distance_m": "4.5 - 6.0 meters directly square of the batting crease",
            "tripod_height_m": "1.0 meter (waist height)",
            "recommended_fps": "60 fps",
            "framing": "Full batter stance, stride footprint, and follow-through balance"
        }
    }

    @classmethod
    def audit_video_quality(cls, video_path: str, activity_type: str = "bowling") -> QualityCheckResult:
        """
        Inspects video dimensions, duration, and frame rate before running deep pose models.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return QualityCheckResult(
                is_suitable=False,
                quality_score=0.0,
                detected_fps=0.0,
                resolution=(0, 0),
                frame_count=0,
                duration_seconds=0.0,
                recommended_angle="side_on",
                checklist_status={"file_readable": False},
                warnings=["Video file could not be opened or is corrupted."],
                tips_for_recording=["Ensure video is an MP4/MOV file and not DRM-protected."]
            )

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        duration = frame_count / fps if fps > 0 else 0.0

        warnings = []
        checklist = {
            "resolution_adequate": (w >= 640 and h >= 480),
            "duration_adequate": (1.0 <= duration <= 25.0),
            "frame_rate_adequate": (fps >= 24.0),
            "high_speed_fps": (fps >= 55.0)
        }

        score = 1.0

        if not checklist["resolution_adequate"]:
            score -= 0.3
            warnings.append(f"Resolution is low ({w}x{h}). High-definition (720p or 1080p) improves joint landmark accuracy.")

        if duration < 1.0:
            score -= 0.5
            warnings.append("Video clip is too short (< 1.0 second). Ensure clip contains gather through release.")
        elif duration > 30.0:
            score -= 0.2
            warnings.append("Video clip is lengthy (> 30 seconds). Trim clip to the delivery stride for fastest processing.")

        if not checklist["high_speed_fps"]:
            warnings.append(f"Video recorded at {fps:.0f}fps. For elbow extension screening, 60fps or higher reduces motion blur at release.")
            score -= 0.15

        score = float(np.clip(score, 0.1, 1.0))

        tips = [
            "Keep the smartphone steady on a tripod or flat surface at waist height.",
            "Record directly side-on (90 degrees to pitch) to check arm straightening.",
            "Make sure the player's head, hands, and feet remain within the frame throughout the entire motion."
        ]

        rec_angle = "side_on" if activity_type.lower() == "bowling" else "side_on"

        return QualityCheckResult(
            is_suitable=(score >= 0.5 and checklist["duration_adequate"]),
            quality_score=round(score, 2),
            detected_fps=round(fps, 1),
            resolution=(w, h),
            frame_count=frame_count,
            duration_seconds=round(duration, 2),
            recommended_angle=rec_angle,
            checklist_status=checklist,
            warnings=warnings,
            tips_for_recording=tips
        )
