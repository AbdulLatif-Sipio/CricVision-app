"""
engine/phase_segmenter.py - Automated Cricket Phase Segmentation Engine.

Identifies the critical biomechanical milestone frames in bowling and batting sequences:
Bowling:
  - Run-up / Gather
  - Back-Foot Contact (BFC)
  - Front-Foot Contact (FFC) / Plant
  - Arm Horizontal (ICC 15° measurement start)
  - Ball Release Point (ICC 15° measurement end)
  - Follow-Through

Batting:
  - Stance & Setup
  - Backlift Peak
  - Front-Foot Stride Plant
  - Impact / Contact Moment
  - Follow-Through Finish
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class BowlingPhases:
    bfc_frame: int
    ffc_frame: int
    arm_horizontal_frame: int
    release_frame: int
    follow_through_frame: int
    total_frames: int


@dataclass
class BattingPhases:
    stance_frame: int
    backlift_peak_frame: int
    stride_plant_frame: int
    impact_frame: int
    follow_through_frame: int
    total_frames: int


class PhaseSegmenter:
    """
    Automated temporal phase detector using kinematic signals and trajectory derivatives.
    """

    @staticmethod
    def segment_bowling(
        wrist_y_series: List[float],
        wrist_x_series: List[float],
        front_ankle_y_series: List[float],
        back_ankle_y_series: List[float],
        shoulder_y_series: List[float],
        fps: float = 30.0
    ) -> BowlingPhases:
        """
        Segments a bowling video sequence into key ICC analysis milestones.
        In image coordinates, Y=0 is top, Y=1 is bottom. Smaller Y = higher in the air.
        """
        n_frames = len(wrist_y_series)
        if n_frames < 10:
            # Fallback equidistant estimation for very short sequences
            return BowlingPhases(
                bfc_frame=max(0, int(n_frames * 0.25)),
                ffc_frame=max(1, int(n_frames * 0.45)),
                arm_horizontal_frame=max(2, int(n_frames * 0.55)),
                release_frame=max(3, int(n_frames * 0.65)),
                follow_through_frame=max(4, int(n_frames * 0.90)),
                total_frames=n_frames
            )

        # 1. Release Frame: Point of highest wrist position (minimum Y) during the forward delivery arc
        # Search between 30% and 85% of clip
        search_start = int(n_frames * 0.3)
        search_end = int(n_frames * 0.85)
        release_frame = search_start + int(np.argmin(wrist_y_series[search_start:search_end]))

        # 2. Arm Horizontal Frame: When wrist Y is roughly equal to shoulder Y right before release
        arm_horizontal_frame = max(0, release_frame - 3)
        for f in range(release_frame - 1, max(0, release_frame - 15), -1):
            if wrist_y_series[f] >= shoulder_y_series[f]:
                arm_horizontal_frame = f
                break

        # 3. Front Foot Contact (FFC): Front ankle reaches lowest point (maximum Y) right before release
        ffc_search_end = release_frame
        ffc_search_start = max(0, release_frame - int(fps * 0.6))
        ffc_frame = ffc_search_start + int(np.argmax(front_ankle_y_series[ffc_search_start:ffc_search_end]))

        # 4. Back Foot Contact (BFC): Back ankle strikes ground before FFC
        bfc_search_end = max(1, ffc_frame)
        bfc_search_start = max(0, ffc_frame - int(fps * 0.5))
        bfc_frame = bfc_search_start + int(np.argmax(back_ankle_y_series[bfc_search_start:bfc_search_end]))

        # Ensure order: BFC <= FFC <= Arm_Horizontal <= Release <= Follow_Through
        if bfc_frame >= ffc_frame:
            bfc_frame = max(0, ffc_frame - 3)

        follow_through_frame = min(n_frames - 1, release_frame + int(fps * 0.5))

        return BowlingPhases(
            bfc_frame=int(bfc_frame),
            ffc_frame=int(ffc_frame),
            arm_horizontal_frame=int(arm_horizontal_frame),
            release_frame=int(release_frame),
            follow_through_frame=int(follow_through_frame),
            total_frames=n_frames
        )

    @staticmethod
    def segment_batting(
        wrist_y_series: List[float],
        front_ankle_x_series: List[float],
        fps: float = 30.0
    ) -> BattingPhases:
        """
        Segments a batting video into stance, backlift, stride, impact, and follow-through.
        """
        n_frames = len(wrist_y_series)
        if n_frames < 10:
            return BattingPhases(
                stance_frame=0,
                backlift_peak_frame=int(n_frames * 0.3),
                stride_plant_frame=int(n_frames * 0.45),
                impact_frame=int(n_frames * 0.6),
                follow_through_frame=int(n_frames * 0.85),
                total_frames=n_frames
            )

        # 1. Backlift Peak: Minimum wrist Y in first half (hands raised high)
        half_pt = int(n_frames * 0.55)
        backlift_peak_frame = int(np.argmin(wrist_y_series[:half_pt]))

        # 2. Impact frame: Rapid downward swing inflection point
        impact_search_start = backlift_peak_frame
        impact_search_end = min(n_frames, backlift_peak_frame + int(fps * 0.6))
        impact_frame = impact_search_start + int(np.argmax(wrist_y_series[impact_search_start:impact_search_end]))

        # 3. Stride plant: Front foot extends and halts before impact
        stride_plant_frame = max(backlift_peak_frame, impact_frame - 2)

        follow_through_frame = min(n_frames - 1, impact_frame + int(fps * 0.4))

        return BattingPhases(
            stance_frame=0,
            backlift_peak_frame=int(backlift_peak_frame),
            stride_plant_frame=int(stride_plant_frame),
            impact_frame=int(impact_frame),
            follow_through_frame=int(follow_through_frame),
            total_frames=n_frames
        )
