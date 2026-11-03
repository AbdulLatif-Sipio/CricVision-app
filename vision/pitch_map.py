"""
vision/pitch_map.py - 2D Cricket Pitch Mapping & Length/Line Zoning System.

Projects ball landing coordinates onto a standard 22-yard 2D cricket pitch:
- Lengths: Yorker, Full / Slot, Good Length, Back of a Length, Short / Bouncer.
- Lines: Outside Off / 5th Stump, Off Stump, Middle Stump, Leg Stump, Down Leg.
- Draws high-resolution 2D pitch diagrams for the coaching report.
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
import numpy as np
import cv2


@dataclass
class PitchZoneClassification:
    length_zone: str  # "Yorker", "Full", "Good Length", "Back of a Length", "Short"
    line_zone: str    # "Outside Off", "Off Stump", "Middle Stump", "Leg Stump", "Down Leg"
    distance_from_crease_m: float
    lateral_offset_m: float  # 0.0 is middle stump, negative is off-side, positive is leg-side
    tactical_verdict: str


class PitchMapper:
    """
    Cricket pitch trajectory and impact zone analyzer.
    """

    @staticmethod
    def classify_landing(
        distance_from_crease_m: float,
        lateral_offset_m: float = 0.15
    ) -> PitchZoneClassification:
        """
        Classifies length and line according to professional cricket coaching standards.
        """
        # 1. Length Classification
        if distance_from_crease_m <= 2.2:
            length = "Yorker"
            tactical = "Toe-crusher / Blockhole. High wicket-taking probability in death overs."
        elif distance_from_crease_m <= 4.8:
            length = "Full / Slot"
            tactical = "Driving length. Invites front-foot drives; high risk of boundary if not swinging."
        elif distance_from_crease_m <= 7.8:
            length = "Good Length"
            tactical = "The corridor of uncertainty. Induces hesitation between front and back foot."
        elif distance_from_crease_m <= 10.5:
            length = "Back of a Length"
            tactical = "Defensive/heavy ball. Forces batter onto the back foot to defend or punch."
        else:
            length = "Short / Bouncer"
            tactical = "Surprise bouncer. Tests batter's pull shot or helmet protection."

        # 2. Line Classification
        if lateral_offset_m < -0.30:
            line = "Wide Outside Off"
        elif lateral_offset_m < -0.10:
            line = "5th Stump Channel (Corridor)"
        elif lateral_offset_m <= 0.08:
            line = "Off & Middle Stump"
        elif lateral_offset_m <= 0.25:
            line = "Leg Stump Line"
        else:
            line = "Down Leg Side"

        return PitchZoneClassification(
            length_zone=length,
            line_zone=line,
            distance_from_crease_m=round(distance_from_crease_m, 2),
            lateral_offset_m=round(lateral_offset_m, 2),
            tactical_verdict=tactical
        )

    @staticmethod
    def generate_pitch_diagram(
        landing_dist_m: float = 6.2,
        lateral_offset_m: float = -0.15,
        width: int = 360,
        height: int = 700
    ) -> np.ndarray:
        """
        Renders a clean, broadcast-style 2D pitch map diagram (BGR image).
        """
        # Pitch canvas: Rich cricket turf green
        pitch_img = np.zeros((height, width, 3), dtype=np.uint8)
        pitch_img[:] = (34, 120, 50)  # Turf green

        # Pitch strip (clay/straw yellow-brown)
        px_margin = 40
        strip_left = px_margin
        strip_right = width - px_margin
        strip_width = strip_right - strip_left

        strip_top = 40
        strip_bottom = height - 40
        strip_height = strip_bottom - strip_top

        cv2.rectangle(pitch_img, (strip_left, strip_top), (strip_right, strip_bottom), (130, 180, 205), -1)
        cv2.rectangle(pitch_img, (strip_left, strip_top), (strip_right, strip_bottom), (255, 255, 255), 2)

        # Draw Stumps & Creases
        # Batter popping crease (near bottom)
        crease_y = strip_bottom - 70
        cv2.line(pitch_img, (strip_left - 15, crease_y), (strip_right + 15, crease_y), (255, 255, 255), 3)

        # Bowling crease (stump line)
        stump_y = strip_bottom - 30
        cv2.line(pitch_img, (strip_left, stump_y), (strip_right, stump_y), (255, 255, 255), 2)

        # Stumps (3 white dots)
        mid_x = (strip_left + strip_right) // 2
        for sx in [mid_x - 12, mid_x, mid_x + 12]:
            cv2.circle(pitch_img, (sx, stump_y), 5, (255, 255, 255), -1)

        # Length Zone horizontal guides
        # Convert distance (0 to 18m) to Y pixel coordinate:
        # 0m is crease_y, 18m is near strip_top
        playable_px = crease_y - strip_top
        def dist_to_y(d_m):
            ratio = np.clip(d_m / 17.68, 0.0, 1.0)
            return int(crease_y - ratio * playable_px)

        # Zone boundaries
        y_yorker = dist_to_y(2.2)
        y_full = dist_to_y(4.8)
        y_good = dist_to_y(7.8)
        y_back = dist_to_y(10.5)

        # Draw subtle zone bands
        # Yorker zone
        cv2.rectangle(pitch_img, (strip_left, y_yorker), (strip_right, crease_y), (100, 150, 180), 1)
        cv2.putText(pitch_img, "YORKER", (strip_left + 8, crease_y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (70, 70, 70), 1)

        # Full zone
        cv2.rectangle(pitch_img, (strip_left, y_full), (strip_right, y_yorker), (110, 160, 190), 1)
        cv2.putText(pitch_img, "FULL", (strip_left + 8, y_yorker - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (70, 70, 70), 1)

        # Good length zone (highlighted in gold/yellow)
        cv2.rectangle(pitch_img, (strip_left, y_good), (strip_right, y_full), (100, 200, 255), 2)
        cv2.putText(pitch_img, "GOOD LENGTH", (strip_left + 8, y_full - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 100, 200), 1)

        # Back of a length
        cv2.rectangle(pitch_img, (strip_left, y_back), (strip_right, y_good), (110, 160, 190), 1)
        cv2.putText(pitch_img, "BACK OF LENGTH", (strip_left + 8, y_good - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (70, 70, 70), 1)

        # Short zone
        cv2.putText(pitch_img, "SHORT / BOUNCER", (strip_left + 8, y_back - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (70, 70, 70), 1)

        # Draw the actual landing impact spot
        ball_y = dist_to_y(landing_dist_m)
        ball_x = int(mid_x + (lateral_offset_m / 1.5) * (strip_width / 2))
        ball_x = np.clip(ball_x, strip_left + 8, strip_right - 8)

        # Glowing ball impact point
        cv2.circle(pitch_img, (ball_x, ball_y), 14, (0, 0, 255), 2)
        cv2.circle(pitch_img, (ball_x, ball_y), 9, (0, 0, 255), -1)
        cv2.circle(pitch_img, (ball_x, ball_y), 4, (255, 255, 255), -1)

        # Coordinates label
        label = f"{landing_dist_m:.1f}m"
        cv2.putText(pitch_img, label, (ball_x + 16, ball_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
        cv2.putText(pitch_img, label, (ball_x + 16, ball_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        return pitch_img
