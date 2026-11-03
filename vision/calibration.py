"""
vision/calibration.py - Metric Perspective & Cricket Ground Truth Calibration.

Implements real-world dimensional calibration using standard cricket regulations:
- Pitch Length: 22 yards = 20.1168 meters (popping crease to popping crease: 17.68m)
- Stump Height: 28 inches = 0.7112 meters
- Pitch Width: 10 feet = 3.048 meters
- Bowling-to-Popping Crease Distance: 4 feet = 1.2192 meters
- Cricket Ball Diameter: 7.2 cm

Allows pixel-to-meter translation for release speed (km/h) and pitch mapping.
"""

from dataclasses import dataclass
from typing import Tuple, Optional, List
import numpy as np
import cv2


@dataclass
class CricketDimensions:
    STUMP_HEIGHT_METERS: float = 0.7112
    STUMP_WIDTH_METERS: float = 0.2286
    PITCH_LENGTH_METERS: float = 20.1168
    POPPING_CREASE_TO_CREASE: float = 17.68
    CREASE_RETURN_WIDTH: float = 2.64
    BALL_DIAMETER_METERS: float = 0.072


class MetricCalibrator:
    """
    Computes real-world scale factors and perspective homography matrices.
    """

    def __init__(self, pixels_per_meter: Optional[float] = None):
        self.pixels_per_meter = pixels_per_meter
        self.homography_matrix: Optional[np.ndarray] = None

    def calibrate_from_stump(self, stump_top_px: Tuple[int, int], stump_base_px: Tuple[int, int]) -> float:
        """
        Calibrates scale from a detected 28-inch (0.7112m) cricket stump in the frame.
        """
        pixel_height = np.sqrt(
            (stump_top_px[0] - stump_base_px[0]) ** 2 +
            (stump_top_px[1] - stump_base_px[1]) ** 2
        )
        if pixel_height > 5:
            self.pixels_per_meter = pixel_height / CricketDimensions.STUMP_HEIGHT_METERS
        else:
            self.pixels_per_meter = 120.0  # sensible default for 720p/1080p footage

        return float(self.pixels_per_meter)

    def calibrate_from_player_height(
        self,
        head_px: Tuple[int, int],
        ankle_px: Tuple[int, int],
        estimated_player_height_m: float = 1.80
    ) -> float:
        """
        Calibrates scale using the detected player's height when standing in the delivery stride.
        """
        pixel_h = abs(head_px[1] - ankle_px[1])
        if pixel_h > 20:
            self.pixels_per_meter = pixel_h / estimated_player_height_m
        else:
            self.pixels_per_meter = 120.0

        return float(self.pixels_per_meter)

    def pixels_to_meters(self, pixel_distance: float) -> float:
        """
        Converts pixel distance to meters.
        """
        ppm = self.pixels_per_meter if self.pixels_per_meter is not None else 120.0
        return float(pixel_distance / ppm)

    def compute_velocity_kmh(self, pixel_displacement: float, dt_seconds: float) -> float:
        """
        Converts pixel displacement over elapsed time into km/h.
        v (m/s) = (pixel_displacement / ppm) / dt
        v (km/h) = v (m/s) * 3.6
        """
        if dt_seconds <= 1e-6:
            return 0.0

        meters = self.pixels_to_meters(pixel_displacement)
        speed_mps = meters / dt_seconds
        speed_kmh = speed_mps * 3.6
        return float(speed_kmh)
