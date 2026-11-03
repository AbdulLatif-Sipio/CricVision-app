"""
engine/kinematics.py - Mathematical & Biomechanical Kinematics Core.

Computes 3D joint angles, angular velocities, rotational torques,
and spatial alignment vectors using rigorous vector geometry.
"""

from typing import Tuple, List, Dict, Optional, Union
import numpy as np


class BiomechanicalKinematics:
    """
    Precision 3D geometric computations for human pose keypoints.
    """

    @staticmethod
    def calculate_3d_angle(
        a: Union[np.ndarray, List[float]],
        b: Union[np.ndarray, List[float]],
        c: Union[np.ndarray, List[float]]
    ) -> float:
        """
        Calculates the 3D angle (in degrees) at joint B formed by limbs BA and BC.
        Uses the dot product: cos(theta) = (BA . BC) / (|BA| * |BC|)
        """
        a_vec = np.array(a, dtype=np.float64)
        b_vec = np.array(b, dtype=np.float64)
        c_vec = np.array(c, dtype=np.float64)

        ba = a_vec - b_vec
        bc = c_vec - b_vec

        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)

        if norm_ba < 1e-7 or norm_bc < 1e-7:
            return 180.0

        cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
        # Numerical stability clip [-1.0, 1.0]
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

        angle = np.degrees(np.arccos(cosine_angle))
        return float(angle)

    @staticmethod
    def calculate_angle_with_vertical(
        proximal: Union[np.ndarray, List[float]],
        distal: Union[np.ndarray, List[float]]
    ) -> float:
        """
        Calculates the angle of a segment (e.g., bowling arm or spine)
        relative to the vertical axis (gravity vector [0, -1, 0] or [0, 1, 0] in image space).
        Returns angle in degrees (0° = pointing straight down, 180° = straight up, 90° = horizontal).
        """
        p = np.array(proximal, dtype=np.float64)
        d = np.array(distal, dtype=np.float64)
        segment = d - p

        # In image coords, Y points downwards; vertical vector pointing upwards is [0, -1, 0]
        vertical_up = np.array([0.0, -1.0, 0.0], dtype=np.float64)

        norm_seg = np.linalg.norm(segment[:2])  # 2D projection or 3D
        if norm_seg < 1e-7:
            return 90.0

        cos_theta = np.dot(segment[:2], vertical_up[:2]) / (norm_seg * np.linalg.norm(vertical_up[:2]))
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_theta)))

    @staticmethod
    def calculate_shoulder_hip_alignment_angle(
        left_shoulder: np.ndarray,
        right_shoulder: np.ndarray,
        left_hip: np.ndarray,
        right_hip: np.ndarray
    ) -> float:
        """
        Calculates the transverse plane counter-rotation angle between the
        shoulder line vector and hip line vector.
        """
        shoulder_vec = np.array(right_shoulder[:2] - left_shoulder[:2], dtype=np.float64)
        hip_vec = np.array(right_hip[:2] - left_hip[:2], dtype=np.float64)

        norm_s = np.linalg.norm(shoulder_vec)
        norm_h = np.linalg.norm(hip_vec)

        if norm_s < 1e-7 or norm_h < 1e-7:
            return 0.0

        cos_val = np.dot(shoulder_vec, hip_vec) / (norm_s * norm_h)
        cos_val = np.clip(cos_val, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_val)))

    @staticmethod
    def calculate_angular_velocity(angles: List[float], fps: float = 30.0) -> List[float]:
        """
        Computes the first derivative of joint angles with respect to time (omega = dtheta / dt) in deg/s.
        """
        if len(angles) < 2:
            return [0.0] * len(angles)

        dt = 1.0 / fps
        velocities = np.gradient(angles, dt)
        return [float(v) for v in velocities]

    @staticmethod
    def calculate_center_of_mass_ratio(
        front_ankle: np.ndarray,
        back_ankle: np.ndarray,
        mid_hip: np.ndarray
    ) -> float:
        """
        Estimates the weight distribution ratio on the front foot (0.0 = 100% on back foot,
        1.0 = 100% on front foot, 0.5 = 50-50 balance).
        """
        # Horizontal X projection
        base_width = abs(front_ankle[0] - back_ankle[0])
        if base_width < 1e-5:
            return 0.5

        # Distance from back foot to hip
        d_from_back = abs(mid_hip[0] - back_ankle[0])
        ratio = d_from_back / base_width
        return float(np.clip(ratio, 0.0, 1.0))
