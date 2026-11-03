"""
data/sample_generator.py - Realistic Cricket Delivery & Stroke Video Generator.

Generates out-of-the-box sample cricket videos (MP4) with realistic human biomechanics:
1. Legal Side-On Pace Delivery (James Anderson / Pat Cummins style)
2. Illegal Throw / Chucking Delivery (Snapping elbow > 15° to trigger ICC Law 21.2)
3. Sling / Round-Arm Delivery (Lasith Malinga / Matheesha Pathirana style)
4. Classic Cover Drive Batting Shot (Virat Kohli / Babar Azam style)
5. Back-Foot Pull Shot (Rohit Sharma style)

Allows zero-setup testing and instant live demonstrations.
"""

import os
from typing import Tuple, List, Dict, Any
import numpy as np
import cv2


class CricketSampleGenerator:
    """
    Renders realistic cricket kinematic actions onto synthetic video files.
    """

    @staticmethod
    def generate_bowling_clip(
        output_path: str,
        delivery_style: str = "legal_side_on",  # "legal_side_on", "illegal_chucking", "sling_round_arm"
        duration_sec: float = 3.0,
        fps: int = 30,
        resolution: Tuple[int, int] = (640, 480)
    ) -> str:
        """
        Creates an MP4 video showing an authentic bowling action sequence.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        w, h = resolution
        n_frames = int(duration_sec * fps)

        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        # Ground and pitch coords
        ground_y = int(h * 0.88)
        base_x = int(w * 0.35)

        for f in range(n_frames):
            frame = np.zeros((h, w, 3), dtype=np.uint8)

            # Draw cricket background: Turf grass & pitch deck
            frame[:ground_y] = (60, 110, 40)   # Outfield green
            frame[ground_y:] = (30, 80, 25)    # Foreground turf
            # Pitch strip
            cv2.rectangle(frame, (int(w * 0.1), ground_y - 20), (int(w * 0.9), ground_y), (120, 170, 195), -1)
            # Crease line
            cv2.line(frame, (int(w * 0.75), ground_y - 25), (int(w * 0.75), ground_y + 10), (255, 255, 255), 2)

            # Player kinematics phase
            # Phase 1: Run-up stride (0 to 35%)
            # Phase 2: Gather & BFC (35% to 50%)
            # Phase 3: FFC & Arm rotation (50% to 75%)
            # Phase 4: Release & Follow-through (75% to 100%)
            progress = f / n_frames

            # Calculate torso position
            forward_x = base_x + int(progress * (w * 0.35))
            torso_y = ground_y - 170

            # Arm rotation & elbow flexion
            if progress < 0.40:
                # Run-up pump
                shoulder_y = torso_y + 20
                shoulder_x = forward_x
                elbow_x = shoulder_x - 20
                elbow_y = shoulder_y + 25
                wrist_x = elbow_x + 10
                wrist_y = elbow_y - 20
                front_knee_ang = 140.0
            elif progress < 0.75:
                # Delivery swing
                swing_prog = (progress - 0.40) / 0.35
                arm_angle_rad = -np.pi / 2 + swing_prog * (2 * np.pi)

                shoulder_x = forward_x
                shoulder_y = torso_y + 10

                arm_len = 55
                forearm_len = 50

                if delivery_style == "sling_round_arm":
                    # Lower round-arm trajectory
                    elbow_x = int(shoulder_x + arm_len * np.cos(arm_angle_rad))
                    elbow_y = int(shoulder_y + arm_len * 0.5 * np.sin(arm_angle_rad))
                    wrist_x = int(elbow_x + forearm_len * np.cos(arm_angle_rad))
                    wrist_y = int(elbow_y + forearm_len * 0.5 * np.sin(arm_angle_rad))
                    front_knee_ang = 152.0
                elif delivery_style == "illegal_chucking":
                    # Elbow starts bent at 120° at horizontal, then aggressively snaps to 175° at release!
                    dynamic_bend = 40.0 * np.sin(swing_prog * np.pi)  # 40 degree snap!
                    elbow_x = int(shoulder_x + arm_len * np.cos(arm_angle_rad))
                    elbow_y = int(shoulder_y + arm_len * np.sin(arm_angle_rad))
                    snap_rad = arm_angle_rad + np.radians(dynamic_bend)
                    wrist_x = int(elbow_x + forearm_len * np.cos(snap_rad))
                    wrist_y = int(elbow_y + forearm_len * np.sin(snap_rad))
                    front_knee_ang = 145.0
                else:  # legal_side_on
                    # Rigid, straight arm throughout circle (extension delta < 6°)
                    elbow_x = int(shoulder_x + arm_len * np.cos(arm_angle_rad))
                    elbow_y = int(shoulder_y + arm_len * np.sin(arm_angle_rad))
                    wrist_x = int(elbow_x + forearm_len * np.cos(arm_angle_rad))
                    wrist_y = int(elbow_y + forearm_len * np.sin(arm_angle_rad))
                    front_knee_ang = 168.0  # textbook braced front leg
            else:
                # Follow through
                ft_prog = (progress - 0.75) / 0.25
                shoulder_x = forward_x
                shoulder_y = torso_y + 30
                elbow_x = shoulder_x + 25
                elbow_y = shoulder_y + 40
                wrist_x = elbow_x - 15
                wrist_y = elbow_y + 30
                front_knee_ang = 160.0

            # Draw Player Silhouette (Head, Torso, Legs, Arms)
            head_center = (forward_x, torso_y - 25)
            cv2.circle(frame, head_center, 16, (220, 220, 220), -1)  # Head
            # Torso
            cv2.line(frame, (forward_x, torso_y - 10), (forward_x, torso_y + 60), (200, 200, 200), 12)
            # Legs
            hip_pos = (forward_x, torso_y + 60)
            front_ankle = (forward_x + 40, ground_y - 5)
            back_ankle = (forward_x - 35, ground_y - 5)
            # Front leg (braced or collapsed)
            knee_x = int(hip_pos[0] + 0.5 * (front_ankle[0] - hip_pos[0]))
            knee_y = int(hip_pos[1] + 0.5 * (front_ankle[1] - hip_pos[1]) + (180.0 - front_knee_ang) * 0.4)
            cv2.line(frame, hip_pos, (knee_x, knee_y), (180, 180, 180), 8)
            cv2.line(frame, (knee_x, knee_y), front_ankle, (180, 180, 180), 8)
            # Back leg
            cv2.line(frame, hip_pos, back_ankle, (150, 150, 150), 6)

            # Bowling arm
            cv2.line(frame, (shoulder_x, shoulder_y), (elbow_x, elbow_y), (230, 230, 230), 7)
            cv2.line(frame, (elbow_x, elbow_y), (wrist_x, wrist_y), (240, 240, 240), 7)
            cv2.circle(frame, (wrist_x, wrist_y), 5, (0, 0, 255), -1)  # Ball in hand

            # Ball trajectory post-release
            if progress >= 0.70:
                ball_f = progress - 0.70
                ball_x = int(wrist_x + ball_f * 450)
                ball_y = int(wrist_y + ball_f * 200)
                if ball_y < ground_y - 5:
                    cv2.circle(frame, (ball_x, ball_y), 5, (0, 0, 255), -1)

            # Status overlay banner
            cv2.putText(
                frame,
                f"CricVision Benchmark: {delivery_style.upper()}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            out.write(frame)

        out.release()
        return output_path

    @staticmethod
    def generate_batting_clip(
        output_path: str,
        shot_type: str = "cover_drive",
        duration_sec: float = 3.0,
        fps: int = 30,
        resolution: Tuple[int, int] = (640, 480)
    ) -> str:
        """
        Creates an MP4 video showing a batting stroke sequence.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        w, h = resolution
        n_frames = int(duration_sec * fps)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        ground_y = int(h * 0.88)
        base_x = int(w * 0.45)

        for f in range(n_frames):
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            frame[:ground_y] = (50, 105, 35)
            frame[ground_y:] = (25, 75, 20)

            # Stumps behind batter
            stump_x = base_x - 70
            for sx in [stump_x - 6, stump_x, stump_x + 6]:
                cv2.line(frame, (sx, ground_y - 65), (sx, ground_y), (255, 255, 255), 2)

            progress = f / n_frames
            torso_y = ground_y - 160

            if shot_type == "cover_drive":
                # Stride forward, high leading elbow
                stride_x = int(progress * 45)
                shoulder_x = base_x + stride_x
                shoulder_y = torso_y + 15
                elbow_x = shoulder_x + 35
                elbow_y = shoulder_y - 25  # High leading elbow
                wrist_x = elbow_x + 15
                wrist_y = elbow_y + 35
                bat_end_x = wrist_x + 40
                bat_end_y = wrist_y + 60
            else:  # pull shot
                # Back and across
                stride_x = -int(progress * 25)
                shoulder_x = base_x + stride_x
                shoulder_y = torso_y + 10
                elbow_x = shoulder_x - 20
                elbow_y = shoulder_y - 15
                wrist_x = elbow_x + 30
                wrist_y = elbow_y + 20
                bat_end_x = wrist_x + 55
                bat_end_y = wrist_y - 10

            # Draw batter
            head_center = (shoulder_x, torso_y - 25)
            cv2.circle(frame, head_center, 16, (220, 220, 220), -1)
            cv2.line(frame, (shoulder_x, torso_y - 10), (shoulder_x, torso_y + 60), (200, 200, 200), 12)

            # Legs
            hip_pos = (shoulder_x, torso_y + 60)
            cv2.line(frame, hip_pos, (shoulder_x + 25, ground_y - 5), (180, 180, 180), 8)
            cv2.line(frame, hip_pos, (shoulder_x - 25, ground_y - 5), (150, 150, 150), 8)

            # Bat & arms
            cv2.line(frame, (shoulder_x, shoulder_y), (elbow_x, elbow_y), (230, 230, 230), 6)
            cv2.line(frame, (elbow_x, elbow_y), (wrist_x, wrist_y), (230, 230, 230), 6)
            # Wooden bat
            cv2.line(frame, (wrist_x, wrist_y), (bat_end_x, bat_end_y), (42, 100, 165), 10)

            cv2.putText(
                frame,
                f"CricVision Batting: {shot_type.upper()}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            out.write(frame)

        out.release()
        return output_path
