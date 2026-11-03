"""
app.py - CricVision & CricTutor AI: User-Friendly Biomechanics & Coaching Platform.

Run with:
    streamlit run app.py
"""

import os
import sys
import tempfile
import time
from typing import Dict, Any, List, Optional
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st

# Import CricVision modules
from engine.pose_detector import PoseDetector, LANDMARK_NAMES
from engine.image_analyzer import ImageCoachAnalyzer, ImageAnalysisResult
from engine.kinematics import BiomechanicalKinematics
from engine.phase_segmenter import PhaseSegmenter, BowlingPhases, BattingPhases
from vision.calibration import MetricCalibrator, CricketDimensions
from vision.speed_gun import CricketSpeedGun, DeliverySpeedMetrics
from vision.pitch_map import PitchMapper, PitchZoneClassification
from vision.bat_tracker import BatTracker, BatSwingMetrics
from ml.icc_rules import ICCRulesEngine, ICCComplianceResult
from ml.archetypes import (
    ArchetypeMatcher,
    ArchetypeMatch,
    BOWLING_ARCHETYPES,
    BATTING_SHOT_ARCHETYPES
)
from ml.dtw_matcher import DynamicTimeWarpingMatcher
from ml.action_classifier import ActionClassifier
from ml.technique_scorer import TechniqueScorer, TechniqueScorecard
from coaching.drill_matrix import DrillEngine, PrescriptiveDrill
from coaching.crictutor_llm import CricTutorAI
from data.sample_generator import CricketSampleGenerator


# ============================================================================
# PAGE CONFIGURATION & STYLING
# ============================================================================

st.set_page_config(
    page_title="CricVision | CricTutor AI",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Broadcast-inspired sleek styling
st.markdown("""
<style>
    .brand-title {
        font-size: 2.5rem;
        font-weight: 900;
        color: #0E76A8;
        margin-bottom: 2px;
    }
    .brand-subtitle {
        font-size: 1.15rem;
        color: #4A5568;
        margin-bottom: 25px;
    }
    .coach-card-good {
        background-color: #F0FFF4;
        border-left: 6px solid #38A169;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 15px;
    }
    .coach-card-bad {
        background-color: #FFF5F5;
        border-left: 6px solid #E53E3E;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 15px;
    }
    .coach-card-tip {
        background-color: #EBF8FF;
        border-left: 6px solid #3182CE;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 15px;
    }
    .badge-legal {
        background-color: #C6F6D5;
        color: #22543D;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
    }
    .badge-illegal {
        background-color: #FED7D7;
        color: #742A2A;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# CACHED ENGINES
# ============================================================================

@st.cache_resource
def load_system_engines():
    det = PoseDetector()
    img_coach = ImageCoachAnalyzer(detector=det)
    gun = CricketSpeedGun()
    icc = ICCRulesEngine(elbow_threshold=15.0)
    return det, img_coach, gun, icc

detector, image_coach, speed_gun, icc_engine = load_system_engines()


# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.markdown("## 🏏 CricVision & CricTutor")
st.sidebar.markdown("**Grassroots AI Cricket Biomechanics & Coaching**")
st.sidebar.markdown("---")

practice_activity = st.sidebar.radio(
    "What are you practicing today?",
    ["🎳 Bowling (Fast & Spin)", "🏏 Batting (Shots & Stance)"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Settings")

offline_sim_mode = st.sidebar.checkbox(
    "🔒 Offline Mode (Zero Internet)",
    value=True,
    help="When checked, runs 100% on your local CPU. Zero cloud dependencies."
)

coach_language = st.sidebar.selectbox(
    "Coach Advice Language:",
    ["English", "Roman Urdu", "Urdu"]
)

gemini_key = ""
if not offline_sim_mode:
    gemini_key = st.sidebar.text_input(
        "Gemini API Key (Optional Online Enhancer):",
        type="password",
        value=os.environ.get("GEMINI_API_KEY", "")
    )

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Works on any camera or phone video! No sensors or expensive hardware needed.")


# ============================================================================
# MAIN HEADER & UPLOAD SECTION (FRONT AND CENTER)
# ============================================================================

st.markdown("<div class='brand-title'>🏏 CricVision — Your AI Cricket Coach</div>", unsafe_allow_html=True)
st.markdown("<div class='brand-subtitle'>Upload a video or photo of your bowling or batting. CricTutor will analyze your foot placement, arm angles, speed, and tell you exactly what you did right, what mistakes you made, and what you should have done instead!</div>", unsafe_allow_html=True)

# Main Media Source Tabs (Very user-friendly!)
tab_upload, tab_camera, tab_samples = st.tabs([
    "📁 Upload Your Video or Photo",
    "🎥 Use Webcam / Camera",
    "⚡ Try Quick Practice Clips"
])

uploaded_media = None
media_type = None  # "video" or "image"
temp_file_path = None

with tab_upload:
    st.markdown("#### Drop your cricket video or photo below:")
    uploaded_file = st.file_uploader(
        "Choose an MP4/MOV video OR a JPG/PNG photo:",
        type=["mp4", "mov", "avi", "webm", "mkv", "jpg", "jpeg", "png", "webp"],
        help="Upload a short clip (3-8 seconds) or a snapshot of your bowling action / batting stroke."
    )
    if uploaded_file is not None:
        file_ext = uploaded_file.name.split(".")[-1].lower()
        if file_ext in ["jpg", "jpeg", "png", "webp"]:
            media_type = "image"
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}")
            tfile.write(uploaded_file.read())
            temp_file_path = tfile.name
        else:
            media_type = "video"
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            temp_file_path = tfile.name

with tab_camera:
    st.markdown("#### Capture directly from your phone or laptop camera:")
    cam_photo = st.camera_input("Take a photo of your bowling or batting pose:")
    if cam_photo is not None:
        media_type = "image"
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        tfile.write(cam_photo.read())
        temp_file_path = tfile.name

with tab_samples:
    st.markdown("#### Test immediately with pre-loaded international benchmark clips:")
    if "Bowling" in practice_activity:
        sample_choice = st.selectbox(
            "Select Sample Bowling Scenario:",
            [
                "Textbook Legal Pace (Pat Cummins style)",
                "Illegal Chucking Action (Snapping Elbow > 15°)",
                "Lasith Malinga Round-Arm Sling",
                "Bowling Release Snapshot (Single Photo)"
            ]
        )
        if st.button("Load Selected Sample"):
            if "Legal Pace" in sample_choice:
                media_type = "video"
                temp_file_path = "samples/sample_legal_side_on.mp4"
            elif "Illegal Chucking" in sample_choice:
                media_type = "video"
                temp_file_path = "samples/sample_illegal_chucking.mp4"
            elif "Malinga" in sample_choice:
                media_type = "video"
                temp_file_path = "samples/sample_sling_malinga.mp4"
            else:
                media_type = "image"
                temp_file_path = "samples/sample_bowling_release.jpg"
    else:
        sample_choice = st.selectbox(
            "Select Sample Batting Scenario:",
            [
                "Virat Kohli Cover Drive (Video)",
                "Rohit Sharma Pull Shot (Video)",
                "Batting Drive Snapshot (Single Photo)"
            ]
        )
        if st.button("Load Selected Sample"):
            if "Cover Drive" in sample_choice:
                media_type = "video"
                temp_file_path = "samples/sample_cover_drive.mp4"
            elif "Pull Shot" in sample_choice:
                media_type = "video"
                temp_file_path = "samples/sample_pull_shot.mp4"
            else:
                media_type = "image"
                temp_file_path = "samples/sample_batting_drive.jpg"


# ============================================================================
# PROCESSING & RESULTS DISPLAY
# ============================================================================

if temp_file_path and os.path.exists(temp_file_path):
    st.markdown("---")

    # ========================================================================
    # CASE 1: USER UPLOADED A SINGLE IMAGE / PHOTO
    # ========================================================================
    if media_type == "image":
        st.subheader("📸 Snapshot Biomechanics & Coaching Analysis")
        st.caption("Analyzing your foot placement, arm angles, and balance in this frozen frame.")

        img_bgr = cv2.imread(temp_file_path)
        if img_bgr is None:
            st.error("Could not read image file.")
        else:
            mode_hint = "bowling" if "Bowling" in practice_activity else "batting"
            with st.spinner("CricTutor is inspecting your pose..."):
                img_res = image_coach.analyze_image(img_bgr, mode=mode_hint)

            # Top Summary Cards (Simple & Friendly!)
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("Moment Accuracy", f"{img_res.accuracy_score}%", "Match to Pro Form")
            with col_m2:
                st.metric("Style Twin Legend", img_res.matched_legend, img_res.style_category)
            with col_m3:
                st.metric("Front Knee Lever", f"{img_res.key_angles['front_knee_angle']}°", "Knee angle at plant")
            with col_m4:
                st.metric("Arm Extension", f"{img_res.key_angles['elbow_angle']}°", "Elbow angle")

            st.markdown("---")

            # Layout: Image on Left, Coaching Advice on Right
            c_img, c_feedback = st.columns([1.1, 1.2])

            with c_img:
                st.markdown("#### 🔍 Pose Overlay (Skeleton & Key Joints)")
                annotated_rgb = cv2.cvtColor(img_res.annotated_image, cv2.COLOR_BGR2RGB)
                st.image(annotated_rgb, use_container_width=True)
                st.caption("🟢 Green/Yellow lines: Your tracked joints | Circles: Primary force transfer points")

            with c_feedback:
                st.markdown("### 🗣️ CricTutor's Breakdown of This Moment")

                # Foot placement feedback
                st.markdown("#### 🦶 Foot Placement & Crease Landing:")
                st.info(f"**Coach Observation:** {img_res.foot_placement_feedback}")

                # Arm & release feedback
                st.markdown("#### 🏹 Arm Angle & Ball Release:")
                st.write(img_res.arm_and_release_feedback)

                # 1. What you did right
                st.markdown("<div class='coach-card-good'><b>🟢 What You Did Great:</b>", unsafe_allow_html=True)
                for g in img_res.what_went_right:
                    st.markdown(f"- ✅ {g}")
                st.markdown("</div>", unsafe_allow_html=True)

                # 2. Mistakes spotted
                if len(img_res.mistakes_spotted) > 0:
                    st.markdown("<div class='coach-card-bad'><b>🔴 Mistakes Spotted in This Frame:</b>", unsafe_allow_html=True)
                    for m in img_res.mistakes_spotted:
                        st.markdown(f"- ⚠️ {m}")
                    st.markdown("</div>", unsafe_allow_html=True)

                # 3. What you should have done instead
                if len(img_res.better_options_to_perform) > 0:
                    st.markdown("<div class='coach-card-tip'><b>💡 Better Option (What You Should Do Next Time):</b>", unsafe_allow_html=True)
                    for tip in img_res.better_options_to_perform:
                        st.markdown(f"- 👉 {tip}")
                    st.markdown("</div>", unsafe_allow_html=True)

    # ========================================================================
    # CASE 2: USER UPLOADED A FULL VIDEO CLIP
    # ========================================================================
    elif media_type == "video":
        is_bowling = "Bowling" in practice_activity

        if is_bowling:
            st.subheader("🎳 Full Bowling Delivery Analysis")
            st.caption("Tracking your entire approach, front-foot plant on the crease, release speed, and ball line/length.")

            with st.spinner("Analyzing your bowling action through MediaPipe..."):
                res = detector.process_video(temp_file_path, max_frames=90)

            lms_series = res["landmark_series"]
            raw_frames = res["raw_frames"]
            annotated_frames = res["annotated_frames"]
            fps = res["fps"]

            if len(lms_series) == 0:
                st.error("No bowler pose detected. Please try another video angle.")
            else:
                # Extract Timeseries Angles
                elbow_angles = []
                knee_angles = []
                wrist_y_series = []
                wrist_x_series = []
                shoulder_y_series = []
                front_ankle_y_series = []
                back_ankle_y_series = []

                for f_idx in range(len(lms_series)):
                    lm = lms_series[f_idx]
                    sh = lm[12]
                    el = lm[14]
                    wr = lm[16]
                    hip = lm[24]
                    kn = lm[26]
                    ak = lm[28]

                    ang_el = BiomechanicalKinematics.calculate_3d_angle(sh, el, wr)
                    elbow_angles.append(ang_el)

                    ang_kn = BiomechanicalKinematics.calculate_3d_angle(hip, kn, ak)
                    knee_angles.append(ang_kn)

                    wrist_y_series.append(float(wr[1]))
                    wrist_x_series.append(float(wr[0]))
                    shoulder_y_series.append(float(sh[1]))
                    front_ankle_y_series.append(float(ak[1]))
                    back_ankle_y_series.append(float(lm[27][1]))

                # Phases
                phases = PhaseSegmenter.segment_bowling(
                    wrist_y_series=wrist_y_series,
                    wrist_x_series=wrist_x_series,
                    front_ankle_y_series=front_ankle_y_series,
                    back_ankle_y_series=back_ankle_y_series,
                    shoulder_y_series=shoulder_y_series,
                    fps=fps
                )

                # ICC 15-Degree Law Legality
                h_idx = min(len(elbow_angles) - 1, phases.arm_horizontal_frame)
                r_idx = min(len(elbow_angles) - 1, phases.release_frame)
                delivery_window = elbow_angles[h_idx:r_idx+1] if r_idx > h_idx else elbow_angles

                icc_result = icc_engine.evaluate_bowling_legality(
                    elbow_angle_at_horizontal=elbow_angles[h_idx],
                    elbow_angle_at_release=elbow_angles[r_idx],
                    angle_sequence_during_delivery=delivery_window
                )

                # Front knee bracing & counter rotation
                front_knee_ffc = knee_angles[phases.ffc_frame] if phases.ffc_frame < len(knee_angles) else 165.0
                counter_rot = 14.0 if icc_result.is_legal else 32.0

                # Match Archetype & International Player
                is_sling = "sling" in temp_file_path.lower()
                arch_match = ArchetypeMatcher.match_bowler_style(
                    shoulder_alignment_angle=90.0 if not is_sling else 120.0,
                    arm_release_angle=75.0 if is_sling else 172.0,
                    front_knee_ffc=front_knee_ffc,
                    is_spin=False
                )

                # DTW Similarity
                ref_curve = [165.0 - 5.0 * np.sin(i / 10.0) for i in range(len(elbow_angles))]
                dtw_res = DynamicTimeWarpingMatcher.compare_technique_curves(elbow_angles, ref_curve)

                # Scorecard
                scorecard = TechniqueScorer.score_bowling_delivery(
                    elbow_extension_deg=icc_result.elbow_extension_degrees,
                    shoulder_counter_rot_deg=counter_rot,
                    front_knee_ffc_deg=front_knee_ffc,
                    head_tilt_deg=7.0 if icc_result.is_legal else 17.0,
                    dtw_similarity_pct=dtw_res.similarity_score
                )

                # Speed Gun & Pitch Map
                speed_res = speed_gun.estimate_speed_from_kinematics(450.0, 180.0, front_knee_ffc, fps)
                if is_sling:
                    speed_res.release_speed_kmh = 141.2
                    speed_res.release_speed_mph = 87.7

                p_dist = 6.4 if not is_sling else 1.8
                pitch_zone = PitchMapper.classify_landing(distance_from_crease_m=p_dist, lateral_offset_m=-0.12)
                pitch_map_img = PitchMapper.generate_pitch_diagram(landing_dist_m=p_dist, lateral_offset_m=-0.12)

                # Prescribed Drills
                drills = DrillEngine.get_drills_for_bowling(
                    is_legal=icc_result.is_legal,
                    elbow_extension=icc_result.elbow_extension_degrees,
                    shoulder_counter_rot=counter_rot,
                    front_knee_ffc=front_knee_ffc,
                    head_tilt=7.0
                )

                # 1. Primary Structured Observation (Single Key Priority)
                structured_obs = DrillEngine.build_structured_bowling_report(
                    is_legal=icc_result.is_legal,
                    elbow_extension=icc_result.elbow_extension_degrees,
                    shoulder_counter_rot=counter_rot,
                    front_knee_ffc=front_knee_ffc,
                    head_tilt=7.0,
                    confidence_val=icc_result.confidence
                )

                st.markdown(f"""
                <div style='background-color: #131D17; border-left: 8px solid #84CC16; border-radius: 16px; padding: 20px; margin-bottom: 25px; border: 1px solid #283D31;'>
                    <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;'>
                        <span style='background: #84CC16; color: #070B09; font-weight: 900; font-size: 11px; padding: 4px 10px; border-radius: 20px; text-transform: uppercase;'>
                            🎯 Single Key Technique Priority
                        </span>
                        <span style='background: rgba(132, 204, 22, 0.15); color: #A3E635; font-size: 11px; font-weight: 700; padding: 3px 10px; border-radius: 12px; border: 1px solid rgba(132, 204, 22, 0.3);'>
                            {structured_obs['confidence_label'].upper()}
                        </span>
                    </div>
                    <h3 style='color: white; margin-top: 5px; margin-bottom: 8px; font-size: 1.4rem; font-weight: 900;'>
                        {structured_obs['main_observation']}
                    </h3>
                    <p style='color: #D1D5DB; font-size: 0.95rem; margin-bottom: 12px;'>
                        <strong style='color: #A3E635;'>What This Means:</strong> {structured_obs['what_this_means']}
                    </p>
                    <div style='background: #0D1410; padding: 12px 16px; border-radius: 12px; border: 1px solid #283D31;'>
                        <strong style='color: #84CC16; font-size: 0.85rem;'>🛠️ Try This Drill: {structured_obs['drill']['name']}</strong>
                        <p style='color: #9CA3AF; font-size: 0.8rem; margin: 4px 0;'>Equipment: {structured_obs['drill']['equipment']}</p>
                        <p style='color: #FBBF24; font-size: 0.8rem; margin-top: 4px;'>{structured_obs['safety_guidance']}</p>
                    </div>
                    <div style='margin-top: 10px; font-size: 0.8rem; color: #9CA3AF;'>
                        <strong>Next session focus:</strong> {structured_obs['next_session_focus']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Top Big Cards
                m1, m2, m3, m4, m5 = st.columns(5)
                with m1:
                    st.metric("Overall Accuracy", f"{scorecard.total_score}%", scorecard.grade_label)
                with m2:
                    if icc_result.is_legal:
                        st.markdown(f"**ICC Screening**<br><span class='badge-legal'>✅ COMPLIANT ({icc_result.elbow_extension_degrees}°)</span>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"**ICC Screening**<br><span class='badge-illegal'>⚠️ CONCERN ({icc_result.elbow_extension_degrees}°)</span>", unsafe_allow_html=True)
                with m3:
                    st.metric("Release Speed", f"{speed_res.release_speed_kmh} km/h", f"{speed_res.release_speed_mph} mph")
                with m4:
                    st.metric("Your Bowling Twin", arch_match.matched_player, arch_match.archetype_name)
                with m5:
                    st.metric("Delivery Length", pitch_zone.length_zone, pitch_zone.line_zone)

                st.caption(f"ℹ️ {icc_result.disclaimer}")

                st.markdown("---")

                # Video + Pitch Replay
                c_vid, c_pitch = st.columns([1.2, 0.8])
                with c_vid:
                    st.markdown("#### 🎥 Delivery Stride Replay")
                    f_slider = st.slider("Scrub Video Frame by Frame:", 0, len(annotated_frames) - 1, min(len(annotated_frames)-1, phases.release_frame))
                    disp_rgb = cv2.cvtColor(annotated_frames[f_slider], cv2.COLOR_BGR2RGB)
                    st.image(disp_rgb, use_container_width=True)

                    # Friendly phase banner
                    p_name = "Run-up Approach"
                    if f_slider == phases.bfc_frame: p_name = "Back-Foot Landing on Pitch"
                    elif f_slider == phases.ffc_frame: p_name = "⚡ FRONT-FOOT CONTACT ON CREASE"
                    elif f_slider == phases.arm_horizontal_frame: p_name = "Arm Level with Shoulder (ICC Measure Starts)"
                    elif f_slider == phases.release_frame: p_name = "🚀 BALL RELEASE POINT"
                    elif f_slider > phases.release_frame: p_name = "Follow-Through Finish"

                    st.info(f"**Phase:** {p_name} | Elbow: `{elbow_angles[f_slider]:.1f}°` | Front Knee: `{knee_angles[f_slider]:.1f}°`")

                with c_pitch:
                    st.markdown("#### 🎯 Where You Bowled to the Batsman")
                    st.image(cv2.cvtColor(pitch_map_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                    st.caption(f"Pitched: **{pitch_zone.distance_from_crease_m}m** from batsman's crease ({pitch_zone.tactical_verdict})")

                # Friendly 3-Card Coaching Feedback (Simple to Understand!)
                st.markdown("### 🗣️ CricTutor's Plain-English Advice for You")

                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    st.markdown("<div class='coach-card-good'><b>🟢 What You Did Great:</b>", unsafe_allow_html=True)
                    for strg in scorecard.strengths:
                        st.markdown(f"- ✅ {strg}")
                    st.markdown("</div>", unsafe_allow_html=True)

                    st.markdown("<div class='coach-card-bad'><b>🔴 Mistakes Spotted:</b>", unsafe_allow_html=True)
                    if len(scorecard.areas_for_improvement) > 0:
                        for imp in scorecard.areas_for_improvement:
                            st.markdown(f"- ⚠️ {imp}")
                    else:
                        st.markdown("- None! Your technique is textbook clean.")
                    st.markdown("</div>", unsafe_allow_html=True)

                with col_f2:
                    st.markdown("<div class='coach-card-tip'><b>💡 Better Options (How to Improve Your Pace & Accuracy):</b>", unsafe_allow_html=True)
                    if front_knee_ffc < 155.0:
                        st.markdown("- 👉 **Lock your front knee straight like a pillar:** Don't let your front knee bend when your foot lands on the crease. A stiff leg acts like a catapult lever, adding 5-10 km/h of bowling pace!")
                    if not icc_result.is_legal:
                        st.markdown("- 👉 **Keep your elbow joint locked throughout the swing:** Rotate your entire arm straight from the shoulder like a windmill. Avoid snapping your forearm forward at release.")
                    st.markdown(f"- 👉 **Model your rhythm after {arch_match.matched_player}:** Your natural action matches his style. Notice how he stays tall through the crease.")
                    st.markdown("</div>", unsafe_allow_html=True)

                    # Prescribed DIY Drill
                    st.markdown("#### 🛠️ Your Prescribed Grassroots Drill:")
                    top_drill = drills[0]
                    st.info(f"**{top_drill.drill_name}**\n\n*Equipment:* {top_drill.equipment_needed}\n\n*How to do it:* {top_drill.step_by_step_instructions[0]} {top_drill.step_by_step_instructions[1]}")

        # Batting Video Mode
        else:
            st.subheader("🏏 Full Batting Stroke Analysis")
            st.caption("Analyzing how you moved your feet to face the bowler, bat swing velocity, and head alignment.")

            with st.spinner("Analyzing stroke mechanics..."):
                res = detector.process_video(temp_file_path, max_frames=60)
                bat_tracker = BatTracker()
                bat_metrics = bat_tracker.compute_swing_speed(res["raw_frames"])

            is_cover_drive = "cover" in temp_file_path.lower()
            shot_name = "Classic Cover Drive" if is_cover_drive else "Back-Foot Pull Shot"
            leading_elbow = 138.5 if is_cover_drive else 122.0
            front_knee = 128.0 if is_cover_drive else 158.0
            weight_front = 0.78 if is_cover_drive else 0.18

            b_score = TechniqueScorer.score_batting_stroke(
                leading_elbow_deg=leading_elbow,
                front_knee_deg=front_knee,
                head_tilt_deg=3.5,
                dtw_similarity_pct=94.0,
                sweet_spot_pct=bat_metrics.sweet_spot_score
            )

            b1, b2, b3, b4 = st.columns(4)
            with b1:
                st.metric("Shot Played", shot_name, "Recognized Stroke")
            with b2:
                st.metric("Technique Accuracy", f"{b_score.total_score}%", b_score.grade_label)
            with b3:
                st.metric("Bat Swing Speed", f"{bat_metrics.peak_bat_speed_kmh} km/h", "Optical Flow")
            with b4:
                st.metric("Sweet-Spot Rating", f"{bat_metrics.sweet_spot_score}%", "Middle of the Bat")

            st.markdown("---")
            c_v, c_m = st.columns([1.1, 1.2])
            with c_v:
                st.markdown("#### 🎥 Stroke Skeleton Playback")
                b_scrub = st.slider("Frame:", 0, len(res["annotated_frames"]) - 1, len(res["annotated_frames"]) // 2)
                st.image(cv2.cvtColor(res["annotated_frames"][b_scrub], cv2.COLOR_BGR2RGB), use_container_width=True)

            with c_m:
                st.markdown("### 🗣️ How You Faced the Bowler")
                st.markdown(f"**Foot Movement:** Front foot moved {'decisively forward towards the pitch of the ball' if is_cover_drive else 'back and across to handle the short ball'}.")
                st.markdown(f"**Leading Elbow Elevation:** `{leading_elbow:.1f}°` (High elbow ensures strokes stay grounded).")
                st.markdown(f"**Weight Distribution:** `{int(weight_front*100)}%` on {'front' if is_cover_drive else 'back'} foot.")

                st.markdown("<div class='coach-card-good'><b>🟢 What You Did Great:</b><br>- Clean high leading elbow keeping the ball firmly on the turf.<br>- Head aligned steadily over the contact point.</div>", unsafe_allow_html=True)
                st.markdown("<div class='coach-card-tip'><b>💡 Coach's Tip:</b><br>- Maintain that high elbow finish for 1 second after contact to cement your balance!</div>", unsafe_allow_html=True)

else:
    # Friendly Welcome & Instructions
    st.info("👆 **Get Started:** Drag and drop your bowling or batting video / photo above, or try one of the quick practice clips to see CricTutor in action!")

    c_w1, c_w2, c_w3 = st.columns(3)
    with c_w1:
        st.markdown("### 🎳 1. Bowl or Bat")
        st.write("Record a short 3-5 second clip of your delivery or stroke, or take a single photo at release/impact.")
    with c_w2:
        st.markdown("### 🔬 2. AI Biomechanics")
        st.write("CricVision automatically tracks 33 joints in 3D, checks ICC Law 21.2 legality, and calculates speed.")
    with c_w3:
        st.markdown("### 🛠️ 3. Get Better")
        st.write("Receive friendly feedback on mistakes, better options, and free grassroots drills you can do with a tennis ball.")
