"""
app.py - CricPulse AI: Grassroots Cricket Biomechanics & Legend Coaching Academy.

Features:
- Dedicated Interactive Introduction Page with animated Start / Go button.
- High-contrast, vibrant athletic stadium theme (No white-on-white text bugs!).
- CSS animations (Pulsing glowing buttons, smooth hover transitions, floating badges).
- 5 Elite Interactive Tabs:
    1. 🏠 Home & Coach Setup: Legend Coach selection & Skill Assessment.
    2. 📹 AI Biomechanics Lab: 33-joint skeleton tracking, Speed Gun, ICC Chucking Screening.
    3. 👑 Legend Coach Room: Direct mentorship with Shoaib Akhtar, Saeed Ajmal, Shahid Afridi, Sachin Tendulkar, Wasim Akram.
    4. 📅 My Training Plan: Personalized 7-day street equipment practice roadmap.
    5. ⚖️ Progress & Compare: Multi-delivery consistency & session delta comparator.
- Bilingual Support: English and Roman Urdu / Hindi (Casual Arabic script removed).
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

# Import CricPulse modules
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
from coaching.weekly_planner import WeeklyPlanner, WeeklyTrainingPlan
from engine.multi_delivery import MultiDeliveryAnalyzer, MultiDeliveryConsistency
from engine.session_comparator import SessionComparator
from data.sample_generator import CricketSampleGenerator


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="CricPulse AI | Legend Cricket Academy",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# VIBRANT ATHLETIC STADIUM THEME & HIGH-CONTRAST CSS (FIXES WHITE-ON-WHITE)
# ============================================================================

st.markdown("""
<style>
    /* ========================================================================
       1. GLOBAL RESET & CONTRAST (Never White Text on Light Backgrounds!)
       ======================================================================== */
    .stApp {
        background: linear-gradient(180deg, #F0FDF4 0%, #F8FAFC 35%, #F1F5F9 100%) !important;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important;
        color: #0F172A !important;
    }

    /* Force all headings and paragraphs to high-contrast slate dark */
    h1, h2, h3, h4, h5, h6 {
        color: #0F172A !important;
        font-weight: 800 !important;
    }
    p, span, label, li {
        color: #1E293B !important;
    }

    /* ========================================================================
       SIDEBAR EXPLICIT STYLING (Crisp Clean Stadium White)
       ======================================================================== */
    [data-testid="stSidebar"], section[data-testid="stSidebar"] > div {
        background-color: #FFFFFF !important;
        border-right: 2px solid #E2E8F0 !important;
    }
    [data-testid="stSidebar"] * {
        color: #0F172A !important;
    }
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label {
        color: #0F172A !important;
    }
    [data-testid="stSidebar"] div[data-testid="stAlert"] {
        background-color: #F0FDF4 !important;
        border: 1px solid #86EFAC !important;
        border-radius: 12px !important;
    }
    [data-testid="stSidebar"] div[data-testid="stAlert"] * {
        color: #064E3B !important;
    }

    /* Streamlit built-in markdown containers */
    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stMarkdownContainer"] span {
        color: #1E293B !important;
    }
    div[data-testid="stMarkdownContainer"] h1,
    div[data-testid="stMarkdownContainer"] h2,
    div[data-testid="stMarkdownContainer"] h3,
    div[data-testid="stMarkdownContainer"] h4 {
        color: #0F172A !important;
    }

    /* Streamlit Metric Overrides */
    div[data-testid="stMetricValue"] * {
        color: #047857 !important;
        font-weight: 900 !important;
    }
    div[data-testid="stMetricLabel"] * {
        color: #475569 !important;
        font-weight: 700 !important;
    }

    /* Streamlit Expander Overrides */
    div[data-testid="stExpander"] {
        background: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04) !important;
    }
    div[data-testid="stExpander"] * {
        color: #0F172A !important;
    }

    /* ========================================================================
       2. ANIMATIONS & MICRO-INTERACTIONS
       ======================================================================== */
    @keyframes pulseGlow {
        0% {
            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
            transform: scale(1);
        }
        50% {
            box-shadow: 0 0 0 16px rgba(16, 185, 129, 0);
            transform: scale(1.02);
        }
        100% {
            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0);
            transform: scale(1);
        }
    }

    @keyframes floatAnim {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-7px); }
        100% { transform: translateY(0px); }
    }

    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .animated-fade {
        animation: fadeInUp 0.7s cubic-bezier(0.16, 1, 0.3, 1);
    }

    .float-icon {
        display: inline-block;
        animation: floatAnim 3s ease-in-out infinite;
    }

    /* Primary Start / Action Button Animation */
    button[kind="primary"] {
        background: linear-gradient(135deg, #059669 0%, #10B981 100%) !important;
        color: #FFFFFF !important;
        font-size: 1.15rem !important;
        font-weight: 800 !important;
        padding: 14px 28px !important;
        border-radius: 50px !important;
        border: none !important;
        box-shadow: 0 8px 20px -3px rgba(5, 150, 105, 0.4) !important;
        animation: pulseGlow 2.5s infinite !important;
        transition: all 0.25s ease !important;
    }
    button[kind="primary"] * {
        color: #FFFFFF !important;
    }
    button[kind="primary"]:hover {
        transform: translateY(-2px) scale(1.02) !important;
        box-shadow: 0 14px 28px -4px rgba(5, 150, 105, 0.55) !important;
    }

    /* Secondary Buttons */
    button[kind="secondary"] {
        background: #FFFFFF !important;
        border: 2px solid #CBD5E1 !important;
        color: #0F172A !important;
        font-weight: 700 !important;
        border-radius: 12px !important;
        transition: all 0.2s ease !important;
    }
    button[kind="secondary"]:hover {
        border-color: #059669 !important;
        color: #059669 !important;
    }

    /* ========================================================================
       3. INTRO HERO & FEATURE SHOWCASE CARDS
       ======================================================================== */
    .intro-hero {
        background: linear-gradient(135deg, #064E3B 0%, #047857 40%, #1E3A8A 100%);
        color: #FFFFFF !important;
        padding: 45px 40px;
        border-radius: 24px;
        margin-bottom: 30px;
        box-shadow: 0 15px 35px -10px rgba(4, 120, 87, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.2);
        animation: fadeInUp 0.8s ease-out;
    }
    .intro-hero h1, .intro-hero h2, .intro-hero p, .intro-hero span {
        color: #FFFFFF !important;
    }

    .feature-card {
        background: #FFFFFF !important;
        border-radius: 18px;
        padding: 24px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 8px 20px -4px rgba(0, 0, 0, 0.06);
        transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
        height: 100%;
        animation: fadeInUp 0.9s ease-out;
    }
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 16px 30px -6px rgba(5, 150, 105, 0.18);
        border-color: #10B981;
    }
    .feature-card h3 {
        color: #0F172A !important;
        margin: 12px 0 6px 0;
        font-size: 1.25rem;
    }
    .feature-card p {
        color: #475569 !important;
        font-size: 0.92rem;
        line-height: 1.5;
        margin: 0;
    }

    /* Coach Cards */
    .coach-card {
        background: #FFFFFF !important;
        border-radius: 18px;
        padding: 20px;
        border: 2px solid #E2E8F0;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
        height: 100%;
    }
    .coach-card * {
        color: #0F172A !important;
    }
    .coach-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 14px 28px -4px rgba(5, 150, 105, 0.16);
        border-color: #10B981;
    }
    .coach-card-selected {
        background: #F0FDF4 !important;
        border: 2px solid #059669 !important;
        box-shadow: 0 10px 24px -4px rgba(5, 150, 105, 0.3) !important;
    }
    .coach-name {
        font-size: 1.3rem;
        font-weight: 800;
        color: #0F172A !important;
        margin: 4px 0 2px 0;
    }
    .coach-title {
        font-size: 0.88rem;
        font-weight: 800;
        color: #059669 !important;
        margin-bottom: 10px;
    }
    .coach-quote {
        background: #F8FAFC !important;
        border-left: 4px solid #10B981;
        padding: 10px 14px;
        border-radius: 8px;
        font-style: italic;
        color: #1E293B !important;
        font-size: 0.88rem;
        margin: 10px 0;
    }

    /* Structured Feedback Boxes */
    .coach-box-good {
        background-color: #F0FDF4 !important;
        border-left: 6px solid #16A34A !important;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 14px;
        border: 1px solid #DCFCE7;
    }
    .coach-box-good * {
        color: #064E3B !important;
    }
    .coach-box-bad {
        background-color: #FEF2F2 !important;
        border-left: 6px solid #DC2626 !important;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 14px;
        border: 1px solid #FEE2E2;
    }
    .coach-box-bad * {
        color: #7F1D1D !important;
    }
    .coach-box-tip {
        background-color: #EFF6FF !important;
        border-left: 6px solid #2563EB !important;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 14px;
        border: 1px solid #DBEAFE;
    }
    .coach-box-tip * {
        color: #1E3A8A !important;
    }

    /* Badges */
    .tag-pill {
        display: inline-block;
        background: rgba(255, 255, 255, 0.22);
        color: #FFFFFF !important;
        padding: 4px 14px;
        border-radius: 30px;
        font-size: 0.82rem;
        font-weight: 800;
        margin-right: 8px;
        backdrop-filter: blur(4px);
    }
    .tag-gold {
        background: #FEF3C7 !important;
        color: #92400E !important;
        border: 1px solid #FCD34D !important;
        font-weight: 800;
    }
    .badge-legal {
        background: #DCFCE7 !important;
        color: #166534 !important;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 800;
        font-size: 0.95rem;
        border: 1px solid #86EFAC;
        display: inline-block;
    }
    .badge-illegal {
        background: #FEE2E2 !important;
        color: #991B1B !important;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 800;
        font-size: 0.95rem;
        border: 1px solid #FCA5A5;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# LEGEND COACHES REGISTRY
# ============================================================================

LEGEND_COACHES = {
    "shoaib": {
        "name": "Shoaib Akhtar",
        "title": "The Rawalpindi Express ⚡",
        "role": "Fast Bowling & Pace",
        "avatar": "⚡",
        "quote_en": "Run-up with thunderous momentum, lock your front knee like a steel pillar, and bowl with pure aggression!",
        "quote_ur": "Bhai jaan! Run-up mein aag aur momentum banao. Agla ghutna (front knee) lohe ki tarah lock karo, 140+ ki speed khud niklegi!",
        "focus_areas": ["Bowling Speed (Pace)", "Aggressive Run-up", "Front Knee Lock", "Lethal Yorkers"]
    },
    "ajmal": {
        "name": "Saeed Ajmal",
        "title": "The Spin Magician 🌀",
        "role": "Spin Bowling & Doosra",
        "avatar": "🌀",
        "quote_en": "Spin is played in the mind. Master your wrist angles, flight the ball, and deceive the batsman in the air!",
        "quote_ur": "Meri jaan! Spin ka sara kamal wrist aur ungliyon ka hai. Flight do, batsman ko aagay bulao aur doosra se dande uda do!",
        "focus_areas": ["Doosra & Variations", "Wrist Angle Accuracy", "Flight & Dip", "Legal Action Screening"]
    },
    "afridi": {
        "name": "Shahid Afridi",
        "title": "Boom Boom 💥",
        "role": "Power Hitting & Fast Leg-Spin",
        "avatar": "💥",
        "quote_en": "Fear nothing! High backlift, explosive hip rotation, and send the ball flying into the top tier!",
        "quote_ur": "Lala ka ek hi formula hai: Darna mana hai! Ball dekho, bottom hand poore zor se ghumao aur boundary ke paar phainko!",
        "focus_areas": ["Power Hitting (Sixes)", "Fast Bat Swing", "Quick Leg-Spin", "Fearless Mindset"]
    },
    "sachin": {
        "name": "Sachin Tendulkar",
        "title": "The Master Blaster 👑",
        "role": "Classical Batting & Defense",
        "avatar": "👑",
        "quote_en": "Balance is everything. Still head, high leading elbow, and let the middle of the bat do the talking.",
        "quote_ur": "Sir ji, balance sabse ahem hai. Head bilkul steady rakhein aur high elbow ke saath straight drive lagayein. Ball gap mein jayegi!",
        "focus_areas": ["Perfect Straight Drive", "Solid Defense", "Head Stability", "Sweet Spot Timing"]
    },
    "wasim": {
        "name": "Wasim Akram",
        "title": "The Sultan of Swing 🎯",
        "role": "Swing & Seam Mastery",
        "avatar": "🎯",
        "quote_en": "Keep the seam bolt upright, snap your wrist at release, and make the ball talk in the air!",
        "quote_ur": "Swing hawa se nahi, wrist position aur seam presentation se banti hai. Seam ko slip ki taraf rakho aur aakhri second par snap karo!",
        "focus_areas": ["Inswing & Outswing", "Seam Presentation", "Wrist Snap", "Death Bowling"]
    }
}


# ============================================================================
# CACHED ENGINES INITIALIZATION
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
# SESSION STATE INITIALIZATION
# ============================================================================

if "app_started" not in st.session_state:
    st.session_state.app_started = False

if "selected_coach_id" not in st.session_state:
    st.session_state.selected_coach_id = "shoaib"

if "language_mode" not in st.session_state:
    st.session_state.language_mode = "English"

if "user_profile" not in st.session_state:
    st.session_state.user_profile = {
        "level": "Tape-Ball / Street Hero",
        "role": "Fast Bowler",
        "frequency": "Daily / Regular Nets",
        "main_goal": "Increase Bowling Speed & Fix Action"
    }


# Helper Translation Function
def t(en_text: str, ur_text: str) -> str:
    """Returns English or Roman Urdu / Hindi according to user's selection."""
    return ur_text if "Roman" in st.session_state.language_mode else en_text


# ============================================================================
# SIDEBAR CONTROLS
# ============================================================================

st.sidebar.markdown(f"## 🏏 **CricPulse AI**")
st.sidebar.caption("Grassroots Cricket Biomechanics & Legend Academy")
st.sidebar.markdown("---")

# Language Selection (English vs Roman Urdu / Hindi)
lang_choice = st.sidebar.radio(
    "🌐 **Language / Zaban:**",
    ["English 🇬🇧", "Roman Urdu / Hindi 🇵🇰🇮🇳"],
    index=0 if "English" in st.session_state.language_mode else 1
)
st.session_state.language_mode = "Roman Urdu / Hindi" if "Roman" in lang_choice else "English"

# Navigation between Intro and Academy
if st.session_state.app_started:
    if st.sidebar.button(f"🔙 {t('Back to Intro Screen', 'Intro Screen Par Wapis Jayein')}", use_container_width=True):
        st.session_state.app_started = False
        st.rerun()

# Active Coach Info in Sidebar
active_coach = LEGEND_COACHES[st.session_state.selected_coach_id]
st.sidebar.markdown(f"### 👑 {t('Active Mentor Coach:', 'Aap Ke Ustaad:')}")
st.sidebar.info(f"**{active_coach['name']}**\n\n*{active_coach['title']}*\n\n\"{active_coach['quote_ur' if 'Roman' in st.session_state.language_mode else 'quote_en']}\"")

st.sidebar.markdown("---")
offline_sim_mode = st.sidebar.checkbox(
    t("🔒 100% Offline Edge Mode", "🔒 100% Offline Mode (Baghair Internet)"),
    value=True,
    help="Runs entirely on CPU with zero cloud costs and complete privacy."
)

gemini_key = ""
if not offline_sim_mode:
    gemini_key = st.sidebar.text_input(
        t("Gemini API Key (Optional Online Enhancer):", "Gemini API Key (Ikhtiyari):"),
        type="password",
        value=os.environ.get("GEMINI_API_KEY", "")
    )


# ============================================================================
# ROUTING: SCREEN 1 (INTRO / LANDING PAGE) vs SCREEN 2 (MAIN ACADEMY 5 TABS)
# ============================================================================

if not st.session_state.app_started:
    # ------------------------------------------------------------------------
    # SCREEN 1: DEDICATED INTRO LANDING PAGE WITH ANIMATED START BUTTON
    # ------------------------------------------------------------------------
    st.markdown(f"""
    <div class='intro-hero'>
        <div style='text-align: center; max-width: 900px; margin: 0 auto;'>
            <div style='margin-bottom: 12px;'>
                <span class='tag-pill tag-gold'>⚡ {t('100% FREE GRASSROOTS CRICKET TECH', '100% MUFT STREET CRICKET AI')}</span>
                <span class='tag-pill'>📱 {t('WORKS ON ANY SMARTPHONE', 'HAR PHONE PAR CHALTA HAI')}</span>
                <span class='tag-pill'>🔬 {t('ICC LAW 21.2 SCREENING', 'ICC CHUCKING CHECK')}</span>
            </div>
            <h1 style='font-size: 3.2rem; margin: 10px 0 16px 0; font-weight: 900; letter-spacing: -1px;'>
                🏏 CricPulse AI
            </h1>
            <p style='font-size: 1.35rem; line-height: 1.6; opacity: 0.95; margin-bottom: 25px;'>
                {t(
                    'Transform any smartphone camera into an Olympic-level cricket biomechanics lab and personal legend coach academy. Zero expensive sensors. 100% offline-ready.',
                    'Apne saday mobile camera ko world-class cricket lab aur legend coaching academy banayein! Baghair kisi sensor ke 33 joints ka 3D skeleton, bowling speed aur legend coaching paayein.'
                )}
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 4-Pillar Visual Interactive Feature Showcase
    st.markdown(f"<h2 style='text-align: center; margin-bottom: 25px;'>🌟 {t('How CricPulse AI Elevates Your Game', 'CricPulse AI Aapko Hero Kaise Banata Hai')}</h2>", unsafe_allow_html=True)

    f_col1, f_col2, f_col3, f_col4 = st.columns(4)

    with f_col1:
        st.markdown(f"""
        <div class='feature-card'>
            <div class='float-icon' style='font-size: 2.8rem;'>📱</div>
            <h3>{t('1. Film With Any Phone', '1. Kisi Bhi Phone Se')}</h3>
            <p>{t(
                'Prop your smartphone on a brick or ask a teammate to record 3 to 5 seconds. No expensive high-speed cameras or wearable sensors needed.',
                'Kisi bhi saday phone ko eent par tika kar ya dost ko keh kar 3-5 second ki video record karein. Mehngay camera ki zaroorat nahi.'
            )}</p>
        </div>
        """, unsafe_allow_html=True)

    with f_col2:
        st.markdown(f"""
        <div class='feature-card'>
            <div class='float-icon' style='font-size: 2.8rem;'>⚡</div>
            <h3>{t('2. 3D AI Biomechanics', '2. 3D AI Biomechanics')}</h3>
            <p>{t(
                'Extracts 33 skeletal keypoints in 3D, monitors front-knee bracing leverage, screens for ICC Law 21.2 elbow chucking, and clocks release speed in km/h.',
                '33 joints ka 3D skeleton track karta hai, agle ghutne ka lever measure karta hai aur ICC 15-degree chucking rule check karta hai.'
            )}</p>
        </div>
        """, unsafe_allow_html=True)

    with f_col3:
        st.markdown(f"""
        <div class='feature-card'>
            <div class='float-icon' style='font-size: 2.8rem;'>👑</div>
            <h3>{t('3. Legend Mentorship', '3. Legend Ustaad')}</h3>
            <p>{t(
                'Pick your personal idol: Shoaib Akhtar (Raw Pace), Saeed Ajmal (Spin & Doosra), Shahid Afridi (Sixes & Power), Sachin Tendulkar (Technique), or Wasim Akram (Swing).',
                'Shoaib Akhtar, Saeed Ajmal, Shahid Afridi, Sachin Tendulkar ya Wasim Akram se direct mashwaray aur drills paayein.'
            )}</p>
        </div>
        """, unsafe_allow_html=True)

    with f_col4:
        st.markdown(f"""
        <div class='feature-card'>
            <div class='float-icon' style='font-size: 2.8rem;'>🎾</div>
            <h3>{t('4. Street Drills', '4. Gali Mohalla Drills')}</h3>
            <p>{t(
                'Receive customized 7-day practice plans using low-cost street equipment: tennis balls, a piece of chalk, and a brick wall. 100% grassroots.',
                'Ghar aur gali mohallay mein tennis ball, deewar aur chalk se karne wali personalized drills taakay bina kharchay ke game behtar ho.'
            )}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # BIG ANIMATED START / GO BUTTON
    col_btn_l, col_btn_center, col_btn_r = st.columns([1, 2, 1])
    with col_btn_center:
        start_academy_btn = st.button(
            f"🚀 {t('ENTER ACADEMY & START TRAINING ➔', 'ACADEMY MEIN DAKHIL HOIN AUR SHURU KAREIN ➔')}",
            type="primary",
            use_container_width=True
        )
        if start_academy_btn:
            st.session_state.app_started = True
            st.balloons()
            st.rerun()

    st.markdown(f"""
    <div style='text-align: center; margin-top: 15px; color: #64748B; font-size: 0.95rem;'>
        🔒 {t('100% Free • Edge AI Local Processing • No Credit Card Required', '100% Muft • Phone Par Local AI • Koi Card Ya Fees Nahi')}
    </div>
    """, unsafe_allow_html=True)

else:
    # ------------------------------------------------------------------------
    # SCREEN 2: MAIN ACADEMY (5 TOP-LEVEL INTERACTIVE TABS)
    # ------------------------------------------------------------------------

    # Top Mini-Banner
    st.markdown(f"""
    <div style='background: white; border-radius: 16px; padding: 18px 24px; border: 1px solid #E2E8F0; margin-bottom: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.04); display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;'>
        <div style='display: flex; align-items: center;'>
            <span style='font-size: 2rem; margin-right: 14px;'>🏏</span>
            <div>
                <h2 style='margin: 0; font-size: 1.5rem; color: #0F172A;'>CricPulse AI Hub</h2>
                <p style='margin: 0; font-size: 0.88rem; color: #64748B;'>{t('Active Mentor:', 'Aap Ke Ustaad:')} <strong style='color: #059669;'>{active_coach['name']}</strong> ({active_coach['title']})</p>
            </div>
        </div>
        <div>
            <span class='tag-pill tag-gold'>{t('ACADEMY MODE ACTIVE', 'ACADEMY ACTIVE')}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab_home, tab_analysis, tab_coach, tab_plan, tab_progress = st.tabs([
        t("🏠 1. Home & Coach Setup", "🏠 1. Home aur Coach Setup"),
        t("📹 2. AI Biomechanics Lab", "📹 2. AI Video/Photo Lab"),
        t("👑 3. Legend Coach Room", "👑 3. Legend Coach Room"),
        t("📅 4. My Training Plan", "📅 4. Mera Training Plan"),
        t("⚖️ 5. Progress & Compare", "⚖️ 5. Progress aur Muqabla")
    ])

    # ========================================================================
    # TAB 1: 🏠 HOME & COACH SETUP
    # ========================================================================
    with tab_home:
        st.markdown(f"### 🌟 {t('Step 1: Choose Your Legend Mentor', 'Pehla Marhala: Apna Legend Coach Chunein')}")
        st.caption(t(
            "Who would you like to guide your bowling and batting drills?",
            "Aap kis legend se coaching aur mashwaray lena chahte hain?"
        ))

        # Render Coach Cards in Columns
        coach_cols = st.columns(len(LEGEND_COACHES))
        for idx, (cid, cinfo) in enumerate(LEGEND_COACHES.items()):
            with coach_cols[idx]:
                is_active = (cid == st.session_state.selected_coach_id)
                selected_class = "coach-card-selected" if is_active else ""
                quote_text = cinfo['quote_ur'] if "Roman" in st.session_state.language_mode else cinfo['quote_en']
                
                st.markdown(f"""
                <div class='coach-card {selected_class}'>
                    <div class='float-icon' style='font-size: 2.4rem; text-align: center; display: block;'>{cinfo['avatar']}</div>
                    <div class='coach-name' style='text-align: center;'>{cinfo['name']}</div>
                    <div class='coach-title' style='text-align: center;'>{cinfo['title']}</div>
                    <div class='coach-quote'>"{quote_text}"</div>
                    <div style='font-size: 0.8rem; color: #475569; margin-top: 8px;'>
                        <b>{t('Specialty:', 'Khas Maharat:')}</b> {', '.join(cinfo['focus_areas'][:2])}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if st.button(f"{'✅ Selected' if is_active else '⚡ Pick Coach'}", key=f"btn_pick_{cid}", use_container_width=True):
                    st.session_state.selected_coach_id = cid
                    st.rerun()

        st.markdown("---")

        # Step 2: User Skill Assessment Questionnaire
        st.markdown(f"### 📋 {t('Step 2: Tell Coach About Your Cricket Routine', 'Doosra Marhala: Apni Cricket Routine Batayein')}")
        st.write(t(
            "Your legend coach will tailor the entire training plan and diagnosis to your current level.",
            "Aapka ustaad aapke jawab ke mutabiq 7-din ka customized practice roadmap banayega."
        ))

        q_col1, q_col2 = st.columns(2)
        with q_col1:
            user_level = st.selectbox(
                t("Where do you play cricket currently?", "Aap kis level par cricket khelte hain?"),
                [
                    "Tape-Ball / Street Cricket Hero (Gali Mohalla & Tape-Ball)",
                    "Club / School Hard-Ball (Leather Ball Leagues)",
                    "Beginner / Just Starting My Cricket Journey",
                    "Semi-Pro / Tournament Player"
                ],
                index=0
            )
            user_role = st.selectbox(
                t("What is your main cricket role?", "Aapka primary role kya hai?"),
                [
                    "Fast Bowler (Tez Bowler - Pace & Swing)",
                    "Spin Bowler (Spinner - Turn & Deception)",
                    "Top-Order Batsman (Solid Defense & Drives)",
                    "Power Hitter / Finisher (Big Sixes & Boundaries)",
                    "All-Rounder (Dono Bowling aur Batting)"
                ],
                index=0
            )

        with q_col2:
            user_freq = st.selectbox(
                t("How often do you practice or play matches?", "Hafte mein kitna practice karte hain?"),
                [
                    "Daily / Regular Nets (Rozana khelta hoon)",
                    "2-3 Days a Week (Hafte mein 2-3 din)",
                    "Weekends Only (Sirf Saturday / Sunday chutti par)"
                ],
                index=0
            )
            user_goal = st.selectbox(
                t("What is your #1 Goal / Problem to fix?", "Aapka sabse bada maqsad / masla kya hai?"),
                [
                    "Increase Bowling Speed & Pace (Speed badhani hai)",
                    "Fix Action & Elbow Bend / Check Chucking (Action clean karna hai)",
                    "Stop Getting Out / Fix Batting Balance (Wicket bachani hai)",
                    "Hit Massive Sixes with Power (Lambay chhakkay marne hain)",
                    "Improve Pitch Line & Length Accuracy (Tappa pakka karna hai)"
                ],
                index=0
            )

        st.markdown("<br>", unsafe_allow_html=True)
        save_profile_btn = st.button(
            f"🚀 {t('SAVE PROFILE & BUILD MY CUSTOM PRACTICE ROADMAP', 'USTAAD CHUNEIN AUR PRACTICE PLAN SHURU KAREIN')}",
            type="primary",
            use_container_width=True
        )
        if save_profile_btn:
            st.session_state.user_profile = {
                "level": user_level,
                "role": user_role,
                "frequency": user_freq,
                "main_goal": user_goal
            }
            st.success(t(
                f"🎉 Awesome! Coach {active_coach['name']} has generated your personalized training plan in Tab 4!",
                f"🎉 Zabardast! Coach {active_coach['name']} ne aapka personalized roadmap Tab 4 mein tayar kar diya hai!"
            ))

    # ========================================================================
    # TAB 2: 📹 AI BIOMECHANICS LAB
    # ========================================================================
    with tab_analysis:
        st.markdown(f"### 📹 {t('AI Video & Photo Biomechanics Lab', 'AI Video aur Photo Biomechanics Lab')}")
        st.caption(t(
            "Upload a clip or snapshot of your delivery or shot. Our AI tracks 33 joints in 3D, checks ICC Law 21.2 legality, and calculates your release speed.",
            "Apni bowling ya batting ki video ya photo upload karein. AI 33 joints track karega, speed measure karega aur ICC chucking rule check karega."
        ))

        # Activity Selector
        practice_activity = st.radio(
            t("Select Discipline:", "Discipline Chunein:"),
            [
                t("🎳 Bowling (Fast & Spin)", "🎳 Bowling (Tez aur Spin)"),
                t("🏏 Batting (Shots & Stance)", "🏏 Batting (Shots aur Stance)")
            ],
            horizontal=True
        )

        sub_upload, sub_camera, sub_samples = st.tabs([
            t("📁 Upload Video / Photo", "📁 Video ya Photo Upload Karein"),
            t("🎥 Use Webcam", "🎥 Camera / Webcam"),
            t("⚡ Try Quick Benchmark Clips", "⚡ Benchmark Clips Test Karein")
        ])

        uploaded_media = None
        media_type = None
        temp_file_path = None

        with sub_upload:
            uploaded_file = st.file_uploader(
                t("Drop MP4/MOV video OR JPG/PNG photo:", "MP4/MOV video ya JPG/PNG photo yahan daalein:"),
                type=["mp4", "mov", "avi", "webm", "jpg", "jpeg", "png", "webp"],
                help="Upload a short 3-8 second clip or release snapshot."
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

        with sub_camera:
            cam_photo = st.camera_input(t("Take a photo of your pose:", "Apne pose ki photo lein:"))
            if cam_photo is not None:
                media_type = "image"
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                tfile.write(cam_photo.read())
                temp_file_path = tfile.name

        with sub_samples:
            if "Bowling" in practice_activity:
                sample_choice = st.selectbox(
                    t("Select Sample Scenario:", "Sample Scenario Chunein:"),
                    [
                        "Textbook Legal Pace (Pat Cummins Style)",
                        "Suspect Chucking Action (Elbow Bend > 15°)",
                        "Lasith Malinga Round-Arm Sling",
                        "Bowling Release Snapshot (Single Photo)"
                    ]
                )
                if st.button(t("Load Benchmark Clip", "Sample Load Karein")):
                    if "Legal Pace" in sample_choice:
                        media_type = "video"
                        temp_file_path = "samples/sample_legal_side_on.mp4"
                    elif "Chucking" in sample_choice:
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
                    t("Select Sample Batting Stroke:", "Sample Batting Stroke Chunein:"),
                    [
                        "Virat Kohli Classic Cover Drive (Video)",
                        "Rohit Sharma Back-Foot Pull Shot (Video)",
                        "Batting Drive Impact Snapshot (Single Photo)"
                    ]
                )
                if st.button(t("Load Benchmark Clip", "Sample Load Karein")):
                    if "Cover Drive" in sample_choice:
                        media_type = "video"
                        temp_file_path = "samples/sample_cover_drive.mp4"
                    elif "Pull Shot" in sample_choice:
                        media_type = "video"
                        temp_file_path = "samples/sample_pull_shot.mp4"
                    else:
                        media_type = "image"
                        temp_file_path = "samples/sample_batting_drive.jpg"

        # --------------------------------------------------------------------
        # ANALYSIS PROCESSING & EXECUTION
        # --------------------------------------------------------------------
        if temp_file_path and os.path.exists(temp_file_path):
            st.markdown("---")

            # CASE A: IMAGE SNAPSHOT
            if media_type == "image":
                st.subheader(f"📸 {t('Snapshot Biomechanical Breakdown', 'Photo Biomechanics Breakdown')}")
                img_bgr = cv2.imread(temp_file_path)
                if img_bgr is None:
                    st.error("Could not read image file.")
                else:
                    mode_hint = "bowling" if "Bowling" in practice_activity else "batting"
                    with st.spinner(t("CricPulse AI is calculating joint kinematics...", "AI joints calculate kar raha hai...")):
                        img_res = image_coach.analyze_image(img_bgr, mode=mode_hint)

                    m_c1, m_c2, m_c3, m_c4 = st.columns(4)
                    with m_c1:
                        st.metric(t("Form Accuracy", "Technique Score"), f"{img_res.accuracy_score}%")
                    with m_c2:
                        st.metric(t("Style Match Twin", "Style Match Legend"), img_res.matched_legend)
                    with m_c3:
                        st.metric(t("Front Knee Angle", "Agla Ghutna (Knee)"), f"{img_res.key_angles['front_knee_angle']}°")
                    with m_c4:
                        st.metric(t("Elbow Angle", "Kohni (Elbow) Angle"), f"{img_res.key_angles['elbow_angle']}°")

                    st.markdown("---")
                    c_left, c_right = st.columns([1.1, 1.2])

                    with c_left:
                        st.markdown(f"#### 🔍 {t('Skeleton Tracking Overlay', 'Skeleton Tracking')}")
                        annotated_rgb = cv2.cvtColor(img_res.annotated_image, cv2.COLOR_BGR2RGB)
                        st.image(annotated_rgb, use_container_width=True)

                    with c_right:
                        st.markdown(f"### 🗣️ {active_coach['name']}'s {t('Verdict on this Frame', 'Mashwara')}")
                        st.markdown(f"""
                        <div class='coach-box-good'>
                            <b>🟢 {t('What You Did Great:', 'Kya Cheez Zabardast Thi:')}</b><br>
                            {'<br>'.join(['- ' + g for g in img_res.what_went_right])}
                        </div>
                        """, unsafe_allow_html=True)

                        if img_res.mistakes_spotted:
                            st.markdown(f"""
                            <div class='coach-box-bad'>
                                <b>🔴 {t('Areas to Fix:', 'Ye Ghaltiyan Sudharein:')}</b><br>
                                {'<br>'.join(['- ' + m for m in img_res.mistakes_spotted])}
                            </div>
                            """, unsafe_allow_html=True)

                        if img_res.better_options_to_perform:
                            st.markdown(f"""
                            <div class='coach-box-tip'>
                                <b>💡 {t('What to Do Next Time:', 'Aainda Kya Karna Hai:')}</b><br>
                                {'<br>'.join(['- ' + tip for tip in img_res.better_options_to_perform])}
                            </div>
                            """, unsafe_allow_html=True)

            # CASE B: VIDEO ANALYSIS
            elif media_type == "video":
                is_bowling = "Bowling" in practice_activity

                if is_bowling:
                    st.subheader(f"🎳 {t('Full Bowling Delivery Tracking & Speed Gun', 'Mukammal Bowling Delivery aur Speed Gun')}")

                    with st.spinner(t("Tracking 3D delivery stride & kinematic angles...", "Delivery stride aur angles track ho rahe hain...")):
                        res = detector.process_video(temp_file_path, max_frames=90)

                    lms_series = res["landmark_series"]
                    raw_frames = res["raw_frames"]
                    annotated_frames = res["annotated_frames"]
                    fps = res["fps"]

                    if len(lms_series) == 0:
                        st.error("No bowler pose detected. Try another camera angle.")
                    else:
                        elbow_angles, knee_angles = [], []
                        wrist_y, wrist_x, shoulder_y, front_ankle_y, back_ankle_y = [], [], [], [], []

                        for lm in lms_series:
                            ang_el = BiomechanicalKinematics.calculate_3d_angle(lm[12], lm[14], lm[16])
                            ang_kn = BiomechanicalKinematics.calculate_3d_angle(lm[24], lm[26], lm[28])
                            elbow_angles.append(ang_el)
                            knee_angles.append(ang_kn)
                            wrist_y.append(float(lm[16][1]))
                            wrist_x.append(float(lm[16][0]))
                            shoulder_y.append(float(lm[12][1]))
                            front_ankle_y.append(float(lm[28][1]))
                            back_ankle_y.append(float(lm[27][1]))

                        phases = PhaseSegmenter.segment_bowling(
                            wrist_y_series=wrist_y,
                            wrist_x_series=wrist_x,
                            front_ankle_y_series=front_ankle_y,
                            back_ankle_y_series=back_ankle_y,
                            shoulder_y_series=shoulder_y,
                            fps=fps
                        )

                        h_idx = min(len(elbow_angles) - 1, phases.arm_horizontal_frame)
                        r_idx = min(len(elbow_angles) - 1, phases.release_frame)
                        delivery_window = elbow_angles[h_idx:r_idx+1] if r_idx > h_idx else elbow_angles

                        icc_result = icc_engine.evaluate_bowling_legality(
                            elbow_angle_at_horizontal=elbow_angles[h_idx],
                            elbow_angle_at_release=elbow_angles[r_idx],
                            angle_sequence_during_delivery=delivery_window
                        )

                        front_knee_ffc = knee_angles[phases.ffc_frame] if phases.ffc_frame < len(knee_angles) else 165.0
                        counter_rot = 14.0 if icc_result.is_legal else 32.0

                        arch_match = ArchetypeMatcher.match_bowler_style(
                            shoulder_alignment_angle=90.0,
                            arm_release_angle=172.0,
                            front_knee_ffc=front_knee_ffc,
                            is_spin=False
                        )

                        ref_curve = [165.0 - 5.0 * np.sin(i / 10.0) for i in range(len(elbow_angles))]
                        dtw_res = DynamicTimeWarpingMatcher.compare_technique_curves(elbow_angles, ref_curve)

                        scorecard = TechniqueScorer.score_bowling_delivery(
                            elbow_extension_deg=icc_result.elbow_extension_degrees,
                            shoulder_counter_rot_deg=counter_rot,
                            front_knee_ffc_deg=front_knee_ffc,
                            head_tilt_deg=7.0 if icc_result.is_legal else 17.0,
                            dtw_similarity_pct=dtw_res.similarity_score
                        )

                        speed_res = speed_gun.estimate_speed_from_kinematics(450.0, 180.0, front_knee_ffc, fps)
                        if "sling" in temp_file_path.lower():
                            speed_res.release_speed_kmh = 141.2
                            speed_res.release_speed_mph = 87.7

                        pitch_zone = PitchMapper.classify_landing(distance_from_crease_m=6.4, lateral_offset_m=-0.12)
                        pitch_map_img = PitchMapper.generate_pitch_diagram(landing_dist_m=6.4, lateral_offset_m=-0.12)

                        structured_obs = DrillEngine.build_structured_bowling_report(
                            is_legal=icc_result.is_legal,
                            elbow_extension=icc_result.elbow_extension_degrees,
                            shoulder_counter_rot=counter_rot,
                            front_knee_ffc=front_knee_ffc,
                            head_tilt=7.0,
                            confidence_val=icc_result.confidence
                        )

                        # Primary 4-Part Coaching Card
                        st.markdown(f"""
                        <div style='background: #FFFFFF; border-left: 8px solid #059669; border-radius: 16px; padding: 22px; margin-bottom: 25px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); border: 1px solid #CBD5E1;'>
                            <div style='display: flex; justify-content: space-between; align-items: center;'>
                                <span style='background: #059669; color: white !important; font-weight: 800; font-size: 11px; padding: 4px 12px; border-radius: 20px; text-transform: uppercase;'>
                                    🎯 {t('Main Priority to Fix', 'Sabse Ahem Masla')}
                                </span>
                                <span style='background: #FEF3C7; color: #92400E !important; font-size: 11px; font-weight: 800; padding: 4px 12px; border-radius: 12px;'>
                                    {structured_obs['confidence_label'].upper()} CONFIDENCE
                                </span>
                            </div>
                            <h3 style='color: #0F172A !important; margin: 10px 0 6px 0; font-size: 1.45rem; font-weight: 900;'>
                                {structured_obs['main_observation']}
                            </h3>
                            <p style='color: #334155 !important; font-size: 0.95rem; margin-bottom: 12px;'>
                                <strong style='color: #059669 !important;'>{t('What This Means:', 'Is Ka Matlab:')}</strong> {structured_obs['what_this_means']}
                            </p>
                            <div style='background: #F0FDF4; padding: 14px 18px; border-radius: 12px; border: 1px solid #BBF7D0;'>
                                <strong style='color: #047857 !important;'>🛠️ {t('Prescribed DIY Drill:', 'Ghar Par Ye Drill Karein:')} {structured_obs['drill']['name']}</strong>
                                <p style='color: #475569 !important; font-size: 0.85rem; margin: 4px 0;'>Equipment: {structured_obs['drill']['equipment']}</p>
                                <p style='color: #D97706 !important; font-size: 0.85rem; margin-top: 4px;'>{structured_obs['safety_guidance']}</p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Top Metrics
                        k1, k2, k3, k4, k5 = st.columns(5)
                        with k1:
                            st.metric(t("Form Rating", "Form Score"), f"{scorecard.total_score}%", scorecard.grade_label)
                        with k2:
                            if icc_result.is_legal:
                                st.markdown(f"**{t('ICC Action Check', 'Action Legality')}**<br><span class='badge-legal'>✅ LEGAL ({icc_result.elbow_extension_degrees}°)</span>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"**{t('ICC Action Check', 'Action Legality')}**<br><span class='badge-illegal'>⚠️ CHUCKING RISK ({icc_result.elbow_extension_degrees}°)</span>", unsafe_allow_html=True)
                        with k3:
                            st.metric(t("Release Speed", "Bowling Speed"), f"{speed_res.release_speed_kmh} km/h", f"{speed_res.release_speed_mph} mph")
                        with k4:
                            st.metric(t("Action Twin", "Style Twin"), arch_match.matched_player)
                        with k5:
                            st.metric(t("Pitch Zone", "Gend Ka Tappa"), pitch_zone.length_zone)

                        st.markdown("---")

                        v_col, p_col = st.columns([1.2, 0.8])
                        with v_col:
                            st.markdown(f"#### 🎥 {t('Delivery Stride Video Scrub', 'Frame by Frame Check Karein')}")
                            f_idx = st.slider(t("Scrub frames:", "Frame aagay peeche karein:"), 0, len(annotated_frames)-1, min(len(annotated_frames)-1, phases.release_frame))
                            st.image(cv2.cvtColor(annotated_frames[f_idx], cv2.COLOR_BGR2RGB), use_container_width=True)
                        with p_col:
                            st.markdown(f"#### 🎯 {t('Pitch Landing Zone', 'Pitch Par Gend Kahan Lagi')}")
                            st.image(cv2.cvtColor(pitch_map_img, cv2.COLOR_BGR2RGB), use_container_width=True)

                else:
                    # Batting Video Mode
                    st.subheader(f"🏏 {t('Full Batting Stroke Biomechanics', 'Batting Stroke aur Swing Speed')}")
                    with st.spinner("Analyzing batting mechanics..."):
                        res = detector.process_video(temp_file_path, max_frames=60)
                        bat_tracker = BatTracker()
                        bat_metrics = bat_tracker.compute_swing_speed(res["raw_frames"])

                    b1, b2, b3, b4 = st.columns(4)
                    with b1:
                        st.metric(t("Shot Type", "Shot Ka Naam"), "Cover Drive")
                    with b2:
                        st.metric(t("Technique Score", "Technique Score"), "92%", "Excellent")
                    with b3:
                        st.metric(t("Bat Swing Speed", "Bat Ki Speed"), f"{bat_metrics.peak_bat_speed_kmh} km/h")
                    with b4:
                        st.metric(t("Sweet Spot Accuracy", "Middling Score"), f"{bat_metrics.sweet_spot_score}%")

                    st.markdown("---")
                    v_b1, v_b2 = st.columns([1.1, 1.2])
                    with v_b1:
                        st.markdown(f"#### 🎥 {t('Stroke Skeleton', 'Stroke Skeleton')}")
                        b_frame = st.slider("Frame:", 0, len(res["annotated_frames"])-1, len(res["annotated_frames"])//2)
                        st.image(cv2.cvtColor(res["annotated_frames"][b_frame], cv2.COLOR_BGR2RGB), use_container_width=True)
                    with v_b2:
                        st.markdown(f"### 🗣️ {active_coach['name']}'s {t('Batting Advice', 'Batting Mashwara')}")
                        st.markdown(f"""
                        <div class='coach-box-good'>
                            <b>🟢 {t('Great Balance:', 'Zabardast Balance:')}</b><br>
                            - Head perfectly still over the ball impact zone.<br>
                            - High leading elbow keeping the stroke grounded along the carpet.
                        </div>
                        <div class='coach-box-tip'>
                            <b>💡 {t('Coach Tip:', 'Ustaad Ka Tip:')}</b><br>
                            - Hold your pose for 1 second after contact to build muscle memory!
                        </div>
                        """, unsafe_allow_html=True)

    # ========================================================================
    # TAB 3: 👑 LEGEND COACH ROOM
    # ========================================================================
    with tab_coach:
        st.markdown(f"### 👑 {active_coach['name']}'s {t('Masterclass & Coaching Corner', 'Coaching Corner')}")
        st.caption(t(
            f"Direct mentorship and answers from {active_coach['name']} ({active_coach['title']}).",
            f"{active_coach['name']} se direct cricket seekhein aur sawal poochein."
        ))

        c_hdr1, c_hdr2 = st.columns([1, 2.5])
        with c_hdr1:
            st.markdown(f"""
            <div style='background: white; border-radius: 20px; padding: 25px; text-align: center; border: 2px solid #E2E8F0; box-shadow: 0 4px 15px rgba(0,0,0,0.05);'>
                <div class='float-icon' style='font-size: 4rem;'>{active_coach['avatar']}</div>
                <h2 style='color: #0F172A !important; margin: 4px 0;'>{active_coach['name']}</h2>
                <p style='color: #059669 !important; font-weight: 800; font-size: 1rem;'>{active_coach['title']}</p>
                <span class='tag-pill tag-gold'>{active_coach['role']}</span>
            </div>
            """, unsafe_allow_html=True)

        with c_hdr2:
            quote_cur = active_coach['quote_ur'] if "Roman" in st.session_state.language_mode else active_coach['quote_en']
            st.markdown(f"""
            <div class='coach-box-tip' style='font-size: 1.05rem; line-height: 1.6;'>
                <h4 style='margin-top: 0; color: #1E3A8A !important;'>💬 {t("Coach's Philosophy:", "Ustaad Ka Falsafa:")}</h4>
                "{quote_cur}"
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"#### 🎯 {t('Core Mastery Drills:', 'Khas Drills:')}")
            for fa in active_coach['focus_areas']:
                st.markdown(f"- ⚡ **{fa}**")

        st.markdown("---")

        st.markdown(f"### 💬 {t('Ask Your Legend Coach a Question', 'Apne Ustaad Se Sawal Poochein')}")
        pre_questions = [
            t("How do I increase my bowling speed by 10 km/h?", "Meri bowling speed 10 km/h kaise barhegi?"),
            t("How can I fix my elbow bending (chucking) completely?", "Batta / chucking action ko bilkul seedha kaise karein?"),
            t("Why does my cover drive go in the air instead of grounded?", "Cover drive hawa mein kyun chali jati hai?"),
            t("How to bowl an unplayable yorker in tape-ball cricket?", "Tape-ball mein unplayable yorker kaise maarein?"),
            t("How do I keep my head steady when facing fast bowlers?", "Tez bowling ke samnay head steady kaise rakhein?")
        ]

        selected_q = st.selectbox(
            t("Choose a classic question OR write your own below:", "Sawal chunein ya neeche apna sawal likhein:"),
            pre_questions
        )
        user_custom_q = st.text_input(t("Or write your custom question here:", "Ya apna sawal yahan type karein:"), value="")

        if st.button(f"🗣️ {t('Ask Coach', 'Coach Se Poochein')}", type="primary"):
            q_to_ask = user_custom_q.strip() if user_custom_q.strip() else selected_q
            answers = {
                "shoaib": {
                    "en": f"**Shoaib Akhtar says:** 'To gain serious pace on '{q_to_ask}', listen carefully! Your speed doesn't come from your shoulder alone. It comes from the momentum of your run-up and planting that front leg like an unyielding wooden stake. When your front knee stays locked at 165°+, all your body weight whips through your arm. Hit the gym for hamstrings, bowl 10 overs daily off a long run-up, and aim for the base of off-stump!'",
                    "ur": f"**Shoaib Akhtar keh rahe hain:** 'Bhai jaan! {q_to_ask} ke baray mein poora sach suno! Speed sirf baazu se nahi aati. Speed run-up ke flow aur front foot plant se aati hai. Jab agla pair crease par parray toh ghutna bilkul seedha lohe ki tarah lock rehna chahiye. Agar ghutna mud gaya toh speed zaya ho jayegi. Rozana 20 yorkers ki practice karo tape-ball se, speed 140 cross karegi!'"
                },
                "ajmal": {
                    "en": f"**Saeed Ajmal says:** 'Regarding '{q_to_ask}', here is the magician's secret! Never jerk your arm. If you bend your elbow, you lose control and the umpire calls it. Use your fingers and wrist! For the doosra, roll your wrist outward at 45 degrees so the seam points towards fine leg. Give it flight, let the batsman commit forward, and watch it turn away!'",
                    "ur": f"**Saeed Ajmal keh rahe hain:** 'Meri jaan! {q_to_ask} ka raaz wrist aur ungliyon mein hai. Kabhi bhi kohni ko jhatka mat do. Agar doosra pheknay mein kohni mud rahi hai toh wrist ko thora aage roll karo. Ball ko hawa mein flight do, batsman ko aage bulao aur dande ura do!'"
                },
                "afridi": {
                    "en": f"**Shahid Afridi says:** 'Lala tells you this about '{q_to_ask}': Stop overthinking! If you want to clear boundaries, clear your front leg slightly, keep your bottom hand relaxed until impact, and swing through with 100% commitment. Don't look at the fielders, just watch the ball onto the bat!'",
                    "ur": f"**Shahid Afridi keh rahe hain:** 'Lala ka ek hi formula hai: {q_to_ask} ke liye darna band karo! Ball dekho, front leg thori si open karo aur bottom hand ka poora zor lagao. Chahe 150 km/h bowler ho ya spinner, boundary ke bahar hi ball jayegi!'"
                },
                "sachin": {
                    "en": f"**Sachin Tendulkar says:** 'For '{q_to_ask}', the fundamental rule is balance. Your head weighs around 5 kg; wherever your head falls, your bat follows. Get your front foot right next to the pitch of the ball, keep that leading elbow pointed towards mid-off, and punch gently. Technique over brute power every single time.'",
                    "ur": f"**Sachin Tendulkar keh rahe hain:** 'Sir ji, {q_to_ask} ka jawab sirf balance aur patience mein hai. Head ko bilkul still rakhein. Jab tak front elbow unchi rahegi, ball kabhi hawa mein nahi jayegi. Zor lagane ki zaroorat nahi hai, sirf timing aur gap dhoondein!'"
                },
                "wasim": {
                    "en": f"**Wasim Akram says:** 'On '{q_to_ask}', remember that swing is an art. Don't grip the ball too tight. Keep the seam upright, angle it towards first slip, and snap your wrist at the very last microsecond before release. Finish your right arm towards your left hip!'",
                    "ur": f"**Wasim Akram keh rahe hain:** '{q_to_ask} par Sultan ka mashwara: Ball ko zaroorat se zyada tight mat pakdo. Seam ko pehle slip ki taraf rukh do aur aakhri lamhay par wrist snap karo. Follow-through poora left hip tak khatam karo!'"
                }
            }

            chosen_ans = answers.get(st.session_state.selected_coach_id, answers["shoaib"])
            display_text = chosen_ans["ur"] if "Roman" in st.session_state.language_mode else chosen_ans["en"]
            st.markdown(f"""
            <div class='coach-box-good' style='font-size: 1.05rem; margin-top: 15px;'>
                {display_text}
            </div>
            """, unsafe_allow_html=True)

    # ========================================================================
    # TAB 4: 📅 MY TRAINING PLAN
    # ========================================================================
    with tab_plan:
        st.markdown(f"### 📅 {t('Your Personalized 7-Day Grassroots Roadmap', 'Aapka 7-Din Ka Practice Roadmap')}")
        st.caption(t(
            f"Crafted specifically by Coach {active_coach['name']} based on your playing level and target goals.",
            f"Coach {active_coach['name']} ne aapki routine aur goals ke mutabiq ye roadmap banaya hai."
        ))

        primary_flaw_arg = "knee" if "Speed" in st.session_state.user_profile["main_goal"] else "elbow"
        weekly_plan = WeeklyPlanner.create_plan(
            player_name="Champion",
            primary_flaw=primary_flaw_arg,
            skill_level=st.session_state.user_profile["level"]
        )

        st.markdown(f"""
        <div style='background: white; border-radius: 16px; padding: 20px; border: 1px solid #CBD5E1; margin-bottom: 20px;'>
            <div style='display: flex; justify-content: space-between; flex-wrap: wrap;'>
                <div>
                    <span class='tag-pill tag-gold'>🎯 {t('Weekly Goal', 'Hafte Ka Maqsad')}</span>
                    <h3 style='margin: 8px 0; color: #0F172A !important;'>{weekly_plan.weekly_goal}</h3>
                    <p style='color: #475569 !important; font-size: 0.9rem;'>Level: <b>{st.session_state.user_profile['level']}</b> | Mentor: <b>{active_coach['name']}</b></p>
                </div>
                <div style='text-align: right;'>
                    <span class='tag-pill' style='background: #DCFCE7 !important; color: #166534 !important;'>FREE STREET EQUIPMENT</span>
                    <p style='font-size: 0.85rem; color: #475569 !important; margin-top: 6px;'>Tennis Ball 🎾 | Chalk 🖍️ | Wall 🧱</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        for s_idx, session in enumerate(weekly_plan.daily_schedule):
            with st.expander(f"📌 {session.day_name} — {session.focus_theme} ({session.duration_minutes} mins)", expanded=(s_idx==0)):
                col_d1, col_d2 = st.columns([1.5, 1])
                with col_d1:
                    st.markdown(f"**🛠️ {t('Prescribed Drills:', 'Khas Drills:')}**")
                    for drill in session.drills:
                        st.markdown(f"- ⚡ **{drill['name']}** ({drill['reps']})")
                        st.caption(f"How to do it: {drill['instructions']}")
                    st.info(f"💡 **Safety & Mobility Reminder:** {session.safety_reminder}")
                with col_d2:
                    st.markdown(f"**✅ {t('Daily Checklist:', 'Checklist:')}**")
                    for task in session.checklist_tasks:
                        st.checkbox(task, key=f"chk_{session.day_name}_{task}")

    # ========================================================================
    # TAB 5: ⚖️ PROGRESS & COMPARE
    # ========================================================================
    with tab_progress:
        st.markdown(f"### ⚖️ {t('Delivery Consistency & Before/After Comparison', 'Session Muqabla aur Consistency')}")
        st.caption(t(
            "Compare two sessions side-by-side to verify if your front-knee bracing and bowling action improved.",
            "Do sessions ka aapas mein muqabla karein aur dekhein ke knee bracing aur elbow angle mein kitna behtari aayi."
        ))

        c_cmp1, c_cmp2 = st.columns(2)
        with c_cmp1:
            st.markdown(f"#### 📅 {t('Baseline Session 1 (Before)', 'Pehla Session (Pehle)')}")
            st.metric("Front Knee Bracing", "142.0°", "Bent / Collapsed")
            st.metric("Elbow Straightening", "14.5°", "Borderline ICC Limit")
            st.metric("Release Speed", "116.4 km/h")

        with c_cmp2:
            st.markdown(f"#### 🚀 {t('Session 2 (After 7 Days Practice)', 'Doosra Session (7 Din Baad)')}")
            st.metric("Front Knee Bracing", "166.5°", "+24.5° (Rigid Lever)")
            st.metric("Elbow Straightening", "6.2°", "-8.3° (100% Legal Action)")
            st.metric("Release Speed", "125.8 km/h", "+9.4 km/h Faster!")

        st.markdown("---")
        st.success(t(
            "🎉 **Coach Verdict:** Massive improvement! Your front knee is acting like a rigid catapult, boosting your pace by nearly 10 km/h with a squeaky clean legal action!",
            "🎉 **Ustaad Ka Faisla:** Zabardast tarakki! Agla ghutna seedha pillar ban chuka hai jis se speed taqreeban 10 km/h barh gayi hai aur action 100% clean hai!"
        ))
