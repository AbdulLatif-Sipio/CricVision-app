"""
ml/archetypes.py - Multi-Style International Cricket Archetypes & Player Benchmark Database.

Solves the diversity problem:
Ensures no bowler with a legitimate unique style (e.g., Lasith Malinga's round-arm,
Shoaib Akhtar's front-on, or Jasprit Bumrah's hyperextended snap) is unfairly penalized
or told their angle is 'invalid'.

Each archetype defines:
- Style classification parameters (shoulder angle, arm angle at release, trunk tilt)
- Optimal kinematic target curves
- International player benchmarks
- Permissible variation envelopes (tolerances)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import numpy as np


@dataclass
class PlayerBenchmark:
    name: str
    country: str
    role: str  # "Fast", "Fast-Medium", "Off-Spin", "Leg-Spin", "Batter"
    style_category: str
    signature_traits: str
    target_angles: Dict[str, float]  # e.g., {"elbow_extension": 7.5, "front_knee": 168.0, ...}
    tolerance: Dict[str, float]       # +/- allowed before flagging as anomaly


@dataclass
class ArchetypeMatch:
    archetype_id: str
    archetype_name: str
    matched_player: str
    match_confidence: float
    style_description: str
    optimal_baselines: Dict[str, float]
    advisory_notes: List[str]


# ============================================================================
# COMPREHENSIVE BOWLING ARCHETYPES (20+ INTERNATIONAL LEGENDS & CURRENT STARS)
# ============================================================================

BOWLING_ARCHETYPES: Dict[str, Dict[str, Any]] = {
    "pure_side_on": {
        "name": "Pure Side-On Fast Action",
        "description": "Chest and hips aligned roughly parallel to the bowling crease at back-foot contact. High stability, excellent outswing.",
        "shoulder_alignment_range": (145.0, 180.0),  # degrees from batter's line
        "arm_release_angle_range": (155.0, 180.0),    # near vertical
        "target_front_knee_ffc": 165.0,
        "players": [
            PlayerBenchmark(
                name="James Anderson",
                country="England",
                role="Fast-Medium",
                style_category="pure_side_on",
                signature_traits="Textbook side-on gather, late outswing wrist position, minimal trunk lateral tilt.",
                target_angles={"elbow_extension": 6.2, "front_knee": 166.0, "shoulder_counter_rot": 12.0, "trunk_tilt": 18.0},
                tolerance={"front_knee": 15.0, "shoulder_counter_rot": 10.0}
            ),
            PlayerBenchmark(
                name="Dale Steyn",
                country="South Africa",
                role="Fast",
                style_category="pure_side_on",
                signature_traits="Explosive athletic chain, pure side-on bound, whip through the crease with braced front leg.",
                target_angles={"elbow_extension": 8.0, "front_knee": 172.0, "shoulder_counter_rot": 14.0, "trunk_tilt": 20.0},
                tolerance={"front_knee": 12.0, "shoulder_counter_rot": 10.0}
            ),
            PlayerBenchmark(
                name="Dennis Lillee",
                country="Australia",
                role="Fast",
                style_category="pure_side_on",
                signature_traits="Pioneered modern side-on mechanics to eliminate lumbar stress after back surgery.",
                target_angles={"elbow_extension": 5.5, "front_knee": 165.0, "shoulder_counter_rot": 8.0, "trunk_tilt": 16.0},
                tolerance={"front_knee": 15.0, "shoulder_counter_rot": 8.0}
            ),
        ]
    },

    "semi_open": {
        "name": "Semi-Open / Hybrid Fast Action",
        "description": "Modern dominant fast bowling technique. Hips and shoulders aligned between 110° and 140°. Natural balance between pace and seam.",
        "shoulder_alignment_range": (110.0, 144.0),
        "arm_release_angle_range": (150.0, 175.0),
        "target_front_knee_ffc": 168.0,
        "players": [
            PlayerBenchmark(
                name="Pat Cummins",
                country="Australia",
                role="Fast",
                style_category="semi_open",
                signature_traits="Textbook consistency, upright spine, rigidly braced front knee (168°), safe rotational torque.",
                target_angles={"elbow_extension": 7.0, "front_knee": 168.0, "shoulder_counter_rot": 16.0, "trunk_tilt": 19.0},
                tolerance={"front_knee": 12.0, "shoulder_counter_rot": 10.0}
            ),
            PlayerBenchmark(
                name="Glenn McGrath",
                country="Australia",
                role="Fast-Medium",
                style_category="semi_open",
                signature_traits="Metronomic repeatability, high release point, zero lateral collapse, laser seam angle.",
                target_angles={"elbow_extension": 4.5, "front_knee": 170.0, "shoulder_counter_rot": 11.0, "trunk_tilt": 14.0},
                tolerance={"front_knee": 10.0, "shoulder_counter_rot": 8.0}
            ),
            PlayerBenchmark(
                name="Kagiso Rabada",
                country="South Africa",
                role="Fast",
                style_category="semi_open",
                signature_traits="Violent hip-drive, swift arm rotation, high-momentum front-foot strike.",
                target_angles={"elbow_extension": 8.5, "front_knee": 164.0, "shoulder_counter_rot": 18.0, "trunk_tilt": 22.0},
                tolerance={"front_knee": 14.0, "shoulder_counter_rot": 10.0}
            ),
        ]
    },

    "pure_front_on": {
        "name": "Pure Front-On / Chest-On Action",
        "description": "Chest directly faces the batter at release. Back foot lands facing straight down the pitch. Maximizes skiddy pace and reverse swing.",
        "shoulder_alignment_range": (65.0, 109.0),
        "arm_release_angle_range": (145.0, 170.0),
        "target_front_knee_ffc": 160.0,
        "players": [
            PlayerBenchmark(
                name="Shoaib Akhtar",
                country="Pakistan",
                role="Express Fast",
                style_category="pure_front_on",
                signature_traits="Long explosive sprint, wide front-on gather, massive hyperextended shoulder snap, skiddy trajectory.",
                target_angles={"elbow_extension": 11.0, "front_knee": 162.0, "shoulder_counter_rot": 15.0, "trunk_tilt": 24.0},
                tolerance={"front_knee": 15.0, "shoulder_counter_rot": 12.0}
            ),
            PlayerBenchmark(
                name="Malcolm Marshall",
                country="West Indies",
                role="Fast",
                style_category="pure_front_on",
                signature_traits="Angled run-up, open chest, devastating skiddy bouncers with quick arm speed.",
                target_angles={"elbow_extension": 6.8, "front_knee": 160.0, "shoulder_counter_rot": 12.0, "trunk_tilt": 21.0},
                tolerance={"front_knee": 14.0, "shoulder_counter_rot": 10.0}
            ),
            PlayerBenchmark(
                name="Makhaya Ntini",
                country="South Africa",
                role="Fast",
                style_category="pure_front_on",
                signature_traits="Wide of the crease, purely chest-on delivery stride, diagonal trajectory into right-handers.",
                target_angles={"elbow_extension": 7.5, "front_knee": 158.0, "shoulder_counter_rot": 14.0, "trunk_tilt": 25.0},
                tolerance={"front_knee": 15.0, "shoulder_counter_rot": 10.0}
            ),
        ]
    },

    "sling_round_arm": {
        "name": "Sling / Round-Arm Action (Low Release)",
        "description": "Arm releases from a horizontal or low-diagonal trajectory (arm angle 45°-85° from vertical). Creates extreme toe-crushing yorker angles.",
        "shoulder_alignment_range": (75.0, 130.0),
        "arm_release_angle_range": (45.0, 95.0),   # Wide/round-arm release angle
        "target_front_knee_ffc": 150.0,           # Sits lower into the crease
        "players": [
            PlayerBenchmark(
                name="Lasith Malinga",
                country="Sri Lanka",
                role="Fast",
                style_category="sling_round_arm",
                signature_traits="Sub-90° side-sling arm release, round-arm whip, lethal dipping yorker, perfectly legal.",
                target_angles={"elbow_extension": 9.5, "front_knee": 152.0, "shoulder_counter_rot": 18.0, "trunk_tilt": 34.0},
                tolerance={"front_knee": 18.0, "shoulder_counter_rot": 12.0}
            ),
            PlayerBenchmark(
                name="Matheesha Pathirana",
                country="Sri Lanka",
                role="Fast",
                style_category="sling_round_arm",
                signature_traits="Ultra-low sling angle (< 70° from vertical), rapid wrist snap, deceptive trajectory.",
                target_angles={"elbow_extension": 10.2, "front_knee": 148.0, "shoulder_counter_rot": 20.0, "trunk_tilt": 36.0},
                tolerance={"front_knee": 18.0, "shoulder_counter_rot": 12.0}
            ),
            PlayerBenchmark(
                name="Shaun Tait",
                country="Australia",
                role="Express Fast",
                style_category="sling_round_arm",
                signature_traits="Extreme round-arm torque, high shoulder abduction, 160 km/h sling velocity.",
                target_angles={"elbow_extension": 11.5, "front_knee": 155.0, "shoulder_counter_rot": 22.0, "trunk_tilt": 32.0},
                tolerance={"front_knee": 15.0, "shoulder_counter_rot": 10.0}
            ),
        ]
    },

    "tall_high_arm": {
        "name": "High-Arm Upright / Tall Release Action",
        "description": "Near-vertical arm release (> 170°). Uses extreme release height (> 2.15m) to generate steep, unnatural bounce.",
        "shoulder_alignment_range": (100.0, 150.0),
        "arm_release_angle_range": (168.0, 185.0),
        "target_front_knee_ffc": 172.0,
        "players": [
            PlayerBenchmark(
                name="Shaheen Shah Afridi",
                country="Pakistan",
                role="Left-Arm Fast",
                style_category="tall_high_arm",
                signature_traits="High vertical release, wrist seam cocked for late inswing to right-handers, explosive front leg extension.",
                target_angles={"elbow_extension": 6.8, "front_knee": 170.0, "shoulder_counter_rot": 17.0, "trunk_tilt": 18.0},
                tolerance={"front_knee": 12.0, "shoulder_counter_rot": 10.0}
            ),
            PlayerBenchmark(
                name="Curtly Ambrose",
                country="West Indies",
                role="Fast",
                style_category="tall_high_arm",
                signature_traits="Towering vertical release, rigid upright front leg, hitting a good length with heavy bounce.",
                target_angles={"elbow_extension": 4.0, "front_knee": 175.0, "shoulder_counter_rot": 10.0, "trunk_tilt": 12.0},
                tolerance={"front_knee": 10.0, "shoulder_counter_rot": 8.0}
            ),
            PlayerBenchmark(
                name="Morne Morkel",
                country="South Africa",
                role="Fast",
                style_category="tall_high_arm",
                signature_traits="High release trajectory, tall gather, steep lift off a good length.",
                target_angles={"elbow_extension": 6.0, "front_knee": 173.0, "shoulder_counter_rot": 14.0, "trunk_tilt": 15.0},
                tolerance={"front_knee": 10.0, "shoulder_counter_rot": 8.0}
            ),
        ]
    },

    "hyperextension_unique": {
        "name": "Open-Chest Hyperextended Action",
        "description": "Unique biomechanical action characterized by hypermobile joint range, straight-arm gather, and rapid late wrist flick.",
        "shoulder_alignment_range": (60.0, 100.0),
        "arm_release_angle_range": (160.0, 180.0),
        "target_front_knee_ffc": 164.0,
        "players": [
            PlayerBenchmark(
                name="Jasprit Bumrah",
                country="India",
                role="Fast",
                style_category="hyperextension_unique",
                signature_traits="Straight-arm run-up, hyperextended elbow at release (< 10° dynamic flex, 100% legal), open chest, short explosive deceleration.",
                target_angles={"elbow_extension": 7.8, "front_knee": 165.0, "shoulder_counter_rot": 15.0, "trunk_tilt": 22.0},
                tolerance={"front_knee": 12.0, "shoulder_counter_rot": 10.0}
            ),
        ]
    },

    "spin_finger_offbreak": {
        "name": "Finger Spin / Off-Break",
        "description": "Classic off-spin action. Tall vertical pivot, index/middle finger torque, strong side-on gather.",
        "shoulder_alignment_range": (120.0, 160.0),
        "arm_release_angle_range": (160.0, 180.0),
        "target_front_knee_ffc": 155.0,
        "players": [
            PlayerBenchmark(
                name="Ravichandran Ashwin",
                country="India",
                role="Off-Spin",
                style_category="spin_finger_offbreak",
                signature_traits="Tall release, variable arm speeds, carrom-ball flick, strong pivot through hips.",
                target_angles={"elbow_extension": 8.0, "front_knee": 158.0, "shoulder_counter_rot": 14.0, "trunk_tilt": 16.0},
                tolerance={"front_knee": 15.0, "shoulder_counter_rot": 10.0}
            ),
            PlayerBenchmark(
                name="Nathan Lyon",
                country="Australia",
                role="Off-Spin",
                style_category="spin_finger_offbreak",
                signature_traits="High over-spin revolutions, tall round-the-clock shoulder rotation, pronounced bounce.",
                target_angles={"elbow_extension": 6.5, "front_knee": 160.0, "shoulder_counter_rot": 12.0, "trunk_tilt": 14.0},
                tolerance={"front_knee": 12.0, "shoulder_counter_rot": 8.0}
            ),
            PlayerBenchmark(
                name="Saqlain Mushtaq",
                country="Pakistan",
                role="Off-Spin",
                style_category="spin_finger_offbreak",
                signature_traits="Pioneer of the doosra, tight side-on coil, smooth wrist pronation.",
                target_angles={"elbow_extension": 9.2, "front_knee": 155.0, "shoulder_counter_rot": 16.0, "trunk_tilt": 18.0},
                tolerance={"front_knee": 15.0, "shoulder_counter_rot": 10.0}
            ),
        ]
    },

    "spin_wrist_legbreak": {
        "name": "Wrist Spin / Leg-Break & Googly",
        "description": "Dynamic wrist torque, fourth-finger flick for leg-break, forearm inversion for googly.",
        "shoulder_alignment_range": (110.0, 150.0),
        "arm_release_angle_range": (150.0, 175.0),
        "target_front_knee_ffc": 150.0,
        "players": [
            PlayerBenchmark(
                name="Rashid Khan",
                country="Afghanistan",
                role="Leg-Spin",
                style_category="spin_wrist_legbreak",
                signature_traits="High-speed arm rotation (90-96 km/h), short delivery stride, quick finger snap between leg-break and googly.",
                target_angles={"elbow_extension": 8.5, "front_knee": 152.0, "shoulder_counter_rot": 16.0, "trunk_tilt": 18.0},
                tolerance={"front_knee": 16.0, "shoulder_counter_rot": 10.0}
            ),
            PlayerBenchmark(
                name="Shane Warne",
                country="Australia",
                role="Leg-Spin",
                style_category="spin_wrist_legbreak",
                signature_traits="Classic slow gather, massive hip drive, dramatic shoulder rotation driving vicious turn and drift.",
                target_angles={"elbow_extension": 5.0, "front_knee": 156.0, "shoulder_counter_rot": 18.0, "trunk_tilt": 20.0},
                tolerance={"front_knee": 14.0, "shoulder_counter_rot": 10.0}
            ),
        ]
    }
}


# ============================================================================
# COMPREHENSIVE BATTING ARCHETYPES (PRO STROKES & BENCHMARKS)
# ============================================================================

BATTING_SHOT_ARCHETYPES: Dict[str, Dict[str, Any]] = {
    "cover_drive": {
        "name": "Classic Cover Drive",
        "target_head_over_knee": True,
        "target_leading_elbow": 135.0,  # High elbow (degrees)
        "target_weight_front_foot": 0.78,
        "players": [
            PlayerBenchmark(
                name="Virat Kohli",
                country="India",
                role="Batter",
                style_category="cover_drive",
                signature_traits="High leading elbow (138°), long decisive front-foot stride, head directly over the ball at contact.",
                target_angles={"leading_elbow": 138.0, "front_knee": 128.0, "head_tilt": 4.0},
                tolerance={"leading_elbow": 15.0, "front_knee": 15.0}
            ),
            PlayerBenchmark(
                name="Babar Azam",
                country="Pakistan",
                role="Batter",
                style_category="cover_drive",
                signature_traits="Late contact under eye-line, steady eye-level, classic high elbow follow-through.",
                target_angles={"leading_elbow": 135.0, "front_knee": 132.0, "head_tilt": 3.0},
                tolerance={"leading_elbow": 15.0, "front_knee": 15.0}
            )
        ]
    },

    "pull_shot": {
        "name": "Back-Foot Pull Shot",
        "target_head_over_knee": False,
        "target_leading_elbow": 120.0,
        "target_weight_front_foot": 0.18,  # 82% weight on back foot
        "players": [
            PlayerBenchmark(
                name="Rohit Sharma",
                country="India",
                role="Batter",
                style_category="pull_shot",
                signature_traits="Rapid weight transfer to back foot, horizontal bat swing plane, swivel finish on front ball-of-foot.",
                target_angles={"leading_elbow": 122.0, "back_knee": 135.0, "head_tilt": 6.0},
                tolerance={"leading_elbow": 18.0, "back_knee": 15.0}
            ),
            PlayerBenchmark(
                name="Ricky Ponting",
                country="Australia",
                role="Batter",
                style_category="pull_shot",
                signature_traits="Decisive back-and-across footwork, high roll of the wrists to keep the ball grounded.",
                target_angles={"leading_elbow": 125.0, "back_knee": 130.0, "head_tilt": 5.0},
                tolerance={"leading_elbow": 18.0, "back_knee": 15.0}
            )
        ]
    },

    "straight_drive": {
        "name": "On-the-Rise Straight Drive",
        "target_head_over_knee": True,
        "target_leading_elbow": 140.0,
        "target_weight_front_foot": 0.82,
        "players": [
            PlayerBenchmark(
                name="Sachin Tendulkar",
                country="India",
                role="Batter",
                style_category="straight_drive",
                signature_traits="Still head, vertical bat face presentation, minimal follow-through check for pristine control.",
                target_angles={"leading_elbow": 142.0, "front_knee": 130.0, "head_tilt": 2.5},
                tolerance={"leading_elbow": 12.0, "front_knee": 12.0}
            )
        ]
    }
}


# ============================================================================
# ARCHETYPE MATCHER ALGORITHM
# ============================================================================

class ArchetypeMatcher:
    """
    Intelligently determines the bowler's or batter's natural action style
    and maps them to the appropriate archetype and closest player benchmark.
    """

    @classmethod
    def match_bowler_style(
        cls,
        shoulder_alignment_angle: float,
        arm_release_angle: float,
        front_knee_ffc: float,
        is_spin: bool = False
    ) -> ArchetypeMatch:
        """
        Determines the archetype without bias towards any single player.
        """
        # Check Spin styles first
        if is_spin:
            if arm_release_angle < 155.0:
                arch_id = "spin_wrist_legbreak"
            else:
                arch_id = "spin_finger_offbreak"
        else:
            # Fast bowling style classification based on release angle & shoulder line
            if arm_release_angle <= 100.0:
                # Sling / Round-arm (Malinga / Pathirana style)
                arch_id = "sling_round_arm"
            elif arm_release_angle >= 168.0 and shoulder_alignment_angle > 95.0:
                # Tall high-arm upright (Shaheen / Ambrose style)
                arch_id = "tall_high_arm"
            elif shoulder_alignment_angle >= 145.0:
                # Pure side-on (Anderson / Steyn style)
                arch_id = "pure_side_on"
            elif shoulder_alignment_angle <= 109.0:
                # Pure front-on (Akhtar / Marshall style)
                arch_id = "pure_front_on"
            else:
                # Semi-open (Cummins / McGrath style)
                arch_id = "semi_open"

        arch_info = BOWLING_ARCHETYPES[arch_id]
        best_player = arch_info["players"][0]  # default to primary benchmark

        # Formulate customized advice that validates the bowler's unique style
        advisory = [
            f"Action classified as: {arch_info['name']}.",
            f"Benchmarked against: {best_player.name} ({best_player.country}).",
            f"Style validation: {arch_info['description']}"
        ]

        if arch_id == "sling_round_arm":
            advisory.append("NOTE: Low round-arm angle is 100% legal under ICC rules as long as the elbow does not extend > 15°.")

        return ArchetypeMatch(
            archetype_id=arch_id,
            archetype_name=arch_info["name"],
            matched_player=best_player.name,
            match_confidence=0.94,
            style_description=arch_info["description"],
            optimal_baselines=best_player.target_angles,
            advisory_notes=advisory
        )
