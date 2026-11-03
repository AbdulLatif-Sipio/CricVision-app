"""
coaching/drill_matrix.py - 100% Offline Grassroots Cricket Drill & Correction Engine.

Provides instant, zero-internet prescriptions for every biomechanical flaw:
Every prescription is designed specifically for grassroots players (gully, school, club)
using accessible equipment (tennis balls, chalk lines, walls, towels) rather than
expensive lab apparatus.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class PrescriptiveDrill:
    flaw_name: str
    biomechanical_cause: str
    target_metric: str
    drill_name: str
    equipment_needed: str
    step_by_step_instructions: List[str]
    reps_and_frequency: str
    pro_model_reference: str


# ============================================================================
# COMPREHENSIVE GRASSROOTS DRILL MATRIX
# ============================================================================

DRILL_CATALOGUE: Dict[str, PrescriptiveDrill] = {
    # 1. ICC 15-Degree Law & Arm Straightening
    "chucking_elbow_flexion": PrescriptiveDrill(
        flaw_name="Dynamic Elbow Extension (Chucking Flaw / ICC Law 21.2)",
        biomechanical_cause="Elbow bends during gather and snaps straight during delivery swing to generate artificial whip.",
        target_metric="Elbow extension change <= 15.0° (Target: Lock elbow at < 8° dynamic change)",
        drill_name="The Rigid Arm Wall-Slide Drill",
        equipment_needed="A smooth brick or plaster wall + tennis ball",
        step_by_step_instructions=[
            "Stand with your bowling arm side 6 inches from a flat wall.",
            "Hold a tennis ball with bowling grip and place the back of your hand against the wall.",
            "Rotate your bowling arm through the full 360-degree circle while keeping your hand touching or grazing the wall throughout.",
            "If your elbow bends, your hand will detach from the wall, giving immediate physical sensory feedback.",
            "Release the ball into a soft net or mattress at the top of the circle with a locked, straight arm."
        ],
        reps_and_frequency="3 sets of 25 revolutions daily before bowling in the nets.",
        pro_model_reference="Glenn McGrath & Pat Cummins"
    ),

    # 2. Collapsed Front Knee at Front-Foot Contact
    "collapsed_front_knee": PrescriptiveDrill(
        flaw_name="Collapsed Front Knee at FFC (Power Leak & Pace Loss)",
        biomechanical_cause="Front leg buckles upon impact (< 140°), absorbing kinetic energy instead of acting as a rigid catapult lever.",
        target_metric="Front knee angle >= 165.0° at release (braced)",
        drill_name="The Step-Over Cone Bracing Drill",
        equipment_needed="A single cone, brick, or chalk mark on the crease",
        step_by_step_instructions=[
            "Place a cone or chalk line at your front foot landing spot.",
            "Take a short 3-step approach.",
            "As your back foot lands, drive your front leg forward and lock the knee firmly BEFORE the heel touches the ground.",
            "Imagine your front leg is a solid wooden pole. Deliver the ball over the braced leg without allowing the knee to dip.",
            "Hold your finish position on the braced leg for 2 seconds to cement muscle memory."
        ],
        reps_and_frequency="4 sets of 10 deliveries off 3 paces, focusing purely on knee lock.",
        pro_model_reference="Dale Steyn & Shaheen Shah Afridi"
    ),

    # 3. Mixed Action / Counter-Rotation (Lumbar Spine Injury Risk)
    "mixed_action_counter_rotation": PrescriptiveDrill(
        flaw_name="Mixed Bowling Action (Lumbar Spondylolysis Stress)",
        biomechanical_cause="Hips land side-on (> 110°) while shoulders counter-rotate chest-on (< 70°), twisting the lower vertebrae under high G-forces.",
        target_metric="Shoulder counter-rotation <= 20.0° (Eliminate spinal torsion)",
        drill_name="The Two-Parallel-Rope Crease Alignment Drill",
        equipment_needed="Two lengths of skipping rope, string, or chalk lines",
        step_by_step_instructions=[
            "Lay two parallel chalk lines 1.5 meters apart pointing straight towards the target stumps.",
            "Decide on ONE pure action: Either Pure Side-On (feet & shoulders aligned 90° to the lines) OR Pure Front-On (facing the target).",
            "Bowl along the corridor ensuring your hips and shoulders remain locked in the SAME rotational plane throughout gather and release.",
            "Do NOT allow your shoulders to twist back after your back foot has landed."
        ],
        reps_and_frequency="30 repetitions at 50% pace to re-wire kinetic alignment.",
        pro_model_reference="Dennis Lillee & James Anderson"
    ),

    # 4. Head Falling Away to the Off-Side
    "head_falling_away": PrescriptiveDrill(
        flaw_name="Head Falling Away to the Off-Side (Loss of Accuracy & Seam)",
        biomechanical_cause="Bowler leans upper body prematurely towards off-side, dragging the release point down and spraying deliveries wide.",
        target_metric="Head tilt angle <= 8.0° from vertical",
        drill_name="The Chin-Tuck Collar Alignment Drill",
        equipment_needed="A coin, folded cloth, or tennis ball tucked under chin",
        step_by_step_instructions=[
            "Tuck a small sponge or tennis ball between your chin and the non-bowling shoulder during gather.",
            "Keep your eyes locked on the top of off-stump throughout your run-up and delivery stride.",
            "Deliver the ball while ensuring your head remains perfectly vertical over the midline of your torso.",
            "Do not allow your chin to drop towards your left armpit (for right-arm bowlers)."
        ],
        reps_and_frequency="5 sets of 8 deliveries with full bowling action.",
        pro_model_reference="Malcolm Marshall & Kagiso Rabada"
    ),

    # 5. Batting: Dropped Leading Elbow on Drives
    "batting_dropped_elbow": PrescriptiveDrill(
        flaw_name="Dropped Leading Elbow (Risk of Aerial Catches)",
        biomechanical_cause="Bottom hand dominates early, collapsing the top-hand lever (< 115°) and slicing the ball into the air.",
        target_metric="Leading elbow elevation >= 135.0° (High elbow)",
        drill_name="The Top-Hand Only Hanging Ball Drill",
        equipment_needed="A tennis ball suspended in a sock or rope + cricket bat",
        step_by_step_instructions=[
            "Grip the bat with your TOP HAND ONLY (left hand for right-handers).",
            "Step forward into a classic cover drive stride towards the hanging ball.",
            "Drive the bat forward leading with the point of the elbow pointing directly at the target.",
            "Hold the follow-through pose with your elbow high and proud ('pointing to the sky').",
            "Perform 50 top-hand strikes, then add the bottom hand with a two-finger light grip."
        ],
        reps_and_frequency="50 top-hand strokes daily against a hanging ball or gentle underarm throws.",
        pro_model_reference="Virat Kohli & Babar Azam"
    ),

    # 6. Batting: Weight Stuck on Back Foot
    "batting_weight_back_foot": PrescriptiveDrill(
        flaw_name="Weight Trapped on Back Foot During Front-Foot Drive",
        biomechanical_cause="Batter fails to transfer center of mass forward, reaching for the ball with hands only and losing stroke power.",
        target_metric="Weight transfer ratio >= 75% on front foot at impact",
        drill_name="The Knee-Bend Coin Stride Drill",
        equipment_needed="A coin or small stone placed on the pitch",
        step_by_step_instructions=[
            "Place a coin 2 feet in front of your batting crease.",
            "As the thrower releases a full-length underarm ball, stride forward so your front knee flexes directly over the coin.",
            "Ensure your nose and front knee form a vertical plumb line over the coin at the exact moment of bat contact.",
            "Lift the heel of your back foot off the ground to prove that 80% of your bodyweight has transferred forward."
        ],
        reps_and_frequency="3 sets of 15 drives off throwdowns.",
        pro_model_reference="Sachin Tendulkar & Joe Root"
    )
}


class DrillEngine:
    """
    Offline prescription engine that pairs detected biomechanical flaws with actionable grassroots drills.
    """

    @classmethod
    def get_drills_for_bowling(
        cls,
        is_legal: bool,
        elbow_extension: float,
        shoulder_counter_rot: float,
        front_knee_ffc: float,
        head_tilt: float
    ) -> List[PrescriptiveDrill]:
        """
        Prescribes targeted drills based on exact quantitative thresholds.
        """
        prescribed = []

        # Check illegal or borderline elbow
        if not is_legal or elbow_extension > 13.0:
            prescribed.append(DRILL_CATALOGUE["chucking_elbow_flexion"])

        # Check lumbar counter-rotation
        if shoulder_counter_rot > 25.0:
            prescribed.append(DRILL_CATALOGUE["mixed_action_counter_rotation"])

        # Check front knee collapse
        if front_knee_ffc < 150.0:
            prescribed.append(DRILL_CATALOGUE["collapsed_front_knee"])

        # Check head tilt
        if head_tilt > 14.0:
            prescribed.append(DRILL_CATALOGUE["head_falling_away"])

        # Fallback if technique is already solid
        if len(prescribed) == 0:
            prescribed.append(DRILL_CATALOGUE["collapsed_front_knee"])

        return prescribed

    @classmethod
    def get_drills_for_batting(
        cls,
        leading_elbow: float,
        front_knee: float,
        weight_front_foot: float,
        head_tilt: float
    ) -> List[PrescriptiveDrill]:
        """
        Prescribes targeted batting drills.
        """
        prescribed = []

        if leading_elbow < 125.0:
            prescribed.append(DRILL_CATALOGUE["batting_dropped_elbow"])

        if weight_front_foot < 0.60:
            prescribed.append(DRILL_CATALOGUE["batting_weight_back_foot"])

        if len(prescribed) == 0:
            prescribed.append(DRILL_CATALOGUE["batting_dropped_elbow"])

        return prescribed

    @classmethod
    def build_structured_bowling_report(
        cls,
        is_legal: bool,
        elbow_extension: float,
        shoulder_counter_rot: float,
        front_knee_ffc: float,
        head_tilt: float,
        confidence_val: float = 0.90
    ) -> Dict[str, Any]:
        """
        Builds a human-friendly coaching report focused on ONE primary issue,
        following the 4-part coaching architecture:
        1. Main Observation
        2. What This Means
        3. Try This Drill (with reps & safety guidance)
        4. Next Session Focus
        """
        import uuid

        # Determine the single most critical technique bottleneck
        if not is_legal or elbow_extension > 14.0:
            main_obs = f"Elbow dynamic extension reached {elbow_extension:.1f}° (Screening Alert)."
            what_means = "Your bowling arm flexed during gather and straightened before release. This creates artificial whip and risks illegal action sanction under ICC Law 21.2."
            drill = DRILL_CATALOGUE["chucking_elbow_flexion"]
            safety = "Safety guidance: Do not force arm locks under fast bowling speeds immediately. Build muscle memory through slow wall slides."
            next_focus = "Record 5 deliveries from side-on and verify elbow extension stays under 12.0°."
        elif front_knee_ffc < 150.0:
            main_obs = f"Front knee collapsed to {front_knee_ffc:.1f}° upon front-foot landing."
            what_means = "Your front leg absorbed the landing shock like a cushion instead of a rigid pole. This leaks approximately 6-10 km/h of bowling pace and strains your knee tendon."
            drill = DRILL_CATALOGUE["collapsed_front_knee"]
            safety = "Safety guidance: Land on flat footwear or proper turf. Warm up quadriceps and hamstrings before bracing drills."
            next_focus = "Compare your front knee angle in your next session with a target of >= 165.0°."
        elif shoulder_counter_rot > 25.0:
            main_obs = f"Mixed action detected: Shoulders counter-rotated by {shoulder_counter_rot:.1f}° relative to hips."
            what_means = "Your hips landed side-on while your upper body twisted chest-on, twisting lower vertebrae under high impact."
            drill = DRILL_CATALOGUE["mixed_action_counter_rotation"]
            safety = "Safety guidance: Stop bowling immediately if experiencing sharp unilateral lower-back pain. Seek qualified physical assessment."
            next_focus = "Check that your hips and shoulders align on the same plane throughout gather."
        elif head_tilt > 14.0:
            main_obs = f"Head fell away by {head_tilt:.1f}° towards the off-side at release."
            what_means = "Leaning your head drops your bowling shoulder, pulling your release channel wide and destabilizing late swing."
            drill = DRILL_CATALOGUE["head_falling_away"]
            safety = "Safety guidance: Keep neck muscles relaxed and eyes level on target."
            next_focus = "Keep eyes locked on top of off-stump from approach through follow-through."
        else:
            main_obs = f"Balanced mechanics observed (Front knee braced at {front_knee_ffc:.1f}°, clean arm circle)."
            what_means = "Your core mechanical alignment is sound. Minor rhythm refinements will unlock greater ball speed and consistency."
            drill = DRILL_CATALOGUE["collapsed_front_knee"]
            safety = "Safety guidance: Maintain regular rotator cuff and core conditioning."
            next_focus = "Compare release speed and delivery repeatability across your next spell."

        conf_label = "High Confidence" if confidence_val >= 0.85 else ("Moderate Confidence" if confidence_val >= 0.70 else "Screening Only")

        return {
            "observation_id": str(uuid.uuid4()),
            "main_observation": main_obs,
            "what_this_means": what_means,
            "confidence_score": round(confidence_val, 2),
            "confidence_label": conf_label,
            "recommended_improvement": f"Master {drill.drill_name} to stabilize this movement pattern.",
            "drill": {
                "name": drill.drill_name,
                "equipment": drill.equipment_needed,
                "instructions": drill.step_by_step_instructions,
                "reps_and_frequency": drill.reps_and_frequency,
                "pro_model_reference": drill.pro_model_reference
            },
            "safety_guidance": safety,
            "next_session_focus": next_focus,
            "can_dismiss_or_correct": True,
            "is_dismissed": False,
            "user_correction": None
        }
