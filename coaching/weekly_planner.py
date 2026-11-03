"""
coaching/weekly_planner.py - 7-Day Grassroots Cricket Practice Planner.

Generates a structured, low-cost weekly training schedule based on detected
technique flaws, available equipment (tennis ball, chalk, walls), and player level.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class DailyPracticeSession:
    day_number: int
    day_name: str          # e.g. "Monday", "Tuesday", etc.
    focus_theme: str       # e.g. "Knee Bracing Foundation", "Arm Straightening Muscle Memory"
    duration_minutes: int
    is_rest_day: bool
    equipment_needed: List[str]
    drills: List[Dict[str, Any]]
    safety_reminder: str
    checklist_tasks: List[str]


@dataclass
class WeeklyTrainingPlan:
    player_name: str
    target_flaw: str
    skill_level: str       # "Beginner", "Intermediate", "Advanced"
    weekly_goal: str
    total_active_days: int
    daily_schedule: List[DailyPracticeSession]
    equipment_checklist: List[str]
    coach_note: str


class WeeklyPlanner:
    """
    Creates personalized 7-day training calendars focused on 1 core technique priority.
    """

    @classmethod
    def create_plan(
        cls,
        player_name: str = "Grassroots Player",
        primary_flaw: str = "collapsed_front_knee",
        skill_level: str = "Intermediate",
        available_equipment: Optional[List[str]] = None
    ) -> WeeklyTrainingPlan:
        equipment = available_equipment or ["tennis_ball", "chalk", "wall", "cone_or_brick"]

        flaw_key = primary_flaw.lower()

        if "knee" in flaw_key:
            theme = "Front Leg Bracing & Kinetic Energy Catapult"
            goal = "Increase front knee angle from bent (<145°) to rigidly locked (>=165°) at release."
            mon_drills = [{
                "name": "Static Standing Stride-Lock",
                "reps": "3 sets of 12 locks",
                "instructions": "Place front foot on chalk line, straighten knee firmly, practice rocking upper body forward."
            }]
            wed_drills = [{
                "name": "Step-Over Cone Bracing Drill",
                "reps": "4 sets of 8 deliveries off 3 paces",
                "instructions": "Land back foot, drive front leg straight, hold follow-through for 2 seconds without knee dipping."
            }]
            fri_drills = [{
                "name": "Full Run-Up Braced Delivery with Video Check",
                "reps": "3 sets of 6 deliveries at 80% pace",
                "instructions": "Bowl with full approach. Focus on the sensory feeling of planting a solid wooden pillar."
            }]
            sat_drills = [{
                "name": "Crease Target Accuracy & Firm Finish",
                "reps": "5 sets of 6 deliveries",
                "instructions": "Hit good-length chalk mark while maintaining head stability and braced knee."
            }]
        elif "elbow" in flaw_key or "chuck" in flaw_key:
            theme = "Rigid Arm Circle & Pure Shoulder Windmill"
            goal = "Eliminate elbow flex/extension during swing (keep dynamic angle change under 8°)."
            mon_drills = [{
                "name": "The Rigid Wall-Slide Drill",
                "reps": "3 sets of 25 revolutions",
                "instructions": "Stand 6 inches from wall, rotate arm in full circle with back of hand brushing wall."
            }]
            wed_drills = [{
                "name": "Towel Under Armpit Gather Drill",
                "reps": "4 sets of 10 gathers",
                "instructions": "Keep bowling arm high without bending at the elbow during back-foot impact."
            }]
            fri_drills = [{
                "name": "Short-Stride High-Arm Windmill Releases",
                "reps": "4 sets of 8 deliveries into soft net",
                "instructions": "Rotate purely from shoulder joint with elbow completely locked."
            }]
            sat_drills = [{
                "name": "Video Progress Recording & Angle Verification",
                "reps": "6 recorded deliveries",
                "instructions": "Record side-on video at 60fps and upload to CricVision to verify elbow straightening."
            }]
        else:
            theme = "Core Balance & Stable Follow-Through"
            goal = "Maintain vertical head alignment and eliminate torso counter-rotation."
            mon_drills = [{
                "name": "Chin-Tuck Collar Alignment Drill",
                "reps": "4 sets of 10 releases",
                "instructions": "Keep tennis ball tucked under chin to prevent head falling away to the off-side."
            }]
            wed_drills = [{
                "name": "Two-Parallel-Rope Runway Corridor Drill",
                "reps": "4 sets of 8 deliveries",
                "instructions": "Bowl within a 1.5-meter chalk corridor ensuring shoulders and hips stay aligned."
            }]
            fri_drills = [{
                "name": "Follow-Through 3-Step Hold Drill",
                "reps": "5 sets of 6 deliveries",
                "instructions": "Bowl and walk through delivery stride, freezing on 3rd step to confirm balance."
            }]
            sat_drills = [{
                "name": "Target Corridor Match Simulation",
                "reps": "5 overs of 6 balls",
                "instructions": "Bowl with match intensity off full run-up, keeping upright posture throughout."
            }]

        schedule = [
            DailyPracticeSession(
                day_number=1,
                day_name="Monday",
                focus_theme=f"{theme}: Neuromuscular Foundation",
                duration_minutes=20,
                is_rest_day=False,
                equipment_needed=["chalk", "tennis_ball"],
                drills=mon_drills,
                safety_reminder="Warm up shoulder rotator cuff and hamstring dynamic stretches before drills.",
                checklist_tasks=["10 min dynamic mobility", "Complete prescribed drill sets", "5 min cool-down"]
            ),
            DailyPracticeSession(
                day_number=2,
                day_name="Tuesday",
                focus_theme="Active Recovery & Mobility",
                duration_minutes=15,
                is_rest_day=True,
                equipment_needed=[],
                drills=[{"name": "Hip flexor and hamstring flexibility stretches", "reps": "3 sets of 30 sec holds", "instructions": "Gentle stretching to enhance stride mobility."}],
                safety_reminder="No fast bowling impact today. Let joints and tendons adapt to movement changes.",
                checklist_tasks=["15 min mobility session", "Stay hydrated"]
            ),
            DailyPracticeSession(
                day_number=3,
                day_name="Wednesday",
                focus_theme=f"{theme}: Short-Approach Repetition",
                duration_minutes=25,
                is_rest_day=False,
                equipment_needed=["chalk", "tennis_ball", "cone_or_brick"],
                drills=wed_drills,
                safety_reminder="Focus on form over bowling speed. Speed will naturally follow clean mechanics.",
                checklist_tasks=["5 min jogging warm-up", "Complete short-approach drills", "Check balance on each delivery"]
            ),
            DailyPracticeSession(
                day_number=4,
                day_name="Thursday",
                focus_theme="Core Strength & Rotational Stability",
                duration_minutes=20,
                is_rest_day=True,
                equipment_needed=[],
                drills=[{"name": "Plank holds + Single-leg balance holds", "reps": "3 sets of 45 sec", "instructions": "Strengthen trunk stabilizers for delivery stride impact."}],
                safety_reminder="Keep back straight during planks; stop if experiencing sharp lumbar discomfort.",
                checklist_tasks=["Core stability routine", "Review previous session video"]
            ),
            DailyPracticeSession(
                day_number=5,
                day_name="Friday",
                focus_theme=f"{theme}: Full Action Integration",
                duration_minutes=30,
                is_rest_day=False,
                equipment_needed=["tennis_ball", "chalk"],
                drills=fri_drills,
                safety_reminder="Ensure bowling shoes have adequate grip on the crease surface.",
                checklist_tasks=["Full warm-up", "Execute full run-up deliveries", "Record 2 reps on camera"]
            ),
            DailyPracticeSession(
                day_number=6,
                day_name="Saturday",
                focus_theme="Measurement & Weekly Progress Recording",
                duration_minutes=30,
                is_rest_day=False,
                equipment_needed=["smartphone_tripod", "tennis_ball"],
                drills=sat_drills,
                safety_reminder="Position camera side-on at 60fps for verification in CricVision.",
                checklist_tasks=["Record 5 test deliveries", "Upload to CricVision", "Compare with Monday baseline"]
            ),
            DailyPracticeSession(
                day_number=7,
                day_name="Sunday",
                focus_theme="Full Rest & Mental Recovery",
                duration_minutes=0,
                is_rest_day=True,
                equipment_needed=[],
                drills=[],
                safety_reminder="Rest is when your neuromuscular system consolidates new movement patterns.",
                checklist_tasks=["Rest and prepare for next week's training"]
            )
        ]

        return WeeklyTrainingPlan(
            player_name=player_name,
            target_flaw=primary_flaw,
            skill_level=skill_level,
            weekly_goal=goal,
            total_active_days=4,
            daily_schedule=schedule,
            equipment_checklist=["Tennis ball", "Chalk or tape", "Flat wall or brick", "Smartphone with camera"],
            coach_note=f"Focus this week exclusively on {theme}. Master one mechanical pillar before moving to the next."
        )
