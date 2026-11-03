"""
coaching/crictutor_llm.py - CricTutor AI Intelligence & Coaching Engine.

Features:
1. 100% Offline Mode:
   - Compiles full biomechanical diagnostics, tactical breakdown, and DIY drill prescriptions
     with zero network calls and zero external dependencies.
2. Optional Online Mode (Gemini API):
   - When internet is active and GEMINI_API_KEY is configured, sends the exact numerical metrics
     to Gemini 2.5 Flash to generate conversational coach dialogues in English, Urdu, or Roman Urdu.
"""

import os
import json
from typing import Dict, Any, List, Optional
import urllib.request
import urllib.error


class CricTutorAI:
    """
    Dual-mode cricket intelligence engine: Local Offline Rule Compiler + Optional Online LLM.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")

    def is_online_available(self) -> bool:
        """Checks if online LLM generation can be performed."""
        return bool(self.api_key and len(self.api_key.strip()) > 5)

    def generate_offline_bowling_report(
        self,
        player_name: str,
        style_name: str,
        matched_pro: str,
        scorecard: Any,
        icc_result: Any,
        speed_metrics: Any,
        pitch_metrics: Any,
        drills: List[Any],
        language: str = "English"
    ) -> str:
        """
        Generates an extensive, professional biomechanical report completely offline.
        """
        legal_badge = "✅ 100% LEGAL ACTION (ICC Law 21.2)" if icc_result.is_legal else "❌ ILLEGAL / CHUCKING FLAGGED"
        
        report = f"""# 🏏 CricTutor AI Coaching Report: {player_name}

### 📊 Performance Summary
* **Biomechanical Technique Score:** **{scorecard.total_score}/100** ({scorecard.grade_label})
* **Classified Bowling Archetype:** **{style_name}**
* **Benchmark International Pro:** **{matched_pro}**
* **ICC Legality Status:** **{legal_badge}**
* **Estimated Release Speed:** **{speed_metrics.release_speed_kmh} km/h** ({speed_metrics.release_speed_mph} mph)
* **Pitch Impact Zone:** **{pitch_metrics.length_zone} | {pitch_metrics.line_zone}** ({pitch_metrics.distance_from_crease_m}m from crease)

---

### 🔬 Biomechanical Breakdown
1. **Elbow Extension (ICC Law 21.2):** Measured at **{icc_result.elbow_extension_degrees}°** (Permissible threshold <= 15.0°).
   - *Analysis:* {icc_result.details}
2. **Kinetic Chain & Front Knee Plant:** Score: **{scorecard.kinetic_power_score}/25**.
   - Bracing the front knee delivers maximum catapult leverage into the release speed.
3. **Stability & Line Integrity:** Score: **{scorecard.stability_score}/25**.
   - Alignment of the head directly determines late swing and seam stability.
4. **Archetype Rhythmic DTW Match:** Score: **{scorecard.archetype_dtw_score}/25**.

---

### 🎯 Key Strengths
"""
        for s in scorecard.strengths:
            report += f"- ✅ {s}\n"

        report += "\n### ⚠️ Priority Mechanical Fixes\n"
        for imp in scorecard.areas_for_improvement:
            report += f"- ⚠️ {imp}\n"

        report += "\n---\n\n### 🛠️ Prescribed Grassroots DIY Drills (Zero Internet / Free)\n"
        for idx, drill in enumerate(drills, 1):
            report += f"#### Drill {idx}: {drill.drill_name}\n"
            report += f"* **Target Flaw:** {drill.flaw_name}\n"
            report += f"* **Biomechanical Cause:** {drill.biomechanical_cause}\n"
            report += f"* **Equipment:** {drill.equipment_needed}\n"
            report += f"* **Frequency:** {drill.reps_and_frequency}\n"
            report += "* **Execution Steps:**\n"
            for step in drill.step_by_step_instructions:
                report += f"  1. {step}\n"
            report += f"* **Pro Reference:** Model your rhythm after *{drill.pro_model_reference}*.\n\n"

        report += "\n> *Generated locally by CricTutor Offline Engine. 100% Free, Private, and Edge-Run.*"
        return report

    def generate_online_coaching_dialogue(
        self,
        player_name: str,
        style_name: str,
        matched_pro: str,
        scorecard: Any,
        icc_result: Any,
        speed_metrics: Any,
        language: str = "English"
    ) -> str:
        """
        Calls Gemini 2.5 Flash to produce a customized, masterclass coach monologue.
        Falls back to offline report if no network or API key.
        """
        if not self.is_online_available():
            return "Online CricTutor AI is currently running in Offline Mode. Switch to offline report for instant breakdown."

        system_prompt = (
            "You are CricTutor, a world-class, encouraging elite cricket coach and biomechanics specialist. "
            "You talk like a passionate mentor who has trained international cricket legends. "
            "Analyze the player's exact numbers, celebrate their strengths, provide tough-love constructive feedback "
            "on flaws, and explain the physics simply. "
        )

        if language == "Urdu":
            system_prompt += "Deliver your complete advice in authentic Urdu script with warmth and encouragement."
        elif language == "Roman Urdu":
            system_prompt += "Deliver your advice in natural Roman Urdu (conversational Urdu written in English alphabets like 'Ustad ji kehti hain...')."
        else:
            system_prompt += "Deliver your advice in inspiring, direct English."

        user_content = f"""
        Player: {player_name}
        Overall Biomechanics Score: {scorecard.total_score}/100
        Bowling Style: {style_name}
        Matched Legend: {matched_pro}
        Elbow Extension: {icc_result.elbow_extension_degrees} degrees (ICC 15° limit)
        Legal Action: {icc_result.is_legal}
        Release Pace: {speed_metrics.release_speed_kmh} km/h
        Key Observations: {', '.join(scorecard.strengths + scorecard.areas_for_improvement)}

        Please provide your interactive coaching debrief.
        """

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={self.api_key}"
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": system_prompt + "\n\n" + user_content}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 800
            }
        }

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=8) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                text = res_data["candidates"][0]["content"]["parts"][0]["text"]
                return text
        except Exception as e:
            return f"(Online CricTutor connection timed out. Falling back to Offline Engine). Error: {str(e)}"
