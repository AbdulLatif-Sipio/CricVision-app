# 🏏 CricVision & CricTutor AI
### *Grassroots AI Cricket Biomechanics, Legality & Coaching Platform*

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://python.org)
[![Offline-First](https://img.shields.io/badge/Edge%20Inference-100%25%20Offline-success.svg)](#)
[![ICC Compliance](https://img.shields.io/badge/ICC%20Law%2021.2-15%C2%B0%20Compliant-red.svg)](#)
[![MediaPipe](https://img.shields.io/badge/Pose%20Tracking-MediaPipe%2033--Landmarks-orange.svg)](#)

---

## 🎯 The Core Thesis & Strategic Wedge

Every commercial cricket AI coaching platform (CricVision.ai, SportsReflector, Matcha) sells **expensive monthly subscriptions and academy licenses** requiring cloud connectivity and expensive infrastructure. 

**None are built for a kid bowling in a gully or on a school ground with no Wi-Fi and zero budget.**

**CricVision & CricTutor AI** is intentionally engineered as an **offline-first, 100% free, edge-native biomechanics laboratory**. It turns a standard phone or laptop camera into an elite cricket laboratory capable of verifying ICC legal compliance, analyzing multi-style bowling mechanics, estimating pace, and prescribing grassroots DIY drills using tennis balls, chalk lines, and brick walls.

---

## 🔬 System Architecture

```
                                  [ CRICVISION / CRICTUTOR ]
                                              │
         ┌────────────────────────────────────┼────────────────────────────────────┐
         ▼                                    ▼                                    ▼
1. BIOMECHANICS ENGINE               2. COMPUTER VISION & PHYSICS         3. ML, ICC & ARCHETYPES
 • MediaPipe 33 3D Pose Landmarks     • Metric Homography Calibration      • ICC Law 21.2 (15° Chucking)
 • Vector Geometry & Euler Angles     • Speed Gun (km/h & mph)             • 20+ International Player Styles
 • Automatic Phase Segmentation       • 2D Pitch Landing Heatmap           • Dynamic Time Warping (DTW)
   (Gather -> BFC -> FFC -> Release)  • Optical Flow Bat Swing Velocity    • 100-Point Biomechanical Index
                                              │
                                              ▼
                                   4. DUAL-MODE CRICTUTOR AI
                      ┌───────────────────────┴───────────────────────┐
                      ▼                                               ▼
             100% OFFLINE ENGINE                             ONLINE CRICTUTOR LLM
    • Zero-Internet Deterministic Matrix            • Optional Gemini 2.5 Flash Enhancer
    • DIY Grassroots Drills (Tennis balls, chalk)   • Multilingual (English / Urdu / Roman Urdu)
```

---

## 🌟 Key Features

### 1. ⚖️ Official ICC Law 21.2 Legality (Chucking Checker)
* Measures elbow angle from the moment the arm reaches horizontal shoulder level to the instant of ball release.
* Enforces the official **$15.0^\circ$ tolerance threshold**. Flags illegal dynamic straightening with millidegree precision.

### 2. 🌍 Diverse Bowling Style Archetypes (20+ Legends)
Prevents false "invalid angle" errors by recognizing that cricket actions are diverse:
* **Pure Side-On:** James Anderson, Dale Steyn, Dennis Lillee
* **Semi-Open / Hybrid:** Pat Cummins, Glenn McGrath, Kagiso Rabada
* **Pure Front-On / Chest-On:** Shoaib Akhtar, Malcolm Marshall, Makhaya Ntini
* **Sling / Round-Arm:** Lasith Malinga, Matheesha Pathirana, Shaun Tait (celebrated as 100% legal, not penalized!)
* **High-Arm Upright:** Shaheen Shah Afridi, Curtly Ambrose, Morne Morkel
* **Unique Hyperextension:** Jasprit Bumrah
* **Spin Archetypes:** Ravichandran Ashwin, Nathan Lyon, Rashid Khan, Shane Warne

### 3. 🎯 Computer Vision Speed Gun & Pitch Radar
* **Release Speed Estimator:** Computes real-world velocity in **km/h and mph** using metric camera perspective homography.
* **2D Pitch Map:** Visualizes landing impact zones (*Yorker, Full, Good Length, Back of Length, Short*) and off-stump channels.

### 4. 🏏 Batting Stroke Academy
* Deep pose stroke classification (*Cover Drive, Pull Shot, Straight Drive, Cut, Sweep*).
* Evaluates classic high leading elbow elevation ($>130^\circ$) and head-over-ball weight transfer.
* Bat swing speed tracker using dense optical flow.

### 5. 🛠️ 100% Offline DIY Grassroots Drill Matrix
Prescribes free DIY drills using accessible street materials:
* *Wall-Slide Drill* for straight-arm muscle memory.
* *Cone Bracing Drill* for front knee leverage.
* *Chin-Tuck Drill* for head stability.
* *Hanging Ball Drill* for high leading elbow drives.

### 6. ☁️ Google Colab Cloud GPU Training Pipeline
Solves the heavy data problem:
* Ingests gigabytes of match broadcasts in the cloud using free T4/A100 GPUs.
* Trains deep temporal models and exports lightweight `.onnx` and `.npz` files for local offline execution.
* Ready-to-run notebook at: `notebooks/cricvision_colab_trainer.ipynb`.

---

## 🚀 Quickstart & Local Setup

### 1. Clone & Enter Directory
```bash
cd c:\Users\Admin\Desktop\CricVision
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch CricVision Dashboard (FastAPI Web + PWA)
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
Open **`http://localhost:8000`** in your browser to access the 5-tab athletic UI. On mobile, tap **"📲 Install App"** to add it directly to your home screen as a standalone app!

### 4. Or Launch Streamlit Dashboard
```bash
streamlit run app.py
```

### 5. Production Docker Deployment
```bash
docker compose up --build
```

### 6. Test Sample Clips
The repository includes 5 pre-rendered benchmark clips in `samples/`:
- `samples/sample_legal_side_on.mp4` (Textbook Pat Cummins / Anderson pace)
- `samples/sample_illegal_chucking.mp4` (Visible 35° elbow snap triggering ICC Law 21.2)
- `samples/sample_sling_malinga.mp4` (Authentic Lasith Malinga round-arm sling)
- `samples/sample_cover_drive.mp4` (Virat Kohli / Babar Azam cover drive)
- `samples/sample_pull_shot.mp4` (Rohit Sharma pull shot)

---

## 🔒 Proving the Offline Claim
To demonstrate zero-internet execution on stage or during live review:
1. Open the sidebar in the app.
2. Ensure **"Offline-Only Mode (Zero Cloud)"** is enabled.
3. Turn off your Wi-Fi or disconnect Ethernet.
4. Run full analysis — the entire pipeline (MediaPipe, DTW, ICC rule engine, Speed Gun, and Drill prescriber) runs with zero latency!
