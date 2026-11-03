import React, { useState, useEffect } from 'react';

export default function App() {
  const [activeTab, setActiveTab] = useState('home');
  const [activityMode, setActivityMode] = useState('Bowling');
  const [selectedFile, setSelectedFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState('Analyzing biomechanics...');
  const [analysisData, setAnalysisData] = useState(null);
  const [history, setHistory] = useState([]);

  // Comparison State
  const [compSessionA, setCompSessionA] = useState('');
  const [compSessionB, setCompSessionB] = useState('');
  const [compResult, setCompResult] = useState(null);
  const [comparing, setComparing] = useState(false);

  // Multi-delivery State
  const [multiResult, setMultiResult] = useState(null);

  // Weekly Planner State
  const [planFlaw, setPlanFlaw] = useState('collapsed_front_knee');
  const [planLevel, setPlanLevel] = useState('Intermediate');
  const [weeklyPlan, setWeeklyPlan] = useState(null);

  // Chat State
  const [chatMessages, setChatMessages] = useState([
    { sender: 'CricTutor', text: "Salaam! I'm your offline-first cricket biomechanics coach. Ask me about your bowling action, pace generation, or drill technique!" }
  ]);
  const [chatInput, setChatInput] = useState('');

  // Modals
  const [showFeedbackModal, setShowFeedbackModal] = useState(false);
  const [showCameraModal, setShowCameraModal] = useState(false);
  const [feedbackText, setFeedbackText] = useState('');

  useEffect(() => {
    fetchHistory();
  }, []);

  const fetchHistory = async () => {
    try {
      const res = await fetch('/api/history');
      if (res.ok) {
        const list = await res.json();
        setHistory(list);
        if (list.length >= 2) {
          setCompSessionA(list[1].session_id);
          setCompSessionB(list[0].session_id);
        }
      }
    } catch (err) {
      console.error('History fetch failed:', err);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      setSelectedFile(file);
      triggerAnalysis(null, file, activityMode);
    }
  };

  const loadPreset = async (presetPath, activity) => {
    setActivityMode(activity);
    setLoading(true);
    setLoadingMessage('Loading broadcast sample video...');
    try {
      const resp = await fetch('/' + presetPath);
      const blob = await resp.blob();
      const fname = presetPath.split('/').pop();
      const file = new File([blob], fname, { type: 'video/mp4' });
      await triggerAnalysis(null, file, activity);
    } catch (err) {
      alert('Failed to load sample preset: ' + err.message);
      setLoading(false);
    }
  };

  const triggerAnalysis = async (presetPath, fileObj, activity) => {
    setLoading(true);
    setLoadingMessage('Extracting 33 3D Pose Landmarks (MediaPipe)...');
    setActiveTab('analyze');

    const formData = new FormData();
    formData.append('activity_type', activity || activityMode);
    formData.append('player_name', 'Grassroots Player');

    try {
      if (fileObj) {
        formData.append('file', fileObj);
      } else {
        alert('Please choose a video or select a preset.');
        setLoading(false);
        return;
      }

      const res = await fetch('/api/analyze/upload', {
        method: 'POST',
        body: formData
      });

      if (!res.ok) {
        const err = await res.json();
        alert('Analysis Error: ' + (err.message || 'Processing failed.'));
        return;
      }

      const data = await res.json();
      setAnalysisData(data);
      fetchHistory();
    } catch (err) {
      alert('Connection failed: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const executeComparison = async () => {
    if (!compSessionA || !compSessionB) {
      alert('Please select both Session A and Session B.');
      return;
    }
    setComparing(true);
    try {
      const res = await fetch('/api/sessions/compare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_a_id: compSessionA, session_b_id: compSessionB })
      });
      if (!res.ok) throw new Error('Comparison failed');
      const data = await res.json();
      setCompResult(data);
    } catch (err) {
      alert('Comparison error: ' + err.message);
    } finally {
      setComparing(false);
    }
  };

  const runSampleMultiDelivery = async () => {
    const sampleDeliveries = [
      { delivery_number: 1, elbow_extension_deg: 7.0, front_knee_angle_deg: 168.0, trunk_tilt_deg: 12.0, release_speed_kmh: 122.0 },
      { delivery_number: 2, elbow_extension_deg: 8.0, front_knee_angle_deg: 166.0, trunk_tilt_deg: 13.0, release_speed_kmh: 123.0 },
      { delivery_number: 3, elbow_extension_deg: 7.5, front_knee_angle_deg: 167.0, trunk_tilt_deg: 12.5, release_speed_kmh: 121.5 },
      { delivery_number: 4, elbow_extension_deg: 8.2, front_knee_angle_deg: 135.0, trunk_tilt_deg: 13.0, release_speed_kmh: 114.0 }
    ];
    try {
      const res = await fetch('/api/analyze/multi-delivery', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ player_name: 'Grassroots Bowler', deliveries: sampleDeliveries })
      });
      const data = await res.json();
      setMultiResult(data);
    } catch (err) {
      alert('Multi-delivery error: ' + err.message);
    }
  };

  const generateWeeklyPlan = async () => {
    try {
      const res = await fetch('/api/tutor/weekly-plan', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ player_name: 'Player', primary_flaw: planFlaw, skill_level: planLevel })
      });
      const data = await res.json();
      setWeeklyPlan(data);
    } catch (err) {
      alert('Weekly plan error: ' + err.message);
    }
  };

  const sendChatMessage = async () => {
    if (!chatInput.trim()) return;
    const msg = chatInput.trim();
    setChatMessages(prev => [...prev, { sender: 'You', text: msg }]);
    setChatInput('');

    try {
      const res = await fetch('/api/coaching/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: msg, language: 'English' })
      });
      const data = await res.json();
      setChatMessages(prev => [...prev, { sender: 'CricTutor', text: data.reply }]);
    } catch (err) {
      setChatMessages(prev => [...prev, { sender: 'CricTutor', text: 'Error connecting to offline coach.' }]);
    }
  };

  const speakMainObservation = () => {
    if (!('speechSynthesis' in window) || !analysisData?.primary_observation) {
      alert('Text-to-speech audio is not supported in this browser.');
      return;
    }
    const obs = analysisData.primary_observation;
    const textToSpeak = `Observation: ${obs.main_observation}. What this means: ${obs.what_this_means}.`;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(textToSpeak);
    utterance.rate = 0.95;
    window.speechSynthesis.speak(utterance);
  };

  const submitFeedback = async () => {
    if (!analysisData?.primary_observation) return;
    const obsId = analysisData.primary_observation.observation_id;
    try {
      await fetch(`/api/observations/${obsId}/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dismiss: true, user_correction: feedbackText })
      });
      alert('Feedback submitted. The observation has been dismissed.');
      setShowFeedbackModal(false);
      setFeedbackText('');
    } catch (err) {
      alert('Error submitting feedback: ' + err.message);
    }
  };

  const deleteSessionVideo = async (sessionId) => {
    if (!confirm('Delete raw video file to free up local storage? Your metrics and coaching reports will be preserved.')) return;
    try {
      const res = await fetch(`/api/sessions/${sessionId}/video`, { method: 'DELETE' });
      const data = await res.json();
      alert(data.message || 'Video deleted.');
      fetchHistory();
    } catch (err) {
      alert('Deletion error: ' + err.message);
    }
  };

  return (
    <div className="min-h-screen bg-[#070B09] text-gray-100 flex flex-col justify-between selection:bg-lime-500 selection:text-black">
      {/* Top Header */}
      <header className="border-b border-[#283D31]/60 bg-[#0D1410]/90 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex items-center justify-between py-3">
          <div className="flex items-center space-x-3 cursor-pointer" onClick={() => setActiveTab('home')}>
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-lime-600 to-lime-400 flex items-center justify-center font-black text-xl shadow-lg shadow-lime-500/20 text-[#070B09]">
              🏏
            </div>
            <div>
              <h1 className="text-xl font-black tracking-tight text-white flex items-center gap-2">
                CRICVISION <span className="text-[10px] px-2 py-0.5 rounded-full bg-lime-500/20 text-lime-400 font-extrabold border border-lime-500/30">REACT 2.0</span>
              </h1>
              <p className="text-[11px] text-gray-400 font-medium">Analyze. Train. Improve.</p>
            </div>
          </div>

          {/* 5 Main Tabs (Desktop) */}
          <nav className="hidden md:flex items-center space-x-1 bg-[#131D17] px-2 py-1 rounded-xl border border-[#283D31]/60 text-xs font-bold">
            {['home', 'analyze', 'progress', 'tutor', 'profile'].map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`px-4 py-2 rounded-lg transition capitalize flex items-center gap-1.5 ${
                  activeTab === tab
                    ? 'text-lime-400 bg-lime-500/10 border-b-2 border-lime-500'
                    : 'text-gray-300 hover:text-white'
                }`}
              >
                {tab === 'home' && '🏠 Home'}
                {tab === 'analyze' && '🎥 Analyze'}
                {tab === 'progress' && '📈 Progress'}
                {tab === 'tutor' && '🏏 Tutor'}
                {tab === 'profile' && '👤 Profile'}
              </button>
            ))}
          </nav>

          <div className="flex items-center space-x-3">
            <div className="hidden sm:flex items-center space-x-2 bg-[#131D17] px-3 py-1.5 rounded-lg border border-[#283D31]/60 text-xs font-semibold">
              <span className="w-2 h-2 rounded-full bg-lime-400 animate-pulse"></span>
              <span className="text-gray-300">100% Offline Edge</span>
            </div>
            <button
              onClick={() => setActiveTab('analyze')}
              className="px-3.5 py-1.5 rounded-lg bg-lime-500 hover:bg-lime-400 text-[#070B09] font-black text-xs transition shadow-md shadow-lime-500/20"
            >
              + New Video
            </button>
          </div>
        </div>
      </header>

      {/* Main Tab Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 w-full flex-grow">

        {/* TAB 1: HOME */}
        {activeTab === 'home' && (
          <div className="space-y-6">
            <div className="bg-[#131D17]/80 border border-lime-500/20 rounded-3xl p-6 sm:p-8 relative overflow-hidden shadow-2xl">
              <div className="relative z-10 max-w-2xl">
                <span className="inline-block px-3 py-1 rounded-full bg-lime-500/10 text-lime-400 border border-lime-500/30 text-xs font-extrabold mb-3">
                  ⚡ On-Device Biomechanics Laboratory
                </span>
                <h2 className="text-3xl sm:text-4xl font-black text-white leading-tight">
                  Analyze your action.<br /><span className="text-lime-400">Fix the #1 flaw. Measure progress.</span>
                </h2>
                <p className="text-sm text-gray-300 mt-2 mb-6">
                  No subscriptions or expensive club licenses. Screen your action under ICC guidelines, get practical DIY drills, and track delivery-to-delivery gains.
                </p>
                <div className="flex flex-wrap items-center gap-3">
                  <button
                    onClick={() => setActiveTab('analyze')}
                    className="px-6 py-3 rounded-xl bg-lime-500 hover:bg-lime-400 text-[#070B09] font-black text-sm transition shadow-lg shadow-lime-500/25"
                  >
                    🎥 Analyze a Video Now
                  </button>
                  <button
                    onClick={() => setShowCameraModal(true)}
                    className="px-4 py-3 rounded-xl bg-[#1B2921] hover:bg-[#283D31] text-gray-200 border border-[#283D31] font-bold text-xs transition"
                  >
                    📐 Camera Setup Guide
                  </button>
                </div>
              </div>
            </div>

            {/* Quick Stats Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <div className="bg-[#131D17]/80 border border-[#283D31] rounded-2xl p-5 border-l-4 border-l-lime-500">
                <p className="text-xs font-bold text-gray-400 uppercase">Training Consistency</p>
                <p className="text-3xl font-black text-white mt-1">4 <span className="text-xs text-lime-400 font-bold">days active</span></p>
                <p className="text-[11px] text-gray-400 mt-1">Weekly practice momentum</p>
              </div>
              <div className="bg-[#131D17]/80 border border-[#283D31] rounded-2xl p-5 border-l-4 border-l-cyan-500">
                <p className="text-xs font-bold text-gray-400 uppercase">Focus Drill</p>
                <p className="text-base font-bold text-cyan-300 mt-1">Rigid Wall-Slide Rotation</p>
                <button onClick={() => setActiveTab('tutor')} className="text-xs text-cyan-400 hover:underline font-bold mt-1 inline-block">View 7-day schedule →</button>
              </div>
              <div className="bg-[#131D17]/80 border border-[#283D31] rounded-2xl p-5 border-l-4 border-l-amber-500">
                <p className="text-xs font-bold text-gray-400 uppercase">Technique Delta</p>
                <p className="text-base font-bold text-amber-300 mt-1">Session A vs. Session B</p>
                <button onClick={() => setActiveTab('progress')} className="text-xs text-amber-400 hover:underline font-bold mt-1 inline-block">Compare Before & After →</button>
              </div>
            </div>

            {/* Recent Sessions */}
            <div className="bg-[#131D17]/80 border border-[#283D31] rounded-2xl p-6">
              <h3 className="text-base font-bold text-white mb-3">Recent Analytical Sessions</h3>
              <div className="space-y-2">
                {history.slice(0, 3).map((s) => (
                  <div key={s.session_id} className="flex items-center justify-between p-3 rounded-xl bg-[#0D1410] border border-[#1B2921] text-xs">
                    <div>
                      <span className="font-bold text-white">{s.activity_type} Session</span>
                      <span className="text-gray-400 ml-2">Score: {s.accuracy_score}/100</span>
                    </div>
                    <div className="flex items-center gap-3">
                      <span className={`font-bold ${s.is_legal ? 'text-lime-400' : 'text-red-400'}`}>
                        {s.is_legal ? 'Legal' : 'Screening Alert'}
                      </span>
                      <span className="text-gray-400">{s.created_at || 'Recent'}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* TAB 2: ANALYZE */}
        {activeTab === 'analyze' && (
          <div className="space-y-6">
            <div className="bg-[#131D17]/80 border border-[#283D31] rounded-2xl p-6">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-4">
                <div>
                  <h2 className="text-xl font-black text-white">Video Analysis & Screening</h2>
                  <p className="text-xs text-gray-400">Upload your video to extract biomechanical angles and receive structured coaching.</p>
                </div>
                <div className="flex items-center bg-[#0D1410] p-1 rounded-xl border border-[#283D31] text-xs font-bold">
                  <button
                    onClick={() => setActivityMode('Bowling')}
                    className={`px-4 py-1.5 rounded-lg transition ${activityMode === 'Bowling' ? 'bg-lime-500 text-[#070B09] font-black' : 'text-gray-300'}`}
                  >
                    Bowling Action
                  </button>
                  <button
                    onClick={() => setActivityMode('Batting')}
                    className={`px-4 py-1.5 rounded-lg transition ${activityMode === 'Batting' ? 'bg-lime-500 text-[#070B09] font-black' : 'text-gray-300'}`}
                  >
                    Batting Stroke
                  </button>
                </div>
              </div>

              {/* Upload Dropzone */}
              <div className="border-2 border-dashed border-[#283D31] hover:border-lime-500/60 rounded-2xl p-8 text-center transition cursor-pointer bg-[#0D1410]/50 relative">
                <input
                  type="file"
                  accept="video/mp4,video/mov,image/jpeg,image/png"
                  onChange={handleFileChange}
                  className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                />
                <div className="space-y-2">
                  <div className="w-12 h-12 rounded-2xl bg-[#1B2921] border border-[#283D31] flex items-center justify-center mx-auto text-xl text-lime-400">
                    📁
                  </div>
                  <p className="text-sm font-bold text-white">Click or drag & drop video here</p>
                  <p className="text-xs text-gray-400">MP4, MOV up to 50MB (60fps side-on recommended)</p>
                </div>
              </div>

              {/* Quick Preset Buttons */}
              <div className="mt-4 pt-4 border-t border-[#283D31]/60 flex flex-wrap items-center gap-2 text-xs">
                <span className="text-gray-400 font-semibold mr-1">Quick Sample Clips:</span>
                <button onClick={() => loadPreset('samples/sample_legal_side_on.mp4', 'Bowling')} className="px-3 py-1.5 rounded-lg bg-[#1B2921] hover:bg-[#283D31] border border-[#283D31] text-gray-200 transition">
                  🟢 Legal Side-On Pace
                </button>
                <button onClick={() => loadPreset('samples/sample_illegal_chucking.mp4', 'Bowling')} className="px-3 py-1.5 rounded-lg bg-[#1B2921] hover:bg-[#283D31] border border-[#283D31] text-gray-200 transition">
                  🔴 Chucking Screening (Elbow Flex)
                </button>
                <button onClick={() => loadPreset('samples/sample_sling_malinga.mp4', 'Bowling')} className="px-3 py-1.5 rounded-lg bg-[#1B2921] hover:bg-[#283D31] border border-[#283D31] text-gray-200 transition">
                  ⚡ Round-Arm Slinger (Malinga)
                </button>
                <button onClick={() => loadPreset('samples/sample_cover_drive.mp4', 'Batting')} className="px-3 py-1.5 rounded-lg bg-[#1B2921] hover:bg-[#283D31] border border-[#283D31] text-gray-200 transition">
                  🏏 Cover Drive (Batting)
                </button>
              </div>
            </div>

            {loading && (
              <div className="bg-[#131D17]/80 border border-[#283D31] rounded-2xl p-8 text-center space-y-3">
                <div className="w-10 h-10 border-4 border-[#283D31] border-t-lime-500 rounded-full animate-spin mx-auto"></div>
                <h3 className="text-base font-bold text-white">{loadingMessage}</h3>
                <p className="text-xs text-gray-400">Tracking 33 3D joints and computing kinematics...</p>
              </div>
            )}

            {/* Results Hierarchy */}
            {analysisData && !loading && (
              <div className="space-y-6">
                {/* 1. Primary Observation Card */}
                {analysisData.primary_observation && (
                  <div className="bg-[#131D17]/90 border-l-8 border-l-lime-500 rounded-3xl p-6 border border-[#283D31] shadow-xl">
                    <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
                      <span className="px-3 py-1 rounded-full text-xs font-black bg-lime-500 text-[#070B09] uppercase tracking-wider">
                        🎯 Single Key Technique Priority
                      </span>
                      <div className="flex items-center gap-2">
                        <span className="px-2.5 py-0.5 rounded-full text-xs font-bold bg-lime-500/10 text-lime-400 border border-lime-500/30">
                          {analysisData.primary_observation.confidence_label.toUpperCase()}
                        </span>
                        <button onClick={speakMainObservation} className="px-2.5 py-1 rounded-lg bg-[#1B2921] hover:bg-[#283D31] text-xs font-bold text-lime-400 flex items-center gap-1">
                          <span>🔊</span> Listen
                        </button>
                      </div>
                    </div>

                    <h3 className="text-xl sm:text-2xl font-black text-white">
                      {analysisData.primary_observation.main_observation}
                    </h3>
                    <p className="text-sm text-gray-300 mt-2">
                      <strong className="text-lime-400">What This Means:</strong> {analysisData.primary_observation.what_this_means}
                    </p>

                    {/* Prescribed Drill */}
                    <div className="mt-4 p-4 rounded-2xl bg-[#0D1410] border border-[#283D31]">
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-xs font-extrabold text-lime-400 uppercase">🛠️ Prescribed Drill</span>
                        <span className="text-xs text-gray-400">3 sets of 10 deliveries</span>
                      </div>
                      <h4 className="text-base font-bold text-white">{analysisData.primary_observation.drill_name}</h4>
                      <p className="text-xs text-gray-400 mt-0.5">Equipment: {analysisData.primary_observation.drill_equipment}</p>
                      <p className="text-[11px] text-amber-300/90 mt-2 font-medium">{analysisData.primary_observation.safety_guidance}</p>
                    </div>

                    <div className="mt-4 flex flex-wrap items-center justify-between gap-3 pt-3 border-t border-[#283D31]/60 text-xs">
                      <div>
                        <span className="text-gray-400">Next session goal: </span>
                        <strong className="text-white">{analysisData.primary_observation.next_session_focus}</strong>
                      </div>
                      <button onClick={() => setShowFeedbackModal(true)} className="text-xs text-gray-400 hover:text-white underline">
                        Dismiss or correct feedback
                      </button>
                    </div>
                  </div>
                )}

                {/* Evidence Grid */}
                <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                  <div className="lg:col-span-5 bg-[#131D17]/80 border border-[#283D31] rounded-2xl p-5 flex flex-col justify-between">
                    <div>
                      <div className="flex items-center justify-between mb-3">
                        <h4 className="text-xs font-bold text-white uppercase tracking-wider">📸 Biomechanical Evidence</h4>
                        <span className={`text-xs px-2.5 py-0.5 rounded-full font-black ${analysisData.is_legal ? 'bg-lime-500/20 text-lime-400 border border-lime-500/30' : 'bg-red-500/20 text-red-400 border border-red-500/30'}`}>
                          {analysisData.icc_status}
                        </span>
                      </div>
                      <div className="aspect-video bg-black/60 rounded-xl overflow-hidden border border-[#283D31] flex items-center justify-center">
                        {analysisData.annotated_media_b64 ? (
                          <img src={`data:image/jpeg;base64,${analysisData.annotated_media_b64}`} alt="Pose Overlay" className="w-full h-full object-contain" />
                        ) : (
                          <span className="text-xs text-gray-500">Pose image available</span>
                        )}
                      </div>
                    </div>
                    <div className="mt-4 pt-3 border-t border-[#283D31]/60 text-xs text-gray-400 flex justify-between">
                      <span>Elbow: <strong className="text-white">{analysisData.elbow_extension_deg}°</strong></span>
                      <span>Front Knee: <strong className="text-white">{analysisData.front_knee_angle_deg}°</strong></span>
                      <span>ICC Tolerance: <strong className="text-lime-400">15.0°</strong></span>
                    </div>
                  </div>

                  <div className="lg:col-span-7 grid grid-cols-2 gap-4">
                    <div className="bg-[#131D17]/80 border border-[#283D31] rounded-xl p-4 border-l-4 border-l-lime-500">
                      <p className="text-xs font-bold text-gray-400 uppercase">Elbow Extension</p>
                      <p className="text-2xl font-black text-white mt-1">{analysisData.elbow_extension_deg}°</p>
                      <p className="text-[11px] text-lime-400 font-bold">{analysisData.icc_status}</p>
                    </div>
                    <div className="bg-[#131D17]/80 border border-[#283D31] rounded-xl p-4 border-l-4 border-l-cyan-500">
                      <p className="text-xs font-bold text-gray-400 uppercase">Release Speed</p>
                      <p className="text-2xl font-black text-cyan-400 mt-1">
                        {analysisData.release_speed_kmh > 0 ? `${analysisData.release_speed_kmh} km/h` : 'Speed Unavailable'}
                      </p>
                      <p className="text-[11px] text-gray-400 mt-0.5">
                        {analysisData.release_speed_kmh > 0 ? `${analysisData.release_speed_mph} mph` : 'Calibrated pitch required'}
                      </p>
                    </div>
                    <div className="bg-[#131D17]/80 border border-[#283D31] rounded-xl p-4 border-l-4 border-l-amber-500">
                      <p className="text-xs font-bold text-gray-400 uppercase">Twin Legend</p>
                      <p className="text-lg font-bold text-amber-300 mt-1 truncate">{analysisData.matched_legend}</p>
                      <p className="text-[11px] text-gray-400 mt-0.5 truncate">{analysisData.style_category}</p>
                    </div>
                    <div className="bg-[#131D17]/80 border border-[#283D31] rounded-xl p-4 border-l-4 border-l-indigo-500">
                      <p className="text-xs font-bold text-gray-400 uppercase">Pitch Zone</p>
                      <p className="text-base font-bold text-indigo-300 mt-1 truncate">{analysisData.pitch_zone}</p>
                      <p className="text-[11px] text-gray-400 mt-0.5">Target channel</p>
                    </div>
                    <div className="col-span-2 bg-[#0D1410] rounded-xl p-3 border border-[#283D31] text-[11px] text-gray-400">
                      <strong>Note:</strong> {analysisData.screening_disclaimer}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* TAB 3: PROGRESS & COMPARISON */}
        {activeTab === 'progress' && (
          <div className="space-y-6">
            <div className="bg-[#131D17]/80 border border-[#283D31] rounded-3xl p-6 shadow-xl">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
                <div>
                  <h2 className="text-xl font-black text-white">Before & After Session Comparator</h2>
                  <p className="text-xs text-gray-400">Verify whether drills are moving your numbers in the right direction.</p>
                </div>
                <button
                  onClick={executeComparison}
                  disabled={comparing}
                  className="px-4 py-2 rounded-xl bg-lime-500 hover:bg-lime-400 text-[#070B09] font-black text-xs transition"
                >
                  {comparing ? 'Comparing...' : 'Compare Sessions'}
                </button>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div>
                  <label className="block text-xs font-bold text-gray-400 mb-1">Session A (Baseline):</label>
                  <select
                    value={compSessionA}
                    onChange={(e) => setCompSessionA(e.target.value)}
                    className="w-full bg-[#0D1410] border border-[#283D31] rounded-xl px-3 py-2 text-xs text-white"
                  >
                    <option value="">Select Baseline Session A...</option>
                    {history.map((s) => (
                      <option key={s.session_id} value={s.session_id}>
                        {s.created_at || s.session_id.slice(0, 8)} — {s.activity_type} ({s.accuracy_score} pts)
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-xs font-bold text-gray-400 mb-1">Session B (Current):</label>
                  <select
                    value={compSessionB}
                    onChange={(e) => setCompSessionB(e.target.value)}
                    className="w-full bg-[#0D1410] border border-[#283D31] rounded-xl px-3 py-2 text-xs text-white"
                  >
                    <option value="">Select Current Session B...</option>
                    {history.map((s) => (
                      <option key={s.session_id} value={s.session_id}>
                        {s.created_at || s.session_id.slice(0, 8)} — {s.activity_type} ({s.accuracy_score} pts)
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              {compResult && (
                <div className="mt-6 p-5 rounded-2xl bg-[#0D1410] border border-[#283D31] space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-xs px-3 py-1 rounded-full font-black bg-lime-500 text-[#070B09]">
                      {compResult.overall_progress_status.replace('_', ' ')}
                    </span>
                    <span className="text-xs text-gray-400">{compResult.days_between} days between sessions</span>
                  </div>
                  <h4 className="text-base font-bold text-white">{compResult.coach_verdict}</h4>
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                    <div className="bg-[#131D17] rounded-xl p-3 border-l-4 border-l-lime-500">
                      <p className="text-[11px] font-bold text-gray-400">Front Knee Bracing</p>
                      <p className="text-lg font-black text-lime-400 mt-0.5">
                        {compResult.metric_deltas.front_knee?.delta_value >= 0 ? '+' : ''}{compResult.metric_deltas.front_knee?.delta_value}°
                      </p>
                    </div>
                    <div className="bg-[#131D17] rounded-xl p-3 border-l-4 border-l-cyan-500">
                      <p className="text-[11px] font-bold text-gray-400">Elbow Straightening</p>
                      <p className="text-lg font-black text-cyan-400 mt-0.5">
                        {compResult.metric_deltas.elbow_extension?.delta_value <= 0 ? '' : '+'}{compResult.metric_deltas.elbow_extension?.delta_value}°
                      </p>
                    </div>
                    <div className="bg-[#131D17] rounded-xl p-3 border-l-4 border-l-amber-500">
                      <p className="text-[11px] font-bold text-gray-400">Scorecard Delta</p>
                      <p className="text-lg font-black text-amber-300 mt-0.5">
                        {compResult.scorecard_delta >= 0 ? '+' : ''}{compResult.scorecard_delta} pts
                      </p>
                    </div>
                  </div>
                  <ul className="text-xs text-gray-300 list-disc list-inside space-y-0.5">
                    {compResult.key_improvements.map((imp, i) => (
                      <li key={i}>{imp}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>

            {/* Multi-Delivery Spell Section */}
            <div className="bg-[#131D17]/80 border border-[#283D31] rounded-3xl p-6">
              <div className="flex items-center justify-between mb-3">
                <div>
                  <h3 className="text-lg font-black text-white">Multi-Delivery Spell Repeatability</h3>
                  <p className="text-xs text-gray-400">Measure action consistency across 4-6 balls in an over.</p>
                </div>
                <button
                  onClick={runSampleMultiDelivery}
                  className="px-3 py-1.5 rounded-lg bg-[#1B2921] hover:bg-[#283D31] border border-[#283D31] text-xs font-bold"
                >
                  Run Sample 4-Ball Net Spell
                </button>
              </div>

              {multiResult ? (
                <div className="p-4 rounded-xl bg-[#0D1410] border border-[#283D31] space-y-2 text-xs">
                  <div className="flex justify-between items-center">
                    <span className="font-bold text-lime-400">{multiResult.repeatability_label}</span>
                    <span className="font-bold text-white">Score: {multiResult.overall_repeatability_score}/100</span>
                  </div>
                  <p className="text-gray-200"><strong>Inconsistency:</strong> {multiResult.primary_inconsistency}</p>
                  {multiResult.fatigue_drift_detected && (
                    <p className="text-amber-300">⚠️ {multiResult.drift_explanation}</p>
                  )}
                </div>
              ) : (
                <p className="text-xs text-gray-400 py-2">Click 'Run Sample 4-Ball Net Spell' to analyze repeatability.</p>
              )}
            </div>

            {/* Session Table */}
            <div className="bg-[#131D17]/80 border border-[#283D31] rounded-2xl p-6">
              <h3 className="text-base font-bold text-white mb-3">Complete Saved Sessions</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead>
                    <tr className="border-b border-[#283D31] text-gray-400 font-semibold">
                      <th className="pb-2">Date</th>
                      <th className="pb-2">Activity</th>
                      <th className="pb-2">Status</th>
                      <th className="pb-2">Score</th>
                      <th className="pb-2">Twin Legend</th>
                      <th className="pb-2 text-right">Action</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#1B2921]">
                    {history.map((s) => (
                      <tr key={s.session_id}>
                        <td className="py-2.5 text-gray-300">{s.created_at || 'Recent'}</td>
                        <td className="py-2.5 font-bold text-white">{s.activity_type}</td>
                        <td className="py-2.5">
                          <span className={`font-bold ${s.is_legal ? 'text-lime-400' : 'text-red-400'}`}>
                            {s.is_legal ? 'Legal' : 'Concern'}
                          </span>
                        </td>
                        <td className="py-2.5 font-bold text-white">{s.accuracy_score}</td>
                        <td className="py-2.5 text-amber-300">{s.matched_legend || '--'}</td>
                        <td className="py-2.5 text-right">
                          <button
                            onClick={() => deleteSessionVideo(s.session_id)}
                            className="text-red-400 hover:text-red-300 text-[11px] font-bold"
                          >
                            Delete Video
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* TAB 4: TUTOR & WEEKLY PLAN */}
        {activeTab === 'tutor' && (
          <div className="space-y-6">
            <div className="bg-[#131D17]/80 border border-[#283D31] rounded-3xl p-6">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
                <div>
                  <h2 className="text-xl font-black text-white">7-Day Grassroots Practice Plan Generator</h2>
                  <p className="text-xs text-gray-400">Structured daily drills using tennis balls, chalk, and flat walls.</p>
                </div>
                <button
                  onClick={generateWeeklyPlan}
                  className="px-4 py-2 rounded-xl bg-lime-500 hover:bg-lime-400 text-[#070B09] font-black text-xs transition"
                >
                  Generate Plan
                </button>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-4">
                <div>
                  <label className="block text-xs font-bold text-gray-400 mb-1">Flaw to Correct:</label>
                  <select
                    value={planFlaw}
                    onChange={(e) => setPlanFlaw(e.target.value)}
                    className="w-full bg-[#0D1410] border border-[#283D31] rounded-xl px-3 py-2 text-xs text-white"
                  >
                    <option value="collapsed_front_knee">Collapsed Front Knee at Impact</option>
                    <option value="chucking_elbow_flexion">Dynamic Elbow Straightening</option>
                    <option value="mixed_action_counter_rotation">Mixed Bowling Action (Torso Counter-Rotation)</option>
                    <option value="head_falling_away">Head Falling Away to Off-Side</option>
                  </select>
                </div>
                <div>
                  <label className="block text-xs font-bold text-gray-400 mb-1">Level:</label>
                  <select
                    value={planLevel}
                    onChange={(e) => setPlanLevel(e.target.value)}
                    className="w-full bg-[#0D1410] border border-[#283D31] rounded-xl px-3 py-2 text-xs text-white"
                  >
                    <option value="Beginner">Beginner (School / Gully)</option>
                    <option value="Intermediate">Intermediate (Club / Academy)</option>
                    <option value="Advanced">Advanced (First-Class Prep)</option>
                  </select>
                </div>
              </div>

              {weeklyPlan && (
                <div className="space-y-3">
                  <p className="text-xs text-lime-400 font-bold">Goal: {weeklyPlan.weekly_goal}</p>
                  <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3">
                    {weeklyPlan.daily_schedule.map((d) => (
                      <div key={d.day_number} className={`rounded-xl p-3 border-l-4 bg-[#0D1410] border border-[#283D31] ${d.is_rest_day ? 'border-l-gray-600 opacity-75' : 'border-l-lime-500'}`}>
                        <div className="flex justify-between items-center mb-1">
                          <span className="font-extrabold text-xs text-white">{d.day_name}</span>
                          <span className="text-[10px] text-gray-400">{d.duration_minutes}m</span>
                        </div>
                        <p className="text-xs font-semibold text-lime-300 truncate">{d.focus_theme}</p>
                        {d.drills.length > 0 ? (
                          <p className="text-[11px] text-gray-300 mt-1">Drill: {d.drills[0].name}</p>
                        ) : (
                          <p className="text-[11px] text-gray-400 mt-1">Active recovery</p>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Chat Box */}
            <div className="bg-[#131D17]/80 border border-[#283D31] rounded-2xl p-6">
              <h3 className="text-base font-bold text-white mb-2">💬 Ask CricTutor AI</h3>
              <div className="h-48 overflow-y-auto bg-[#0D1410] rounded-xl p-3 border border-[#283D31] space-y-2 mb-3 text-xs">
                {chatMessages.map((m, idx) => (
                  <div key={idx} className={`p-2.5 rounded-lg ${m.sender === 'You' ? 'bg-[#1B2921] text-white' : 'bg-[#131D17] text-gray-200 border border-[#283D31]'}`}>
                    <strong className={m.sender === 'You' ? 'text-lime-300' : 'text-lime-400'}>{m.sender}: </strong>
                    {m.text}
                  </div>
                ))}
              </div>
              <div className="flex gap-2">
                <input
                  type="text"
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && sendChatMessage()}
                  placeholder="Ask a technical question..."
                  className="flex-grow bg-[#0D1410] border border-[#283D31] rounded-xl px-3 py-2 text-xs text-white"
                />
                <button
                  onClick={sendChatMessage}
                  className="px-4 py-2 rounded-xl bg-lime-500 hover:bg-lime-400 text-[#070B09] font-black text-xs transition"
                >
                  Send
                </button>
              </div>
            </div>
          </div>
        )}

        {/* TAB 5: PROFILE & PRIVACY */}
        {activeTab === 'profile' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-[#131D17]/80 border border-[#283D31] rounded-2xl p-6 space-y-4">
                <h3 className="text-lg font-bold text-white">👤 Player Profile</h3>
                <div>
                  <label className="block text-xs font-bold text-gray-400 mb-1">Player Name:</label>
                  <input type="text" defaultValue="Grassroots Player" className="w-full bg-[#0D1410] border border-[#283D31] rounded-xl px-3 py-2 text-xs text-white" />
                </div>
                <div>
                  <label className="block text-xs font-bold text-gray-400 mb-1">Role:</label>
                  <select defaultValue="Fast Bowler" className="w-full bg-[#0D1410] border border-[#283D31] rounded-xl px-3 py-2 text-xs text-white">
                    <option value="Fast Bowler">Fast / Medium Bowler</option>
                    <option value="Spin Bowler">Spin Bowler</option>
                    <option value="Top-Order Batter">Top-Order Batter</option>
                  </select>
                </div>
                <button onClick={() => alert('Profile saved locally!')} className="px-4 py-2 rounded-xl bg-lime-500 hover:bg-lime-400 text-[#070B09] font-black text-xs transition">
                  Save Preferences
                </button>
              </div>

              <div className="bg-[#131D17]/80 border border-[#283D31] rounded-2xl p-6 space-y-4">
                <h3 className="text-lg font-bold text-white">🔒 Privacy & Storage</h3>
                <div className="p-3 rounded-xl bg-[#0D1410] border border-[#283D31] text-xs">
                  <p className="font-bold text-lime-400 mb-1">🛡️ 100% On-Device Processing</p>
                  <p className="text-gray-300">Biomechanical pose estimates and joint tracking run locally on your hardware. We do not sell video profiles.</p>
                </div>
                <button
                  onClick={() => analysisData ? deleteSessionVideo(analysisData.session_id) : alert('No active session.')}
                  className="px-3 py-2 rounded-xl bg-[#1B2921] hover:bg-[#283D31] text-red-400 border border-red-500/30 text-xs font-bold"
                >
                  🗑️ Delete Latest Session Video File
                </button>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Mobile Bottom Navigation */}
      <nav className="md:hidden sticky bottom-0 z-50 bg-[#0D1410]/95 backdrop-blur-md border-t border-[#283D31]/60 flex justify-around py-2 text-[10px] font-bold">
        {['home', 'analyze', 'progress', 'tutor', 'profile'].map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`p-1.5 flex flex-col items-center capitalize ${activeTab === tab ? 'text-lime-400' : 'text-gray-400'}`}
          >
            <span>{tab === 'home' ? '🏠' : tab === 'analyze' ? '🎥' : tab === 'progress' ? '📈' : tab === 'tutor' ? '🏏' : '👤'}</span>
            <span>{tab}</span>
          </button>
        ))}
      </nav>

      {/* Camera Guidance Modal */}
      {showCameraModal && (
        <div className="fixed inset-0 z-50 bg-black/75 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-[#131D17] border border-[#283D31] rounded-3xl p-6 max-w-lg w-full space-y-4">
            <div className="flex justify-between items-center">
              <h3 className="text-lg font-bold text-white">📐 Camera Setup Guidance</h3>
              <button onClick={() => setShowCameraModal(false)} className="text-gray-400 hover:text-white">&times;</button>
            </div>
            <div className="space-y-3 text-xs text-gray-300">
              <div className="p-3 bg-[#0D1410] rounded-xl border border-[#283D31]">
                <strong className="text-lime-400 block mb-1">Side-On (Perpendicular to Pitch) — Best for Elbows</strong>
                Position 5-7 meters from the bowling crease at waist height (1.0m - 1.2m). Ensure head-to-toe visibility across delivery stride.
              </div>
              <div className="p-3 bg-[#0D1410] rounded-xl border border-[#283D31]">
                <strong className="text-cyan-400 block mb-1">Front-On (Behind Stumps) — Best for Alignment</strong>
                Position 6-8 meters behind target stumps at chest height (1.2m - 1.5m). Ideal for detecting trunk tilt and head stability.
              </div>
            </div>
            <button
              onClick={() => setShowCameraModal(false)}
              className="w-full py-2.5 rounded-xl bg-lime-500 hover:bg-lime-400 text-[#070B09] font-black text-xs transition"
            >
              Got it!
            </button>
          </div>
        </div>
      )}

      {/* Feedback Modal */}
      {showFeedbackModal && (
        <div className="fixed inset-0 z-50 bg-black/75 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-[#131D17] border border-[#283D31] rounded-3xl p-6 max-w-md w-full space-y-4">
            <h3 className="text-lg font-bold text-white">✍️ Feedback on Observation</h3>
            <p className="text-xs text-gray-300">If camera placement caused a false reading, you can dismiss or provide a coach correction note.</p>
            <textarea
              rows={3}
              value={feedbackText}
              onChange={(e) => setFeedbackText(e.target.value)}
              placeholder="e.g. Angle was tilted; arm was straight in reality..."
              className="w-full bg-[#0D1410] border border-[#283D31] rounded-xl p-2.5 text-xs text-white"
            />
            <div className="flex justify-end gap-2 text-xs">
              <button onClick={() => setShowFeedbackModal(false)} className="px-4 py-2 rounded-xl bg-[#1B2921] text-gray-300">Cancel</button>
              <button onClick={submitFeedback} className="px-4 py-2 rounded-xl bg-lime-500 text-[#070B09] font-black">Dismiss & Submit</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
