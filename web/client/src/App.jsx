import { useEffect, useRef, useState } from 'react';
import axios from 'axios';

const POSES = [
  { value: 'squat', label: 'Squat' },
  { value: 'plank', label: 'Plank' },
  { value: 'downdog', label: 'Downward Dog' },
  { value: 'tree', label: 'Tree' },
  { value: 'warrior2', label: 'Warrior II' },
  { value: 'goddess', label: 'Goddess' }
];

const INTERVAL_OPTIONS = [
  { value: 700, label: 'Fast (0.7s)' },
  { value: 1200, label: 'Balanced (1.2s)' },
  { value: 1800, label: 'Light (1.8s)' }
];

const CAPTURE_MAX_WIDTH = 512;
const CAPTURE_JPEG_QUALITY = 0.65;
const VOICE_COOLDOWN_MS = 6500;
const STABILITY_WINDOW = 5;
const STABILITY_REQUIRED = 3;

function formatDuration(totalSeconds) {
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${String(seconds).padStart(2, '0')}`;
}

export default function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const selectedPoseRef = useRef('squat');
  const lastSpokenRef = useRef({ text: '', at: 0 });
  const feedbackWindowRef = useRef([]);
  const squatTrackerRef = useRef({ seenBottom: false, lastPhase: '', lastRepAt: 0 });

  const [selectedPose, setSelectedPose] = useState('squat');
  const [intervalMs, setIntervalMs] = useState(1200);
  const [isRealtime, setIsRealtime] = useState(false);
  const [voiceEnabled, setVoiceEnabled] = useState(false);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [feedback, setFeedback] = useState(null);
  const [cameraReady, setCameraReady] = useState(false);
  const [lastUpdated, setLastUpdated] = useState('');
  const [stabilityMessage, setStabilityMessage] = useState('');
  const [sessionSeconds, setSessionSeconds] = useState(0);
  const [poseSeconds, setPoseSeconds] = useState(0);
  const [squatReps, setSquatReps] = useState(0);
  const [poseDetected, setPoseDetected] = useState(false);

  useEffect(() => {
    selectedPoseRef.current = selectedPose;
  }, [selectedPose]);

  useEffect(() => {
    let stream;

    async function initCamera() {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: 'user',
            width: { ideal: 640 },
            height: { ideal: 480 }
          },
          audio: false
        });

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          setCameraReady(true);
        }
      } catch {
        setError('Unable to access camera. Please allow webcam permissions.');
      }
    }

    initCamera();

    return () => {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  useEffect(() => {
    if (
      !voiceEnabled ||
      !feedback?.success ||
      feedback.status === 'good' ||
      !feedback.feedback ||
      !('speechSynthesis' in window)
    ) {
      return;
    }

    const now = Date.now();
    const text = feedback.feedback;
    if (
      text === lastSpokenRef.current.text &&
      now - lastSpokenRef.current.at < VOICE_COOLDOWN_MS
    ) {
      return;
    }

    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 0.92;
    utterance.pitch = 1;
    utterance.volume = 0.85;
    window.speechSynthesis.speak(utterance);
    lastSpokenRef.current = { text, at: now };
  }, [feedback, voiceEnabled]);

  function handleVoiceToggle(enabled) {
    setVoiceEnabled(enabled);
    if (!('speechSynthesis' in window)) {
      return;
    }

    window.speechSynthesis.cancel();
    if (enabled) {
      window.speechSynthesis.resume();
      lastSpokenRef.current = { text: '', at: 0 };
    }
  }

  function feedbackKey(result) {
    if (!result?.success) {
      return `camera:${result?.error || 'not_ready'}`;
    }

    return `${result.pose || selectedPoseRef.current}:${result.phase || 'hold'}:${result.issue || result.status}`;
  }

  function acceptStableFeedback(result) {
    if (!result) {
      return false;
    }

    if (!result.success) {
      feedbackWindowRef.current = [];
      setStabilityMessage('');
      return true;
    }

    const key = feedbackKey(result);
    const nextWindow = [...feedbackWindowRef.current, key].slice(-STABILITY_WINDOW);
    feedbackWindowRef.current = nextWindow;

    const matches = nextWindow.filter((item) => item === key).length;
    const isCorrection = result.status !== 'good';
    const requiredMatches = isCorrection ? STABILITY_REQUIRED : 2;
    const isStable = matches >= requiredMatches;

    setStabilityMessage(isStable ? '' : 'Checking consistency...');
    return isStable;
  }

  function updateWorkoutStats(result) {
    const isSquat = selectedPoseRef.current === 'squat';
    setPoseDetected(Boolean(result?.success));

    if (!isSquat) {
      squatTrackerRef.current = { seenBottom: false, lastPhase: '', lastRepAt: 0 };
      return;
    }

    if (!result?.success || result.pose !== 'squat') {
      return;
    }

    const phase = result.phase || '';
    const tracker = squatTrackerRef.current;

    if (phase === 'bottom') {
      tracker.seenBottom = true;
    }

    if (tracker.seenBottom && phase === 'standing' && tracker.lastPhase !== 'standing') {
      const now = Date.now();
      if (now - tracker.lastRepAt > 1800) {
        setSquatReps((count) => count + 1);
        tracker.lastRepAt = now;
      }
      tracker.seenBottom = false;
    }

    tracker.lastPhase = phase;
  }

  function resetSession() {
    setSessionSeconds(0);
    setPoseSeconds(0);
    setSquatReps(0);
    setPoseDetected(false);
    squatTrackerRef.current = { seenBottom: false, lastPhase: '', lastRepAt: 0 };
  }

  async function analyzeFrame() {
    if (!videoRef.current || !canvasRef.current || loading) {
      return;
    }

    setLoading(true);
    setError('');

    try {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const sourceWidth = video.videoWidth || 640;
      const sourceHeight = video.videoHeight || 480;
      const scale = Math.min(1, CAPTURE_MAX_WIDTH / sourceWidth);
      const width = Math.round(sourceWidth * scale);
      const height = Math.round(sourceHeight * scale);

      canvas.width = width;
      canvas.height = height;

      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0, width, height);

      const blob = await new Promise((resolve) => {
        canvas.toBlob(resolve, 'image/jpeg', CAPTURE_JPEG_QUALITY);
      });

      if (!blob) {
        throw new Error('Failed to capture image');
      }

      const formData = new FormData();
      formData.append('frame', blob, 'frame.jpg');
      formData.append('selectedPose', selectedPoseRef.current);

      const { data } = await axios.post('/api/analyze', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      setLastUpdated(new Date().toLocaleTimeString());
      updateWorkoutStats(data);
      if (acceptStableFeedback(data)) {
        setFeedback(data);
      }
    } catch (err) {
      const message = err.response?.data?.detail || err.response?.data?.error || err.message;
      setError(message || 'Analysis failed');
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (!isRealtime || !cameraReady) {
      return undefined;
    }

    let stopped = false;
    let timerId;

    async function loop() {
      if (stopped) {
        return;
      }
      await analyzeFrame();
      if (!stopped) {
        timerId = setTimeout(loop, intervalMs);
      }
    }

    loop();

    return () => {
      stopped = true;
      clearTimeout(timerId);
    };
  }, [isRealtime, intervalMs, cameraReady]);

  useEffect(() => {
    if (!isRealtime) {
      return undefined;
    }

    const timerId = setInterval(() => {
      setSessionSeconds((seconds) => seconds + 1);
      if (poseDetected) {
        setPoseSeconds((seconds) => seconds + 1);
      }
    }, 1000);

    return () => clearInterval(timerId);
  }, [isRealtime, poseDetected]);

  useEffect(() => {
    feedbackWindowRef.current = [];
    setFeedback(null);
    setStabilityMessage('');
    setPoseSeconds(0);
    setPoseDetected(false);
    squatTrackerRef.current = { seenBottom: false, lastPhase: '', lastRepAt: 0 };
    lastSpokenRef.current = { text: '', at: 0 };
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }
  }, [selectedPose]);

  return (
    <div className="app-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">AI FORM COACH</p>
          <h1>Your Gym Buddy</h1>
          <p className="hero-sub">Choose your target pose and get continuous coaching feedback.</p>
        </div>
        <div className="status-chip">
          <span className={`dot ${isRealtime ? 'live' : ''}`} />
          {isRealtime ? 'Live Analysis On' : 'Live Analysis Off'}
        </div>
      </header>

      <main className="layout">
        <section className="panel camera-panel">
          <div className="controls-grid">
            <div className="field">
              <label htmlFor="pose">Target Pose</label>
              <select
                id="pose"
                value={selectedPose}
                onChange={(e) => setSelectedPose(e.target.value)}
              >
                {POSES.map((pose) => (
                  <option key={pose.value} value={pose.value}>
                    {pose.label}
                  </option>
                ))}
              </select>
            </div>

            <div className="field">
              <label htmlFor="interval">Update Rate</label>
              <select
                id="interval"
                value={intervalMs}
                onChange={(e) => setIntervalMs(Number(e.target.value))}
              >
                {INTERVAL_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="actions">
            <button
              className="btn-primary"
              onClick={() => setIsRealtime((prev) => !prev)}
              disabled={!cameraReady}
            >
              {isRealtime ? 'Stop Real-Time Coach' : 'Start Real-Time Coach'}
            </button>
            <button className="btn-secondary" onClick={analyzeFrame} disabled={!cameraReady || loading}>
              {loading ? 'Analyzing...' : 'Analyze One Frame'}
            </button>
            <label className="voice-toggle">
              <input
                type="checkbox"
                checked={voiceEnabled}
                onChange={(e) => handleVoiceToggle(e.target.checked)}
              />
              Voice cues
            </label>
            <button className="btn-secondary" onClick={resetSession}>
              Reset Stats
            </button>
          </div>

          <div className="video-wrap">
            <video ref={videoRef} autoPlay playsInline muted />
            <canvas ref={canvasRef} className="hidden-canvas" />
            {!cameraReady && <div className="overlay">Waiting for camera...</div>}
          </div>
        </section>

        <section className="panel feedback-panel">
          <h2>Coach Feedback</h2>

          <div className="session-stats">
            <p>
              <span>Workout Time</span>
              <strong>{formatDuration(sessionSeconds)}</strong>
            </p>
            <p>
              <span>{selectedPose === 'squat' ? 'Squat Time' : 'Pose Hold'}</span>
              <strong>{formatDuration(poseSeconds)}</strong>
            </p>
            <p>
              <span>Squat Reps</span>
              <strong>{squatReps}</strong>
            </p>
          </div>

          {error && <p className="error">{error}</p>}

          {!error && !feedback && <p className="muted">Start real-time mode or analyze one frame.</p>}
          {!error && feedback && stabilityMessage && <p className="muted">{stabilityMessage}</p>}

          {feedback && (
            <>
              <p className={feedback.status === 'good' ? 'status-good' : 'status-warn'}>
                {!feedback.success
                  ? 'Adjust Camera'
                  : feedback.status === 'good'
                    ? 'Good Form'
                    : 'Needs Adjustment'}
              </p>
              <p className="tip">{feedback.feedback}</p>

              <div className="metrics">
                <p>
                  <span>Knee Angle</span>
                  <strong>{feedback.metrics?.knee_angle ?? '-'} deg</strong>
                </p>
                <p>
                  <span>Hip Angle</span>
                  <strong>{feedback.metrics?.hip_angle ?? '-'} deg</strong>
                </p>
                <p>
                  <span>Torso Lean</span>
                  <strong>{feedback.metrics?.torso_lean ?? '-'}</strong>
                </p>
                <p>
                  <span>Visible Side</span>
                  <strong>{feedback.metrics?.side ?? '-'}</strong>
                </p>
                <p>
                  <span>Visibility</span>
                  <strong>{feedback.visibility?.score ?? '-'}</strong>
                </p>
                <p>
                  <span>Issue</span>
                  <strong>{feedback.issue ?? feedback.error ?? '-'}</strong>
                </p>
                <p>
                  <span>Phase</span>
                  <strong>{feedback.phase ?? '-'}</strong>
                </p>
                <p>
                  <span>Updated</span>
                  <strong>{lastUpdated || '-'}</strong>
                </p>
              </div>
            </>
          )}
        </section>
      </main>
    </div>
  );
}
