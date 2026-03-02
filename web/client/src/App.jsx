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

export default function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const selectedPoseRef = useRef('squat');

  const [selectedPose, setSelectedPose] = useState('squat');
  const [intervalMs, setIntervalMs] = useState(1200);
  const [isRealtime, setIsRealtime] = useState(false);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [feedback, setFeedback] = useState(null);
  const [cameraReady, setCameraReady] = useState(false);
  const [lastUpdated, setLastUpdated] = useState('');

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
            width: { ideal: 960 },
            height: { ideal: 720 }
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

  async function analyzeFrame() {
    if (!videoRef.current || !canvasRef.current || loading) {
      return;
    }

    setLoading(true);
    setError('');

    try {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const width = video.videoWidth || 640;
      const height = video.videoHeight || 480;

      canvas.width = width;
      canvas.height = height;

      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0, width, height);

      const blob = await new Promise((resolve) => {
        canvas.toBlob(resolve, 'image/jpeg', 0.9);
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

      setFeedback(data);
      setLastUpdated(new Date().toLocaleTimeString());
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
          </div>

          <div className="video-wrap">
            <video ref={videoRef} autoPlay playsInline muted />
            <canvas ref={canvasRef} className="hidden-canvas" />
            {!cameraReady && <div className="overlay">Waiting for camera...</div>}
          </div>
        </section>

        <section className="panel feedback-panel">
          <h2>Coach Feedback</h2>

          {error && <p className="error">{error}</p>}

          {!error && !feedback && <p className="muted">Start real-time mode or analyze one frame.</p>}

          {feedback && (
            <>
              <p className={feedback.status === 'good' ? 'status-good' : 'status-warn'}>
                {feedback.status === 'good' ? 'Good Form' : 'Needs Adjustment'}
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
