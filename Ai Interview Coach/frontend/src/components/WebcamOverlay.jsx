import { useState, useEffect, useRef, useCallback } from 'react';

export default function WebcamOverlay({ onMetrics }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const runningRef = useRef(false);
  const detectorRef = useRef(null);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState(null);
  const [isMirrored, setIsMirrored] = useState(true);
  const [metrics, setMetrics] = useState({
    eye_contact_score: 0.5,
    movement_score: 0.5,
    confidence_score: 0.5,
    stress_score: 0.5,
    hand_count: 0,
    has_face: false,
  });

  const prevNose = useRef(null);
  const smoothedMetricsRef = useRef({
    eye_contact_score: 0.5,
    movement_score: 0.5,
    confidence_score: 0.5,
    stress_score: 0.5,
  });
  const lastUpdateRef = useRef(0);

  const loadDetector = useCallback(async () => {
    try {
      const { FaceLandmarker, HandLandmarker, FilesetResolver } = await import('@mediapipe/tasks-vision');
      const vision = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
      );

      const faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
          delegate: 'GPU',
        },
        runningMode: 'VIDEO',
        numFaces: 1,
        outputFaceBlendshapes: false,
        outputFacialTransformationMatrixes: false,
      });

      const handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
          delegate: 'GPU',
        },
        runningMode: 'VIDEO',
        numHands: 2,
      });

      detectorRef.current = { faceLandmarker, handLandmarker };
      setReady(true);
    } catch {
      setError('Failed to load vision models. Using basic detection.');
      setReady(true);
    }
  }, []);

  useEffect(() => {
    loadDetector();
    return () => {
      runningRef.current = false;
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }
    };
  }, [loadDetector]);

  useEffect(() => {
    if (!ready) return;

    async function start() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false });
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
        }
        runningRef.current = true;
        processFrames();
      } catch {
        setError('Camera access denied.');
      }
    }

    start();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready]);

  function processFrames() {
    if (!runningRef.current) return;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const det = detectorRef.current;
    if (!video || !canvas || video.readyState < 2) {
      requestAnimationFrame(processFrames);
      return;
    }

    const w = video.videoWidth || 640;
    const h = video.videoHeight || 480;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, w, h);
    ctx.drawImage(video, 0, 0, w, h);

    let eye = 0.3, move = 0.6, handCount = 0, hasFace = false;

    if (det) {
      try {
        const result = det.faceLandmarker.detectForVideo(video, performance.now());
        if (result.faceLandmarks && result.faceLandmarks.length > 0) {
          hasFace = true;
          const lm = result.faceLandmarks[0];
          const nose = lm[1];
          const leftCheek = lm[234];
          const rightCheek = lm[454];
          const forehead = lm[10];
          const chin = lm[152];

          const centerX = (leftCheek.x + rightCheek.x) / 2;
          const centerY = (forehead.y + chin.y) / 2;
          const yaw = Math.abs(nose.x - centerX);
          const pitch = Math.abs(nose.y - centerY);
          eye = Math.max(0, Math.min(1, 1 - (yaw * 3.5 + pitch * 2.5)));

          const curr = { x: nose.x, y: nose.y };
          if (prevNose.current) {
            const motion = Math.abs(curr.x - prevNose.current.x) + Math.abs(curr.y - prevNose.current.y);
            move = Math.max(0, Math.min(1, 1 - motion * 8));
          } else {
            move = 0.75;
          }
          prevNose.current = curr;

          if (result.faceLandmarks[0].length > 470) {
            const li = result.faceLandmarks[0];
            const leftEAR = earRatio(li, 33, 160, 158, 133, 153, 144);
            const rightEAR = earRatio(li, 263, 385, 387, 362, 373, 380);
            const eyeOpen = Math.max(0, Math.min(1, ((leftEAR + rightEAR) / 2 - 0.12) / 0.18));
            eye = 0.75 * eye + 0.25 * eyeOpen;
          }

          drawFaceMesh(ctx, lm, w, h);
        }

        const handResult = det.handLandmarker.detectForVideo(video, performance.now());
        if (handResult.handLandmarks) {
          handCount = handResult.handLandmarks.length;
          drawHands(ctx, handResult.handLandmarks, w, h);
        }
      } catch {
        // fallback below
      }
    }

    if (!hasFace) {
      const gray = grayscaleFromCanvas(canvas, w, h);
      const basic = basicFaceDetect(gray, w, h);
      if (basic.found) {
        hasFace = true;
        eye = basic.eyeContact;
      }
      prevNose.current = null;
    }

    const handSignal = handCount > 0 ? 0.15 : 0;
    const confidence = Math.max(0, Math.min(1, 0.55 * eye + 0.35 * move + handSignal));
    const stress = Math.max(0, Math.min(1, 1 - (0.45 * eye + 0.35 * move + 0.20 * 0.6)));

    // Exponential Moving Average (EMA) smoothing to eliminate noise jitter
    const alpha = 0.08;
    smoothedMetricsRef.current.eye_contact_score = alpha * eye + (1 - alpha) * smoothedMetricsRef.current.eye_contact_score;
    smoothedMetricsRef.current.movement_score = alpha * move + (1 - alpha) * smoothedMetricsRef.current.movement_score;
    smoothedMetricsRef.current.confidence_score = alpha * confidence + (1 - alpha) * smoothedMetricsRef.current.confidence_score;
    smoothedMetricsRef.current.stress_score = alpha * stress + (1 - alpha) * smoothedMetricsRef.current.stress_score;

    // Throttle React state and prop updates to once every 300ms for high readability
    const now = performance.now();
    if (now - lastUpdateRef.current > 300) {
      const newMetrics = {
        eye_contact_score: Math.round(smoothedMetricsRef.current.eye_contact_score * 1000) / 1000,
        movement_score: Math.round(smoothedMetricsRef.current.movement_score * 1000) / 1000,
        confidence_score: Math.round(smoothedMetricsRef.current.confidence_score * 1000) / 1000,
        stress_score: Math.round(smoothedMetricsRef.current.stress_score * 1000) / 1000,
        hand_count: handCount,
        has_face: hasFace,
      };
      setMetrics(newMetrics);
      if (onMetrics) onMetrics(newMetrics);
      lastUpdateRef.current = now;
    }

    requestAnimationFrame(processFrames);
  }

  return (
    <div className="card" style={{ padding: 0, overflow: 'hidden', position: 'relative', width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#000' }}>
      <video ref={videoRef} style={{ display: 'none' }} playsInline />
      <canvas
        ref={canvasRef}
        style={{
          maxWidth: '100%',
          maxHeight: '100%',
          objectFit: 'contain',
          display: 'block',
          borderRadius: 12,
          transform: isMirrored ? 'scaleX(-1)' : 'none',
          transition: 'transform 0.3s ease',
        }}
      />

      {/* Floating Toggle Mirror Button */}
      <button
        onClick={() => setIsMirrored(prev => !prev)}
        style={{
          position: 'absolute',
          top: 12,
          right: 12,
          background: 'rgba(0, 0, 0, 0.6)',
          border: '1px solid rgba(255, 255, 255, 0.2)',
          borderRadius: '50%',
          width: 36,
          height: 36,
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: '#fff',
          fontSize: '1rem',
          backdropFilter: 'blur(4px)',
          transition: 'all 0.2s',
          zIndex: 10,
        }}
        title="Toggle Mirror View"
        onMouseEnter={e => {
          e.currentTarget.style.background = 'rgba(0, 0, 0, 0.8)';
          e.currentTarget.style.transform = 'scale(1.05)';
        }}
        onMouseLeave={e => {
          e.currentTarget.style.background = 'rgba(0, 0, 0, 0.6)';
          e.currentTarget.style.transform = 'scale(1)';
        }}
      >
        🔄
      </button>

      {error && (
        <div style={{ position: 'absolute', bottom: 12, left: 12, right: 12, padding: 8, background: 'rgba(234,67,53,0.9)', color: '#fff', borderRadius: 8, fontSize: '0.82rem', textAlign: 'center', zIndex: 10 }}>
          {error}
        </div>
      )}
      {!ready && !error && (
        <div className="flex center" style={{ height: 200, zIndex: 5 }}>
          <div className="spinner" />
        </div>
      )}
    </div>
  );
}

function earRatio(lm, p1, p2, p3, p4, p5, p6) {
  const a = dist(lm[p1], lm[p2]);
  const b = dist(lm[p3], lm[p4]);
  const c = dist(lm[p5], lm[p6]);
  const d = dist(lm[p1], lm[p4]);
  return (a + b + c) / (3 * Math.max(d, 0.001));
}

function dist(a, b) {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
}

function drawFaceMesh(ctx, lm, w, h) {
  if (!lm) return;
  ctx.strokeStyle = 'rgba(26,115,232,0.4)';
  ctx.lineWidth = 1;
  // Draw face oval connections
  const faceConnections = [
    [10, 338], [338, 297], [297, 332], [332, 284], [284, 251],
    [251, 389], [389, 356], [356, 454], [454, 323], [323, 361],
    [361, 288], [288, 397], [397, 365], [365, 379], [379, 378],
    [378, 400], [400, 377], [377, 152], [152, 148], [148, 176],
    [176, 149], [149, 150], [150, 136], [136, 172], [172, 58],
    [58, 132], [132, 93], [93, 234], [234, 127], [127, 162],
    [162, 21], [21, 54], [54, 103], [103, 67], [67, 109],
    [109, 10],
  ];
  for (const [i, j] of faceConnections) {
    ctx.beginPath();
    ctx.moveTo(lm[i].x * w, lm[i].y * h);
    ctx.lineTo(lm[j].x * w, lm[j].y * h);
    ctx.stroke();
  }

  // Draw eye landmarks
  ctx.fillStyle = 'rgba(26,115,232,0.6)';
  const eyePoints = [33, 133, 362, 263, 468, 473];
  for (const i of eyePoints) {
    ctx.beginPath();
    ctx.arc(lm[i].x * w, lm[i].y * h, 2, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawHands(ctx, hands, w, h) {
  for (const hand of hands) {
    ctx.strokeStyle = 'rgba(52,168,83,0.5)';
    ctx.lineWidth = 2;
    const connections = [
      [0, 1], [1, 2], [2, 3], [3, 4],
      [0, 5], [5, 6], [6, 7], [7, 8],
      [0, 9], [9, 10], [10, 11], [11, 12],
      [0, 13], [13, 14], [14, 15], [15, 16],
      [0, 17], [17, 18], [18, 19], [19, 20],
    ];
    for (const [i, j] of connections) {
      ctx.beginPath();
      ctx.moveTo(hand[i].x * w, hand[i].y * h);
      ctx.lineTo(hand[j].x * w, hand[j].y * h);
      ctx.stroke();
    }
    ctx.fillStyle = 'rgba(52,168,83,0.8)';
    for (const p of hand) {
      ctx.beginPath();
      ctx.arc(p.x * w, p.y * h, 2, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

function grayscaleFromCanvas(canvas, w, h) {
  const ctx = canvas.getContext('2d');
  const imageData = ctx.getImageData(0, 0, w, h);
  const gray = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    const idx = i * 4;
    gray[i] = 0.299 * imageData.data[idx] + 0.587 * imageData.data[idx + 1] + 0.114 * imageData.data[idx + 2];
  }
  return gray;
}

function basicFaceDetect(gray, w, h) {
  // Simple brightness-based face detection fallback
  const centerW = Math.floor(w * 0.3);
  const centerH = Math.floor(h * 0.3);
  const cx = Math.floor(w * 0.5);
  const cy = Math.floor(h * 0.4);
  let sum = 0, count = 0;
  for (let y = cy - centerH / 2; y < cy + centerH / 2 && y < h; y++) {
    for (let x = cx - centerW / 2; x < cx + centerW / 2 && x < w; x++) {
      if (y >= 0 && x >= 0) { sum += gray[y * w + x]; count++; }
    }
  }
  const avg = count > 0 ? sum / count : 0;
  const centerOffset = Math.abs(0.5 - 0.5);
  const eye = Math.max(0, Math.min(1, 1 - centerOffset * 2));
  const brightnessNorm = Math.max(0, Math.min(1, avg / 200));
  return { found: brightnessNorm > 0.15, eyeContact: eye * 0.5 + brightnessNorm * 0.5 };
}
