import { useState, useEffect, useRef } from 'react';

const SIZE = 260;

export default function AiAvatar({ imageUrl, state = 'idle', question }) {
  const canvasRef = useRef(null);
  const imgRef = useRef(null);
  const lmRef = useRef(null);
  const animRef = useRef(null);
  const mouthRef = useRef(0);
  const [ready, setReady] = useState(false);
  const [err, setErr] = useState(null);

  useEffect(() => {
    if (!imageUrl) { setReady(true); return; }
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      imgRef.current = img;
      detectFace(img);
    };
    img.onerror = () => { setErr('Image failed to load'); setReady(true); };
    img.src = imageUrl;
  }, [imageUrl]);

  async function detectFace(img) {
    try {
      const { FaceLandmarker, FilesetResolver } = await import('@mediapipe/tasks-vision');
      const vision = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
      );
      const fl = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
          delegate: 'GPU',
        },
        runningMode: 'IMAGE',
        numFaces: 1,
      });
      const c = document.createElement('canvas');
      c.width = img.width; c.height = img.height;
      const ctx = c.getContext('2d');
      ctx.drawImage(img, 0, 0);
      const r = fl.detect(c);
      if (r.faceLandmarks?.length > 0) lmRef.current = r.faceLandmarks[0];
    } catch (e) {
      console.warn('Face landmark detection failed:', e);
    }
    setReady(true);
  }

  useEffect(() => {
    if (!ready || state !== 'speaking') {
      if (animRef.current) { cancelAnimationFrame(animRef.current); animRef.current = null; }
      if (state !== 'speaking') mouthRef.current = 0;
      drawFrame();
      return;
    }
    const t0 = performance.now();
    function loop() {
      const t = (performance.now() - t0) / 1000;
      const freq = 4.5 + Math.sin(t * 0.7) * 1.2;
      const open = (Math.sin(t * freq * Math.PI * 2) * 0.5 + 0.5) * (0.5 + Math.sin(t * 1.3) * 0.2 + 0.3);
      mouthRef.current = Math.max(0.05, Math.min(1, open));
      drawFrame();
      animRef.current = requestAnimationFrame(loop);
    }
    animRef.current = requestAnimationFrame(loop);
    return () => { if (animRef.current) cancelAnimationFrame(animRef.current); };
  }, [ready, state]);

  function drawFrame() {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = SIZE, H = SIZE;
    canvas.width = W; canvas.height = H;
    ctx.clearRect(0, 0, W, H);

    if (!img || err) {
      ctx.fillStyle = 'var(--bg3)';
      ctx.beginPath(); ctx.arc(W / 2, H / 2, W / 2, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = 'var(--text2)';
      ctx.font = '40px sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(err ? '😅' : '🤖', W / 2, H / 2 - 10);
      ctx.font = '12px sans-serif';
      ctx.fillText(err || 'AI Coach', W / 2, H / 2 + 30);
      return;
    }

    const lm = lmRef.current;
    let sx = 0, sy = 0, sw = img.width, sh = img.height;
    if (lm) {
      const xs = lm.map(p => p.x), ys = lm.map(p => p.y);
      const mnX = Math.min(...xs), mxX = Math.max(...xs);
      const mnY = Math.min(...ys), mxY = Math.max(...ys);
      const fw = mxX - mnX, fh = mxY - mnY, m = 0.12;
      sx = Math.max(0, (mnX - m * fw) * img.width);
      sy = Math.max(0, (mnY - m * fh * 0.5) * img.height);
      sw = Math.min(img.width - sx, (fw + 2 * m * fw) * img.width);
      sh = Math.min(img.height - sy, (fh + 2 * m * fh) * img.height);
    }
    ctx.save();
    ctx.beginPath(); ctx.arc(W / 2, H / 2, W / 2, 0, Math.PI * 2); ctx.clip();
    ctx.drawImage(img, sx, sy, sw, sh, 0, 0, W, H);

    if (lm && state === 'speaking') {
      const open = mouthRef.current;
      const ul = lm[13], ll = lm[14], lc = lm[61], rc = lm[291];
      if (ul && ll && lc && rc) {
        const mX = p => ((p.x * img.width - sx) / sw) * W;
        const mY = p => ((p.y * img.height - sy) / sh) * H;
        const cx = (mX(lc) + mX(rc)) / 2;
        const bw = (mX(rc) - mX(lc)) * 0.85;
        const gap = (mY(ll) - mY(ul)) * (0.2 + open * 1.2);
        const cy = mY(ul) + gap * 0.3;

        ctx.save();
        ctx.beginPath();
        ctx.ellipse(cx, cy, bw * 0.5, gap * 0.5, 0, 0, Math.PI * 2);
        ctx.fillStyle = '#1a1a1a';
        ctx.fill();
        ctx.beginPath();
        ctx.ellipse(cx, cy - gap * 0.15, bw * 0.45, gap * 0.25, 0, Math.PI, Math.PI * 2);
        ctx.fillStyle = '#d4a0a0';
        ctx.fill();
        ctx.beginPath();
        ctx.ellipse(cx, cy + gap * 0.35, bw * 0.45, gap * 0.2, 0, 0, Math.PI);
        ctx.fillStyle = '#d4a0a0';
        ctx.fill();
        ctx.restore();
      }
    }
    ctx.restore();
  }

  const borderColor = state === 'speaking' ? 'var(--green)' : state === 'listening' ? 'var(--blue)' : state === 'thinking' ? 'var(--yellow)' : 'var(--border)';

  return (
    <div style={{ position: 'relative', width: SIZE, margin: '0 auto' }}>
      {question && (state === 'speaking' || state === 'thinking') && (
        <div style={{
          position: 'absolute', top: '100%', left: '50%', transform: 'translateX(-50%)',
          marginTop: 16, padding: '10px 16px', background: 'var(--bg3)',
          borderRadius: 14, border: '1px solid var(--border)',
          fontSize: '0.9rem', color: 'var(--text)', lineHeight: 1.5,
          minWidth: 200, maxWidth: 350, textAlign: 'center',
          boxShadow: '0 4px 16px rgba(0,0,0,0.1)', zIndex: 10,
        }}>
          <div style={{
            position: 'absolute', bottom: '100%', left: '50%', transform: 'translateX(-50%)',
            width: 0, height: 0,
            borderLeft: '7px solid transparent', borderRight: '7px solid transparent',
            borderBottom: '7px solid var(--border)',
          }} />
          {state === 'thinking' ? (
            <span>Thinking{'.'.repeat(Math.floor(Date.now() / 600) % 4)}</span>
          ) : question}
        </div>
      )}
      <canvas ref={canvasRef} style={{
        width: SIZE, height: SIZE, borderRadius: '50%',
        border: '3px solid', borderColor,
        boxShadow: state === 'speaking'
          ? '0 0 30px rgba(52,168,83,0.5), 0 0 60px rgba(52,168,83,0.2)'
          : state !== 'idle'
            ? `0 0 24px ${borderColor.replace('var', 'rgba').replace(')', ',0.25)')}`
            : 'none',
        animation: state === 'speaking' ? 'pulseGlow 1.5s ease-in-out infinite' : 'none',
        transition: 'border-color 0.3s, box-shadow 0.3s',
        display: 'block',
        background: 'var(--bg)',
      }} />
      <div style={{
        position: 'absolute', bottom: 6, right: 6,
        width: 14, height: 14, borderRadius: '50%',
        background: borderColor,
        border: '2px solid var(--bg)',
        transition: 'background 0.3s',
      }} />
    </div>
  );
}
