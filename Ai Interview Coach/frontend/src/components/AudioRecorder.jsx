import { useState, useRef, useEffect, useCallback } from 'react';

export default function AudioRecorder({ onRecordingComplete }) {
  const [state, setState] = useState('idle');
  const mediaRecorder = useRef(null);
  const chunks = useRef([]);
  const timer = useRef(null);
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    return () => {
      if (timer.current) clearInterval(timer.current);
      if (mediaRecorder.current && mediaRecorder.current.state !== 'inactive') {
        mediaRecorder.current.stop();
      }
    };
  }, []);

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });
      mediaRecorder.current = recorder;
      chunks.current = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.current.push(e.data);
      };

      recorder.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(chunks.current, { type: 'audio/webm' });
        if (onRecordingComplete) onRecordingComplete(blob);
        setElapsed(0);
      };

      recorder.start(250);
      setState('recording');
      let sec = 0;
      timer.current = setInterval(() => {
        sec += 1;
        setElapsed(sec);
        if (sec >= 300) stopRecording();
      }, 1000);
    } catch {
      setState('error');
    }
  }, [onRecordingComplete]);

  const stopRecording = useCallback(() => {
    if (mediaRecorder.current && mediaRecorder.current.state !== 'inactive') {
      mediaRecorder.current.stop();
    }
    if (timer.current) {
      clearInterval(timer.current);
      timer.current = null;
    }
    setState('idle');
  }, []);

  const formatTime = (s) => {
    const m = Math.floor(s / 60);
    const sec = s % 60;
    return `${m}:${sec.toString().padStart(2, '0')}`;
  };

  if (state === 'recording') {
    return (
      <div className="flex center gap-8" style={{ padding: '8px 0' }}>
        <div style={{ width: 10, height: 10, borderRadius: '50%', background: 'var(--red)', animation: 'pulse 1s infinite' }} />
        <span style={{ color: 'var(--red)', fontWeight: 600, fontSize: '0.9rem' }}>{formatTime(elapsed)}</span>
        <button className="btn btn-danger" onClick={stopRecording} style={{ padding: '6px 16px', fontSize: '0.82rem' }}>
          Stop
        </button>
      </div>
    );
  }

  if (state === 'error') {
    return (
      <div className="flex center gap-8" style={{ padding: '8px 0' }}>
        <span style={{ color: 'var(--red)', fontSize: '0.85rem' }}>Microphone access denied</span>
        <button className="btn btn-secondary" onClick={() => setState('idle')} style={{ padding: '4px 12px', fontSize: '0.78rem' }}>
          Retry
        </button>
      </div>
    );
  }

  return (
    <button className="btn btn-secondary" onClick={startRecording} style={{ padding: '8px 20px', fontSize: '0.88rem' }}>
      <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" style={{ marginRight: 6 }}>
        <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z"/>
        <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/>
      </svg>
      Record
    </button>
  );
}
