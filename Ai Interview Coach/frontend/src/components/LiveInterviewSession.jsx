import { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import AiAvatar from './AiAvatar';
import WebcamOverlay from './WebcamOverlay';
import MovementSuggestions from './MovementSuggestions';

const SILENCE_THRESHOLD = 18;
const SILENCE_DURATION_MS = 1500;
const POLL_INTERVAL = 100;

function formatTime(s) {
  const m = Math.floor(s / 60);
  const sec = s % 60;
  return `${m}:${sec.toString().padStart(2, '0')}`;
}

function MetricCard({ value, label, color, icon }) {
  const pct = Math.round(value * 100);
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      className="card"
      style={{ padding: '10px 14px', minWidth: 100, textAlign: 'center', flex: 1 }}
    >
      <div style={{ fontSize: '1rem', marginBottom: 1 }}>{icon}</div>
      <div style={{ fontSize: '1.3rem', fontWeight: 700, color }}>{pct}%</div>
      <div style={{ fontSize: '0.7rem', color: 'var(--text2)', marginTop: 1 }}>{label}</div>
      <div className="progress-track" style={{ marginTop: 6, height: 4 }}>
        <motion.div
          className="progress-fill"
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.5, ease: 'easeOut' }}
          style={{ background: color, height: '100%', borderRadius: 2 }}
        />
      </div>
    </motion.div>
  );
}

export default function LiveInterviewSession({
  interviewId,
  currentQuestion,
  phase,
  onAnswer,
  onEnd,
  loading,
  totalEstimated,
  answersSoFar,
  remainingSeconds,
  timeExpired,
  avatarImage,
  useElevenLabs,
  elevenlabsVoiceId,
  elevenlabsRef,
  speakText,
  isSpeaking,
  greetingMessage,
  clarifyingQuestions,
  clarifyingIndex,
  onClarificationAnswer,
}) {
  const [question, setQuestion] = useState(currentQuestion || '');
  const [avatarState, setAvatarState] = useState('idle');
  const [lastFeedback, setLastFeedback] = useState(null);
  const [sessionPhase, setSessionPhase] = useState('ready');
  const [timer, setTimer] = useState(remainingSeconds || 0);
  const [completed, setCompleted] = useState(false);
  const [webcamMetrics, setWebcamMetrics] = useState(null);
  const [audioLevel, setAudioLevel] = useState(0);
  const [muted, setMuted] = useState(false);
  const [clarifyingSubmitting, setClarifyingSubmitting] = useState(false);

  const streamRef = useRef(null);
  const recorderRef = useRef(null);
  const chunksRef = useRef([]);
  const audioCtxRef = useRef(null);
  const analyserRef = useRef(null);
  const silenceTimerRef = useRef(null);
  const pollRef = useRef(null);
  const speakingRef = useRef(false);
  const isSpeakingRef = useRef(false);
  const handleEndRef = useRef(null);
  const questionSpokenRef = useRef(false);
  const clarificationsSpokenRef = useRef(0);
  // Stable ref so speak effects never go stale when speakText changes
  const speakTextRef = useRef(speakText);
  useEffect(() => { speakTextRef.current = speakText; }, [speakText]);
  const isGreeting = phase === 'greeting';

  useEffect(() => { setTimer(remainingSeconds || 0); }, [remainingSeconds]);

  // Timer countdown
  useEffect(() => {
    if (timer <= 0 || completed || isGreeting) return;
    const interval = setInterval(() => {
      setTimer((t) => {
        if (t <= 1) { clearInterval(interval); handleEndRef.current?.(); return 0; }
        return t - 1;
      });
    }, 1000);
    return () => clearInterval(interval);
  }, [timer, completed, isGreeting]);

  // React to parent `isSpeaking` changes for avatar state transitions
  const prevIsSpeakingRef = useRef(false);
  useEffect(() => {
    // Handle BOTH greeting and session phases — the early return was preventing
    // the avatar from ever showing 'speaking' state during the greeting
    if (isSpeaking && !prevIsSpeakingRef.current) {
      setAvatarState('speaking');
    } else if (!isSpeaking && prevIsSpeakingRef.current) {
      if (!isGreeting) {
        setAvatarState('listening');
        setSessionPhase('listening');
      } else {
        setAvatarState('idle');
      }
    }
    prevIsSpeakingRef.current = isSpeaking;
  }, [isSpeaking, isGreeting]);

  useEffect(() => {
    if (timeExpired && !completed) handleEndRef.current?.();
  }, [timeExpired]);

  useEffect(() => {
    if (currentQuestion) setQuestion(currentQuestion);
  }, [currentQuestion]);

  // Speak the question when it arrives — use speakTextRef to avoid stale closure
  useEffect(() => {
    if (!question || isGreeting || completed || sessionPhase !== 'ready' || questionSpokenRef.current) return;
    questionSpokenRef.current = true;
    const t = setTimeout(() => {
      speakTextRef.current?.(question);
    }, 500);
    return () => clearTimeout(t);
  }, [question, isGreeting, completed, sessionPhase]); // speakText intentionally excluded — use ref

  // Speak clarifying questions during greeting phase
  // Uses speakTextRef (not speakText directly) so the effect is stable and the
  // clarificationsSpokenRef guard doesn't fire again when speakText is recreated
  useEffect(() => {
    if (!isGreeting || !clarifyingQuestions?.length) return;
    if (clarifyingIndex >= clarifyingQuestions.length) return;
    if (clarificationsSpokenRef.current > clarifyingIndex) return;

    clarificationsSpokenRef.current = clarifyingIndex + 1;

    const text = clarifyingIndex === 0 && greetingMessage
      ? `${greetingMessage} ${clarifyingQuestions[0]}`
      : clarifyingQuestions[clarifyingIndex];

    if (!text) return;

    const t = setTimeout(() => {
      speakTextRef.current?.(text);
    }, 600);
    return () => clearTimeout(t);
  }, [isGreeting, clarifyingIndex, clarifyingQuestions, greetingMessage]); // speakText excluded — use ref

  const handleClarifyContinue = async () => {
    if (clarifyingSubmitting) return;
    setClarifyingSubmitting(true);
    try {
      await onClarificationAnswer(null);
    } catch {
    } finally {
      setClarifyingSubmitting(false);
    }
  };

  // Mic setup
  const startMic = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const source = ctx.createMediaStreamSource(stream);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 512;
      source.connect(analyser);
      audioCtxRef.current = ctx;
      analyserRef.current = analyser;
      // Safari doesn't support audio/webm;codecs=opus — pick a supported MIME type
      const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus'
        : MediaRecorder.isTypeSupported('audio/mp4')
        ? 'audio/mp4'
        : '';
      const recorderOpts = mimeType ? { mimeType } : {};
      const recorder = new MediaRecorder(stream, recorderOpts);
      recorder.ondataavailable = (e) => { if (e.data.size > 0) chunksRef.current.push(e.data); };
      recorder.onstop = () => {
        if (chunksRef.current.length > 0) {
          const blob = new Blob(chunksRef.current, { type: mimeType || 'audio/webm' });
          chunksRef.current = [];
          handleAudioBlob(blob);
        }
      };
      recorderRef.current = recorder;
    } catch (e) {
      console.error('Mic access denied:', e);
    }
  }, []);

  useEffect(() => {
    if (question && !completed && !isGreeting) startMic();
    return () => stopMic();
  }, [question, completed, isGreeting]);

  function stopMic() {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
    if (silenceTimerRef.current) { clearTimeout(silenceTimerRef.current); silenceTimerRef.current = null; }
    if (recorderRef.current && recorderRef.current.state !== 'inactive') recorderRef.current.stop();
    if (streamRef.current) { streamRef.current.getTracks().forEach(t => t.stop()); streamRef.current = null; }
    if (audioCtxRef.current) { audioCtxRef.current.close(); audioCtxRef.current = null; }
    isSpeakingRef.current = false;
  }

  // VAD polling
  const startPolling = useCallback(() => {
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = setInterval(() => {
      const analyser = analyserRef.current;
      if (!analyser) return;
      const data = new Uint8Array(analyser.frequencyBinCount);
      analyser.getByteFrequencyData(data);
      const avg = data.reduce((a, b) => a + b, 0) / data.length;
      setAudioLevel(Math.min(1, avg / 100));
      if (sessionPhase !== 'listening' && sessionPhase !== 'processing' && sessionPhase !== 'feedback') return;
      if (avg > SILENCE_THRESHOLD) {
        if (!isSpeakingRef.current) {
          isSpeakingRef.current = true;
          speakingRef.current = true;
          chunksRef.current = [];
          if (recorderRef.current && recorderRef.current.state === 'inactive') recorderRef.current.start(250);
        }
        if (silenceTimerRef.current) { clearTimeout(silenceTimerRef.current); silenceTimerRef.current = null; }
      } else if (isSpeakingRef.current) {
        if (!silenceTimerRef.current) {
          silenceTimerRef.current = setTimeout(() => {
            isSpeakingRef.current = false;
            speakingRef.current = false;
            setAvatarState('thinking');
            setSessionPhase('processing');
            if (recorderRef.current && recorderRef.current.state !== 'inactive') recorderRef.current.stop();
          }, SILENCE_DURATION_MS);
        }
      }
    }, POLL_INTERVAL);
  }, [sessionPhase]);

  useEffect(() => {
    if ((sessionPhase === 'listening' || sessionPhase === 'ready') && !isGreeting) startPolling();
    else if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [sessionPhase, startPolling, isGreeting]);

  const handleAudioBlob = async (blob) => {
    try {
      const result = await onAnswer(blob);
      setLastFeedback(result);
      if (result.is_complete) {
        if (elevenlabsRef?.current) elevenlabsRef.current.stop();
        window.speechSynthesis.cancel();
        setCompleted(true);
        stopMic();
        return;
      }
      setSessionPhase('feedback');
      setAvatarState('thinking');
      questionSpokenRef.current = false;
      setTimeout(() => {
        if (result.next_question) {
          setQuestion(result.next_question);
          setAvatarState('idle');
          setSessionPhase('ready');
        }
      }, 2500);
    } catch (err) {
      console.error('Answer failed:', err);
      setAvatarState('idle');
    }
  };

  const handleToggleMute = () => {
    const newMuted = !muted;
    setMuted(newMuted);
    if (streamRef.current) {
      streamRef.current.getAudioTracks().forEach((t) => { t.enabled = !newMuted; });
    }
  };

  const handleEnd = () => {
    if (completed) return;
    if (elevenlabsRef?.current) elevenlabsRef.current.stop();
    window.speechSynthesis.cancel();
    stopMic();
    setCompleted(true);
    onEnd();
  };
  handleEndRef.current = handleEnd;

  if (completed) return null;

  const statusLabel = isGreeting ? 'Introduction'
    : sessionPhase === 'listening' ? (isSpeakingRef.current ? 'Speaking' : 'Listening')
    : sessionPhase === 'processing' ? 'Analyzing'
    : sessionPhase === 'feedback' ? 'Feedback'
    : 'Ready';

  const statusColor = isGreeting ? 'var(--purple)'
    : sessionPhase === 'listening' && isSpeakingRef.current ? 'var(--green)'
    : sessionPhase === 'processing' ? 'var(--yellow)'
    : sessionPhase === 'feedback' ? 'var(--blue)'
    : 'var(--text3)';

  const questionText = isGreeting
    ? (clarifyingQuestions?.[clarifyingIndex] || 'Preparing your interview...')
    : (question || 'Waiting for question...');

  return (
    <div style={{
      position: 'fixed', inset: 0, zIndex: 100,
      background: '#0a0c10',
      display: 'flex', flexDirection: 'column',
      color: 'var(--text)',
    }}>
      <style>{`
        @keyframes pulseGlow {
          0%,100% { box-shadow: 0 0 20px rgba(52,168,83,0.4), 0 0 40px rgba(52,168,83,0.15); }
          50% { box-shadow: 0 0 35px rgba(52,168,83,0.7), 0 0 70px rgba(52,168,83,0.3); }
        }
        @keyframes audioPulse {
          0%,100% { transform: scaleY(0.3); }
          50% { transform: scaleY(1); }
        }
      `}</style>

      {/* Top bar */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '12px 24px', background: 'rgba(0,0,0,0.4)',
        borderBottom: '1px solid rgba(255,255,255,0.06)',
        flexShrink: 0,
      }}>
        <div className="flex" style={{ alignItems: 'center', gap: 20 }}>
          <div style={{
            fontVariantNumeric: 'tabular-nums', fontSize: '1.2rem', fontWeight: 700,
            color: timer < 60 ? 'var(--red)' : 'var(--text)',
            fontFamily: 'monospace',
          }}>
            {formatTime(timer)}
          </div>
          <div style={{
            padding: '3px 10px', borderRadius: 12,
            background: 'rgba(255,255,255,0.06)',
            fontSize: '0.82rem', color: 'var(--text2)',
          }}>
            Q{isGreeting ? '-' : answersSoFar + 1}
          </div>
          <div className="flex" style={{ alignItems: 'center', gap: 6 }}>
            <div style={{
              width: 8, height: 8, borderRadius: '50%',
              background: statusColor,
              animation: sessionPhase === 'listening' && isSpeakingRef.current ? 'pulse 1s infinite' : 'none',
            }} />
            <span style={{ fontSize: '0.78rem', color: 'var(--text2)' }}>{statusLabel}</span>
          </div>
        </div>
        <div className="flex" style={{ gap: 8 }}>
          <button
            onClick={handleToggleMute}
            className="btn btn-secondary"
            style={{ padding: '8px 14px', fontSize: '0.85rem', borderRadius: 24 }}
            title={muted ? 'Unmute' : 'Mute'}
          >
            {muted ? '🔇' : '🎤'}
          </button>
          <button
            onClick={handleEnd}
            className="btn btn-danger"
            style={{ padding: '8px 20px', fontSize: '0.85rem', borderRadius: 24, fontWeight: 700 }}
          >
            End Call
          </button>
        </div>
      </div>

      {/* Main content */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', padding: 16, gap: 12 }}>
        {/* Video area */}
        <div style={{ flex: 1, display: 'flex', gap: 16, minHeight: 0 }}>
          {/* AI Avatar panel */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="card"
            style={{
              flex: '0 0 280px',
              padding: 20,
              display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
              background: 'rgba(255,255,255,0.03)',
              border: avatarState === 'speaking'
                ? '2px solid rgba(52,168,83,0.4)'
                : '1px solid rgba(255,255,255,0.06)',
              borderRadius: 20,
            }}
          >
            <div style={{
              animation: avatarState === 'speaking' ? 'pulseGlow 1.5s ease-in-out infinite' : 'none',
              borderRadius: '50%', display: 'inline-flex',
            }}>
              <AiAvatar
                imageUrl={avatarImage}
                state={avatarState}
                question={null}
              />
            </div>
            <div style={{ marginTop: 12, fontSize: '0.8rem', color: 'var(--text2)', fontWeight: 500 }}>
              {avatarState === 'speaking' ? 'Speaking...'
                : avatarState === 'listening' ? 'Listening...'
                : avatarState === 'thinking' ? 'Analyzing...'
                : avatarState === 'idle' && isGreeting ? 'Waiting for you...'
                : 'AI Coach'}
            </div>
          </motion.div>

          {/* Webcam panel */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.1 }}
            style={{ flex: 1, position: 'relative', borderRadius: 20, overflow: 'hidden', background: '#000' }}
          >
            <WebcamOverlay onMetrics={setWebcamMetrics} />

            {/* Audio level indicator - stable heights avoid re-render spam */}
            {audioLevel > 0.05 && !muted && (
              <div style={{
                position: 'absolute', bottom: 16, left: 16,
                display: 'flex', alignItems: 'flex-end', gap: 3, height: 24,
              }}>
                {[0.5, 0.8, 1.0, 0.7, 0.4].map((scale, i) => (
                  <div key={i} style={{
                    width: 4, borderRadius: 2, background: 'var(--green)',
                    height: `${Math.max(4, audioLevel * 40 * scale)}px`,
                    animation: 'audioPulse 0.4s ease-in-out infinite',
                    animationDelay: `${i * 0.08}s`,
                    opacity: isSpeakingRef.current ? 1 : 0.4,
                  }} />
                ))}
              </div>
            )}

            {isSpeakingRef.current && sessionPhase === 'listening' && (
              <div style={{
                position: 'absolute', top: 12, left: 12,
                padding: '4px 10px', borderRadius: 12,
                background: 'rgba(52,168,83,0.8)', color: '#fff',
                fontSize: '0.75rem', fontWeight: 600,
              }}>
                You're speaking
              </div>
            )}
          </motion.div>
        </div>

        {/* Question caption bar */}
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          className="card"
          style={{
            padding: '12px 20px', textAlign: 'center', flexShrink: 0,
            background: isGreeting ? 'rgba(147,52,230,0.08)' : 'rgba(26,115,232,0.08)',
            border: isGreeting ? '1px solid rgba(147,52,230,0.15)' : '1px solid rgba(26,115,232,0.15)',
          }}
        >
          {isGreeting && (
            <div style={{ fontSize: '0.75rem', color: 'var(--purple)', fontWeight: 600, marginBottom: 4, textTransform: 'uppercase', letterSpacing: 1 }}>
              Quick Question {clarifyingIndex + 1} of {clarifyingQuestions?.length || 0}
            </div>
          )}
          <div style={{ fontSize: '1rem', color: 'var(--text)', lineHeight: 1.5 }}>
            {questionText}
          </div>
        </motion.div>

        {/* Metrics row + Coaching */}
        <div className="flex" style={{ gap: 12, flexShrink: 0 }}>
          <div className="flex" style={{ gap: 8, flex: 1 }}>
            <MetricCard value={webcamMetrics?.eye_contact_score || 0} label="Eye Contact" color="var(--blue)" icon="👁" />
            <MetricCard value={webcamMetrics?.confidence_score || 0} label="Confidence" color="var(--green)" icon="🧠" />
            <MetricCard value={webcamMetrics?.movement_score || 0} label="Movement" color="var(--yellow)" icon="🔄" />
            <MetricCard value={webcamMetrics?.stress_score !== undefined ? 1 - webcamMetrics.stress_score : 0} label="Composure" color="var(--purple)" icon="😌" />
          </div>
          <div style={{ flex: '0 0 220px' }}>
            <MovementSuggestions metrics={webcamMetrics} />
          </div>
        </div>
      </div>

      {/* Bottom bar */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 16,
        padding: '8px 24px', background: 'rgba(0,0,0,0.4)',
        borderTop: '1px solid rgba(255,255,255,0.06)',
        flexShrink: 0,
      }}>
        <div className="flex" style={{ alignItems: 'center', gap: 8 }}>
          <div style={{ width: 10, height: 10, borderRadius: '50%', background: muted ? 'var(--red)' : 'var(--green)' }} />
          <span style={{ fontSize: '0.82rem', color: 'var(--text2)' }}>Mic {muted ? 'Muted' : 'Live'}</span>
        </div>

        {isGreeting && (
          <button
            className="btn btn-primary"
            onClick={handleClarifyContinue}
            disabled={clarifyingSubmitting}
            style={{ borderRadius: 24, padding: '8px 28px' }}
          >
            {clarifyingSubmitting ? 'Processing...' : 'Continue'}
          </button>
        )}

        {sessionPhase === 'feedback' && lastFeedback && (
          <div style={{ fontSize: '0.82rem', color: 'var(--text2)' }}>
            {lastFeedback.feedback?.slice(0, 80) || 'Answer recorded'}
          </div>
        )}
      </div>
    </div>
  );
}
