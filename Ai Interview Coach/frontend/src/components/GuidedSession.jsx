import { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import AudioRecorder from './AudioRecorder';
import ScoreGauge from './ScoreGauge';
import { ElevenLabsSpeaker } from '../utils/ElevenLabsSpeaker';

function formatTime(seconds) {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}:${s.toString().padStart(2, '0')}`;
}

export default function GuidedSession({
  interviewId,
  currentQuestion,
  onAnswer,
  onEnd,
  loading,
  totalEstimated,
  answersSoFar,
  remainingSeconds,
  timeExpired,
  useElevenLabs,
  elevenlabsVoiceId,
}) {
  const [question, setQuestion] = useState(currentQuestion || '');
  const [recording, setRecording] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [completed, setCompleted] = useState(false);
  const [lastFeedback, setLastFeedback] = useState(null);
  const [timer, setTimer] = useState(remainingSeconds || 0);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const synthRef = useRef(null);
  const elevenlabsRef = useRef(null);
  const handleEndRef = useRef(null);

  // Cleanup TTS on unmount
  useEffect(() => {
    return () => {
      if (elevenlabsRef.current) elevenlabsRef.current.stop();
      if (window.speechSynthesis) window.speechSynthesis.cancel();
    };
  }, []);

  useEffect(() => {
    if (useElevenLabs && elevenlabsVoiceId) {
      if (!elevenlabsRef.current) {
        elevenlabsRef.current = new ElevenLabsSpeaker({
          voiceId: elevenlabsVoiceId,
          playbackRate: 1.15,
          onStart: () => setIsSpeaking(true),
          onEnd: () => setIsSpeaking(false),
        });
      } else {
        elevenlabsRef.current.setVoice(elevenlabsVoiceId);
      }
    }
  }, [useElevenLabs, elevenlabsVoiceId]);

  // Sync question prop
  useEffect(() => {
    if (currentQuestion) setQuestion(currentQuestion);
  }, [currentQuestion]);

  // Timer countdown — use handleEndRef to avoid stale closure
  useEffect(() => {
    if (timer <= 0 || completed) return;
    const interval = setInterval(() => {
      setTimer((t) => {
        if (t <= 1) {
          clearInterval(interval);
          handleEndRef.current?.();
          return 0;
        }
        return t - 1;
      });
    }, 1000);
    return () => clearInterval(interval);
  }, [timer, completed]);

  // Update timer when prop changes
  useEffect(() => {
    setTimer(remainingSeconds || 0);
  }, [remainingSeconds]);

  // Auto-expire
  useEffect(() => {
    if (timeExpired && !completed) handleEndRef.current?.();
  }, [timeExpired]);

  // TTS
  const speakQuestion = useCallback((text) => {
    if (!text) return;

    if (useElevenLabs && elevenlabsRef.current) {
      elevenlabsRef.current.speak(text);
      return;
    }

    if (!window.speechSynthesis) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1.0;
    utterance.pitch = 1;
    utterance.onstart = () => setIsSpeaking(true);
    utterance.onend = () => setIsSpeaking(false);
    window.speechSynthesis.speak(utterance);
  }, [useElevenLabs]);

  useEffect(() => {
    if (question && !completed) {
      const timer = setTimeout(() => speakQuestion(question), 300);
      return () => clearTimeout(timer);
    }
  }, [question, completed]);

  const handleRecordingComplete = async (blob) => {
    setSubmitting(true);
    try {
      const result = await onAnswer(blob);
      setLastFeedback(result);
      if (result.is_complete) {
        setCompleted(true);
      } else if (result.next_question) {
        setQuestion(result.next_question);
      }
    } catch (err) {
      console.error('Answer submission failed:', err);
    } finally {
      setSubmitting(false);
    }
  };

  const handleEnd = () => {
    if (completed) return;
    setCompleted(true);
    if (elevenlabsRef.current) elevenlabsRef.current.stop();
    if (window.speechSynthesis) window.speechSynthesis.cancel();
    onEnd();
  };
  handleEndRef.current = handleEnd;

  if (completed) return null;

  return (
    <div style={{ maxWidth: 800, margin: '0 auto', padding: '80px 24px 40px' }}>
      {/* Top bar */}
      <div className="card" style={{ padding: '12px 20px', marginBottom: 16, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div className="flex" style={{ alignItems: 'center', gap: 16 }}>
          <div style={{ fontVariantNumeric: 'tabular-nums', fontSize: '1.3rem', fontWeight: 700, color: timer < 60 ? 'var(--red)' : 'var(--text)' }}>
            {formatTime(timer)}
          </div>
          <div style={{ color: 'var(--text2)', fontSize: '0.9rem' }}>
            Q{answersSoFar + 1}/{totalEstimated}
          </div>
        </div>
        <button className="btn btn-outline" onClick={handleEnd} style={{ color: 'var(--red)', borderColor: 'var(--red)' }}>
          End Interview
        </button>
      </div>

      {/* Question */}
      <AnimatePresence mode="wait">
        <motion.div
          key={question}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -20 }}
          transition={{ duration: 0.35 }}
          className="card"
          style={{ padding: 28, marginBottom: 16 }}
        >
          <div className="flex" style={{ alignItems: 'flex-start', gap: 12 }}>
            <div style={{ flex: 1 }}>
              <div style={{ color: 'var(--text2)', fontSize: '0.8rem', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 8 }}>
                Current Question
              </div>
              <div style={{ fontSize: '1.15rem', fontWeight: 600, lineHeight: 1.5, color: 'var(--text)' }}>
                {question}
              </div>
            </div>
            <button
              onClick={() => speakQuestion(question)}
              className="btn btn-outline"
              style={{ padding: '8px 12px', flexShrink: 0, fontSize: '0.85rem', minWidth: 0 }}
              title={isSpeaking ? 'Playing...' : 'Replay question'}
            >
              {isSpeaking ? '🔊' : '🔈'}
            </button>
          </div>
        </motion.div>
      </AnimatePresence>

      {/* Recording */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.3, delay: 0.15 }}
        className="card"
        style={{ padding: 24, marginBottom: 16 }}
      >
        <AudioRecorder
          onRecordingComplete={handleRecordingComplete}
          disabled={submitting}
        />
        {submitting && (
          <div className="flex center" style={{ padding: 20 }}>
            <div className="spinner" />
            <span style={{ marginLeft: 12, color: 'var(--text2)' }}>Analyzing your answer...</span>
          </div>
        )}
      </motion.div>

      {/* Last feedback */}
      <AnimatePresence>
        {lastFeedback && (
          <motion.div
            key={lastFeedback.qa_id || 'fb'}
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 16 }}
            className="card"
            style={{ padding: 20 }}
          >
            <div style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text2)', marginBottom: 12 }}>
              Last Answer Feedback
            </div>
            <div className="flex" style={{ gap: 16, marginBottom: 12, flexWrap: 'wrap' }}>
              <ScoreGauge score={lastFeedback.scores?.content || 0} label="Content" />
              <ScoreGauge score={lastFeedback.scores?.relevance || 0} label="Relevance" />
              <ScoreGauge score={lastFeedback.scores?.fluency || 0} label="Fluency" />
              <ScoreGauge score={lastFeedback.scores?.confidence || 0} label="Confidence" />
            </div>
            {lastFeedback.feedback && (
              <div style={{ color: 'var(--text2)', fontSize: '0.9rem', lineHeight: 1.6, whiteSpace: 'pre-wrap' }}>
                {lastFeedback.feedback.slice(0, 300)}
                {lastFeedback.feedback.length > 300 ? '...' : ''}
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
