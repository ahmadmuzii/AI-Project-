import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import ScoreGauge from './ScoreGauge';

function formatTime(seconds) {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}:${s.toString().padStart(2, '0')}`;
}

export default function ChatSession({
  interviewId,
  currentQuestion,
  onAnswer,
  onEnd,
  loading,
  totalEstimated,
  answersSoFar,
  remainingSeconds,
  timeExpired,
}) {
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [completed, setCompleted] = useState(false);
  const [timer, setTimer] = useState(remainingSeconds || 0);
  const chatEndRef = useRef(null);
  const inputRef = useRef(null);
  const handleEndRef = useRef(null);

  useEffect(() => {
    if (!currentQuestion) return;
    setMessages(prev => {
      const last = prev[prev.length - 1];
      if (last && last.type === 'question' && last.content === currentQuestion) return prev;
      return [...prev, { type: 'question', content: currentQuestion }];
    });
  }, [currentQuestion]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

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

  useEffect(() => { setTimer(remainingSeconds || 0); }, [remainingSeconds]);
  useEffect(() => { if (timeExpired && !completed) handleEndRef.current?.(); }, [timeExpired]);

  const handleSubmit = async () => {
    const text = inputText.trim();
    if (!text || submitting) return;
    setInputText('');
    setSubmitting(true);

    setMessages(prev => [...prev, { type: 'answer', content: text }]);

    try {
      const result = await onAnswer(text);
      setMessages(prev => [...prev, {
        type: 'feedback',
        scores: result.scores,
        feedback: result.feedback,
      }]);

      if (result.is_complete) {
        setCompleted(true);
      }
    } catch (err) {
      console.error('Failed to submit answer:', err);
      setMessages(prev => [...prev, {
        type: 'system',
        content: 'Failed to submit answer. Please try again.',
      }]);
    } finally {
      setSubmitting(false);
      inputRef.current?.focus();
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleEnd = () => {
    if (completed) return;
    setCompleted(true);
    onEnd();
  };
  handleEndRef.current = handleEnd;

  if (completed) return null;

  return (
    <div style={{ maxWidth: 800, margin: '0 auto', padding: '80px 24px 0', height: '100vh', display: 'flex', flexDirection: 'column' }}>
      <div className="card" style={{ padding: '12px 20px', marginBottom: 16, display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexShrink: 0 }}>
        <div className="flex" style={{ alignItems: 'center', gap: 16 }}>
          <div style={{ fontVariantNumeric: 'tabular-nums', fontSize: '1.3rem', fontWeight: 700, color: timer < 60 ? 'var(--red)' : 'var(--text)' }}>
            {formatTime(timer)}
          </div>
          <div style={{ color: 'var(--text2)', fontSize: '0.9rem' }}>
            Q{answersSoFar + 1}/{totalEstimated}
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <div style={{
              width: 10, height: 10, borderRadius: '50%',
              background: submitting ? 'var(--yellow)' : 'var(--green)',
              animation: submitting ? 'pulse 1s infinite' : 'none',
            }} />
            <span style={{ fontSize: '0.8rem', color: 'var(--text2)' }}>
              {submitting ? 'Analyzing' : 'Ready'}
            </span>
          </div>
        </div>
        <button className="btn btn-outline" onClick={handleEnd} style={{ color: 'var(--red)', borderColor: 'var(--red)' }}>
          End Interview
        </button>
      </div>

      <div style={{ flex: 1, overflowY: 'auto', paddingRight: 8, marginBottom: 16 }}>
        <AnimatePresence>
          {messages.map((msg, i) => (
            <motion.div
              key={`${msg.type}-${i}`}
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
              style={{ marginBottom: 12 }}
            >
              {msg.type === 'question' && (
                <div style={{ display: 'flex', justifyContent: 'flex-start' }}>
                  <div className="card" style={{ maxWidth: '75%', padding: '16px 20px', background: 'var(--bg3)' }}>
                    <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--blue)', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 6 }}>
                      Interviewer
                    </div>
                    <div style={{ fontSize: '0.95rem', lineHeight: 1.6, color: 'var(--text)' }}>
                      {msg.content}
                    </div>
                  </div>
                </div>
              )}
              {msg.type === 'answer' && (
                <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                  <div style={{ maxWidth: '75%', padding: '14px 18px', borderRadius: '16px 16px 4px 16px', background: 'var(--blue)', color: '#fff', fontSize: '0.95rem', lineHeight: 1.5 }}>
                    {msg.content}
                  </div>
                </div>
              )}
              {msg.type === 'feedback' && (
                <div style={{ display: 'flex', justifyContent: 'flex-start' }}>
                  <div className="card" style={{ maxWidth: '85%', padding: 16, background: 'var(--bg3)' }}>
                    <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--green)', textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 8 }}>
                      Feedback
                    </div>
                    {msg.scores && (
                      <div className="flex" style={{ gap: 8, marginBottom: 10, flexWrap: 'wrap' }}>
                        <ScoreGauge score={msg.scores.content || 0} label="Content" />
                        <ScoreGauge score={msg.scores.relevance || 0} label="Relevance" />
                        <ScoreGauge score={msg.scores.fluency || 0} label="Fluency" />
                        <ScoreGauge score={msg.scores.confidence || 0} label="Confidence" />
                      </div>
                    )}
                    {msg.feedback && (
                      <div style={{ color: 'var(--text2)', fontSize: '0.88rem', lineHeight: 1.6, whiteSpace: 'pre-wrap' }}>
                        {msg.feedback}
                      </div>
                    )}
                  </div>
                </div>
              )}
              {msg.type === 'system' && (
                <div style={{ display: 'flex', justifyContent: 'center' }}>
                  <div style={{ padding: '8px 16px', fontSize: '0.82rem', color: 'var(--text3)', fontStyle: 'italic' }}>
                    {msg.content}
                  </div>
                </div>
              )}
            </motion.div>
          ))}
        </AnimatePresence>

        {submitting && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            style={{ display: 'flex', justifyContent: 'flex-start', marginBottom: 12 }}
          >
            <div className="card" style={{ padding: '12px 16px', background: 'var(--bg3)' }}>
              <div className="flex" style={{ alignItems: 'center', gap: 8 }}>
                <div className="spinner" />
                <span style={{ fontSize: '0.85rem', color: 'var(--text2)' }}>Analyzing your answer...</span>
              </div>
            </div>
          </motion.div>
        )}

        <div ref={chatEndRef} />
      </div>

      <div style={{ paddingBottom: 24, flexShrink: 0 }}>
        <div className="card" style={{ display: 'flex', gap: 12, alignItems: 'center', padding: '12px 16px' }}>
          <input
            ref={inputRef}
            type="text"
            className="input"
            placeholder="Type your answer..."
            value={inputText}
            onChange={e => setInputText(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={submitting || loading}
            autoFocus
          />
          <button
            className="btn btn-primary"
            onClick={handleSubmit}
            disabled={!inputText.trim() || submitting || loading}
            style={{ padding: '10px 20px', flexShrink: 0 }}
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}
