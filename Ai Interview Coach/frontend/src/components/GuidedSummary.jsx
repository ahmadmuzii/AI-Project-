import { useState } from 'react';
import { motion } from 'framer-motion';
import ScoreGauge from './ScoreGauge';
import { useNavigate } from 'react-router-dom';

export default function GuidedSummary({ interview, onRetry }) {
  const navigate = useNavigate();
  const [expanded, setExpanded] = useState(null);

  const interviewData = interview?.interview || {};
  const qaPairs = interview?.qa_pairs?.filter((q) => q.transcript) || [];
  const summaryRaw = interviewData.summary;
  let summary = { summary: '', strengths: [], top_improvements: [], action_plan: [], readiness_estimate: 'N/A' };
  try {
    if (summaryRaw) summary = typeof summaryRaw === 'string' ? JSON.parse(summaryRaw) : summaryRaw;
  } catch {}

  const overallScore = interviewData.overall_score || 0;

  const scoreColor = overallScore >= 80 ? 'var(--green)' : overallScore >= 60 ? 'var(--yellow)' : overallScore >= 40 ? 'var(--orange)' : 'var(--red)';
  const readinessColor = {
    'Not ready': 'var(--red)',
    'Needs work': 'var(--orange)',
    'Almost ready': 'var(--yellow)',
    'Ready': 'var(--green)',
    'Highly prepared': 'var(--blue)',
  }[summary.readiness_estimate] || 'var(--text2)';

  return (
    <div style={{ maxWidth: 800, margin: '0 auto', padding: '80px 24px 40px' }}>
      <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
        {/* Hero Score */}
        <div className="card" style={{ textAlign: 'center', padding: 40, marginBottom: 16 }}>
          <div style={{ fontSize: '1.5rem', fontWeight: 700, marginBottom: 4, color: 'var(--text)' }}>
            Interview Complete
          </div>
          <div style={{ color: 'var(--text2)', marginBottom: 24 }}>
            {qaPairs.length} questions answered &middot; {interviewData.target_company ? `${interviewData.target_company} focus` : 'General practice'}
          </div>
          <div className="flex center" style={{ gap: 24, flexWrap: 'wrap' }}>
            <ScoreGauge score={overallScore / 100} label="Overall" color={scoreColor} />
          </div>
          <div style={{ marginTop: 16 }}>
            <span
              style={{
                display: 'inline-block',
                padding: '4px 16px',
                borderRadius: 20,
                background: readinessColor + '18',
                color: readinessColor,
                fontWeight: 600,
                fontSize: '0.9rem',
              }}
            >
              {summary.readiness_estimate}
            </span>
          </div>
        </div>

        {/* Score breakdown */}
        <div className="card" style={{ padding: 24, marginBottom: 16 }}>
          <div className="section-title" style={{ marginBottom: 16 }}>Score Breakdown</div>
          <div className="flex" style={{ gap: 16, flexWrap: 'wrap', justifyContent: 'center' }}>
            {qaPairs.length > 0 && (
              <>
                <ScoreGauge score={qaPairs.reduce((s, q) => s + (q.content_score || 0), 0) / qaPairs.length} label="Content" color="var(--blue)" />
                <ScoreGauge score={qaPairs.reduce((s, q) => s + (q.relevance_score || 0), 0) / qaPairs.length} label="Relevance" color="var(--green)" />
                <ScoreGauge score={qaPairs.reduce((s, q) => s + (q.fluency_score || 0), 0) / qaPairs.length} label="Fluency" color="var(--purple)" />
                <ScoreGauge score={qaPairs.reduce((s, q) => s + (q.confidence_score || 0), 0) / qaPairs.length} label="Confidence" color="var(--orange)" />
              </>
            )}
          </div>
        </div>

        {/* LLM Summary */}
        {summary.summary && (
          <div className="card" style={{ padding: 24, marginBottom: 16 }}>
            <div className="section-title" style={{ marginBottom: 8 }}>Coach Feedback</div>
            <p style={{ color: 'var(--text)', lineHeight: 1.7, fontSize: '0.95rem', whiteSpace: 'pre-wrap' }}>
              {summary.summary}
            </p>
          </div>
        )}

        {/* Strengths */}
        {summary.strengths?.length > 0 && (
          <div className="card" style={{ padding: 24, marginBottom: 16 }}>
            <div className="section-title" style={{ marginBottom: 12 }}>Strengths</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {summary.strengths.map((s, i) => (
                <div key={i} className="flex" style={{ alignItems: 'flex-start', gap: 10 }}>
                  <span style={{ color: 'var(--green)', fontSize: '1.1rem' }}>&#10003;</span>
                  <span style={{ color: 'var(--text)' }}>{s}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Top Improvements */}
        {summary.top_improvements?.length > 0 && (
          <div className="card" style={{ padding: 24, marginBottom: 16 }}>
            <div className="section-title" style={{ marginBottom: 12 }}>Top Improvements</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {summary.top_improvements.map((imp, i) => (
                <div key={i} className="flex" style={{ alignItems: 'flex-start', gap: 10 }}>
                  <span style={{ color: 'var(--yellow)', fontSize: '1.1rem', fontWeight: 700 }}>{i + 1}.</span>
                  <span style={{ color: 'var(--text)' }}>{imp}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Action Plan */}
        {summary.action_plan?.length > 0 && (
          <div className="card" style={{ padding: 24, marginBottom: 16 }}>
            <div className="section-title" style={{ marginBottom: 12 }}>Action Plan</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {summary.action_plan.map((step, i) => (
                <div key={i} className="flex" style={{ alignItems: 'flex-start', gap: 10 }}>
                  <span style={{
                    width: 24, height: 24, borderRadius: 12,
                    background: 'var(--blue-bg, #e3f2fd)', color: 'var(--blue)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontSize: '0.75rem', fontWeight: 700, flexShrink: 0,
                  }}>
                    {i + 1}
                  </span>
                  <span style={{ color: 'var(--text)', paddingTop: 2 }}>{step}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Per-question accordion */}
        <div className="card" style={{ padding: 24, marginBottom: 16 }}>
          <div className="section-title" style={{ marginBottom: 12 }}>Question Review</div>
          {qaPairs.length === 0 && (
            <p style={{ color: 'var(--text2)', textAlign: 'center', padding: 20 }}>No answers recorded.</p>
          )}
          {qaPairs.map((qa, i) => (
            <div key={qa.id} style={{ borderBottom: i < qaPairs.length - 1 ? '1px solid var(--border)' : 'none', marginBottom: i < qaPairs.length - 1 ? 8 : 0 }}>
              <button
                onClick={() => setExpanded(expanded === qa.id ? null : qa.id)}
                style={{
                  width: '100%', background: 'none', border: 'none',
                  display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                  padding: '10px 4px', cursor: 'pointer', color: 'var(--text)',
                  textAlign: 'left', gap: 12,
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, minWidth: 0 }}>
                  <span style={{
                    width: 26, height: 26, borderRadius: 13,
                    background: 'var(--bg3)', display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontSize: '0.75rem', fontWeight: 600, flexShrink: 0,
                  }}>
                    {i + 1}
                  </span>
                  <span style={{ fontSize: '0.9rem', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {qa.question?.slice(0, 80)}{qa.question?.length > 80 ? '...' : ''}
                  </span>
                </div>
                <div className="flex" style={{ alignItems: 'center', gap: 8, flexShrink: 0 }}>
                  <span style={{ color: 'var(--blue)', fontWeight: 600, fontSize: '0.85rem' }}>
                    {Math.round((qa.content_score || 0) * 100)}%
                  </span>
                  <span style={{ color: 'var(--text2)', transition: 'transform 0.2s', transform: expanded === qa.id ? 'rotate(180deg)' : '' }}>
                    &#9660;
                  </span>
                </div>
              </button>
              {expanded === qa.id && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: 'auto', opacity: 1 }}
                  transition={{ duration: 0.25 }}
                  style={{ overflow: 'hidden', padding: '0 4px 12px' }}
                >
                  <div style={{ marginBottom: 8, fontSize: '0.9rem', color: 'var(--text)', lineHeight: 1.5, fontStyle: 'italic' }}>
                    Q: {qa.question}
                  </div>
                  {qa.transcript && (
                    <div style={{ marginBottom: 8, fontSize: '0.9rem', color: 'var(--text2)', lineHeight: 1.5, padding: '8px 12px', background: 'var(--bg3)', borderRadius: 6 }}>
                      {qa.transcript}
                    </div>
                  )}
                  {qa.feedback && (
                    <div style={{ fontSize: '0.85rem', color: 'var(--text2)', lineHeight: 1.5, whiteSpace: 'pre-wrap' }}>
                      {qa.feedback.slice(0, 400)}{qa.feedback.length > 400 ? '...' : ''}
                    </div>
                  )}
                  <div className="flex" style={{ gap: 12, marginTop: 8, flexWrap: 'wrap' }}>
                    <span style={{ fontSize: '0.8rem', color: 'var(--blue)' }}>Content: {Math.round((qa.content_score || 0) * 100)}%</span>
                    <span style={{ fontSize: '0.8rem', color: 'var(--green)' }}>Relevance: {Math.round((qa.relevance_score || 0) * 100)}%</span>
                    <span style={{ fontSize: '0.8rem', color: 'var(--purple)' }}>Fluency: {Math.round((qa.fluency_score || 0) * 100)}%</span>
                    <span style={{ fontSize: '0.8rem', color: 'var(--orange)' }}>Confidence: {Math.round((qa.confidence_score || 0) * 100)}%</span>
                  </div>
                </motion.div>
              )}
            </div>
          ))}
        </div>

        {/* Actions */}
        <div className="flex" style={{ justifyContent: 'center', gap: 12, marginTop: 8 }}>
          <button className="btn btn-primary" onClick={onRetry}>
            Practice Again
          </button>
          <button className="btn btn-outline" onClick={() => navigate('/dashboard')}>
            View Dashboard
          </button>
        </div>
      </motion.div>
    </div>
  );
}
