import { motion } from 'framer-motion';
import ScoreGauge from './ScoreGauge';
import WordAnalysis from './WordAnalysis';

export default function AnalysisResults({ analysis }) {
  if (!analysis) return null;

  const scores = analysis.scores || {};
  const nlp = analysis.nlp || {};
  const transcript = analysis.transcript || '';
  const feedback = analysis.feedback || '';
  const wordAnalysis = analysis.word_analysis || [];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
    >
      <div className="section-title" style={{ margin: '20px 0 12px' }}>Analysis Results</div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, marginBottom: 16 }}>
        <ScoreGauge score={scores.fluency || 0} label="Fluency" color="var(--blue)" delay={0} />
        <ScoreGauge score={scores.confidence || 0} label="Confidence" color="var(--green)" delay={0.1} />
        <ScoreGauge score={scores.composure || 0} label="Composure" color="var(--purple)" delay={0.2} />
        <ScoreGauge score={scores.overall || 0} label="Overall" color="var(--red)" delay={0.3} />
      </div>

      <div className="card" style={{ marginBottom: 16 }}>
        <div className="section-title" style={{ marginBottom: 12 }}>NLP Analysis</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12 }}>
          <div className="metric">
            <div className="metric-value" style={{ fontSize: '1.1rem' }}>{(nlp.sentiment || 'n/a').charAt(0).toUpperCase() + (nlp.sentiment || 'n/a').slice(1)}</div>
            <div className="metric-label">Sentiment</div>
          </div>
          <div className="metric">
            <div className="metric-value" style={{ fontSize: '1.1rem' }}>{nlp.star_score || 0}</div>
            <div className="metric-label">STAR Score</div>
          </div>
          <div className="metric">
            <div className="metric-value" style={{ fontSize: '1.1rem' }}>{nlp.coherence_score || 0}</div>
            <div className="metric-label">Coherence</div>
          </div>
          <div className="metric">
            <div className="metric-value" style={{ fontSize: '1.1rem' }}>{nlp.keyword_relevance || 0}</div>
            <div className="metric-label">Relevance</div>
          </div>
        </div>
        {nlp.weak_topics && nlp.weak_topics.length > 0 && (
          <div className="flex gap-8" style={{ marginTop: 12, flexWrap: 'wrap' }}>
            {nlp.weak_topics.map((t, i) => (
              <span key={i} className="badge badge-yellow">{t}</span>
            ))}
          </div>
        )}
      </div>

      {transcript && (
        <div className="card" style={{ marginBottom: 16 }}>
          <div className="section-title" style={{ marginBottom: 8 }}>Transcript</div>
          <p style={{ color: 'var(--text2)', lineHeight: 1.7, fontSize: '0.9rem' }}>{transcript}</p>
        </div>
      )}

      {feedback && (
        <div className="card" style={{ marginBottom: 16, borderLeft: '3px solid var(--green)' }}>
          <div className="section-title" style={{ marginBottom: 8 }}>Coaching Feedback</div>
          <p style={{ color: 'var(--text2)', lineHeight: 1.7, fontSize: '0.9rem', whiteSpace: 'pre-wrap' }}>{feedback}</p>
        </div>
      )}

      {wordAnalysis.length > 0 && <WordAnalysis items={wordAnalysis} />}
    </motion.div>
  );
}
