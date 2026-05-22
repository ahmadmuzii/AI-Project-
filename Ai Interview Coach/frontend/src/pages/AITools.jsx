import { useState, useRef } from 'react';
import { motion } from 'framer-motion';
import Navbar from '../components/Navbar';
import WebcamOverlay from '../components/WebcamOverlay';
import ResumeManager from '../components/ResumeManager';
import { useAuth } from '../context/AuthContext';
import {
  nlpAnalyzeAnswer,
  resumeAnalyze,
  companyMode,
  studyPlan as apiStudyPlan,
  stressEvaluate,
  adaptiveNextQuestions,
} from '../api/client';

function SectionCard({ title, subtitle, children, delay = 0 }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay }}
      style={{ marginBottom: 24 }}
    >
      <div className="section-title" style={{ fontSize: '1.05rem', marginBottom: 2 }}>{title}</div>
      {subtitle && <div className="section-sub" style={{ margin: '0 0 12px' }}>{subtitle}</div>}
      <div className="card" style={{ padding: 20 }}>
        {children}
      </div>
    </motion.div>
  );
}

function MetricGrid({ items }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, marginTop: 12 }}>
      {items.map((item, i) => (
        <div key={i} className="metric">
          <div className="metric-value" style={{ fontSize: '1.1rem', color: item.color }}>{item.value}</div>
          <div className="metric-label">{item.label}</div>
        </div>
      ))}
    </div>
  );
}

export default function AITools() {
  const { user } = useAuth();
  const [role, setRole] = useState('backend');

  // NLP Analyzer
  const [nlpText, setNlpText] = useState('');
  const [nlpResult, setNlpResult] = useState(null);
  const [nlpLoading, setNlpLoading] = useState(false);
  const [nlpQuestions, setNlpQuestions] = useState([]);

  // Company Mode
  const [company, setCompany] = useState('google');
  const [companyResult, setCompanyResult] = useState(null);
  const [companyLoading, setCompanyLoading] = useState(false);

  // Study Plan
  const [studyResult, setStudyResult] = useState(null);
  const [studyLoading, setStudyLoading] = useState(false);

  // Stress
  const [eyeVal, setEyeVal] = useState(0.6);
  const [moveVal, setMoveVal] = useState(0.6);
  const [voiceVal, setVoiceVal] = useState(0.6);
  const [stressResult, setStressResult] = useState(null);
  const [stressLoading, setStressLoading] = useState(false);

  // Webcam frame
  const [webcamStreaming, setWebcamStreaming] = useState(false);
  const [webcamMetrics, setWebcamMetrics] = useState(null);
  const videoRef = useRef(null);

  async function handleNlp() {
    if (!nlpText.trim()) return;
    setNlpLoading(true);
    try {
      const out = await nlpAnalyzeAnswer(nlpText, role);
      setNlpResult(out);
      const qs = await adaptiveNextQuestions(role, out.weak_topics || ['general'], []);
      setNlpQuestions(qs.questions || []);
    } catch {}
    setNlpLoading(false);
  }

  async function handleCompany() {
    setCompanyLoading(true);
    try {
      const out = await companyMode(company, role);
      setCompanyResult(out);
    } catch {}
    setCompanyLoading(false);
  }

  async function handleStudy() {
    if (!user) return;
    setStudyLoading(true);
    try {
      const out = await apiStudyPlan(user.user_id);
      setStudyResult(out);
    } catch {}
    setStudyLoading(false);
  }

  async function handleStress() {
    setStressLoading(true);
    try {
      const out = await stressEvaluate(eyeVal, moveVal, voiceVal);
      setStressResult(out);
    } catch {}
    setStressLoading(false);
  }

  function handleFileChange(e) {
    const file = e.target.files[0];
    if (!file) return;
    setResumeFile(file);
    const reader = new FileReader();
    reader.onload = (ev) => {
      const text = ev.target.result;
      setResumeText(text);
    };
    reader.readAsText(file);
  }

  function extractPdfText(file) {
    // For PDF files, we just read as text (limited but works for text PDFs)
    // Full PDF parsing requires a library like pdfjs
    // For now, we use the text area for manual paste
    const reader = new FileReader();
    reader.onload = (ev) => setResumeText(ev.target.result);
    reader.readAsText(file);
  }

  const stressLevelColor = stressResult
    ? stressResult.stress_score < 0.4 ? 'var(--green)' : stressResult.stress_score < 0.67 ? 'var(--yellow)' : 'var(--red)'
    : 'var(--text2)';

  return (
    <div>
      <Navbar />
      <div style={{ padding: '80px 24px 40px', maxWidth: 1100, margin: '0 auto' }}>
        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}>
          <div className="flex" style={{ justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
            <div>
              <div className="section-title" style={{ fontSize: '1.3rem' }}>AI Tools</div>
              <div className="section-sub" style={{ margin: 0 }}>Analysis, coaching, and practice tools</div>
            </div>
            <select className="select" value={role} onChange={(e) => setRole(e.target.value)} style={{ width: 160 }}>
              <option value="backend">Backend</option>
              <option value="data science">Data Science</option>
              <option value="frontend">Frontend</option>
              <option value="general">General</option>
            </select>
          </div>

          {/* NLP Answer Analyzer */}
          <SectionCard title="NLP Answer Analyzer" subtitle="Paste an answer for instant NLP feedback">
            <textarea
              className="input"
              placeholder="Paste your interview answer here..."
              value={nlpText}
              onChange={(e) => setNlpText(e.target.value)}
              style={{ minHeight: 100, resize: 'vertical', marginBottom: 12 }}
            />
            <button className="btn btn-primary" onClick={handleNlp} disabled={nlpLoading || !nlpText.trim()}>
              {nlpLoading ? <><span className="spinner" style={{ width: 16, height: 16 }} /> Analyzing...</> : 'Run NLP Analysis'}
            </button>

            {nlpResult && (
              <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} style={{ marginTop: 16 }}>
                <MetricGrid
                  items={[
                    { label: 'Sentiment', value: (nlpResult.sentiment || 'n/a').charAt(0).toUpperCase() + (nlpResult.sentiment || 'n/a').slice(1), color: 'var(--blue)' },
                    { label: 'STAR Score', value: nlpResult.star_score ?? 0, color: 'var(--green)' },
                    { label: 'Coherence', value: nlpResult.coherence_score ?? 0, color: 'var(--purple)' },
                    { label: 'Relevance', value: nlpResult.keyword_relevance ?? 0, color: 'var(--yellow)' },
                  ]}
                />
                {nlpResult.weak_topics?.length > 0 && (
                  <div className="flex gap-8" style={{ marginTop: 12, flexWrap: 'wrap' }}>
                    {nlpResult.weak_topics.map((t, i) => (
                      <span key={i} className="badge badge-yellow">{t}</span>
                    ))}
                  </div>
                )}
                {nlpQuestions.length > 0 && (
                  <div style={{ marginTop: 12 }}>
                    <div style={{ fontWeight: 600, fontSize: '0.85rem', color: 'var(--text)', marginBottom: 8 }}>Suggested Practice Questions</div>
                    {nlpQuestions.map((q, i) => (
                      <div key={i} style={{ padding: '8px 12px', marginBottom: 4, background: 'var(--bg)', borderRadius: 8, fontSize: '0.85rem', color: 'var(--text2)' }}>
                        {q}
                      </div>
                    ))}
                  </div>
                )}
              </motion.div>
            )}
          </SectionCard>

          {/* Resume Analyzer */}
          <SectionCard title="Resume Analyzer" subtitle="Upload, manage, and analyze your resumes" delay={0.1}>
            <ResumeManager />
          </SectionCard>

          {/* Company Mode + Study Plan */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <SectionCard title="Company Mode" subtitle="Practice for specific companies" delay={0.2}>
              <select className="select" value={company} onChange={(e) => setCompany(e.target.value)} style={{ marginBottom: 12 }}>
                <option value="google">Google</option>
                <option value="meta">Meta</option>
                <option value="mckinsey">McKinsey</option>
              </select>
              <button className="btn btn-primary btn-full" onClick={handleCompany} disabled={companyLoading}>
                {companyLoading ? <><span className="spinner" style={{ width: 16, height: 16 }} /> Loading...</> : 'Generate Questions'}
              </button>

              {companyResult && (
                <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} style={{ marginTop: 12 }}>
                  <div className="flex gap-8" style={{ alignItems: 'center', marginBottom: 12 }}>
                    <span className="badge badge-blue">{company.charAt(0).toUpperCase() + company.slice(1)} Mode</span>
                    <span style={{ fontSize: '0.8rem', color: 'var(--text2)' }}>Focus: {companyResult.focus || 'general'}</span>
                  </div>
                  {(companyResult.questions || []).map((q, i) => (
                    <div key={i} style={{ padding: '10px 12px', marginBottom: 6, background: 'var(--bg)', borderRadius: 8, fontSize: '0.85rem', color: 'var(--text2)', borderLeft: '3px solid var(--blue)' }}>
                      {q}
                    </div>
                  ))}
                </motion.div>
              )}
            </SectionCard>

            <SectionCard title="Study Plan" subtitle="Personalized daily tasks" delay={0.25}>
              <button className="btn btn-primary btn-full" onClick={handleStudy} disabled={studyLoading}>
                {studyLoading ? <><span className="spinner" style={{ width: 16, height: 16 }} /> Loading...</> : 'Generate Plan'}
              </button>

              {studyResult && (
                <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} style={{ marginTop: 12 }}>
                  {studyResult.weak_topics?.length > 0 && (
                    <div style={{ marginBottom: 12 }}>
                      <div style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text)', marginBottom: 6 }}>Focus Areas</div>
                      <div className="flex gap-8" style={{ flexWrap: 'wrap' }}>
                        {studyResult.weak_topics.map((t, i) => (
                          <span key={i} className="badge badge-yellow">{t}</span>
                        ))}
                      </div>
                    </div>
                  )}
                  {(studyResult.plan || []).map((p, i) => (
                    <div key={i} className="flex" style={{ gap: 12, padding: '10px 0', borderBottom: i < studyResult.plan.length - 1 ? '1px solid var(--border)' : 'none' }}>
                      <span style={{ background: 'var(--blue)', color: '#fff', width: 24, height: 24, borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.75rem', fontWeight: 600, flexShrink: 0 }}>
                        {p.day || i + 1}
                      </span>
                      <div>
                        <div style={{ fontWeight: 500, fontSize: '0.9rem', color: 'var(--text)' }}>
                          {(p.focus || '').replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())}
                        </div>
                        <div style={{ fontSize: '0.85rem', color: 'var(--text2)', marginTop: 2 }}>{p.task}</div>
                      </div>
                    </div>
                  ))}
                </motion.div>
              )}
            </SectionCard>
          </div>

          {/* Stress & Webcam */}
          <SectionCard title="Stress & Webcam Analysis" subtitle="Evaluate your stress level and analyze webcam feed" delay={0.3}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
              <div>
                <div style={{ fontWeight: 600, fontSize: '0.9rem', marginBottom: 12 }}>Stress Estimator</div>
                <div style={{ marginBottom: 8 }}>
                  <label style={{ fontSize: '0.82rem', color: 'var(--text2)', display: 'block', marginBottom: 4 }}>Eye Contact: {eyeVal.toFixed(2)}</label>
                  <input type="range" min="0" max="1" step="0.05" value={eyeVal} onChange={(e) => setEyeVal(parseFloat(e.target.value))} style={{ width: '100%' }} />
                </div>
                <div style={{ marginBottom: 8 }}>
                  <label style={{ fontSize: '0.82rem', color: 'var(--text2)', display: 'block', marginBottom: 4 }}>Movement: {moveVal.toFixed(2)}</label>
                  <input type="range" min="0" max="1" step="0.05" value={moveVal} onChange={(e) => setMoveVal(parseFloat(e.target.value))} style={{ width: '100%' }} />
                </div>
                <div style={{ marginBottom: 12 }}>
                  <label style={{ fontSize: '0.82rem', color: 'var(--text2)', display: 'block', marginBottom: 4 }}>Voice Energy: {voiceVal.toFixed(2)}</label>
                  <input type="range" min="0" max="1" step="0.05" value={voiceVal} onChange={(e) => setVoiceVal(parseFloat(e.target.value))} style={{ width: '100%' }} />
                </div>
                <button className="btn btn-primary btn-full" onClick={handleStress} disabled={stressLoading}>
                  {stressLoading ? <><span className="spinner" style={{ width: 16, height: 16 }} /> Evaluating...</> : 'Evaluate Stress'}
                </button>

                {stressResult && (
                  <motion.div
                    className="card"
                    style={{ textAlign: 'center', marginTop: 12, borderLeft: `3px solid ${stressLevelColor}` }}
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                  >
                    <div style={{ fontSize: '0.85rem', color: 'var(--text2)', marginBottom: 4 }}>Stress Level</div>
                    <div style={{ fontSize: '2rem', fontWeight: 700, color: stressLevelColor, marginBottom: 4 }}>
                      {stressResult.stress_score?.toFixed(2) || '0.00'}
                    </div>
                    <span className="badge" style={{ background: `${stressLevelColor}20`, color: stressLevelColor, fontSize: '0.85rem', padding: '4px 16px' }}>
                      {(stressResult.stress_level || 'unknown').charAt(0).toUpperCase() + (stressResult.stress_level || 'unknown').slice(1)}
                    </span>
                  </motion.div>
                )}
              </div>

              <div>
                <div style={{ fontWeight: 600, fontSize: '0.9rem', marginBottom: 12 }}>Live Webcam Analysis</div>
                {!webcamStreaming ? (
                  <div className="flex center col" style={{ padding: 40, background: 'var(--bg)', borderRadius: 12, border: '1px dashed var(--border2)' }}>
                    <p style={{ color: 'var(--text2)', fontSize: '0.9rem', marginBottom: 12 }}>Start webcam for real-time analysis</p>
                    <button className="btn btn-primary" onClick={() => setWebcamStreaming(true)}>
                      Start Webcam
                    </button>
                  </div>
                ) : (
                  <>
                    <WebcamOverlay onMetrics={setWebcamMetrics} />
                    {webcamMetrics && (
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginTop: 8 }}>
                        <div className="metric">
                          <div className="metric-value" style={{ fontSize: '0.9rem', color: 'var(--blue)' }}>{webcamMetrics.eye_contact_score.toFixed(2)}</div>
                          <div className="metric-label">Eye Contact</div>
                        </div>
                        <div className="metric">
                          <div className="metric-value" style={{ fontSize: '0.9rem', color: 'var(--green)' }}>{webcamMetrics.movement_score.toFixed(2)}</div>
                          <div className="metric-label">Movement</div>
                        </div>
                        <div className="metric">
                          <div className="metric-value" style={{ fontSize: '0.9rem', color: 'var(--purple)' }}>{webcamMetrics.confidence_score.toFixed(2)}</div>
                          <div className="metric-label">Confidence</div>
                        </div>
                        <div className="metric">
                          <div className="metric-value" style={{ fontSize: '0.9rem', color: 'var(--red)' }}>{webcamMetrics.stress_score.toFixed(2)}</div>
                          <div className="metric-label">Stress</div>
                        </div>
                      </div>
                    )}
                    <button className="btn btn-secondary btn-full" onClick={() => setWebcamStreaming(false)} style={{ marginTop: 8 }}>
                      Stop Webcam
                    </button>
                  </>
                )}
              </div>
            </div>
          </SectionCard>
        </motion.div>
      </div>
    </div>
  );
}
