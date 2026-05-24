import { useState } from 'react';
import { motion } from 'framer-motion';
import Navbar from '../components/Navbar';
import ResumeManager from '../components/ResumeManager';
import { useAuth } from '../context/AuthContext';
import {
  nlpAnalyzeAnswer,
  companyMode,
  studyPlan as apiStudyPlan,
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
      <div className="apple-glass" style={{ padding: 20 }}>
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

  return (
    <div>
      <Navbar />
      <div style={{ padding: '90px 24px 40px', maxWidth: 1100, margin: '0 auto' }}>
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

        </motion.div>
      </div>
    </div>
  );
}
