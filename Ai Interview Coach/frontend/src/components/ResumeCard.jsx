import { useState } from 'react';
import { motion } from 'framer-motion';
import ResumeAnalyzer from './ResumeAnalyzer';
import { analyzeResumeProfile, analyzeResume } from '../api/client';

export default function ResumeCard({
  resume,
  onSetPrimary,
  onDelete,
}) {
  const [expanded, setExpanded] = useState(false);
  const [profile, setProfile] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [profileLoading, setProfileLoading] = useState(false);
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const [selectedRole, setSelectedRole] = useState('');

  if (!resume) return null;

  async function handleAnalyze() {
    setProfileLoading(true);
    setProfile(null);
    setAnalysis(null);
    try {
      const [profileResult, analysisResult] = await Promise.all([
        analyzeResumeProfile(resume.id),
        selectedRole ? analyzeResume(resume.id, selectedRole) : Promise.resolve(null),
      ]);
      setProfile(profileResult);
      if (analysisResult) setAnalysis(analysisResult);
    } catch (e) {
      console.error('Analysis failed:', e);
    } finally {
      setProfileLoading(false);
    }
  }

  async function handleRoleChange(role) {
    setSelectedRole(role);
    if (!profile) return;
    setAnalysisLoading(true);
    setAnalysis(null);
    try {
      const result = await analyzeResume(resume.id, role);
      setAnalysis(result);
    } catch (e) {
      console.error('Role analysis failed:', e);
    } finally {
      setAnalysisLoading(false);
    }
  }

  return (
    <motion.div
      className="card"
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      style={{
        marginBottom: 8, padding: '14px 16px',
        borderLeft: resume.is_primary ? '3px solid var(--blue)' : '1px solid var(--border)',
      }}
    >
      <div className="flex" style={{ alignItems: 'center', justifyContent: 'space-between', gap: 12 }}>
        <div className="flex" style={{ alignItems: 'center', gap: 12, flex: 1, minWidth: 0 }}>
          <span style={{ fontSize: '1.3rem' }}>📄</span>
          <div style={{ minWidth: 0 }}>
            <div className="flex gap-8" style={{ alignItems: 'center', flexWrap: 'wrap' }}>
              <span style={{ fontWeight: 600, fontSize: '0.9rem', color: 'var(--text)' }}>{resume.name}</span>
              {resume.is_primary && (
                <span className="badge badge-blue" style={{ fontSize: '0.7rem' }}>Primary</span>
              )}
            </div>
            <div style={{ fontSize: '0.78rem', color: 'var(--text2)', marginTop: 2 }}>
              {resume.skills?.length || 0} skills
              {resume.experience_years ? ` · ${resume.experience_years}y exp` : ''}
              {resume.created_at ? ` · ${resume.created_at.slice(0, 10)}` : ''}
            </div>
          </div>
        </div>

        <div className="flex gap-8" style={{ flexShrink: 0 }}>
          <button className="btn btn-secondary" onClick={() => {
            setExpanded(!expanded);
            if (!expanded && !profile) handleAnalyze();
          }} style={{ padding: '4px 10px', fontSize: '0.78rem' }}>
            {expanded ? 'Collapse' : 'Analyze'}
          </button>
          <button className="btn btn-secondary" onClick={() => onDelete?.(resume.id)}
            style={{ padding: '4px 10px', fontSize: '0.78rem', color: 'var(--red)' }}
            title="Delete resume">
            🗑
          </button>
        </div>
      </div>

      {expanded && (
        <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }}
          transition={{ duration: 0.2 }} style={{ marginTop: 12, overflow: 'hidden' }}>
          <ResumeAnalyzer
            profile={profile}
            analysis={analysis}
            profileLoading={profileLoading}
            analysisLoading={analysisLoading}
            selectedRole={selectedRole}
            onRoleChange={handleRoleChange}
            onAnalyze={handleAnalyze}
          />

          <div className="flex gap-8" style={{ marginTop: 12, flexWrap: 'wrap', borderTop: '1px solid var(--border)', paddingTop: 12 }}>
            {!resume.is_primary && (
              <button className="btn btn-secondary" onClick={() => onSetPrimary?.(resume.id)}
                style={{ fontSize: '0.78rem', padding: '6px 14px' }}>
                Set as Primary
              </button>
            )}
            <button className="btn btn-secondary" onClick={() => onDelete?.(resume.id)}
              style={{ fontSize: '0.78rem', padding: '6px 14px', color: 'var(--red)' }}>
              Delete
            </button>
          </div>
        </motion.div>
      )}
    </motion.div>
  );
}
