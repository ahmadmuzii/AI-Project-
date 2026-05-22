import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import Navbar from '../components/Navbar';
import { listGuidedInterviews } from '../api/client';
import { useAuth } from '../context/AuthContext';

const STATUS_LABEL = {
  completed: { label: 'Completed', color: 'var(--green)' },
  in_progress: { label: 'In Progress', color: 'var(--yellow)' },
  setup: { label: 'Setup', color: 'var(--text3)' },
};

function scoreColor(s) {
  if (s == null) return 'var(--text3)';
  if (s >= 80) return 'var(--green)';
  if (s >= 60) return 'var(--yellow)';
  return 'var(--red)';
}

export default function HistoryPage() {
  const { user } = useAuth();
  const [interviews, setInterviews] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!user) return;
    listGuidedInterviews()
      .then(setInterviews)
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [user]);

  return (
    <div>
      <Navbar />
      <div style={{ padding: '80px 24px 40px', maxWidth: 900, margin: '0 auto' }}>
        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}>
          <h1 style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--text)', marginBottom: 4 }}>Interview History</h1>
          <p style={{ color: 'var(--text2)', fontSize: '0.9rem', marginBottom: 24 }}>Review your past guided interviews</p>

          {loading ? (
            <div className="flex center" style={{ padding: 60 }}>
              <div className="spinner" />
            </div>
          ) : interviews.length === 0 ? (
            <div className="card" style={{ textAlign: 'center', padding: 60 }}>
              <div style={{ fontSize: '3rem', marginBottom: 12, opacity: 0.6 }}>📋</div>
              <div style={{ fontWeight: 500, color: 'var(--text)', marginBottom: 4 }}>No interviews yet</div>
              <p style={{ color: 'var(--text2)', fontSize: '0.9rem' }}>
                Complete a <Link to="/interview/new" style={{ color: 'var(--blue)' }}>guided interview</Link> to see it here.
              </p>
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {interviews.map((i, idx) => {
                const st = STATUS_LABEL[i.status] || { label: i.status, color: 'var(--text3)' };
                return (
                  <Link
                    key={i.id}
                    to={i.status === 'completed' ? `/interview/${i.id}` : '#'}
                    style={{ textDecoration: 'none' }}
                  >
                    <motion.div
                      className="card"
                      initial={{ opacity: 0, y: 12 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: idx * 0.04 }}
                      style={{
                        display: 'flex', alignItems: 'center', gap: 16,
                        cursor: i.status === 'completed' ? 'pointer' : 'default',
                      }}
                    >
                      <div style={{
                        width: 44, height: 44, borderRadius: 12, flexShrink: 0,
                        background: i.overall_score != null && i.overall_score >= 60
                          ? 'linear-gradient(135deg, var(--green), #34D399)'
                          : 'linear-gradient(135deg, var(--blue), var(--purple))',
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        fontSize: '1.1rem', color: '#fff', fontWeight: 700,
                      }}>
                        {i.overall_score != null ? Math.round(i.overall_score) : '—'}
                      </div>
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ fontWeight: 600, fontSize: '0.92rem', color: 'var(--text)', marginBottom: 2 }}>
                          {i.aim ? i.aim.slice(0, 60) : `Interview #${i.id}`}
                          {i.aim && i.aim.length > 60 ? '...' : ''}
                        </div>
                        <div style={{ display: 'flex', gap: 12, fontSize: '0.8rem', color: 'var(--text2)', flexWrap: 'wrap' }}>
                          <span>{i.started_at?.slice(0, 10) || ''}</span>
                          {i.target_company && <span>• {i.target_company}</span>}
                          <span>• {i.duration_minutes} min</span>
                          {i.recording_count > 0 && <span>• {i.recording_count} answers</span>}
                        </div>
                      </div>
                      <div style={{ textAlign: 'right', flexShrink: 0 }}>
                        <div style={{
                          fontSize: '0.75rem', fontWeight: 600, padding: '3px 10px',
                          borderRadius: 12, background: `${st.color}18`, color: st.color,
                          display: 'inline-block',
                        }}>
                          {st.label}
                        </div>
                        {i.overall_score != null && (
                          <div style={{ fontSize: '0.75rem', color: scoreColor(i.overall_score), marginTop: 4, fontWeight: 600 }}>
                            {Math.round(i.overall_score)}%
                          </div>
                        )}
                      </div>
                    </motion.div>
                  </Link>
                );
              })}
            </div>
          )}
        </motion.div>
      </div>
    </div>
  );
}
