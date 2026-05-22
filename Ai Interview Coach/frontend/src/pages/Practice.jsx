import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import Navbar from '../components/Navbar';
import AudioPlayer from '../components/AudioPlayer';
import { useAuth } from '../context/AuthContext';
import {
  listRecordings as apiListRecordings,
  listSessions,
  recordingAudioUrl,
  clearSessions,
  deleteRecording,
  listGuidedInterviews,
} from '../api/client';

function confidenceLabel(v) {
  if (v == null || v === 0) return null;
  if (v >= 0.7) return { label: 'High', color: 'var(--green)' };
  if (v >= 0.4) return { label: 'Medium', color: 'var(--yellow)' };
  return { label: 'Low', color: 'var(--red)' };
}

function scoreColor(s) {
  if (s == null || s === 0) return 'var(--text3)';
  if (s >= 70) return 'var(--green)';
  if (s >= 40) return 'var(--yellow)';
  return 'var(--red)';
}

export default function Practice() {
  const { user } = useAuth();
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState('practice'); // 'practice' or 'guided'
  const [sessions, setSessions] = useState([]);
  const [currentSession, setCurrentSession] = useState(null);
  const [recordings, setRecordings] = useState([]);
  const [guidedInterviews, setGuidedInterviews] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [expandedRecId, setExpandedRecId] = useState(null);

  useEffect(() => {
    if (!user) return;
    refreshData();
  }, [user]);

  useEffect(() => {
    if (activeTab === 'practice' && currentSession) {
      refreshRecordings();
    }
  }, [activeTab, currentSession]);

  async function refreshData() {
    setLoading(true);
    setError(null);
    try {
      const sessList = await listSessions();
      // Filter sessions to practice type for the Self-Practice tab
      const practiceSessions = sessList.filter(s => s.session_type === 'practice');
      setSessions(practiceSessions);
      if (practiceSessions.length > 0) {
        setCurrentSession(practiceSessions[0].id);
      }

      const guidedList = await listGuidedInterviews();
      setGuidedInterviews(guidedList || []);
    } catch (e) {
      setError('Could not load Review Center history.');
    } finally {
      setLoading(false);
    }
  }

  async function refreshRecordings() {
    if (!currentSession) return;
    try {
      const data = await apiListRecordings(currentSession);
      setRecordings(data.recordings || []);
      setExpandedRecId(null);
    } catch {
      setRecordings([]);
    }
  }

  async function handleDeleteRecording(recordingId) {
    if (!window.confirm('Are you sure you want to delete this recording?')) return;
    try {
      await deleteRecording(recordingId);
      refreshRecordings();
    } catch (e) {
      setError(e.message);
    }
  }

  async function handleClearAllPractice() {
    if (!window.confirm('WARNING: This will permanently delete all self-practice sessions and audio files. Continue?')) return;
    try {
      await clearSessions();
      setSessions([]);
      setCurrentSession(null);
      setRecordings([]);
    } catch (e) {
      setError('Failed to clear sessions.');
    }
  }

  const currentPracticeSess = sessions.find(s => s.id === currentSession);

  return (
    <div>
      <Navbar />
      <div style={{ padding: '90px 24px 40px', maxWidth: 1200, margin: '0 auto' }}>
        
        {/* Page Title Header */}
        <div className="flex" style={{ justifyContent: 'space-between', alignItems: 'center', marginBottom: 24, flexWrap: 'wrap', gap: 16 }}>
          <div>
            <div className="flex center gap-12" style={{ justifyContent: 'flex-start' }}>
              <h1 className="section-title" style={{ fontSize: '1.75rem', margin: 0 }}>Review Center</h1>
              <span className="grok-badge">⚡ Grok Evaluated</span>
            </div>
            <p className="section-sub" style={{ margin: '4px 0 0 0' }}>Analyze recordings, review speech analytics, and study guided session history</p>
          </div>
          {activeTab === 'practice' && sessions.length > 0 && (
            <button className="btn btn-danger" onClick={handleClearAllPractice} style={{ fontSize: '0.8rem', padding: '8px 16px' }}>
              Clear All Practice
            </button>
          )}
        </div>

        {error && (
          <div className="card" style={{ borderLeft: '4px solid var(--red)', color: 'var(--red)', marginBottom: 20, fontSize: '0.9rem' }}>
            <div className="flex" style={{ justifyContent: 'space-between', alignItems: 'center' }}>
              <span>{error}</span>
              <button className="btn btn-secondary" style={{ padding: '2px 8px', fontSize: '0.78rem' }} onClick={() => setError(null)}>Dismiss</button>
            </div>
          </div>
        )}

        {/* Dynamic Framer Motion Tabs */}
        <div style={{ position: 'relative', marginBottom: 24 }}>
          <div className="tab-bar" style={{ maxWidth: 440, padding: 4 }}>
            <button
              className={`tab-item ${activeTab === 'practice' ? 'active' : ''}`}
              onClick={() => setActiveTab('practice')}
              style={{ position: 'relative', overflow: 'visible', zIndex: 1 }}
            >
              <span style={{ position: 'relative', zIndex: 2 }}>Self-Practice Logs</span>
              {activeTab === 'practice' && (
                <motion.div
                  layoutId="activeReviewTab"
                  transition={{ type: 'spring', stiffness: 350, damping: 28 }}
                  style={{
                    position: 'absolute', inset: 0, background: 'var(--blue)', borderRadius: 'var(--radius)', zIndex: 1
                  }}
                />
              )}
            </button>
            <button
              className={`tab-item ${activeTab === 'guided' ? 'active' : ''}`}
              onClick={() => setActiveTab('guided')}
              style={{ position: 'relative', overflow: 'visible', zIndex: 1 }}
            >
              <span style={{ position: 'relative', zIndex: 2 }}>Guided Interviews</span>
              {activeTab === 'guided' && (
                <motion.div
                  layoutId="activeReviewTab"
                  transition={{ type: 'spring', stiffness: 350, damping: 28 }}
                  style={{
                    position: 'absolute', inset: 0, background: 'var(--blue)', borderRadius: 'var(--radius)', zIndex: 1
                  }}
                />
              )}
            </button>
          </div>
        </div>

        {loading && (
          <div className="flex center" style={{ padding: 100 }}>
            <div className="spinner" />
          </div>
        )}

        {/* Tab View Panels */}
        {!loading && (
          <AnimatePresence mode="wait">
            
            {/* SELF-PRACTICE TAB */}
            {activeTab === 'practice' && (
              <motion.div
                key="practice-pane"
                className="flex"
                style={{ gap: 24, flexWrap: 'wrap' }}
                initial={{ opacity: 0, y: 15 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -15 }}
                transition={{ duration: 0.35 }}
              >
                
                {/* Left Sidebar */}
                <div style={{ width: '100%', maxWidth: 280, flexShrink: 0 }}>
                  <div className="card" style={{ padding: 18, height: 'fit-content' }}>
                    <div style={{ fontSize: '0.82rem', fontWeight: 700, color: 'var(--text2)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 12 }}>
                      Sessions ({sessions.length})
                    </div>
                    
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 6, maxHeight: 460, overflowY: 'auto' }}>
                      {sessions.length === 0 && (
                        <p style={{ color: 'var(--text3)', fontSize: '0.85rem', textAlign: 'center', padding: 24 }}>No practice sessions</p>
                      )}
                      
                      {sessions.map((s) => {
                        const conf = confidenceLabel(s.avg_confidence);
                        const isActive = currentSession === s.id;
                        return (
                          <motion.div
                            key={s.id}
                            onClick={() => { setCurrentSession(s.id); setError(null); }}
                            style={{
                              padding: '12px 14px',
                              borderRadius: 'var(--radius)',
                              cursor: 'pointer',
                              border: `1px solid ${isActive ? 'rgba(37,99,235,0.25)' : 'var(--border)'}`,
                              background: isActive ? 'rgba(37, 99, 235, 0.05)' : 'transparent',
                              boxShadow: isActive ? 'var(--glow-blue)' : 'none',
                              transition: 'all 0.2s',
                            }}
                            whileHover={{ scale: 1.01, background: isActive ? 'rgba(37, 99, 235, 0.07)' : 'rgba(255,255,255,0.01)' }}
                          >
                            <div className="flex" style={{ alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 }}>
                              <span style={{ fontWeight: 700, fontSize: '0.85rem', color: isActive ? 'var(--blue)' : 'var(--text)' }}>
                                Session #{s.session_number || s.id}
                              </span>
                              {s.overall_score > 0 && (
                                <span style={{ fontWeight: 800, fontSize: '0.85rem', color: scoreColor(s.overall_score) }}>
                                  {Math.round(s.overall_score)}%
                                </span>
                              )}
                            </div>
                            
                            <div style={{ fontSize: '0.75rem', color: 'var(--text2)', textTransform: 'capitalize', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                              {s.role !== 'general' ? s.role : 'General Coach'} · {s.topic !== 'general' ? s.topic : 'General'}
                            </div>

                            <div className="flex" style={{ justifyContent: 'space-between', alignItems: 'center', marginTop: 6, fontSize: '0.72rem', color: 'var(--text3)' }}>
                              <span>{s.started_at?.slice(0, 10) || ''}</span>
                              {conf && (
                                <span style={{ color: conf.color, fontWeight: 600 }}>
                                  ● {conf.label}
                                </span>
                              )}
                            </div>
                          </motion.div>
                        );
                      })}
                    </div>
                  </div>
                </div>

                {/* Right Content Area */}
                <div style={{ flex: 1, minWidth: 280 }}>
                  
                  {/* Session Overview Header */}
                  {currentPracticeSess && (
                    <div className="card-glass" style={{ padding: 20, marginBottom: 16, border: '1px solid var(--border)' }}>
                      <div className="flex" style={{ justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 12 }}>
                        <div>
                          <div style={{ fontSize: '0.8rem', color: 'var(--blue)', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Selected Session</div>
                          <h3 className="section-title" style={{ margin: '2px 0 4px', fontSize: '1.25rem' }}>
                            Session #{currentPracticeSess.session_number || currentPracticeSess.id}
                          </h3>
                          <div style={{ fontSize: '0.82rem', color: 'var(--text2)' }}>
                            Focus Area: <span style={{ textTransform: 'capitalize', fontWeight: 600, color: 'var(--text)' }}>{currentPracticeSess.role}</span> (Topic: {currentPracticeSess.topic})
                          </div>
                        </div>
                        {currentPracticeSess.overall_score > 0 && (
                          <div style={{ display: 'flex', alignItems: 'center', gap: 12, background: 'rgba(255,255,255,0.02)', padding: '8px 16px', borderRadius: 'var(--radius)', border: '1px solid var(--border)' }}>
                            <div style={{ textAlign: 'right' }}>
                              <div style={{ fontSize: '1.4rem', fontWeight: 800, color: scoreColor(currentPracticeSess.overall_score) }}>
                                {Math.round(currentPracticeSess.overall_score)}%
                              </div>
                              <div style={{ fontSize: '0.7rem', color: 'var(--text2)', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600 }}>Average Score</div>
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Recordings List */}
                  <div>
                    {recordings.length > 0 && (
                      <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                        {recordings.map((rec, i) => {
                          const isExpanded = expandedRecId === rec.id;
                          return (
                            <motion.div
                              key={rec.id}
                              className="card"
                              style={{ padding: 18, border: isExpanded ? '1px solid var(--border2)' : '1px solid var(--border)' }}
                              initial={{ opacity: 0, y: 10 }}
                              animate={{ opacity: 1, y: 0 }}
                              transition={{ delay: i * 0.05 }}
                              layout
                            >
                              <div className="flex" style={{ alignItems: 'center', justifyContent: 'space-between', gap: 16, flexWrap: 'wrap' }}>
                                <div className="flex gap-12" style={{ alignItems: 'center' }}>
                                  <div style={{ width: 36, height: 36, borderRadius: 10, background: 'rgba(37,99,235,0.08)', display: 'flex', alignItems: 'center', justify: 'center', color: 'var(--blue)', fontWeight: 800, fontSize: '0.9rem' }}>
                                    #{i + 1}
                                  </div>
                                  <div>
                                    <div style={{ fontWeight: 700, fontSize: '0.88rem', color: 'var(--text)' }}>Recording Answer</div>
                                    <div style={{ fontSize: '0.72rem', color: 'var(--text3)', marginTop: 2 }}>{rec.created_at?.slice(11, 16) || ''} · {rec.created_at?.slice(0, 10) || ''}</div>
                                  </div>
                                </div>
                                
                                <div className="flex gap-12" style={{ alignItems: 'center' }}>
                                  <AudioPlayer src={recordingAudioUrl(rec.id)} />
                                  
                                  <button
                                    className="btn btn-secondary"
                                    onClick={() => setExpandedRecId(isExpanded ? null : rec.id)}
                                    style={{ padding: '8px 14px', fontSize: '0.78rem', gap: 4 }}
                                  >
                                    {isExpanded ? 'Collapse' : 'Analyze'} {isExpanded ? '▲' : '▼'}
                                  </button>

                                  <button
                                    onClick={() => handleDeleteRecording(rec.id)}
                                    style={{
                                      background: 'none', border: 'none', cursor: 'pointer',
                                      color: 'var(--text3)', fontSize: '1rem', padding: 6,
                                      transition: 'color 0.15s', lineHeight: 1,
                                    }}
                                    onMouseEnter={e => e.currentTarget.style.color = 'var(--red)'}
                                    onMouseLeave={e => e.currentTarget.style.color = 'var(--text3)'}
                                    title="Delete recording"
                                  >
                                    🗑️
                                  </button>
                                </div>
                              </div>

                              {/* Accordion Expansion (Framer Motion) */}
                              <AnimatePresence>
                                {isExpanded && (rec.transcript || rec.feedback) && (
                                  <motion.div
                                    initial={{ height: 0, opacity: 0 }}
                                    animate={{ height: 'auto', opacity: 1 }}
                                    exit={{ height: 0, opacity: 0 }}
                                    transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
                                    style={{ overflow: 'hidden', marginTop: 16 }}
                                  >
                                    <div style={{ borderTop: '1px solid var(--border)', paddingTop: 16, display: 'flex', flexDirection: 'column', gap: 12 }}>
                                      
                                      {rec.transcript && (
                                        <div style={{ background: 'var(--bg3)', padding: 16, borderRadius: 'var(--radius)', border: '1px solid var(--border)' }}>
                                          <div style={{ fontWeight: 700, fontSize: '0.8rem', color: 'var(--text2)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 6 }}>Transcript</div>
                                          <p style={{ fontSize: '0.86rem', color: 'var(--text)', lineHeight: 1.5 }}>"{rec.transcript}"</p>
                                        </div>
                                      )}
                                      
                                      {rec.feedback && (
                                        <div style={{ background: 'rgba(0,230,118,0.02)', padding: 16, borderRadius: 'var(--radius)', border: '1px solid rgba(0,230,118,0.1)', borderLeft: '4px solid var(--green)' }}>
                                          <div style={{ fontWeight: 700, fontSize: '0.8rem', color: '#00E676', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 6 }}>Grok AI Feedback</div>
                                          <p style={{ fontSize: '0.86rem', color: 'var(--text)', whiteSpace: 'pre-wrap', lineHeight: 1.5 }}>{rec.feedback}</p>
                                        </div>
                                      )}

                                      {rec.word_analysis?.length > 0 && (
                                        <div style={{ background: 'var(--bg3)', borderRadius: 'var(--radius)', border: '1px solid var(--border)', overflow: 'hidden' }}>
                                          <div style={{ padding: '12px 16px 8px', fontWeight: 700, fontSize: '0.8rem', color: 'var(--text2)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Word-Level analysis</div>
                                          <div style={{ overflowX: 'auto' }}>
                                            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.8rem' }}>
                                              <thead>
                                                <tr style={{ background: 'rgba(255,255,255,0.02)', borderBottom: '1px solid var(--border)' }}>
                                                  <th style={{ padding: '8px 16px', textAlign: 'left', color: 'var(--text2)', fontWeight: 600 }}>Timestamp</th>
                                                  <th style={{ padding: '8px 16px', textAlign: 'left', color: 'var(--text2)', fontWeight: 600 }}>Word</th>
                                                  <th style={{ padding: '8px 16px', textAlign: 'left', color: 'var(--text2)', fontWeight: 600 }}>Issue Detected</th>
                                                  <th style={{ padding: '8px 16px', textAlign: 'left', color: 'var(--text2)', fontWeight: 600 }}>Grok Suggestion</th>
                                                </tr>
                                              </thead>
                                              <tbody>
                                                {rec.word_analysis.map((w, j) => (
                                                  <tr key={j} style={{ borderBottom: j < rec.word_analysis.length - 1 ? '1px solid var(--border)' : 'none' }}>
                                                    <td style={{ padding: '10px 16px', fontFamily: 'monospace', color: 'var(--text3)' }}>{w.timestamp || '0:00'}</td>
                                                    <td style={{ padding: '10px 16px', fontWeight: 700, color: 'var(--text)' }}>{w.word}</td>
                                                    <td style={{ padding: '10px 16px' }}>
                                                      <span className="badge badge-red" style={{ fontSize: '0.7rem' }}>{w.issue}</span>
                                                    </td>
                                                    <td style={{ padding: '10px 16px', color: 'var(--text2)' }}>{w.suggestion}</td>
                                                  </tr>
                                                ))}
                                              </tbody>
                                            </table>
                                          </div>
                                        </div>
                                      )}
                                    </div>
                                  </motion.div>
                                )}
                              </AnimatePresence>
                            </motion.div>
                          );
                        })}
                      </div>
                    )}

                    {recordings.length === 0 && currentSession && (
                      <div className="card" style={{ textAlign: 'center', padding: 64 }}>
                        <div style={{ fontSize: '3rem', marginBottom: 12 }}>🎙️</div>
                        <h4 style={{ fontWeight: 700, marginBottom: 4 }}>Empty Practice Session</h4>
                        <p style={{ color: 'var(--text2)', fontSize: '0.88rem', maxWidth: 360, margin: '0 auto 16px' }}>
                          You have created this practice session but haven't uploaded any recordings yet.
                        </p>
                        <button className="btn btn-primary" onClick={() => navigate('/interview/new')}>
                          Record an Answer
                        </button>
                      </div>
                    )}

                    {!currentSession && sessions.length === 0 && (
                      <div className="card" style={{ textAlign: 'center', padding: 80 }}>
                        <div style={{ fontSize: '3.5rem', marginBottom: 16 }}>🎯</div>
                        <h3 style={{ fontWeight: 800, fontSize: '1.25rem', marginBottom: 6 }}>Start Self-Practice</h3>
                        <p style={{ color: 'var(--text2)', fontSize: '0.88rem', maxWidth: 440, margin: '0 auto 20px' }}>
                          Self-practice logs individual recordings, checks for filler words, evaluates response rates, and gets detailed speech scores.
                        </p>
                        <button className="btn btn-primary" onClick={() => navigate('/interview/new')}>
                          Practice New Question
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              </motion.div>
            )}

            {/* GUIDED INTERVIEWS TAB */}
            {activeTab === 'guided' && (
              <motion.div
                key="guided-pane"
                initial={{ opacity: 0, y: 15 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -15 }}
                transition={{ duration: 0.35 }}
              >
                {guidedInterviews.length === 0 ? (
                  <div className="card" style={{ textAlign: 'center', padding: 80 }}>
                    <div style={{ fontSize: '3.5rem', marginBottom: 16, filter: 'drop-shadow(0 0 10px rgba(139,92,246,0.2))' }}>💼</div>
                    <h3 style={{ fontWeight: 800, fontSize: '1.25rem', marginBottom: 6 }}>No Guided Interviews Yet</h3>
                    <p style={{ color: 'var(--text2)', fontSize: '0.88rem', maxWidth: 480, margin: '0 auto 24px' }}>
                      Take a complete guided mock interview mimicking real company standards. The Grok engine will ask follow-up questions based on your resume and generate an overall scorecard!
                    </p>
                    <button className="btn btn-primary" onClick={() => navigate('/interview/new')}>
                      Start Mock Interview
                    </button>
                  </div>
                ) : (
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(340px, 1fr))', gap: 16 }}>
                    {guidedInterviews.map((gi, idx) => {
                      const score = gi.overall_score;
                      const hasScore = score != null && score > 0;
                      return (
                        <motion.div
                          key={gi.id}
                          className="card"
                          whileHover={{ y: -3, scale: 1.01, border: '1px solid var(--border2)' }}
                          transition={{ type: 'spring', stiffness: 350, damping: 25 }}
                          initial={{ opacity: 0, y: 12 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: idx * 0.05 }}
                          style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between', minHeight: 220 }}
                        >
                          <div>
                            <div className="flex" style={{ justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 12 }}>
                              <span className="badge badge-purple" style={{ fontSize: '0.7rem' }}>
                                #{gi.id} Mock Interview
                              </span>
                              {gi.status === 'in_progress' ? (
                                <span className="badge badge-yellow" style={{ fontSize: '0.7rem' }}>In Progress</span>
                              ) : (
                                <span className="badge badge-green" style={{ fontSize: '0.7rem' }}>Completed</span>
                              )}
                            </div>
                            
                            <h4 style={{ fontSize: '1.05rem', fontWeight: 800, margin: '0 0 4px 0', color: 'var(--text)' }}>
                              {gi.aim || 'General Interview'}
                            </h4>
                            
                            {gi.target_company && (
                              <div style={{ fontSize: '0.8rem', color: 'var(--text2)', fontWeight: 600, marginBottom: 8 }}>
                                Target Company: <span style={{ color: 'var(--blue)' }}>{gi.target_company}</span>
                              </div>
                            )}

                            <div style={{ display: 'flex', gap: 10, fontSize: '0.75rem', color: 'var(--text3)', flexWrap: 'wrap', marginBottom: 14 }}>
                              <span>Difficulty: <strong style={{ color: 'var(--text2)', textTransform: 'capitalize' }}>{gi.difficulty}</strong></span>
                              <span>•</span>
                              <span>{gi.duration_minutes} mins</span>
                              <span>•</span>
                              <span>{gi.recording_count} answers</span>
                            </div>
                          </div>

                          <div style={{ borderTop: '1px solid var(--border)', paddingTop: 14, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <div>
                              {hasScore ? (
                                <div>
                                  <span style={{ fontSize: '1.25rem', fontWeight: 800, color: scoreColor(score) }}>{Math.round(score)}%</span>
                                  <span style={{ fontSize: '0.7rem', color: 'var(--text3)', display: 'block', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Overall Match</span>
                                </div>
                              ) : (
                                <span style={{ fontSize: '0.75rem', color: 'var(--text3)', italic: true }}>Pending completion</span>
                              )}
                            </div>

                            <button
                              className="btn btn-secondary"
                              onClick={() => navigate(`/interview/${gi.id}`)}
                              style={{ padding: '6px 14px', fontSize: '0.78rem' }}
                            >
                              {gi.status === 'in_progress' ? 'Resume Session' : 'View Summary'}
                            </button>
                          </div>
                        </motion.div>
                      );
                    })}
                  </div>
                )}
              </motion.div>
            )}
          </AnimatePresence>
        )}
      </div>
    </div>
  );
}
