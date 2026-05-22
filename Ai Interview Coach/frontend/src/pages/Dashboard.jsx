import { useState, useEffect, useMemo } from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import Navbar from '../components/Navbar';
import { useAuth } from '../context/AuthContext';
import { getDashboard, getLeaderboard } from '../api/client';

function SimpleLineChart({ data, keys, colors }) {
  if (!data.length) return null;

  const w = 600, h = 220, pad = 40;
  const allValues = data.flatMap((d) => keys.map((k) => d[k] || 0));
  const max = Math.max(...allValues, 0.1);
  const xStep = (w - pad * 2) / Math.max(data.length - 1, 1);

  function path(key, color) {
    const pts = data.map((d, i) => ({
      x: pad + i * xStep,
      y: h - pad - ((d[key] || 0) / max) * (h - pad * 2),
    }));
    let dStr = pts.length === 1
      ? `M${pts[0].x},${pts[0].y} L${pts[0].x + 1},${pts[0].y}`
      : pts.map((p, i) => `${i === 0 ? 'M' : 'L'}${p.x},${p.y}`).join(' ');
    
    return (
      <g key={key}>
        {/* Glow effect under the path */}
        <path
          d={dStr}
          fill="none"
          stroke={color}
          strokeWidth={6}
          strokeLinecap="round"
          strokeLinejoin="round"
          style={{ opacity: 0.15, filter: 'blur(4px)' }}
        />
        <path
          d={dStr}
          fill="none"
          stroke={color}
          strokeWidth={2.5}
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </g>
    );
  }

  return (
    <svg viewBox={`0 0 ${w} ${h}`} style={{ width: '100%', maxHeight: 240, overflow: 'visible' }}>
      {Array.from({ length: 5 }).map((_, i) => (
        <line
          key={i}
          x1={pad}
          x2={w - pad}
          y1={pad + (i * (h - pad * 2)) / 4}
          y2={pad + (i * (h - pad * 2)) / 4}
          stroke="var(--border)"
          strokeWidth={1}
          strokeDasharray="4 4"
        />
      ))}
      {keys.map((k, i) => path(k, colors[i]))}
      {keys.map((k, i) => (
        data.map((d, j) => (
          <motion.circle
            key={`${k}-${j}`}
            cx={pad + j * xStep}
            cy={h - pad - ((d[k] || 0) / max) * (h - pad * 2)}
            r={4}
            fill={colors[i]}
            stroke="var(--bg)"
            strokeWidth={1.5}
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: j * 0.05, type: 'spring', stiffness: 300 }}
            whileHover={{ scale: 7 }}
          />
        ))
      ))}
    </svg>
  );
}

function HeatmapBars({ data }) {
  const entries = Object.entries(data || {}).sort((a, b) => b[1] - a[1]);
  if (!entries.length) return null;

  return (
    <div className="card" style={{ marginBottom: 16 }}>
      <div className="section-title" style={{ marginBottom: 4 }}>Weak Topic Heatmap</div>
      <div className="section-sub" style={{ marginBottom: 16 }}>Topics needing focus based on evaluations</div>
      {entries.map(([topic, val]) => {
        const pct = Math.min(val, 1);
        const barColor = pct > 0.7 ? 'var(--red)' : pct > 0.4 ? 'var(--yellow)' : 'var(--green)';
        return (
          <div key={topic} style={{ marginBottom: 12 }}>
            <div className="flex" style={{ justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: 4 }}>
              <span style={{ color: 'var(--text2)', fontWeight: 500, textTransform: 'capitalize' }}>{topic}</span>
              <span style={{ color: 'var(--text)', fontWeight: 600 }}>{Math.round(pct * 100)}%</span>
            </div>
            <div className="progress-track" style={{ height: 10 }}>
              <motion.div
                className="progress-fill"
                initial={{ width: 0 }}
                animate={{ width: `${pct * 100}%` }}
                transition={{ duration: 1, ease: [0.16, 1, 0.3, 1] }}
                style={{ background: barColor }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}

const medals = ['🥇', '🥈', '🥉'];

export default function Dashboard() {
  const { user } = useAuth();
  const navigate = useNavigate();
  const [role, setRole] = useState('backend');
  const [dash, setDash] = useState(null);
  const [board, setBoard] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!user) return;
    setLoading(true);
    setError(null);
    Promise.all([
      getDashboard().catch((e) => { setError(e.message); return null; }),
      getLeaderboard(role).catch(() => null),
    ]).then(([d, b]) => {
      setDash(d);
      setBoard(b);
      setLoading(false);
    });
  }, [user, role]);

  const series = dash?.series || [];
  const heatmap = dash?.heatmap || {};
  const streak = dash?.streak_days ?? 0;
  const readiness = dash?.readiness_days;
  const percentile = dash?.comparison_percentile;
  const leaders = board?.leaders || [];

  const chartData = useMemo(() => series.map((s) => ({
    date: s.date?.slice(0, 10) || '',
    overall: s.overall || 0,
    confidence: s.confidence || 0,
  })), [series]);

  return (
    <div>
      <Navbar />
      <div style={{ padding: '90px 24px 40px', maxWidth: 1100, margin: '0 auto' }}>
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
        >
          {/* Header Row */}
          <div className="flex" style={{ justifyContent: 'space-between', alignItems: 'center', marginBottom: 28, flexWrap: 'wrap', gap: 16 }}>
            <div>
              <div className="flex center gap-12" style={{ justifyContent: 'flex-start', flexWrap: 'wrap' }}>
                <h1 className="section-title" style={{ fontSize: '1.75rem', margin: 0 }}>Dashboard</h1>
                <span className="grok-badge">
                  <span style={{ fontSize: '0.9rem', marginRight: 2 }}>⚡</span>
                  Grok-2 AI Engine
                </span>
              </div>
              <div className="section-sub" style={{ margin: '4px 0 0 0' }}>Your interview preparation metrics at a glance</div>
            </div>
            
            <div className="flex" style={{ gap: 12, alignItems: 'center' }}>
              <select
                className="select"
                value={role}
                onChange={(e) => setRole(e.target.value)}
                style={{ width: 150, padding: '10px 32px 10px 14px', fontSize: '0.85rem' }}
              >
                <option value="backend">Backend</option>
                <option value="data science">Data Science</option>
                <option value="frontend">Frontend</option>
                <option value="general">General</option>
              </select>
              <button
                className="btn btn-primary"
                onClick={() => navigate('/interview/new')}
                style={{ fontSize: '0.85rem', padding: '10px 20px' }}
              >
                Start Interview
              </button>
            </div>
          </div>

          {loading && (
            <div className="flex center" style={{ padding: 120 }}>
              <div className="spinner" />
            </div>
          )}

          {error && (
            <div className="card" style={{ textAlign: 'center', padding: 40, borderLeft: '4px solid var(--red)', color: 'var(--red)' }}>
              <p style={{ fontWeight: 600 }}>{error}</p>
            </div>
          )}

          {!loading && !error && !series.length && (
            <div className="card" style={{ textAlign: 'center', padding: 80 }}>
              <div style={{ fontSize: '3.5rem', marginBottom: 16, filter: 'drop-shadow(0 0 10px rgba(37,99,235,0.2))' }}>📈</div>
              <h3 style={{ fontSize: '1.2rem', fontWeight: 700, marginBottom: 8 }}>No Performance History</h3>
              <p style={{ color: 'var(--text2)', maxWidth: 440, margin: '0 auto 24px', fontSize: '0.9rem' }}>
                Complete a mock interview or self-practice recording to analyze your skills with the Grok engine and view analytics.
              </p>
              <button className="btn btn-primary" onClick={() => navigate('/interview/new')}>
                Take First Interview
              </button>
            </div>
          )}

          {!loading && !error && series.length > 0 && (
            <>
              {/* Metrics Grid */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 16, marginBottom: 20 }}>
                <motion.div
                  className="metric"
                  whileHover={{ y: -3, scale: 1.01 }}
                  transition={{ type: 'spring', stiffness: 400, damping: 25 }}
                >
                  <div className="metric-value" style={{ color: 'var(--blue)', filter: 'drop-shadow(0 0 10px rgba(37,99,235,0.25))' }}>
                    {streak}🔥
                  </div>
                  <div className="metric-label">Day Streak</div>
                </motion.div>
                
                <motion.div
                  className="metric"
                  whileHover={{ y: -3, scale: 1.01 }}
                  transition={{ type: 'spring', stiffness: 400, damping: 25 }}
                >
                  <div className="metric-value" style={{ color: 'var(--green)', filter: 'drop-shadow(0 0 10px rgba(0,230,118,0.25))' }}>
                    {readiness ?? 'n/a'}
                  </div>
                  <div className="metric-label">Readiness (days)</div>
                </motion.div>
                
                <motion.div
                  className="metric"
                  whileHover={{ y: -3, scale: 1.01 }}
                  transition={{ type: 'spring', stiffness: 400, damping: 25 }}
                >
                  <div className="metric-value" style={{ color: 'var(--purple)', filter: 'drop-shadow(0 0 10px rgba(139,92,246,0.25))' }}>
                    {percentile != null ? `${percentile}%` : 'n/a'}
                  </div>
                  <div className="metric-label">Peer Percentile</div>
                </motion.div>
              </div>

              {/* Progress Graph */}
              <div className="card" style={{ marginBottom: 20 }}>
                <div className="flex" style={{ justifyContent: 'space-between', alignItems: 'center', marginBottom: 12, flexWrap: 'wrap', gap: 8 }}>
                  <div>
                    <h3 className="section-title" style={{ margin: 0 }}>Progress Curve</h3>
                    <p style={{ fontSize: '0.8rem', color: 'var(--text2)', margin: 0 }}>Overall score and confidence trends</p>
                  </div>
                  <div className="flex gap-16" style={{ fontSize: '0.8rem', color: 'var(--text2)' }}>
                    <span className="flex center gap-8"><span style={{ width: 8, height: 8, borderRadius: '50%', background: 'var(--blue)', display: 'inline-block' }} /> Overall Score</span>
                    <span className="flex center gap-8"><span style={{ width: 8, height: 8, borderRadius: '50%', background: 'var(--green)', display: 'inline-block' }} /> Confidence</span>
                  </div>
                </div>
                <SimpleLineChart
                  data={chartData}
                  keys={['overall', 'confidence']}
                  colors={['var(--blue)', 'var(--green)']}
                />
              </div>

              {/* Weak topics and Leaderboard Split */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(450px, 1fr))', gap: 20 }}>
                <HeatmapBars data={heatmap} />

                <div className="card" style={{ height: 'fit-content' }}>
                  <div className="section-title" style={{ marginBottom: 4 }}>Leaderboard</div>
                  <div className="section-sub" style={{ marginBottom: 16 }}>Top performing mock interviews for <span style={{ textTransform: 'capitalize', fontWeight: 600, color: 'var(--blue)' }}>{role}</span></div>
                  
                  {leaders.length === 0 && (
                    <p style={{ color: 'var(--text2)', textAlign: 'center', padding: 30, fontSize: '0.9rem' }}>No leaderboard data for this role yet.</p>
                  )}
                  
                  {leaders.length > 0 && (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                      {leaders.map((l, i) => (
                        <motion.div
                          key={l.rank || i}
                          className="flex"
                          style={{
                            alignItems: 'center',
                            justifyContent: 'space-between',
                            padding: '12px 14px',
                            borderRadius: 'var(--radius)',
                            background: i === 0 ? 'rgba(37,99,235,0.04)' : 'transparent',
                            border: `1px solid ${i === 0 ? 'rgba(37,99,235,0.1)' : 'transparent'}`,
                          }}
                          initial={{ opacity: 0, x: -10 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ duration: 0.3, delay: i * 0.06 }}
                          whileHover={{ x: 3, background: 'rgba(255,255,255,0.02)' }}
                        >
                          <div className="flex gap-12" style={{ alignItems: 'center' }}>
                            <span style={{ fontSize: '1.25rem', width: 28, textAlign: 'center' }}>
                              {i < 3 ? medals[i] : `#${l.rank}`}
                            </span>
                            <span style={{ fontWeight: 600, color: i === 0 ? 'var(--text)' : 'var(--text2)', fontSize: '0.9rem' }}>
                              {l.name}
                            </span>
                          </div>
                          <span style={{ fontWeight: 700, color: 'var(--blue)', fontSize: '0.95rem' }}>{l.score}</span>
                        </motion.div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </>
          )}
        </motion.div>
      </div>
    </div>
  );
}
