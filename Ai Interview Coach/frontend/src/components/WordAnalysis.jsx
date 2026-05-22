import { motion } from 'framer-motion';

export default function WordAnalysis({ items = [] }) {
  if (!items.length) {
    return (
      <div className="card" style={{ color: 'var(--text2)', fontSize: '0.9rem' }}>
        No word-level issues flagged. Clean delivery!
      </div>
    );
  }

  return (
    <motion.div
      className="card" style={{ padding: 0, overflow: 'hidden' }}
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
    >
      <div style={{ padding: '16px 16px 8px', fontWeight: 600, fontSize: '0.9rem', color: 'var(--text)' }}>
        <span className="badge badge-yellow" style={{ marginRight: 8 }}>🔍</span>
        Word-Level Analysis
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
          <thead>
            <tr style={{ background: 'var(--bg2)' }}>
              <th style={{ padding: '8px 12px', textAlign: 'left', color: 'var(--text2)', fontWeight: 500 }}>Time</th>
              <th style={{ padding: '8px 12px', textAlign: 'left', color: 'var(--text2)', fontWeight: 500 }}>Word</th>
              <th style={{ padding: '8px 12px', textAlign: 'left', color: 'var(--text2)', fontWeight: 500 }}>Issue</th>
              <th style={{ padding: '8px 12px', textAlign: 'left', color: 'var(--text2)', fontWeight: 500 }}>Suggestion</th>
            </tr>
          </thead>
          <tbody>
            {items.map((w, i) => (
              <tr key={i} style={{ borderTop: '1px solid var(--border)' }}>
                <td style={{ padding: '8px 12px', fontFamily: 'monospace', color: 'var(--text2)', fontSize: '0.8rem' }}>{w.timestamp || ''}</td>
                <td style={{ padding: '8px 12px', fontWeight: 600, color: 'var(--text)' }}>{w.word}</td>
                <td style={{ padding: '8px 12px' }}>
                  <span className="badge badge-red">{w.issue}</span>
                </td>
                <td style={{ padding: '8px 12px', color: 'var(--text2)', fontSize: '0.8rem' }}>{w.suggestion}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </motion.div>
  );
}
