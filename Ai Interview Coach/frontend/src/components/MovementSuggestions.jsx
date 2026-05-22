import { useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const TIPS = {
  eye_contact: {
    good: { text: 'Great eye contact', color: 'var(--green)', icon: '✓' },
    warn: { text: 'Look toward the camera', color: 'var(--yellow)', icon: '⚠' },
    bad: { text: 'Keep your eyes on the camera', color: 'var(--red)', icon: '✗' },
  },
  movement: {
    good: { text: 'Steady posture', color: 'var(--green)', icon: '✓' },
    warn: { text: 'Try to minimize fidgeting', color: 'var(--yellow)', icon: '⚠' },
    bad: { text: 'Too much movement, stay calm', color: 'var(--red)', icon: '✗' },
  },
  stress: {
    good: { text: 'You appear confident', color: 'var(--green)', icon: '✓' },
    warn: { text: 'Take a deep breath', color: 'var(--yellow)', icon: '⚠' },
    bad: { text: 'Relax — you got this', color: 'var(--red)', icon: '✗' },
  },
  hands: {
    good: { text: 'Hands are steady', color: 'var(--green)', icon: '✓' },
    warn: { text: 'Keep hands visible', color: 'var(--yellow)', icon: '⚠' },
    bad: { text: 'Minimize hand gestures', color: 'var(--red)', icon: '✗' },
  },
};

function level(value, ascending) {
  if (ascending) {
    if (value > 0.7) return 'good';
    if (value > 0.35) return 'warn';
    return 'bad';
  }
  if (value < 0.3) return 'good';
  if (value < 0.6) return 'warn';
  return 'bad';
}

export default function MovementSuggestions({ metrics }) {
  const suggestions = useMemo(() => {
    if (!metrics) return [];
    const items = [
      { key: 'eye_contact', tip: TIPS.eye_contact[level(metrics.eye_contact_score, true)] },
      { key: 'movement', tip: TIPS.movement[level(metrics.movement_score, true)] },
      { key: 'stress', tip: TIPS.stress[level(metrics.stress_score, false)] },
      { key: 'hands', tip: TIPS.hands[level(1 - Math.min(metrics.hand_count / 4, 1), false)] },
    ];
    return items;
  }, [metrics]);

  if (!metrics) return null;

  return (
    <div className="card" style={{ padding: 16 }}>
      <div style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text2)', marginBottom: 12 }}>
        Live Coaching
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        <AnimatePresence mode="popLayout">
          {suggestions.map(({ key, tip }) => (
            <motion.div
              key={key}
              layout
              initial={{ opacity: 0, x: -8 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 8 }}
              transition={{ duration: 0.25 }}
              style={{
                display: 'flex', alignItems: 'center', gap: 8,
                padding: '6px 10px', borderRadius: 8,
                background: tip.color + '12',
                fontSize: '0.82rem', color: tip.color,
              }}
            >
              <span style={{ fontWeight: 700, flexShrink: 0, fontSize: '0.9rem' }}>{tip.icon}</span>
              <span>{tip.text}</span>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    </div>
  );
}
