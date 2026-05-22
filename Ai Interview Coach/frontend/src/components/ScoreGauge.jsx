import { useEffect, useState, useRef } from 'react';
import { motion, useInView } from 'framer-motion';

const SIZE = 140;
const STROKE = 10;
const RADIUS = (SIZE - STROKE) / 2;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS;

export default function ScoreGauge({ score = 0, label = '', color = '#1A73E8', delay = 0 }) {
  const [display, setDisplay] = useState(0);
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: '-40px' });
  const animated = useRef(false);

  useEffect(() => {
    if (inView && !animated.current) {
      animated.current = true;
      const startTime = performance.now();
      const duration = 1200;
      const target = Math.round(score * 100);

      function animate(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        setDisplay(Math.round(eased * target));
        if (progress < 1) requestAnimationFrame(animate);
      }
      requestAnimationFrame(animate);
    }
  }, [inView, score]);

  const offset = CIRCUMFERENCE * (1 - score);

  return (
    <motion.div
      ref={ref}
      className="flex col center"
      initial={{ opacity: 0, y: 20 }}
      animate={inView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.5, delay, ease: [0.16, 1, 0.3, 1] }}
    >
      <svg width={SIZE} height={SIZE} viewBox={`0 0 ${SIZE} ${SIZE}`}>
        <circle
          cx={SIZE / 2} cy={SIZE / 2} r={RADIUS}
          fill="none" stroke="var(--bg3)" strokeWidth={STROKE} strokeLinecap="round"
        />
        <motion.circle
          cx={SIZE / 2} cy={SIZE / 2} r={RADIUS}
          fill="none" stroke={color} strokeWidth={STROKE} strokeLinecap="round"
          strokeDasharray={CIRCUMFERENCE}
          initial={{ strokeDashoffset: CIRCUMFERENCE }}
          animate={inView ? { strokeDashoffset: offset } : {}}
          transition={{ duration: 1.2, delay: delay + 0.1, ease: [0.16, 1, 0.3, 1] }}
          transform={`rotate(-90 ${SIZE / 2} ${SIZE / 2})`}
        />
        <text
          x={SIZE / 2} y={SIZE / 2 - 4} textAnchor="middle"
          dominantBaseline="central"
          fontSize="2rem" fontWeight="700" fill="var(--text)"
        >
          {display}%
        </text>
        <text
          x={SIZE / 2} y={SIZE / 2 + 24} textAnchor="middle"
          dominantBaseline="central"
          fontSize="0.7rem" fill="var(--text2)"
        >
          {label}
        </text>
      </svg>
    </motion.div>
  );
}
