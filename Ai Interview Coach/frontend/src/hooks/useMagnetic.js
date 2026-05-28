import { useRef, useState, useCallback } from 'react';

export default function useMagnetic({ stiffness = 150, damping = 15 } = {}) {
  const ref = useRef(null);
  const [pos, setPos] = useState({ x: 0, y: 0 });

  const handleMouseMove = useCallback((e) => {
    const el = ref.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const cx = rect.left + rect.width / 2;
    const cy = rect.top + rect.height / 2;
    const dx = e.clientX - cx;
    const dy = e.clientY - cy;
    const dist = Math.sqrt(dx * dx + dy * dy);
    const maxDist = 200;
    if (dist > maxDist) {
      setPos({ x: 0, y: 0 });
      return;
    }
    const strength = 0.25;
    setPos({ x: dx * strength, y: dy * strength });
  }, []);

  const handleMouseLeave = useCallback(() => {
    setPos({ x: 0, y: 0 });
  }, []);

  return { ref, pos, handleMouseMove, handleMouseLeave };
}
