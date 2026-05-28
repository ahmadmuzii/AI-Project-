import { useRef, useEffect, useCallback } from 'react';
import { useTheme } from '../context/ThemeContext';

export default function CursorRipple() {
  const { isDark } = useTheme();
  const containerRef = useRef(null);
  const idRef = useRef(0);

  const addRipple = useCallback((x, y) => {
    if (isDark) return;
    const id = ++idRef.current;
    const el = document.createElement('div');
    el.className = 'cursor-ripple';
    el.style.left = x + 'px';
    el.style.top = y + 'px';
    el.dataset.rippleId = id;
    containerRef.current?.appendChild(el);
    el.addEventListener('animationend', () => el.remove(), { once: true });
  }, [isDark]);

  useEffect(() => {
    let last = 0;
    function onMove(e) {
      const now = Date.now();
      if (now - last < 100) return;
      last = now;
      addRipple(e.clientX, e.clientY);
    }
    window.addEventListener('mousemove', onMove);
    return () => window.removeEventListener('mousemove', onMove);
  }, [addRipple]);

  if (isDark) return null;

  return <div ref={containerRef} style={{ position: 'fixed', inset: 0, pointerEvents: 'none', zIndex: 9998 }} />;
}
