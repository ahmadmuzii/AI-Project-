import { useState, useEffect } from 'react';
import ParticleField from './ParticleField';
import CursorRipple from './CursorRipple';
import { useTheme } from '../context/ThemeContext';

const blurOrbs = [
  { x: 10, y: 20, size: 500, color: 'rgba(59,130,246,0.15)', duration: 28, delay: 0 },
  { x: 70, y: 50, size: 600, color: 'rgba(139,92,246,0.12)', duration: 32, delay: -5 },
  { x: 40, y: 80, size: 450, color: 'rgba(6,182,212,0.1)', duration: 34, delay: -10 },
  { x: 85, y: 15, size: 400, color: 'rgba(99,102,241,0.1)', duration: 30, delay: -8 },
  { x: 20, y: 60, size: 550, color: 'rgba(59,130,246,0.1)', duration: 26, delay: -3 },
];

export default function GlobalEffects() {
  const [mousePos, setMousePos] = useState({ x: -200, y: -200 });
  const { isDark } = useTheme();

  useEffect(() => {
    let raf;
    let px = -200, py = -200;
    function onMove(e) {
      px = e.clientX;
      py = e.clientY;
    }
    window.addEventListener('mousemove', onMove);

    function tick() {
      setMousePos({ x: px, y: py });
      raf = requestAnimationFrame(tick);
    }
    raf = requestAnimationFrame(tick);

    return () => {
      window.removeEventListener('mousemove', onMove);
      cancelAnimationFrame(raf);
    };
  }, []);

  return (
    <>
      <div className="mesh-bg" />
      <ParticleField />

      {blurOrbs.map((orb, i) => {
        const c = isDark ? orb.color : orb.color.replace('0.15', '0.06').replace('0.12', '0.05').replace('0.1', '0.04');
        return (
          <div
            key={i}
            className="blur-orb"
            style={{
              left: orb.x + '%',
              top: orb.y + '%',
              width: orb.size + 'px',
              height: orb.size + 'px',
              background: c,
              animation: `orbFloat${i} ${orb.duration}s ease-in-out ${orb.delay}s infinite`,
            }}
          />
        );
      })}
      <style>{blurOrbs.map((_, i) => `
@keyframes orbFloat${i} {
  0%, 100% { transform: translate(0, 0) scale(1); }
  25% { transform: translate(${30 + i * 10}px, ${-20 - i * 5}px) scale(${1.04 + i * 0.01}); }
  50% { transform: translate(${-20 - i * 8}px, ${30 + i * 8}px) scale(${0.96 - i * 0.01}); }
  75% { transform: translate(${20 + i * 6}px, ${15 + i * 10}px) scale(${1.02 + i * 0.005}); }
}`).join('\n')}</style>

      <CursorRipple />
      <div className="cursor-glow" style={{ left: mousePos.x, top: mousePos.y }} />
    </>
  );
}
