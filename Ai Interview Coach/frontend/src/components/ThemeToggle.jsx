import { useTheme } from '../context/ThemeContext';

export default function ThemeToggle() {
  const { theme, toggleTheme } = useTheme();

  return (
    <button
      onClick={toggleTheme}
      aria-label="Toggle theme"
      style={{
        background: 'none', border: 'none', cursor: 'pointer',
        padding: 6, borderRadius: 8, display: 'flex',
        alignItems: 'center', justifyContent: 'center',
        color: 'var(--text2)', fontSize: '1.15rem', lineHeight: 1,
        transition: 'var(--transition)',
      }}
      onMouseEnter={e => e.currentTarget.style.color = 'var(--text)'}
      onMouseLeave={e => e.currentTarget.style.color = 'var(--text2)'}
    >
      {theme === 'dark' ? '☀️' : '🌙'}
    </button>
  );
}
