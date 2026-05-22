import { useState, useRef, useEffect } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { useAuth } from '../context/AuthContext';
import ThemeToggle from './ThemeToggle';
import './Navbar.css';

export default function Navbar() {
  const { user, profile, logout } = useAuth();
  const location = useLocation();
  const navigate = useNavigate();
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef(null);

  useEffect(() => {
    function handleClick(e) {
      if (menuRef.current && !menuRef.current.contains(e.target)) {
        setMenuOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, []);

  const handleLogout = () => {
    setMenuOpen(false);
    logout();
    navigate('/');
  };

  const navLinks = [
    { to: '/dashboard', label: 'Dashboard' },
    { to: '/interview/new', label: 'Start Interview' },
    { to: '/practice',  label: 'Review Center' },
    { to: '/tools',     label: 'AI Tools' },
  ];

  const avatarUrl = profile?.avatar_url || null;

  return (
    <nav className="navbar">
      <div className="nav-inner">
        <Link to="/" className="nav-logo">
          <span className="nav-logo-icon" style={{ filter: 'drop-shadow(0 0 8px rgba(37,99,235,0.4))' }}>🎙️</span>
          <span style={{ fontFamily: "'Outfit', sans-serif", fontSize: '1.05rem', fontWeight: 800, background: 'linear-gradient(90deg, #fff, var(--text2))', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>Interview Coach</span>
        </Link>

        {user && (
          <div className="nav-links" style={{ position: 'relative' }}>
            {navLinks.map(l => {
              const isActive = location.pathname === l.to || (l.to === '/practice' && location.pathname.startsWith('/interview/') && !location.pathname.endsWith('/new'));
              return (
                <Link
                  key={l.to}
                  to={l.to}
                  className={`nav-link ${isActive ? 'active' : ''}`}
                  style={{ position: 'relative', overflow: 'visible', zIndex: 1 }}
                >
                  <span style={{ position: 'relative', zIndex: 2 }}>{l.label}</span>
                  {isActive && (
                    <motion.div
                      layoutId="activeNavTab"
                      transition={{ type: 'spring', stiffness: 380, damping: 30 }}
                      style={{
                        position: 'absolute',
                        inset: 0,
                        background: 'rgba(37, 99, 235, 0.08)',
                        border: '1px solid rgba(37, 99, 235, 0.15)',
                        borderRadius: 'var(--radius-sm)',
                        zIndex: 1,
                      }}
                    />
                  )}
                </Link>
              );
            })}
          </div>
        )}

        <div className="nav-right">
          <ThemeToggle />
          {user ? (
            <div className="nav-user" ref={menuRef} style={{ position: 'relative' }}>
              <div className="nav-avatar" onClick={() => setMenuOpen(!menuOpen)}
                style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 8 }}>
                {avatarUrl ? (
                  <img src={avatarUrl} alt="avatar"
                    style={{ width: 32, height: 32, borderRadius: '50%', objectFit: 'cover' }} />
                ) : (
                  <div style={{
                    width: 32, height: 32, borderRadius: '50%',
                    background: 'var(--blue)', color: '#fff',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontSize: '0.85rem', fontWeight: 600,
                  }}>
                    {user.name?.[0]?.toUpperCase() || '?'}
                  </div>
                )}
                <span className="hide-mobile" style={{ fontSize: '0.88rem', color: 'var(--text)', fontWeight: 500 }}>
                  {profile?.display_name || user.name}
                </span>
              </div>

              <AnimatePresence>
                {menuOpen && (
                  <motion.div
                    initial={{ opacity: 0, y: 10, scale: 0.95 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, y: 10, scale: 0.95 }}
                    transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
                    style={{
                      position: 'absolute', top: '100%', right: 0, marginTop: 8,
                      minWidth: 200, background: 'var(--bg2)', borderRadius: 12,
                      border: '1px solid var(--border2)', boxShadow: 'var(--shadow-lg)',
                      overflow: 'hidden', zIndex: 100,
                    }}
                  >
                    <Link to="/profile" onClick={() => setMenuOpen(false)}
                      style={{
                        display: 'block', padding: '12px 16px', fontSize: '0.88rem',
                        color: 'var(--text)', textDecoration: 'none',
                        borderBottom: '1px solid var(--border)',
                        transition: 'background 0.15s',
                      }}
                      onMouseEnter={(e) => e.target.style.background = 'var(--bg3)'}
                      onMouseLeave={(e) => e.target.style.background = 'transparent'}
                    >
                      Profile
                    </Link>
                    <Link to="/settings" onClick={() => setMenuOpen(false)}
                      style={{
                        display: 'block', padding: '12px 16px', fontSize: '0.88rem',
                        color: 'var(--text)', textDecoration: 'none',
                        borderBottom: '1px solid var(--border)',
                        transition: 'background 0.15s',
                      }}
                      onMouseEnter={(e) => e.target.style.background = 'var(--bg3)'}
                      onMouseLeave={(e) => e.target.style.background = 'transparent'}
                    >
                      Settings
                    </Link>
                    <button onClick={handleLogout}
                      style={{
                        display: 'block', width: '100%', padding: '12px 16px', fontSize: '0.88rem',
                        color: 'var(--red)', background: 'none', border: 'none', cursor: 'pointer',
                        textAlign: 'left', transition: 'background 0.15s',
                      }}
                      onMouseEnter={(e) => e.target.style.background = 'var(--bg3)'}
                      onMouseLeave={(e) => e.target.style.background = 'transparent'}
                    >
                      Sign Out
                    </button>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          ) : (
            <Link to="/login" className="btn btn-primary"
              style={{ padding: '8px 20px', fontSize: '0.88rem' }}>
              Get Started
            </Link>
          )}
        </div>
      </div>
    </nav>
  );
}
