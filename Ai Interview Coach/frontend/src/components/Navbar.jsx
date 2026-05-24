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
    { to: '/interview/new', label: 'Interview' },
    { to: '/practice',  label: 'Review' },
    { to: '/tools',     label: 'Tools' },
  ];

  const avatarUrl = profile?.avatar_url || null;

  return (
    <nav className="navbar">
      <div className="nav-inner">
        <Link to="/" className="nav-logo">
          <span className="nav-logo-icon">🎙️</span>
          <span>Coach</span>
        </Link>

        {user && (
          <div className="nav-links">
            {navLinks.map(l => {
              const isActive = location.pathname === l.to || (l.to === '/practice' && location.pathname.startsWith('/interview/') && !location.pathname.endsWith('/new'));
              return (
                <Link
                  key={l.to}
                  to={l.to}
                  className={`nav-link ${isActive ? 'active' : ''}`}
                >
                  <span style={{ position: 'relative', zIndex: 2 }}>{l.label}</span>
                  {isActive && (
                    <motion.div
                      layoutId="activeNavTab"
                      transition={{ type: 'spring', stiffness: 380, damping: 30 }}
                      style={{
                        position: 'absolute',
                        inset: 0,
                        background: 'rgba(0,0,0,0.06)',
                        borderRadius: 999,
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
            <div className="nav-user" ref={menuRef}>
              <div className="nav-avatar" onClick={() => setMenuOpen(!menuOpen)}>
                {avatarUrl ? (
                  <img src={avatarUrl} alt="avatar" className="nav-avatar-img" />
                ) : (
                  user.name?.[0]?.toUpperCase() || '?'
                )}
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
                      minWidth: 180, background: 'rgba(255,255,255,0.85)',
                      backdropFilter: 'blur(24px)',
                      borderRadius: 16, border: '1px solid rgba(255,255,255,0.4)',
                      boxShadow: '0 12px 40px rgba(0,0,0,0.1)',
                      overflow: 'hidden', zIndex: 100,
                    }}
                    className="dark:bg-black/80 dark:border-white/10"
                  >
                    <Link to="/profile" onClick={() => setMenuOpen(false)}
                      style={{
                        display: 'block', padding: '12px 16px', fontSize: '13px',
                        color: '#1d1d1f', textDecoration: 'none',
                        borderBottom: '1px solid rgba(0,0,0,0.04)',
                        transition: 'background 0.15s',
                      }}
                      onMouseEnter={(e) => e.target.style.background = 'rgba(0,0,0,0.04)'}
                      onMouseLeave={(e) => e.target.style.background = 'transparent'}
                    >
                      Profile
                    </Link>
                    <Link to="/settings" onClick={() => setMenuOpen(false)}
                      style={{
                        display: 'block', padding: '12px 16px', fontSize: '13px',
                        color: '#1d1d1f', textDecoration: 'none',
                        borderBottom: '1px solid rgba(0,0,0,0.04)',
                        transition: 'background 0.15s',
                      }}
                      onMouseEnter={(e) => e.target.style.background = 'rgba(0,0,0,0.04)'}
                      onMouseLeave={(e) => e.target.style.background = 'transparent'}
                    >
                      Settings
                    </Link>
                    <button onClick={handleLogout}
                      style={{
                        display: 'block', width: '100%', padding: '12px 16px', fontSize: '13px',
                        color: '#ff4b5c', background: 'none', border: 'none', cursor: 'pointer',
                        textAlign: 'left', transition: 'background 0.15s',
                      }}
                      onMouseEnter={(e) => e.target.style.background = 'rgba(0,0,0,0.04)'}
                      onMouseLeave={(e) => e.target.style.background = 'transparent'}
                    >
                      Sign Out
                    </button>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          ) : (
            <Link to="/login" className="nav-link"
              style={{ color: '#1d1d1f', fontWeight: 600 }}>
              Sign In
            </Link>
          )}
        </div>
      </div>
    </nav>
  );
}
