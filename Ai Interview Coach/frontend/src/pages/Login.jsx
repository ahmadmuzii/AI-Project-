import { useState } from 'react';
import { Link, Navigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { useAuth } from '../context/AuthContext';
import { login as apiLogin, register as apiRegister } from '../api/client';
import './Login.css';

function Toast({ message, type, onClose }) {
  if (!message) return null;
  const bg = type === 'error' ? '#EA4335' : type === 'success' ? '#34A853' : '#1A73E8';
  return (
    <motion.div
      className="login-toast"
      style={{ background: bg }}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 20 }}
      transition={{ duration: 0.3 }}
    >
      {message}
    </motion.div>
  );
}

export default function Login() {
  const { login: setAuth, user } = useAuth();
  const [tab, setTab] = useState('login');
  const [loading, setLoading] = useState(false);
  const [toast, setToast] = useState({ message: '', type: '' });

  const [loginEmail, setLoginEmail] = useState('');
  const [loginPassword, setLoginPassword] = useState('');
  const [regName, setRegName] = useState('');
  const [regEmail, setRegEmail] = useState('');
  const [regPassword, setRegPassword] = useState('');

  if (user) return <Navigate to="/dashboard" replace />;

  function showToast(message, type = '') {
    setToast({ message, type });
    setTimeout(() => setToast({ message: '', type: '' }), 3000);
  }

  async function handleLogin(e) {
    e.preventDefault();
    if (!loginEmail || !loginPassword) {
      showToast('Please fill in all fields.', 'error');
      return;
    }
    setLoading(true);
    try {
      const data = await apiLogin(loginEmail, loginPassword);
      setAuth(data.access_token, { user_id: data.user_id, name: data.name, email: data.email });

      showToast(`Welcome back, ${data.name}!`, 'success');
    } catch (err) {
      showToast(err.message, 'error');
    } finally {
      setLoading(false);
    }
  }

  async function handleRegister(e) {
    e.preventDefault();
    if (!regName || !regEmail || !regPassword) {
      showToast('Please fill in all fields.', 'error');
      return;
    }
    setLoading(true);
    try {
      const data = await apiRegister(regName, regEmail, regPassword);
      setAuth(data.access_token, { user_id: data.user_id, name: data.name, email: data.email });

      showToast(`Welcome, ${data.name}!`, 'success');
    } catch (err) {
      showToast(err.message, 'error');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="login-page">
      <div className="login-bg">
        <div className="login-bg-gradient" />
      </div>

      <Link to="/" className="login-logo">AI Interview Coach</Link>

      <motion.div
        className="login-card"
        initial={{ opacity: 0, y: 40, scale: 0.96 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
      >
        <div className="login-tabs">
          <button
            className={`login-tab ${tab === 'login' ? 'active' : ''}`}
            onClick={() => setTab('login')}
          >
            Sign In
          </button>
          <button
            className={`login-tab ${tab === 'register' ? 'active' : ''}`}
            onClick={() => setTab('register')}
          >
            Get Started
          </button>
        </div>

        <AnimatePresence mode="wait">
          {tab === 'login' && (
            <motion.form
              key="login"
              className="login-form"
              onSubmit={handleLogin}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 10 }}
              transition={{ duration: 0.25, ease: 'easeInOut' }}
            >
              <div className="login-input-group">
                <input
                  type="email"
                  className="login-input"
                  placeholder=" "
                  value={loginEmail}
                  onChange={(e) => setLoginEmail(e.target.value)}
                  required
                />
                <label className="login-label">Email address</label>
              </div>
              <div className="login-input-group">
                <input
                  type="password"
                  className="login-input"
                  placeholder=" "
                  value={loginPassword}
                  onChange={(e) => setLoginPassword(e.target.value)}
                  required
                />
                <label className="login-label">Password</label>
              </div>
              <button type="submit" className="login-submit" disabled={loading}>
                {loading ? <span className="login-spinner" /> : 'Continue'}
              </button>
              <p className="login-hint">
                Don't have an account?{' '}
                <a href="#" onClick={(e) => { e.preventDefault(); setTab('register'); }}>Get started</a>
              </p>
            </motion.form>
          )}

          {tab === 'register' && (
            <motion.form
              key="register"
              className="login-form"
              onSubmit={handleRegister}
              initial={{ opacity: 0, x: 10 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -10 }}
              transition={{ duration: 0.25, ease: 'easeInOut' }}
            >
              <div className="login-input-group">
                <input
                  type="text"
                  className="login-input"
                  placeholder=" "
                  value={regName}
                  onChange={(e) => setRegName(e.target.value)}
                  required
                />
                <label className="login-label">Full name</label>
              </div>
              <div className="login-input-group">
                <input
                  type="email"
                  className="login-input"
                  placeholder=" "
                  value={regEmail}
                  onChange={(e) => setRegEmail(e.target.value)}
                  required
                />
                <label className="login-label">Email address</label>
              </div>
              <div className="login-input-group">
                <input
                  type="password"
                  className="login-input"
                  placeholder=" "
                  value={regPassword}
                  onChange={(e) => setRegPassword(e.target.value)}
                  required
                />
                <label className="login-label">Password</label>
              </div>
              <button type="submit" className="login-submit" disabled={loading}>
                {loading ? <span className="login-spinner" /> : 'Create Account'}
              </button>
              <p className="login-hint">
                Already have an account?{' '}
                <a href="#" onClick={(e) => { e.preventDefault(); setTab('login'); }}>Sign in</a>
              </p>
            </motion.form>
          )}
        </AnimatePresence>
      </motion.div>

      <AnimatePresence>
        {toast.message && <Toast message={toast.message} type={toast.type} />}
      </AnimatePresence>
    </div>
  );
}
