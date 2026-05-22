import { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route, Navigate, useLocation } from 'react-router-dom';
import { AuthProvider, useAuth } from './context/AuthContext';
import OnboardingModal from './components/OnboardingModal';
import Landing from './pages/Landing';
import Login from './pages/Login';
import Dashboard from './pages/Dashboard';
import Practice from './pages/Practice';
import AITools from './pages/AITools';
import Profile from './pages/Profile';
import HistoryPage from './pages/HistoryPage';
import SettingsPage from './pages/SettingsPage';
import GuidedInterviewPage from './pages/GuidedInterview';
import './index.css';

function ProtectedRoute({ children }) {
  const { user, loading } = useAuth();
  if (loading) return <div style={{ padding: 120, textAlign: 'center', color: 'var(--text2)' }}>Loading...</div>;
  return user ? children : <Navigate to="/login" replace />;
}

function OnboardingGate({ children }) {
  const { profile, profileCompleted, loading } = useAuth();
  const [show, setShow] = useState(false);
  const location = useLocation();

  useEffect(() => {
    if (!loading && profile !== null && !profileCompleted && location.pathname !== '/profile' && location.pathname !== '/settings') {
      setShow(true);
    } else {
      setShow(false);
    }
  }, [loading, profile, profileCompleted, location.pathname]);

  return (
    <>
      {children}
      {show && <OnboardingModal onComplete={() => setShow(false)} />}
    </>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <OnboardingGate>
          <Routes>
            <Route path="/"         element={<Landing />} />
            <Route path="/login"    element={<Login />} />
            <Route path="/dashboard" element={<ProtectedRoute><Dashboard /></ProtectedRoute>} />
            <Route path="/practice"  element={<ProtectedRoute><Practice /></ProtectedRoute>} />
            <Route path="/tools"     element={<ProtectedRoute><AITools /></ProtectedRoute>} />
            <Route path="/profile"   element={<ProtectedRoute><Profile /></ProtectedRoute>} />
            <Route path="/history"   element={<Navigate to="/practice" replace />} />
            <Route path="/settings"  element={<ProtectedRoute><SettingsPage /></ProtectedRoute>} />
            <Route path="/interview/new" element={<ProtectedRoute><GuidedInterviewPage /></ProtectedRoute>} />
            <Route path="/interview/:id" element={<ProtectedRoute><GuidedInterviewPage /></ProtectedRoute>} />
            <Route path="*"          element={<Navigate to="/" replace />} />
          </Routes>
        </OnboardingGate>
      </BrowserRouter>
    </AuthProvider>
  );
}
