import { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { getMe, getProfile } from '../api/client';

const AuthContext = createContext(null);

const AUTH_KEY = 'aic_auth';

function loadAuth() {
  try {
    const raw = localStorage.getItem(AUTH_KEY);
    if (!raw) return null;
    const data = JSON.parse(raw);
    if (data && data.access_token && data.user) return data;
  } catch {}
  return null;
}

function saveAuth(data) {
  localStorage.setItem(AUTH_KEY, JSON.stringify(data));
}

function clearAuth() {
  localStorage.removeItem(AUTH_KEY);
}

export function AuthProvider({ children }) {
  const [auth, setAuth] = useState(loadAuth);
  const [profile, setProfile] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const stored = loadAuth();
    if (stored && stored.access_token) {
      getMe()
        .then((user) => {
          const updated = { ...stored, user: { user_id: user.user_id, name: user.name, email: user.email } };
          saveAuth(updated);
          setAuth(updated);
          return getProfile();
        })
        .then((p) => setProfile(p))
        .catch(() => {
          clearAuth();
          setAuth(null);
          setProfile(null);
        })
        .finally(() => setLoading(false));
    } else {
      setLoading(false);
    }
  }, []);

  const login = useCallback((access_token, userData) => {
    const data = { access_token, user: { user_id: userData.user_id, name: userData.name, email: userData.email } };
    saveAuth(data);
    setAuth(data);
    getProfile()
      .then((p) => setProfile(p))
      .catch(() => setProfile(null));
  }, []);

  const refreshProfile = useCallback(() => {
    return getProfile()
      .then((p) => {
        setProfile(p);
        return p;
      })
      .catch(() => {
        setProfile(null);
        return null;
      });
  }, []);

  const logout = useCallback(() => {
    clearAuth();
    setAuth(null);
    setProfile(null);
  }, []);

  const user = auth?.user || null;
  const accessToken = auth?.access_token || null;
  const profileCompleted = profile?.profile_completed ?? false;

  return (
    <AuthContext.Provider value={{ user, accessToken, profile, profileCompleted, loading, login, logout, refreshProfile }}>
      {children}
    </AuthContext.Provider>
  );
}

export const useAuth = () => {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within AuthProvider');
  return ctx;
};
