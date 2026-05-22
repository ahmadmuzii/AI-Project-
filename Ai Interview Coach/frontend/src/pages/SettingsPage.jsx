import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import Navbar from '../components/Navbar';
import { useAuth } from '../context/AuthContext';
import {
  getProfile, updateProfile, changePassword, deleteAccount,
  listElevenLabsVoices,
} from '../api/client';

export default function SettingsPage() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  const [pwOld, setPwOld] = useState('');
  const [pwNew, setPwNew] = useState('');
  const [pwMsg, setPwMsg] = useState(null);

  const [notifyEmail, setNotifyEmail] = useState(true);
  const [notifyReminders, setNotifyReminders] = useState(true);
  const [notifyMsg, setNotifyMsg] = useState(null);

  const [deleteConfirm, setDeleteConfirm] = useState('');
  const [deletePw, setDeletePw] = useState('');

  const [voices, setVoices] = useState([]);
  const [useElevenLabs, setUseElevenLabs] = useState(false);
  const [elevenlabsVoiceId, setElevenlabsVoiceId] = useState('');
  const [elevenlabsMsg, setElevenlabsMsg] = useState(null);

  const [savingNotify, setSavingNotify] = useState(false);
  const [savingEleven, setSavingEleven] = useState(false);

  useEffect(() => {
    getProfile()
      .then((p) => {
        setNotifyEmail(p.notify_email_digests ?? true);
        setNotifyReminders(p.notify_session_reminders ?? true);
        setUseElevenLabs(p.use_elevenlabs ?? false);
        setElevenlabsVoiceId(p.elevenlabs_voice_id || '');
      })
      .catch(() => {});
    listElevenLabsVoices()
      .then((v) => setVoices(v.voices || v || []))
      .catch(() => {});
  }, []);

  async function handleChangePassword(e) {
    e.preventDefault();
    setPwMsg(null);
    try {
      await changePassword(pwOld, pwNew);
      setPwMsg({ type: 'success', text: 'Password changed' });
      setPwOld('');
      setPwNew('');
    } catch (e) {
      setPwMsg({ type: 'error', text: e.message });
    }
  }

  async function handleSaveNotifications() {
    setSavingNotify(true);
    setNotifyMsg(null);
    try {
      await updateProfile({
        notify_email_digests: notifyEmail,
        notify_session_reminders: notifyReminders,
      });
      setNotifyMsg({ type: 'success', text: 'Notification preferences saved' });
    } catch (e) {
      setNotifyMsg({ type: 'error', text: e.message });
    } finally {
      setSavingNotify(false);
    }
  }

  async function handleSaveElevenLabs() {
    setSavingEleven(true);
    setElevenlabsMsg(null);
    try {
      await updateProfile({
        use_elevenlabs: useElevenLabs,
        elevenlabs_voice_id: elevenlabsVoiceId || '',
      });
      setElevenlabsMsg({ type: 'success', text: 'Voice settings saved' });
    } catch (e) {
      setElevenlabsMsg({ type: 'error', text: e.message });
    } finally {
      setSavingEleven(false);
    }
  }

  async function handleDeleteAccount() {
    if (deleteConfirm !== 'DELETE') return;
    try {
      await deleteAccount(deletePw);
      logout();
      navigate('/');
    } catch (e) {
      setNotifyMsg({ type: 'error', text: e.message });
    }
  }

  const sectionTitle = { fontSize: '1.1rem', fontWeight: 700, color: 'var(--text)', marginBottom: 16, marginTop: 8 };
  const input = {
    width: '100%', padding: '11px 14px', borderRadius: 10,
    border: '1px solid var(--border)', background: 'var(--bg)',
    color: 'var(--text)', fontSize: '0.9rem', outline: 'none',
    boxSizing: 'border-box',
  };
  const label = { display: 'block', fontSize: '0.82rem', fontWeight: 600, color: 'var(--text)', marginBottom: 6 };

  return (
    <div>
      <Navbar />
      <div style={{ padding: '80px 24px 40px', maxWidth: 720, margin: '0 auto' }}>
        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}>
          <h1 style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--text)', marginBottom: 4 }}>Settings</h1>
          <p style={{ color: 'var(--text2)', fontSize: '0.9rem', marginBottom: 24 }}>Account & app preferences</p>

          {/* Change Password */}
          <div className="card" style={{ marginBottom: 16 }}>
            <div style={sectionTitle}>Change Password</div>
            {pwMsg && (
              <div style={{ padding: '8px 12px', marginBottom: 12, borderRadius: 8,
                background: pwMsg.type === 'success' ? 'rgba(52,168,83,0.1)' : 'rgba(234,67,53,0.1)',
                color: pwMsg.type === 'success' ? 'var(--green)' : 'var(--red)', fontSize: '0.85rem' }}>
                {pwMsg.text}
              </div>
            )}
            <form onSubmit={handleChangePassword} style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              <input style={input} type="password" placeholder="Current password" value={pwOld}
                onChange={(e) => setPwOld(e.target.value)} required />
              <input style={input} type="password" placeholder="New password (min 4 chars)" value={pwNew}
                onChange={(e) => setPwNew(e.target.value)} required minLength={4} />
              <button type="submit" className="btn btn-primary"
                style={{ alignSelf: 'flex-start', fontSize: '0.85rem', padding: '10px 24px' }}>
                Change Password
              </button>
            </form>
          </div>

          {/* Notifications */}
          <div className="card" style={{ marginBottom: 16 }}>
            <div style={sectionTitle}>Notifications</div>
            {notifyMsg && (
              <div style={{ padding: '8px 12px', marginBottom: 12, borderRadius: 8,
                background: notifyMsg.type === 'success' ? 'rgba(52,168,83,0.1)' : 'rgba(234,67,53,0.1)',
                color: notifyMsg.type === 'success' ? 'var(--green)' : 'var(--red)', fontSize: '0.85rem' }}>
                {notifyMsg.text}
              </div>
            )}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12, marginBottom: 16 }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer', fontSize: '0.9rem', color: 'var(--text)' }}>
                <input type="checkbox" checked={notifyEmail}
                  onChange={(e) => setNotifyEmail(e.target.checked)} />
                Email Digests
              </label>
              <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer', fontSize: '0.9rem', color: 'var(--text)' }}>
                <input type="checkbox" checked={notifyReminders}
                  onChange={(e) => setNotifyReminders(e.target.checked)} />
                Session Reminders
              </label>
            </div>
            <button className="btn btn-primary" onClick={handleSaveNotifications} disabled={savingNotify}
              style={{ fontSize: '0.85rem', padding: '10px 24px' }}>
              {savingNotify ? 'Saving...' : 'Save'}
            </button>
          </div>

          {/* ElevenLabs */}
          <div className="card" style={{ marginBottom: 16 }}>
            <div style={sectionTitle}>AI Voice (ElevenLabs)</div>
            {elevenlabsMsg && (
              <div style={{ padding: '8px 12px', marginBottom: 12, borderRadius: 8,
                background: elevenlabsMsg.type === 'success' ? 'rgba(52,168,83,0.1)' : 'rgba(234,67,53,0.1)',
                color: elevenlabsMsg.type === 'success' ? 'var(--green)' : 'var(--red)', fontSize: '0.85rem' }}>
                {elevenlabsMsg.text}
              </div>
            )}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer', fontSize: '0.9rem', color: 'var(--text)' }}>
                <input type="checkbox" checked={useElevenLabs}
                  onChange={(e) => setUseElevenLabs(e.target.checked)} />
                Use ElevenLabs for AI interviewer voice
              </label>
              {useElevenLabs && (
                <div>
                  <label style={label}>Voice</label>
                  <select
                    style={{ ...input, appearance: 'auto' }}
                    value={elevenlabsVoiceId}
                    onChange={(e) => setElevenlabsVoiceId(e.target.value)}
                  >
                    <option value="">Default</option>
                    {voices.map((v) => (
                      <option key={v.voice_id} value={v.voice_id}>{v.name}</option>
                    ))}
                    <option value="T7eLpgAAhoXHlrNajG8v">Gracie</option>
                  </select>
                </div>
              )}
            </div>
            <button className="btn btn-primary" onClick={handleSaveElevenLabs} disabled={savingEleven}
              style={{ marginTop: 12, fontSize: '0.85rem', padding: '10px 24px' }}>
              {savingEleven ? 'Saving...' : 'Save'}
            </button>
          </div>

          {/* Delete Account */}
          <div className="card" style={{ marginBottom: 16, borderColor: 'var(--red)' }}>
            <div style={{ ...sectionTitle, color: 'var(--red)' }}>Delete Account</div>
            <p style={{ fontSize: '0.85rem', color: 'var(--text2)', marginBottom: 12 }}>
              This permanently deletes your account and all data. This cannot be undone.
            </p>
            <input style={{ ...input, marginBottom: 8 }} type="password" placeholder="Enter your password"
              value={deletePw} onChange={(e) => setDeletePw(e.target.value)} />
            <input style={input} placeholder='Type "DELETE" to confirm'
              value={deleteConfirm} onChange={(e) => setDeleteConfirm(e.target.value)} />
            <button className="btn btn-primary" onClick={handleDeleteAccount}
              disabled={deleteConfirm !== 'DELETE' || !deletePw}
              style={{ marginTop: 12, background: 'var(--red)', fontSize: '0.85rem', padding: '10px 24px' }}>
              Delete My Account
            </button>
          </div>
        </motion.div>
      </div>
    </div>
  );
}
