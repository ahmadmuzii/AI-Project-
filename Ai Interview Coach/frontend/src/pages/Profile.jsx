import { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import Navbar from '../components/Navbar';
import ResumeManager from '../components/ResumeManager';
import { useAuth } from '../context/AuthContext';
import {
  getProfile, updateProfile, uploadAvatar, uploadResume,
  changePassword, deleteAccount,
} from '../api/client';

const INDUSTRIES = [
  { value: 'software', label: 'Software / Technology' },
  { value: 'finance', label: 'Finance & Banking' },
  { value: 'consulting', label: 'Consulting' },
  { value: 'healthcare', label: 'Healthcare & Pharmaceuticals' },
  { value: 'education', label: 'Education & Academia' },
  { value: 'ecommerce', label: 'E-commerce & Retail' },
  { value: 'media', label: 'Media & Entertainment' },
  { value: 'telecommunications', label: 'Telecommunications' },
  { value: 'energy', label: 'Energy & Utilities' },
  { value: 'manufacturing', label: 'Manufacturing' },
  { value: 'aerospace', label: 'Aerospace & Defense' },
  { value: 'automotive', label: 'Automotive' },
  { value: 'biotechnology', label: 'Biotechnology' },
  { value: 'government', label: 'Government / Public Sector' },
  { value: 'nonprofit', label: 'Non-Profit' },
  { value: 'real_estate', label: 'Real Estate' },
  { value: 'hospitality', label: 'Hospitality & Tourism' },
  { value: 'legal', label: 'Legal Services' },
  { value: 'agriculture', label: 'Agriculture' },
  { value: 'logistics', label: 'Logistics & Supply Chain' },
];

const FOCUS_AREAS = [
  { value: 'behavioral', label: 'Behavioral / STAR Stories' },
  { value: 'technical', label: 'Technical Skills' },
  { value: 'system_design', label: 'System Design' },
  { value: 'case_studies', label: 'Case Studies' },
  { value: 'leadership', label: 'Leadership & Management' },
  { value: 'communication', label: 'Communication & Presentation' },
  { value: 'problem_solving', label: 'Problem Solving' },
  { value: 'coding', label: 'Coding / Algorithms' },
  { value: 'domain_knowledge', label: 'Domain Knowledge' },
  { value: 'conflict_resolution', label: 'Conflict Resolution' },
  { value: 'project_management', label: 'Project Management' },
  { value: 'data_analysis', label: 'Data Analysis & Analytics' },
  { value: 'situational', label: 'Situational Judgment' },
  { value: 'cultural_fit', label: 'Cultural Fit & Values' },
  { value: 'negotiation', label: 'Salary Negotiation' },
  { value: 'storytelling', label: 'Storytelling & Narrative' },
  { value: 'adaptability', label: 'Adaptability & Resilience' },
  { value: 'teamwork', label: 'Teamwork & Collaboration' },
  { value: 'time_management', label: 'Time Management & Prioritization' },
  { value: 'critical_thinking', label: 'Critical Thinking' },
];

const SENIORITY_LEVELS = [
  { value: 'intern', label: 'Intern' },
  { value: 'junior', label: 'Junior' },
  { value: 'mid', label: 'Mid-Level' },
  { value: 'senior', label: 'Senior' },
  { value: 'staff', label: 'Staff' },
  { value: 'manager', label: 'Manager' },
  { value: 'executive', label: 'Executive' },
];

const EDUCATION_LEVELS = [
  { value: 'high_school', label: 'High School' },
  { value: 'associate', label: 'Associate Degree' },
  { value: 'bachelor', label: "Bachelor's Degree" },
  { value: 'master', label: "Master's Degree" },
  { value: 'phd', label: 'PhD / Doctorate' },
  { value: 'other', label: 'Other' },
];

export default function Profile() {
  const { user, logout, refreshProfile } = useAuth();
  const navigate = useNavigate();
  const avatarRef = useRef(null);
  const interviewerAvatarRef = useRef(null);

  const [form, setForm] = useState({
    display_name: '', phone: '', target_role: '', target_industry: '',
    seniority_level: '', years_of_experience: '', current_company: '',
    education_level: '', linkedin_url: '', focus_areas: [],
    upcoming_interview_date: '', preferred_difficulty: 'intermediate',
    timezone: '', locale: 'en-US',
    notify_email_digests: true, notify_session_reminders: true,
    mic_default: '', camera_default: '',
  });
  const [avatarUrl, setAvatarUrl] = useState(null);
  const [interviewerAvatar, setInterviewerAvatar] = useState(() => {
    try { return localStorage.getItem('aic_interviewer_avatar') || null; } catch { return null; }
  });
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState(null);
  const [pwOld, setPwOld] = useState('');
  const [pwNew, setPwNew] = useState('');
  const [pwMsg, setPwMsg] = useState(null);
  const [deleteConfirm, setDeleteConfirm] = useState('');
  const [deletePw, setDeletePw] = useState('');

  useEffect(() => {
    getProfile()
      .then((p) => {
        setForm({
          display_name: p.display_name || '',
          phone: p.phone || '',
          target_role: p.target_role || '',
          target_industry: p.target_industry || '',
          seniority_level: p.seniority_level || '',
          years_of_experience: p.years_of_experience ?? '',
          current_company: p.current_company || '',
          education_level: p.education_level || '',
          linkedin_url: p.linkedin_url || '',
          focus_areas: p.focus_areas || [],
          upcoming_interview_date: p.upcoming_interview_date ? p.upcoming_interview_date.slice(0, 10) : '',
          preferred_difficulty: p.preferred_difficulty || 'intermediate',
          timezone: p.timezone || Intl.DateTimeFormat().resolvedOptions().timeZone || '',
          locale: p.locale || navigator.language || 'en-US',
          notify_email_digests: p.notify_email_digests ?? true,
          notify_session_reminders: p.notify_session_reminders ?? true,
          mic_default: p.mic_default || '',
          camera_default: p.camera_default || '',
        });
        setAvatarUrl(p.avatar_url || null);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  function setField(key, val) {
    setForm((prev) => ({ ...prev, [key]: val }));
  }

  function toggleFocus(val) {
    setForm((prev) => ({
      ...prev,
      focus_areas: prev.focus_areas.includes(val)
        ? prev.focus_areas.filter((f) => f !== val)
        : [...prev.focus_areas, val],
    }));
  }

  async function handleAvatarUpload(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const res = await uploadAvatar(file);
      setAvatarUrl(res.avatar_url);
      await refreshProfile();
      setMessage({ type: 'success', text: 'Avatar updated' });
    } catch {
      setMessage({ type: 'error', text: 'Avatar upload failed' });
    }
  }

  function handleInterviewerAvatarUpload(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = reader.result;
      try { localStorage.setItem('aic_interviewer_avatar', dataUrl); } catch {}
      setInterviewerAvatar(dataUrl);
      setMessage({ type: 'success', text: 'Interviewer avatar saved' });
    };
    reader.onerror = () => setMessage({ type: 'error', text: 'Failed to read file' });
    reader.readAsDataURL(file);
  }

  function handleRemoveInterviewerAvatar() {
    try { localStorage.removeItem('aic_interviewer_avatar'); } catch {}
    setInterviewerAvatar(null);
    setMessage({ type: 'success', text: 'Interviewer avatar removed' });
  }

  async function handleSave() {
    setSaving(true);
    setMessage(null);
    try {
      await updateProfile({
        ...form,
        years_of_experience: form.years_of_experience ? parseInt(form.years_of_experience) : undefined,
        upcoming_interview_date: form.upcoming_interview_date || undefined,
      });
      await refreshProfile();
      setMessage({ type: 'success', text: 'Profile saved' });
    } catch (e) {
      setMessage({ type: 'error', text: e.message });
    } finally {
      setSaving(false);
    }
  }

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

  async function handleDeleteAccount() {
    if (deleteConfirm !== 'DELETE') return;
    try {
      await deleteAccount(deletePw);
      logout();
      navigate('/');
    } catch (e) {
      setMessage({ type: 'error', text: e.message });
    }
  }

  if (loading) {
    return (
      <div>
        <Navbar />
        <div style={{ padding: '120px 24px', textAlign: 'center', color: 'var(--text2)' }}>
          Loading profile...
        </div>
      </div>
    );
  }

  const row = { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 };
  const sectionTitle = { fontSize: '1.1rem', fontWeight: 700, color: 'var(--text)', marginBottom: 16, marginTop: 8 };
  const label = { display: 'block', fontSize: '0.82rem', fontWeight: 600, color: 'var(--text)', marginBottom: 6 };
  const input = {
    width: '100%', padding: '11px 14px', borderRadius: 10,
    border: '1px solid var(--border)', background: 'var(--bg)',
    color: 'var(--text)', fontSize: '0.9rem', outline: 'none',
    boxSizing: 'border-box',
  };
  const select = { ...input, appearance: 'auto' };

  return (
    <div>
      <Navbar />
      <div style={{ padding: '80px 24px 40px', maxWidth: 720, margin: '0 auto' }}>
        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}>
          <h1 style={{ fontSize: '1.5rem', fontWeight: 700, color: 'var(--text)', marginBottom: 4 }}>Profile Settings</h1>
          <p style={{ color: 'var(--text2)', fontSize: '0.9rem', marginBottom: 24 }}>Manage your account and preferences</p>

          {message && (
            <div className="card" style={{
              padding: '12px 16px', marginBottom: 16,
              borderLeft: `3px solid ${message.type === 'success' ? 'var(--green)' : 'var(--red)'}`,
              color: message.type === 'success' ? 'var(--green)' : 'var(--red)',
              fontSize: '0.9rem',
            }}>
              {message.text}
              <button className="btn btn-secondary" style={{ marginLeft: 12, padding: '2px 10px', fontSize: '0.78rem' }}
                onClick={() => setMessage(null)}>Dismiss</button>
            </div>
          )}

          {/* Avatar */}
          <div className="card" style={{ marginBottom: 16 }}>
            <div style={sectionTitle}>Photo</div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
              <div style={{
                width: 80, height: 80, borderRadius: '50%',
                background: 'var(--bg2)', border: '2px solid var(--border)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                overflow: 'hidden', fontSize: '1.8rem', color: 'var(--text2)', flexShrink: 0,
              }}>
                {avatarUrl ? (
                  <img src={avatarUrl} alt="avatar" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                ) : (
                  user?.name?.[0]?.toUpperCase() || '?'
                )}
              </div>
              <div>
                <button className="btn btn-secondary" onClick={() => avatarRef.current?.click()}
                  style={{ fontSize: '0.82rem' }}>Upload Photo</button>
                <input ref={avatarRef} type="file" accept="image/*" style={{ display: 'none' }}
                  onChange={handleAvatarUpload} />
              </div>
            </div>
          </div>

          {/* Interviewer Avatar */}
          <div className="card" style={{ marginBottom: 16 }}>
            <div style={sectionTitle}>Interviewer Avatar</div>
            <p style={{ color: 'var(--text2)', fontSize: '0.82rem', marginBottom: 12 }}>
              This photo appears as the AI interviewer in live interviews. Upload your own or leave blank for the default options.
            </p>
            <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
              <div style={{
                width: 80, height: 80, borderRadius: '50%',
                background: 'var(--bg2)', border: '2px solid var(--border)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                overflow: 'hidden', fontSize: '1.8rem', color: 'var(--text2)', flexShrink: 0,
              }}>
                {interviewerAvatar ? (
                  <img src={interviewerAvatar} alt="interviewer avatar" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                ) : (
                  '🤖'
                )}
              </div>
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                <button className="btn btn-secondary" onClick={() => interviewerAvatarRef.current?.click()}
                  style={{ fontSize: '0.82rem' }}>Upload</button>
                <input ref={interviewerAvatarRef} type="file" accept="image/*" style={{ display: 'none' }}
                  onChange={handleInterviewerAvatarUpload} />
                {interviewerAvatar && (
                  <button className="btn btn-outline" onClick={handleRemoveInterviewerAvatar}
                    style={{ fontSize: '0.82rem', color: 'var(--red)', borderColor: 'var(--red)' }}>
                    Remove
                  </button>
                )}
              </div>
            </div>
          </div>

          {/* Identity */}
          <div className="card" style={{ marginBottom: 16 }}>
            <div style={sectionTitle}>Identity</div>
            <div style={row}>
              <div>
                <label style={label}>Display Name</label>
                <input style={input} value={form.display_name}
                  onChange={(e) => setField('display_name', e.target.value)} />
              </div>
              <div>
                <label style={label}>Email</label>
                <input style={{ ...input, opacity: 0.6 }} value={user?.email || ''} disabled />
              </div>
            </div>
            <div style={{ marginTop: 12 }}>
              <label style={label}>Phone</label>
              <input style={input} value={form.phone}
                onChange={(e) => setField('phone', e.target.value)} />
            </div>
          </div>

          {/* Career Context */}
          <div className="card" style={{ marginBottom: 16 }}>
            <div style={sectionTitle}>Career Context</div>
            <div>
              <label style={label}>Target Role</label>
              <input style={input} value={form.target_role}
                onChange={(e) => setField('target_role', e.target.value)} />
            </div>
            <div style={row}>
              <div>
                <label style={label}>Industry</label>
                <select style={select} value={form.target_industry}
                  onChange={(e) => setField('target_industry', e.target.value)}>
                  <option value="">Select</option>
                  {INDUSTRIES.map((i) => (
                    <option key={i.value} value={i.value}>{i.label}</option>
                  ))}
                </select>
              </div>
              <div>
                <label style={label}>Seniority</label>
                <select style={select} value={form.seniority_level}
                  onChange={(e) => setField('seniority_level', e.target.value)}>
                  <option value="">Select</option>
                  {SENIORITY_LEVELS.map((s) => (
                    <option key={s.value} value={s.value}>{s.label}</option>
                  ))}
                </select>
              </div>
            </div>
          </div>

          {/* Background */}
          <div className="card" style={{ marginBottom: 16 }}>
            <div style={sectionTitle}>Background & Experience</div>
            <div style={row}>
              <div>
                <label style={label}>Years of Experience</label>
                <input style={input} type="number" min={0} value={form.years_of_experience}
                  onChange={(e) => setField('years_of_experience', e.target.value)} />
              </div>
              <div>
                <label style={label}>Education</label>
                <select style={select} value={form.education_level}
                  onChange={(e) => setField('education_level', e.target.value)}>
                  <option value="">Select</option>
                  {EDUCATION_LEVELS.map((e) => (
                    <option key={e.value} value={e.value}>{e.label}</option>
                  ))}
                </select>
              </div>
            </div>
            <div style={{ marginTop: 12 }}>
              <label style={label}>Current / Last Company</label>
              <input style={input} value={form.current_company}
                onChange={(e) => setField('current_company', e.target.value)} />
            </div>
            <div style={{ marginTop: 12 }}>
              <label style={label}>LinkedIn URL</label>
              <input style={input} value={form.linkedin_url}
                onChange={(e) => setField('linkedin_url', e.target.value)} />
            </div>
            <div style={{ marginTop: 12 }}>
              <label style={label}>Resume</label>
              <ResumeManager />
            </div>
          </div>

          {/* Goals */}
          <div className="card" style={{ marginBottom: 16 }}>
            <div style={sectionTitle}>Goals & Focus</div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 16 }}>
              {FOCUS_AREAS.map((f) => {
                const sel = form.focus_areas.includes(f.value);
                return (
                  <button key={f.value} type="button" onClick={() => toggleFocus(f.value)}
                    style={{
                      padding: '7px 14px', borderRadius: 20, border: '1px solid var(--border)',
                      background: sel ? 'var(--blue)' : 'var(--bg2)',
                      color: sel ? '#fff' : 'var(--text)',
                      fontSize: '0.8rem', cursor: 'pointer', fontWeight: sel ? 600 : 400,
                      transition: 'all 0.15s',
                    }}>
                    {f.label}
                  </button>
                );
              })}
            </div>
            <div style={row}>
              <div>
                <label style={label}>Upcoming Interview Date</label>
                <input style={input} type="date" value={form.upcoming_interview_date}
                  onChange={(e) => setField('upcoming_interview_date', e.target.value)} />
              </div>
              <div>
                <label style={label}>Preferred Difficulty</label>
                <select style={select} value={form.preferred_difficulty}
                  onChange={(e) => setField('preferred_difficulty', e.target.value)}>
                  <option value="beginner">Beginner</option>
                  <option value="intermediate">Intermediate</option>
                  <option value="advanced">Advanced</option>
                  <option value="expert">Expert</option>
                </select>
              </div>
            </div>
          </div>

          {/* Preferences */}
          <div className="card" style={{ marginBottom: 16 }}>
            <div style={sectionTitle}>Preferences</div>
            <div style={row}>
              <div>
                <label style={label}>Timezone</label>
                <input style={input} value={form.timezone}
                  onChange={(e) => setField('timezone', e.target.value)} />
              </div>
              <div>
                <label style={label}>Locale</label>
                <input style={input} value={form.locale}
                  onChange={(e) => setField('locale', e.target.value)} />
              </div>
            </div>
            <div style={{ display: 'flex', gap: 24, marginTop: 12 }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer', fontSize: '0.9rem' }}>
                <input type="checkbox" checked={form.notify_email_digests}
                  onChange={(e) => setField('notify_email_digests', e.target.checked)} />
                Email Digests
              </label>
              <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer', fontSize: '0.9rem' }}>
                <input type="checkbox" checked={form.notify_session_reminders}
                  onChange={(e) => setField('notify_session_reminders', e.target.checked)} />
                Session Reminders
              </label>
            </div>
          </div>

          <button className="btn btn-primary" onClick={handleSave} disabled={saving}
            style={{ width: '100%', padding: '14px', fontSize: '1rem', marginBottom: 32 }}>
            {saving ? 'Saving...' : 'Save Changes'}
          </button>

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
