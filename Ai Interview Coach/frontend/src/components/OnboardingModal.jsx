import { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useAuth } from '../context/AuthContext';
import { updateProfile, uploadAvatar, uploadResume } from '../api/client';
import ResumeDropzone from './ResumeDropzone';

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

const DIFFICULTY_LEVELS = [
  { value: 'beginner', label: 'Beginner' },
  { value: 'intermediate', label: 'Intermediate' },
  { value: 'advanced', label: 'Advanced' },
  { value: 'expert', label: 'Expert' },
];

const STEPS = [
  { id: 'identity', title: 'About You', subtitle: 'Tell us who you are' },
  { id: 'career', title: 'Career Context', subtitle: 'What role are you targeting?' },
  { id: 'background', title: 'Background', subtitle: 'Your experience & education' },
  { id: 'goals', title: 'Goals & Focus', subtitle: 'What do you want to improve?' },
  { id: 'preferences', title: 'Preferences', subtitle: 'Customize your experience' },
];

const slideVariants = {
  enter: (dir) => ({ x: dir > 0 ? 300 : -300, opacity: 0 }),
  center: { x: 0, opacity: 1 },
  exit: (dir) => ({ x: dir > 0 ? -300 : 300, opacity: 0 }),
};

export default function OnboardingModal({ onComplete }) {
  const { user, refreshProfile } = useAuth();
  const [step, setStep] = useState(0);
  const [dir, setDir] = useState(1);
  const [saving, setSaving] = useState(false);
  const fileInputRef = useRef(null);

  const [form, setForm] = useState({
    display_name: user?.name || '',
    phone: '',
    target_role: '',
    target_industry: '',
    seniority_level: '',
    years_of_experience: '',
    current_company: '',
    education_level: '',
    linkedin_url: '',
    focus_areas: [],
    upcoming_interview_date: '',
    preferred_difficulty: 'intermediate',
    timezone: Intl.DateTimeFormat().resolvedOptions().timeZone || '',
    locale: navigator.language || 'en-US',
    notify_email_digests: true,
    notify_session_reminders: true,
    mic_default: '',
    camera_default: '',
    avatarFile: null,
    resumeFile: null,
  });

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

  function goNext() {
    setDir(1);
    setStep((s) => Math.min(s + 1, STEPS.length - 1));
  }

  function goBack() {
    setDir(-1);
    setStep((s) => Math.max(s - 1, 0));
  }

  async function handleSave() {
    setSaving(true);
    try {
      if (form.avatarFile) {
        await uploadAvatar(form.avatarFile);
      }
      if (form.resumeFile) {
        await uploadResume(form.resumeFile);
      }
      await updateProfile({
        display_name: form.display_name,
        phone: form.phone,
        target_role: form.target_role,
        target_industry: form.target_industry,
        seniority_level: form.seniority_level,
        years_of_experience: form.years_of_experience ? parseInt(form.years_of_experience) : undefined,
        current_company: form.current_company,
        education_level: form.education_level,
        linkedin_url: form.linkedin_url,
        focus_areas: form.focus_areas,
        upcoming_interview_date: form.upcoming_interview_date || undefined,
        preferred_difficulty: form.preferred_difficulty,
        timezone: form.timezone,
        locale: form.locale,
        notify_email_digests: form.notify_email_digests,
        notify_session_reminders: form.notify_session_reminders,
        mic_default: form.mic_default,
        camera_default: form.camera_default,
        profile_completed: true,
      });
      await refreshProfile();
      onComplete?.();
    } catch (e) {
      console.error('Profile save failed:', e);
    } finally {
      setSaving(false);
    }
  }

  async function handleSkip() {
    try {
      await updateProfile({ profile_completed: true });
      await refreshProfile();
      onComplete?.();
    } catch {
      onComplete?.();
    }
  }

  const canProceed = () => {
    if (step === 0) return form.display_name.trim().length > 0;
    return true;
  };

  const inputStyle = {
    width: '100%',
    padding: '12px 14px',
    borderRadius: 10,
    border: '1px solid var(--border)',
    background: 'var(--bg)',
    color: 'var(--text)',
    fontSize: '0.9rem',
    outline: 'none',
    boxSizing: 'border-box',
  };

  const labelStyle = {
    display: 'block',
    fontSize: '0.82rem',
    fontWeight: 600,
    color: 'var(--text)',
    marginBottom: 6,
  };

  const selectStyle = { ...inputStyle, appearance: 'auto' };

  return (
    <div style={{
      position: 'fixed', inset: 0, zIndex: 9999,
      background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(6px)',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      padding: 20,
    }}>
      <motion.div
        className="card"
        initial={{ opacity: 0, scale: 0.95, y: 20 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
        style={{
          width: '100%', maxWidth: 580, maxHeight: '90vh', overflow: 'hidden',
          display: 'flex', flexDirection: 'column', padding: 0,
        }}
      >
        {/* Header */}
        <div style={{ padding: '28px 32px 0' }}>
          <div style={{ display: 'flex', gap: 6, marginBottom: 20 }}>
            {STEPS.map((s, i) => (
              <div key={s.id} style={{
                flex: 1, height: 3, borderRadius: 2,
                background: i <= step ? 'var(--blue)' : 'var(--border)',
                transition: 'background 0.3s',
              }} />
            ))}
          </div>
          <h2 style={{ margin: 0, fontSize: '1.3rem', fontWeight: 700, color: 'var(--text)' }}>
            {STEPS[step].title}
          </h2>
          <p style={{ margin: '4px 0 0', fontSize: '0.88rem', color: 'var(--text2)' }}>
            {STEPS[step].subtitle}
          </p>
        </div>

        {/* Body */}
        <div style={{ flex: 1, overflow: 'auto', padding: '24px 32px', minHeight: 280 }}>
          <AnimatePresence mode="wait" custom={dir}>
            <motion.div
              key={step}
              custom={dir}
              variants={slideVariants}
              initial="enter"
              animate="center"
              exit="exit"
              transition={{ duration: 0.25, ease: 'easeInOut' }}
            >
              {/* Step 0: Identity */}
              {step === 0 && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
                    <div
                      onClick={() => fileInputRef.current?.click()}
                      style={{
                        width: 72, height: 72, borderRadius: '50%',
                        background: 'var(--bg2)', border: '2px dashed var(--border)',
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        cursor: 'pointer', overflow: 'hidden', flexShrink: 0,
                        fontSize: '1.6rem', color: 'var(--text2)',
                      }}
                    >
                      {form.avatarFile ? (
                        <img src={URL.createObjectURL(form.avatarFile)} alt="avatar"
                          style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                      ) : (
                        user?.name?.[0]?.toUpperCase() || '?'
                      )}
                    </div>
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept="image/*"
                      style={{ display: 'none' }}
                      onChange={(e) => {
                        if (e.target.files[0]) setField('avatarFile', e.target.files[0]);
                      }}
                    />
                    <div>
                      <div style={{ fontWeight: 600, fontSize: '0.9rem', color: 'var(--text)' }}>
                        Profile Photo
                      </div>
                      <div style={{ fontSize: '0.8rem', color: 'var(--text2)', marginTop: 2 }}>
                        Click to upload
                      </div>
                    </div>
                  </div>
                  <div>
                    <label style={labelStyle}>Display Name *</label>
                    <input style={inputStyle} placeholder="How should we address you?"
                      value={form.display_name}
                      onChange={(e) => setField('display_name', e.target.value)} />
                  </div>
                  <div>
                    <label style={labelStyle}>Email</label>
                    <input style={{ ...inputStyle, opacity: 0.6 }} value={user?.email || ''} disabled />
                  </div>
                  <div>
                    <label style={labelStyle}>Phone (optional)</label>
                    <input style={inputStyle} placeholder="+1 555-123-4567"
                      value={form.phone}
                      onChange={(e) => setField('phone', e.target.value)} />
                  </div>
                </div>
              )}

              {/* Step 1: Career Context */}
              {step === 1 && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                  <div>
                    <label style={labelStyle}>Target Job Title</label>
                    <input style={inputStyle} placeholder="e.g. Senior Backend Engineer"
                      value={form.target_role}
                      onChange={(e) => setField('target_role', e.target.value)} />
                  </div>
                  <div>
                    <label style={labelStyle}>Target Industry</label>
                    <select style={selectStyle} value={form.target_industry}
                      onChange={(e) => setField('target_industry', e.target.value)}>
                      <option value="">Select industry</option>
                      {INDUSTRIES.map((i) => (
                        <option key={i.value} value={i.value}>{i.label}</option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label style={labelStyle}>Seniority Level</label>
                    <select style={selectStyle} value={form.seniority_level}
                      onChange={(e) => setField('seniority_level', e.target.value)}>
                      <option value="">Select level</option>
                      {SENIORITY_LEVELS.map((s) => (
                        <option key={s.value} value={s.value}>{s.label}</option>
                      ))}
                    </select>
                  </div>
                </div>
              )}

              {/* Step 2: Background */}
              {step === 2 && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                    <div>
                      <label style={labelStyle}>Years of Experience</label>
                      <input style={inputStyle} type="number" min={0} placeholder="0"
                        value={form.years_of_experience}
                        onChange={(e) => setField('years_of_experience', e.target.value)} />
                    </div>
                    <div>
                      <label style={labelStyle}>Education</label>
                      <select style={selectStyle} value={form.education_level}
                        onChange={(e) => setField('education_level', e.target.value)}>
                        <option value="">Select</option>
                        {EDUCATION_LEVELS.map((e) => (
                          <option key={e.value} value={e.value}>{e.label}</option>
                        ))}
                      </select>
                    </div>
                  </div>
                  <div>
                    <label style={labelStyle}>Current / Last Company</label>
                    <input style={inputStyle} placeholder="e.g. Google, Stripe, ..."
                      value={form.current_company}
                      onChange={(e) => setField('current_company', e.target.value)} />
                  </div>
                  <div>
                    <label style={labelStyle}>LinkedIn URL (optional)</label>
                    <input style={inputStyle} placeholder="https://linkedin.com/in/..."
                      value={form.linkedin_url}
                      onChange={(e) => setField('linkedin_url', e.target.value)} />
                  </div>
                  <div>
                    <label style={labelStyle}>Resume (PDF)</label>
                    <ResumeDropzone onFile={(f) => setField('resumeFile', f)} />
                    {form.resumeFile && (
                      <div style={{ fontSize: '0.8rem', color: 'var(--green)', marginTop: 4 }}>
                        ✓ {form.resumeFile.name}
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Step 3: Goals & Focus */}
              {step === 3 && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                  <div>
                    <label style={labelStyle}>What do you want to improve?</label>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, maxHeight: 200, overflow: 'auto' }}>
                      {FOCUS_AREAS.map((f) => {
                        const selected = form.focus_areas.includes(f.value);
                        return (
                          <button key={f.value} type="button"
                            onClick={() => toggleFocus(f.value)}
                            style={{
                              padding: '8px 14px', borderRadius: 20, border: '1px solid var(--border)',
                              background: selected ? 'var(--blue)' : 'var(--bg2)',
                              color: selected ? '#fff' : 'var(--text)',
                              fontSize: '0.8rem', cursor: 'pointer', fontWeight: selected ? 600 : 400,
                              transition: 'all 0.15s',
                            }}
                          >
                            {f.label}
                          </button>
                        );
                      })}
                    </div>
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                    <div>
                      <label style={labelStyle}>Upcoming Interview Date</label>
                      <input style={inputStyle} type="date"
                        value={form.upcoming_interview_date}
                        onChange={(e) => setField('upcoming_interview_date', e.target.value)} />
                    </div>
                    <div>
                      <label style={labelStyle}>Preferred Difficulty</label>
                      <select style={selectStyle} value={form.preferred_difficulty}
                        onChange={(e) => setField('preferred_difficulty', e.target.value)}>
                        {DIFFICULTY_LEVELS.map((d) => (
                          <option key={d.value} value={d.value}>{d.label}</option>
                        ))}
                      </select>
                    </div>
                  </div>
                </div>
              )}

              {/* Step 4: Preferences */}
              {step === 4 && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                  <div>
                    <label style={labelStyle}>Timezone</label>
                    <input style={inputStyle}
                      value={form.timezone}
                      onChange={(e) => setField('timezone', e.target.value)} />
                  </div>
                  <div>
                    <label style={labelStyle}>Language / Locale</label>
                    <input style={inputStyle}
                      value={form.locale}
                      onChange={(e) => setField('locale', e.target.value)} />
                  </div>
                  <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap' }}>
                    <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer', fontSize: '0.9rem', color: 'var(--text)' }}>
                      <input type="checkbox" checked={form.notify_email_digests}
                        onChange={(e) => setField('notify_email_digests', e.target.checked)} />
                      Email Digests
                    </label>
                    <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer', fontSize: '0.9rem', color: 'var(--text)' }}>
                      <input type="checkbox" checked={form.notify_session_reminders}
                        onChange={(e) => setField('notify_session_reminders', e.target.checked)} />
                      Session Reminders
                    </label>
                  </div>
                </div>
              )}
            </motion.div>
          </AnimatePresence>
        </div>

        {/* Footer */}
        <div style={{
          padding: '16px 32px 24px',
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          borderTop: '1px solid var(--border)',
        }}>
          <button className="btn btn-secondary" onClick={handleSkip}
            style={{ fontSize: '0.85rem', padding: '10px 20px' }}>
            Complete Later
          </button>
          <div style={{ display: 'flex', gap: 10 }}>
            {step > 0 && (
              <button className="btn btn-secondary" onClick={goBack}
                style={{ fontSize: '0.85rem', padding: '10px 24px' }}>
                Back
              </button>
            )}
            {step < STEPS.length - 1 ? (
              <button className="btn btn-primary" onClick={goNext} disabled={!canProceed()}
                style={{ fontSize: '0.85rem', padding: '10px 24px' }}>
                Continue
              </button>
            ) : (
              <button className="btn btn-primary" onClick={handleSave} disabled={saving}
                style={{ fontSize: '0.85rem', padding: '10px 24px' }}>
                {saving ? <><span className="spinner" style={{ width: 16, height: 16 }} /> Saving...</> : 'Complete Setup'}
              </button>
            )}
          </div>
        </div>
      </motion.div>
    </div>
  );
}
