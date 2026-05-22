import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import ResumeDropzone from './ResumeDropzone';
import { listElevenLabsVoices } from '../api/client';
import { primeAudio } from '../utils/audioUnlock';

const DIFFICULTIES = ['beginner', 'intermediate', 'advanced', 'expert'];

const POPULAR_COMPANIES = [
  'Google', 'Meta', 'Amazon', 'Microsoft', 'Apple', 'Netflix', 'Stripe', 'McKinsey', 'Goldman Sachs', 'Deloitte',
  'Systems Limited', 'Arbisoft', 'Devsinc', 'NorthBay Solutions', 'Folio3', 'VentureDive',
  '10Pearls', 'Contour Software', 'Tintash', 'TekRevol', 'Convo', 'Careem',
  'Zameen.com', 'Jeeny', 'Bykea', 'Tajir', 'Bazaar', 'Finja',
];

const LS_AIM = 'aic_guided_aim';
const LS_COMPANY = 'aic_guided_company';
const LS_DURATION = 'aic_guided_duration';

function lsItem(key, fallback) {
  try { return localStorage.getItem(key) || fallback; } catch { return fallback; }
}
function lsSet(key, val) {
  try { localStorage.setItem(key, val); } catch {}
}

export default function GuidedSetup({ profile, onStart, loading }) {
  const [step, setStep] = useState(0);
  const [aim, setAim] = useState(() => lsItem(LS_AIM, ''));
  const [targetCompany, setTargetCompany] = useState(() => lsItem(LS_COMPANY, ''));
  const [durationMinutes, setDurationMinutes] = useState(() => {
    const v = lsItem(LS_DURATION, '30');
    return parseInt(v) || 30;
  });
  const [difficulty, setDifficulty] = useState(profile?.preferred_difficulty || 'intermediate');
  const [focusAreas, setFocusAreas] = useState(() => {
    try { return JSON.parse(profile?.focus_areas || '[]'); } catch { return []; }
  });
  const [focusInput, setFocusInput] = useState('');
  const [mode, setMode] = useState('text');
  const [avatarImage, setAvatarImage] = useState(() => {
    try { return localStorage.getItem('aic_interviewer_avatar') || null; } catch { return null; }
  });
  const [useElevenLabs, setUseElevenLabs] = useState(false);
  const [elevenlabsVoiceId, setElevenlabsVoiceId] = useState('');
  const [voices, setVoices] = useState([]);
  const [voicesLoading, setVoicesLoading] = useState(false);

  const hasResume = !!profile?.resume_text;
  const totalSteps = 3;

  useEffect(() => {
    if (useElevenLabs && voices.length === 0 && !voicesLoading) {
      setVoicesLoading(true);
      listElevenLabsVoices()
        .then((data) => {
          setVoices(data);
          if (data.length > 0 && !elevenlabsVoiceId) {
            setElevenlabsVoiceId(data[0].voice_id);
          }
        })
        .catch((err) => {
          console.warn('Failed to load ElevenLabs voices:', err);
        })
        .finally(() => setVoicesLoading(false));
    }
  }, [useElevenLabs]);

  useEffect(() => {
    if (mode === 'live' && !avatarImage) setAvatarImage('/avatars/male.jpg');
  }, [mode]);

  useEffect(() => { lsSet(LS_AIM, aim); }, [aim]);
  useEffect(() => { lsSet(LS_COMPANY, targetCompany); }, [targetCompany]);
  useEffect(() => { lsSet(LS_DURATION, String(durationMinutes)); }, [durationMinutes]);

  const addFocus = () => {
    const val = focusInput.trim();
    if (val && !focusAreas.includes(val)) { setFocusAreas([...focusAreas, val]); setFocusInput(''); }
  };

  const removeFocus = (idx) => setFocusAreas(focusAreas.filter((_, i) => i !== idx));

  const handleBegin = () => {
    primeAudio();
    onStart({
      aim,
      target_company: targetCompany,
      duration_minutes: durationMinutes,
      difficulty,
      focus_areas: focusAreas,
      mode,
      avatar_image: avatarImage,
      use_elevenlabs: useElevenLabs,
      elevenlabs_voice_id: elevenlabsVoiceId,
    });
  };

  return (
    <div style={{ maxWidth: 640, margin: '0 auto', padding: '80px 24px 40px' }}>
      <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
        <div style={{ fontSize: '1.6rem', fontWeight: 700, marginBottom: 4, color: 'var(--text)' }}>
          Guided Interview
        </div>
        <div style={{ color: 'var(--text2)', marginBottom: 32 }}>
          Step {step + 1} of {totalSteps} —
          {step === 0 ? ' Resume Check' : step === 1 ? ' Interview Details' : ' Interview Mode'}
        </div>

        {/* Stepper */}
        <div className="flex" style={{ gap: 8, marginBottom: 32 }}>
          {Array.from({ length: totalSteps }).map((_, i) => (
            <div key={i} style={{
              flex: 1, height: 4, borderRadius: 2,
              background: i <= step ? 'var(--blue)' : 'var(--border)',
              transition: 'background 0.3s',
            }} />
          ))}
        </div>

        {/* Step 0: Resume */}
        {step === 0 && (
          <motion.div key="s0" initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }}>
            <div className="card" style={{ padding: 24 }}>
              <div style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: 8, color: 'var(--text)' }}>Resume Check</div>
              <p style={{ color: 'var(--text2)', marginBottom: 16, lineHeight: 1.5 }}>
                Your resume helps us tailor questions. {hasResume ? 'You already have one.' : 'Upload one below.'}
              </p>
              {hasResume ? (
                <div style={{ padding: '16px 20px', background: 'var(--green-bg, #e8f5e9)', borderRadius: 8, border: '1px solid var(--green, #4caf50)', color: 'var(--green, #2e7d32)', display: 'flex', alignItems: 'center', gap: 12 }}>
                  <span style={{ fontSize: '1.3rem' }}>&#10003;</span>
                  <span>Resume loaded</span>
                </div>
              ) : <ResumeDropzone />}
            </div>
            <div className="flex" style={{ justifyContent: 'flex-end', marginTop: 20 }}>
              <button className="btn btn-primary" onClick={() => setStep(1)}>Continue</button>
            </div>
          </motion.div>
        )}

        {/* Step 1: Details */}
        {step === 1 && (
          <motion.div key="s1" initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }}>
            <div className="card" style={{ padding: 24, marginBottom: 16 }}>
              <div className="section-title" style={{ marginBottom: 4 }}>Aim / Purpose</div>
              <div className="section-sub" style={{ margin: '0 0 12px' }}>Why are you doing this interview?</div>
              <textarea className="input" rows={3} placeholder="e.g. Preparing for a senior backend role at Google" value={aim} onChange={e => setAim(e.target.value)} style={{ width: '100%', resize: 'vertical' }} />
            </div>

            <div className="card" style={{ padding: 24, marginBottom: 16 }}>
              <div className="section-title" style={{ marginBottom: 4 }}>Target Company</div>
              <div className="section-sub" style={{ margin: '0 0 12px' }}>Which company? (optional)</div>
              <div className="flex" style={{ flexWrap: 'wrap', gap: 6, marginBottom: 10 }}>
                {POPULAR_COMPANIES.map(c => (
                  <button key={c} onClick={() => setTargetCompany(targetCompany === c ? '' : c)} style={{
                    padding: '4px 12px', borderRadius: 16, border: `1px solid ${targetCompany === c ? 'var(--blue)' : 'var(--border)'}`,
                    background: targetCompany === c ? 'var(--blue-bg, #e3f2fd)' : 'transparent',
                    color: targetCompany === c ? 'var(--blue)' : 'var(--text2)', cursor: 'pointer', fontSize: '0.85rem', transition: 'all 0.2s',
                  }}>{c}</button>
                ))}
              </div>
              <input className="input" placeholder="Or type..." value={targetCompany} onChange={e => setTargetCompany(e.target.value)} style={{ width: '100%' }} />
            </div>

            <div className="card" style={{ padding: 24, marginBottom: 16 }}>
              <div className="section-title" style={{ marginBottom: 4 }}>Duration</div>
              <div className="section-sub" style={{ margin: '0 0 12px' }}>How long? <strong>{durationMinutes} min</strong></div>
              <input
                type="range"
                min={1}
                max={90}
                step={1}
                value={durationMinutes}
                onChange={(e) => setDurationMinutes(Number(e.target.value))}
                style={{
                  width: '100%', height: 6, borderRadius: 3, appearance: 'none',
                  background: `linear-gradient(to right, var(--blue) ${((durationMinutes - 1) / 89) * 100}%, var(--border) ${((durationMinutes - 1) / 89) * 100}%)`,
                  outline: 'none', cursor: 'pointer',
                }}
              />
              <div className="flex" style={{ justifyContent: 'space-between', marginTop: 4, fontSize: '0.78rem', color: 'var(--text3)' }}>
                <span>1 min</span>
                <span>90 min</span>
              </div>
            </div>

            <div className="card" style={{ padding: 24, marginBottom: 16 }}>
              <div className="section-title" style={{ marginBottom: 4 }}>Difficulty</div>
              <div className="section-sub" style={{ margin: '0 0 12px' }}>How challenging?</div>
              <div className="flex" style={{ gap: 8 }}>
                {DIFFICULTIES.map(d => (
                  <button key={d} className={`btn ${difficulty === d ? 'btn-primary' : 'btn-outline'}`} onClick={() => setDifficulty(d)} style={{ flex: 1 }}>
                    {d.charAt(0).toUpperCase() + d.slice(1)}
                  </button>
                ))}
              </div>
            </div>

            <div className="card" style={{ padding: 24, marginBottom: 16 }}>
              <div className="section-title" style={{ marginBottom: 4 }}>Focus Areas</div>
              <div className="section-sub" style={{ margin: '0 0 12px' }}>Topics to emphasize</div>
              <div className="flex" style={{ flexWrap: 'wrap', gap: 6, marginBottom: 10 }}>
                {focusAreas.map((a, i) => (
                  <span key={i} style={{ display: 'inline-flex', alignItems: 'center', gap: 4, padding: '4px 10px', borderRadius: 16, background: 'var(--blue-bg, #e3f2fd)', color: 'var(--blue)', fontSize: '0.85rem' }}>
                    {a}
                    <button onClick={() => removeFocus(i)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--blue)', padding: 0, fontSize: '1rem', lineHeight: 1 }}>&times;</button>
                  </span>
                ))}
              </div>
              <div className="flex" style={{ gap: 8 }}>
                <input className="input" placeholder="e.g. system design, behavioral" value={focusInput} onChange={e => setFocusInput(e.target.value)} onKeyDown={e => e.key === 'Enter' && addFocus()} style={{ flex: 1 }} />
                <button className="btn btn-outline" onClick={addFocus}>Add</button>
              </div>
            </div>

            <div className="flex" style={{ justifyContent: 'space-between', marginTop: 20 }}>
              <button className="btn btn-outline" onClick={() => setStep(0)}>Back</button>
              <button className="btn btn-primary" onClick={() => setStep(2)}>Continue</button>
            </div>
          </motion.div>
        )}

        {/* Step 2: Mode + Avatar */}
        {step === 2 && (
          <motion.div key="s2" initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }}>
            <div style={{ fontSize: '1.1rem', fontWeight: 600, marginBottom: 16, color: 'var(--text)' }}>
              Choose Interview Mode
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 24 }}>
              <button onClick={() => setMode('text')} style={{
                padding: 24, borderRadius: 12, border: `2px solid ${mode === 'text' ? 'var(--blue)' : 'var(--border)'}`,
                background: mode === 'text' ? 'var(--blue-bg, #e3f2fd)' : 'var(--bg2)',
                cursor: 'pointer', textAlign: 'left', transition: 'all 0.2s',
              }}>
                <div style={{ fontSize: '2rem', marginBottom: 8 }}>🎤</div>
                <div style={{ fontWeight: 600, fontSize: '0.95rem', color: 'var(--text)', marginBottom: 4 }}>Text Interview</div>
                <div style={{ fontSize: '0.82rem', color: 'var(--text2)', lineHeight: 1.4 }}>
                  Chat-based interview. AI asks questions, you type your answers. Simple and focused.
                </div>
              </button>

              <button onClick={() => setMode('live')} style={{
                padding: 24, borderRadius: 12, border: `2px solid ${mode === 'live' ? 'var(--blue)' : 'var(--border)'}`,
                background: mode === 'live' ? 'var(--blue-bg, #e3f2fd)' : 'var(--bg2)',
                cursor: 'pointer', textAlign: 'left', transition: 'all 0.2s',
              }}>
                <div style={{ fontSize: '2rem', marginBottom: 8 }}>🎥</div>
                <div style={{ fontWeight: 600, fontSize: '0.95rem', color: 'var(--text)', marginBottom: 4 }}>Live Interview</div>
                <div style={{ fontSize: '0.82rem', color: 'var(--text2)', lineHeight: 1.4 }}>
                  AI avatar conducts the interview. Webcam tracks eye contact &amp; body language in real time.
                </div>
              </button>
            </div>

            {/* Avatar selection (only for live mode) */}
            {mode === 'live' && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                className="card"
                style={{ padding: 24, marginBottom: 16, overflow: 'hidden' }}
              >
                <div className="section-title" style={{ marginBottom: 4 }}>AI Interviewer Avatar</div>
                <div className="section-sub" style={{ margin: '0 0 12px' }}>
                  Choose the avatar that will conduct your interview.
                </div>
                <div className="flex" style={{ gap: 16, justifyContent: 'center', flexWrap: 'wrap' }}>
                  {[
                    { id: 'male', label: 'Male', path: '/avatars/male.jpg', icon: '👨' },
                    { id: 'female', label: 'Female', path: '/avatars/female.jpg', icon: '👩' },
                    ...(() => {
                      const custom = (() => { try { return localStorage.getItem('aic_interviewer_avatar') || ''; } catch { return ''; } })();
                      return custom ? [{ id: 'custom', label: 'Custom', path: custom, icon: null }] : [];
                    })(),
                  ].map(({ id, label, path, icon }) => (
                    <button
                      key={id}
                      onClick={() => setAvatarImage(path)}
                      style={{
                        width: 120, padding: 16, borderRadius: 12,
                        border: `2px solid ${avatarImage === path ? 'var(--blue)' : 'var(--border)'}`,
                        background: avatarImage === path ? 'var(--blue-bg, #e3f2fd)' : 'var(--bg2)',
                        cursor: 'pointer', textAlign: 'center',
                        transition: 'all 0.2s',
                      }}
                    >
                      {icon ? (
                        <div style={{ fontSize: '2.5rem', marginBottom: 8 }}>{icon}</div>
                      ) : (
                        <div style={{
                          width: 48, height: 48, borderRadius: '50%', margin: '0 auto 8px',
                          overflow: 'hidden', border: '2px solid var(--border)',
                        }}>
                          <img src={path} alt="custom" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                        </div>
                      )}
                      <div style={{
                        fontWeight: avatarImage === path ? 600 : 400,
                        fontSize: '0.9rem', color: avatarImage === path ? 'var(--blue)' : 'var(--text)',
                      }}>
                        {label}
                      </div>
                    </button>
                  ))}
                </div>
              </motion.div>
            )}

            {/* ElevenLabs Voice Toggle */}
            <div className="card" style={{ padding: 24, marginBottom: 16 }}>
              <div className="flex" style={{ alignItems: 'center', justifyContent: 'space-between', marginBottom: useElevenLabs ? 12 : 0 }}>
                <div>
                  <div className="section-title" style={{ marginBottom: 2 }}>ElevenLabs Voice</div>
                  <div className="section-sub" style={{ margin: 0 }}>Use AI voice for questions</div>
                </div>
                <label style={{ position: 'relative', display: 'inline-block', width: 44, height: 24, cursor: 'pointer' }}>
                  <input type="checkbox" checked={useElevenLabs} onChange={e => setUseElevenLabs(e.target.checked)} style={{ opacity: 0, width: 0, height: 0 }} />
                  <span style={{
                    position: 'absolute', inset: 0, borderRadius: 24,
                    background: useElevenLabs ? 'var(--blue)' : 'var(--border)',
                    transition: 'background 0.2s',
                  }}>
                    <span style={{
                      position: 'absolute', left: useElevenLabs ? 22 : 2, top: 2,
                      width: 20, height: 20, borderRadius: '50%', background: '#fff',
                      transition: 'left 0.2s',
                    }} />
                  </span>
                </label>
              </div>
              {useElevenLabs && (
                <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }}>
                  {voicesLoading ? (
                    <div style={{ color: 'var(--text2)', fontSize: '0.85rem' }}>Loading voices...</div>
                  ) : (
                    <div>
                      <select
                        className="input"
                        value={elevenlabsVoiceId}
                        onChange={e => setElevenlabsVoiceId(e.target.value)}
                        style={{ width: '100%', padding: '8px 12px' }}
                      >
                        {voices.length === 0 && <option value="">No voices available</option>}
                        {voices.map((v) => (
                          <option key={v.voice_id} value={v.voice_id}>
                            {v.name} {v.category ? `(${v.category})` : ''}
                          </option>
                        ))}
                      </select>
                      {voices.length > 0 && (
                        <div style={{ marginTop: 8, fontSize: '0.78rem', color: 'var(--text3)' }}>
                          {voices.length} voice{voices.length !== 1 ? 's' : ''} available
                        </div>
                      )}
                    </div>
                  )}
                </motion.div>
              )}
            </div>

            <div className="flex" style={{ justifyContent: 'space-between', marginTop: 20 }}>
              <button className="btn btn-outline" onClick={() => setStep(1)}>Back</button>
              <button className="btn btn-primary" onClick={handleBegin} disabled={loading}>
                {loading ? 'Starting...' : 'Begin Interview'}
              </button>
            </div>
          </motion.div>
        )}
      </motion.div>
    </div>
  );
}
