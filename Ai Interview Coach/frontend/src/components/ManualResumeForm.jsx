import { useState } from 'react';

export default function ManualResumeForm({ onSave, saving }) {
  const [name, setName] = useState('Manual Entry');
  const [skillsInput, setSkillsInput] = useState('');
  const [experienceYears, setExperienceYears] = useState('');
  const [educationEntries, setEducationEntries] = useState([]);
  const [summary, setSummary] = useState('');
  const [degree, setDegree] = useState('');
  const [institution, setInstitution] = useState('');

  function addEducation() {
    if (!degree.trim()) return;
    setEducationEntries((prev) => [...prev, { degree: degree.trim(), institution: institution.trim() }]);
    setDegree('');
    setInstitution('');
  }

  function removeEducation(i) {
    setEducationEntries((prev) => prev.filter((_, idx) => idx !== i));
  }

  async function handleSubmit(e) {
    e.preventDefault();
    const skills = skillsInput.split(',').map((s) => s.trim()).filter(Boolean);
    await onSave({ name, skills, experience_years: parseInt(experienceYears) || 0, education: educationEntries, summary });
  }

  const inputStyle = {
    width: '100%', padding: '10px 12px', borderRadius: 8,
    border: '1px solid var(--border)', background: 'var(--bg)',
    color: 'var(--text)', fontSize: '0.85rem', outline: 'none',
    boxSizing: 'border-box',
  };
  const label = { display: 'block', fontSize: '0.8rem', fontWeight: 600, color: 'var(--text)', marginBottom: 4 };

  return (
    <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      <div>
        <label style={label}>Resume Name</label>
        <input style={inputStyle} value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. My Software Resume" required />
      </div>
      <div>
        <label style={label}>Skills (comma separated)</label>
        <input style={inputStyle} value={skillsInput} onChange={(e) => setSkillsInput(e.target.value)}
          placeholder="Python, React, Docker, SQL, ..." />
      </div>
      <div>
        <label style={label}>Years of Experience</label>
        <input style={inputStyle} type="number" min={0} value={experienceYears}
          onChange={(e) => setExperienceYears(e.target.value)} placeholder="0" />
      </div>
      <div>
        <label style={label}>Education</label>
        <div className="flex gap-8" style={{ alignItems: 'center' }}>
          <input style={{ ...inputStyle, flex: 1 }} value={degree} onChange={(e) => setDegree(e.target.value)}
            placeholder="Degree (e.g. B.S. Computer Science)" />
          <input style={{ ...inputStyle, flex: 1 }} value={institution} onChange={(e) => setInstitution(e.target.value)}
            placeholder="Institution" />
          <button type="button" className="btn btn-secondary" onClick={addEducation}
            style={{ padding: '10px 14px', fontSize: '0.8rem', flexShrink: 0 }}>+</button>
        </div>
        {educationEntries.length > 0 && (
          <div style={{ marginTop: 6 }}>
            {educationEntries.map((e, i) => (
              <div key={i} className="flex" style={{ alignItems: 'center', gap: 8, marginBottom: 4, fontSize: '0.82rem', color: 'var(--text2)' }}>
                <span>🎓 {e.degree}{e.institution ? ` at ${e.institution}` : ''}</span>
                <button type="button" className="btn btn-secondary" onClick={() => removeEducation(i)}
                  style={{ padding: '2px 8px', fontSize: '0.7rem', color: 'var(--red)' }}>✕</button>
              </div>
            ))}
          </div>
        )}
      </div>
      <div>
        <label style={label}>Professional Summary</label>
        <textarea style={{ ...inputStyle, minHeight: 80, resize: 'vertical' }} value={summary}
          onChange={(e) => setSummary(e.target.value)}
          placeholder="Brief summary of your experience, achievements, and career goals..." />
      </div>
      <button type="submit" className="btn btn-primary" disabled={saving}
        style={{ alignSelf: 'flex-start', fontSize: '0.85rem', padding: '10px 24px' }}>
        {saving ? 'Saving...' : 'Save Resume'}
      </button>
    </form>
  );
}
