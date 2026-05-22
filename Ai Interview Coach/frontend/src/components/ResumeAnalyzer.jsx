import { motion } from 'framer-motion';

const ROLES = [
  'backend', 'frontend', 'fullstack', 'data science', 'devops',
  'mobile', 'product management', 'data engineering', 'ml engineering', 'security',
];

const label = { fontSize: '0.75rem', color: 'var(--text2)', marginBottom: 2 };

function Badge({ children, color, bg }) {
  return (
    <span style={{
      display: 'inline-block', padding: '3px 10px', borderRadius: 12,
      fontSize: '0.78rem', fontWeight: 500,
      background: bg || 'var(--bg3)', color: color || 'var(--text)',
      margin: '2px 4px 2px 0',
    }}>{children}</span>
  );
}

function ScoreRing({ score, size = 56, label: lbl }) {
  const color = score > 70 ? 'var(--green)' : score > 40 ? 'var(--yellow)' : 'var(--red)';
  return (
    <div style={{ textAlign: 'center' }}>
      <div style={{
        width: size, height: size, borderRadius: '50%',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        background: `${color}15`, border: `3px solid ${color}`,
        margin: '0 auto',
      }}>
        <span style={{ fontSize: size > 50 ? '1rem' : '0.85rem', fontWeight: 700, color }}>{score}</span>
      </div>
      {lbl && <div style={{ fontSize: '0.7rem', color: 'var(--text2)', marginTop: 3 }}>{lbl}</div>}
    </div>
  );
}

export default function ResumeAnalyzer({
  profile,
  analysis,
  profileLoading,
  analysisLoading,
  selectedRole,
  onRoleChange,
  onAnalyze,
}) {
  if (profileLoading) {
    return (
      <div style={{ textAlign: 'center', padding: 16 }}>
        <div className="flex center gap-8">
          <span className="spinner" style={{ width: 16, height: 16 }} />
          <span style={{ color: 'var(--text2)', fontSize: '0.82rem' }}>Analyzing...</span>
        </div>
      </div>
    );
  }

  if (!profile) return null;

  return (
    <div>
      {/* General Profile */}
      <div style={{ marginBottom: 12 }}>
        {profile.description && (
          <p style={{ fontSize: '0.85rem', color: 'var(--text2)', lineHeight: 1.5, marginBottom: 12 }}>
            {profile.description}
          </p>
        )}

        <div className="flex" style={{ gap: 16, marginBottom: 12, flexWrap: 'wrap', alignItems: 'center' }}>
          {profile.overall_structure_score != null && (
            <ScoreRing score={profile.overall_structure_score} label="Structure" />
          )}
          {profile.structure?.sections_found?.length > 0 && (
            <div>
              <div style={label}>Sections</div>
              <div style={{ marginTop: 2 }}>
                {profile.structure.sections_found.map((s) => (
                  <Badge key={s} color="var(--green)" bg="rgba(76,175,80,0.08)">{s}</Badge>
                ))}
              </div>
            </div>
          )}
          {profile.structure?.format_quality && (
            <div>
              <div style={label}>Format</div>
              <Badge color="var(--purple)" bg="rgba(156,39,176,0.1)">
                {profile.structure.format_quality}
              </Badge>
            </div>
          )}
          {profile.structure?.readability && (
            <div>
              <div style={label}>Readability</div>
              <Badge color="var(--green)" bg="rgba(76,175,80,0.1)">
                {profile.structure.readability}
              </Badge>
            </div>
          )}
        </div>

        {profile.suggested_roles?.length > 0 && (
          <div style={{ marginBottom: 8 }}>
            <div style={label}>Suited For</div>
            <div style={{ marginTop: 2 }}>
              {profile.suggested_roles.map((r) => (
                <Badge key={r} color="var(--blue)" bg="var(--blue-bg, #e3f2fd)">{r}</Badge>
              ))}
            </div>
          </div>
        )}

        {profile.structure?.issues?.length > 0 && (
          <div style={{ marginTop: 8 }}>
            <div style={{ fontSize: '0.75rem', color: 'var(--yellow)', fontWeight: 600, marginBottom: 3 }}>Issues</div>
            <ul style={{ margin: 0, paddingLeft: 16, fontSize: '0.8rem', color: 'var(--text2)' }}>
              {profile.structure.issues.map((s, i) => <li key={i} style={{ marginBottom: 2 }}>{s}</li>)}
            </ul>
          </div>
        )}
      </div>

      {/* Role Tailor Section */}
      <div style={{
        padding: '12px 14px', borderRadius: 8, background: 'var(--bg)',
        border: '1px solid var(--border)',
      }}>
        <div style={{ fontWeight: 600, fontSize: '0.85rem', color: 'var(--text)', marginBottom: 8 }}>
          Tailor for a Role
        </div>
        <div className="flex" style={{ gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
          <select
            className="input"
            value={selectedRole}
            onChange={(e) => onRoleChange(e.target.value)}
            style={{ flex: 1, minWidth: 160, padding: '7px 10px', fontSize: '0.85rem' }}
          >
            <option value="">-- Select a role --</option>
            {ROLES.map((r) => (
              <option key={r} value={r}>{r.charAt(0).toUpperCase() + r.slice(1)}</option>
            ))}
          </select>
          <button
            className="btn btn-primary"
            onClick={onAnalyze}
            disabled={analysisLoading}
            style={{ fontSize: '0.85rem', padding: '7px 16px', whiteSpace: 'nowrap' }}
          >
            {analysisLoading ? 'Analyzing...' : profile && selectedRole ? 'Update Feedback' : 'Run Analysis'}
          </button>
        </div>
      </div>

      {/* Role-specific results */}
      {analysisLoading && (
        <div className="flex center gap-8" style={{ padding: 16 }}>
          <span className="spinner" style={{ width: 16, height: 16 }} />
          <span style={{ color: 'var(--text2)', fontSize: '0.82rem' }}>Analyzing for {selectedRole}...</span>
        </div>
      )}

      {analysis && !analysisLoading && (
        <motion.div initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} style={{ marginTop: 12 }}>
          {/* ATS + metrics row */}
          <div className="flex" style={{ alignItems: 'center', gap: 16, marginBottom: 12, flexWrap: 'wrap' }}>
            <ScoreRing score={analysis.ats?.ats_score || 0} label="ATS Score" size={52} />
            <div style={{ flex: 1, minWidth: 200 }}>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px 14px' }}>
                {[
                  { l: 'Keywords', v: `${analysis.ats?.keyword_matches || 0}/${analysis.ats?.total_keywords || 0}`, c: 'var(--blue)' },
                  { l: 'Action Verbs', v: `${analysis.ats?.action_verbs_found || 0} found`, c: 'var(--green)' },
                  { l: 'Format', v: `${((analysis.ats?.format_score || 0) * 100).toFixed(0)}%`, c: 'var(--purple)' },
                  { l: 'Quantification', v: `${((analysis.ats?.quantification_score || 0) * 100).toFixed(0)}%`, c: 'var(--yellow)' },
                ].map(({ l, v, c }) => (
                  <div key={l}>
                    <div style={label}>{l}</div>
                    <div style={{ fontSize: '0.82rem', fontWeight: 600, color: c }}>{v}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Skills gap */}
          {analysis.skills_gap && (analysis.skills_gap.matched?.length > 0 || analysis.skills_gap.missing?.length > 0) && (
            <div style={{ marginBottom: 12 }}>
              <div style={{ fontSize: '0.82rem', fontWeight: 600, color: 'var(--text)', marginBottom: 6 }}>
                Skills Gap for <span style={{ color: 'var(--blue)', textTransform: 'capitalize' }}>{selectedRole}</span>
              </div>
              <div className="flex" style={{ gap: 6, flexWrap: 'wrap', marginBottom: 6 }}>
                {analysis.skills_gap.matched?.map((s) => (
                  <Badge key={s} color="var(--green)" bg="rgba(76,175,80,0.12)">{s}</Badge>
                ))}
                {analysis.skills_gap.missing?.slice(0, 10).map((s) => (
                  <Badge key={s} color="var(--red)" bg="rgba(234,67,53,0.1)">{s}</Badge>
                ))}
              </div>
              {analysis.skills_gap.suggestions?.length > 0 && (
                <ul style={{ margin: '6px 0 0', paddingLeft: 16, fontSize: '0.8rem', color: 'var(--text2)' }}>
                  {analysis.skills_gap.suggestions.map((s, i) => (
                    <li key={i} style={{ marginBottom: 3 }}>{s}</li>
                  ))}
                </ul>
              )}
            </div>
          )}

          {/* Summary */}
          {analysis.summary && (
            <div style={{
              background: 'var(--bg)', borderLeft: '3px solid var(--blue)',
              padding: '10px 12px', borderRadius: 6, fontSize: '0.82rem',
              color: 'var(--text2)', lineHeight: 1.5,
            }}>
              {analysis.summary}
            </div>
          )}
        </motion.div>
      )}
    </div>
  );
}
