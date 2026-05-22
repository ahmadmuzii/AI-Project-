import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import ResumeDropzone from './ResumeDropzone';
import ResumeCard from './ResumeCard';
import ManualResumeForm from './ManualResumeForm';
import {
  listResumes, uploadResumeFile, createManualResume,
  setPrimaryResume, deleteResume,
} from '../api/client';
import { useAuth } from '../context/AuthContext';

export default function ResumeManager({ compact = false }) {
  const { refreshProfile } = useAuth();
  const [resumes, setResumes] = useState([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [showManual, setShowManual] = useState(false);
  const [error, setError] = useState(null);

  const fetchResumes = useCallback(async () => {
    try {
      const data = await listResumes();
      setResumes(data || []);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchResumes(); }, [fetchResumes]);

  async function handleUpload(file) {
    setUploading(true);
    setError(null);
    try {
      const name = file.name.replace(/\.[^.]+$/, '');
      await uploadResumeFile(file, name);
      await fetchResumes();
      await refreshProfile();
    } catch (e) {
      setError(e.message);
    } finally {
      setUploading(false);
    }
  }

  async function handleManualSave(data) {
    setUploading(true);
    setError(null);
    try {
      await createManualResume(data);
      await fetchResumes();
      await refreshProfile();
      setShowManual(false);
    } catch (e) {
      setError(e.message);
    } finally {
      setUploading(false);
    }
  }

  async function handleSetPrimary(id) {
    try {
      await setPrimaryResume(id);
      await fetchResumes();
      await refreshProfile();
    } catch (e) {
      setError(e.message);
    }
  }

  async function handleDelete(id) {
    try {
      await deleteResume(id);
      setResumes((prev) => prev.filter((r) => r.id !== id));
      await refreshProfile();
    } catch (e) {
      setError(e.message);
    }
  }

  return (
    <div>
      {error && (
        <div className="card" style={{ padding: '10px 14px', marginBottom: 12, borderLeft: '3px solid var(--red)', color: 'var(--red)', fontSize: '0.82rem' }}>
          {error}
          <button className="btn btn-secondary" style={{ marginLeft: 8, padding: '2px 8px', fontSize: '0.7rem' }} onClick={() => setError(null)}>✕</button>
        </div>
      )}

      {/* Upload Area */}
      {!showManual && (
        <div style={{ marginBottom: 12 }}>
          <ResumeDropzone onFile={handleUpload} disabled={uploading} />
          {uploading && (
            <div className="flex center gap-8" style={{ marginTop: 8, fontSize: '0.82rem', color: 'var(--text2)' }}>
              <span className="spinner" style={{ width: 14, height: 14 }} /> Uploading...
            </div>
          )}
          <button className="btn btn-secondary" onClick={() => setShowManual(true)}
            style={{ marginTop: 8, fontSize: '0.82rem', padding: '8px 16px', width: '100%' }}>
            Or enter details manually
          </button>
        </div>
      )}

      {/* Manual Form */}
      <AnimatePresence>
        {showManual && (
          <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} exit={{ opacity: 0, height: 0 }}>
            <div className="card" style={{ marginBottom: 12 }}>
              <div className="flex" style={{ justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
                <span style={{ fontWeight: 600, fontSize: '0.9rem', color: 'var(--text)' }}>Manual Entry</span>
                <button className="btn btn-secondary" onClick={() => setShowManual(false)}
                  style={{ padding: '4px 12px', fontSize: '0.78rem' }}>Back</button>
              </div>
              <ManualResumeForm onSave={handleManualSave} saving={uploading} />
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Resume List */}
      {resumes.length > 0 && (
        <div style={{ marginTop: 12 }}>
          {!compact && (
            <div style={{ fontWeight: 600, fontSize: '0.85rem', color: 'var(--text)', marginBottom: 8 }}>
              Resumes ({resumes.length})
            </div>
          )}
          {loading ? (
            <div className="flex center" style={{ padding: 20 }}><span className="spinner" /></div>
          ) : (
            resumes.map((r) => (
              <ResumeCard
                key={r.id}
                resume={r}
                onSetPrimary={handleSetPrimary}
                onDelete={handleDelete}
              />
            ))
          )}
        </div>
      )}

      {!loading && resumes.length === 0 && (
        <p style={{ textAlign: 'center', color: 'var(--text2)', fontSize: '0.85rem', padding: 16 }}>
          No resumes yet. Upload a PDF or enter details manually.
        </p>
      )}
    </div>
  );
}
