import { useState, useRef } from 'react';

export default function ResumeDropzone({ onFile, accept = '.pdf', disabled = false }) {
  const [dragOver, setDragOver] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const inputRef = useRef(null);

  function handleDrop(e) {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer?.files?.[0];
    if (f && f.type === 'application/pdf') {
      setSelectedFile(f);
      onFile?.(f);
    }
  }

  function handleInput(e) {
    const f = e.target.files?.[0];
    if (f) {
      setSelectedFile(f);
      onFile?.(f);
    }
  }

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
      onClick={() => !disabled && inputRef.current?.click()}
      style={{
        border: `2px dashed ${dragOver ? 'var(--blue)' : selectedFile ? 'var(--green)' : 'var(--border)'}`,
        borderRadius: 12, padding: '28px 20px', textAlign: 'center', cursor: disabled ? 'default' : 'pointer',
        background: dragOver ? 'rgba(26,115,232,0.05)' : selectedFile ? 'rgba(52,168,83,0.05)' : 'var(--bg2)',
        transition: 'all 0.2s',
      }}
    >
      <input ref={inputRef} type="file" accept={accept} style={{ display: 'none' }} onChange={handleInput} disabled={disabled} />
      {selectedFile ? (
        <div>
          <div style={{ fontSize: '1.8rem', marginBottom: 6 }}>📄</div>
          <div style={{ fontWeight: 600, fontSize: '0.9rem', color: 'var(--green)' }}>{selectedFile.name}</div>
          <div style={{ fontSize: '0.78rem', color: 'var(--text2)', marginTop: 2 }}>
            {(selectedFile.size / 1024).toFixed(1)} KB — click to change
          </div>
        </div>
      ) : (
        <div>
          <div style={{ fontSize: '2rem', marginBottom: 8, opacity: 0.6 }}>📎</div>
          <div style={{ fontWeight: 500, fontSize: '0.9rem', color: 'var(--text)' }}>Drop your resume PDF here</div>
          <div style={{ fontSize: '0.8rem', color: 'var(--text2)', marginTop: 4 }}>or click to browse</div>
        </div>
      )}
    </div>
  );
}
