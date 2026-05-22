import { useState } from 'react';

export default function AudioPlayer({ src, label = 'Play recording' }) {
  const [playing, setPlaying] = useState(false);

  function handlePlay() {
    setPlaying(true);
  }

  function handleEnded() {
    setPlaying(false);
  }

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
      <audio
        src={src}
        controls
        preload="none"
        style={{ height: 36, borderRadius: 8, maxWidth: 260 }}
        onPlay={handlePlay}
        onEnded={handleEnded}
      />
    </div>
  );
}
