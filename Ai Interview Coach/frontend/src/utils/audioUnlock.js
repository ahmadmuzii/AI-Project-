let audioCtx = null;

export function unlockAudio() {
  try {
    if (!audioCtx) {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (audioCtx.state === 'suspended') {
      audioCtx.resume();
    }
    const buf = audioCtx.createBuffer(1, 1, 22050);
    const src = audioCtx.createBufferSource();
    src.buffer = buf;
    src.connect(audioCtx.destination);
    src.start(0);
    return true;
  } catch {
    return false;
  }
}

/**
 * Unlock HTML5 Audio element autoplay.
 * ElevenLabs uses `new Audio()` which has a SEPARATE autoplay policy
 * from the Web Audio API (AudioContext). Must be called during a user gesture.
 */
export function unlockHtmlAudio() {
  try {
    // Create a minimal silent audio (1-frame WAV) as a data URI
    // This primes the browser's HTML5 audio autoplay permission
    const silentWav = 'data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA=';
    const audio = new Audio(silentWav);
    audio.volume = 0;
    const p = audio.play();
    if (p !== undefined) {
      p.catch(() => {
        // Autoplay still blocked — user hasn't interacted enough yet
        // This is non-fatal; ElevenLabs will try again on next speak call
      });
    }
    return true;
  } catch {
    return false;
  }
}

export function primeSpeechSynthesis() {
  if (!window.speechSynthesis) return false;
  try {
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance('');
    u.volume = 0;
    window.speechSynthesis.speak(u);
    window.speechSynthesis.cancel();
    return true;
  } catch {
    return false;
  }
}

export function primeAudio() {
  unlockAudio();
  unlockHtmlAudio(); // unlock HTML5 Audio (used by ElevenLabs) separately from AudioContext
  primeSpeechSynthesis();
}

