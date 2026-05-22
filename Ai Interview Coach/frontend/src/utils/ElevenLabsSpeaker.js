import { getElevenLabsTtsUrl } from '../api/client';

const DEFAULT_PAUSE_MS = 300;
const QUESTION_PAUSE_MS = 800;
const LIST_PAUSE_MS = 600;

function getToken() {
  try {
    const stored = localStorage.getItem('aic_auth');
    if (stored) return JSON.parse(stored).access_token;
  } catch { }
  return null;
}

export class ElevenLabsSpeaker {
  constructor(options = {}) {
    this.voiceId = options.voiceId || '21m00Tcm4TlvDq8ikWAM';
    this.onStart = options.onStart || null;
    this.onEnd = options.onEnd || null;
    this.onError = options.onError || null;
    this.playbackRate = options.playbackRate || 1.15;

    this._speaking = false;
    this._cancelled = false;
    this._audioRef = null;
    this._objectUrls = [];
  }

  setPlaybackRate(rate) {
    if (rate > 0) this.playbackRate = rate;
  }

  speak(text) {
    if (!text || typeof text !== 'string') return;

    this._cancelled = false;
    this._speaking = true;
    if (this.onStart) this.onStart();

    const clauses = this._chunk(text);
    this._speakClauses(clauses, 0);
  }

  isSpeaking() {
    return this._speaking;
  }

  stop() {
    this._cancelled = true;
    this._speaking = false;
    if (this._audioRef) {
      this._audioRef.pause();
      this._audioRef = null;
    }
    this._revokeAll();
    if (this.onEnd) this.onEnd();
  }

  setVoice(voiceId) {
    if (voiceId) this.voiceId = voiceId;
  }

  _chunk(text) {
    const cleaned = text.replace(/\s+/g, ' ').trim();
    if (!cleaned) return [];

    const raw = cleaned.split(/(?<=[.!?])\s+|(?<=[:;])\s+/);
    const chunks = [];

    for (const segment of raw) {
      const trimmed = segment.trim();
      if (!trimmed) continue;

      const isQuestion = trimmed.endsWith('?');
      const isListItem = /^[\-\*\d+\.\s]/.test(trimmed);

      chunks.push({
        text: trimmed,
        pauseAfter: isQuestion ? QUESTION_PAUSE_MS : isListItem ? LIST_PAUSE_MS : DEFAULT_PAUSE_MS,
      });
    }

    return chunks;
  }

  _speakClauses(clauses, idx) {
    if (this._cancelled || idx >= clauses.length) {
      this._speaking = false;
      if (this.onEnd) this.onEnd();
      return;
    }

    this._speakSingle(clauses[idx].text, () => {
      if (this._cancelled) {
        this._speaking = false;
        this._revokeAll();
        if (this.onEnd) this.onEnd();
        return;
      }
      setTimeout(() => {
        if (!this._cancelled) {
          this._speakClauses(clauses, idx + 1);
        }
      }, clauses[idx].pauseAfter);
    });
  }

  async _speakElevenLabs(text) {
    const url = getElevenLabsTtsUrl(text, this.voiceId);
    const token = getToken();

    if (!token) {
      throw new Error('ElevenLabs TTS: No auth token found — user may need to log in again');
    }

    const res = await fetch(url, {
      headers: { Authorization: `Bearer ${token}` },
    });

    if (!res.ok) {
      const body = await res.text().catch(() => '');
      throw new Error(`TTS request failed (${res.status}): ${body}`);
    }

    const blob = await res.blob();
    if (!blob || blob.size === 0) {
      throw new Error('ElevenLabs returned empty audio blob');
    }

    const objectUrl = URL.createObjectURL(blob);
    this._objectUrls.push(objectUrl);

    const audio = new Audio(objectUrl);
    audio.playbackRate = this.playbackRate;
    this._audioRef = audio;

    return new Promise((resolve, reject) => {
      audio.onended = () => {
        this._audioRef = null;
        const idx = this._objectUrls.indexOf(objectUrl);
        if (idx > -1) this._objectUrls.splice(idx, 1);
        URL.revokeObjectURL(objectUrl);
        resolve();
      };
      audio.onerror = (e) => {
        this._audioRef = null;
        const idx = this._objectUrls.indexOf(objectUrl);
        if (idx > -1) this._objectUrls.splice(idx, 1);
        URL.revokeObjectURL(objectUrl);
        reject(new Error(`Audio element error: ${e.message || 'unknown'}`));
      };
      const playPromise = audio.play();
      if (playPromise !== undefined) {
        playPromise.catch((err) => {
          // NotAllowedError = browser autoplay policy blocked playback
          if (err.name === 'NotAllowedError') {
            console.warn('[ElevenLabs] Autoplay blocked by browser. A user gesture is required before audio can play.');
            reject(new Error('Autoplay blocked — interact with the page first'));
          } else {
            console.warn('[ElevenLabs] audio.play() failed:', err);
            reject(err);
          }
        });
      }
    });
  }

  _speakBrowserTTS(text) {
    return new Promise((resolve, reject) => {
      if (!window.speechSynthesis) {
        reject(new Error('Browser speechSynthesis not available'));
        return;
      }
      window.speechSynthesis.cancel();
      const u = new SpeechSynthesisUtterance(text);
      u.rate = 1.0;
      u.onend = resolve;
      u.onerror = (e) => reject(new Error(`SpeechSynthesis error: ${e.error || 'unknown'}`));
      window.speechSynthesis.speak(u);
    });
  }

  async _speakSingle(text, onDone) {
    if (!text) {
      onDone();
      return;
    }

    let lastError = null;

    try {
      await this._speakElevenLabs(text);
      onDone();
      return;
    } catch (err) {
      console.warn('[ElevenLabs] ElevenLabs TTS failed, falling back to browser TTS:', err.message);
      lastError = err;
    }

    // Fallback: browser speech synthesis
    try {
      await this._speakBrowserTTS(text);
      onDone();
      return;
    } catch (err) {
      console.warn('[ElevenLabs] Browser TTS also failed:', err.message);
      lastError = err;
    }

    // Both failed — report the error
    console.error('[ElevenLabs] All TTS methods failed:', lastError?.message);
    if (this.onError) this.onError(lastError || new Error('All TTS methods failed'));
    onDone();
  }

  _revokeAll() {
    for (const url of this._objectUrls) URL.revokeObjectURL(url);
    this._objectUrls = [];
  }
}

export default ElevenLabsSpeaker;
