import numpy as np
import librosa
import random
import math  # for formatting time
import os

# ---------- Groq LLM Feedback ----------
def generate_llm_feedback(temporal, fluency, lexical, acoustic, scores, transcript):
    try:
        from groq import Groq

        api_key = os.environ.get("GROQ_API_KEY")
        print(f"GROQ_API_KEY found: {bool(api_key)}")
        print(f"Key starts with: {api_key[:8] if api_key else 'NONE'}")
        if not api_key:
            return None

        client = Groq(api_key=api_key)

        prompt = f"""You are an expert interview coach. Analyze the following speech metrics from a candidate's interview response and provide personalized, actionable coaching feedback.

TRANSCRIPT:
{transcript}

SPEECH METRICS:
- Speaking Rate: {temporal['wpm']} words per minute (ideal: 120-160 WPM)
- Total Duration: {temporal['total_duration']} seconds
- Long Pauses: {temporal['long_pause_count']} pauses over 1.2 seconds
- Average Pause: {temporal['avg_pause']} seconds
- Filler Words (um, uh, like, etc.): {fluency['filler_count']} times ({fluency['filler_ratio']*100:.1f}% of words)
- Word Repetitions: {fluency['repetition_count']}
- Hedging Words (maybe, I think, probably): {lexical['hedge_count']}
- Vocabulary Variety (TTR): {lexical['ttr']:.2f} (1.0 = all unique words)
- Pitch Variation: {acoustic['pitch_std']:.2f} Hz
- Voice Energy: {acoustic['energy_mean']:.4f}

SCORES (0 to 1):
- Fluency: {scores['fluency']}
- Confidence: {scores['confidence']}
- Composure: {scores['composure']}
- Overall: {scores['overall']}

Please provide:
1. A brief overall assessment (1-2 sentences)
2. The top 2-3 specific areas to improve with concrete, practical tips
3. One thing they did well (if applicable)
4. An encouraging closing sentence

Keep the feedback conversational, specific to their actual metrics, and between 150-250 words. Do not use bullet points or headers — write it as natural flowing paragraphs like a real coach would speak."""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.7,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Groq feedback failed: {e}")
        return None  # Fall back to rule-based


# ---------- Helper: format time (seconds -> MM:SS) ----------
def format_time(seconds):
    """Convert seconds to MM:SS format."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}:{secs:02d}"

# ---------- Word-Level Analysis ----------
def analyze_word_level(words):
    """
    Analyze each word and return a list of issues (dictionaries) for words
    that need improvement.
    """
    if words is None or len(words) == 0:
        return []
    
    issues = []
    
    try:
        word_texts = [w["word"].lower() for w in words if isinstance(w, dict) and "word" in w]
    except (TypeError, KeyError, AttributeError):
        return []

    pauses = []
    for i in range(len(words)-1):
        try:
            if not isinstance(words[i], dict) or not isinstance(words[i+1], dict):
                pauses.append(0)
                continue
            if "end" not in words[i] or "start" not in words[i+1]:
                pauses.append(0)
                continue
            pause = words[i+1]["start"] - words[i]["end"]
            pauses.append(max(0, pause))
        except (TypeError, KeyError):
            pauses.append(0)

    LONG_PAUSE_THRESHOLD = 1.2
    FILLER_WORDS = {"um", "uh", "like", "you know", "actually", "basically", "literally"}
    HEDGE_WORDS = {"maybe", "i think", "probably", "kind of", "sort of", "perhaps", "might"}

    for i, w in enumerate(words):
        if not isinstance(w, dict):
            continue
        if "word" not in w or "start" not in w or "end" not in w:
            continue
            
        try:
            word = w["word"].lower()
            start = w["start"]
            end = w["end"]
        except (AttributeError, KeyError):
            continue

        if word in FILLER_WORDS:
            issues.append({
                "time": format_time(start),
                "word": w["word"],
                "issue": "filler",
                "suggestion": "Replace with a brief pause or remove it. Filler words reduce clarity."
            })

        if word in HEDGE_WORDS:
            issues.append({
                "time": format_time(start),
                "word": w["word"],
                "issue": "hedging",
                "suggestion": "Use more definitive language to sound confident. For example, say 'I will' instead of 'I think I will'."
            })

        if i > 0 and i < len(word_texts) and word_texts[i] == word_texts[i-1]:
            issues.append({
                "time": format_time(start),
                "word": w["word"],
                "issue": "repetition",
                "suggestion": "You repeated this word. Slow down and avoid echoing the same word."
            })

        if i > 0 and i-1 < len(pauses) and pauses[i-1] > LONG_PAUSE_THRESHOLD:
            issues.append({
                "time": format_time(start),
                "word": w["word"],
                "issue": "long_pause_before",
                "suggestion": f"A long pause ({pauses[i-1]:.1f}s) occurred before this word. Try to keep your flow by using bridging phrases."
            })

    return issues

# ---------- Extract words with timestamps ----------
def extract_words_with_timestamps(segments):
    """Return list of dicts with 'word', 'start', 'end' for each word."""
    if segments is None:
        return []
        
    words = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        for w in seg.get("words", []):
            if not isinstance(w, dict):
                continue
            try:
                words.append({
                    "word": str(w.get("word", "")).strip(),
                    "start": float(w.get("start", 0)),
                    "end": float(w.get("end", 0))
                })
            except (ValueError, TypeError):
                continue
    return words

def compute_temporal_features(words):
    if not words or len(words) < 2:
        return {
            "total_duration": 0,
            "wpm": 0,
            "avg_pause": 0,
            "long_pause_count": 0,
            "pause_rate_per_min": 0
        }

    try:
        first_start = words[0]["start"]
        last_end = words[-1]["end"]
        total_duration = last_end - first_start

        num_words = len(words)
        wpm = (num_words / total_duration) * 60 if total_duration > 0 else 0

        pauses = []
        for i in range(len(words)-1):
            try:
                pause = words[i+1]["start"] - words[i]["end"]
                pauses.append(max(0, pause))
            except (KeyError, TypeError):
                pauses.append(0)

        avg_pause = np.mean(pauses) if pauses else 0
        long_pause_threshold = 1.2
        long_pause_count = sum(1 for p in pauses if p > long_pause_threshold)
        pause_rate = (long_pause_count / total_duration) * 60 if total_duration > 0 else 0

        return {
            "total_duration": round(total_duration, 2),
            "wpm": round(wpm, 2),
            "avg_pause": round(avg_pause, 2),
            "long_pause_count": long_pause_count,
            "pause_rate_per_min": round(pause_rate, 2)
        }
    except (KeyError, TypeError, ZeroDivisionError):
        return {
            "total_duration": 0,
            "wpm": 0,
            "avg_pause": 0,
            "long_pause_count": 0,
            "pause_rate_per_min": 0
        }

FILLER_WORDS = {"um", "uh", "like", "you know", "actually", "basically", "literally"}

def compute_fluency_features(words):
    if not words:
        return {"repetition_count": 0, "repetition_rate": 0, "filler_count": 0, "filler_ratio": 0}
    
    try:
        word_texts = [w["word"].lower() for w in words if isinstance(w, dict) and "word" in w]
        total_words = len(word_texts)

        if total_words == 0:
            return {"repetition_count": 0, "repetition_rate": 0, "filler_count": 0, "filler_ratio": 0}

        repetition_count = 0
        for i in range(len(word_texts)-1):
            if word_texts[i] == word_texts[i+1]:
                repetition_count += 1

        filler_count = sum(1 for w in word_texts if w in FILLER_WORDS)

        return {
            "repetition_count": repetition_count,
            "repetition_rate": round(repetition_count / total_words, 4),
            "filler_count": filler_count,
            "filler_ratio": round(filler_count / total_words, 4)
        }
    except (KeyError, TypeError, ZeroDivisionError):
        return {"repetition_count": 0, "repetition_rate": 0, "filler_count": 0, "filler_ratio": 0}

HEDGE_WORDS = {"maybe", "i think", "probably", "kind of", "sort of", "perhaps", "might"}

def compute_lexical_features(words):
    if not words:
        return {"hedge_count": 0, "hedge_ratio": 0, "unique_words": 0, "ttr": 0}
    
    try:
        word_texts = [w["word"].lower() for w in words if isinstance(w, dict) and "word" in w]
        total_words = len(word_texts)

        if total_words == 0:
            return {"hedge_count": 0, "hedge_ratio": 0, "unique_words": 0, "ttr": 0}

        hedge_count = sum(1 for w in word_texts if w in HEDGE_WORDS)
        unique_words = len(set(word_texts))
        ttr = unique_words / total_words

        return {
            "hedge_count": hedge_count,
            "hedge_ratio": round(hedge_count / total_words, 4),
            "unique_words": unique_words,
            "ttr": round(ttr, 4)
        }
    except (KeyError, TypeError, ZeroDivisionError):
        return {"hedge_count": 0, "hedge_ratio": 0, "unique_words": 0, "ttr": 0}

def compute_acoustic_features(filepath):
    try:
        y, sr = librosa.load(filepath, sr=16000)

        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=float(librosa.note_to_hz('C2')),
            fmax=float(librosa.note_to_hz('C7'))
        )
        f0 = f0[~np.isnan(f0)]

        pitch_mean = float(np.mean(f0)) if len(f0) > 0 else 0
        pitch_std = float(np.std(f0)) if len(f0) > 0 else 0

        rms = librosa.feature.rms(y=y)[0]
        energy_mean = float(np.mean(rms))
        energy_var = float(np.var(rms))

        if len(f0) > 1:
            diff_f0 = np.abs(np.diff(f0))
            jitter = float(np.mean(diff_f0) / pitch_mean) if pitch_mean > 0 else 0
        else:
            jitter = 0

        if len(y) > 1:
            amp = np.abs(y)
            diff_amp = np.abs(np.diff(amp))
            shimmer = float(np.mean(diff_amp) / np.mean(amp)) if np.mean(amp) > 0 else 0
        else:
            shimmer = 0

        return {
            "pitch_mean": round(pitch_mean, 2),
            "pitch_std": round(pitch_std, 2),
            "energy_mean": round(energy_mean, 4),
            "energy_var": round(energy_var, 4),
            "jitter": round(jitter, 4),
            "shimmer": round(shimmer, 4)
        }
    except Exception as e:
        print(f"Acoustic analysis failed: {e}")
        return {
            "pitch_mean": 0,
            "pitch_std": 0,
            "energy_mean": 0,
            "energy_var": 0,
            "jitter": 0,
            "shimmer": 0
        }

def compute_scores(temporal, fluency, lexical, acoustic):
    try:
        pause_rate = temporal["pause_rate_per_min"] / 20
        rep_rate = fluency["repetition_rate"] * 5
        filler_ratio = fluency["filler_ratio"]
        fluency_score = 1 - (0.3 * pause_rate + 0.3 * rep_rate + 0.2 * filler_ratio + 0.2 * (temporal["avg_pause"]/3))
        fluency_score = max(0, min(1, fluency_score))

        hedge_ratio = lexical["hedge_ratio"]
        pitch_var_norm = acoustic["pitch_std"] / 100
        jitter_norm = acoustic["jitter"] * 10
        filler_ratio = fluency["filler_ratio"]
        confidence = 1 - (0.3 * hedge_ratio + 0.2 * pitch_var_norm + 0.2 * jitter_norm + 0.3 * filler_ratio)
        confidence = max(0, min(1, confidence))

        energy_var_norm = acoustic["energy_var"] * 100
        long_pause_rate = temporal["pause_rate_per_min"] / 20
        pitch_std_norm = acoustic["pitch_std"] / 100
        composure = 1 - (0.4 * energy_var_norm + 0.3 * long_pause_rate + 0.3 * pitch_std_norm)
        composure = max(0, min(1, composure))

        overall = 0.4 * fluency_score + 0.35 * confidence + 0.25 * composure

        return {
            "fluency": round(fluency_score, 3),
            "confidence": round(confidence, 3),
            "composure": round(composure, 3),
            "overall": round(overall, 3)
        }
    except (KeyError, TypeError, ZeroDivisionError):
        return {
            "fluency": 0,
            "confidence": 0,
            "composure": 0,
            "overall": 0
        }

# ---------- Rule-Based Feedback (Fallback) ----------
def generate_rule_based_feedback(temporal, fluency, lexical, acoustic, scores):
    feedback_parts = []

    if temporal["total_duration"] == 0:
        return "No speech detected. Please try again with a longer recording."

    wpm = temporal["wpm"]
    pace_templates = {
        "slow": [
            "Your speaking rate of {wpm} words per minute is on the slower side. This can make you sound hesitant or uncertain. Try practicing with a metronome at 120-140 bpm to gradually increase your pace.",
            "At {wpm} WPM, you're speaking below the typical conversational range (120-160 WPM). Consider reading aloud daily to build momentum and reduce those longer pauses between phrases.",
        ],
        "good": [
            "Your speaking rate of {wpm} WPM falls within the optimal range for interviews. This pace is conversational yet professional, giving you credibility while remaining easy to follow.",
            "Great job maintaining {wpm} words per minute! This balanced pace shows you're comfortable with the material without rushing.",
        ],
        "fast": [
            "You're speaking at {wpm} WPM, which is quite rapid. Fast speech can signal nervousness and make it harder for interviewers to absorb your key points. Practice inserting micro-pauses after important statements.",
            "Your rapid pace ({wpm} WPM) might be overwhelming listeners. Try the 'comma technique' - mentally insert commas between ideas and actually pause there.",
        ]
    }
    if wpm < 100:
        pace_key = "slow"
    elif wpm <= 160:
        pace_key = "good"
    else:
        pace_key = "fast"
    feedback_parts.append(random.choice(pace_templates[pace_key]).format(wpm=wpm))

    if temporal["long_pause_count"] > 0:
        feedback_parts.append(
            f"I noticed {temporal['long_pause_count']} significant pauses (over 1.2 seconds) in your speech. "
            f"While occasional pauses are natural, extended silences can disrupt your flow. "
            f"Try using transitional phrases like 'Let me think about that' to buy thinking time naturally."
        )

    if fluency["filler_ratio"] > 0.03:
        feedback_parts.append(
            f"Filler words appeared {fluency['filler_count']} times ({fluency['filler_ratio']*100:.1f}% of total words). "
            f"Try the 'pause instead' technique: whenever you feel 'um' coming, simply pause silently."
        )
    else:
        feedback_parts.append(
            f"Great job keeping filler words to just {fluency['filler_count']} instances. "
            f"This level of fluency is what top performers achieve with practice."
        )

    if lexical["hedge_ratio"] > 0.04:
        feedback_parts.append(
            "Hedging words like 'maybe', 'I think', and 'probably' appeared frequently. "
            "Practice making definitive statements - instead of 'I think I could...' try 'I can...'"
        )

    if scores["overall"] > 0.8:
        feedback_parts.append(f"\n\nOverall, you delivered a strong performance with a score of {scores['overall']:.2f}. Keep practicing daily - you're on the right track!")
    elif scores["overall"] > 0.6:
        feedback_parts.append(f"\n\nYour overall score of {scores['overall']:.2f} shows solid foundational skills with room to grow. With focused practice, you could see significant gains in just 2-3 weeks.")
    else:
        feedback_parts.append(f"\n\nYour overall score of {scores['overall']:.2f} suggests you're still building your interview communication skills. Focus on one metric at a time and record yourself weekly to see progress.")

    return " ".join(feedback_parts)


# ---------- Main generate_feedback ----------
def generate_feedback(temporal, fluency, lexical, acoustic, scores, words, transcript=""):
    """
    Generate general feedback (text) and word-level analysis (list of dicts).
    Tries Groq LLM first, falls back to rule-based if unavailable.
    Returns a dict with keys 'general' and 'word_analysis'.
    """
    if words is None:
        words = []

    # 1. Word-level analysis (always rule-based)
    word_analysis = analyze_word_level(words)

    # 2. Try LLM feedback first
    if temporal["total_duration"] == 0:
        general = "No speech detected. Please try again with a longer recording."
        return {"general": general, "word_analysis": word_analysis}

    llm_feedback = generate_llm_feedback(temporal, fluency, lexical, acoustic, scores, transcript)

    if llm_feedback:
        general = llm_feedback
        print("✅ Using Groq LLM feedback")
    else:
        general = generate_rule_based_feedback(temporal, fluency, lexical, acoustic, scores)
        print("⚠️  Using rule-based feedback (Groq unavailable)")

    return {"general": general, "word_analysis": word_analysis}
