import numpy as np
import librosa
import random
import math  # for formatting time

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
    # Safety check: if words is None or empty, return empty list
    # THIS MUST COME BEFORE ANY LIST COMPREHENSION
    if words is None or len(words) == 0:
        return []
    
    issues = []
    
    # Now it's safe to create word_texts
    try:
        word_texts = [w["word"].lower() for w in words if isinstance(w, dict) and "word" in w]
    except (TypeError, KeyError, AttributeError):
        return []  # If we can't process words, return empty issues list

    # Precompute pauses (gaps between words)
    pauses = []
    for i in range(len(words)-1):
        try:
            # Ensure both words have required keys
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

    # Constants
    LONG_PAUSE_THRESHOLD = 1.2
    FILLER_WORDS = {"um", "uh", "like", "you know", "actually", "basically", "literally"}
    HEDGE_WORDS = {"maybe", "i think", "probably", "kind of", "sort of", "perhaps", "might"}

    for i, w in enumerate(words):
        # Safety check: ensure word is a dict and has required keys
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

        # 1. Filler words
        if word in FILLER_WORDS:
            issues.append({
                "time": format_time(start),
                "word": w["word"],
                "issue": "filler",
                "suggestion": "Replace with a brief pause or remove it. Filler words reduce clarity."
            })

        # 2. Hedge words
        if word in HEDGE_WORDS:
            issues.append({
                "time": format_time(start),
                "word": w["word"],
                "issue": "hedging",
                "suggestion": "Use more definitive language to sound confident. For example, say 'I will' instead of 'I think I will'."
            })

        # 3. Repetitions (check consecutive words)
        if i > 0 and i < len(word_texts) and word_texts[i] == word_texts[i-1]:
            issues.append({
                "time": format_time(start),
                "word": w["word"],
                "issue": "repetition",
                "suggestion": "You repeated this word. Slow down and avoid echoing the same word."
            })

        # 4. Long pause before this word (if not first word)
        if i > 0 and i-1 < len(pauses) and pauses[i-1] > LONG_PAUSE_THRESHOLD:
            issues.append({
                "time": format_time(start),
                "word": w["word"],
                "issue": "long_pause_before",
                "suggestion": f"A long pause ({pauses[i-1]:.1f}s) occurred before this word. Try to keep your flow by using bridging phrases."
            })

    return issues

# ---------- Extract words with timestamps (FIXED) ----------
def extract_words_with_timestamps(segments):
    """Return list of dicts with 'word', 'start', 'end' for each word."""
    # Safety check
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
    # Safety check
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
    # Safety check
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
    # Safety check
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

# ---------- Modified generate_feedback ----------
def generate_feedback(temporal, fluency, lexical, acoustic, scores, words):
    """
    Generate general feedback (text) and word-level analysis (list of dicts).
    Returns a dict with keys 'general' and 'word_analysis'.
    """
    # Safety check for words
    if words is None:
        words = []
    
    # 1. Word-level analysis
    word_analysis = analyze_word_level(words)

    # 2. General feedback (same as before, but we'll build it into a variable)
    if temporal["total_duration"] == 0:
        general = "No speech detected. Please try again with a longer recording."
        return {"general": general, "word_analysis": word_analysis}

    feedback_parts = []

    # ----- Timing & Pace -----
    wpm = temporal["wpm"]
    pace_templates = {
        "slow": [
            "Your speaking rate of {wpm} words per minute is on the slower side. This can make you sound hesitant or uncertain. Try practicing with a metronome at 120-140 bpm to gradually increase your pace.",
            "At {wpm} WPM, you're speaking below the typical conversational range (120-160 WPM). Consider reading aloud daily to build momentum and reduce those longer pauses between phrases.",
            "Your current pace ({wpm} WPM) suggests you might be thinking too much while speaking. A helpful exercise: record yourself answering common interview questions, then practice delivering them more fluidly without the thinking pauses."
        ],
        "good": [
            "Your speaking rate of {wpm} WPM falls within the optimal range for interviews. This pace is conversational yet professional, giving you credibility while remaining easy to follow.",
            "Great job maintaining {wpm} words per minute! This balanced pace shows you're comfortable with the material without rushing. You're in the sweet spot for keeping listener engagement.",
            "At {wpm} WPM, you've found the ideal tempo - not too fast to seem rehearsed, not too slow to lose attention. This natural flow builds rapport with interviewers."
        ],
        "fast": [
            "You're speaking at {wpm} WPM, which is quite rapid. Fast speech can signal nervousness and make it harder for interviewers to absorb your key points. Practice inserting micro-pauses after important statements.",
            "Your rapid pace ({wpm} WPM) might be overwhelming listeners. Try the 'comma technique' - mentally insert commas between ideas and actually pause there. This gives you breathing room and your audience time to process.",
            "At {wpm} words per minute, you're in danger of sounding rehearsed or anxious. Slow down by 10-15% and notice how much more authoritative you sound. Your key achievements deserve to be heard clearly."
        ]
    }
    if wpm < 100:
        pace_key = "slow"
    elif wpm <= 160:
        pace_key = "good"
    else:
        pace_key = "fast"
    feedback_parts.append(random.choice(pace_templates[pace_key]).format(wpm=wpm))

    # ----- Pause Analysis -----
    if temporal["long_pause_count"] > 0:
        pause_templates = [
            f"I noticed {temporal['long_pause_count']} significant pauses (over 1.2 seconds) in your speech. While occasional pauses are natural, extended silences can disrupt your flow. Try using transitional phrases like 'Let me think about that' or 'That's an excellent question' to buy thinking time naturally.",
            f"Your recording contained {temporal['long_pause_count']} lengthy pauses. This often happens when we're searching for the right word. A powerful technique: memorize 3-5 bridging phrases that you can use instinctively when you need a moment to gather thoughts.",
            f"With {temporal['long_pause_count']} long pauses detected, your speech has some choppy moments. Try the 'continuous speaking' exercise: pick any topic and speak for 2 minutes without stopping - even if you repeat yourself."
        ]
        feedback_parts.append(random.choice(pause_templates))

    # ----- Filler Words -----
    if fluency["filler_ratio"] > 0.03:
        filler_templates = [
            f"Filler words like 'um', 'uh', and 'like' appeared {fluency['filler_count']} times in your speech ({fluency['filler_ratio']*100:.1f}% of total words). These small words can undermine your authority. Try the 'pause instead' technique: whenever you feel 'um' coming, simply pause.",
            f"You used {fluency['filler_count']} filler words in this recording. Each one is a tiny credibility leak. A practical fix: record yourself daily and count your fillers. Awareness alone reduces them by 30-40% within a week.",
            f"Your filler word density is {fluency['filler_ratio']*100:.1f}%. To sound more polished, practice the 'prepared pause' technique. When you need to think, pause deliberately for 1-2 seconds instead of filling with 'um'."
        ]
        feedback_parts.append(random.choice(filler_templates))
    else:
        if random.choice([True, False]):  # Occasionally give positive reinforcement
            good_filler_templates = [
                f"Excellent control over filler words! Only {fluency['filler_count']} in this recording shows real polish. This makes you sound confident and well-prepared.",
                f"Your minimal use of filler words ({fluency['filler_count']}) is impressive. Clean speech like this signals executive presence and clarity of thought.",
                f"Great job keeping filler words to just {fluency['filler_count']} instances. This level of fluency is what top performers achieve with practice."
            ]
            feedback_parts.append(random.choice(good_filler_templates))

    # ----- Repetitions -----
    if fluency["repetition_rate"] > 0.02:
        rep_templates = [
            f"I detected some word repetitions in your speech. This often happens when we're thinking ahead while still speaking. Try slowing down slightly and finishing your current thought completely before moving to the next one.",
            f"Repetitions can make you sound uncertain. A helpful exercise: practice explaining complex topics to a friend and ask them to signal whenever you repeat yourself.",
            f"Your repetition rate suggests you might be circling around ideas instead of stating them directly. Try writing out key points before speaking and sticking to that structure."
        ]
        feedback_parts.append(random.choice(rep_templates))

    # ----- Hedging -----
    if lexical["hedge_ratio"] > 0.04:
        hedge_templates = [
            f"Hedging words like 'maybe', 'I think', and 'probably' appeared frequently. These words subtly undermine your confidence. Practice making definitive statements - instead of 'I think I could...' try 'I can...'",
            f"Your speech contained several tentative phrases. While diplomacy has its place, interviews reward directness. Try reviewing your transcript and rewriting each hedged statement as a confident assertion.",
            f"Hedging language can make you sound less authoritative. The next time you practice, challenge yourself to eliminate all qualifying words. You'll sound noticeably more confident."
        ]
        feedback_parts.append(random.choice(hedge_templates))

    # ----- Vocabulary Richness -----
    if lexical["ttr"] < 0.6 and lexical["unique_words"] > 10:
        ttr_templates = [
            f"Your vocabulary variety (Type-Token Ratio: {lexical['ttr']:.2f}) suggests some word repetition. Expanding your active vocabulary for interview topics can make you sound more articulate. Try learning 3 new synonyms each week for common interview words.",
            f"With a TTR of {lexical['ttr']:.2f}, you're using a limited vocabulary set. This is normal in conversation, but interviews reward precision. Practice describing your experience using varied, specific language.",
            f"Your word choice shows some repetition patterns. To sound more sophisticated, prepare 2-3 ways to describe each of your key accomplishments. Variety signals mastery."
        ]
        feedback_parts.append(random.choice(ttr_templates))

    # ----- Score-Based Feedback -----
    if scores["fluency"] < 0.6:
        fluency_score_templates = [
            f"Your fluency score of {scores['fluency']:.2f} indicates room for improvement in smoothness. Focus on reducing pauses and filler words through daily practice.",
            f"At {scores['fluency']:.2f}, your fluency is the area with most growth potential. Try the 'continuous speech' exercise: speak for 90 seconds without stopping on any topic.",
            f"To boost your fluency score from {scores['fluency']:.2f}, practice linking your ideas with transition phrases like 'building on that point' or 'another aspect to consider'."
        ]
        feedback_parts.append(random.choice(fluency_score_templates))

    if scores["confidence"] < 0.6:
        confidence_score_templates = [
            f"Your confidence score of {scores['confidence']:.2f} suggests vocal uncertainty. Work on eliminating hedging words and stabilizing your pitch through breathing exercises.",
            f"To improve your confidence score ({scores['confidence']:.2f}), practice speaking slightly louder and with more varied intonation. Record yourself and listen for tentative phrases.",
            f"The confidence score ({scores['confidence']:.2f}) reflects both word choice and vocal stability. Try power poses before speaking and practice stating your achievements without qualifiers."
        ]
        feedback_parts.append(random.choice(confidence_score_templates))

    if scores["composure"] < 0.6:
        composure_score_templates = [
            f"Your composure score of {scores['composure']:.2f} indicates some vocal tension. Deep breathing before speaking can help. Also try to consciously relax your jaw and shoulders while talking.",
            f"To improve composure ({scores['composure']:.2f}), practice speaking in a slightly lower pitch range. Lower voices typically sound more composed and authoritative.",
            f"The composure score suggests some nervous energy in your voice. Try the 'pause and breathe' technique: before answering any question, take a deliberate breath."
        ]
        feedback_parts.append(random.choice(composure_score_templates))

    # ----- Overall Summary -----
    if scores["overall"] > 0.8:
        summary_templates = [
            f"\n\nOverall, you delivered a strong performance with a score of {scores['overall']:.2f}. Your speech is clear, confident, and well-paced. The suggestions above will help you move from good to exceptional. Keep practicing daily - you're on the right track!",
            f"\n\nGreat work! Your overall score of {scores['overall']:.2f} shows you have strong interview communication skills. Focus on the specific areas mentioned to polish your delivery even further. Consistency is key - record yourself weekly to track improvement.",
            f"\n\nWith an overall score of {scores['overall']:.2f}, you're already performing well above average. The refinements suggested above will help you achieve that elite level of communication that distinguishes top candidates."
        ]
    elif scores["overall"] > 0.6:
        summary_templates = [
            f"\n\nYour overall score of {scores['overall']:.2f} shows solid foundational skills with room to grow. The areas highlighted above are your quickest path to improvement. With focused practice on these specific points, you could see significant gains in just 2-3 weeks.",
            f"\n\nAt {scores['overall']:.2f}, you have good interview mechanics but some patterns are holding you back. The feedback above targets your biggest opportunities. Pick just ONE area to focus on this week - mastering it will pull other metrics up naturally.",
            f"\n\nYour overall score of {scores['overall']:.2f} places you in the developing range. The good news: the metrics above show exactly where to focus. Start with reducing filler words - it's often the highest-impact change you can make quickly."
        ]
    else:
        summary_templates = [
            f"\n\nYour overall score of {scores['overall']:.2f} suggests you're still building your interview communication skills. This is completely normal and improvable with structured practice. Don't be discouraged - every great communicator started exactly where you are. Focus on one metric at a time and record yourself weekly to see progress.",
            f"\n\nWith an overall score of {scores['overall']:.2f}, you have clear opportunities for growth. The analysis above isn't criticism - it's a roadmap. Professional speakers and executives all worked on these exact skills. Your journey to mastery starts with awareness, and now you have it.",
            f"\n\nYour score of {scores['overall']:.2f} reflects where many people start before focused practice. The beauty of this system is that every metric is improvable. Choose the lowest score among fluency, confidence, or composure and make that your priority for the next two weeks."
        ]
    feedback_parts.append(random.choice(summary_templates))

    # Combine general feedback
    general = " ".join(feedback_parts)

    # Word count enforcement (optional, but we keep it for general feedback)
    word_count = len(general.split())
    if word_count < 30:
        filler_templates = [
            " Remember that consistent practice is key to improvement. Try recording yourself daily, even for just 2 minutes, and track your progress on these metrics over time.",
            " The best speakers weren't born great - they practiced deliberately. Focus on one area at a time, and you'll see steady improvement in your scores.",
            " Consider joining a speaking group or practicing with a friend who can give you real-time feedback on these specific areas.",
            " Recording yourself is the first step. Now that you have this baseline data, you can track improvement over the coming weeks.",
            " Every point of improvement in these scores translates to more confident, compelling interviews. Keep up the great work!"
        ]
        general += random.choice(filler_templates)
        if len(general.split()) < 30:
            general += " " + random.choice([
                "Small daily improvements lead to stunning results over time.",
                "The key is consistency - practice a little every day rather than for hours once a week.",
                "You're building a skill that will serve you throughout your entire career.",
                "Each recording makes you more aware of your speech patterns, and awareness is the first step to improvement."
            ])

    return {"general": general, "word_analysis": word_analysis}