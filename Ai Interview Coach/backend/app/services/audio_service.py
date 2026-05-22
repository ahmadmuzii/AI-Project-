# audio_service.py
import asyncio
import os
import uuid
from datetime import datetime
from pathlib import PurePath
from .analysis_service import (
    extract_words_with_timestamps,
    compute_temporal_features,
    compute_fluency_features,
    compute_lexical_features,
    compute_acoustic_features,
    compute_scores,
    generate_feedback
)

UPLOAD_DIR = "uploads"
ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".webm", ".ogg", ".flac", ".aac"}

async def save_and_transcribe(file, model):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    ext = PurePath(file.filename or "audio.wav").suffix.lower()
    if ext not in ALLOWED_AUDIO_EXTENSIONS:
        ext = ".wav"
    safe_name = f"{datetime.now().timestamp()}_{uuid.uuid4().hex[:12]}{ext}"
    filepath = os.path.join(UPLOAD_DIR, safe_name)

    contents = await file.read()
    with open(filepath, "wb") as f:
        f.write(contents)

    result = await asyncio.to_thread(
        model.transcribe,
        filepath,
        word_timestamps=True
    )

    transcript = result["text"]
    segments = result["segments"]
    words = extract_words_with_timestamps(segments)

    temporal = compute_temporal_features(words)
    fluency = compute_fluency_features(words)
    lexical = compute_lexical_features(words)
    acoustic = compute_acoustic_features(filepath)
    scores = compute_scores(temporal, fluency, lexical, acoustic)

    feedback_result = generate_feedback(temporal, fluency, lexical, acoustic, scores, words, transcript)

    return {
        "filepath": filepath,
        "transcript": transcript,
        "temporal": temporal,
        "fluency": fluency,
        "lexical": lexical,
        "acoustic": acoustic,
        "scores": scores,
        "feedback": feedback_result["general"],
        "word_analysis": feedback_result["word_analysis"]
    }
