# audio_service.py
import asyncio
import os
from datetime import datetime
from .analysis_service import (
    extract_words_with_timestamps,
    compute_temporal_features,
    compute_fluency_features,
    compute_lexical_features,
    compute_acoustic_features,
    compute_scores,
    generate_feedback   # now returns dict
)

UPLOAD_DIR = "uploads"

async def save_and_transcribe(file, model):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    filename = f"{datetime.now().timestamp()}_{file.filename}"
    filepath = os.path.join(UPLOAD_DIR, filename)

    contents = await file.read()
    with open(filepath, "wb") as f:
        f.write(contents)

    # Transcribe with word timestamps
    result = await asyncio.to_thread(
        model.transcribe,
        filepath,
        word_timestamps=True
    )

    transcript = result["text"]
    segments = result["segments"]
    words = extract_words_with_timestamps(segments)

    # Compute all feature groups
    temporal = compute_temporal_features(words)
    fluency = compute_fluency_features(words)
    lexical = compute_lexical_features(words)
    acoustic = compute_acoustic_features(filepath)
    scores = compute_scores(temporal, fluency, lexical, acoustic)

    # Generate feedback (now returns dict with 'general' and 'word_analysis')
    feedback_result = generate_feedback(temporal, fluency, lexical, acoustic, scores, words)

    return {
        "filepath": filepath,
        "transcript": transcript,
        "temporal": temporal,
        "fluency": fluency,
        "lexical": lexical,
        "acoustic": acoustic,
        "scores": scores,
        "feedback": feedback_result["general"],          # for backward compatibility
        "word_analysis": feedback_result["word_analysis"]  # new field
    }