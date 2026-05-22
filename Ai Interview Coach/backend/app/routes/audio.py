import logging
import traceback
from pathlib import PurePath

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import InterviewSession, Recording, RecordingMetric, User, WordAnalysis
from app.routes.auth import get_current_user
from app.services.audio_service import ALLOWED_AUDIO_EXTENSIONS, save_and_transcribe
from app.services.intelligence_service import analyze_answer_nlp, score_answer_content_llm, analyze_resume_text_llm

router = APIRouter()
log = logging.getLogger("uvicorn.error")


@router.get("/upload-audio")
def upload_audio_get():
    """
    Browsers (or devtools) often probe this URL with GET and then log 405.
    This route explains that only POST is valid for uploads.
    """
    return JSONResponse(
        status_code=200,
        content={
            "message": "Audio upload accepts POST only.",
            "method": "POST",
            "content_type": "multipart/form-data",
            "form_fields": ["session_id", "role", "topic", "question", "resume_text", "file"],
        },
    )


@router.post("/upload-audio")
async def upload_audio(
    request: Request,
    session_id: int = Form(...),
    role: str = Form("general"),
    topic: str = Form("general"),
    question: str = Form(""),
    resume_text: str = Form(""),
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ext = PurePath(file.filename or "audio.wav").suffix.lower()
    if ext not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Invalid audio format: {ext}. Allowed: {', '.join(sorted(ALLOWED_AUDIO_EXTENSIONS))}")
    if file.size and file.size > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 50 MB.")

    sess = (
        db.query(InterviewSession)
        .filter(
            InterviewSession.id == session_id,
            InterviewSession.user_id == current_user.id,
        )
        .first()
    )
    if not sess:
        raise HTTPException(status_code=400, detail="Invalid session")

    model = request.app.state.whisper_model
    try:
        result = await save_and_transcribe(file, model)
        nlp = analyze_answer_nlp(result.get("transcript") or "", role=role)

        recording = Recording(
            session_id=session_id,
            file_path=result["filepath"],
            transcript=result.get("transcript") or "",
            feedback=result.get("feedback") or "",
        )
        db.add(recording)
        db.flush()

        for item in result.get("word_analysis") or []:
            ts = item.get("time") or item.get("timestamp") or ""
            db.add(
                WordAnalysis(
                    recording_id=recording.id,
                    word=str(item.get("word", "")),
                    issue=str(item.get("issue", "")),
                    suggestion=str(item.get("suggestion", "")),
                    timestamp=str(ts),
                )
            )

        scores = result.get("scores") or {}

        content = score_answer_content_llm(question, result.get("transcript") or "", role)
        resume_result = analyze_resume_text_llm(resume_text, role) if resume_text else None

        db.add(
            RecordingMetric(
                recording_id=recording.id,
                role=role.strip().lower() or "general",
                topic=topic.strip().lower() or "general",
                fluency=float(scores.get("fluency", 0) or 0),
                confidence=float(scores.get("confidence", 0) or 0),
                composure=float(scores.get("composure", 0) or 0),
                overall=float(scores.get("overall", 0) or 0),
                sentiment=nlp.sentiment,
                star_score=float(nlp.star_score),
                coherence_score=float(nlp.coherence_score),
                content_score=float(content.get("content_quality", 0)),
                question_relevance=float(content.get("relevance", 0)),
                resume_score=int(resume_result["score"]) if resume_result else 0,
            )
        )

        db.commit()
        db.refresh(recording)

        return {
            "status": "success",
            "recording_id": recording.id,
            "filepath": result["filepath"],
            "transcript": result["transcript"],
            "temporal": result.get("temporal"),
            "fluency": result.get("fluency"),
            "lexical": result.get("lexical"),
            "acoustic": result.get("acoustic"),
            "scores": result.get("scores"),
            "nlp": {
                "sentiment": nlp.sentiment,
                "sentiment_score": nlp.sentiment_score,
                "star_score": nlp.star_score,
                "coherence_score": nlp.coherence_score,
                "keyword_relevance": nlp.keyword_relevance,
                "weak_topics": nlp.weak_topics,
            },
            "feedback": result.get("feedback"),
            "word_analysis": result.get("word_analysis"),
            "content_scoring": content,
            "resume_analysis": resume_result,
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        log.error("upload-audio failed: %s\n%s", e, traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Audio processing failed: {e!s}",
        ) from e
