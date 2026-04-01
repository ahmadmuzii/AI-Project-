from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import InterviewSession, Recording, WordAnalysis
from app.services.audio_service import save_and_transcribe

router = APIRouter()


@router.post("/upload-audio")
async def upload_audio(
    request: Request,
    user_id: int = Form(...),
    session_id: int = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    sess = (
        db.query(InterviewSession)
        .filter(
            InterviewSession.id == session_id,
            InterviewSession.user_id == user_id,
        )
        .first()
    )
    if not sess:
        raise HTTPException(status_code=400, detail="Invalid user_id or session_id")

    model = request.app.state.whisper_model
    result = await save_and_transcribe(file, model)

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
        "feedback": result.get("feedback"),
        "word_analysis": result.get("word_analysis"),
    }
