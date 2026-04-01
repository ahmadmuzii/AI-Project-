import os
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import InterviewSession, Recording, User, WordAnalysis
from app.schemas import CreateUserBody, RecordingOut, StartSessionBody, WordAnalysisOut

router = APIRouter(tags=["interview"])


def _recording_to_out(db: Session, r: Recording) -> RecordingOut:
    rows = (
        db.query(WordAnalysis)
        .filter(WordAnalysis.recording_id == r.id)
        .order_by(WordAnalysis.id.asc())
        .all()
    )
    return RecordingOut(
        id=r.id,
        session_id=r.session_id,
        file_path=r.file_path,
        transcript=r.transcript,
        feedback=r.feedback,
        created_at=r.created_at,
        word_analysis=[
            WordAnalysisOut(
                timestamp=w.timestamp,
                word=w.word,
                issue=w.issue,
                suggestion=w.suggestion,
            )
            for w in rows
        ],
    )


@router.post("/create-user")
def create_user(body: CreateUserBody, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == body.email.strip().lower()).first()
    if existing:
        return {
            "user_id": existing.id,
            "name": existing.name,
            "email": existing.email,
            "existing": True,
        }
    user = User(name=body.name.strip(), email=body.email.strip().lower())
    db.add(user)
    db.commit()
    db.refresh(user)
    return {
        "user_id": user.id,
        "name": user.name,
        "email": user.email,
        "existing": False,
    }


@router.post("/start-session")
def start_session(body: StartSessionBody, db: Session = Depends(get_db)):
    user = db.get(User, body.user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    sess = InterviewSession(user_id=body.user_id)
    db.add(sess)
    db.commit()
    db.refresh(sess)
    return {"session_id": sess.id, "started_at": sess.started_at.isoformat()}


@router.get("/sessions/{user_id}")
def list_sessions(user_id: int, db: Session = Depends(get_db)):
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    sessions = (
        db.query(InterviewSession)
        .filter(InterviewSession.user_id == user_id)
        .order_by(InterviewSession.started_at.desc())
        .all()
    )
    return [
        {"id": s.id, "user_id": s.user_id, "started_at": s.started_at.isoformat()}
        for s in sessions
    ]


@router.get("/recordings/{session_id}")
def list_recordings(session_id: int, db: Session = Depends(get_db)):
    sess = db.get(InterviewSession, session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")
    recs = (
        db.query(Recording)
        .filter(Recording.session_id == session_id)
        .order_by(Recording.created_at.asc())
        .all()
    )
    return {"recordings": [_recording_to_out(db, r).model_dump(mode="json") for r in recs]}


@router.get("/recording/{recording_id}")
def get_recording(recording_id: int, db: Session = Depends(get_db)):
    r = db.get(Recording, recording_id)
    if not r:
        raise HTTPException(status_code=404, detail="Recording not found")
    return _recording_to_out(db, r).model_dump(mode="json")


def _guess_media_type(path: str) -> str:
    ext = Path(path).suffix.lower()
    return {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".m4a": "audio/mp4",
        ".webm": "audio/webm",
        ".ogg": "audio/ogg",
    }.get(ext, "application/octet-stream")


@router.get("/recording/{recording_id}/audio")
def stream_recording_audio(recording_id: int, db: Session = Depends(get_db)):
    r = db.get(Recording, recording_id)
    if not r:
        raise HTTPException(status_code=404, detail="Recording not found")
    path = Path(r.file_path).resolve()
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Audio file missing on disk")
    upload_root = Path("uploads").resolve()
    try:
        path.relative_to(upload_root)
    except ValueError:
        raise HTTPException(status_code=403, detail="Invalid file path")
    return FileResponse(os.fspath(path), media_type=_guess_media_type(os.fspath(path)))
