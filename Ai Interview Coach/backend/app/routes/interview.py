import os
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import InterviewSession, Recording, User, WordAnalysis
from app.routes.auth import get_current_user
from app.schemas import CreateUserBody, RecordingOut, WordAnalysisOut


def _backfill_session_numbers(db: Session, user_id: int) -> None:
    zero = (
        db.query(InterviewSession)
        .filter(InterviewSession.user_id == user_id, InterviewSession.session_number == 0)
        .order_by(InterviewSession.started_at.asc())
        .all()
    )
    if not zero:
        return
    max_existing = (
        db.query(InterviewSession.session_number)
        .filter(InterviewSession.user_id == user_id, InterviewSession.session_number > 0)
        .order_by(InterviewSession.session_number.desc())
        .first()
    )
    start = (max_existing[0] if max_existing else 0) + 1
    for i, sess in enumerate(zero, start=start):
        sess.session_number = i
    db.commit()

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
def start_session(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    session_type: str = "practice",
    role: str = "general",
    topic: str = "general",
):
    max_num = (
        db.query(InterviewSession.session_number)
        .filter(InterviewSession.user_id == current_user.id)
        .order_by(InterviewSession.session_number.desc())
        .first()
    )
    next_num = (max_num[0] if max_num and max_num[0] else 0) + 1
    sess = InterviewSession(
        user_id=current_user.id,
        session_number=next_num,
        session_type=session_type,
        role=role.strip().lower(),
        topic=topic.strip().lower(),
    )
    db.add(sess)
    db.commit()
    db.refresh(sess)
    return {"session_id": sess.id, "session_number": sess.session_number, "started_at": sess.started_at.isoformat()}


@router.get("/sessions")
def list_sessions(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _backfill_session_numbers(db, current_user.id)
    sessions = (
        db.query(InterviewSession)
        .filter(InterviewSession.user_id == current_user.id)
        .order_by(InterviewSession.started_at.desc())
        .all()
    )
    result = []
    for s in sessions:
        rec_count = len(s.recordings)
        scores = []
        confs = []
        for r in s.recordings:
            for m in r.metrics:
                if m.overall > 0:
                    scores.append(m.overall)
                if m.confidence > 0:
                    confs.append(m.confidence)
        overall = round(sum(scores) / len(scores), 1) if scores else 0
        confidence = round(sum(confs) / len(confs), 2) if confs else 0
        if overall != s.overall_score or confidence != s.avg_confidence:
            s.overall_score = overall
            s.avg_confidence = confidence
            db.commit()
        result.append({
            "id": s.id,
            "user_id": s.user_id,
            "session_number": s.session_number,
            "session_type": s.session_type,
            "role": s.role,
            "topic": s.topic,
            "overall_score": overall,
            "avg_confidence": confidence,
            "recording_count": rec_count,
            "started_at": s.started_at.isoformat(),
        })
    return result


@router.delete("/sessions")
def clear_sessions(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    sessions = (
        db.query(InterviewSession)
        .filter(InterviewSession.user_id == current_user.id)
        .all()
    )
    if not sessions:
        return {"deleted": 0}

    paths = []
    for s in sessions:
        for r in s.recordings:
            if r.file_path:
                paths.append(Path(r.file_path))

    for s in sessions:
        db.delete(s)
    db.commit()

    for p in paths:
        try:
            if p.is_file():
                p.unlink()
        except OSError:
            pass

    return {"deleted": len(sessions)}


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


@router.delete("/recording/{recording_id}")
def delete_recording(
    recording_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    r = db.get(Recording, recording_id)
    if not r:
        raise HTTPException(status_code=404, detail="Recording not found")
    sess = r.session
    if not sess or sess.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not your recording")
    path = Path(r.file_path).resolve() if r.file_path else None
    db.delete(r)
    db.commit()
    if path and path.is_file():
        try:
            path.unlink()
        except OSError:
            pass
    return {"deleted": recording_id}


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
