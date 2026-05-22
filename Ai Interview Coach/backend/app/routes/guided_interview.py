import json
import logging
import os
import traceback
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import GuidedInterview, InterviewQA, InterviewSession, Recording, RecordingMetric, User
from app.routes.auth import get_current_user
from app.services.audio_service import save_and_transcribe
from app.services.intelligence_service import (
    analyze_answer_nlp,
    generate_interview_summary,
    generate_next_question,
    generate_greeting_and_clarifying_questions,
    generate_first_interview_question,
    score_answer_content_llm,
)

router = APIRouter()
log = logging.getLogger("uvicorn.error")


def _get_interview_or_404(interview_id: int, db: Session, current_user: User | None = None) -> GuidedInterview:
    interview = db.query(GuidedInterview).filter(GuidedInterview.id == interview_id).first()
    if not interview:
        raise HTTPException(status_code=404, detail="Interview not found")
    if current_user and interview.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not your interview")
    return interview


def _get_user_or_404(user_id: int, db: Session) -> User:
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


def _profile_dict(user: User) -> dict:
    return {
        "target_role": user.target_role or "",
        "years_of_experience": user.years_of_experience,
        "seniority_level": user.seniority_level or "",
        "focus_areas": user.focus_areas or "",
        "resume_text": user.resume_text or "",
        "target_industry": user.target_industry or "",
    }


@router.get("/guided/list")
def list_guided_interviews(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    interviews = (
        db.query(GuidedInterview)
        .filter(GuidedInterview.user_id == current_user.id)
        .order_by(GuidedInterview.started_at.desc())
        .all()
    )
    return [
        {
            "id": i.id,
            "aim": i.aim,
            "target_company": i.target_company,
            "difficulty": i.difficulty,
            "status": i.status,
            "overall_score": i.overall_score,
            "started_at": i.started_at.isoformat(),
            "completed_at": i.completed_at.isoformat() if i.completed_at else None,
            "recording_count": i.recording_count,
            "duration_minutes": i.duration_minutes,
        }
        for i in interviews
    ]


@router.post("/guided/start")
def start_guided_interview(
    aim: str = Form(""),
    target_company: str = Form(""),
    duration_minutes: int = Form(30),
    difficulty: str = Form("intermediate"),
    focus_areas: str = Form("[]"),
    mode: str = Form("text"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    profile = _profile_dict(current_user)

    # Create interview
    interview = GuidedInterview(
        user_id=current_user.id,
        aim=aim,
        target_company=target_company,
        duration_minutes=duration_minutes,
        difficulty=difficulty,
        focus_areas=focus_areas,
        mode=mode,
        status="in_progress",
    )
    db.add(interview)
    db.flush()

    # Generate greeting + clarifying questions
    greeting_data = generate_greeting_and_clarifying_questions(profile, aim, target_company)

    interview.status = "in_progress"
    interview.clarifying_questions = json.dumps({
        "greeting": greeting_data["greeting"],
        "questions": greeting_data["clarifying_questions"],
    })
    interview.clarification_answers = json.dumps([])
    db.commit()
    db.refresh(interview)

    return {
        "interview_id": interview.id,
        "greeting_message": greeting_data["greeting"],
        "clarifying_questions": greeting_data["clarifying_questions"],
        "status": "greeting",
        "started_at": interview.started_at.isoformat(),
        "mode": mode,
    }


@router.post("/guided/{interview_id}/answer-clarification")
async def answer_clarification(
    interview_id: int,
    request: Request,
    index: int = Form(...),
    file: UploadFile = File(None),
    text: str = Form(""),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    interview = _get_interview_or_404(interview_id, db, current_user)
    if interview.status != "in_progress":
        raise HTTPException(status_code=400, detail="Interview is not in progress")

    profile = _profile_dict(current_user)

    # Parse clarifying questions
    cq_data = json.loads(interview.clarifying_questions or "{}")
    questions = cq_data.get("questions", [])

    if index < 0 or index >= len(questions):
        raise HTTPException(status_code=400, detail="Invalid clarification question index")

    # Parse existing answers
    answers = json.loads(interview.clarification_answers or "[]")

    if text:
        transcript = text.strip()
    elif file:
        model = request.app.state.whisper_model
        try:
            result = await save_and_transcribe(file, model)
        except Exception as e:
            log.error("Clarification transcription failed: %s", e)
            raise HTTPException(status_code=500, detail=f"Audio processing failed: {e!s}")
        transcript = result.get("transcript") or ""
    else:
        transcript = ""

    # Store answer
    while len(answers) <= index:
        answers.append("")
    answers[index] = transcript
    interview.clarification_answers = json.dumps(answers)
    db.commit()

    # Check if all answered (allow empty answers — text mode skips clarifications)
    all_answered = len(answers) >= len(questions)

    if all_answered:
        # Generate first real interview question
        aim = interview.aim or ""
        company = interview.target_company or ""
        first_question = generate_first_interview_question(profile, aim, company)

        estimated_count = max(3, interview.duration_minutes * 2)
        max_possible = estimated_count

        qa = InterviewQA(
            interview_id=interview.id,
            question=first_question,
            order=0,
        )
        db.add(qa)
        db.commit()
        db.refresh(qa)

        return {
            "done": True,
            "next_index": None,
            "first_question": first_question,
            "current_qa_id": qa.id,
            "total_estimated": max_possible,
            "questions_remaining": max_possible - 1,
            "mode": interview.mode,
        }

    return {
        "done": False,
        "next_index": index + 1,
        "first_question": None,
    }


@router.post("/guided/{interview_id}/answer")
async def answer_guided_question(
    interview_id: int,
    request: Request,
    file: UploadFile = File(None),
    text: str = Form(""),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    interview = _get_interview_or_404(interview_id, db, current_user)
    if interview.status != "in_progress":
        raise HTTPException(status_code=400, detail="Interview is not in progress")

    if not file and not text:
        raise HTTPException(status_code=400, detail="Provide either audio file or text answer")

    profile = _profile_dict(current_user)

    # Find current unanswered QA
    current_qa = (
        db.query(InterviewQA)
        .filter(InterviewQA.interview_id == interview_id, InterviewQA.transcript == "")
        .order_by(InterviewQA.order)
        .first()
    )
    if not current_qa:
        raise HTTPException(status_code=400, detail="No pending question to answer")

    # Create a temporary session for this Q&A
    sess = InterviewSession(
        user_id=interview.user_id,
        session_type="guided",
        role=(profile.get("target_role") or "general").strip().lower(),
        topic="guided",
    )
    db.add(sess)
    db.flush()

    is_text_mode = bool(text.strip())

    if is_text_mode:
        # Text mode — use provided text directly, no audio processing
        transcript = text.strip()
        result = {"transcript": transcript, "scores": {}, "feedback": "", "filepath": ""}
    else:
        # Audio mode — transcribe via Whisper
        model = request.app.state.whisper_model
        try:
            result = await save_and_transcribe(file, model)
        except Exception as e:
            db.rollback()
            log.error("Transcription failed: %s", e)
            raise HTTPException(status_code=500, detail=f"Audio processing failed: {e!s}")
        transcript = result.get("transcript") or ""

    # Create recording
    recording = Recording(
        session_id=sess.id,
        file_path=result.get("filepath", ""),
        transcript=transcript,
        feedback="",
    )
    db.add(recording)
    db.flush()

    # NLP + content scoring
    nlp = analyze_answer_nlp(transcript, role=profile.get("target_role", "general"))
    content = score_answer_content_llm(current_qa.question, transcript, profile.get("target_role", "general"))

    # Create metric
    scores = result.get("scores") or {}
    db.add(
        RecordingMetric(
            recording_id=recording.id,
            role=(profile.get("target_role") or "general").strip().lower(),
            topic="guided",
            fluency=float(scores.get("fluency", 0) or 0),
            confidence=float(scores.get("confidence", 0) or 0),
            composure=float(scores.get("composure", 0) or 0),
            overall=float(scores.get("overall", 0) or 0),
            sentiment=nlp.sentiment,
            star_score=float(nlp.star_score),
            coherence_score=float(nlp.coherence_score),
            content_score=float(content.get("content_quality", 0)),
            question_relevance=float(content.get("relevance", 0)),
        )
    )

    # Get feedback (skip audio analysis for text mode — no temporal data)
    feedback_text = result.get("feedback") or ""
    if not feedback_text and transcript and result.get("temporal"):
        from app.services.analysis_service import generate_feedback
        fb = generate_feedback(
            result.get("temporal") or {},
            result.get("fluency") or {},
            result.get("lexical") or {},
            result.get("acoustic") or {},
            scores,
            result.get("words") or [],
            transcript,
            current_user.resume_text or "",
        )
        feedback_text = fb.get("general", "")

    # Update QA
    current_qa.recording_id = recording.id
    current_qa.transcript = transcript
    current_qa.feedback = feedback_text
    current_qa.content_score = float(content.get("content_quality", 0))
    current_qa.relevance_score = float(content.get("relevance", 0))
    current_qa.fluency_score = float(scores.get("fluency", 0))
    current_qa.confidence_score = float(scores.get("confidence", 0))

    # Update interview recording count
    interview.recording_count = (
        db.query(InterviewQA)
        .filter(InterviewQA.interview_id == interview_id, InterviewQA.transcript != "")
        .count()
    )

    db.commit()
    db.refresh(current_qa)

    # Check if interview is done
    total_estimated = max(3, interview.duration_minutes * 2)
    elapsed_count = interview.recording_count

    # Check time elapsed
    now = datetime.now(timezone.utc)
    started = interview.started_at.replace(tzinfo=timezone.utc) if interview.started_at.tzinfo is None else interview.started_at
    time_elapsed = (now - started).total_seconds()
    time_limit = interview.duration_minutes * 60
    time_up = time_elapsed >= time_limit

    is_complete = False
    next_question = ""
    next_qa_id = None

    if time_up:
        is_complete = True
        interview.status = "completed"
        interview.completed_at = now
    else:
        # Build context for next question
        qa_history = (
            db.query(InterviewQA)
            .filter(InterviewQA.interview_id == interview_id)
            .order_by(InterviewQA.order)
            .all()
        )
        history = [
            {
                "question": q.question,
                "transcript": q.transcript,
                "content_score": q.content_score,
                "relevance_score": q.relevance_score,
                "fluency_score": q.fluency_score,
                "confidence_score": q.confidence_score,
            }
            for q in qa_history
            if q.transcript
        ]

        context = {
            "target_company": interview.target_company,
            "difficulty": interview.difficulty,
        }

        next_question = generate_next_question(context, history, profile)

        # Store next question
        next_qa = InterviewQA(
            interview_id=interview_id,
            question=next_question,
            order=elapsed_count,
        )
        db.add(next_qa)
        db.commit()
        db.refresh(next_qa)
        next_qa_id = next_qa.id

    seconds_elapsed = int((now - started).total_seconds())
    questions_remaining = max(0, total_estimated - elapsed_count) if not is_complete else 0

    return {
        "qa_id": current_qa.id,
        "transcript": transcript,
        "feedback": feedback_text,
        "scores": {
            "content": current_qa.content_score,
            "relevance": current_qa.relevance_score,
            "fluency": current_qa.fluency_score,
            "confidence": current_qa.confidence_score,
        },
        "is_complete": is_complete,
        "next_question": next_question,
        "next_qa_id": next_qa_id,
        "elapsed_seconds": seconds_elapsed,
        "questions_remaining": questions_remaining,
    }


@router.get("/guided/{interview_id}")
def get_guided_interview(
    interview_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    interview = _get_interview_or_404(interview_id, db, current_user)
    qa_list = (
        db.query(InterviewQA)
        .filter(InterviewQA.interview_id == interview_id)
        .order_by(InterviewQA.order)
        .all()
    )

    now = datetime.now(timezone.utc)
    started = interview.started_at.replace(tzinfo=timezone.utc) if interview.started_at.tzinfo is None else interview.started_at
    time_elapsed = (now - started).total_seconds()
    time_limit = interview.duration_minutes * 60
    remaining_seconds = max(0, int(time_limit - time_elapsed))

    # Determine phase
    has_qa = len(qa_list) > 0
    has_clarification_answers = bool(interview.clarification_answers)
    if interview.status == "completed":
        phase = "summary"
    elif has_qa:
        phase = "session"
    else:
        phase = "greeting"

    # Get the pending question (only for session phase)
    pending_qa = None
    if phase == "session":
        for qa in qa_list:
            if not qa.transcript:
                pending_qa = {"id": qa.id, "question": qa.question, "order": qa.order}
                break

    # Parse clarification data for greeting phase
    clarification_answers_list = []
    if has_clarification_answers:
        try:
            clarification_answers_list = json.loads(interview.clarification_answers)
        except (json.JSONDecodeError, TypeError):
            pass

    # Parse stored clarifying questions for greeting phase
    clarifying_questions_list = []
    greeting_message = ""
    if interview.clarifying_questions:
        try:
            cq_data = json.loads(interview.clarifying_questions)
            greeting_message = cq_data.get("greeting", "")
            clarifying_questions_list = cq_data.get("questions", [])
        except (json.JSONDecodeError, TypeError):
            pass

    return {
        "interview": {
            "id": interview.id,
            "user_id": interview.user_id,
            "aim": interview.aim,
            "target_company": interview.target_company,
            "duration_minutes": interview.duration_minutes,
            "difficulty": interview.difficulty,
            "focus_areas": interview.get_focus_areas(),
            "mode": interview.mode,
            "status": interview.status,
            "started_at": interview.started_at.isoformat(),
            "completed_at": interview.completed_at.isoformat() if interview.completed_at else None,
            "overall_score": interview.overall_score,
            "recording_count": interview.recording_count,
            "summary": interview.summary,
            "phase": phase,
            "clarification_answers": clarification_answers_list,
            "greeting_message": greeting_message,
            "clarifying_questions": clarifying_questions_list,
        },
        "qa_pairs": [
            {
                "id": qa.id,
                "question": qa.question,
                "transcript": qa.transcript,
                "feedback": qa.feedback,
                "content_score": qa.content_score,
                "relevance_score": qa.relevance_score,
                "fluency_score": qa.fluency_score,
                "confidence_score": qa.confidence_score,
                "order": qa.order,
                "created_at": qa.created_at.isoformat(),
            }
            for qa in qa_list
        ],
        "pending_question": pending_qa,
        "remaining_seconds": remaining_seconds,
        "time_expired": time_elapsed >= time_limit,
    }


@router.post("/guided/{interview_id}/end")
def end_guided_interview(
    interview_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    interview = _get_interview_or_404(interview_id, db, current_user)
    if interview.status == "completed":
        raise HTTPException(status_code=400, detail="Interview already completed")

    profile = _profile_dict(current_user)

    # Mark as completed
    interview.status = "completed"
    interview.completed_at = datetime.now(timezone.utc)

    # Gather all answered QAs
    qa_list = (
        db.query(InterviewQA)
        .filter(InterviewQA.interview_id == interview_id, InterviewQA.transcript != "")
        .order_by(InterviewQA.order)
        .all()
    )

    if qa_list:
        # Calculate scores
        avg_content = sum(q.content_score for q in qa_list) / len(qa_list)
        avg_relevance = sum(q.relevance_score for q in qa_list) / len(qa_list)
        avg_fluency = sum(q.fluency_score for q in qa_list) / len(qa_list)
        avg_confidence = sum(q.confidence_score for q in qa_list) / len(qa_list)
        overall = round((avg_content * 0.3 + avg_relevance * 0.2 + avg_fluency * 0.25 + avg_confidence * 0.25) * 100, 1)

        interview.overall_score = overall

        scores = {
            "overall": overall,
            "avg_content": round(avg_content * 100, 1),
            "avg_relevance": round(avg_relevance * 100, 1),
            "avg_fluency": round(avg_fluency * 100, 1),
            "avg_confidence": round(avg_confidence * 100, 1),
        }

        qa_data = [
            {
                "question": q.question,
                "transcript": q.transcript,
                "feedback": q.feedback,
                "content_score": q.content_score,
                "relevance_score": q.relevance_score,
                "fluency_score": q.fluency_score,
                "confidence_score": q.confidence_score,
            }
            for q in qa_list
        ]

        summary_result = generate_interview_summary(qa_data, profile, scores)
        interview.summary = json.dumps(summary_result)
    else:
        interview.overall_score = 0
        scores = {"overall": 0}
        summary_result = {
            "summary": "No answers recorded.",
            "strengths": [],
            "top_improvements": ["Start by answering the first question"],
            "action_plan": ["Practice answering interview questions"],
            "readiness_estimate": "Not ready",
        }
        interview.summary = json.dumps(summary_result)

    # Remove any pending (unanswered) QAs
    db.query(InterviewQA).filter(
        InterviewQA.interview_id == interview_id,
        InterviewQA.transcript == "",
    ).delete()

    db.commit()
    db.refresh(interview)

    end = interview.completed_at.replace(tzinfo=timezone.utc) if interview.completed_at and interview.completed_at.tzinfo is None else interview.completed_at
    start = interview.started_at.replace(tzinfo=timezone.utc) if interview.started_at.tzinfo is None else interview.started_at
    time_elapsed = int((end - start).total_seconds())

    return {
        "interview_id": interview.id,
        "status": "completed",
        "overall_score": interview.overall_score,
        "total_answered": len(qa_list),
        "time_elapsed_seconds": time_elapsed,
        "summary": summary_result,
    }


@router.post("/guided/{interview_id}/upload-resume")
async def upload_resume_guided(
    interview_id: int,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    interview = _get_interview_or_404(interview_id, db, current_user)

    from app.utils.pdf_extractor import extract_text_from_pdf

    content = await file.read()
    raw_text = extract_text_from_pdf(content)

    current_user.resume_text = raw_text
    db.commit()

    return {
        "status": "ok",
        "message": "Resume uploaded and extracted.",
        "text_length": len(raw_text),
    }
