from datetime import date, datetime

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import InterviewSession, Recording, RecordingMetric, User
from app.routes.auth import get_current_user
from app.services.intelligence_service import (
    analyze_answer_nlp,
    analyze_resume_text,
    analyze_resume_text_llm,
    build_study_plan,
    company_mode_questions,
    evaluate_stress,
    generate_company_questions_llm,
    generate_follow_up_llm,
    generate_questions_llm,
    generate_session_summary_llm,
    predict_readiness_days,
    score_answer_content_llm,
    suggest_questions,
    topic_heatmap,
)
from app.services.webcam_service import analyze_webcam_frame

router = APIRouter(tags=["analytics"])


# ════════════════════════════════════════════
#  Feature 1: LLM Question Generation
# ════════════════════════════════════════════


@router.post("/adaptive/next-questions")
def adaptive_next_questions(
    role: str = Form("general"),
    weak_topics: str = Form("general"),
    previous_questions: str = Form(""),
    resume_text: str = Form(""),
):
    weak = [w.strip() for w in weak_topics.split(",") if w.strip()]
    previous = [q.strip() for q in previous_questions.split("||") if q.strip()]
    questions = generate_questions_llm(
        role=role,
        weak_topics=weak if weak else ["general"],
        previous_questions=previous,
        max_items=4,
        resume_text=resume_text,
    )
    return {"questions": questions}


@router.get("/company-mode")
def company_mode(company: str, role: str = "general"):
    return generate_company_questions_llm(company, role)


# ════════════════════════════════════════════
#  Feature 2: Resume-Aware Feedback
# ════════════════════════════════════════════


@router.post("/resume/analyze")
def resume_analyze(resume_text: str = Form(...), role: str = Form("backend")):
    return analyze_resume_text_llm(resume_text, role)


# ════════════════════════════════════════════
#  Feature 3: Session Summary
# ════════════════════════════════════════════


@router.get("/session-summary/{session_id}")
def session_summary(session_id: int, db: Session = Depends(get_db)):
    sess = db.get(InterviewSession, session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")

    recordings = (
        db.query(Recording)
        .filter(Recording.session_id == session_id)
        .order_by(Recording.created_at.asc())
        .all()
    )

    recordings_data = []
    for r in recordings:
        metrics = (
            db.query(RecordingMetric)
            .filter(RecordingMetric.recording_id == r.id)
            .first()
        )
        scores = {}
        if metrics:
            scores = {
                "fluency": metrics.fluency,
                "confidence": metrics.confidence,
                "composure": metrics.composure,
                "overall": metrics.overall,
            }
        recordings_data.append(
            {
                "id": r.id,
                "transcript": r.transcript,
                "feedback": r.feedback,
                "scores": scores,
                "created_at": r.created_at.isoformat() if r.created_at else "",
            }
        )

    result = generate_session_summary_llm(recordings_data)
    result["session_id"] = session_id
    result["recording_count"] = len(recordings_data)
    return result


# ════════════════════════════════════════════
#  Feature 4: Follow-up Questions
# ════════════════════════════════════════════


@router.post("/follow-up")
def follow_up(
    question: str = Form(...),
    answer: str = Form(...),
    role: str = Form("general"),
):
    fq = generate_follow_up_llm(question, answer, role)
    return {"follow_up": fq}


# ════════════════════════════════════════════
#  Feature 5: Content Scoring (also in audio.py)
# ════════════════════════════════════════════


@router.post("/score-content")
def score_content(
    question: str = Form(...),
    answer: str = Form(...),
    role: str = Form("general"),
):
    return score_answer_content_llm(question, answer, role)


# ════════════════════════════════════════════
#  Existing endpoints (unchanged)
# ════════════════════════════════════════════


@router.post("/nlp/analyze-answer")
def nlp_analyze_answer(answer: str = Form(...), role: str = Form("general")):
    result = analyze_answer_nlp(answer, role)
    return {
        "sentiment": result.sentiment,
        "sentiment_score": result.sentiment_score,
        "star_score": result.star_score,
        "coherence_score": result.coherence_score,
        "keyword_relevance": result.keyword_relevance,
        "weak_topics": result.weak_topics,
    }


@router.get("/dashboard")
def dashboard(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    q = (
        db.query(
            RecordingMetric.created_at,
            RecordingMetric.overall,
            RecordingMetric.confidence,
            RecordingMetric.topic,
            RecordingMetric.role,
        )
        .join(Recording, Recording.id == RecordingMetric.recording_id)
        .join(InterviewSession, InterviewSession.id == Recording.session_id)
        .filter(InterviewSession.user_id == current_user.id)
        .order_by(RecordingMetric.created_at.asc())
    )
    rows = q.all()
    if not rows:
        return {
            "series": [],
            "heatmap": {},
            "streak_days": 0,
            "readiness_days": None,
            "comparison_percentile": None,
        }

    overall_scores = [float(r.overall or 0.0) for r in rows]
    series = [
        {
            "date": (
                r.created_at.isoformat()
                if isinstance(r.created_at, datetime)
                else str(r.created_at)
            ),
            "overall": float(r.overall or 0.0),
            "confidence": float(r.confidence or 0.0),
            "topic": r.topic or "general",
            "role": r.role or "general",
        }
        for r in rows
    ]
    heatmap = topic_heatmap(
        [(r.topic or "general", float(r.overall or 0.0)) for r in rows]
    )

    days = sorted(
        {
            r.created_at.date() if isinstance(r.created_at, datetime) else date.today()
            for r in rows
        },
        reverse=True,
    )
    streak = 0
    cursor = date.today()
    for d in days:
        if d == cursor:
            streak += 1
            cursor = date.fromordinal(cursor.toordinal() - 1)
        elif d == date.fromordinal(cursor.toordinal() - 1) and streak == 0:
            streak += 1
            cursor = d
        elif d < cursor:
            break

    readiness_days = predict_readiness_days(overall_scores, target_score=0.8)

    user_avg = sum(overall_scores) / max(1, len(overall_scores))
    all_avgs = (
        db.query(func.avg(RecordingMetric.overall).label("avg_score"))
        .group_by(RecordingMetric.role)
        .all()
    )
    peer_vals = [float(v.avg_score or 0) for v in all_avgs]
    percentile = None
    if peer_vals:
        below = sum(1 for p in peer_vals if p <= user_avg)
        percentile = int(round((below / len(peer_vals)) * 100))

    return {
        "series": series,
        "heatmap": heatmap,
        "streak_days": streak,
        "readiness_days": readiness_days,
        "comparison_percentile": percentile,
    }


@router.get("/leaderboard")
def leaderboard(role: str = "general", limit: int = 20, db: Session = Depends(get_db)):
    rows = (
        db.query(
            User.name.label("name"),
            func.avg(RecordingMetric.overall).label("score"),
        )
        .join(InterviewSession, InterviewSession.user_id == User.id)
        .join(Recording, Recording.session_id == InterviewSession.id)
        .join(RecordingMetric, RecordingMetric.recording_id == Recording.id)
        .filter(RecordingMetric.role == role.lower())
        .group_by(User.id, User.name)
        .order_by(func.avg(RecordingMetric.overall).desc())
        .limit(limit)
        .all()
    )
    return {
        "role": role.lower(),
        "leaders": [
            {"rank": idx + 1, "name": r.name, "score": round(float(r.score or 0), 3)}
            for idx, r in enumerate(rows)
        ],
    }


@router.get("/study-plan")
def study_plan(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    metrics = (
        db.query(RecordingMetric.topic, RecordingMetric.overall)
        .join(Recording, Recording.id == RecordingMetric.recording_id)
        .join(InterviewSession, InterviewSession.id == Recording.session_id)
        .filter(InterviewSession.user_id == current_user.id)
        .order_by(RecordingMetric.created_at.desc())
        .limit(20)
        .all()
    )
    if not metrics:
        weak = ["general"]
    else:
        hm = topic_heatmap(
            [(m.topic or "general", float(m.overall or 0.0)) for m in metrics]
        )
        weak = [
            k
            for k, v in sorted(hm.items(), key=lambda kv: kv[1], reverse=True)[:3]
        ]
    return {"weak_topics": weak, "plan": build_study_plan(weak)}


@router.post("/stress/evaluate")
def stress_evaluate(
    eye_contact_score: float = Form(...),
    movement_score: float = Form(...),
    voice_energy: float = Form(...),
):
    return evaluate_stress(eye_contact_score, movement_score, voice_energy)


@router.post("/stress/analyze-webcam")
async def stress_analyze_webcam(
    stream_id: str = Form("default"),
    voice_energy: float = Form(0.6),
    frame: UploadFile = File(...),
):
    frame_bytes = await frame.read()
    cv_out = analyze_webcam_frame(frame_bytes, stream_id=stream_id)
    if not cv_out.get("ok"):
        raise HTTPException(status_code=400, detail=cv_out.get("error", "Webcam analysis failed"))
    stress = evaluate_stress(
        eye_contact_score=float(cv_out["eye_contact_score"]),
        movement_score=float(cv_out["movement_score"]),
        voice_energy=float(voice_energy),
    )
    return {
        "vision": cv_out,
        "stress": stress,
    }
