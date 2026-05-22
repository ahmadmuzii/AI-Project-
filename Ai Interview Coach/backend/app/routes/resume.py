import json
import os
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User, UserResume
from app.routes.auth import get_current_user
from app.services.resume_service import (
    analyze_skills_gap,
    analyze_skills_gap_llm,
    extract_structured_data,
    extract_structured_data_llm,
    generate_resume_profile_llm,
    generate_resume_summary_llm,
    score_resume_ats,
)

router = APIRouter(prefix="/resume", tags=["resume"])
security = HTTPBearer()

UPLOAD_DIR = Path(__file__).resolve().parent.parent.parent / "uploads" / "resumes"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.get("/list")
def list_resumes(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    resumes = (
        db.query(UserResume)
        .filter(UserResume.user_id == current_user.id)
        .order_by(UserResume.is_primary.desc(), UserResume.created_at.desc())
        .all()
    )
    return [
        {
            "id": r.id,
            "name": r.name,
            "file_path": r.file_path,
            "raw_text": r.raw_text[:200] if r.raw_text else "",
            "skills": json.loads(r.skills) if r.skills else [],
            "experience_years": r.experience_years,
            "education": json.loads(r.education) if r.education else [],
            "is_primary": bool(r.is_primary),
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "analysis": None,
        }
        for r in resumes
    ]


@router.post("/upload")
async def upload_resume(
    name: str = Form("Untitled Resume"),
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    content = await file.read()

    from app.utils.pdf_extractor import extract_text_from_pdf

    raw_text = extract_text_from_pdf(content)
    if not raw_text:
        raise HTTPException(status_code=400, detail="Could not extract text from PDF")

    ext = os.path.splitext(file.filename or "resume.pdf")[1] or ".pdf"
    filename = f"resume_{current_user.id}_{uuid.uuid4().hex[:8]}{ext}"
    filepath = UPLOAD_DIR / filename
    with open(filepath, "wb") as f:
        f.write(content)

    structured = extract_structured_data_llm(raw_text)
    skills = structured.get("skills", [])
    education = structured.get("education", [])
    experience_years = structured.get("experience_years", 0)

    resume = UserResume(
        user_id=current_user.id,
        name=name,
        file_path=str(filepath),
        raw_text=raw_text[:10000],
        skills=json.dumps(skills),
        experience_years=experience_years or None,
        education=json.dumps(education),
    )
    db.add(resume)
    db.flush()

    # Update primary fields on User
    user = db.get(User, current_user.id)
    if not user.resume_text:
        user.resume_text = raw_text[:10000]

    db.commit()
    db.refresh(resume)

    return {
        "id": resume.id,
        "name": resume.name,
        "skills": skills,
        "experience_years": experience_years,
        "education": education,
        "raw_text_preview": raw_text[:300],
    }


@router.post("/manual")
def create_manual_resume(
    name: str = Form("Manual Entry"),
    skills: str = Form("[]"),
    experience_years: int = Form(0),
    education: str = Form("[]"),
    summary: str = Form(""),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        skills_list = json.loads(skills) if isinstance(skills, str) else skills
    except (json.JSONDecodeError, TypeError):
        skills_list = []

    try:
        education_list = json.loads(education) if isinstance(education, str) else education
    except (json.JSONDecodeError, TypeError):
        education_list = []

    raw_text = summary or f"Skills: {', '.join(skills_list)}. Experience: {experience_years} years."
    if education_list:
        raw_text += " Education: " + ", ".join(
            f"{e.get('degree', '')} at {e.get('institution', '')}" for e in education_list
        )

    resume = UserResume(
        user_id=current_user.id,
        name=name,
        raw_text=raw_text,
        skills=json.dumps(skills_list),
        experience_years=experience_years or None,
        education=json.dumps(education_list),
    )
    db.add(resume)
    db.commit()
    db.refresh(resume)

    return {
        "id": resume.id,
        "name": resume.name,
        "skills": skills_list,
        "experience_years": experience_years,
        "education": education_list,
    }


@router.put("/{resume_id}/primary")
def set_primary(
    resume_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    resume = db.query(UserResume).filter(
        UserResume.id == resume_id,
        UserResume.user_id == current_user.id,
    ).first()
    if not resume:
        raise HTTPException(status_code=404, detail="Resume not found")

    db.query(UserResume).filter(
        UserResume.user_id == current_user.id,
        UserResume.is_primary == True,
    ).update({"is_primary": False})

    resume.is_primary = True

    user = db.get(User, current_user.id)
    user.resume_text = resume.raw_text[:10000] if resume.raw_text else ""

    db.commit()
    return {"message": f"'{resume.name}' set as primary resume"}


@router.delete("/{resume_id}")
def delete_resume(
    resume_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    resume = db.query(UserResume).filter(
        UserResume.id == resume_id,
        UserResume.user_id == current_user.id,
    ).first()
    if not resume:
        raise HTTPException(status_code=404, detail="Resume not found")

    if resume.file_path and os.path.exists(resume.file_path):
        try:
            os.remove(resume.file_path)
        except OSError:
            pass

    was_primary = resume.is_primary
    db.delete(resume)

    if was_primary:
        remaining = (
            db.query(UserResume)
            .filter(UserResume.user_id == current_user.id)
            .order_by(UserResume.created_at.desc())
            .first()
        )
        user = db.get(User, current_user.id)
        if remaining:
            remaining.is_primary = True
            user.resume_text = remaining.raw_text[:10000] if remaining.raw_text else ""
        else:
            user.resume_text = ""

    db.commit()
    return {"message": "Resume deleted"}


@router.get("/{resume_id}/profile")
def get_profile(
    resume_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    resume = db.query(UserResume).filter(
        UserResume.id == resume_id,
        UserResume.user_id == current_user.id,
    ).first()
    if not resume:
        raise HTTPException(status_code=404, detail="Resume not found")

    text = resume.raw_text or ""
    return generate_resume_profile_llm(text)


@router.get("/{resume_id}/analysis")
def get_analysis(
    resume_id: int,
    role: str | None = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    resume = db.query(UserResume).filter(
        UserResume.id == resume_id,
        UserResume.user_id == current_user.id,
    ).first()
    if not resume:
        raise HTTPException(status_code=404, detail="Resume not found")

    text = resume.raw_text or ""
    ats = score_resume_ats(text, role or "general")

    if role:
        skills = json.loads(resume.skills) if resume.skills else []
        gap = analyze_skills_gap_llm(skills or extract_structured_data(text).get("skills", []), role)
        summary = generate_resume_summary_llm(text, role, ats["ats_score"], gap)
    else:
        gap = {"matched": [], "missing": [], "irrelevant": [], "suggestions": []}
        summary = ""

    return {
        "ats": ats,
        "skills_gap": gap,
        "summary": summary,
    }
