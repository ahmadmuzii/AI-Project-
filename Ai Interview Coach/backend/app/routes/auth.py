import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User
from app.utils.pdf_extractor import extract_text_from_pdf

SECRET_KEY = "aic-secret-key-change-in-production-2026"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()
router = APIRouter(prefix="/auth", tags=["auth"])

UPLOAD_DIR = Path(__file__).resolve().parent.parent.parent / "uploads" / "avatars"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class RegisterBody(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    email: str = Field(..., min_length=3, max_length=255)
    password: str = Field(..., min_length=4, max_length=128)


class LoginBody(BaseModel):
    email: str = Field(..., min_length=3, max_length=255)
    password: str = Field(..., min_length=1, max_length=128)


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: int
    name: str
    email: str


class UserOut(BaseModel):
    user_id: int
    name: str
    email: str


class ProfileOut(BaseModel):
    user_id: int
    name: str
    email: str
    display_name: str | None = None
    phone: str | None = None
    avatar_url: str | None = None
    bio: str | None = None
    target_role: str | None = None
    target_industry: str | None = None
    seniority_level: str | None = None
    years_of_experience: int | None = None
    current_company: str | None = None
    education_level: str | None = None
    linkedin_url: str | None = None
    resume_text: str | None = None
    focus_areas: list[str] = []
    upcoming_interview_date: str | None = None
    preferred_difficulty: str | None = None
    locale: str | None = None
    timezone: str | None = None
    theme: str = "light"
    notify_email_digests: bool = True
    notify_session_reminders: bool = True
    mic_default: str | None = None
    camera_default: str | None = None
    use_elevenlabs: bool = False
    elevenlabs_voice_id: str | None = None
    profile_completed: bool = False
    created_at: str | None = None


class ChangePasswordBody(BaseModel):
    old_password: str = Field(..., min_length=1)
    new_password: str = Field(..., min_length=4, max_length=128)


class DeleteAccountBody(BaseModel):
    password: str = Field(..., min_length=1)


def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
) -> User:
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("user_id")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


def _profile_to_dict(user: User) -> dict:
    focus = []
    if user.focus_areas:
        try:
            focus = json.loads(user.focus_areas)
        except (json.JSONDecodeError, TypeError):
            focus = []
    avatar_url = None
    if user.avatar_path:
        avatar_url = f"/uploads/avatars/{os.path.basename(user.avatar_path)}"
    return {
        "user_id": user.id,
        "name": user.name,
        "email": user.email,
        "display_name": user.display_name,
        "phone": user.phone,
        "avatar_url": avatar_url,
        "bio": user.bio,
        "target_role": user.target_role,
        "target_industry": user.target_industry,
        "seniority_level": user.seniority_level,
        "years_of_experience": user.years_of_experience,
        "current_company": user.current_company,
        "education_level": user.education_level,
        "linkedin_url": user.linkedin_url,
        "resume_text": user.resume_text,
        "focus_areas": focus,
        "upcoming_interview_date": user.upcoming_interview_date.isoformat() if user.upcoming_interview_date else None,
        "preferred_difficulty": user.preferred_difficulty,
        "locale": user.locale,
        "timezone": user.timezone,
        "theme": user.theme or "light",
        "notify_email_digests": bool(user.notify_email_digests),
        "notify_session_reminders": bool(user.notify_session_reminders),
        "mic_default": user.mic_default,
        "camera_default": user.camera_default,
        "use_elevenlabs": bool(user.use_elevenlabs),
        "elevenlabs_voice_id": user.elevenlabs_voice_id,
        "profile_completed": bool(user.profile_completed),
        "created_at": user.created_at.isoformat() if user.created_at else None,
    }


@router.post("/register", response_model=TokenOut)
def register(body: RegisterBody, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == body.email.strip().lower()).first()
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")
    user = User(
        name=body.name.strip(),
        email=body.email.strip().lower(),
        password_hash=pwd_context.hash(body.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    token = create_access_token({"user_id": user.id, "email": user.email})
    return TokenOut(
        access_token=token,
        user_id=user.id,
        name=user.name,
        email=user.email,
    )


@router.post("/login", response_model=TokenOut)
def login(body: LoginBody, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == body.email.strip().lower()).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    if not user.password_hash:
        raise HTTPException(status_code=401, detail="Account has no password set. Please register again.")
    if not pwd_context.verify(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_access_token({"user_id": user.id, "email": user.email})
    return TokenOut(
        access_token=token,
        user_id=user.id,
        name=user.name,
        email=user.email,
    )


@router.get("/me", response_model=UserOut)
def me(current_user: User = Depends(get_current_user)):
    return UserOut(
        user_id=current_user.id,
        name=current_user.name,
        email=current_user.email,
    )


# ── Profile ──


@router.get("/profile")
def get_profile(current_user: User = Depends(get_current_user)):
    return _profile_to_dict(current_user)


@router.put("/profile")
async def update_profile(
    display_name: str = Form(None),
    phone: str = Form(None),
    bio: str = Form(None),
    target_role: str = Form(None),
    target_industry: str = Form(None),
    seniority_level: str = Form(None),
    years_of_experience: int = Form(None),
    current_company: str = Form(None),
    education_level: str = Form(None),
    linkedin_url: str = Form(None),
    focus_areas: str = Form(None),
    upcoming_interview_date: str = Form(None),
    preferred_difficulty: str = Form(None),
    locale: str = Form(None),
    timezone: str = Form(None),
    theme: str = Form(None),
    notify_email_digests: bool = Form(None),
    notify_session_reminders: bool = Form(None),
    mic_default: str = Form(None),
    camera_default: str = Form(None),
    profile_completed: bool = Form(None),
    use_elevenlabs: bool = Form(None),
    elevenlabs_voice_id: str = Form(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    user = db.get(User, current_user.id)

    fields = {
        "display_name": display_name,
        "phone": phone,
        "bio": bio,
        "target_role": target_role,
        "target_industry": target_industry,
        "seniority_level": seniority_level,
        "years_of_experience": years_of_experience,
        "current_company": current_company,
        "education_level": education_level,
        "linkedin_url": linkedin_url,
        "preferred_difficulty": preferred_difficulty,
        "locale": locale,
        "timezone": timezone,
        "theme": theme,
        "mic_default": mic_default,
        "camera_default": camera_default,
        "elevenlabs_voice_id": elevenlabs_voice_id,
    }
    for key, val in fields.items():
        if val is not None:
            setattr(user, key, val)

    if focus_areas is not None:
        try:
            parsed = json.loads(focus_areas) if isinstance(focus_areas, str) else focus_areas
            user.focus_areas = json.dumps(parsed) if isinstance(parsed, list) else focus_areas
        except (json.JSONDecodeError, TypeError):
            user.focus_areas = focus_areas

    if upcoming_interview_date is not None:
        try:
            user.upcoming_interview_date = datetime.fromisoformat(upcoming_interview_date)
        except (ValueError, TypeError):
            pass

    if notify_email_digests is not None:
        user.notify_email_digests = notify_email_digests
    if notify_session_reminders is not None:
        user.notify_session_reminders = notify_session_reminders
    if profile_completed is not None:
        user.profile_completed = profile_completed
    if use_elevenlabs is not None:
        user.use_elevenlabs = use_elevenlabs

    db.commit()
    db.refresh(user)
    return _profile_to_dict(user)


@router.post("/upload-avatar")
async def upload_avatar(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ext = os.path.splitext(file.filename or "avatar.jpg")[1] or ".jpg"
    filename = f"avatar_{current_user.id}_{uuid.uuid4().hex[:8]}{ext}"
    filepath = UPLOAD_DIR / filename

    content = await file.read()
    with open(filepath, "wb") as f:
        f.write(content)

    user = db.get(User, current_user.id)
    old = user.avatar_path
    user.avatar_path = str(filepath)
    db.commit()

    if old and os.path.exists(old):
        try:
            os.remove(old)
        except OSError:
            pass

    return {"avatar_url": f"/uploads/avatars/{filename}"}


@router.post("/upload-resume")
async def upload_resume(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    content = await file.read()
    text = extract_text_from_pdf(content)

    if not text:
        raise HTTPException(status_code=400, detail="Could not extract text from PDF")

    user = db.get(User, current_user.id)
    user.resume_text = text[:10000]
    db.commit()

    return {"resume_text": user.resume_text}


@router.post("/change-password")
def change_password(
    body: ChangePasswordBody,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    user = db.get(User, current_user.id)
    if not pwd_context.verify(body.old_password, user.password_hash):
        raise HTTPException(status_code=400, detail="Current password is incorrect")
    user.password_hash = pwd_context.hash(body.new_password)
    db.commit()
    return {"message": "Password changed successfully"}


@router.post("/delete-account")
def delete_account(
    body: DeleteAccountBody,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    user = db.get(User, current_user.id)
    if not pwd_context.verify(body.password, user.password_hash):
        raise HTTPException(status_code=400, detail="Password is incorrect")
    db.delete(user)
    db.commit()
    return {"message": "Account deleted permanently"}
