import json
from datetime import datetime
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    password_hash: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    # Identity
    display_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    phone: Mapped[str | None] = mapped_column(String(50), nullable=True)
    avatar_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    bio: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Career context
    target_role: Mapped[str | None] = mapped_column(String(255), nullable=True)
    target_industry: Mapped[str | None] = mapped_column(String(128), nullable=True)
    seniority_level: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Background & experience
    years_of_experience: Mapped[int | None] = mapped_column(Integer, nullable=True)
    current_company: Mapped[str | None] = mapped_column(String(255), nullable=True)
    education_level: Mapped[str | None] = mapped_column(String(64), nullable=True)
    linkedin_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    resume_text: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Goals & focus
    focus_areas: Mapped[str | None] = mapped_column(Text, nullable=True)
    upcoming_interview_date: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    preferred_difficulty: Mapped[str | None] = mapped_column(String(32), nullable=True)

    # Preferences
    locale: Mapped[str | None] = mapped_column(String(16), nullable=True)
    timezone: Mapped[str | None] = mapped_column(String(64), nullable=True)
    theme: Mapped[str] = mapped_column(String(16), default="light")
    notify_email_digests: Mapped[bool] = mapped_column(Boolean, default=True)
    notify_session_reminders: Mapped[bool] = mapped_column(Boolean, default=True)
    mic_default: Mapped[str | None] = mapped_column(String(64), nullable=True)
    camera_default: Mapped[str | None] = mapped_column(String(64), nullable=True)
    use_elevenlabs: Mapped[bool] = mapped_column(Boolean, default=False)
    elevenlabs_voice_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Auth & meta
    profile_completed: Mapped[bool] = mapped_column(Boolean, default=False)
    refresh_token: Mapped[str | None] = mapped_column(String(512), nullable=True)

    sessions: Mapped[list["InterviewSession"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class InterviewSession(Base):
    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    session_number: Mapped[int] = mapped_column(Integer, default=0)
    session_type: Mapped[str] = mapped_column(String(16), default="practice")
    role: Mapped[str] = mapped_column(String(128), default="general")
    topic: Mapped[str] = mapped_column(String(128), default="general")
    overall_score: Mapped[float] = mapped_column(Float, default=0)
    avg_confidence: Mapped[float] = mapped_column(Float, default=0)
    started_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    user: Mapped["User"] = relationship(back_populates="sessions")
    recordings: Mapped[list["Recording"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )


class Recording(Base):
    __tablename__ = "recordings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
    )
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    transcript: Mapped[str] = mapped_column(Text, default="")
    feedback: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    session: Mapped["InterviewSession"] = relationship(back_populates="recordings")
    word_analyses: Mapped[list["WordAnalysis"]] = relationship(
        back_populates="recording", cascade="all, delete-orphan"
    )
    metrics: Mapped[list["RecordingMetric"]] = relationship(
        back_populates="recording", cascade="all, delete-orphan"
    )


class WordAnalysis(Base):
    __tablename__ = "word_analysis"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    recording_id: Mapped[int] = mapped_column(
        ForeignKey("recordings.id", ondelete="CASCADE"), nullable=False
    )
    word: Mapped[str] = mapped_column(String(255), nullable=False)
    issue: Mapped[str] = mapped_column(String(128), nullable=False)
    suggestion: Mapped[str] = mapped_column(Text, default="")
    timestamp: Mapped[str] = mapped_column(String(32), nullable=False)

    recording: Mapped["Recording"] = relationship(back_populates="word_analyses")


class UserResume(Base):
    __tablename__ = "user_resumes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), default="Untitled Resume")
    file_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    raw_text: Mapped[str] = mapped_column(Text, default="")
    skills: Mapped[str] = mapped_column(Text, default="[]")
    experience_years: Mapped[int | None] = mapped_column(Integer, nullable=True)
    education: Mapped[str] = mapped_column(Text, default="[]")
    is_primary: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class RecordingMetric(Base):
    __tablename__ = "recording_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    recording_id: Mapped[int] = mapped_column(
        ForeignKey("recordings.id", ondelete="CASCADE"), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(String(128), default="general", index=True)
    topic: Mapped[str] = mapped_column(String(128), default="general", index=True)
    fluency: Mapped[float] = mapped_column(Float, default=0)
    confidence: Mapped[float] = mapped_column(Float, default=0)
    composure: Mapped[float] = mapped_column(Float, default=0)
    overall: Mapped[float] = mapped_column(Float, default=0, index=True)
    sentiment: Mapped[str] = mapped_column(String(32), default="neutral")
    star_score: Mapped[float] = mapped_column(Float, default=0)
    coherence_score: Mapped[float] = mapped_column(Float, default=0)
    content_score: Mapped[float] = mapped_column(Float, default=0)
    question_relevance: Mapped[float] = mapped_column(Float, default=0)
    resume_score: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), index=True)

    recording: Mapped["Recording"] = relationship(back_populates="metrics")


class GuidedInterview(Base):
    __tablename__ = "guided_interviews"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    aim: Mapped[str] = mapped_column(Text, default="")
    target_company: Mapped[str] = mapped_column(String(255), default="")
    duration_minutes: Mapped[int] = mapped_column(Integer, default=30)
    difficulty: Mapped[str] = mapped_column(String(32), default="intermediate")
    focus_areas: Mapped[str] = mapped_column(Text, default="[]")
    status: Mapped[str] = mapped_column(String(32), default="setup")
    started_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    overall_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    mode: Mapped[str] = mapped_column(String(16), default="text")
    recording_count: Mapped[int] = mapped_column(Integer, default=0)
    clarification_answers: Mapped[str | None] = mapped_column(Text, nullable=True)
    clarifying_questions: Mapped[str | None] = mapped_column(Text, nullable=True)

    qa_pairs: Mapped[list["InterviewQA"]] = relationship(
        back_populates="interview", cascade="all, delete-orphan"
    )

    def get_focus_areas(self) -> list:
        try:
            return json.loads(self.focus_areas) if self.focus_areas else []
        except (json.JSONDecodeError, TypeError):
            return []


class InterviewQA(Base):
    __tablename__ = "interview_qa"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    interview_id: Mapped[int] = mapped_column(
        ForeignKey("guided_interviews.id", ondelete="CASCADE")
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    recording_id: Mapped[int | None] = mapped_column(
        ForeignKey("recordings.id", ondelete="SET NULL"), nullable=True
    )
    transcript: Mapped[str] = mapped_column(Text, default="")
    feedback: Mapped[str] = mapped_column(Text, default="")
    content_score: Mapped[float] = mapped_column(Float, default=0)
    relevance_score: Mapped[float] = mapped_column(Float, default=0)
    fluency_score: Mapped[float] = mapped_column(Float, default=0)
    confidence_score: Mapped[float] = mapped_column(Float, default=0)
    order: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    interview: Mapped["GuidedInterview"] = relationship(back_populates="qa_pairs")
