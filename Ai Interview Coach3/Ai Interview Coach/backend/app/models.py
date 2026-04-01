from datetime import datetime
from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    sessions: Mapped[list["InterviewSession"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class InterviewSession(Base):
    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
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
