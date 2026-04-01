from datetime import datetime
from pydantic import BaseModel, Field


class CreateUserBody(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    email: str = Field(..., min_length=3, max_length=255)


class StartSessionBody(BaseModel):
    user_id: int = Field(..., ge=1)


class WordAnalysisOut(BaseModel):
    timestamp: str
    word: str
    issue: str
    suggestion: str

    class Config:
        from_attributes = True


class RecordingOut(BaseModel):
    id: int
    session_id: int
    file_path: str
    transcript: str
    feedback: str
    created_at: datetime
    word_analysis: list[WordAnalysisOut]

    class Config:
        from_attributes = True


class SessionOut(BaseModel):
    id: int
    user_id: int
    started_at: datetime

    class Config:
        from_attributes = True
