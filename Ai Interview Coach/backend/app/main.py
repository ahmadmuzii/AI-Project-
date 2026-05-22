from contextlib import asynccontextmanager

import asyncio
import whisper
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import app.config  # loads .env before anything else
from app.database import Base, engine
from app.routes import analytics, audio, guided_interview, interview, auth, resume, elevenlabs

@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)

    # Migration: add missing columns for existing tables
    # NOTE: inspector must be created AFTER create_all so new tables are visible
    import sqlalchemy as sa
    inspector = sa.inspect(engine)
    migrations = [
        ("guided_interviews", "mode", "VARCHAR(16)", "'text'"),
        ("guided_interviews", "clarification_answers", "TEXT", "NULL"),
        ("guided_interviews", "clarifying_questions", "TEXT", "NULL"),
        ("users", "use_elevenlabs", "BOOLEAN", "0"),
        ("users", "elevenlabs_voice_id", "VARCHAR(64)", "NULL"),
    ]
    for table, column, col_type, default in migrations:
        try:
            cols = [c["name"] for c in inspector.get_columns(table)]
            if column not in cols:
                with engine.connect() as conn:
                    conn.execute(sa.text(f"ALTER TABLE {table} ADD COLUMN {column} {col_type} DEFAULT {default}"))
                    conn.commit()
                    print(f"Migration: added `{column}` to {table} \u2705")
        except Exception as e:
            print(f"Migration skipped ({table}.{column}): {e}")

    print("Database tables ready ✅")

    model_name = "tiny"
    app.state.whisper_model = await asyncio.to_thread(
        whisper.load_model, model_name
    )
    print("Whisper model loaded ✅")

    yield

    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

app.include_router(auth.router)
app.include_router(interview.router)
app.include_router(audio.router)
app.include_router(analytics.router)
app.include_router(resume.router)
app.include_router(guided_interview.router)
app.include_router(elevenlabs.router)

@app.get("/")
def root():
    return {"message": "AI Interview Coach Running. Frontend runs on localhost:3000."}