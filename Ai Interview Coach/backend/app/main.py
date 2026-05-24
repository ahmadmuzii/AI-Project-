from contextlib import asynccontextmanager

import asyncio
import whisper
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import app.config  # loads .env before anything else
from app.config import CORS_ORIGINS_CSV
from app.database import Base, engine
from app.routes import analytics, audio, guided_interview, interview, auth, resume, elevenlabs

@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)

    import sqlalchemy as sa

    # Create migration tracking table
    with engine.connect() as conn:
        conn.execute(sa.text(
            "CREATE TABLE IF NOT EXISTS _migrations (name VARCHAR(255) PRIMARY KEY, applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        ))
        conn.commit()

    inspector = sa.inspect(engine)
    migrations = [
        ("guided_interviews", "mode", "VARCHAR(16)", "'text'", "v1_guided_mode"),
        ("guided_interviews", "clarification_answers", "TEXT", "NULL", "v1_clarification_answers"),
        ("guided_interviews", "clarifying_questions", "TEXT", "NULL", "v1_clarifying_questions"),
        ("users", "use_elevenlabs", "BOOLEAN", "0", "v1_use_elevenlabs"),
        ("users", "elevenlabs_voice_id", "VARCHAR(64)", "NULL", "v1_elevenlabs_voice_id"),
    ]
    for table, column, col_type, default, name in migrations:
        try:
            with engine.connect() as conn:
                already = conn.execute(sa.text("SELECT 1 FROM _migrations WHERE name = :n"), {"n": name}).fetchone()
                if already:
                    continue
                cols = [c["name"] for c in inspector.get_columns(table)]
                if column not in cols:
                    conn.execute(sa.text(f"ALTER TABLE {table} ADD COLUMN {column} {col_type} DEFAULT {default}"))
                conn.execute(sa.text("INSERT INTO _migrations (name) VALUES (:n)"), {"n": name})
                conn.commit()
                print(f"Migration: `{name}` applied [OK]")
        except Exception as e:
            print(f"Migration skipped ({name}): {e}")

    print("Database tables ready [OK]")

    model_name = "tiny"
    app.state.whisper_model = await asyncio.to_thread(
        whisper.load_model, model_name
    )
    print("Whisper model loaded [OK]")

    yield

    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS_CSV.split(",") if CORS_ORIGINS_CSV != "*" else ["*"],
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