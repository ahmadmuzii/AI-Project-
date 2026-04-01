from contextlib import asynccontextmanager

import asyncio
import whisper
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.database import Base, engine
from app.routes import audio, interview

@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    print("Database tables ready ✅")

    model_name = "small"
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

app.include_router(interview.router)
app.include_router(audio.router)

# IMPORTANT: DO NOT mount at "/"
app.mount("/frontend", StaticFiles(directory="frontend", html=True), name="frontend")

@app.get("/")
def root():
    return {"message": "AI Interview Coach Running"}