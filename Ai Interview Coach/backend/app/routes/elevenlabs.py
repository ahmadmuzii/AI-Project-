from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response

from app.database import get_db
from app.routes.auth import get_current_user
from app.models import User
from app.services.elevenlabs_service import list_voices, text_to_speech

router = APIRouter(prefix="/api/elevenlabs", tags=["elevenlabs"])


@router.get("/voices")
async def get_voices(
    current_user: User = Depends(get_current_user),
):
    return await list_voices()


@router.get("/tts")
async def get_tts(
    text: str = Query(..., min_length=1, max_length=2000),
    voice_id: str = Query(..., min_length=1),
    current_user: User = Depends(get_current_user),
):
    audio_bytes = await text_to_speech(text, voice_id)
    return Response(content=audio_bytes, media_type="audio/mpeg")
