import httpx
from fastapi import HTTPException

from app.config import ELEVENLABS_API_KEY

BASE_URL = "https://api.elevenlabs.io/v1"

HEADERS = {
    "xi-api-key": ELEVENLABS_API_KEY,
    "Content-Type": "application/json",
}


async def list_voices():
    if not ELEVENLABS_API_KEY:
        raise HTTPException(status_code=400, detail="ElevenLabs API key not configured")

    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{BASE_URL}/voices", headers=HEADERS)

    if resp.status_code != 200:
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"ElevenLabs API error: {resp.text}",
        )

    data = resp.json()
    voices = data.get("voices", [])
    return [
        {
            "voice_id": v["voice_id"],
            "name": v["name"],
            "category": v.get("category", "generated"),
            "description": v.get("description", ""),
            "preview_url": v.get("preview_url", ""),
            "labels": v.get("labels", {}),
        }
        for v in voices
    ]


async def text_to_speech(text: str, voice_id: str):
    if not ELEVENLABS_API_KEY:
        raise HTTPException(status_code=400, detail="ElevenLabs API key not configured")

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{BASE_URL}/text-to-speech/{voice_id}/stream",
            headers={
                "xi-api-key": ELEVENLABS_API_KEY,
                "Content-Type": "application/json",
                "Accept": "audio/mpeg",
            },
            json={
                "text": text,
                "model_id": "eleven_multilingual_v2",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.75,
                },
            },
        )

    if resp.status_code != 200:
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"ElevenLabs TTS error: {resp.text}",
        )

    return resp.content
