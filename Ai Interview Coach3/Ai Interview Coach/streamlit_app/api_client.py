import os
from typing import Any

import requests

DEFAULT_API = os.environ.get("INTERVIEW_API_URL", "http://127.0.0.1:8000")


def _url(path: str) -> str:
    base = DEFAULT_API.rstrip("/")
    return f"{base}{path}"


def create_user(name: str, email: str) -> dict[str, Any]:
    r = requests.post(
        _url("/create-user"),
        json={"name": name, "email": email},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def start_session(user_id: int) -> dict[str, Any]:
    r = requests.post(
        _url("/start-session"),
        json={"user_id": user_id},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def list_sessions(user_id: int) -> list[dict[str, Any]]:
    r = requests.get(_url(f"/sessions/{user_id}"), timeout=30)
    r.raise_for_status()
    return r.json()


def list_recordings(session_id: int) -> dict[str, Any]:
    r = requests.get(_url(f"/recordings/{session_id}"), timeout=30)
    r.raise_for_status()
    return r.json()


def get_recording(recording_id: int) -> dict[str, Any]:
    r = requests.get(_url(f"/recording/{recording_id}"), timeout=30)
    r.raise_for_status()
    return r.json()


def upload_audio(
    user_id: int,
    session_id: int,
    file_bytes: bytes,
    filename: str,
    content_type: str | None = None,
) -> dict[str, Any]:
    ct = content_type or "application/octet-stream"
    files = {"file": (filename, file_bytes, ct)}
    data = {"user_id": str(user_id), "session_id": str(session_id)}
    r = requests.post(_url("/upload-audio"), files=files, data=data, timeout=600)
    r.raise_for_status()
    return r.json()


def recording_audio_url(recording_id: int) -> str:
    return _url(f"/recording/{recording_id}/audio")
