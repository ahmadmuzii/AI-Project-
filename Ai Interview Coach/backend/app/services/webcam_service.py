from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

_HAS_MEDIAPIPE_SOLUTIONS = False
try:
    import mediapipe as mp

    if hasattr(mp, "solutions"):
        mp_face = mp.solutions.face_mesh
        mp_hands = mp.solutions.hands
        _HAS_MEDIAPIPE_SOLUTIONS = True
except Exception:
    _HAS_MEDIAPIPE_SOLUTIONS = False

# Light-weight global analyzers (good enough for single-process local demo).
_FACE_MESH = None
_HANDS = None
if _HAS_MEDIAPIPE_SOLUTIONS:
    _FACE_MESH = mp_face.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    _HANDS = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

_FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# In-memory state for temporal movement estimation.
_PREV_STATE: dict[str, dict[str, float]] = {}


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _decode_image(image_bytes: bytes) -> np.ndarray | None:
    arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame


def _eye_aspect_ratio(face_landmarks, width: int, height: int, left: bool) -> float:
    # Approximate EAR with MediaPipe FaceMesh ids.
    if left:
        p1, p2, p3, p4, p5, p6 = 33, 160, 158, 133, 153, 144
    else:
        p1, p2, p3, p4, p5, p6 = 263, 385, 387, 362, 373, 380

    lm = face_landmarks.landmark
    a = (lm[p1].x * width, lm[p1].y * height)
    b = (lm[p2].x * width, lm[p2].y * height)
    c = (lm[p3].x * width, lm[p3].y * height)
    d = (lm[p4].x * width, lm[p4].y * height)
    e = (lm[p5].x * width, lm[p5].y * height)
    f = (lm[p6].x * width, lm[p6].y * height)

    vertical = (_dist(b, e) + _dist(c, f)) / 2.0
    horizontal = max(1e-6, _dist(a, d))
    return vertical / horizontal


def analyze_webcam_frame(image_bytes: bytes, stream_id: str = "default") -> dict[str, Any]:
    frame = _decode_image(image_bytes)
    if frame is None:
        return {
            "ok": False,
            "error": "Invalid image data",
        }

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_result = _FACE_MESH.process(rgb) if _FACE_MESH is not None else None
    hands_result = _HANDS.process(rgb) if _HANDS is not None else None

    has_face = bool(face_result and face_result.multi_face_landmarks)
    hand_count = len(hands_result.multi_hand_landmarks) if (hands_result and hands_result.multi_hand_landmarks) else 0

    eye_contact_score = 0.2
    movement_score = 0.5
    confidence_label = "low"

    curr_state = {"nose_x": 0.0, "nose_y": 0.0, "face_center_x": 0.0, "face_center_y": 0.0}
    if has_face:
        face = face_result.multi_face_landmarks[0]
        lm = face.landmark

        nose = lm[1]
        left_cheek = lm[234]
        right_cheek = lm[454]
        forehead = lm[10]
        chin = lm[152]

        center_x = (left_cheek.x + right_cheek.x) / 2.0
        center_y = (forehead.y + chin.y) / 2.0
        yaw_offset = abs(nose.x - center_x)
        pitch_offset = abs(nose.y - center_y)

        # Higher when face is centered toward camera.
        eye_contact_score = _clamp(1.0 - (yaw_offset * 3.5 + pitch_offset * 2.5))

        # Eye openness contributes a bit (blink-heavy/closed eyes reduce perceived contact).
        left_ear = _eye_aspect_ratio(face, w, h, left=True)
        right_ear = _eye_aspect_ratio(face, w, h, left=False)
        eye_open_score = _clamp(((left_ear + right_ear) / 2.0 - 0.12) / 0.18)
        eye_contact_score = _clamp(0.75 * eye_contact_score + 0.25 * eye_open_score)

        curr_state = {
            "nose_x": nose.x,
            "nose_y": nose.y,
            "face_center_x": center_x,
            "face_center_y": center_y,
        }
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = _FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4, minSize=(60, 60))
        if len(faces) > 0:
            has_face = True
            x, y, fw, fh = max(faces, key=lambda bb: bb[2] * bb[3])
            center_x = (x + fw / 2) / w
            center_y = (y + fh / 2) / h
            cx_off = abs(center_x - 0.5)
            cy_off = abs(center_y - 0.5)
            eye_contact_score = _clamp(1.0 - (cx_off * 2.0 + cy_off * 1.5))
            curr_state = {
                "nose_x": center_x,
                "nose_y": center_y,
                "face_center_x": center_x,
                "face_center_y": center_y,
            }

    prev = _PREV_STATE.get(stream_id)
    if prev and has_face:
        dx = abs(curr_state["nose_x"] - prev["nose_x"])
        dy = abs(curr_state["nose_y"] - prev["nose_y"])
        motion = dx + dy
        # Less abrupt movement => higher stability score.
        movement_score = _clamp(1.0 - motion * 8.0)
    elif has_face:
        movement_score = 0.75

    # Hand visibility can raise confidence signal in mock interview communication.
    hand_signal = 0.15 if hand_count > 0 else 0.0
    confidence_score = _clamp(0.55 * eye_contact_score + 0.35 * movement_score + hand_signal)
    if confidence_score > 0.7:
        confidence_label = "high"
    elif confidence_score > 0.45:
        confidence_label = "moderate"

    _PREV_STATE[stream_id] = curr_state

    return {
        "ok": True,
        "engine": "mediapipe" if _HAS_MEDIAPIPE_SOLUTIONS else "opencv-fallback",
        "has_face": has_face,
        "hand_count": hand_count,
        "eye_contact_score": round(float(eye_contact_score), 3),
        "movement_score": round(float(movement_score), 3),
        "confidence_score": round(float(confidence_score), 3),
        "confidence_label": confidence_label,
    }
