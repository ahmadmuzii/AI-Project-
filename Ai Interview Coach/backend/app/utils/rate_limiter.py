import time
from collections import defaultdict

from fastapi import HTTPException

_attempts: dict[str, list[float]] = defaultdict(list)
LIMIT = 10
WINDOW = 60


def check_rate_limit(key: str) -> None:
    now = time.time()
    window_start = now - WINDOW
    _attempts[key] = [t for t in _attempts[key] if t > window_start]
    if len(_attempts[key]) >= LIMIT:
        raise HTTPException(status_code=429, detail="Too many requests. Try again later.")
    _attempts[key].append(now)
