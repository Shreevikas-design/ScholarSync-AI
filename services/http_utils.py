from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

_RATE_LOCK = threading.Lock()
_NEXT_REQUEST: dict[str, float] = {}


def wait_for_host(host: str, min_interval: float) -> None:
    """Process-wide request pacing shared by concurrent user sessions."""
    with _RATE_LOCK:
        now = time.monotonic()
        next_at = _NEXT_REQUEST.get(host, now)
        delay = max(0.0, next_at - now)
        _NEXT_REQUEST[host] = max(now, next_at) + min_interval
    if delay:
        time.sleep(delay)


def retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value.strip()))
    except (TypeError, ValueError):
        try:
            target = parsedate_to_datetime(value)
            if target.tzinfo is None:
                target = target.replace(tzinfo=timezone.utc)
            return max(0.0, (target - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None
