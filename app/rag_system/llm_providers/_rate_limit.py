"""Client-side rate limiter (token bucket over a sliding window).

Used to keep us comfortably under Gemini free-tier per-minute limits so we
don't burst into 429s and crash the ingest. Thread-safe.

Usage:
    limiter = RateLimiter(max_calls=80, window_s=60)
    limiter.acquire(n=1)   # blocks until a slot is free
    call_the_api()
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import deque

log = logging.getLogger(__name__)


class RateLimiter:
    """Token-bucket over a sliding time window."""

    def __init__(self, max_calls: int, window_s: float, *, name: str = "ratelimit"):
        self.max_calls = max_calls
        self.window_s = window_s
        self.name = name
        self._times: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self, n: int = 1) -> None:
        """Block until `n` calls fit inside the window, then record them.

        If `n` exceeds max_calls in a single bucket, we acquire in smaller
        chunks recursively — caller's intent is honored without crashing.
        """
        if n > self.max_calls:
            # Split into chunks of max_calls; each will wait its own window
            remaining = n
            while remaining > 0:
                take = min(remaining, self.max_calls)
                self.acquire(take)
                remaining -= take
            return
        while True:
            with self._lock:
                now = time.monotonic()
                cutoff = now - self.window_s
                while self._times and self._times[0] < cutoff:
                    self._times.popleft()
                if len(self._times) + n <= self.max_calls:
                    for _ in range(n):
                        self._times.append(now)
                    return
                # Need to wait for the oldest tokens to expire enough
                need_to_drop = (len(self._times) + n) - self.max_calls
                target = self._times[need_to_drop - 1]
                wait_s = max(0.1, (target + self.window_s) - now + 0.1)
            log.info(
                "%s: rate limit reached (%d/%d in last %.0fs), sleeping %.1fs",
                self.name, len(self._times), self.max_calls, self.window_s, wait_s,
            )
            time.sleep(wait_s)


# ---------------------------------------------------------------------------
# 429 helpers
# ---------------------------------------------------------------------------
_RETRY_RE = re.compile(r"['\"]retryDelay['\"]\s*:\s*['\"](\d+)(?:\.\d+)?s['\"]")


def parse_retry_after_seconds(err_msg: str, *, default: float = 20.0) -> float:
    """Pull the suggested wait time out of a Gemini 429 error message."""
    m = _RETRY_RE.search(err_msg)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return default


def is_rate_limit_error(err: Exception) -> bool:
    msg = str(err)
    return (
        "429" in msg
        or "RESOURCE_EXHAUSTED" in msg
        or "quota" in msg.lower()
        or "rate" in msg.lower() and "limit" in msg.lower()
    )
