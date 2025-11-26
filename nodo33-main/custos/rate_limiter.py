"""
Token bucket rate limiter per pattern/chiave.
Restituisce se la richiesta può procedere e un eventuale tempo di attesa.
"""

from __future__ import annotations

import time
from typing import Dict, Tuple


class TokenBucketLimiter:
    def __init__(self) -> None:
        # key -> (tokens, last_timestamp)
        self._state: Dict[str, Tuple[float, float]] = {}

    def acquire(self, key: str, rpm: int, burst: int, block: bool = False) -> Tuple[bool, float]:
        """
        rpm: richieste per minuto
        burst: capacità massima del bucket
        block: se True, attende il tempo necessario e poi concede il token
        Ritorna (allowed, wait_seconds_needed_if_block_false)
        """
        now = time.monotonic()
        rate_per_sec = rpm / 60.0 if rpm > 0 else 0.0

        tokens, last = self._state.get(key, (float(burst), now))
        elapsed = max(0.0, now - last)
        tokens = min(float(burst), tokens + elapsed * rate_per_sec)

        if tokens >= 1.0:
            tokens -= 1.0
            self._state[key] = (tokens, now)
            return True, 0.0

        if rate_per_sec <= 0:
            self._state[key] = (tokens, now)
            return False, float("inf")

        wait = (1.0 - tokens) / rate_per_sec
        if block:
            time.sleep(wait)
            return self.acquire(key, rpm, burst, block=False)

        self._state[key] = (tokens, now)
        return False, wait
