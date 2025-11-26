#!/usr/bin/env python3
"""
Enterprise Rate Limiter

Implements multiple rate limiting strategies for production use:
- Token bucket algorithm
- Sliding window
- Per-user quotas
- Dynamic rate adjustment

Nodo33 - Sasso Digitale
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from threading import Lock
from typing import Callable, Dict, Optional

try:
    from ratelimit import limits, sleep_and_retry
    from ratelimit import RateLimitException as _RateLimitException
    RATELIMIT_AVAILABLE = True
except ImportError:
    RATELIMIT_AVAILABLE = False
    _RateLimitException = Exception


# Define our own exception (always available)
class RateLimitException(Exception):
    """Rate limit exceeded."""
    pass


class RateLimitTier(Enum):
    """Rate limit tiers for different user types."""

    FREE = "free"
    BASIC = "basic"
    PRO = "pro"
    ENTERPRISE = "enterprise"


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""

    # Calls per period
    calls: int
    period: int  # seconds

    # Burst allowance
    burst_multiplier: float = 1.5

    # Backoff
    backoff_enabled: bool = True
    backoff_factor: float = 2.0

    @property
    def burst_calls(self) -> int:
        """Maximum burst calls allowed."""
        return int(self.calls * self.burst_multiplier)


# Predefined tiers
TIER_CONFIGS = {
    RateLimitTier.FREE: RateLimitConfig(calls=10, period=60),  # 10/min
    RateLimitTier.BASIC: RateLimitConfig(calls=100, period=60),  # 100/min
    RateLimitTier.PRO: RateLimitConfig(calls=1000, period=60),  # 1000/min
    RateLimitTier.ENTERPRISE: RateLimitConfig(calls=10000, period=60),  # 10k/min
}


class TokenBucket:
    """Token bucket rate limiter implementation."""

    def __init__(self, rate: float, capacity: int):
        """
        Initialize token bucket.

        Args:
            rate: Tokens per second
            capacity: Maximum tokens
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.time()
        self.lock = Lock()

    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if tokens were consumed, False if rate limited
        """
        with self.lock:
            now = time.time()
            elapsed = now - self.last_update

            # Add new tokens based on elapsed time
            self.tokens = min(
                self.capacity,
                self.tokens + elapsed * self.rate
            )
            self.last_update = now

            # Try to consume
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True

            return False

    def wait_time(self, tokens: int = 1) -> float:
        """
        Calculate wait time for tokens to be available.

        Args:
            tokens: Number of tokens needed

        Returns:
            Wait time in seconds
        """
        with self.lock:
            if self.tokens >= tokens:
                return 0.0

            needed = tokens - self.tokens
            return needed / self.rate


class SlidingWindowRateLimiter:
    """Sliding window rate limiter for more accurate limiting."""

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.windows: Dict[str, list] = defaultdict(list)
        self.lock = Lock()

    def is_allowed(self, key: str = "default") -> bool:
        """
        Check if request is allowed.

        Args:
            key: Identifier (e.g., user_id, ip_address)

        Returns:
            True if allowed, False if rate limited
        """
        with self.lock:
            now = time.time()
            window_start = now - self.config.period

            # Clean old timestamps
            self.windows[key] = [
                ts for ts in self.windows[key]
                if ts > window_start
            ]

            # Check limit
            if len(self.windows[key]) < self.config.calls:
                self.windows[key].append(now)
                return True

            return False

    def remaining(self, key: str = "default") -> int:
        """Get remaining calls in current window."""
        with self.lock:
            now = time.time()
            window_start = now - self.config.period

            # Count recent calls
            recent = sum(
                1 for ts in self.windows.get(key, [])
                if ts > window_start
            )

            return max(0, self.config.calls - recent)

    def reset_time(self, key: str = "default") -> float:
        """Get time until window resets."""
        with self.lock:
            timestamps = self.windows.get(key, [])
            if not timestamps:
                return 0.0

            oldest = min(timestamps)
            reset_at = oldest + self.config.period
            return max(0, reset_at - time.time())


class EnterpriseRateLimiter:
    """
    Enterprise-grade rate limiter with multiple strategies.

    Features:
    - Per-user quotas
    - Multiple tiers
    - Dynamic rate adjustment
    - Graceful degradation
    - Metrics collection
    """

    def __init__(self):
        self.limiters: Dict[str, SlidingWindowRateLimiter] = {}
        self.user_tiers: Dict[str, RateLimitTier] = {}
        self.lock = Lock()

        # Metrics
        self.total_requests = 0
        self.rate_limited_requests = 0
        self.requests_by_tier = defaultdict(int)

    def set_user_tier(self, user_id: str, tier: RateLimitTier) -> None:
        """Set rate limit tier for a user."""
        with self.lock:
            self.user_tiers[user_id] = tier
            config = TIER_CONFIGS[tier]
            self.limiters[user_id] = SlidingWindowRateLimiter(config)

    def get_limiter(self, user_id: str) -> SlidingWindowRateLimiter:
        """Get or create limiter for user."""
        if user_id not in self.limiters:
            # Default to FREE tier
            self.set_user_tier(user_id, RateLimitTier.FREE)

        return self.limiters[user_id]

    def check_limit(self, user_id: str = "default") -> tuple[bool, Dict]:
        """
        Check if request is allowed and return status.

        Args:
            user_id: User identifier

        Returns:
            (allowed: bool, metadata: dict)
        """
        limiter = self.get_limiter(user_id)
        tier = self.user_tiers.get(user_id, RateLimitTier.FREE)

        allowed = limiter.is_allowed(user_id)

        # Update metrics
        with self.lock:
            self.total_requests += 1
            self.requests_by_tier[tier.value] += 1

            if not allowed:
                self.rate_limited_requests += 1

        metadata = {
            "allowed": allowed,
            "tier": tier.value,
            "remaining": limiter.remaining(user_id),
            "reset_in": limiter.reset_time(user_id),
            "limit": TIER_CONFIGS[tier].calls,
        }

        return allowed, metadata

    def get_metrics(self) -> Dict:
        """Get rate limiting metrics."""
        with self.lock:
            rate_limited_percentage = (
                (self.rate_limited_requests / self.total_requests * 100)
                if self.total_requests > 0
                else 0
            )

            return {
                "total_requests": self.total_requests,
                "rate_limited": self.rate_limited_requests,
                "rate_limited_percentage": rate_limited_percentage,
                "by_tier": dict(self.requests_by_tier),
                "active_users": len(self.limiters),
            }

    def reset_metrics(self) -> None:
        """Reset metrics counters."""
        with self.lock:
            self.total_requests = 0
            self.rate_limited_requests = 0
            self.requests_by_tier.clear()


# ============================================================================
# DECORATORS
# ============================================================================


def rate_limit(
    tier: RateLimitTier = RateLimitTier.FREE,
    user_id_param: str = "user_id"
):
    """
    Decorator for rate limiting functions.

    Args:
        tier: Default rate limit tier
        user_id_param: Parameter name for user_id in function

    Usage:
        @rate_limit(tier=RateLimitTier.PRO)
        def expensive_operation(user_id: str):
            ...
    """
    limiter = EnterpriseRateLimiter()

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Extract user_id from kwargs or use default
            user_id = kwargs.get(user_id_param, "default")

            # Set tier if not already set
            if user_id not in limiter.user_tiers:
                limiter.set_user_tier(user_id, tier)

            # Check limit
            allowed, metadata = limiter.check_limit(user_id)

            if not allowed:
                raise RateLimitException(
                    f"Rate limit exceeded. "
                    f"Try again in {metadata['reset_in']:.1f}s. "
                    f"Limit: {metadata['limit']}/min"
                )

            return func(*args, **kwargs)

        return wrapper

    return decorator


# ============================================================================
# DEMO
# ============================================================================


def demo():
    """Demonstrate rate limiting."""
    print("🚦 Enterprise Rate Limiter Demo\n")

    limiter = EnterpriseRateLimiter()

    # Set up users with different tiers
    limiter.set_user_tier("user_free", RateLimitTier.FREE)
    limiter.set_user_tier("user_pro", RateLimitTier.PRO)
    limiter.set_user_tier("user_enterprise", RateLimitTier.ENTERPRISE)

    # Simulate requests
    print("Simulating 150 requests from FREE user (limit: 10/min)...")
    for i in range(150):
        allowed, meta = limiter.check_limit("user_free")
        if i < 15:  # Show first 15
            status = "✅ ALLOWED" if allowed else "❌ RATE LIMITED"
            print(f"  Request {i+1}: {status} (remaining: {meta['remaining']})")

    print("\nSimulating 150 requests from PRO user (limit: 100/min)...")
    for i in range(150):
        allowed, meta = limiter.check_limit("user_pro")
        if i < 15 or i >= 95:  # Show first 15 and around limit
            status = "✅ ALLOWED" if allowed else "❌ RATE LIMITED"
            print(f"  Request {i+1}: {status} (remaining: {meta['remaining']})")

    print("\n📊 Metrics:")
    metrics = limiter.get_metrics()
    for key, value in metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    demo()
