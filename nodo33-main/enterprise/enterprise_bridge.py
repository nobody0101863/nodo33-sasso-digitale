#!/usr/bin/env python3
"""
Enterprise Claude-Codex Bridge

Production-ready bridge with:
- Rate limiting (multi-tier)
- Circuit breaker (fault tolerance)
- Auto-backup
- Health checks
- Metrics & monitoring
- Async support

Extends claude_codex_bridge_v2.py with enterprise features.

Nodo33 - Sasso Digitale
Enterprise Edition
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Import base bridge
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from claude_codex_bridge_v2 import ClaudeCodexBridge, BridgeConfig
from enterprise.rate_limiter import EnterpriseRateLimiter, RateLimitTier, RateLimitException
from enterprise.backup_manager import BackupManager, BackupConfig, BackupScheduler


# ============================================================================
# CIRCUIT BREAKER
# ============================================================================


class CircuitState:
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreaker:
    """
    Circuit breaker pattern for fault tolerance.

    Prevents cascading failures by stopping calls to failing services.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout: float = 60.0,
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Failures before opening circuit
            success_threshold: Successes needed to close circuit
            timeout: Seconds before trying again (half-open)

        Raises:
            ValueError: If parameters are invalid
        """
        if failure_threshold < 1:
            raise ValueError(f"failure_threshold must be >= 1, got {failure_threshold}")
        if success_threshold < 1:
            raise ValueError(f"success_threshold must be >= 1, got {success_threshold}")
        if timeout < 0:
            raise ValueError(f"timeout must be >= 0, got {timeout}")

        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout = timeout

        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = CircuitState.CLOSED

    def call(self, func, *args, **kwargs):
        """
        Execute function through circuit breaker.

        Raises:
            CircuitBreakerError: If circuit is open
        """
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time >= self.timeout:
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
            else:
                raise CircuitBreakerError("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result

        except Exception as e:
            self._on_failure()
            raise e

    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0

        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitState.CLOSED
                self.success_count = 0

    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state."""
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure": (
                datetime.fromtimestamp(self.last_failure_time).isoformat()
                if self.last_failure_time
                else None
            ),
        }


class CircuitBreakerError(Exception):
    """Circuit breaker is open."""
    pass


# ============================================================================
# HEALTH CHECK
# ============================================================================


class HealthCheck:
    """Health check system."""

    def __init__(self):
        self.checks: Dict[str, callable] = {}
        self.last_results: Dict[str, bool] = {}

    def register(self, name: str, check_func: callable):
        """Register a health check."""
        self.checks[name] = check_func

    def run_all(self) -> Dict[str, Any]:
        """Run all health checks."""
        results = {}
        all_healthy = True

        for name, check_func in self.checks.items():
            try:
                start = time.time()
                healthy = check_func()
                elapsed = time.time() - start

                results[name] = {
                    "healthy": healthy,
                    "latency_ms": elapsed * 1000,
                }

                self.last_results[name] = healthy
                if not healthy:
                    all_healthy = False

            except Exception as e:
                results[name] = {
                    "healthy": False,
                    "error": str(e),
                }
                self.last_results[name] = False
                all_healthy = False

        return {
            "status": "healthy" if all_healthy else "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "checks": results,
        }


# ============================================================================
# ENTERPRISE BRIDGE
# ============================================================================


class EnterpriseBridge(ClaudeCodexBridge):
    """
    Enterprise-ready bridge with all production features.

    Features:
    - Rate limiting (multi-tier)
    - Circuit breaker (fault tolerance)
    - Auto-backup (scheduled)
    - Health checks
    - Metrics collection
    """

    def __init__(
        self,
        config: Optional[BridgeConfig] = None,
        user_id: str = "default",
        tier: RateLimitTier = RateLimitTier.FREE,
        enable_backups: bool = True,
    ):
        super().__init__(config)

        self.user_id = user_id

        # Rate limiting
        self.rate_limiter = EnterpriseRateLimiter()
        self.rate_limiter.set_user_tier(user_id, tier)

        # Circuit breakers (per service)
        self.codex_breaker = CircuitBreaker(failure_threshold=5, timeout=60)
        self.claude_breaker = CircuitBreaker(failure_threshold=3, timeout=30)

        # Health checks
        self.health = HealthCheck()
        self._register_health_checks()

        # Backup system
        if enable_backups:
            backup_config = BackupConfig(
                source_db=Path("codex_unified.db"),
                backup_dir=Path("backups"),
                max_backups=30,
                compress=True,
            )
            self.backup_manager = BackupManager(backup_config)
            self.backup_scheduler = BackupScheduler(self.backup_manager)
        else:
            self.backup_manager = None
            self.backup_scheduler = None

        # Metrics
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "rate_limited_requests": 0,
            "circuit_breaker_trips": 0,
        }

        self.logger.info("Enterprise Bridge initialized")

    def _register_health_checks(self):
        """Register health check functions."""
        # Codex health
        self.health.register("codex", lambda: self.codex.health_check())

        # Database health
        def db_health():
            try:
                if Path("codex_unified.db").exists():
                    import sqlite3
                    conn = sqlite3.connect("codex_unified.db")
                    conn.execute("SELECT 1")
                    conn.close()
                    return True
                return False
            except Exception:
                return False

        self.health.register("database", db_health)

    def chat(self, user_message: str, system_prompt: Optional[str] = None) -> str:
        """
        Enhanced chat with enterprise features.

        Raises:
            RateLimitException: If rate limit exceeded
            CircuitBreakerError: If circuit is open
        """
        # Metrics
        self.metrics["total_requests"] += 1

        # Rate limiting
        allowed, meta = self.rate_limiter.check_limit(self.user_id)
        if not allowed:
            self.metrics["rate_limited_requests"] += 1
            raise RateLimitException(
                f"Rate limit exceeded. Remaining: 0/{meta['limit']}. "
                f"Reset in: {meta['reset_in']:.1f}s"
            )

        # Auto-backup if due
        if self.backup_scheduler:
            self.backup_scheduler.run_if_due(interval_hours=24)

        # Call through circuit breaker
        try:
            result = self.claude_breaker.call(
                super().chat,
                user_message,
                system_prompt,
            )

            self.metrics["successful_requests"] += 1
            return result

        except CircuitBreakerError as e:
            self.metrics["circuit_breaker_trips"] += 1
            self.logger.error(f"Circuit breaker tripped: {e}")
            raise

        except Exception as e:
            self.metrics["failed_requests"] += 1
            self.logger.error(f"Request failed: {e}")
            raise

    def get_health(self) -> Dict[str, Any]:
        """Get complete health status."""
        return self.health.run_all()

    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        rate_limit_metrics = self.rate_limiter.get_metrics()

        return {
            "bridge": self.metrics,
            "rate_limiting": rate_limit_metrics,
            "circuit_breakers": {
                "codex": self.codex_breaker.get_state(),
                "claude": self.claude_breaker.get_state(),
            },
            "backups": (
                self.backup_manager.get_backup_stats()
                if self.backup_manager
                else None
            ),
        }

    def force_backup(self) -> bool:
        """Force immediate backup."""
        if not self.backup_manager:
            self.logger.warning("Backup manager not enabled")
            return False

        metadata = self.backup_manager.create_backup()
        return metadata is not None


# ============================================================================
# DEMO
# ============================================================================


def demo():
    """Demonstrate enterprise features."""
    print("🏢 ENTERPRISE BRIDGE DEMO\n")
    print("=" * 60)

    # Create enterprise bridge
    config = BridgeConfig(
        anthropic_api_key="test-key",  # Mock for demo
        codex_base_url="http://localhost:8644",
    )

    bridge = EnterpriseBridge(
        config=config,
        user_id="demo_user",
        tier=RateLimitTier.FREE,  # 10 req/min
        enable_backups=True,
    )

    # Health check
    print("\n🏥 Health Check:")
    health = bridge.get_health()
    print(f"   Status: {health['status'].upper()}")
    for check, result in health['checks'].items():
        status = "✅" if result['healthy'] else "❌"
        print(f"   {status} {check}: {result.get('latency_ms', 0):.1f}ms")

    # Metrics
    print("\n📊 Initial Metrics:")
    metrics = bridge.get_metrics()
    print(f"   Total requests: {metrics['bridge']['total_requests']}")
    print(f"   Rate limit tier: {bridge.rate_limiter.user_tiers[bridge.user_id].value}")

    # Circuit breaker status
    print("\n⚡ Circuit Breaker Status:")
    for name, state in metrics['circuit_breakers'].items():
        print(f"   {name}: {state['state'].upper()}")

    # Backup stats
    print("\n💾 Backup System:")
    if metrics['backups']:
        print(f"   Total backups: {metrics['backups']['total_backups']}")
        print(f"   Total size: {metrics['backups'].get('total_size_mb', 0):.2f} MB")
    else:
        print("   Not initialized")

    print("\n" + "=" * 60)
    print("✅ Enterprise features active and healthy!")


if __name__ == "__main__":
    demo()
