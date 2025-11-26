"""
Gestione policy dominio: rate-limit, headers, rispetto robots/TOS e skip sicuro.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import yaml


@dataclass
class DomainPolicy:
    pattern: str
    requests_per_minute: int
    burst: int
    headers: Dict[str, str]
    respect_robots: bool
    tos_blocked: bool


@dataclass
class EffectivePolicy:
    pattern: str
    user_agent: str
    requests_per_minute: int
    burst: int
    headers: Dict[str, str]
    respect_robots: bool
    tos_blocked: bool
    timeout_seconds: int = 10  # Default timeout for HTTP requests


def load_domain_policies(path: str = "domains.yaml") -> tuple[dict, List[DomainPolicy]]:
    """
    Carica defaults e policy specifiche per dominio/pattern.
    """
    data = yaml.safe_load(Path(path).read_text()) or {}
    defaults = data.get("defaults", {})
    policies: List[DomainPolicy] = []

    for d in data.get("domains", []):
        policies.append(
            DomainPolicy(
                pattern=d["pattern"],
                requests_per_minute=d.get("requests_per_minute", defaults.get("requests_per_minute", 30)),
                burst=d.get("burst", defaults.get("burst", 10)),
                headers=d.get("headers", {}),
                respect_robots=d.get("respect_robots", defaults.get("respect_robots", True)),
                tos_blocked=d.get("tos_blocked", False),
            )
        )
    return defaults, policies


def resolve_policy_for_url(url: str, defaults: dict, policies: List[DomainPolicy]) -> EffectivePolicy:
    """
    Prende la prima policy che matcha il pattern dato. Fallback sui defaults.
    """
    matched: Optional[DomainPolicy] = None
    for p in policies:
        if fnmatch.fnmatch(url, p.pattern):
            matched = p
            break

    if matched is None:
        matched = DomainPolicy(
            pattern="*",
            requests_per_minute=defaults.get("requests_per_minute", 30),
            burst=defaults.get("burst", 10),
            headers={},
            respect_robots=defaults.get("respect_robots", True),
            tos_blocked=False,
        )

    return EffectivePolicy(
        pattern=matched.pattern,
        user_agent=defaults.get("user_agent", "Nodo33-CustodianBot/1.0"),
        requests_per_minute=matched.requests_per_minute,
        burst=matched.burst,
        headers=matched.headers,
        respect_robots=matched.respect_robots,
        tos_blocked=matched.tos_blocked,
    )


def should_skip_url(
    url: str,
    defaults: dict,
    policies: List[DomainPolicy],
    robots_checker: Optional[Callable[[str, str], bool]] = None,
    fail_open: bool = False,
) -> Tuple[bool, str, EffectivePolicy]:
    """
    Restituisce (skip, reason, policy) valutando TOS e robots.txt.
    - robots_checker: callable opzionale (url, user_agent) -> bool.
      Se mancante e respect_robots=True, skip a meno di fail_open=True.
    """
    policy = resolve_policy_for_url(url, defaults, policies)

    if policy.tos_blocked:
        return True, "TOS-blocked domain", policy

    if policy.respect_robots:
        if robots_checker is None and not fail_open:
            return True, "robots check unavailable (fail closed)", policy
        if robots_checker is not None and not robots_checker(url, policy.user_agent):
            return True, "robots.txt disallows", policy

    return False, "", policy


def fetch_with_policy(
    url: str,
    defaults: dict,
    policies: List[DomainPolicy],
    robots_checker: Optional[Callable[[str, str], bool]] = None,
    fail_open: bool = False,
):
    """
    Esempio di utilizzo: chiama should_skip_url e applica rate-limit/headers.
    Sostituisci l'HTTP client con la tua implementazione.
    """
    skip, reason, policy = should_skip_url(url, defaults, policies, robots_checker, fail_open=fail_open)
    if skip:
        print(f"[SKIP] {url} -> {reason}")
        return None

    headers = {"User-Agent": policy.user_agent, **policy.headers}
    # rate_limiter.acquire(policy.pattern, policy.requests_per_minute, policy.burst)  # integrazione attesa
    # return httpx.get(url, headers=headers, timeout=10)  # placeholder: integra il tuo client
    return {"url": url, "headers": headers, "policy": policy}


# ═══════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS FOR AGENT_EXECUTOR
# ═══════════════════════════════════════════════════════════

_cached_policies: Optional[Tuple[dict, List[DomainPolicy]]] = None


def get_policies(path: str = "domains.yaml") -> Tuple[dict, List[DomainPolicy]]:
    """
    Load and cache domain policies.
    Returns: (defaults, policies)
    """
    global _cached_policies
    if _cached_policies is None:
        _cached_policies = load_domain_policies(path)
    return _cached_policies


def check_and_wait(
    url: str,
    robots_checker: Optional[Callable[[str, str], bool]] = None,
    fail_open: bool = True,
) -> Tuple[bool, str, EffectivePolicy]:
    """
    Check if URL should be processed (inverse of should_skip_url).

    Returns:
        (should_proceed, skip_reason, policy)
        - should_proceed: True if URL can be fetched, False if should skip
        - skip_reason: Reason for skipping (empty if should_proceed=True)
        - policy: EffectivePolicy for the URL
    """
    defaults, policies = get_policies()
    skip, reason, policy = should_skip_url(url, defaults, policies, robots_checker, fail_open)

    # Invert the skip flag to get should_proceed
    should_proceed = not skip

    return should_proceed, reason, policy
