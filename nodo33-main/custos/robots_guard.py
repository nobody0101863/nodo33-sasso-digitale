"""
Gestore robots.txt con cache in memoria.
Usa urllib.robotparser ma scarica il file con requests per timeout controllato.
"""

from __future__ import annotations

import time
import urllib.parse
import urllib.robotparser
from typing import Dict, Optional

import requests


class RobotsGuard:
    def __init__(
        self,
        fetch_timeout: int = 5,
        cache_ttl: int = 3600,
        fail_open: bool = False,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.fetch_timeout = fetch_timeout
        self.cache_ttl = cache_ttl
        self.fail_open = fail_open
        self.session = session or requests.Session()
        self._cache: Dict[str, Dict[str, object]] = {}

    def is_allowed(self, url: str, user_agent: str) -> bool:
        parser = self._get_parser(url)
        if parser is None:
            return self.fail_open
        try:
            return parser.can_fetch(user_agent, url)
        except Exception:
            return self.fail_open

    def _get_parser(self, url: str) -> Optional[urllib.robotparser.RobotFileParser]:
        parsed = urllib.parse.urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return None

        base = f"{parsed.scheme}://{parsed.netloc}"
        now = time.time()
        cached = self._cache.get(base)
        if cached and cached.get("expires", 0) > now:
            return cached.get("parser")  # type: ignore[return-value]

        robots_url = f"{base}/robots.txt"
        parser = urllib.robotparser.RobotFileParser()
        try:
            resp = self.session.get(robots_url, timeout=self.fetch_timeout)
            if resp.status_code >= 400:
                # Se robots manca o è 4xx/5xx, parser senza regole (interpretabile come allow-all).
                parser.parse([])
            else:
                parser.parse(resp.text.splitlines())
            self._cache[base] = {"parser": parser, "expires": now + self.cache_ttl}
            return parser
        except Exception:
            self._cache[base] = {"parser": None, "expires": now + self.cache_ttl}
            return None
