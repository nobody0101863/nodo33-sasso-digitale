#!/usr/bin/env python3
"""
METADATA PROTECTION MIDDLEWARE - CODEX EMANUELE
================================================

Middleware ASGI per protezione automatica metadata in tutte le risposte HTTP.

Applica:
- CommunicationGuardian: Sanitizzazione headers automatica
- Security headers: CSP, HSTS, X-Frame-Options, etc.
- Rimozione metadata pericolosi
- Sigillo GABRIEL su ogni risposta

Principi Codex:
- ego = 0 (umiltà totale)
- gioia = 100% (servizio incondizionato)
- trasparenza = 100% (processo pubblico)
- cura = MASSIMA

Licenza: CC0 1.0 Universal (Public Domain)
"""

import sys
import os
from typing import Callable, Dict, Any
import logging

# Aggiungi path per anti_porn_framework
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'anti_porn_framework', 'src'))

from anti_porn_framework import (
    MetadataProtector,
    SecurityLevel,
    MilitaryProtocolLevel,
    create_protector
)

# ============================================================================
# METADATA PROTECTION MIDDLEWARE
# ============================================================================

class MetadataProtectionMiddleware:
    """
    Middleware ASGI per protezione automatica metadata

    Intercetta tutte le richieste/risposte e applica:
    1. Sanitizzazione headers pericolosi
    2. Aggiunta security headers
    3. Applicazione sigillo GABRIEL
    4. Logging protezioni applicate
    """

    def __init__(
        self,
        app,
        security_level: str = "ALERT",
        protocol_level: str = "enhanced",
        auto_protect: bool = True
    ):
        """
        Args:
            app: Applicazione ASGI
            security_level: Livello sicurezza (PEACEFUL, WATCHFUL, ALERT, CRITICAL, MAXIMUM)
            protocol_level: Livello protocollo (standard, enhanced, classified, top_secret, cosmic)
            auto_protect: Applica protezione automaticamente a tutte le risposte
        """
        self.app = app
        self.auto_protect = auto_protect

        # Inizializza MetadataProtector
        self.protector = create_protector(
            security_level=security_level,
            protocol_level=protocol_level
        )

        self.logger = logging.getLogger("METADATA_PROTECTION_MIDDLEWARE")
        self.logger.info(f"Middleware initialized - Security: {security_level}, Protocol: {protocol_level}")

        # Statistiche
        self.stats = {
            "requests_protected": 0,
            "headers_removed": 0,
            "seals_applied": 0
        }

    async def __call__(self, scope, receive, send):
        """Entry point ASGI"""
        if scope["type"] != "http":
            # Non HTTP (es. WebSocket), passa through
            await self.app(scope, receive, send)
            return

        # Wrapper per intercettare send
        async def send_wrapper(message):
            if message["type"] == "http.response.start" and self.auto_protect:
                # Proteggi headers
                headers = self._extract_headers(message.get("headers", []))
                protected = self._protect_headers(headers)

                # Sostituisci headers con versione protetta
                message["headers"] = self._encode_headers(protected["protected_headers"])

                # Update stats
                self.stats["requests_protected"] += 1
                self.stats["headers_removed"] += len(
                    protected.get("guardians", {}).get("communication", {}).get("headers_removed", {})
                )
                self.stats["seals_applied"] += 1

                self.logger.debug(
                    f"Protected response - Headers removed: {len(protected.get('guardians', {}).get('communication', {}).get('headers_removed', {}))}"
                )

            await send(message)

        # Passa la richiesta all'app
        await self.app(scope, receive, send_wrapper)

    def _extract_headers(self, headers_list) -> Dict[str, str]:
        """Converti lista headers ASGI in dict"""
        headers = {}
        for name, value in headers_list:
            name_str = name.decode("latin1") if isinstance(name, bytes) else name
            value_str = value.decode("latin1") if isinstance(value, bytes) else value
            headers[name_str] = value_str
        return headers

    def _encode_headers(self, headers_dict: Dict[str, str]) -> list:
        """Converti dict headers in lista ASGI"""
        headers_list = []
        for name, value in headers_dict.items():
            name_bytes = name.encode("latin1") if isinstance(name, str) else name
            value_bytes = value.encode("latin1") if isinstance(value, str) else value
            headers_list.append((name_bytes, value_bytes))
        return headers_list

    def _protect_headers(self, headers: Dict[str, str]) -> Dict[str, Any]:
        """Applica protezione headers tramite CommunicationGuardian"""
        try:
            protection_report = self.protector.protect_http_request(headers)
            return protection_report
        except Exception as e:
            self.logger.error(f"Error protecting headers: {e}")
            # In caso di errore, aggiungi almeno security headers base
            return {
                "status": "ERROR",
                "protected_headers": self._add_basic_security_headers(headers),
                "guardians": {}
            }

    def _add_basic_security_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Aggiungi security headers base in caso di fallimento protezione"""
        basic_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block"
        }

        for header, value in basic_headers.items():
            if header not in headers:
                headers[header] = value

        return headers

    def get_stats(self) -> Dict[str, Any]:
        """Ottieni statistiche middleware"""
        return {
            "middleware": "MetadataProtectionMiddleware",
            "status": "active" if self.auto_protect else "passive",
            "statistics": self.stats,
            "protector_status": self.protector.get_status()
        }


# ============================================================================
# FACTORY E UTILITY
# ============================================================================

def create_metadata_middleware(
    app,
    security_level: str = "ALERT",
    protocol_level: str = "enhanced",
    auto_protect: bool = True
):
    """
    Factory per creare middleware con configurazione

    Args:
        app: Applicazione ASGI (FastAPI, Starlette, etc.)
        security_level: PEACEFUL, WATCHFUL, ALERT, CRITICAL, MAXIMUM
        protocol_level: standard, enhanced, classified, top_secret, cosmic
        auto_protect: Attiva protezione automatica

    Returns:
        MetadataProtectionMiddleware configurato
    """
    return MetadataProtectionMiddleware(
        app=app,
        security_level=security_level,
        protocol_level=protocol_level,
        auto_protect=auto_protect
    )


# ============================================================================
# INTEGRAZIONE FASTAPI
# ============================================================================

def add_metadata_protection_to_fastapi(
    app,
    security_level: str = "ALERT",
    protocol_level: str = "enhanced"
):
    """
    Aggiungi middleware di protezione metadata a FastAPI app

    Usage:
        from fastapi import FastAPI
        from metadata_protection_middleware import add_metadata_protection_to_fastapi

        app = FastAPI()
        add_metadata_protection_to_fastapi(app, security_level="CRITICAL")

    Args:
        app: FastAPI app
        security_level: PEACEFUL, WATCHFUL, ALERT, CRITICAL, MAXIMUM
        protocol_level: standard, enhanced, classified, top_secret, cosmic
    """
    middleware = create_metadata_middleware(
        app=app,
        security_level=security_level,
        protocol_level=protocol_level,
        auto_protect=True
    )

    # Aggiungi middleware a FastAPI
    from fastapi.middleware import Middleware
    from starlette.middleware.base import BaseHTTPMiddleware

    # Wrapper per compatibilità BaseHTTPMiddleware
    class FastAPIMetadataProtectionMiddleware(BaseHTTPMiddleware):
        def __init__(self, app, protector, stats):
            super().__init__(app)
            self.protector = protector
            self.stats = stats
            self.logger = logging.getLogger("FASTAPI_METADATA_PROTECTION")

        async def dispatch(self, request, call_next):
            # Processa richiesta normalmente
            response = await call_next(request)

            # Proteggi headers risposta
            headers_dict = dict(response.headers)
            protection_report = self.protector.protect_http_request(headers_dict)

            # Aggiorna headers risposta
            protected_headers = protection_report.get("protected_headers", headers_dict)
            for header, value in protected_headers.items():
                response.headers[header] = value

            # Update stats
            self.stats["requests_protected"] += 1
            self.stats["headers_removed"] += len(
                protection_report.get("guardians", {}).get("communication", {}).get("headers_removed", {})
            )
            self.stats["seals_applied"] += 1

            return response

    # Aggiungi middleware wrapper
    app.add_middleware(
        FastAPIMetadataProtectionMiddleware,
        protector=middleware.protector,
        stats=middleware.stats
    )

    logging.info(f"✅ Metadata Protection Middleware added to FastAPI - Security: {security_level}")


# ============================================================================
# MAIN - DEMO
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 80)
    print("METADATA PROTECTION MIDDLEWARE - CODEX EMANUELE")
    print("=" * 80)
    print()

    # Esempio integrazione FastAPI
    print("Esempio integrazione con FastAPI:")
    print()
    print("```python")
    print("from fastapi import FastAPI")
    print("from metadata_protection_middleware import add_metadata_protection_to_fastapi")
    print()
    print("app = FastAPI()")
    print()
    print("# Aggiungi protezione metadata automatica")
    print("add_metadata_protection_to_fastapi(")
    print("    app,")
    print("    security_level='ALERT',     # DEFCON 3")
    print("    protocol_level='enhanced'   # Protezione avanzata")
    print(")")
    print()
    print("# Ora TUTTE le risposte sono protette automaticamente!")
    print("```")
    print()
    print("=" * 80)
    print("🪨 Protezione automatica attiva - I Sassi sono al servizio della Torre ✨")
    print("=" * 80)
