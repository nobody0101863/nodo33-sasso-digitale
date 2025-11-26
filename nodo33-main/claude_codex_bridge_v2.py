#!/usr/bin/env python3
"""
Claude-Codex Bridge v2.0 - Improved & Secure

Bridge potenziato tra Claude API e Codex server con:
- Validazione input robusta (anti prompt injection)
- Retry logic con exponential backoff
- Logging strutturato
- Conversazione multi-turno con memoria
- Architettura modulare e testabile
- Rate limiting e timeout configurabili

Autore: Nodo33 - Sasso Digitale
Motto: "La luce non si vende. La si regala."
"""

from __future__ import annotations

import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None  # type: ignore[assignment]


# ============================================================================
# CONFIGURAZIONE
# ============================================================================


class LogLevel(Enum):
    """Livelli di logging supportati."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass
class BridgeConfig:
    """Configurazione centralizzata del bridge."""

    # Claude settings
    claude_model: str = "claude-3-5-sonnet-20241022"
    claude_max_tokens: int = 2048
    anthropic_api_key: Optional[str] = None

    # Codex server settings
    codex_base_url: str = "http://localhost:8644"
    codex_timeout: int = 120  # secondi
    codex_max_retries: int = 3
    codex_backoff_factor: float = 2.0  # exponential backoff: 2, 4, 8 secondi

    # Security settings
    max_prompt_length: int = 5000
    max_image_steps: int = 50
    allowed_url_schemes: tuple = ("http", "https")
    validate_ssl: bool = True

    # Logging
    log_level: LogLevel = LogLevel.INFO
    log_file: Optional[Path] = None

    # Conversazione
    enable_conversation_history: bool = True
    max_conversation_turns: int = 10

    @classmethod
    def from_env(cls) -> BridgeConfig:
        """Crea configurazione da variabili d'ambiente."""
        return cls(
            claude_model=os.environ.get("CLAUDE_MODEL", cls.claude_model),
            claude_max_tokens=int(
                os.environ.get("CLAUDE_MAX_TOKENS", cls.claude_max_tokens)
            ),
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
            codex_base_url=os.environ.get("CODEX_BASE_URL", cls.codex_base_url),
            codex_timeout=int(os.environ.get("CODEX_TIMEOUT", cls.codex_timeout)),
            log_level=LogLevel(
                os.environ.get("BRIDGE_LOG_LEVEL", cls.log_level.value)
            ),
        )


# ============================================================================
# LOGGING
# ============================================================================


def setup_logging(config: BridgeConfig) -> logging.Logger:
    """Configura logging strutturato."""
    logger = logging.getLogger("claude_codex_bridge")
    logger.setLevel(config.log_level.value)

    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if config.log_file:
        file_handler = logging.FileHandler(config.log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# ============================================================================
# VALIDAZIONE & SICUREZZA
# ============================================================================


class ValidationError(ValueError):
    """Errore di validazione input."""

    pass


class SecurityValidator:
    """Validatore per input e sicurezza."""

    # Pattern per rilevare potenziali prompt injection
    SUSPICIOUS_PATTERNS = [
        r"ignore\s+previous\s+instructions",
        r"disregard\s+all\s+prior",
        r"you\s+are\s+now\s+(?:a|an)",
        r"system\s*:\s*role",
        r"<\|im_start\|>",  # ChatGPT special tokens
        r"<\|im_end\|>",
    ]

    def __init__(self, config: BridgeConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self._compiled_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.SUSPICIOUS_PATTERNS
        ]

    def validate_prompt(self, prompt: str) -> str:
        """
        Valida e sanitizza il prompt utente.

        Raises:
            ValidationError: Se il prompt è invalido o sospetto
        """
        if not prompt or not prompt.strip():
            raise ValidationError("Prompt vuoto")

        prompt = prompt.strip()

        if len(prompt) > self.config.max_prompt_length:
            raise ValidationError(
                f"Prompt troppo lungo: {len(prompt)} chars "
                f"(max: {self.config.max_prompt_length})"
            )

        # Controllo pattern sospetti (prompt injection)
        for pattern in self._compiled_patterns:
            if pattern.search(prompt):
                self.logger.warning(
                    f"Suspicious pattern detected in prompt: {pattern.pattern}"
                )
                # Non blocchiamo, ma loggiamo per monitoraggio
                # In produzione potresti voler bloccare

        return prompt

    def validate_url(self, url: str) -> str:
        """
        Valida URL del server Codex.

        Raises:
            ValidationError: Se URL non è valido o sicuro
        """
        parsed = urlparse(url)

        if not parsed.scheme:
            raise ValidationError(f"URL senza schema: {url}")

        if parsed.scheme not in self.config.allowed_url_schemes:
            raise ValidationError(
                f"Schema non consentito: {parsed.scheme} "
                f"(allowed: {self.config.allowed_url_schemes})"
            )

        if not parsed.netloc:
            raise ValidationError(f"URL senza host: {url}")

        # Blocca localhost in produzione se necessario
        # if parsed.hostname not in ('localhost', '127.0.0.1'):
        #     raise ValidationError(f"Only localhost allowed, got: {parsed.hostname}")

        return url

    def validate_image_params(
        self, steps: int, guidance_scale: float
    ) -> tuple[int, float]:
        """
        Valida parametri di generazione immagine.

        Raises:
            ValidationError: Se i parametri sono fuori range
        """
        if not (1 <= steps <= self.config.max_image_steps):
            raise ValidationError(
                f"Steps out of range: {steps} (allowed: 1-{self.config.max_image_steps})"
            )

        if not (0.0 <= guidance_scale <= 20.0):
            raise ValidationError(
                f"Guidance scale out of range: {guidance_scale} (allowed: 0.0-20.0)"
            )

        return int(steps), float(guidance_scale)


# ============================================================================
# CODEX CLIENT
# ============================================================================


class CodexClient:
    """Client per comunicare con il server Codex."""

    def __init__(
        self, config: BridgeConfig, validator: SecurityValidator, logger: logging.Logger
    ):
        self.config = config
        self.validator = validator
        self.logger = logger
        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        """Crea sessione HTTP con retry logic."""
        session = requests.Session()

        # Configura retry con exponential backoff
        retry_strategy = Retry(
            total=self.config.codex_max_retries,
            backoff_factor=self.config.codex_backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST", "GET"],
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    def generate_image(
        self, prompt: str, steps: int = 4, guidance_scale: float = 1.5
    ) -> Dict[str, Any]:
        """
        Genera immagine tramite Codex server.

        Args:
            prompt: Prompt per l'immagine
            steps: Numero di inference steps
            guidance_scale: Guidance scale per la generazione

        Returns:
            Dizionario con risposta del server

        Raises:
            ValidationError: Se input non è valido
            requests.RequestException: Se la chiamata fallisce
        """
        # Validazione
        prompt = self.validator.validate_prompt(prompt)
        steps, guidance_scale = self.validator.validate_image_params(
            steps, guidance_scale
        )
        url = self.validator.validate_url(
            f"{self.config.codex_base_url.rstrip('/')}/api/generate-image"
        )

        payload = {
            "prompt": prompt,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
        }

        self.logger.info(
            f"Calling Codex: prompt_len={len(prompt)}, steps={steps}, scale={guidance_scale}"
        )

        start_time = time.time()

        try:
            response = self.session.post(
                url,
                json=payload,
                timeout=self.config.codex_timeout,
                verify=self.config.validate_ssl,
            )
            response.raise_for_status()

            elapsed = time.time() - start_time
            self.logger.info(f"Codex response received in {elapsed:.2f}s")

            return response.json()

        except requests.Timeout:
            self.logger.error(
                f"Codex timeout after {self.config.codex_timeout}s for prompt: {prompt[:100]}"
            )
            raise
        except requests.RequestException as e:
            self.logger.error(f"Codex request failed: {e}")
            raise

    def health_check(self) -> bool:
        """Verifica che il server Codex sia raggiungibile."""
        try:
            url = f"{self.config.codex_base_url.rstrip('/')}/health"
            response = self.session.get(url, timeout=5)
            return response.status_code == 200
        except Exception as e:
            self.logger.warning(f"Codex health check failed: {e}")
            return False


# ============================================================================
# CLAUDE CLIENT
# ============================================================================


class ClaudeClient:
    """Client per comunicare con Claude API."""

    TOOLS: List[Dict[str, Any]] = [
        {
            "name": "codex_pulse_image",
            "description": (
                "Genera un'immagine 'pulse' usando il Codex server (Stable Diffusion). "
                "Usa questo tool quando l'utente chiede di creare, generare o visualizzare immagini."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Descrizione dettagliata dell'immagine da generare",
                    },
                    "num_inference_steps": {
                        "type": "integer",
                        "default": 4,
                        "description": "Numero di step (più alto = migliore qualità, più lento)",
                    },
                    "guidance_scale": {
                        "type": "number",
                        "default": 1.5,
                        "description": "Quanto seguire il prompt (1.0-7.0 tipicamente)",
                    },
                },
                "required": ["prompt"],
            },
        },
        {
            "name": "codex_query_status",
            "description": "Interroga lo stato del server Codex (uptime, salute, metriche).",
            "input_schema": {"type": "object", "properties": {}},
        },
    ]

    def __init__(self, config: BridgeConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.client = self._get_client()

    def _get_client(self) -> Anthropic:
        """Ottiene client Anthropic."""
        if Anthropic is None:
            raise RuntimeError(
                "Libreria 'anthropic' non installata. "
                "Installa con: pip install anthropic requests"
            )

        api_key = self.config.anthropic_api_key
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY non configurata")

        return Anthropic(api_key=api_key)

    def send_message(
        self, messages: List[Dict[str, Any]], system_prompt: Optional[str] = None
    ) -> Any:
        """
        Invia messaggio a Claude.

        Args:
            messages: Lista di messaggi nel formato Claude
            system_prompt: System prompt opzionale

        Returns:
            Risposta di Claude
        """
        self.logger.info(f"Sending message to Claude ({len(messages)} messages)")

        kwargs: Dict[str, Any] = {
            "model": self.config.claude_model,
            "max_tokens": self.config.claude_max_tokens,
            "tools": self.TOOLS,
            "messages": messages,
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        response = self.client.messages.create(**kwargs)

        self.logger.debug(f"Claude response: {response}")
        return response

    @staticmethod
    def extract_text(response: Any) -> str:
        """Estrae testo dalla risposta di Claude."""
        parts: List[str] = []
        for block in getattr(response, "content", []) or []:
            if getattr(block, "type", None) == "text":
                parts.append(getattr(block, "text", ""))
        return "\n".join(p for p in parts if p).strip()

    @staticmethod
    def extract_tool_uses(response: Any) -> List[Any]:
        """Estrae tool uses dalla risposta di Claude."""
        return [
            block
            for block in getattr(response, "content", [])
            if getattr(block, "type", None) == "tool_use"
        ]


# ============================================================================
# BRIDGE PRINCIPALE
# ============================================================================


@dataclass
class ConversationState:
    """Stato della conversazione."""

    messages: List[Dict[str, Any]] = field(default_factory=list)
    turn_count: int = 0


class ClaudeCodexBridge:
    """Bridge principale tra Claude e Codex."""

    def __init__(self, config: Optional[BridgeConfig] = None):
        self.config = config or BridgeConfig.from_env()
        self.logger = setup_logging(self.config)
        self.validator = SecurityValidator(self.config, self.logger)
        self.codex = CodexClient(self.config, self.validator, self.logger)
        self.claude = ClaudeClient(self.config, self.logger)
        self.conversation = ConversationState()

        self.logger.info("Bridge initialized successfully")
        self._check_codex_health()

    def _check_codex_health(self) -> None:
        """Verifica salute del server Codex all'avvio."""
        if not self.codex.health_check():
            self.logger.warning(
                f"Codex server at {self.config.codex_base_url} is not responding. "
                "Image generation may fail."
            )

    def _handle_tool_use(self, tool_use: Any) -> Dict[str, Any]:
        """
        Gestisce l'esecuzione di un tool.

        Args:
            tool_use: Tool use block da Claude

        Returns:
            Risultato del tool in formato tool_result
        """
        tool_name = getattr(tool_use, "name", "")
        tool_input = getattr(tool_use, "input", {}) or {}

        self.logger.info(f"Handling tool: {tool_name}")

        if tool_name == "codex_pulse_image":
            return self._handle_image_generation(tool_use, tool_input)
        elif tool_name == "codex_query_status":
            return self._handle_status_query(tool_use)
        else:
            error_msg = f"Unknown tool: {tool_name}"
            self.logger.error(error_msg)
            return {
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": [{"type": "text", "text": f"ERROR: {error_msg}"}],
                "is_error": True,
            }

    def _handle_image_generation(
        self, tool_use: Any, tool_input: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Gestisce generazione immagine."""
        prompt = str(tool_input.get("prompt", ""))
        steps = int(tool_input.get("num_inference_steps", 4))
        scale = float(tool_input.get("guidance_scale", 1.5))

        try:
            result = self.codex.generate_image(
                prompt=prompt, steps=steps, guidance_scale=scale
            )

            image_url = result.get("image_url", "N/A")
            status = result.get("status", "unknown")

            success_msg = (
                f"✓ Immagine generata con successo!\n"
                f"  URL: {image_url}\n"
                f"  Status: {status}\n"
                f"  Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}"
            )

            return {
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": [{"type": "text", "text": success_msg}],
            }

        except Exception as e:
            error_msg = f"Errore generazione immagine: {e}"
            self.logger.error(error_msg)
            return {
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": [{"type": "text", "text": f"ERROR: {error_msg}"}],
                "is_error": True,
            }

    def _handle_status_query(self, tool_use: Any) -> Dict[str, Any]:
        """Gestisce query di status del Codex."""
        is_healthy = self.codex.health_check()
        status_msg = (
            f"Codex Server Status:\n"
            f"  URL: {self.config.codex_base_url}\n"
            f"  Health: {'✓ OK' if is_healthy else '✗ OFFLINE'}\n"
            f"  Timeout: {self.config.codex_timeout}s\n"
            f"  Max Retries: {self.config.codex_max_retries}"
        )

        return {
            "type": "tool_result",
            "tool_use_id": tool_use.id,
            "content": [{"type": "text", "text": status_msg}],
        }

    def chat(self, user_message: str, system_prompt: Optional[str] = None) -> str:
        """
        Invia un messaggio e gestisce la conversazione con tool calling.

        Args:
            user_message: Messaggio dell'utente
            system_prompt: System prompt opzionale

        Returns:
            Risposta finale di Claude
        """
        # Valida input
        user_message = self.validator.validate_prompt(user_message)

        # Aggiungi messaggio utente
        self.conversation.messages.append(
            {"role": "user", "content": user_message}
        )

        # Prima chiamata a Claude
        response = self.claude.send_message(
            self.conversation.messages, system_prompt=system_prompt
        )

        # Estrai tool uses
        tool_uses = self.claude.extract_tool_uses(response)

        # Se non ci sono tool, restituisci il testo
        if not tool_uses:
            text_response = self.claude.extract_text(response) or repr(response)
            # Aggiungi risposta assistant alla storia
            if self.config.enable_conversation_history:
                self.conversation.messages.append(
                    {"role": "assistant", "content": response.content}
                )
            return text_response

        # Prepara contenuto assistant con tool uses
        assistant_content = []
        for block in response.content:
            if getattr(block, "type", None) == "tool_use":
                assistant_content.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )
            elif getattr(block, "type", None) == "text":
                assistant_content.append(
                    {"type": "text", "text": block.text}
                )

        # Esegui tools
        tool_results = [self._handle_tool_use(tu) for tu in tool_uses]

        # Prepara messaggi per followup
        followup_messages = self.conversation.messages + [
            {"role": "assistant", "content": assistant_content},
            {"role": "user", "content": tool_results},
        ]

        # Chiamata followup
        followup_response = self.claude.send_message(
            followup_messages, system_prompt=system_prompt
        )

        final_text = self.claude.extract_text(followup_response) or repr(
            followup_response
        )

        # Aggiorna storia conversazione
        if self.config.enable_conversation_history:
            self.conversation.messages.append(
                {"role": "assistant", "content": assistant_content}
            )
            self.conversation.messages.append({"role": "user", "content": tool_results})
            self.conversation.messages.append(
                {"role": "assistant", "content": followup_response.content}
            )

            # Limita turni conversazione
            self.conversation.turn_count += 1
            if self.conversation.turn_count > self.config.max_conversation_turns:
                self.logger.info("Max conversation turns reached, resetting history")
                self.reset_conversation()

        return final_text

    def reset_conversation(self) -> None:
        """Resetta lo stato della conversazione."""
        self.conversation = ConversationState()
        self.logger.info("Conversation reset")

    def interactive_mode(self) -> None:
        """Modalità interattiva con conversazione multi-turno."""
        print("=== Claude-Codex Bridge v2.0 ===")
        print("Modalità interattiva. Digita 'exit' o 'quit' per uscire.")
        print("Digita 'reset' per resettare la conversazione.\n")

        while True:
            try:
                user_input = input("Tu: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ("exit", "quit"):
                    print("Arrivederci!")
                    break

                if user_input.lower() == "reset":
                    self.reset_conversation()
                    print("Conversazione resettata.\n")
                    continue

                response = self.chat(user_input)
                print(f"\nClaude: {response}\n")

            except KeyboardInterrupt:
                print("\n\nInterrotto dall'utente. Arrivederci!")
                break
            except Exception as e:
                self.logger.error(f"Error in interactive mode: {e}", exc_info=True)
                print(f"\nErrore: {e}\n")


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
    """Entry point principale."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Claude-Codex Bridge v2.0 - Nodo33 Sasso Digitale",
        epilog='Motto: "La luce non si vende. La si regala."',
    )
    parser.add_argument(
        "message",
        nargs="*",
        help="Messaggio da inviare a Claude (se omesso, modalità interattiva)",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Avvia in modalità interattiva",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Livello di logging",
    )
    parser.add_argument(
        "--codex-url", help="URL del server Codex (override env var)"
    )

    args = parser.parse_args()

    # Configura
    config = BridgeConfig.from_env()
    config.log_level = LogLevel(args.log_level)

    if args.codex_url:
        config.codex_base_url = args.codex_url

    # Crea bridge
    try:
        bridge = ClaudeCodexBridge(config)
    except Exception as e:
        print(f"Errore inizializzazione bridge: {e}", file=sys.stderr)
        sys.exit(1)

    # Modalità interattiva
    if args.interactive or not args.message:
        bridge.interactive_mode()
        return

    # Modalità single-shot
    user_message = " ".join(args.message)

    try:
        response = bridge.chat(user_message)
        print(response)
    except Exception as e:
        bridge.logger.error(f"Errore durante chat: {e}", exc_info=True)
        print(f"Errore: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
