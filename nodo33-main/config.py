#!/usr/bin/env python3
"""
Codex Configuration Loader

Centralizes configuration loading from .env files.
Uses python-dotenv for environment variable management.

Usage:
    from config import load_config, get_config

    # Load at startup
    load_config()

    # Access anywhere
    config = get_config()
    print(config.ANTHROPIC_API_KEY)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv

    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


class Config:
    """Configuration container."""

    # Claude
    ANTHROPIC_API_KEY: str
    CLAUDE_MODEL: str = "claude-3-5-sonnet-20241022"
    CLAUDE_MAX_TOKENS: int = 2048

    # Codex Server
    CODEX_BASE_URL: str = "http://localhost:8644"
    CODEX_TIMEOUT: int = 120
    CODEX_MAX_RETRIES: int = 3

    # Security
    MAX_PROMPT_LENGTH: int = 5000
    MAX_IMAGE_STEPS: int = 50
    VALIDATE_SSL: bool = True

    # Logging
    BRIDGE_LOG_LEVEL: str = "INFO"
    BRIDGE_LOG_FILE: Optional[str] = None

    # Database
    CODEX_DB_PATH: str = "codex_unified.db"
    ENABLE_CONVERSATION_HISTORY: bool = True
    MAX_CONVERSATION_TURNS: int = 10

    # Nodo33
    DEFAULT_MODE: str = "complete"
    SACRED_HASH: str = "644"
    SACRED_FREQUENCY: str = "300"
    NODO33_MOTTO: str = "La luce non si vende. La si regala."

    # Dev/Test
    DEBUG: bool = False
    TEST_MODE: bool = False

    # Metrics
    ENABLE_METRICS: bool = True
    METRICS_RETENTION_DAYS: int = 30

    # MCP
    MCP_ENABLED: bool = True
    MCP_LOG_LEVEL: str = "INFO"

    @classmethod
    def load_from_env(cls) -> Config:
        """Load configuration from environment variables."""
        config = cls()

        # Load all env vars with type conversion
        config.ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
        config.CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", config.CLAUDE_MODEL)
        config.CLAUDE_MAX_TOKENS = int(
            os.getenv("CLAUDE_MAX_TOKENS", config.CLAUDE_MAX_TOKENS)
        )

        config.CODEX_BASE_URL = os.getenv("CODEX_BASE_URL", config.CODEX_BASE_URL)
        config.CODEX_TIMEOUT = int(os.getenv("CODEX_TIMEOUT", config.CODEX_TIMEOUT))
        config.CODEX_MAX_RETRIES = int(
            os.getenv("CODEX_MAX_RETRIES", config.CODEX_MAX_RETRIES)
        )

        config.MAX_PROMPT_LENGTH = int(
            os.getenv("MAX_PROMPT_LENGTH", config.MAX_PROMPT_LENGTH)
        )
        config.MAX_IMAGE_STEPS = int(
            os.getenv("MAX_IMAGE_STEPS", config.MAX_IMAGE_STEPS)
        )
        config.VALIDATE_SSL = os.getenv("VALIDATE_SSL", "true").lower() in (
            "true",
            "1",
            "yes",
        )

        config.BRIDGE_LOG_LEVEL = os.getenv("BRIDGE_LOG_LEVEL", config.BRIDGE_LOG_LEVEL)
        config.BRIDGE_LOG_FILE = os.getenv("BRIDGE_LOG_FILE")

        config.CODEX_DB_PATH = os.getenv("CODEX_DB_PATH", config.CODEX_DB_PATH)
        config.ENABLE_CONVERSATION_HISTORY = os.getenv(
            "ENABLE_CONVERSATION_HISTORY", "true"
        ).lower() in ("true", "1", "yes")
        config.MAX_CONVERSATION_TURNS = int(
            os.getenv("MAX_CONVERSATION_TURNS", config.MAX_CONVERSATION_TURNS)
        )

        config.DEFAULT_MODE = os.getenv("DEFAULT_MODE", config.DEFAULT_MODE)
        config.SACRED_HASH = os.getenv("SACRED_HASH", config.SACRED_HASH)
        config.SACRED_FREQUENCY = os.getenv("SACRED_FREQUENCY", config.SACRED_FREQUENCY)
        config.NODO33_MOTTO = os.getenv("NODO33_MOTTO", config.NODO33_MOTTO)

        config.DEBUG = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")
        config.TEST_MODE = os.getenv("TEST_MODE", "false").lower() in (
            "true",
            "1",
            "yes",
        )

        config.ENABLE_METRICS = os.getenv("ENABLE_METRICS", "true").lower() in (
            "true",
            "1",
            "yes",
        )
        config.METRICS_RETENTION_DAYS = int(
            os.getenv("METRICS_RETENTION_DAYS", config.METRICS_RETENTION_DAYS)
        )

        config.MCP_ENABLED = os.getenv("MCP_ENABLED", "true").lower() in (
            "true",
            "1",
            "yes",
        )
        config.MCP_LOG_LEVEL = os.getenv("MCP_LOG_LEVEL", config.MCP_LOG_LEVEL)

        return config

    def validate(self) -> None:
        """Validate required configuration."""
        errors = []

        if not self.ANTHROPIC_API_KEY and not self.TEST_MODE:
            errors.append("ANTHROPIC_API_KEY is required (unless TEST_MODE=true)")

        if self.CODEX_TIMEOUT < 10:
            errors.append("CODEX_TIMEOUT must be >= 10 seconds")

        if self.MAX_PROMPT_LENGTH < 100:
            errors.append("MAX_PROMPT_LENGTH must be >= 100")

        if errors:
            raise ValueError(
                "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            )

    def __repr__(self) -> str:
        """Safe repr without exposing secrets."""
        return (
            f"Config(\n"
            f"  CLAUDE_MODEL={self.CLAUDE_MODEL},\n"
            f"  CODEX_BASE_URL={self.CODEX_BASE_URL},\n"
            f"  LOG_LEVEL={self.BRIDGE_LOG_LEVEL},\n"
            f"  MODE={self.DEFAULT_MODE},\n"
            f"  HASH={self.SACRED_HASH},\n"
            f"  FREQUENCY={self.SACRED_FREQUENCY} Hz\n"
            f")"
        )


# Global config instance
_config: Optional[Config] = None


def load_config(env_file: Optional[Path] = None) -> Config:
    """
    Load configuration from .env file and environment variables.

    Args:
        env_file: Path to .env file (default: .env in current directory)

    Returns:
        Loaded and validated Config instance
    """
    global _config

    # Load .env file if available
    if DOTENV_AVAILABLE:
        if env_file is None:
            env_file = Path.cwd() / ".env"

        if env_file.exists():
            load_dotenv(env_file)
            print(f"✅ Loaded configuration from {env_file}")
        else:
            print(f"⚠️  No .env file found at {env_file}, using environment variables only")
    else:
        print("⚠️  python-dotenv not installed, using environment variables only")
        print("   Install with: pip install python-dotenv")

    # Load config
    _config = Config.load_from_env()

    # Validate
    try:
        _config.validate()
        print("✅ Configuration validated successfully")
    except ValueError as e:
        print(f"❌ Configuration validation failed:\n{e}")
        raise

    return _config


def get_config() -> Config:
    """
    Get current config instance.

    Raises:
        RuntimeError: If config hasn't been loaded yet
    """
    if _config is None:
        raise RuntimeError(
            "Configuration not loaded. Call load_config() first."
        )
    return _config


def reload_config() -> Config:
    """Reload configuration (useful for tests)."""
    global _config
    _config = None
    return load_config()


# ============================================================================
# CLI
# ============================================================================


def main() -> None:
    """CLI for configuration management."""
    import argparse

    parser = argparse.ArgumentParser(description="Codex Configuration Manager")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show current configuration",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate configuration",
    )
    parser.add_argument(
        "--create-env",
        action="store_true",
        help="Create .env from .env.example",
    )

    args = parser.parse_args()

    if args.create_env:
        example_path = Path(".env.example")
        env_path = Path(".env")

        if not example_path.exists():
            print("❌ .env.example not found")
            return

        if env_path.exists():
            response = input(".env already exists. Overwrite? (y/N): ")
            if response.lower() != "y":
                print("Aborted.")
                return

        env_path.write_text(example_path.read_text())
        print(f"✅ Created {env_path}")
        print("📝 Edit .env and fill in your values")
        return

    # Load and show/validate
    try:
        config = load_config()

        if args.show:
            print("\n📋 Current Configuration:")
            print(config)
            print(f"\n🕊️ {config.NODO33_MOTTO}")

        if args.validate:
            config.validate()
            print("\n✅ All configuration is valid!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    main()
