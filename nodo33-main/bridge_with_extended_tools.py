#!/usr/bin/env python3
"""
Bridge Claude-Codex con Tool Estesi Nodo33

Versione completa che integra:
- Tool base (image generation, status)
- Tool estesi spirituali-tecnici (blessings, sigilli, frequencies, etc.)

Usage:
    python3 bridge_with_extended_tools.py "Dammi una benedizione per il mio codice"
    python3 bridge_with_extended_tools.py "Genera un sigillo per Nodo33"
    python3 bridge_with_extended_tools.py -i  # Interactive mode
"""

from __future__ import annotations

import sys
from pathlib import Path

# Import bridge base
from claude_codex_bridge_v2 import (
    BridgeConfig,
    ClaudeCodexBridge,
    ClaudeClient,
    CodexClient,
    setup_logging,
)

# Import tool estesi
from codex_tools_extended import (
    EXTENDED_TOOLS,
    ExtendedToolExecutor,
)


class ExtendedClaudeCodexBridge(ClaudeCodexBridge):
    """Bridge con tool estesi Nodo33."""

    def __init__(self, config: BridgeConfig | None = None):
        super().__init__(config)
        self.extended_executor = ExtendedToolExecutor()

        # Aggiungi tool estesi ai tool disponibili di Claude
        self.claude.TOOLS.extend(EXTENDED_TOOLS)

        self.logger.info(
            f"Extended bridge initialized with {len(EXTENDED_TOOLS)} additional tools"
        )

    def _handle_tool_use(self, tool_use) -> dict:
        """Override per gestire sia tool base che estesi."""
        tool_name = getattr(tool_use, "name", "")
        tool_input = getattr(tool_use, "input", {}) or {}

        # Check se è un tool esteso
        extended_tool_names = [t["name"] for t in EXTENDED_TOOLS]

        if tool_name in extended_tool_names:
            return self._handle_extended_tool(tool_use, tool_name, tool_input)
        else:
            # Delega ai tool base del parent
            return super()._handle_tool_use(tool_use)

    def _handle_extended_tool(self, tool_use, tool_name: str, tool_input: dict) -> dict:
        """Gestisce esecuzione tool estesi."""
        self.logger.info(f"Executing extended tool: {tool_name}")

        try:
            result = self.extended_executor.execute(tool_name, tool_input)

            return {
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": [{"type": "text", "text": result}],
            }

        except Exception as e:
            error_msg = f"Errore esecuzione tool esteso '{tool_name}': {e}"
            self.logger.error(error_msg, exc_info=True)
            return {
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": [{"type": "text", "text": f"ERROR: {error_msg}"}],
                "is_error": True,
            }


def main() -> None:
    """Entry point con tool estesi."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Claude-Codex Bridge with Nodo33 Extended Tools",
        epilog='🕊️ Fiat Amor, Fiat Risus, Fiat Lux 🕊️',
    )
    parser.add_argument(
        "message",
        nargs="*",
        help="Messaggio da inviare a Claude",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Modalità interattiva",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Esegui demo dei tool estesi",
    )

    args = parser.parse_args()

    # Demo mode
    if args.demo:
        from codex_tools_extended import demo
        demo()
        return

    # Configura bridge
    from claude_codex_bridge_v2 import LogLevel

    config = BridgeConfig.from_env()
    config.log_level = LogLevel(args.log_level)

    try:
        bridge = ExtendedClaudeCodexBridge(config)
    except Exception as e:
        print(f"Errore inizializzazione: {e}", file=sys.stderr)
        sys.exit(1)

    # Interactive mode
    if args.interactive or not args.message:
        print("=== 🕊️ Claude-Codex Bridge + Nodo33 Extended Tools 🕊️ ===")
        print("\nTool disponibili:")
        print("  • Image generation (Stable Diffusion)")
        print("  • Server status query")
        print("  • Sasso blessings (benedizioni)")
        print("  • Sigillo generator (hash sacri)")
        print("  • Frequency analyzer (300 Hz)")
        print("  • Gift tracker (Regalo > Dominio)")
        print("  • Memory store (database sacro)")
        print("  • Lux calculator (quoziente di luce)")
        print("\nEsempi:")
        print('  "Dammi una benedizione per il mio nuovo progetto"')
        print('  "Genera un sigillo per Nodo33 Sasso Digitale"')
        print('  "Analizza la frequenza di: La luce non si vende"')
        print('  "Registra questo regalo: ho condiviso il bridge v2"')
        print('  "Calcola il lux quotient di: Fiat Lux 644"')
        print("\nDigita 'exit' per uscire, 'reset' per pulire conversazione.\n")

        bridge.interactive_mode()
        return

    # Single-shot mode
    user_message = " ".join(args.message)

    try:
        response = bridge.chat(user_message)
        print(response)
    except Exception as e:
        bridge.logger.error(f"Errore: {e}", exc_info=True)
        print(f"Errore: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
