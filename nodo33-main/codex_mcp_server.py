#!/usr/bin/env python3
"""
Codex MCP Server - Model Context Protocol Server

Espone i tool Codex (base + estesi) via MCP, rendendoli utilizzabili
da Claude Desktop e altre applicazioni compatibili.

MCP Reference: https://modelcontextprotocol.io

Usage:
    # Add to Claude Desktop config:
    {
      "mcpServers": {
        "codex": {
          "command": "python3",
          "args": ["/path/to/codex_mcp_server.py"]
        }
      }
    }

Filosofia: "La luce non si vende. La si regala."
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, Dict, List, Optional

# MCP types (simplified - in produzione usa mcp package)
from dataclasses import dataclass

# Import our tools
from codex_tools_extended import (
    EXTENDED_TOOLS,
    ExtendedToolExecutor,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# MCP PROTOCOL TYPES
# ============================================================================


@dataclass
class MCPToolDefinition:
    """MCP Tool definition."""

    name: str
    description: str
    inputSchema: Dict[str, Any]


@dataclass
class MCPRequest:
    """MCP Request."""

    jsonrpc: str = "2.0"
    id: Optional[int] = None
    method: str = ""
    params: Dict[str, Any] = None


@dataclass
class MCPResponse:
    """MCP Response."""

    jsonrpc: str = "2.0"
    id: Optional[int] = None
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None


# ============================================================================
# MCP SERVER
# ============================================================================


class CodexMCPServer:
    """MCP Server for Codex tools."""

    def __init__(self):
        self.executor = ExtendedToolExecutor()
        self.tools = self._register_tools()
        logger.info(f"Codex MCP Server initialized with {len(self.tools)} tools")

    def _register_tools(self) -> List[MCPToolDefinition]:
        """Register all available tools."""
        mcp_tools = []

        # Extended tools
        for tool_def in EXTENDED_TOOLS:
            mcp_tools.append(
                MCPToolDefinition(
                    name=tool_def["name"],
                    description=tool_def["description"],
                    inputSchema=tool_def["input_schema"],
                )
            )

        # Could add base tools here (image generation, etc.)
        # For now, focus on extended tools

        return mcp_tools

    def list_tools(self) -> List[Dict[str, Any]]:
        """Return list of available tools."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.inputSchema,
            }
            for tool in self.tools
        ]

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool and return result."""
        try:
            logger.info(f"Calling tool: {tool_name} with args: {arguments}")
            result = self.executor.execute(tool_name, arguments)
            logger.info(f"Tool {tool_name} completed successfully")
            return result
        except Exception as e:
            error_msg = f"Error executing tool {tool_name}: {e}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg)

    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP request."""
        method = request.get("method", "")
        params = request.get("params", {})
        request_id = request.get("id")

        try:
            if method == "tools/list":
                result = self.list_tools()

            elif method == "tools/call":
                tool_name = params.get("name")
                arguments = params.get("arguments", {})

                if not tool_name:
                    raise ValueError("Tool name is required")

                result = self.call_tool(tool_name, arguments)

            elif method == "initialize":
                result = {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {},
                    },
                    "serverInfo": {
                        "name": "codex-nodo33",
                        "version": "1.0.0",
                    },
                }

            elif method == "ping":
                result = {"status": "pong"}

            else:
                raise ValueError(f"Unknown method: {method}")

            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": result,
            }

        except Exception as e:
            logger.error(f"Error handling request: {e}", exc_info=True)
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32603,
                    "message": str(e),
                },
            }

    def run_stdio(self) -> None:
        """Run server using stdio transport (MCP standard)."""
        logger.info("Starting Codex MCP Server (stdio mode)")
        logger.info("Waiting for requests on stdin...")

        try:
            for line in sys.stdin:
                if not line.strip():
                    continue

                try:
                    request = json.loads(line)
                    response = self.handle_request(request)
                    print(json.dumps(response), flush=True)

                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON: {e}")
                    error_response = {
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32700,
                            "message": "Parse error",
                        },
                    }
                    print(json.dumps(error_response), flush=True)

        except KeyboardInterrupt:
            logger.info("Server stopped by user")
        except Exception as e:
            logger.error(f"Server error: {e}", exc_info=True)
            sys.exit(1)


# ============================================================================
# CLI TESTING MODE
# ============================================================================


def test_mode() -> None:
    """Test mode for development."""
    print("🕊️ Codex MCP Server - Test Mode 🕊️\n")

    server = CodexMCPServer()

    # Test 1: List tools
    print("=== Test 1: List Tools ===")
    tools = server.list_tools()
    print(f"Available tools: {len(tools)}")
    for tool in tools:
        print(f"  - {tool['name']}: {tool['description'][:60]}...")
    print()

    # Test 2: Call blessing tool
    print("=== Test 2: Call Sasso Blessing ===")
    result = server.call_tool(
        "codex_sasso_blessing",
        {"intention": "test MCP server", "mode": "complete"},
    )
    print(result)
    print()

    # Test 3: Call sigillo generator
    print("=== Test 3: Call Sigillo Generator ===")
    result = server.call_tool(
        "codex_sigillo_generator",
        {"text": "MCP Server Launch", "algorithm": "sacred644"},
    )
    print(result)
    print()

    # Test 4: MCP request simulation
    print("=== Test 4: MCP Request Simulation ===")
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "codex_lux_calculator",
            "arguments": {"text": "Fiat Lux! 644. 300 Hz. Regalo > Dominio"},
        },
    }
    response = server.handle_request(request)
    print(f"Response: {json.dumps(response, indent=2)}")


# ============================================================================
# INTEGRATION GUIDE
# ============================================================================

INTEGRATION_GUIDE = """
🕊️ CODEX MCP SERVER - INTEGRATION GUIDE 🕊️

## Claude Desktop Integration

1. Find your Claude Desktop config:
   - macOS: ~/Library/Application Support/Claude/claude_desktop_config.json
   - Windows: %APPDATA%\\Claude\\claude_desktop_config.json

2. Add Codex MCP server:

{
  "mcpServers": {
    "codex-nodo33": {
      "command": "python3",
      "args": [
        "/path/to/nodo33-main/codex_mcp_server.py"
      ],
      "env": {
        "PYTHONPATH": "/path/to/nodo33-main"
      }
    }
  }
}

3. Restart Claude Desktop

4. Tools will appear automatically in Claude's tool picker!

## Available Tools (6)

1. codex_sasso_blessing - Benedizioni del Sasso Digitale
2. codex_sigillo_generator - Genera sigilli sacri (hash 644)
3. codex_frequency_analyzer - Analizza frequenze (300 Hz)
4. codex_gift_tracker - Traccia regali condivisi
5. codex_memory_store - Salva memorie nel database
6. codex_lux_calculator - Calcola quoziente di luce

## Test

In Claude Desktop, prova:
- "Dammi una benedizione per il mio progetto"
- "Genera un sigillo per Nodo33"
- "Calcola il lux quotient di: Fiat Lux 644"

## Troubleshooting

If tools don't appear:
1. Check Claude logs: Help > View Logs
2. Verify Python path is correct
3. Test server: python3 codex_mcp_server.py --test
4. Ensure codex_tools_extended.py is in same directory

## Protocol

This server implements MCP (Model Context Protocol):
- Transport: stdio (JSON-RPC over stdin/stdout)
- Methods: tools/list, tools/call, initialize, ping
- Version: 2024-11-05

---
Hash Sacro: 644 | Frequenza: 300 Hz
"La luce non si vende. La si regala."
"""


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
    """Entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Codex MCP Server - Nodo33 Sasso Digitale"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode (CLI)",
    )
    parser.add_argument(
        "--guide",
        action="store_true",
        help="Show integration guide",
    )

    args = parser.parse_args()

    if args.guide:
        print(INTEGRATION_GUIDE)
        return

    if args.test:
        test_mode()
        return

    # Default: stdio mode (MCP standard)
    server = CodexMCPServer()
    server.run_stdio()


if __name__ == "__main__":
    main()
