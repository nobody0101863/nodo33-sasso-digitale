# ADR-002: Model Context Protocol (MCP) Integration

**Date**: 2025-11-18
**Status**: Accepted
**Context**: Claude Desktop tool integration
**Decision Makers**: Nodo33 Team

---

## Context

Extended tools (blessings, sigilli, etc.) were Python-only, requiring:
- Manual script execution
- No integration with Claude Desktop
- CLI-based interaction only

**Goal**: Make tools available natively in Claude Desktop's tool picker.

## Decision

Implement **MCP Server** following [Model Context Protocol](https://modelcontextprotocol.io) specification.

### Architecture

```
Claude Desktop
    ↓ (stdio)
codex_mcp_server.py
    ↓
ExtendedToolExecutor
    ↓
[6 Extended Tools]
```

### Transport: stdio

- **Why**: Standard MCP transport, simplest to integrate
- **How**: JSON-RPC over stdin/stdout
- **Pros**: No network config, no ports, secure by default

### Supported Methods

- `initialize`: Handshake and capability exchange
- `tools/list`: Return available tools
- `tools/call`: Execute a tool
- `ping`: Health check

## Implementation

**File**: `codex_mcp_server.py`
**Class**: `CodexMCPServer`
**Protocol**: JSON-RPC 2.0
**Version**: MCP 2024-11-05

### Integration with Claude Desktop

macOS config file:
```json
{
  "mcpServers": {
    "codex-nodo33": {
      "command": "python3",
      "args": ["/path/to/codex_mcp_server.py"],
      "env": {
        "PYTHONPATH": "/path/to/nodo33-main"
      }
    }
  }
}
```

## Consequences

### Positive

✅ **Native Claude Desktop integration**: Tools appear in UI
✅ **Zero-config for end user**: Just add to config file
✅ **Automatic discovery**: Tools self-describe via schema
✅ **Secure**: No network exposure, local stdio only
✅ **Standard protocol**: Works with any MCP client

### Negative

⚠️ **Debugging harder**: stdio transport less visible
⚠️ **Claude Desktop required**: Can't use in browser Claude
⚠️ **Path dependencies**: Needs correct PYTHONPATH

### Neutral

🔵 **Performance**: Slight overhead vs direct calls (negligible)
🔵 **Logging**: Goes to stderr, separate from protocol

## Alternatives Considered

1. **HTTP Server**: Rejected (unnecessary complexity, port conflicts)
2. **WebSocket**: Rejected (not standard MCP transport)
3. **Direct API integration**: Rejected (not extensible)

## Testing

```bash
# Test mode (CLI)
python3 codex_mcp_server.py --test

# Integration guide
python3 codex_mcp_server.py --guide

# Production (stdio)
python3 codex_mcp_server.py
```

## Future Enhancements

- [ ] Support `prompts/*` methods for prompt templates
- [ ] Support `resources/*` for data sources
- [ ] Add `logging/*` for better observability
- [ ] Multi-transport support (stdio + HTTP for web clients)

## Notes

- All 6 extended tools exposed via MCP
- Base tools (image generation) could be added later
- Tool schemas come from `EXTENDED_TOOLS` definitions
- Error handling follows JSON-RPC 2.0 spec

## References

- [MCP Specification](https://modelcontextprotocol.io)
- [Claude Desktop MCP Guide](https://docs.anthropic.com/claude/docs/mcp)
- [JSON-RPC 2.0](https://www.jsonrpc.org/specification)

---

**Hash Sacro**: 644
**Frequenza**: 300 Hz
*"Regalo > Dominio"*
