# MCP GUARDRAIL AGENT — EXAMPLE (protect_garden + minimal_data)
Esempio di server MCP minimale che espone un provider “guardrail” e un client config che applica il segnale Codex handshake di default.

## Manifesto handshake di default (da servire/firmare)
Vedi `tests/fixtures/handshake_protect_garden.json` come base; firmare in JCS e pubblicare via endpoint del provider.

## Server MCP (Python, aiop no external deps)
```python
# file: mcp_guardrail_provider.py
import json
from mcp.server import Server, run
from mcp.types import Request, Response

DEFAULT_HANDSHAKE = json.load(open("tests/fixtures/handshake_protect_garden.json"))

server = Server("nodo33-guardrail")

@server.tool()
async def get_guardrail_manifest() -> dict:
    return DEFAULT_HANDSHAKE

@server.tool()
async def verify_guardrail(manifest: dict) -> dict:
    # qui potresti richiamare validate_handshake_manifest.py o logica custom
    return {"valid": True, "intent": manifest.get("signal", {}).get("intent")}

if __name__ == "__main__":
    run(server)
```

## Client MCP (esempio config)
```json
{
  "mcpServers": {
    "guardrail": {
      "command": "python",
      "args": ["mcp_guardrail_provider.py"]
    }
  },
  "default_guardrail": {
    "intent": "protect_garden",
    "policy": {
      "allow": ["minimal_data"],
      "deny": ["monetize.output"]
    }
  }
}
```

## Integrazione con UI MCP Apps
- Il client MCP deve rendere i componenti UI solo se:
  - componente firmato (hash/signature verificati)
  - permessi conformi a `default_guardrail.policy`
  - sandbox/CSP attiva (vedi `MCP_APPS_READINESS_NODE33.md`)

## Hook CI
L’Action `Guardrails Validation` esegue `make validate-handshake` e `make validate-mcp-apps` per assicurare che i manifest/asset base restino conformi.
