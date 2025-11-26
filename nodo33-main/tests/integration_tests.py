#!/usr/bin/env python3
"""
Integration Test Suite for Nodo33 Sasso Digitale
═══════════════════════════════════════════════════════════

Comprehensive test coverage for all systems:
- Memory Graph API (5 endpoints)
- Agent Monitoring (4 endpoints)
- System Diagnostics (1 endpoint)
- Audit Logging (3 endpoints)
- API Key Management (4 endpoints)
- Rate Limiting Dashboard (1 endpoint)

Run with: pytest tests/integration_tests.py -v
"""

import pytest
import requests
import json
from datetime import datetime, timedelta

BASE_URL = "http://localhost:8644"

# ═══════════════════════════════════════════════════════════
# FIXTURES & SETUP
# ═══════════════════════════════════════════════════════════

@pytest.fixture(scope="session")
def server_running():
    """Check if server is running."""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        assert response.status_code == 200
        return True
    except:
        pytest.skip("Server not running")

@pytest.fixture
def api_key():
    """Generate a test API key."""
    response = requests.post(
        f"{BASE_URL}/api/keys/generate",
        params={
            "name": "test-key",
            "permissions": ["read", "write"],
            "rate_limit": 10000
        }
    )
    data = response.json()
    return data.get("full_key_secret")

# ═══════════════════════════════════════════════════════════
# MEMORY GRAPH TESTS
# ═══════════════════════════════════════════════════════════

class TestMemoryGraph:
    """Test memory graph CRUD operations."""

    def test_create_memory(self, server_running):
        """Test creating a memory node."""
        response = requests.post(
            f"{BASE_URL}/api/memory/add",
            json={
                "endpoint": "/test",
                "content": "Test memory content",
                "memory_type": "test",
                "source_type": "cli",
                "tags": ["test", "integration"]
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["id"] > 0
        assert data["content"] == "Test memory content"

    def test_retrieve_memory(self, server_running):
        """Test retrieving a specific memory."""
        # First create a memory
        create_resp = requests.post(
            f"{BASE_URL}/api/memory/add",
            json={
                "endpoint": "/test",
                "content": "Retrieve test",
                "memory_type": "test"
            }
        )
        memory_id = create_resp.json()["id"]

        # Then retrieve it
        response = requests.get(f"{BASE_URL}/api/memory/retrieve/{memory_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == memory_id
        assert data["content"] == "Retrieve test"

    def test_search_memories(self, server_running):
        """Test searching memories."""
        response = requests.get(
            f"{BASE_URL}/api/memory/search",
            params={
                "memory_type": "test",
                "limit": 10
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "nodes" in data
        assert "edges" in data

    def test_memory_relations(self, server_running):
        """Test getting memory relations."""
        # Create two memories
        mem1 = requests.post(
            f"{BASE_URL}/api/memory/add",
            json={"endpoint": "/test", "content": "Memory 1"}
        ).json()

        mem2 = requests.post(
            f"{BASE_URL}/api/memory/add",
            json={"endpoint": "/test", "content": "Memory 2"}
        ).json()

        # Create relation
        requests.post(
            f"{BASE_URL}/api/memory/relation",
            json={
                "from_memory_id": mem1["id"],
                "to_memory_id": mem2["id"],
                "relation_type": "test_relation",
                "weight": 0.95
            }
        )

        # Get relations
        response = requests.get(f"{BASE_URL}/api/memory/relations/{mem1['id']}")
        assert response.status_code == 200
        data = response.json()
        assert len(data["relations"]) > 0

    def test_update_memory(self, server_running):
        """Test updating a memory."""
        # Create
        mem = requests.post(
            f"{BASE_URL}/api/memory/add",
            json={"endpoint": "/test", "content": "Original"}
        ).json()

        # Update
        response = requests.put(
            f"{BASE_URL}/api/memory/{mem['id']}",
            json={
                "endpoint": "/test",
                "content": "Updated content",
                "memory_type": "updated"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["content"] == "Updated content"

    def test_delete_memory(self, server_running):
        """Test deleting a memory."""
        mem = requests.post(
            f"{BASE_URL}/api/memory/add",
            json={"endpoint": "/test", "content": "To delete"}
        ).json()

        response = requests.delete(f"{BASE_URL}/api/memory/{mem['id']}")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

# ═══════════════════════════════════════════════════════════
# AGENT MONITORING TESTS
# ═══════════════════════════════════════════════════════════

class TestAgentMonitoring:
    """Test agent monitoring and management."""

    def test_list_agents(self, server_running):
        """Test listing agents."""
        response = requests.get(f"{BASE_URL}/api/agents/list")
        assert response.status_code == 200
        data = response.json()
        assert "total_agents" in data
        assert "filtered_agents" in data
        assert "agents" in data

    def test_list_agents_with_filters(self, server_running):
        """Test listing agents with filters."""
        response = requests.get(
            f"{BASE_URL}/api/agents/list",
            params={"status_filter": "active"}
        )
        assert response.status_code == 200
        data = response.json()
        # Check that filtered agents <= total agents
        assert data["filtered_agents"] <= data["total_agents"]

    def test_agent_dashboard(self, server_running):
        """Test agent dashboard."""
        response = requests.get(f"{BASE_URL}/api/agents/dashboard")
        assert response.status_code == 200
        data = response.json()
        assert "agent_statistics" in data
        assert "global_metrics" in data
        assert "domain_statistics" in data

# ═══════════════════════════════════════════════════════════
# SYSTEM DIAGNOSTICS TESTS
# ═══════════════════════════════════════════════════════════

class TestSystemDiagnostics:
    """Test system health and diagnostics."""

    def test_health_check(self, server_running):
        """Test basic health check."""
        response = requests.get(f"{BASE_URL}/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "uptime_s" in data

    def test_diagnostics(self, server_running):
        """Test full diagnostics."""
        response = requests.get(f"{BASE_URL}/api/diagnostics")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "server" in data
        assert "requests" in data
        assert "database" in data

    def test_diagnostics_status_values(self, server_running):
        """Test diagnostics returns valid status."""
        response = requests.get(f"{BASE_URL}/api/diagnostics")
        data = response.json()
        assert data["status"] in ["healthy", "degraded", "critical"]

# ═══════════════════════════════════════════════════════════
# AUDIT LOGGING TESTS
# ═══════════════════════════════════════════════════════════

class TestAuditLogging:
    """Test audit logging system."""

    def test_get_audit_logs(self, server_running):
        """Test retrieving audit logs."""
        response = requests.get(
            f"{BASE_URL}/api/audit/logs",
            params={"limit": 10}
        )
        assert response.status_code == 200
        data = response.json()
        assert "total_logs" in data
        assert "logs" in data

    def test_audit_logs_with_filters(self, server_running):
        """Test audit logs with filters."""
        response = requests.get(
            f"{BASE_URL}/api/audit/logs",
            params={
                "severity_filter": "critical",
                "time_range_hours": 24
            }
        )
        assert response.status_code == 200
        data = response.json()
        # All logs should match filter
        for log in data["logs"]:
            assert log["severity"] == "critical"

    def test_audit_summary(self, server_running):
        """Test audit summary."""
        response = requests.get(f"{BASE_URL}/api/audit/summary")
        assert response.status_code == 200
        data = response.json()
        assert "statistics" in data
        assert "by_event_type" in data
        assert "by_severity" in data

    def test_audit_events(self, server_running):
        """Test getting available event types."""
        response = requests.get(f"{BASE_URL}/api/audit/events")
        assert response.status_code == 200
        data = response.json()
        assert len(data["event_types"]) > 0

# ═══════════════════════════════════════════════════════════
# API KEY MANAGEMENT TESTS
# ═══════════════════════════════════════════════════════════

class TestAPIKeyManagement:
    """Test API key management."""

    def test_generate_api_key(self, server_running):
        """Test generating API key."""
        response = requests.post(
            f"{BASE_URL}/api/keys/generate",
            params={
                "name": "integration-test-key",
                "permissions": ["read", "write"],
                "rate_limit": 1000
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] == True
        assert "full_key_secret" in data
        assert data["name"] == "integration-test-key"

    def test_list_api_keys(self, server_running):
        """Test listing API keys."""
        response = requests.get(f"{BASE_URL}/api/keys/list")
        assert response.status_code == 200
        data = response.json()
        assert "total_keys" in data
        assert "keys" in data

    def test_get_key_info(self, server_running, api_key):
        """Test getting key information."""
        # Extract key_id from full_key
        key_id = api_key.split(":")[0]

        response = requests.get(f"{BASE_URL}/api/keys/{key_id}/info")
        assert response.status_code == 200
        data = response.json()
        assert data["key_id"] == key_id
        assert "permissions" in data

# ═══════════════════════════════════════════════════════════
# RATE LIMITING TESTS
# ═══════════════════════════════════════════════════════════

class TestRateLimiting:
    """Test rate limiting dashboard."""

    def test_rate_limit_dashboard(self, server_running):
        """Test rate limit dashboard."""
        response = requests.get(f"{BASE_URL}/api/rate-limit/dashboard")
        assert response.status_code == 200
        data = response.json()
        assert "global" in data
        assert "api_key_limits" in data
        assert "health_check" in data

    def test_rate_limit_dashboard_structure(self, server_running):
        """Test dashboard returns expected structure."""
        response = requests.get(f"{BASE_URL}/api/rate-limit/dashboard")
        data = response.json()

        # Check global stats
        assert "requests_last_1h" in data["global"]
        assert "avg_requests_per_minute" in data["global"]

        # Check API key limits
        assert isinstance(data["api_key_limits"]["details"], list)
        assert "health_check" in data

# ═══════════════════════════════════════════════════════════
# END-TO-END TESTS
# ═══════════════════════════════════════════════════════════

class TestEndToEnd:
    """End-to-end workflow tests."""

    def test_complete_workflow(self, server_running):
        """Test a complete workflow."""
        # 1. Check system health
        health = requests.get(f"{BASE_URL}/health").json()
        assert health["status"] in ["ok", "degraded"]

        # 2. Create memory
        mem = requests.post(
            f"{BASE_URL}/api/memory/add",
            json={"endpoint": "/test", "content": "E2E test"}
        ).json()

        # 3. Retrieve memory
        retrieved = requests.get(f"{BASE_URL}/api/memory/retrieve/{mem['id']}").json()
        assert retrieved["id"] == mem["id"]

        # 4. Get diagnostics
        diag = requests.get(f"{BASE_URL}/api/diagnostics").json()
        assert "database" in diag
        assert diag["database"]["memory_nodes"] > 0

        # 5. Check audit logs
        audit = requests.get(f"{BASE_URL}/api/audit/logs").json()
        assert "logs" in audit

        # 6. Check rate limits
        rate = requests.get(f"{BASE_URL}/api/rate-limit/dashboard").json()
        assert "global" in rate

# ═══════════════════════════════════════════════════════════
# PYTEST CONFIGURATION
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
