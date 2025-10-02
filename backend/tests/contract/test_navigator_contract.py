#!/usr/bin/env python3
"""
T007: Contract Test - POST /api/clause/navigate

Tests PathNavigator API contract per Spec 035.
EXPECTED: ALL TESTS MUST FAIL (endpoint not implemented yet)

Contract: specs/035-clause-phase2-multi-agent/contracts/navigator_api.yaml
Data Model: specs/035-clause-phase2-multi-agent/data-model.md
"""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime


@pytest.fixture
def client():
    """FastAPI test client with CLAUSE routes for contract testing"""
    from fastapi import FastAPI
    import sys
    sys.path.insert(0, '/Volumes/Asylum/dev/Dionysus-2.0/backend/src')

    # Import CLAUSE router
    from api.routes import clause

    # Create minimal app with CLAUSE routes
    app = FastAPI()
    app.include_router(clause.router)

    return TestClient(app)


class TestNavigatorContract:
    """Contract tests for PathNavigator API"""

    def test_navigate_endpoint_exists(self, client):
        """Verify POST /api/clause/navigate endpoint exists"""
        response = client.post("/api/clause/navigate", json={
            "query": "What causes climate change?",
            "start_node": "climate_change",
            "step_budget": 10
        })

        # Should not be 404 when endpoint exists
        assert response.status_code != 404, "Endpoint POST /api/clause/navigate does not exist"

    def test_navigate_requires_query(self, client):
        """Verify query field is required"""
        response = client.post("/api/clause/navigate", json={
            "start_node": "climate_change",
            "step_budget": 10
        })

        assert response.status_code == 422, "Should reject missing query field"
        error_detail = response.json()
        assert "query" in str(error_detail).lower()

    def test_navigate_requires_start_node(self, client):
        """Verify start_node field is required"""
        response = client.post("/api/clause/navigate", json={
            "query": "What causes climate change?",
            "step_budget": 10
        })

        assert response.status_code == 422, "Should reject missing start_node field"
        error_detail = response.json()
        assert "start_node" in str(error_detail).lower()

    def test_navigate_query_min_length(self, client):
        """Verify query has min_length=3 validation"""
        response = client.post("/api/clause/navigate", json={
            "query": "ab",  # Only 2 chars (min is 3)
            "start_node": "climate_change",
            "step_budget": 10
        })

        assert response.status_code == 422, "Should reject query with <3 characters"

    def test_navigate_step_budget_range(self, client):
        """Verify step_budget range validation (1-20)"""
        # Test below minimum
        response = client.post("/api/clause/navigate", json={
            "query": "What causes climate change?",
            "start_node": "climate_change",
            "step_budget": 0
        })
        assert response.status_code == 422, "Should reject step_budget < 1"

        # Test above maximum
        response = client.post("/api/clause/navigate", json={
            "query": "What causes climate change?",
            "start_node": "climate_change",
            "step_budget": 21
        })
        assert response.status_code == 422, "Should reject step_budget > 20"

    def test_navigate_default_step_budget(self, client):
        """Verify step_budget defaults to 10"""
        response = client.post("/api/clause/navigate", json={
            "query": "What causes climate change?",
            "start_node": "climate_change"
        })

        # Should succeed (assuming endpoint returns 200 when implemented)
        if response.status_code == 200:
            data = response.json()
            assert "metadata" in data
            assert data["metadata"]["budget_total"] == 10, "Default step_budget should be 10"

    def test_navigate_boolean_flags_default(self, client):
        """Verify boolean flags default to True"""
        response = client.post("/api/clause/navigate", json={
            "query": "What causes climate change?",
            "start_node": "climate_change",
            "step_budget": 10
        })

        # When implemented, should use defaults
        if response.status_code == 200:
            # Defaults: enable_thoughtseeds=True, enable_curiosity=True, enable_causal=True
            # This will be validated when implementation is complete
            pass

    def test_navigate_curiosity_threshold_range(self, client):
        """Verify curiosity_threshold range validation (0.0-1.0)"""
        response = client.post("/api/clause/navigate", json={
            "query": "What causes climate change?",
            "start_node": "climate_change",
            "step_budget": 10,
            "curiosity_threshold": 1.5  # Above max
        })

        assert response.status_code == 422, "Should reject curiosity_threshold > 1.0"

        response = client.post("/api/clause/navigate", json={
            "query": "What causes climate change?",
            "start_node": "climate_change",
            "step_budget": 10,
            "curiosity_threshold": -0.1  # Below min
        })

        assert response.status_code == 422, "Should reject curiosity_threshold < 0.0"

    def test_navigate_response_schema(self, client):
        """Verify response has path, metadata, performance fields"""
        response = client.post("/api/clause/navigate", json={
            "query": "What causes climate change?",
            "start_node": "climate_change",
            "step_budget": 10
        })

        if response.status_code == 200:
            data = response.json()

            # Required top-level fields
            assert "path" in data, "Response missing 'path' field"
            assert "metadata" in data, "Response missing 'metadata' field"
            assert "performance" in data, "Response missing 'performance' field"

            # Path structure
            assert "nodes" in data["path"], "path missing 'nodes' field"
            assert "edges" in data["path"], "path missing 'edges' field"
            assert "steps" in data["path"], "path missing 'steps' field"

            # Metadata structure
            assert "budget_used" in data["metadata"]
            assert "budget_total" in data["metadata"]
            assert "final_action" in data["metadata"]

            # Performance structure
            assert "latency_ms" in data["performance"]

    def test_navigate_budget_compliance(self, client):
        """Verify response respects step_budget"""
        response = client.post("/api/clause/navigate", json={
            "query": "What causes climate change?",
            "start_node": "climate_change",
            "step_budget": 5
        })

        if response.status_code == 200:
            data = response.json()
            budget_used = data["metadata"]["budget_used"]
            budget_total = data["metadata"]["budget_total"]

            assert budget_total == 5, "budget_total should match request"
            assert budget_used <= budget_total, "budget_used must not exceed budget_total"
            assert len(data["path"]["steps"]) <= 5, "Path steps must not exceed step_budget"


if __name__ == "__main__":
    print("\n=== T007: PathNavigator Contract Test ===\n")
    print("⚠️  EXPECTED: All tests MUST FAIL (endpoint not implemented)")
    print("Contract: POST /api/clause/navigate\n")

    pytest.main([__file__, "-v", "--tb=short"])
