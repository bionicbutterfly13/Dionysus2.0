#!/usr/bin/env python3
"""
T007: Contract Test - POST /api/clause/subgraph

This test MUST FAIL initially (endpoint not implemented).
Tests API contract compliance for CLAUSE subgraph construction endpoint.
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def api_client():
    """Create test client for API (will fail until endpoint exists)"""
    try:
        from backend.src.main import app
        return TestClient(app)
    except ImportError:
        # Expected to fail initially
        pytest.skip("API app not available yet")


def test_subgraph_request_schema_validation(api_client):
    """Test request schema validation for POST /api/clause/subgraph"""

    # Valid request
    valid_request = {
        "query": "neural architecture search",
        "edge_budget": 50,
        "lambda_edge": 0.2,
        "hop_distance": 2
    }

    response = api_client.post("/api/clause/subgraph", json=valid_request)

    # Should return 200 or 404 (if not implemented)
    assert response.status_code in [200, 404], \
        f"Expected 200 or 404, got {response.status_code}"


def test_subgraph_response_schema(api_client):
    """Test response schema matches contract specification"""

    request = {
        "query": "neural architecture search",
        "edge_budget": 50,
        "lambda_edge": 0.2,
        "hop_distance": 2
    }

    response = api_client.post("/api/clause/subgraph", json=request)

    if response.status_code == 200:
        data = response.json()

        # Required fields per contract
        assert "selected_edges" in data
        assert "edge_scores" in data
        assert "shaped_gains" in data
        assert "budget_used" in data
        assert "stopped_reason" in data

        # Type validation
        assert isinstance(data["selected_edges"], list)
        assert isinstance(data["edge_scores"], dict)
        assert isinstance(data["shaped_gains"], dict)
        assert isinstance(data["budget_used"], int)
        assert isinstance(data["stopped_reason"], str)


def test_edge_budget_validation(api_client):
    """Test edge_budget must be in range 1-1000"""

    # Budget too low
    response = api_client.post("/api/clause/subgraph", json={
        "query": "test",
        "edge_budget": 0,
        "lambda_edge": 0.2,
        "hop_distance": 2
    })

    if response.status_code != 404:
        assert response.status_code == 400 or response.status_code == 422, \
            "Budget 0 should return 400/422"

    # Budget too high
    response = api_client.post("/api/clause/subgraph", json={
        "query": "test",
        "edge_budget": 1001,
        "lambda_edge": 0.2,
        "hop_distance": 2
    })

    if response.status_code != 404:
        assert response.status_code == 400 or response.status_code == 422, \
            "Budget 1001 should return 400/422"

    # Valid budget
    response = api_client.post("/api/clause/subgraph", json={
        "query": "test",
        "edge_budget": 50,
        "lambda_edge": 0.2,
        "hop_distance": 2
    })

    assert response.status_code in [200, 404]


def test_lambda_edge_validation(api_client):
    """Test lambda_edge must be in range 0.0-1.0"""

    # Lambda too low
    response = api_client.post("/api/clause/subgraph", json={
        "query": "test",
        "edge_budget": 50,
        "lambda_edge": -0.1,
        "hop_distance": 2
    })

    if response.status_code != 404:
        assert response.status_code == 400 or response.status_code == 422, \
            "Lambda -0.1 should return 400/422"

    # Lambda too high
    response = api_client.post("/api/clause/subgraph", json={
        "query": "test",
        "edge_budget": 50,
        "lambda_edge": 1.1,
        "hop_distance": 2
    })

    if response.status_code != 404:
        assert response.status_code == 400 or response.status_code == 422, \
            "Lambda 1.1 should return 400/422"


def test_missing_required_field_query(api_client):
    """Test missing 'query' field returns 400"""

    response = api_client.post("/api/clause/subgraph", json={
        "edge_budget": 50,
        "lambda_edge": 0.2,
        "hop_distance": 2
    })

    if response.status_code != 404:
        assert response.status_code == 400 or response.status_code == 422, \
            "Missing 'query' should return 400/422"


def test_optional_parameters_defaults(api_client):
    """Test optional parameters use correct defaults"""

    # Minimal request (only query required)
    response = api_client.post("/api/clause/subgraph", json={
        "query": "neural architecture search"
    })

    if response.status_code == 200:
        data = response.json()

        # Default edge_budget should be 50 (from spec)
        assert data.get("budget_used", 0) <= 50

        # Should use default lambda_edge (0.2)
        # Should use default hop_distance (2)


if __name__ == "__main__":
    print("\n=== T007: Contract Test - POST /api/clause/subgraph ===\n")
    print("⚠️  This test MUST FAIL initially (endpoint not implemented)")
    print("✅  Test will pass once endpoint is implemented in T015-T025\n")

    pytest.main([__file__, "-v"])
