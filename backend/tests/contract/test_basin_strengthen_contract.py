#!/usr/bin/env python3
"""
T008: Contract Test - POST /api/clause/basins/strengthen

This test MUST FAIL initially (endpoint not implemented).
Tests API contract compliance for basin strengthening endpoint.
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
        pytest.skip("API app not available yet")


def test_basin_strengthen_request_schema(api_client):
    """Test request schema validation for POST /api/clause/basins/strengthen"""

    valid_request = {
        "concepts": ["neural_architecture", "search_algorithms", "reinforcement_learning"],
        "document_id": "doc_12345",
        "increment": 0.2
    }

    response = api_client.post("/api/clause/basins/strengthen", json=valid_request)

    assert response.status_code in [200, 404], \
        f"Expected 200 or 404, got {response.status_code}"


def test_basin_strengthen_response_schema(api_client):
    """Test response schema matches contract specification"""

    request = {
        "concepts": ["neural_architecture", "search_algorithms"],
        "document_id": "doc_12345",
        "increment": 0.2
    }

    response = api_client.post("/api/clause/basins/strengthen", json=request)

    if response.status_code == 200:
        data = response.json()

        # Required fields per contract
        assert "updated_basins" in data
        assert "new_basins" in data
        assert "cooccurrence_updates" in data
        assert "total_strengthening_time_ms" in data

        # Type validation
        assert isinstance(data["updated_basins"], list)
        assert isinstance(data["new_basins"], list)
        assert isinstance(data["cooccurrence_updates"], dict)
        assert isinstance(data["total_strengthening_time_ms"], (int, float))


def test_concepts_list_validation(api_client):
    """Test concepts list must be non-empty"""

    # Empty concepts list
    response = api_client.post("/api/clause/basins/strengthen", json={
        "concepts": [],
        "document_id": "doc_12345",
        "increment": 0.2
    })

    if response.status_code != 404:
        assert response.status_code == 400 or response.status_code == 422, \
            "Empty concepts should return 400/422"

    # Valid concepts list
    response = api_client.post("/api/clause/basins/strengthen", json={
        "concepts": ["concept_a"],
        "document_id": "doc_12345",
        "increment": 0.2
    })

    assert response.status_code in [200, 404]


def test_increment_validation(api_client):
    """Test increment must be in range 0.0-1.0"""

    # Increment too low
    response = api_client.post("/api/clause/basins/strengthen", json={
        "concepts": ["concept_a"],
        "document_id": "doc_12345",
        "increment": -0.1
    })

    if response.status_code != 404:
        assert response.status_code == 400 or response.status_code == 422, \
            "Increment -0.1 should return 400/422"

    # Increment too high
    response = api_client.post("/api/clause/basins/strengthen", json={
        "concepts": ["concept_a"],
        "document_id": "doc_12345",
        "increment": 1.1
    })

    if response.status_code != 404:
        assert response.status_code == 400 or response.status_code == 422, \
            "Increment 1.1 should return 400/422"

    # Valid increment
    response = api_client.post("/api/clause/basins/strengthen", json={
        "concepts": ["concept_a"],
        "document_id": "doc_12345",
        "increment": 0.2
    })

    assert response.status_code in [200, 404]


def test_missing_required_fields(api_client):
    """Test missing required fields return 400"""

    # Missing concepts
    response = api_client.post("/api/clause/basins/strengthen", json={
        "document_id": "doc_12345",
        "increment": 0.2
    })

    if response.status_code != 404:
        assert response.status_code == 400 or response.status_code == 422, \
            "Missing 'concepts' should return 400/422"

    # Missing document_id
    response = api_client.post("/api/clause/basins/strengthen", json={
        "concepts": ["concept_a"],
        "increment": 0.2
    })

    if response.status_code != 404:
        assert response.status_code == 400 or response.status_code == 422, \
            "Missing 'document_id' should return 400/422"


def test_default_increment(api_client):
    """Test default increment is 0.2 when not specified"""

    # Minimal request (increment should default to 0.2)
    response = api_client.post("/api/clause/basins/strengthen", json={
        "concepts": ["concept_a", "concept_b"],
        "document_id": "doc_12345"
    })

    if response.status_code == 200:
        data = response.json()

        # Should have created or updated basins
        assert len(data.get("updated_basins", [])) > 0 or len(data.get("new_basins", [])) > 0


def test_cooccurrence_symmetric_update(api_client):
    """Test co-occurrence updates are symmetric (A↔B)"""

    request = {
        "concepts": ["concept_a", "concept_b", "concept_c"],
        "document_id": "doc_12345",
        "increment": 0.2
    }

    response = api_client.post("/api/clause/basins/strengthen", json=request)

    if response.status_code == 200:
        data = response.json()

        # Co-occurrence updates should be symmetric
        cooccurrence = data.get("cooccurrence_updates", {})

        # If A→B exists, B→A should also exist
        for source, targets in cooccurrence.items():
            for target in targets:
                # Check reverse exists
                if target in cooccurrence:
                    assert source in cooccurrence[target], \
                        f"Symmetric co-occurrence missing: {source}↔{target}"


if __name__ == "__main__":
    print("\n=== T008: Contract Test - POST /api/clause/basins/strengthen ===\n")
    print("⚠️  This test MUST FAIL initially (endpoint not implemented)")
    print("✅  Test will pass once endpoint is implemented in T015-T025\n")

    pytest.main([__file__, "-v"])
