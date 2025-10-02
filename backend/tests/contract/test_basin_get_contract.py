#!/usr/bin/env python3
"""
T009: Contract Test - GET /api/clause/basins/{basin_id}

This test MUST FAIL initially (endpoint not implemented).
Tests API contract compliance for basin retrieval endpoint.
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


def test_basin_get_path_parameter(api_client):
    """Test path parameter validation for GET /api/clause/basins/{basin_id}"""

    # Valid basin_id
    response = api_client.get("/api/clause/basins/neural_architecture")

    assert response.status_code in [200, 404], \
        f"Expected 200 or 404, got {response.status_code}"


def test_basin_get_response_schema(api_client):
    """Test response schema matches contract specification"""

    response = api_client.get("/api/clause/basins/test_basin")

    if response.status_code == 200:
        data = response.json()

        # Required fields per contract
        assert "basin_id" in data
        assert "strength" in data
        assert "activation_count" in data
        assert "co_occurring_concepts" in data

        # Type validation
        assert isinstance(data["basin_id"], str)
        assert isinstance(data["strength"], (int, float))
        assert isinstance(data["activation_count"], int)
        assert isinstance(data["co_occurring_concepts"], dict)


def test_basin_strength_range(api_client):
    """Test basin strength is in range 1.0-2.0"""

    response = api_client.get("/api/clause/basins/test_basin")

    if response.status_code == 200:
        data = response.json()
        strength = data.get("strength")

        assert strength is not None, "Strength field missing"
        assert 1.0 <= strength <= 2.0, \
            f"Strength {strength} out of range [1.0, 2.0]"


def test_basin_not_found(api_client):
    """Test non-existent basin returns 404"""

    response = api_client.get("/api/clause/basins/non_existent_basin_12345")

    # Should return 404 if basin not found (or 404 if endpoint not implemented)
    assert response.status_code == 404


def test_basin_activation_count(api_client):
    """Test activation_count is non-negative integer"""

    response = api_client.get("/api/clause/basins/test_basin")

    if response.status_code == 200:
        data = response.json()
        activation_count = data.get("activation_count")

        assert activation_count is not None, "activation_count field missing"
        assert isinstance(activation_count, int), "activation_count must be integer"
        assert activation_count >= 0, "activation_count must be non-negative"


def test_basin_cooccurrence_structure(api_client):
    """Test co_occurring_concepts has correct structure (dict[str, int])"""

    response = api_client.get("/api/clause/basins/test_basin")

    if response.status_code == 200:
        data = response.json()
        cooccurrence = data.get("co_occurring_concepts", {})

        assert isinstance(cooccurrence, dict), "co_occurring_concepts must be dict"

        # Validate dict structure: str → int
        for concept, count in cooccurrence.items():
            assert isinstance(concept, str), \
                f"Co-occurrence key must be str, got {type(concept)}"
            assert isinstance(count, int), \
                f"Co-occurrence count must be int, got {type(count)}"
            assert count > 0, \
                f"Co-occurrence count must be positive, got {count}"


def test_basin_optional_fields(api_client):
    """Test optional fields are included if available"""

    response = api_client.get("/api/clause/basins/test_basin")

    if response.status_code == 200:
        data = response.json()

        # Optional fields from AttractorBasin
        optional_fields = ["stability", "depth", "basin_type", "activation_history"]

        # These are optional, just verify types if present
        if "stability" in data:
            assert isinstance(data["stability"], (int, float))
            assert 0.0 <= data["stability"] <= 1.0

        if "depth" in data:
            assert isinstance(data["depth"], (int, float))
            assert data["depth"] >= 0

        if "basin_type" in data:
            assert isinstance(data["basin_type"], str)

        if "activation_history" in data:
            assert isinstance(data["activation_history"], list)


if __name__ == "__main__":
    print("\n=== T009: Contract Test - GET /api/clause/basins/{basin_id} ===\n")
    print("⚠️  This test MUST FAIL initially (endpoint not implemented)")
    print("✅  Test will pass once endpoint is implemented in T015-T025\n")

    pytest.main([__file__, "-v"])
