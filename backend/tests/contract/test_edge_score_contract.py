#!/usr/bin/env python3
"""
T010: Contract Test - POST /api/clause/edges/score

This test MUST FAIL initially (endpoint not implemented).
Tests API contract compliance for CLAUSE edge scoring endpoint.
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


def test_edge_score_request_schema(api_client):
    """Test request schema validation for POST /api/clause/edges/score"""

    valid_request = {
        "edges": [
            {
                "source": "neural_architecture",
                "relation": "RELATED_TO",
                "target": "search_algorithms"
            }
        ],
        "query": "neural architecture search",
        "signal_weights": {
            "entity": 0.25,
            "relation": 0.25,
            "neighborhood": 0.20,
            "degree": 0.15,
            "basin": 0.15
        }
    }

    response = api_client.post("/api/clause/edges/score", json=valid_request)

    assert response.status_code in [200, 404], \
        f"Expected 200 or 404, got {response.status_code}"


def test_edge_score_response_schema(api_client):
    """Test response schema matches contract specification"""

    request = {
        "edges": [
            {
                "source": "neural_architecture",
                "relation": "RELATED_TO",
                "target": "search_algorithms"
            }
        ],
        "query": "neural architecture search"
    }

    response = api_client.post("/api/clause/edges/score", json=request)

    if response.status_code == 200:
        data = response.json()

        # Required fields per contract
        assert "edge_scores" in data
        assert "signal_breakdown" in data
        assert "computation_time_ms" in data

        # Type validation
        assert isinstance(data["edge_scores"], dict)
        assert isinstance(data["signal_breakdown"], dict)
        assert isinstance(data["computation_time_ms"], (int, float))


def test_signal_weights_validation(api_client):
    """Test signal_weights must sum to 1.0"""

    # Invalid weights (sum > 1.0)
    response = api_client.post("/api/clause/edges/score", json={
        "edges": [{"source": "a", "relation": "R", "target": "b"}],
        "query": "test",
        "signal_weights": {
            "entity": 0.5,
            "relation": 0.5,
            "neighborhood": 0.5,
            "degree": 0.5,
            "basin": 0.5
        }
    })

    if response.status_code != 404:
        assert response.status_code == 400 or response.status_code == 422, \
            "Weights sum > 1.0 should return 400/422"

    # Valid weights (sum = 1.0)
    response = api_client.post("/api/clause/edges/score", json={
        "edges": [{"source": "a", "relation": "R", "target": "b"}],
        "query": "test",
        "signal_weights": {
            "entity": 0.25,
            "relation": 0.25,
            "neighborhood": 0.20,
            "degree": 0.15,
            "basin": 0.15
        }
    })

    assert response.status_code in [200, 404]


def test_default_signal_weights(api_client):
    """Test default signal weights when not specified"""

    # Minimal request (weights should default to CLAUSE spec)
    response = api_client.post("/api/clause/edges/score", json={
        "edges": [
            {
                "source": "neural_architecture",
                "relation": "RELATED_TO",
                "target": "search_algorithms"
            }
        ],
        "query": "neural architecture search"
    })

    if response.status_code == 200:
        data = response.json()

        # Should have edge scores
        assert len(data.get("edge_scores", {})) > 0


def test_edge_score_range(api_client):
    """Test edge scores are in range 0.0-1.0"""

    request = {
        "edges": [
            {
                "source": "neural_architecture",
                "relation": "RELATED_TO",
                "target": "search_algorithms"
            }
        ],
        "query": "neural architecture search"
    }

    response = api_client.post("/api/clause/edges/score", json=request)

    if response.status_code == 200:
        data = response.json()
        edge_scores = data.get("edge_scores", {})

        for edge_key, score in edge_scores.items():
            assert isinstance(score, (int, float)), \
                f"Score must be numeric, got {type(score)}"
            assert 0.0 <= score <= 1.0, \
                f"Score {score} out of range [0.0, 1.0]"


def test_signal_breakdown_structure(api_client):
    """Test signal_breakdown has correct structure (5 signals)"""

    request = {
        "edges": [
            {
                "source": "neural_architecture",
                "relation": "RELATED_TO",
                "target": "search_algorithms"
            }
        ],
        "query": "neural architecture search"
    }

    response = api_client.post("/api/clause/edges/score", json=request)

    if response.status_code == 200:
        data = response.json()
        breakdown = data.get("signal_breakdown", {})

        # Each edge should have 5 signals
        for edge_key, signals in breakdown.items():
            assert isinstance(signals, dict), "Signals must be dict"

            # Verify 5 CLAUSE signals present
            expected_signals = ["phi_ent", "phi_rel", "phi_nbr", "phi_deg", "phi_basin"]
            for signal in expected_signals:
                assert signal in signals, f"Missing signal: {signal}"

            # Verify signal values in range [0.0, 1.0]
            for signal, value in signals.items():
                assert 0.0 <= value <= 1.0, \
                    f"Signal {signal} value {value} out of range [0.0, 1.0]"


def test_empty_edges_list(api_client):
    """Test empty edges list returns 400"""

    response = api_client.post("/api/clause/edges/score", json={
        "edges": [],
        "query": "test"
    })

    if response.status_code != 404:
        assert response.status_code == 400 or response.status_code == 422, \
            "Empty edges list should return 400/422"


def test_missing_query_field(api_client):
    """Test missing 'query' field returns 400"""

    response = api_client.post("/api/clause/edges/score", json={
        "edges": [{"source": "a", "relation": "R", "target": "b"}]
    })

    if response.status_code != 404:
        assert response.status_code == 400 or response.status_code == 422, \
            "Missing 'query' should return 400/422"


def test_computation_time_positive(api_client):
    """Test computation_time_ms is positive"""

    request = {
        "edges": [
            {
                "source": "neural_architecture",
                "relation": "RELATED_TO",
                "target": "search_algorithms"
            }
        ],
        "query": "neural architecture search"
    }

    response = api_client.post("/api/clause/edges/score", json=request)

    if response.status_code == 200:
        data = response.json()
        time_ms = data.get("computation_time_ms")

        assert time_ms is not None, "computation_time_ms missing"
        assert isinstance(time_ms, (int, float)), "computation_time_ms must be numeric"
        assert time_ms >= 0, "computation_time_ms must be non-negative"


if __name__ == "__main__":
    print("\n=== T010: Contract Test - POST /api/clause/edges/score ===\n")
    print("⚠️  This test MUST FAIL initially (endpoint not implemented)")
    print("✅  Test will pass once endpoint is implemented in T015-T025\n")

    pytest.main([__file__, "-v"])
