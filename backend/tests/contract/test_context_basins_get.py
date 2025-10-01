#!/usr/bin/env python3
"""
Contract Test: GET /api/v1/context-engineering/basins
Test retrieval of Context Engineering attractor basins and neural field states
"""

import pytest
from fastapi.testclient import TestClient


class TestContextBasinsContract:
    """Contract tests for Context Engineering basins endpoint"""

    @pytest.fixture
    def client(self):
        """Test client - will fail until endpoint implemented"""
        from src.main import app  # This import will fail initially
        return TestClient(app)

    def test_context_basins_get_success(self, client):
        """Test successful basins retrieval"""
        # This test MUST fail initially - endpoint doesn't exist yet
        response = client.get("/api/v1/context-engineering/basins")

        # Contract requirements
        assert response.status_code == 200
        response_data = response.json()

        # Required response structure per contract
        assert "basins" in response_data
        assert "neural_field_state" in response_data
        assert "total_basins" in response_data
        assert "active_basins" in response_data
        assert "consciousness_coherence" in response_data

        # Data type validations
        assert isinstance(response_data["basins"], list)
        assert isinstance(response_data["neural_field_state"], dict)
        assert isinstance(response_data["total_basins"], int)
        assert isinstance(response_data["active_basins"], int)
        assert 0.0 <= response_data["consciousness_coherence"] <= 1.0

    def test_context_basins_schema_validation(self, client):
        """Test attractor basin schema compliance"""
        response = client.get("/api/v1/context-engineering/basins")

        if response.status_code == 200:
            response_data = response.json()

            if response_data["basins"]:
                basin = response_data["basins"][0]

                # Required AttractorBasin fields per contract
                required_fields = [
                    "basin_id", "basin_name", "stability", "depth",
                    "activation_threshold", "current_activation", "connected_basins",
                    "neural_field_influence", "consciousness_contribution", "last_updated"
                ]

                for field in required_fields:
                    assert field in basin

                # Data type validations
                assert isinstance(basin["basin_id"], str)
                assert isinstance(basin["basin_name"], str)
                assert 0.0 <= basin["stability"] <= 1.0
                assert isinstance(basin["depth"], (int, float))
                assert basin["depth"] > 0.0
                assert 0.0 <= basin["activation_threshold"] <= 1.0
                assert 0.0 <= basin["current_activation"] <= 1.0
                assert isinstance(basin["connected_basins"], list)
                assert isinstance(basin["neural_field_influence"], dict)
                assert 0.0 <= basin["consciousness_contribution"] <= 1.0
                assert isinstance(basin["last_updated"], str)  # ISO datetime

    def test_context_basins_neural_field_structure(self, client):
        """Test neural field state structure"""
        response = client.get("/api/v1/context-engineering/basins")

        if response.status_code == 200:
            response_data = response.json()
            neural_field = response_data["neural_field_state"]

            # Required neural field components
            required_field_components = [
                "field_strength", "coherence_level", "entropy_measure",
                "field_gradient", "oscillation_frequency", "phase_coupling"
            ]

            for component in required_field_components:
                assert component in neural_field

            # Data type and range validations
            assert 0.0 <= neural_field["field_strength"] <= 1.0
            assert 0.0 <= neural_field["coherence_level"] <= 1.0
            assert isinstance(neural_field["entropy_measure"], (int, float))
            assert isinstance(neural_field["field_gradient"], dict)
            assert isinstance(neural_field["oscillation_frequency"], (int, float))
            assert 0.0 <= neural_field["phase_coupling"] <= 1.0

    def test_context_basins_with_activation_filter(self, client):
        """Test activation level filtering"""
        min_activation = 0.5
        response = client.get(f"/api/v1/context-engineering/basins?min_activation={min_activation}")

        if response.status_code == 200:
            response_data = response.json()
            # All returned basins should meet minimum activation
            for basin in response_data["basins"]:
                assert basin["current_activation"] >= min_activation

    def test_context_basins_with_stability_filter(self, client):
        """Test stability threshold filtering"""
        min_stability = 0.7
        response = client.get(f"/api/v1/context-engineering/basins?min_stability={min_stability}")

        if response.status_code == 200:
            response_data = response.json()
            # All returned basins should meet minimum stability
            for basin in response_data["basins"]:
                assert basin["stability"] >= min_stability

    def test_context_basins_basin_connections(self, client):
        """Test basin connection structure"""
        response = client.get("/api/v1/context-engineering/basins")

        if response.status_code == 200:
            response_data = response.json()

            for basin in response_data["basins"]:
                connections = basin["connected_basins"]

                # Each connection should have connection data
                for connection in connections:
                    assert "basin_id" in connection
                    assert "connection_strength" in connection
                    assert "connection_type" in connection
                    assert isinstance(connection["basin_id"], str)
                    assert 0.0 <= connection["connection_strength"] <= 1.0
                    assert connection["connection_type"] in [
                        "excitatory", "inhibitory", "modulatory", "bidirectional"
                    ]

    def test_context_basins_consciousness_coherence(self, client):
        """Test consciousness coherence calculation"""
        response = client.get("/api/v1/context-engineering/basins")

        if response.status_code == 200:
            response_data = response.json()
            coherence = response_data["consciousness_coherence"]
            active_basins = response_data["active_basins"]
            total_basins = response_data["total_basins"]

            # Coherence should correlate with basin activation
            if active_basins == 0:
                assert coherence <= 0.1  # Very low coherence with no active basins
            elif active_basins == total_basins and active_basins > 0:
                # High coherence when all basins active (if basins exist)
                assert coherence >= 0.3

    def test_context_basins_neural_field_influence(self, client):
        """Test neural field influence structure"""
        response = client.get("/api/v1/context-engineering/basins")

        if response.status_code == 200:
            response_data = response.json()

            for basin in response_data["basins"]:
                influence = basin["neural_field_influence"]

                # Each basin should have field influence data
                assert "field_contribution" in influence
                assert "spatial_extent" in influence
                assert "temporal_persistence" in influence

                assert 0.0 <= influence["field_contribution"] <= 1.0
                assert isinstance(influence["spatial_extent"], (int, float))
                assert isinstance(influence["temporal_persistence"], (int, float))

    def test_context_basins_invalid_filters(self, client):
        """Test invalid filter parameter handling"""
        # Invalid activation (> 1.0)
        response = client.get("/api/v1/context-engineering/basins?min_activation=1.5")
        assert response.status_code in [400, 200]

        # Invalid stability (negative)
        response = client.get("/api/v1/context-engineering/basins?min_stability=-0.1")
        assert response.status_code in [400, 200]

    def test_context_basins_empty_result(self, client):
        """Test empty result handling"""
        # Filter for very high thresholds that shouldn't match anything
        response = client.get("/api/v1/context-engineering/basins?min_activation=0.99&min_stability=0.99")

        if response.status_code == 200:
            response_data = response.json()
            assert isinstance(response_data["basins"], list)
            assert response_data["total_basins"] >= 0
            assert response_data["active_basins"] >= 0

    def test_context_basins_performance_requirement(self, client):
        """Test performance requirement: <300ms basin retrieval"""
        import time
        start_time = time.time()

        response = client.get("/api/v1/context-engineering/basins")

        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000

        # Performance requirement for Context Engineering queries
        assert response_time_ms < 300  # <300ms

    def test_context_basins_with_limit(self, client):
        """Test result limiting"""
        limit = 5
        response = client.get(f"/api/v1/context-engineering/basins?limit={limit}")

        if response.status_code == 200:
            response_data = response.json()
            # Should not exceed limit
            assert len(response_data["basins"]) <= limit

    def test_context_basins_field_gradient_structure(self, client):
        """Test neural field gradient structure"""
        response = client.get("/api/v1/context-engineering/basins")

        if response.status_code == 200:
            response_data = response.json()
            field_gradient = response_data["neural_field_state"]["field_gradient"]

            # Should have spatial gradient components
            expected_gradient_fields = ["x_component", "y_component", "z_component", "magnitude"]

            for field in expected_gradient_fields:
                assert field in field_gradient
                assert isinstance(field_gradient[field], (int, float))

            # Magnitude should be positive
            assert field_gradient["magnitude"] >= 0.0