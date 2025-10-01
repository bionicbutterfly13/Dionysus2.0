#!/usr/bin/env python3
"""
Contract Test: POST /api/v1/research/query
Test ASI-GO-2 research query processing with ThoughtSeed competition
"""

import pytest
import json
from httpx import AsyncClient
from fastapi.testclient import TestClient


class TestResearchQueryContract:
    """Contract tests for research query endpoint"""

    @pytest.fixture
    def client(self):
        """Test client - will fail until endpoint implemented"""
        from src.main import app  # This import will fail initially
        return TestClient(app)

    @pytest.fixture
    def valid_research_query(self):
        """Valid research query payload"""
        return {
            "query": "What are the key mechanisms of consciousness emergence in neural networks?",
            "context": {
                "domain_focus": ["neuroscience", "artificial_intelligence"],
                "consciousness_level_required": 0.7
            }
        }

    def test_research_query_post_success(self, client, valid_research_query):
        """Test successful research query processing"""
        # This test MUST fail initially - endpoint doesn't exist yet
        response = client.post("/api/v1/research/query", json=valid_research_query)

        # Contract requirements
        assert response.status_code == 200
        response_data = response.json()

        # Required response fields per contract
        assert "query_id" in response_data
        assert "synthesis" in response_data
        assert "confidence_score" in response_data
        assert "patterns_used" in response_data
        assert "thoughtseed_workspace_id" in response_data
        assert "consciousness_level" in response_data
        assert "processing_time_ms" in response_data
        assert "attractor_basins_activated" in response_data

        # Data type validations
        assert isinstance(response_data["query_id"], str)
        assert isinstance(response_data["synthesis"], str)
        assert 0.0 <= response_data["confidence_score"] <= 1.0
        assert isinstance(response_data["patterns_used"], list)
        assert isinstance(response_data["thoughtseed_workspace_id"], str)
        assert 0.0 <= response_data["consciousness_level"] <= 1.0
        assert isinstance(response_data["processing_time_ms"], int)
        assert isinstance(response_data["attractor_basins_activated"], list)

        # Business logic validations
        assert len(response_data["synthesis"]) > 0
        assert response_data["processing_time_ms"] > 0
        assert len(response_data["patterns_used"]) >= 0  # Can be empty for new patterns

    def test_research_query_post_invalid_payload(self, client):
        """Test invalid payload handling"""
        # Missing required 'query' field
        invalid_payload = {"context": {"domain_focus": ["ai"]}}

        response = client.post("/api/v1/research/query", json=invalid_payload)
        assert response.status_code == 400

    def test_research_query_post_empty_query(self, client):
        """Test empty query handling"""
        empty_query = {"query": ""}

        response = client.post("/api/v1/research/query", json=empty_query)
        assert response.status_code == 400

    def test_research_query_post_consciousness_level_validation(self, client):
        """Test consciousness level requirement validation"""
        query_with_invalid_consciousness = {
            "query": "Test query",
            "context": {"consciousness_level_required": 1.5}  # Invalid > 1.0
        }

        response = client.post("/api/v1/research/query", json=query_with_invalid_consciousness)
        assert response.status_code == 400

    def test_research_query_processing_time_performance(self, client, valid_research_query):
        """Test performance requirement: <2s processing time"""
        response = client.post("/api/v1/research/query", json=valid_research_query)

        if response.status_code == 200:
            response_data = response.json()
            # Performance requirement from specification
            assert response_data["processing_time_ms"] < 2000  # <2s

    def test_research_query_consciousness_detection(self, client, valid_research_query):
        """Test consciousness level detection in response"""
        response = client.post("/api/v1/research/query", json=valid_research_query)

        if response.status_code == 200:
            response_data = response.json()
            # For consciousness-related queries, should detect some level of consciousness
            if "consciousness" in valid_research_query["query"].lower():
                assert response_data["consciousness_level"] > 0.5

    def test_research_query_thoughtseed_competition(self, client, valid_research_query):
        """Test ThoughtSeed competition generates workspace"""
        response = client.post("/api/v1/research/query", json=valid_research_query)

        if response.status_code == 200:
            response_data = response.json()
            # Must have thoughtseed workspace for pattern competition
            assert len(response_data["thoughtseed_workspace_id"]) > 0
            # Should activate some patterns for competition
            assert isinstance(response_data["patterns_used"], list)