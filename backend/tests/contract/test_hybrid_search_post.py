#!/usr/bin/env python3
"""
Contract Test: POST /api/v1/hybrid/search
Test hybrid vector+graph database search with AutoSchemaKG integration
"""

import pytest
from fastapi.testclient import TestClient


class TestHybridSearchContract:
    """Contract tests for hybrid search endpoint"""

    @pytest.fixture
    def client(self):
        """Test client - will fail until endpoint implemented"""
        from src.main import app  # This import will fail initially
        return TestClient(app)

    @pytest.fixture
    def valid_search_query(self):
        """Valid hybrid search query payload"""
        return {
            "query": "consciousness emergence in neural networks",
            "search_type": "hybrid",
            "context": {
                "semantic_weight": 0.6,
                "graph_weight": 0.4,
                "max_results": 10
            }
        }

    @pytest.fixture
    def vector_only_query(self):
        """Vector-only search query"""
        return {
            "query": "neural architecture patterns",
            "search_type": "vector",
            "context": {
                "semantic_weight": 1.0,
                "graph_weight": 0.0,
                "max_results": 5
            }
        }

    @pytest.fixture
    def graph_only_query(self):
        """Graph-only search query"""
        return {
            "query": "pattern relationships",
            "search_type": "graph",
            "context": {
                "semantic_weight": 0.0,
                "graph_weight": 1.0,
                "max_results": 15
            }
        }

    def test_hybrid_search_post_success(self, client, valid_search_query):
        """Test successful hybrid search"""
        # This test MUST fail initially - endpoint doesn't exist yet
        response = client.post("/api/v1/hybrid/search", json=valid_search_query)

        # Contract requirements
        assert response.status_code == 200
        response_data = response.json()

        # Required response fields per contract
        required_fields = [
            "search_id", "results", "total_results", "search_metadata",
            "processing_time_ms", "autoschema_evolution"
        ]

        for field in required_fields:
            assert field in response_data

        # Data type validations
        assert isinstance(response_data["search_id"], str)
        assert isinstance(response_data["results"], list)
        assert isinstance(response_data["total_results"], int)
        assert isinstance(response_data["search_metadata"], dict)
        assert isinstance(response_data["processing_time_ms"], int)
        assert isinstance(response_data["autoschema_evolution"], dict)

        # Business logic validations
        assert response_data["processing_time_ms"] > 0
        assert response_data["total_results"] >= 0
        assert len(response_data["results"]) <= valid_search_query["context"]["max_results"]

    def test_hybrid_search_result_structure(self, client, valid_search_query):
        """Test search result structure per contract"""
        response = client.post("/api/v1/hybrid/search", json=valid_search_query)

        if response.status_code == 200:
            response_data = response.json()

            if response_data["results"]:
                result = response_data["results"][0]

                # Required result fields per contract
                required_result_fields = [
                    "result_id", "content_type", "title", "content_summary",
                    "relevance_score", "vector_score", "graph_score",
                    "source_metadata", "relationships"
                ]

                for field in required_result_fields:
                    assert field in result

                # Data type validations for results
                assert isinstance(result["result_id"], str)
                assert result["content_type"] in ["document", "pattern", "research_query", "thoughtseed_trace"]
                assert isinstance(result["title"], str)
                assert isinstance(result["content_summary"], str)
                assert 0.0 <= result["relevance_score"] <= 1.0
                assert 0.0 <= result["vector_score"] <= 1.0
                assert 0.0 <= result["graph_score"] <= 1.0
                assert isinstance(result["source_metadata"], dict)
                assert isinstance(result["relationships"], list)

    def test_hybrid_search_metadata_structure(self, client, valid_search_query):
        """Test search metadata structure"""
        response = client.post("/api/v1/hybrid/search", json=valid_search_query)

        if response.status_code == 200:
            response_data = response.json()
            metadata = response_data["search_metadata"]

            # Required metadata fields
            required_metadata_fields = [
                "vector_database_hits", "graph_database_hits", "fusion_strategy",
                "semantic_embedding_model", "query_expansion", "filters_applied"
            ]

            for field in required_metadata_fields:
                assert field in metadata

            # Data type validations
            assert isinstance(metadata["vector_database_hits"], int)
            assert isinstance(metadata["graph_database_hits"], int)
            assert metadata["fusion_strategy"] in ["weighted_sum", "rrf", "hybrid_rank"]
            assert isinstance(metadata["semantic_embedding_model"], str)
            assert isinstance(metadata["query_expansion"], list)
            assert isinstance(metadata["filters_applied"], list)

    def test_hybrid_search_autoschema_evolution(self, client, valid_search_query):
        """Test AutoSchemaKG evolution tracking"""
        response = client.post("/api/v1/hybrid/search", json=valid_search_query)

        if response.status_code == 200:
            response_data = response.json()
            evolution = response_data["autoschema_evolution"]

            # Required evolution tracking fields
            required_evolution_fields = [
                "schema_version", "new_relationships_discovered", "schema_modifications",
                "confidence_updates", "knowledge_graph_expansion"
            ]

            for field in required_evolution_fields:
                assert field in evolution

            # Data type validations
            assert isinstance(evolution["schema_version"], str)
            assert isinstance(evolution["new_relationships_discovered"], list)
            assert isinstance(evolution["schema_modifications"], list)
            assert isinstance(evolution["confidence_updates"], dict)
            assert isinstance(evolution["knowledge_graph_expansion"], bool)

    def test_hybrid_search_vector_only(self, client, vector_only_query):
        """Test vector-only search mode"""
        response = client.post("/api/v1/hybrid/search", json=vector_only_query)

        if response.status_code == 200:
            response_data = response.json()
            metadata = response_data["search_metadata"]

            # Vector-only should have zero graph hits
            assert metadata["graph_database_hits"] == 0
            assert metadata["vector_database_hits"] > 0

            # All results should have vector scores, graph scores may be 0
            for result in response_data["results"]:
                assert result["vector_score"] > 0.0
                # Graph score should be 0 or very low for vector-only

    def test_hybrid_search_graph_only(self, client, graph_only_query):
        """Test graph-only search mode"""
        response = client.post("/api/v1/hybrid/search", json=graph_only_query)

        if response.status_code == 200:
            response_data = response.json()
            metadata = response_data["search_metadata"]

            # Graph-only should have zero vector hits
            assert metadata["vector_database_hits"] == 0
            assert metadata["graph_database_hits"] > 0

            # All results should have graph scores, vector scores may be 0
            for result in response_data["results"]:
                assert result["graph_score"] > 0.0

    def test_hybrid_search_relationship_structure(self, client, valid_search_query):
        """Test relationship structure in results"""
        response = client.post("/api/v1/hybrid/search", json=valid_search_query)

        if response.status_code == 200:
            response_data = response.json()

            for result in response_data["results"]:
                relationships = result["relationships"]

                for relationship in relationships:
                    # Each relationship should have required fields
                    assert "target_id" in relationship
                    assert "relationship_type" in relationship
                    assert "confidence" in relationship
                    assert isinstance(relationship["target_id"], str)
                    assert isinstance(relationship["relationship_type"], str)
                    assert 0.0 <= relationship["confidence"] <= 1.0

    def test_hybrid_search_invalid_payload(self, client):
        """Test invalid payload handling"""
        # Missing required 'query' field
        invalid_payload = {"search_type": "hybrid"}

        response = client.post("/api/v1/hybrid/search", json=invalid_payload)
        assert response.status_code == 400

    def test_hybrid_search_empty_query(self, client):
        """Test empty query handling"""
        empty_query = {"query": "", "search_type": "hybrid"}

        response = client.post("/api/v1/hybrid/search", json=empty_query)
        assert response.status_code == 400

    def test_hybrid_search_invalid_weights(self, client):
        """Test invalid weight validation"""
        invalid_weights = {
            "query": "test query",
            "search_type": "hybrid",
            "context": {
                "semantic_weight": 0.7,
                "graph_weight": 0.8,  # Weights sum > 1.0
                "max_results": 10
            }
        }

        response = client.post("/api/v1/hybrid/search", json=invalid_weights)
        assert response.status_code == 400

    def test_hybrid_search_invalid_search_type(self, client):
        """Test invalid search type"""
        invalid_type = {
            "query": "test query",
            "search_type": "invalid_type"
        }

        response = client.post("/api/v1/hybrid/search", json=invalid_type)
        assert response.status_code == 400

    def test_hybrid_search_performance_requirement(self, client, valid_search_query):
        """Test performance requirement: <1s hybrid search"""
        response = client.post("/api/v1/hybrid/search", json=valid_search_query)

        if response.status_code == 200:
            response_data = response.json()
            # Performance requirement from specification
            assert response_data["processing_time_ms"] < 1000  # <1s

    def test_hybrid_search_result_ranking(self, client, valid_search_query):
        """Test result ranking by relevance score"""
        response = client.post("/api/v1/hybrid/search", json=valid_search_query)

        if response.status_code == 200:
            response_data = response.json()
            results = response_data["results"]

            if len(results) > 1:
                # Results should be sorted by relevance_score (descending)
                for i in range(len(results) - 1):
                    assert results[i]["relevance_score"] >= results[i + 1]["relevance_score"]

    def test_hybrid_search_max_results_limit(self, client):
        """Test maximum results limit enforcement"""
        large_limit_query = {
            "query": "test query",
            "search_type": "hybrid",
            "context": {
                "semantic_weight": 0.5,
                "graph_weight": 0.5,
                "max_results": 1000  # Very large limit
            }
        }

        response = client.post("/api/v1/hybrid/search", json=large_limit_query)

        if response.status_code == 200:
            response_data = response.json()
            # Should enforce reasonable maximum (e.g., 100)
            assert len(response_data["results"]) <= 100

    def test_hybrid_search_query_expansion(self, client, valid_search_query):
        """Test query expansion tracking"""
        response = client.post("/api/v1/hybrid/search", json=valid_search_query)

        if response.status_code == 200:
            response_data = response.json()
            expansion = response_data["search_metadata"]["query_expansion"]

            # Should have expansion terms
            for term in expansion:
                assert isinstance(term, str)
                assert len(term) > 0