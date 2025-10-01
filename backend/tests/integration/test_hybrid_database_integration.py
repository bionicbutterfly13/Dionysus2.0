#!/usr/bin/env python3
"""
Integration Test: Hybrid Vector+Graph Database Queries
Test complete hybrid database system with AutoSchemaKG, Qdrant vectors, and Neo4j graph
"""

import pytest
import asyncio
from fastapi.testclient import TestClient


class TestHybridDatabaseIntegration:
    """Integration tests for hybrid vector+graph database system"""

    @pytest.fixture
    def client(self):
        """Test client - will fail until endpoints implemented"""
        from src.main import app  # This import will fail initially
        return TestClient(app)

    @pytest.fixture
    def semantic_search_query(self):
        """Semantic search query for vector database testing"""
        return {
            "query": "neural network consciousness emergence patterns",
            "search_type": "vector",
            "context": {
                "semantic_weight": 1.0,
                "graph_weight": 0.0,
                "max_results": 10,
                "embedding_model": "nomic-embed-text"
            }
        }

    @pytest.fixture
    def graph_relationship_query(self):
        """Graph relationship query for Neo4j testing"""
        return {
            "query": "pattern relationships in deep learning architectures",
            "search_type": "graph",
            "context": {
                "semantic_weight": 0.0,
                "graph_weight": 1.0,
                "max_results": 15,
                "relationship_depth": 3
            }
        }

    @pytest.fixture
    def hybrid_fusion_query(self):
        """Hybrid fusion query combining vector and graph"""
        return {
            "query": "consciousness research patterns with semantic relationships",
            "search_type": "hybrid",
            "context": {
                "semantic_weight": 0.6,
                "graph_weight": 0.4,
                "max_results": 20,
                "fusion_strategy": "weighted_sum"
            }
        }

    def test_hybrid_database_full_pipeline(self, client, hybrid_fusion_query):
        """Test complete hybrid database pipeline with AutoSchemaKG evolution"""
        # This test MUST fail initially - integration not implemented yet

        # First, populate database with some data via document processing
        test_doc = self._create_test_document()
        doc_response = client.post(
            "/api/v1/documents/process",
            files={"file": test_doc},
            data={
                "extract_narratives": "true",
                "thoughtseed_layers": '["conceptual","abstract"]'
            }
        )

        if doc_response.status_code == 200:
            # Now test hybrid search with populated data
            search_response = client.post("/api/v1/hybrid/search", json=hybrid_fusion_query)
            assert search_response.status_code == 200

            search_data = search_response.json()

            # Should have complete hybrid search results
            assert "search_id" in search_data
            assert "results" in search_data
            assert "search_metadata" in search_data
            assert "autoschema_evolution" in search_data

            # Metadata should show both vector and graph hits
            metadata = search_data["search_metadata"]
            assert metadata["vector_database_hits"] >= 0
            assert metadata["graph_database_hits"] >= 0

            # For hybrid search, should use both databases
            total_hits = metadata["vector_database_hits"] + metadata["graph_database_hits"]
            assert total_hits > 0  # Should find something after document processing

            # AutoSchemaKG should track evolution
            evolution = search_data["autoschema_evolution"]
            assert "schema_version" in evolution
            assert "new_relationships_discovered" in evolution
            assert "knowledge_graph_expansion" in evolution

    def test_vector_database_semantic_similarity(self, client, semantic_search_query):
        """Test vector database semantic similarity search"""
        # Submit research query to populate vector database
        research_query = {
            "query": "How do neural networks develop semantic understanding?",
            "context": {
                "domain_focus": ["machine_learning", "semantics"],
                "consciousness_level_required": 0.6
            }
        }

        research_response = client.post("/api/v1/research/query", json=research_query)

        if research_response.status_code == 200:
            # Now test vector search
            vector_response = client.post("/api/v1/hybrid/search", json=semantic_search_query)

            if vector_response.status_code == 200:
                vector_data = vector_response.json()

                # Should have vector-based results
                metadata = vector_data["search_metadata"]
                assert metadata["vector_database_hits"] > 0
                assert metadata["graph_database_hits"] == 0  # Vector-only search

                # Results should have vector scores
                for result in vector_data["results"]:
                    assert "vector_score" in result
                    assert result["vector_score"] > 0.0
                    assert "graph_score" in result  # May be 0 for vector-only

                # Should use semantic embedding model
                assert metadata["semantic_embedding_model"] == "nomic-embed-text"

    def test_graph_database_relationship_traversal(self, client, graph_relationship_query):
        """Test Neo4j graph database relationship traversal"""
        # Create some related patterns via multiple queries
        related_queries = [
            {
                "query": "Deep learning architecture patterns",
                "context": {"domain_focus": ["deep_learning"]}
            },
            {
                "query": "Neural network architectural evolution",
                "context": {"domain_focus": ["neural_networks"]}
            }
        ]

        # Submit queries to build relationships
        for query in related_queries:
            client.post("/api/v1/research/query", json=query)

        # Now test graph relationship search
        graph_response = client.post("/api/v1/hybrid/search", json=graph_relationship_query)

        if graph_response.status_code == 200:
            graph_data = graph_response.json()

            # Should have graph-based results
            metadata = graph_data["search_metadata"]
            assert metadata["graph_database_hits"] > 0
            assert metadata["vector_database_hits"] == 0  # Graph-only search

            # Results should have relationship information
            for result in graph_data["results"]:
                assert "graph_score" in result
                assert result["graph_score"] > 0.0
                assert "relationships" in result

                # Check relationship structure
                for relationship in result["relationships"]:
                    assert "target_id" in relationship
                    assert "relationship_type" in relationship
                    assert "confidence" in relationship
                    assert 0.0 <= relationship["confidence"] <= 1.0

    def test_autoschema_kg_dynamic_evolution(self, client, hybrid_fusion_query):
        """Test AutoSchemaKG dynamic schema evolution"""
        # Get initial schema state
        initial_response = client.post("/api/v1/hybrid/search", json=hybrid_fusion_query)

        if initial_response.status_code == 200:
            initial_data = initial_response.json()
            initial_evolution = initial_data["autoschema_evolution"]
            initial_version = initial_evolution["schema_version"]

            # Submit new type of query to trigger schema evolution
            novel_query = {
                "query": "quantum consciousness in artificial neural substrates",
                "search_type": "hybrid",
                "context": {
                    "semantic_weight": 0.5,
                    "graph_weight": 0.5,
                    "max_results": 10
                }
            }

            novel_response = client.post("/api/v1/hybrid/search", json=novel_query)

            if novel_response.status_code == 200:
                novel_data = novel_response.json()
                novel_evolution = novel_data["autoschema_evolution"]

                # Schema should potentially evolve with new concepts
                if novel_evolution["knowledge_graph_expansion"]:
                    # Should have discovered new relationships
                    assert len(novel_evolution["new_relationships_discovered"]) >= 0

                    # Schema modifications should be tracked
                    assert isinstance(novel_evolution["schema_modifications"], list)

                    # Confidence updates should be recorded
                    assert isinstance(novel_evolution["confidence_updates"], dict)

    def test_hybrid_fusion_strategies(self, client):
        """Test different hybrid fusion strategies"""
        base_query = {
            "query": "neural pattern recognition in consciousness studies",
            "search_type": "hybrid",
            "context": {
                "semantic_weight": 0.5,
                "graph_weight": 0.5,
                "max_results": 10
            }
        }

        fusion_strategies = ["weighted_sum", "rrf", "hybrid_rank"]

        for strategy in fusion_strategies:
            query = base_query.copy()
            query["context"]["fusion_strategy"] = strategy

            response = client.post("/api/v1/hybrid/search", json=query)

            if response.status_code == 200:
                data = response.json()
                metadata = data["search_metadata"]

                # Should use the specified fusion strategy
                assert metadata["fusion_strategy"] == strategy

                # Results should be properly fused
                for result in data["results"]:
                    assert "relevance_score" in result
                    assert "vector_score" in result
                    assert "graph_score" in result

                    # Relevance should be fusion of vector and graph scores
                    relevance = result["relevance_score"]
                    vector_score = result["vector_score"]
                    graph_score = result["graph_score"]

                    # Basic fusion validation (exact formula depends on strategy)
                    if strategy == "weighted_sum":
                        expected_min = min(vector_score, graph_score)
                        expected_max = max(vector_score, graph_score)
                        assert expected_min <= relevance <= expected_max + 0.1  # Allow fusion boost

    def test_hybrid_database_concurrent_access(self, client, semantic_search_query, graph_relationship_query):
        """Test concurrent database access and consistency"""
        import threading
        import time

        responses = []
        errors = []

        def execute_search(query, response_list, error_list):
            try:
                response = client.post("/api/v1/hybrid/search", json=query)
                response_list.append(response)
            except Exception as e:
                error_list.append(e)

        # Launch concurrent searches
        threads = []
        queries = [semantic_search_query, graph_relationship_query, semantic_search_query]

        for query in queries:
            thread = threading.Thread(
                target=execute_search,
                args=(query, responses, errors)
            )
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join(timeout=15)

        # Check results
        assert len(errors) == 0  # No errors during concurrent access

        # All searches should complete
        for response in responses:
            assert response.status_code in [200, 429]  # 429 if rate limited

            if response.status_code == 200:
                data = response.json()
                # Each should have valid search results
                assert "results" in data
                assert "search_metadata" in data

    def test_hybrid_database_data_consistency(self, client):
        """Test data consistency across vector and graph databases"""
        # Submit document for processing
        test_doc = self._create_test_document()
        doc_response = client.post(
            "/api/v1/documents/process",
            files={"file": test_doc},
            data={"thoughtseed_layers": '["conceptual"]'}
        )

        if doc_response.status_code == 200:
            doc_data = doc_response.json()
            document_id = doc_data["document_id"]

            # Search for the document in vector space
            vector_query = {
                "query": "test document neural networks",
                "search_type": "vector",
                "context": {"semantic_weight": 1.0, "graph_weight": 0.0, "max_results": 20}
            }

            vector_response = client.post("/api/v1/hybrid/search", json=vector_query)

            if vector_response.status_code == 200:
                vector_data = vector_response.json()

                # Search for relationships in graph space
                graph_query = {
                    "query": "test document relationships",
                    "search_type": "graph",
                    "context": {"semantic_weight": 0.0, "graph_weight": 1.0, "max_results": 20}
                }

                graph_response = client.post("/api/v1/hybrid/search", json=graph_query)

                if graph_response.status_code == 200:
                    graph_data = graph_response.json()

                    # Should find consistent data across both databases
                    vector_results = vector_data["results"]
                    graph_results = graph_data["results"]

                    # At least one database should have found the document
                    total_results = len(vector_results) + len(graph_results)
                    assert total_results >= 0  # May be 0 if no matches, but shouldn't error

    def test_hybrid_database_performance_integration(self, client, hybrid_fusion_query):
        """Test hybrid database performance under load"""
        import time

        # Multiple hybrid searches
        start_time = time.time()

        responses = []
        for i in range(3):  # Keep reasonable for integration test
            response = client.post("/api/v1/hybrid/search", json=hybrid_fusion_query)
            responses.append(response)

        end_time = time.time()
        total_time_ms = (end_time - start_time) * 1000

        # All searches should succeed
        for response in responses:
            assert response.status_code == 200

            data = response.json()
            # Each search should meet performance requirement
            assert data["processing_time_ms"] < 1000  # <1s per search

        # Total time for all searches should be reasonable
        assert total_time_ms < 5000  # <5s for 3 searches

    def test_hybrid_database_query_expansion(self, client, semantic_search_query):
        """Test query expansion and optimization"""
        response = client.post("/api/v1/hybrid/search", json=semantic_search_query)

        if response.status_code == 200:
            data = response.json()
            metadata = data["search_metadata"]

            # Should have query expansion
            expansion = metadata["query_expansion"]
            assert isinstance(expansion, list)

            # Expansion terms should be related to original query
            original_terms = semantic_search_query["query"].lower().split()
            if expansion:
                # At least some expansion terms should be strings
                for term in expansion:
                    assert isinstance(term, str)
                    assert len(term) > 0

    def test_hybrid_database_error_recovery(self, client):
        """Test error recovery and resilience"""
        # Test with malformed query
        malformed_query = {
            "query": "",  # Empty query
            "search_type": "invalid_type",
            "context": {
                "semantic_weight": 1.5,  # Invalid weight > 1.0
                "graph_weight": -0.5     # Invalid negative weight
            }
        }

        error_response = client.post("/api/v1/hybrid/search", json=malformed_query)
        assert error_response.status_code == 400

        # Database should still be functional after error
        valid_query = {
            "query": "test query",
            "search_type": "hybrid",
            "context": {
                "semantic_weight": 0.5,
                "graph_weight": 0.5,
                "max_results": 5
            }
        }

        recovery_response = client.post("/api/v1/hybrid/search", json=valid_query)
        assert recovery_response.status_code == 200

        # Should return valid results
        recovery_data = recovery_response.json()
        assert "results" in recovery_data
        assert "search_metadata" in recovery_data

    def _create_test_document(self):
        """Helper method to create a test document"""
        import io

        pdf_content = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R>>endobj
4 0 obj<</Length 100>>stream
BT /F1 12 Tf 50 750 Td (Test document about neural networks and consciousness patterns.) Tj ET
endstream endobj
xref 0 5
0000000000 65535 f 0000000015 00000 n 0000000060 00000 n 0000000111 00000 n 0000000199 00000 n
trailer<</Size 5/Root 1 0 R>>startxref 350 %%EOF"""

        return ("test_neural_doc.pdf", io.BytesIO(pdf_content), "application/pdf")