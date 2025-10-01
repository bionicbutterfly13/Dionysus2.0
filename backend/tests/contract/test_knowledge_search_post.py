"""Contract test for POST /api/v1/knowledge/search endpoint."""

import pytest
from fastapi.testclient import TestClient

# This test MUST FAIL until the endpoint is implemented

class TestKnowledgeSearchPost:
    """Contract tests for knowledge search endpoint."""

    @pytest.fixture
    def client(self):
        """Test client fixture - will fail until main app is created."""
        from src.main import app  # This import will fail initially
        return TestClient(app)

    @pytest.fixture
    def valid_search_query(self) -> dict:
        """Create a valid search query for testing."""
        return {
            "query": "consciousness and neural networks",
            "search_type": "SEMANTIC",
            "filters": {
                "document_types": ["RESEARCH_PAPER", "ARTICLE"],
                "date_range": {
                    "start": "2020-01-01",
                    "end": "2024-12-31"
                }
            },
            "limit": 20
        }

    @pytest.fixture
    def minimal_search_query(self) -> dict:
        """Create a minimal search query for testing."""
        return {
            "query": "artificial intelligence"
        }

    def test_knowledge_search_success(self, client: TestClient, valid_search_query: dict):
        """Test successful knowledge search."""
        response = client.post("/api/v1/knowledge/search", json=valid_search_query)

        assert response.status_code == 200
        response_data = response.json()

        # Required fields from API contract
        assert "results" in response_data
        assert "total_count" in response_data
        assert "search_metadata" in response_data
        assert "knowledge_graph_insights" in response_data

        # Validate results structure
        results = response_data["results"]
        assert isinstance(results, list)

        # Validate total count
        total_count = response_data["total_count"]
        assert isinstance(total_count, int)
        assert total_count >= 0
        assert total_count >= len(results)

    def test_knowledge_search_minimal_query(self, client: TestClient, minimal_search_query: dict):
        """Test knowledge search with minimal query."""
        response = client.post("/api/v1/knowledge/search", json=minimal_search_query)

        assert response.status_code == 200
        response_data = response.json()

        # Should still have required fields
        assert "results" in response_data
        assert "total_count" in response_data

    def test_knowledge_search_results_structure(self, client: TestClient, valid_search_query: dict):
        """Test knowledge search results structure."""
        response = client.post("/api/v1/knowledge/search", json=valid_search_query)

        if response.status_code == 200:
            response_data = response.json()
            results = response_data["results"]

            if results:
                result = results[0]
                # Core result fields
                assert "result_id" in result
                assert "result_type" in result
                assert "title" in result
                assert "content_snippet" in result
                assert "relevance_score" in result
                assert "source_document" in result

                # Validate result type
                valid_result_types = ["DOCUMENT", "KNOWLEDGE_TRIPLE", "THOUGHTSEED", "ATTRACTOR_BASIN"]
                assert result["result_type"] in valid_result_types

                # Validate relevance score
                relevance_score = result["relevance_score"]
                assert isinstance(relevance_score, (int, float))
                assert 0.0 <= relevance_score <= 1.0

                # Source document information
                source_document = result["source_document"]
                assert "document_id" in source_document
                assert "filename" in source_document
                assert "document_type" in source_document

    def test_knowledge_search_semantic_type(self, client: TestClient):
        """Test semantic knowledge search."""
        search_query = {
            "query": "machine learning consciousness",
            "search_type": "SEMANTIC",
            "limit": 10
        }

        response = client.post("/api/v1/knowledge/search", json=search_query)

        if response.status_code == 200:
            response_data = response.json()
            search_metadata = response_data["search_metadata"]

            # Semantic search specific metadata
            assert "semantic_similarity_threshold" in search_metadata
            assert "vector_space_model" in search_metadata
            assert "embedding_model" in search_metadata

            # Validate embedding model
            embedding_model = search_metadata["embedding_model"]
            assert "model_name" in embedding_model
            assert "dimensions" in embedding_model
            assert embedding_model["dimensions"] == 384  # From our 384-dimensional vectors

    def test_knowledge_search_graph_type(self, client: TestClient):
        """Test graph-based knowledge search."""
        search_query = {
            "query": "neural networks",
            "search_type": "GRAPH",
            "graph_traversal": {
                "max_depth": 3,
                "relationship_types": ["RELATED_TO", "INFLUENCES", "DERIVED_FROM"]
            }
        }

        response = client.post("/api/v1/knowledge/search", json=search_query)

        if response.status_code == 200:
            response_data = response.json()
            search_metadata = response_data["search_metadata"]

            # Graph search specific metadata
            assert "graph_traversal_info" in search_metadata
            traversal_info = search_metadata["graph_traversal_info"]
            assert "nodes_visited" in traversal_info
            assert "edges_traversed" in traversal_info
            assert "max_depth_reached" in traversal_info

    def test_knowledge_search_hybrid_type(self, client: TestClient):
        """Test hybrid knowledge search (semantic + graph)."""
        search_query = {
            "query": "consciousness emergence",
            "search_type": "HYBRID",
            "semantic_weight": 0.7,
            "graph_weight": 0.3
        }

        response = client.post("/api/v1/knowledge/search", json=search_query)

        if response.status_code == 200:
            response_data = response.json()
            search_metadata = response_data["search_metadata"]

            # Hybrid search specific metadata
            assert "hybrid_weights" in search_metadata
            hybrid_weights = search_metadata["hybrid_weights"]
            assert "semantic_weight" in hybrid_weights
            assert "graph_weight" in hybrid_weights
            assert abs(hybrid_weights["semantic_weight"] + hybrid_weights["graph_weight"] - 1.0) < 1e-6

    def test_knowledge_search_filters(self, client: TestClient):
        """Test knowledge search with various filters."""
        search_query = {
            "query": "artificial intelligence",
            "filters": {
                "document_types": ["RESEARCH_PAPER"],
                "thoughtseed_layers": ["CONCEPTUAL", "ABSTRACT"],
                "consciousness_threshold": 0.5,
                "attractor_strength_min": 0.3,
                "date_range": {
                    "start": "2023-01-01",
                    "end": "2024-12-31"
                }
            }
        }

        response = client.post("/api/v1/knowledge/search", json=search_query)

        if response.status_code == 200:
            response_data = response.json()
            search_metadata = response_data["search_metadata"]

            # Filter metadata
            assert "applied_filters" in search_metadata
            applied_filters = search_metadata["applied_filters"]
            assert "document_types" in applied_filters
            assert "thoughtseed_layers" in applied_filters
            assert "consciousness_threshold" in applied_filters

    def test_knowledge_search_graph_insights(self, client: TestClient, valid_search_query: dict):
        """Test knowledge graph insights in search results."""
        response = client.post("/api/v1/knowledge/search", json=valid_search_query)

        if response.status_code == 200:
            response_data = response.json()
            kg_insights = response_data["knowledge_graph_insights"]

            # Core knowledge graph insights
            assert "related_concepts" in kg_insights
            assert "conceptual_clusters" in kg_insights
            assert "knowledge_triples" in kg_insights
            assert "graph_metrics" in kg_insights

            # Related concepts
            related_concepts = kg_insights["related_concepts"]
            assert isinstance(related_concepts, list)

            if related_concepts:
                concept = related_concepts[0]
                assert "concept" in concept
                assert "relevance_score" in concept
                assert "frequency" in concept

            # Conceptual clusters
            conceptual_clusters = kg_insights["conceptual_clusters"]
            assert isinstance(conceptual_clusters, list)

            if conceptual_clusters:
                cluster = conceptual_clusters[0]
                assert "cluster_id" in cluster
                assert "concepts" in cluster
                assert "cluster_coherence" in cluster

            # Knowledge triples from AutoSchemaKG
            knowledge_triples = kg_insights["knowledge_triples"]
            assert isinstance(knowledge_triples, list)

            if knowledge_triples:
                triple = knowledge_triples[0]
                assert "subject" in triple
                assert "predicate" in triple
                assert "object" in triple
                assert "confidence_score" in triple
                assert "source_document_id" in triple

    def test_knowledge_search_thoughtseed_integration(self, client: TestClient):
        """Test ThoughtSeed integration in knowledge search."""
        search_query = {
            "query": "consciousness neural processing",
            "include_thoughtseed_analysis": True,
            "thoughtseed_layers": ["PERCEPTUAL", "CONCEPTUAL", "ABSTRACT"]
        }

        response = client.post("/api/v1/knowledge/search", json=search_query)

        if response.status_code == 200:
            response_data = response.json()

            # ThoughtSeed analysis should be included
            assert "thoughtseed_analysis" in response_data
            thoughtseed_analysis = response_data["thoughtseed_analysis"]

            assert "layer_activations" in thoughtseed_analysis
            assert "consciousness_patterns" in thoughtseed_analysis
            assert "memory_integrations" in thoughtseed_analysis

            # Layer activations
            layer_activations = thoughtseed_analysis["layer_activations"]
            expected_layers = ["PERCEPTUAL", "CONCEPTUAL", "ABSTRACT"]
            for layer in expected_layers:
                assert layer in layer_activations
                layer_data = layer_activations[layer]
                assert "activation_strength" in layer_data
                assert "relevant_concepts" in layer_data

    def test_knowledge_search_attractor_basin_integration(self, client: TestClient):
        """Test attractor basin integration in knowledge search."""
        search_query = {
            "query": "machine learning patterns",
            "include_attractor_analysis": True,
            "attractor_influence_types": ["REINFORCEMENT", "SYNTHESIS"]
        }

        response = client.post("/api/v1/knowledge/search", json=search_query)

        if response.status_code == 200:
            response_data = response.json()

            # Attractor basin analysis should be included
            assert "attractor_analysis" in response_data
            attractor_analysis = response_data["attractor_analysis"]

            assert "relevant_basins" in attractor_analysis
            assert "conceptual_attractions" in attractor_analysis
            assert "basin_modifications" in attractor_analysis

            # Relevant basins
            relevant_basins = attractor_analysis["relevant_basins"]
            assert isinstance(relevant_basins, list)

            if relevant_basins:
                basin = relevant_basins[0]
                assert "basin_id" in basin
                assert "concept" in basin
                assert "attraction_strength" in basin
                assert "influence_type" in basin

    def test_knowledge_search_pagination(self, client: TestClient):
        """Test knowledge search pagination."""
        search_query = {
            "query": "artificial intelligence",
            "limit": 5,
            "offset": 10
        }

        response = client.post("/api/v1/knowledge/search", json=search_query)

        if response.status_code == 200:
            response_data = response.json()

            # Pagination info
            assert "pagination" in response_data
            pagination = response_data["pagination"]
            assert "limit" in pagination
            assert "offset" in pagination
            assert "has_more" in pagination

            assert pagination["limit"] == 5
            assert pagination["offset"] == 10
            assert len(response_data["results"]) <= 5

    def test_knowledge_search_invalid_query(self, client: TestClient):
        """Test knowledge search with invalid query."""
        # Empty query
        response = client.post("/api/v1/knowledge/search", json={"query": ""})
        assert response.status_code == 422

        # Missing query
        response = client.post("/api/v1/knowledge/search", json={})
        assert response.status_code == 422

    def test_knowledge_search_invalid_search_type(self, client: TestClient):
        """Test knowledge search with invalid search type."""
        search_query = {
            "query": "test",
            "search_type": "INVALID_TYPE"
        }

        response = client.post("/api/v1/knowledge/search", json=search_query)
        assert response.status_code == 422

    def test_knowledge_search_invalid_filters(self, client: TestClient):
        """Test knowledge search with invalid filters."""
        # Invalid date range
        search_query = {
            "query": "test",
            "filters": {
                "date_range": {
                    "start": "invalid-date",
                    "end": "2024-12-31"
                }
            }
        }

        response = client.post("/api/v1/knowledge/search", json=search_query)
        assert response.status_code == 422

        # Invalid consciousness threshold
        search_query = {
            "query": "test",
            "filters": {
                "consciousness_threshold": 1.5  # Should be 0.0-1.0
            }
        }

        response = client.post("/api/v1/knowledge/search", json=search_query)
        assert response.status_code == 422

    def test_knowledge_search_autoschemakg_integration(self, client: TestClient):
        """Test AutoSchemaKG integration in knowledge search."""
        search_query = {
            "query": "neural networks deep learning",
            "include_autoschemakg": True,
            "autoschemakg_confidence_threshold": 0.7
        }

        response = client.post("/api/v1/knowledge/search", json=search_query)

        if response.status_code == 200:
            response_data = response.json()

            # AutoSchemaKG results should be included
            assert "autoschemakg_results" in response_data
            autoschemakg_results = response_data["autoschemakg_results"]

            assert "extracted_entities" in autoschemakg_results
            assert "relationship_patterns" in autoschemakg_results
            assert "schema_suggestions" in autoschemakg_results

            # Extracted entities
            extracted_entities = autoschemakg_results["extracted_entities"]
            assert isinstance(extracted_entities, list)

            if extracted_entities:
                entity = extracted_entities[0]
                assert "entity_name" in entity
                assert "entity_type" in entity
                assert "confidence_score" in entity
                assert "extraction_method" in entity

    def test_knowledge_search_research_integration(self, client: TestClient):
        """Test research integration markers in knowledge search."""
        search_query = {
            "query": "consciousness memory integration",
            "include_research_markers": True
        }

        response = client.post("/api/v1/knowledge/search", json=search_query)

        if response.status_code == 200:
            response_data = response.json()

            # Research integration markers
            assert "research_integration" in response_data
            research_integration = response_data["research_integration"]

            # MIT MEM1 memory markers
            assert "mit_mem1_relevance" in research_integration
            # IBM Zurich neural efficiency markers
            assert "ibm_zurich_relevance" in research_integration
            # Shanghai AI Lab active inference markers
            assert "shanghai_ai_relevance" in research_integration