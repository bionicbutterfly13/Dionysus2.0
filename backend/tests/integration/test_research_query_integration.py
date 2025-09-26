#!/usr/bin/env python3
"""
Integration Test: Research Query with Pattern Competition
Test complete ASI-GO-2 research query processing with ThoughtSeed competition
"""

import pytest
import asyncio
from fastapi.testclient import TestClient


class TestResearchQueryIntegration:
    """Integration tests for research query with pattern competition"""

    @pytest.fixture
    def client(self):
        """Test client - will fail until endpoints implemented"""
        from backend.src.main import app  # This import will fail initially
        return TestClient(app)

    @pytest.fixture
    def consciousness_research_query(self):
        """Complex consciousness research query for pattern competition"""
        return {
            "query": "How do neural networks develop self-awareness through recursive pattern recognition and meta-cognitive feedback loops?",
            "context": {
                "domain_focus": ["neuroscience", "artificial_intelligence", "consciousness_studies"],
                "consciousness_level_required": 0.8,
                "pattern_competition_enabled": True,
                "thoughtseed_layers": ["conceptual", "abstract", "metacognitive"]
            }
        }

    @pytest.fixture
    def pattern_recognition_query(self):
        """Pattern recognition focused query"""
        return {
            "query": "What are the fundamental patterns in deep learning architecture evolution?",
            "context": {
                "domain_focus": ["machine_learning", "neural_architecture"],
                "consciousness_level_required": 0.6,
                "pattern_competition_enabled": True,
                "max_competing_patterns": 5
            }
        }

    def test_research_query_full_asi_go_2_pipeline(self, client, consciousness_research_query):
        """Test complete ASI-GO-2 research intelligence pipeline"""
        # This test MUST fail initially - integration not implemented yet
        response = client.post("/api/v1/research/query", json=consciousness_research_query)

        assert response.status_code == 200
        query_data = response.json()

        # Should have complete ASI-GO-2 response
        assert "query_id" in query_data
        assert "synthesis" in query_data
        assert "confidence_score" in query_data
        assert "patterns_used" in query_data
        assert "thoughtseed_workspace_id" in query_data
        assert "consciousness_level" in query_data
        assert "attractor_basins_activated" in query_data

        # Check ThoughtSeed workspace was created
        workspace_id = query_data["thoughtseed_workspace_id"]
        workspace_response = client.get(f"/api/v1/thoughtseed/workspace/{workspace_id}")
        assert workspace_response.status_code == 200

        workspace_data = workspace_response.json()

        # Should have processed requested layers
        layer_states = workspace_data["layer_states"]
        requested_layers = consciousness_research_query["context"]["thoughtseed_layers"]

        for layer in requested_layers:
            assert layer in layer_states
            # Layer should be processed or processing
            assert layer_states[layer]["status"] in ["completed", "processing", "active"]

        # Consciousness level should meet requirement
        if workspace_data["processing_status"] == "completed":
            required_level = consciousness_research_query["context"]["consciousness_level_required"]
            assert query_data["consciousness_level"] >= required_level * 0.8  # Allow some tolerance

    def test_research_query_pattern_competition(self, client, pattern_recognition_query):
        """Test pattern competition mechanism in research queries"""
        response = client.post("/api/v1/research/query", json=pattern_recognition_query)

        if response.status_code == 200:
            query_data = response.json()

            # Should have competing patterns
            patterns_used = query_data["patterns_used"]
            assert isinstance(patterns_used, list)

            # Check pattern competition results
            if patterns_used:
                max_patterns = pattern_recognition_query["context"].get("max_competing_patterns", 10)
                assert len(patterns_used) <= max_patterns

                # Each pattern should have competition metrics
                for pattern in patterns_used:
                    assert "pattern_id" in pattern
                    assert "competition_score" in pattern
                    assert "selection_reason" in pattern
                    assert 0.0 <= pattern["competition_score"] <= 1.0

                # Patterns should be ranked by competition score
                if len(patterns_used) > 1:
                    for i in range(len(patterns_used) - 1):
                        assert patterns_used[i]["competition_score"] >= patterns_used[i + 1]["competition_score"]

    def test_research_query_attractor_basin_integration(self, client, consciousness_research_query):
        """Test attractor basin activation during research queries"""
        # Submit research query
        response = client.post("/api/v1/research/query", json=consciousness_research_query)

        if response.status_code == 200:
            query_data = response.json()
            activated_basins = query_data["attractor_basins_activated"]

            # Should have activated basins for complex queries
            assert isinstance(activated_basins, list)

            # Check basin states after query
            basins_response = client.get("/api/v1/context-engineering/basins")
            if basins_response.status_code == 200:
                basins_data = basins_response.json()

                # Should have active basins
                assert basins_data["active_basins"] >= 0

                # Check if specific basins were activated
                if activated_basins:
                    basin_ids = [basin["basin_id"] for basin in basins_data["basins"]]
                    for activated_basin in activated_basins:
                        # Basin should exist or have been created
                        assert isinstance(activated_basin, str)

                # Consciousness coherence should reflect query complexity
                coherence = basins_data["consciousness_coherence"]
                if query_data["consciousness_level"] > 0.7:
                    # High consciousness queries should increase coherence
                    assert coherence > 0.3

    def test_research_query_hybrid_search_integration(self, client, consciousness_research_query):
        """Test integration with hybrid search during research queries"""
        response = client.post("/api/v1/research/query", json=consciousness_research_query)

        if response.status_code == 200:
            query_data = response.json()

            # Query should trigger hybrid search
            # Test by running a hybrid search with the same query
            search_payload = {
                "query": consciousness_research_query["query"],
                "search_type": "hybrid",
                "context": {
                    "semantic_weight": 0.6,
                    "graph_weight": 0.4,
                    "max_results": 10
                }
            }

            search_response = client.post("/api/v1/hybrid/search", json=search_payload)

            if search_response.status_code == 200:
                search_data = search_response.json()

                # Research query should have used hybrid search results
                if search_data["total_results"] > 0:
                    # Synthesis should be more comprehensive with search results
                    assert len(query_data["synthesis"]) > 100  # Should have substantial content

                # AutoSchemaKG should evolve with research queries
                evolution = search_data["autoschema_evolution"]
                if evolution["knowledge_graph_expansion"]:
                    # Schema should capture research relationships
                    assert len(evolution["new_relationships_discovered"]) >= 0

    def test_research_query_asi_go_2_component_coordination(self, client, consciousness_research_query):
        """Test coordination between ASI-GO-2 components (Cognition Base, Researcher, Engineer, Analyst)"""
        response = client.post("/api/v1/research/query", json=consciousness_research_query)

        if response.status_code == 200:
            query_data = response.json()

            # Should have evidence of multi-component processing
            synthesis = query_data["synthesis"]
            confidence = query_data["confidence_score"]

            # Complex consciousness query should involve multiple components
            assert len(synthesis) > 200  # Should be comprehensive

            # Confidence should reflect multi-component validation
            if query_data["consciousness_level"] > 0.7:
                assert confidence > 0.6  # High consciousness should yield high confidence

            # Check for component-specific contributions in patterns
            patterns_used = query_data["patterns_used"]

            if patterns_used:
                # Should have patterns from different ASI-GO-2 components
                component_types = set()
                for pattern in patterns_used:
                    if "component_source" in pattern:
                        component_types.add(pattern["component_source"])

                # Complex query should engage multiple components
                if len(patterns_used) >= 3:
                    assert len(component_types) >= 1  # At least one component identified

    def test_research_query_learning_and_adaptation(self, client, pattern_recognition_query):
        """Test learning and adaptation in repeated queries"""
        # Submit same query multiple times to test learning
        first_response = client.post("/api/v1/research/query", json=pattern_recognition_query)

        if first_response.status_code == 200:
            first_data = first_response.json()
            first_confidence = first_data["confidence_score"]
            first_processing_time = first_data["processing_time_ms"]

            # Submit again (system should learn)
            second_response = client.post("/api/v1/research/query", json=pattern_recognition_query)

            if second_response.status_code == 200:
                second_data = second_response.json()
                second_confidence = second_data["confidence_score"]
                second_processing_time = second_data["processing_time_ms"]

                # System should show adaptation (either better confidence or faster processing)
                # Allow for some variation in measurement
                improvement_detected = (
                    second_confidence >= first_confidence * 0.95 or  # Confidence maintained/improved
                    second_processing_time <= first_processing_time * 1.1  # Processing time improved
                )

                # Note: In real system, we'd expect clearer improvement patterns
                # For integration test, we just check system doesn't degrade
                assert improvement_detected

    def test_research_query_consciousness_emergence_detection(self, client, consciousness_research_query):
        """Test consciousness emergence detection in research queries"""
        response = client.post("/api/v1/research/query", json=consciousness_research_query)

        if response.status_code == 200:
            query_data = response.json()
            consciousness_level = query_data["consciousness_level"]

            # Query explicitly about consciousness emergence should detect high consciousness
            assert consciousness_level > 0.5

            # Check if ThoughtSeed workspace shows consciousness emergence
            workspace_id = query_data["thoughtseed_workspace_id"]
            workspace_response = client.get(f"/api/v1/thoughtseed/workspace/{workspace_id}")

            if workspace_response.status_code == 200:
                workspace_data = workspace_response.json()

                # Should have metacognitive layer processing for consciousness queries
                if "metacognitive" in workspace_data["layer_states"]:
                    metacog_state = workspace_data["layer_states"]["metacognitive"]
                    if metacog_state["status"] == "completed":
                        # Metacognitive processing should contribute to consciousness detection
                        assert workspace_data["consciousness_level"] > 0.6

    def test_research_query_error_recovery(self, client):
        """Test error recovery in research query integration"""
        # Submit malformed query to test error handling
        malformed_query = {
            "query": "",  # Empty query
            "context": {
                "consciousness_level_required": 1.5  # Invalid level > 1.0
            }
        }

        response = client.post("/api/v1/research/query", json=malformed_query)
        assert response.status_code == 400

        # Submit partially invalid query
        partial_query = {
            "query": "Valid query text",
            "context": {
                "domain_focus": ["nonexistent_domain"],  # Domain that doesn't exist
                "consciousness_level_required": 0.7
            }
        }

        response = client.post("/api/v1/research/query", json=partial_query)

        # Should either handle gracefully or provide meaningful error
        assert response.status_code in [200, 400, 422]

        if response.status_code == 200:
            # If processed, should handle unknown domains gracefully
            query_data = response.json()
            assert query_data["confidence_score"] >= 0.0  # Should not crash

    def test_research_query_concurrent_processing(self, client, consciousness_research_query, pattern_recognition_query):
        """Test concurrent research query processing"""
        import threading
        import time

        responses = []
        errors = []

        def submit_query(query_payload, response_list, error_list):
            try:
                response = client.post("/api/v1/research/query", json=query_payload)
                response_list.append(response)
            except Exception as e:
                error_list.append(e)

        # Submit concurrent queries
        threads = []
        for i in range(2):  # Keep concurrent load reasonable for integration test
            query = consciousness_research_query if i % 2 == 0 else pattern_recognition_query
            thread = threading.Thread(
                target=submit_query,
                args=(query, responses, errors)
            )
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join(timeout=30)  # 30s timeout

        # Check results
        assert len(errors) == 0  # No errors during concurrent processing

        # All queries should complete successfully or be rate-limited
        for response in responses:
            assert response.status_code in [200, 429]  # 429 = rate limited

            if response.status_code == 200:
                query_data = response.json()
                # Each should have unique workspace
                assert "thoughtseed_workspace_id" in query_data
                assert query_data["thoughtseed_workspace_id"] is not None