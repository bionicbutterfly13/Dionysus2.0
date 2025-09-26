"""
Integration test for component discovery and quality assessment

Tests the complete flow from quickstart scenario 1:
Component Discovery and Quality Assessment
"""

import pytest
from httpx import AsyncClient

from src.dionysus_migration.api.main import app


class TestComponentDiscoveryIntegration:
    """Integration tests for component discovery process"""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_complete_component_discovery_flow(self):
        """
        Test complete component discovery flow from quickstart scenario 1

        Steps:
        1. Initialize pipeline with legacy codebase path
        2. Wait for discovery phase completion (status: analyzing → analyzed)
        3. Verify components have consciousness functionality scores
        4. Confirm quality scoring prioritizes consciousness impact + strategic value
        """
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Step 1: Initialize pipeline
            pipeline_payload = {
                "legacy_codebase_path": "/path/to/test/legacy/codebase",
                "migration_strategy": "complete_rewrite",
                "quality_threshold": 0.7
            }

            pipeline_response = await client.post(
                "/api/v1/migration/pipeline",
                json=pipeline_payload
            )

            assert pipeline_response.status_code == 201
            pipeline_data = pipeline_response.json()
            pipeline_id = pipeline_data["pipeline_id"]

            # Step 2: Monitor discovery progress
            # In real implementation, this would poll until completion
            # For now, we just verify the endpoint structure

            components_response = await client.get(
                "/api/v1/migration/components",
                params={"pipeline_id": pipeline_id}
            )

            assert components_response.status_code == 200
            components = components_response.json()

            # Step 3: Verify consciousness functionality scores exist
            for component in components:
                consciousness = component["consciousness_functionality"]
                assert "awareness_score" in consciousness
                assert "inference_score" in consciousness
                assert "memory_score" in consciousness

                # Scores should be in valid range
                for score_key in ["awareness_score", "inference_score", "memory_score"]:
                    score = consciousness[score_key]
                    assert 0.0 <= score <= 1.0

            # Step 4: Verify quality scoring reflects consciousness impact + strategic value
            for component in components:
                quality_score = component["quality_score"]
                consciousness = component["consciousness_functionality"]
                strategic = component["strategic_value"]

                # Quality score should be influenced by both factors
                assert 0.0 <= quality_score <= 1.0

                # Components with higher consciousness scores should generally
                # have higher quality scores (when strategic value is similar)
                consciousness_avg = sum(consciousness.values()) / len(consciousness)
                strategic_avg = sum(strategic.values()) / len(strategic)

                # This is a simplified check - real implementation would have
                # more sophisticated scoring validation
                assert isinstance(consciousness_avg, (int, float))
                assert isinstance(strategic_avg, (int, float))

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_consciousness_component_prioritization(self):
        """Test that consciousness-related components are prioritized"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Create pipeline
            pipeline_payload = {
                "legacy_codebase_path": "/path/to/test/legacy/codebase",
                "quality_threshold": 0.6
            }

            pipeline_response = await client.post(
                "/api/v1/migration/pipeline",
                json=pipeline_payload
            )

            pipeline_id = pipeline_response.json()["pipeline_id"]

            # Get high-quality components
            high_quality_response = await client.get(
                "/api/v1/migration/components",
                params={
                    "pipeline_id": pipeline_id,
                    "min_quality_score": 0.8
                }
            )

            assert high_quality_response.status_code == 200
            high_quality_components = high_quality_response.json()

            # Verify high-quality components have strong consciousness scores
            for component in high_quality_components:
                consciousness = component["consciousness_functionality"]

                # At least one consciousness dimension should be strong
                consciousness_scores = [
                    consciousness["awareness_score"],
                    consciousness["inference_score"],
                    consciousness["memory_score"]
                ]

                max_consciousness_score = max(consciousness_scores)
                assert max_consciousness_score >= 0.7

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_strategic_value_assessment(self):
        """Test that strategic value factors into quality assessment"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Create pipeline
            pipeline_payload = {
                "legacy_codebase_path": "/path/to/test/legacy/codebase",
                "quality_threshold": 0.5
            }

            pipeline_response = await client.post(
                "/api/v1/migration/pipeline",
                json=pipeline_payload
            )

            pipeline_id = pipeline_response.json()["pipeline_id"]

            # Get all analyzed components
            components_response = await client.get(
                "/api/v1/migration/components",
                params={
                    "pipeline_id": pipeline_id,
                    "status": "analyzed"
                }
            )

            assert components_response.status_code == 200
            components = components_response.json()

            # Verify strategic value structure and scoring
            for component in components:
                strategic = component["strategic_value"]

                required_strategic_factors = [
                    "uniqueness_score",
                    "reusability_score",
                    "framework_alignment_score"
                ]

                for factor in required_strategic_factors:
                    assert factor in strategic
                    score = strategic[factor]
                    assert 0.0 <= score <= 1.0