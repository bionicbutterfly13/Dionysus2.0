"""
Contract test for GET /api/v1/migration/components endpoint

Tests the API contract for listing legacy components according to the
OpenAPI specification. This test MUST FAIL until the endpoint is implemented.
"""

import pytest
from httpx import AsyncClient

from src.dionysus_migration.api.main import app


class TestMigrationComponentsGet:
    """Contract tests for migration components listing endpoint"""

    @pytest.mark.asyncio
    @pytest.mark.contract
    async def test_list_components_with_pipeline_id(self):
        """Test listing components with required pipeline_id parameter"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            pipeline_id = "550e8400-e29b-41d4-a716-446655440000"

            response = await client.get(
                "/api/v1/migration/components",
                params={"pipeline_id": pipeline_id}
            )

            assert response.status_code == 200

            data = response.json()
            assert isinstance(data, list)

            # Validate component structure if components exist
            for component in data:
                self._validate_component_structure(component)

    @pytest.mark.asyncio
    @pytest.mark.contract
    async def test_list_components_missing_pipeline_id(self):
        """Test that pipeline_id parameter is required"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/api/v1/migration/components")

            # Should return 422 for missing required parameter
            assert response.status_code == 422

    @pytest.mark.asyncio
    @pytest.mark.contract
    async def test_list_components_filter_by_status(self):
        """Test filtering components by status"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            pipeline_id = "550e8400-e29b-41d4-a716-446655440000"

            # Test each valid status filter
            valid_statuses = [
                "pending",
                "analyzing",
                "analyzed",
                "migrating",
                "completed",
                "failed"
            ]

            for status in valid_statuses:
                response = await client.get(
                    "/api/v1/migration/components",
                    params={
                        "pipeline_id": pipeline_id,
                        "status": status
                    }
                )

                assert response.status_code == 200

                data = response.json()
                assert isinstance(data, list)

                # If components exist, verify they have the requested status
                for component in data:
                    assert component["analysis_status"] == status

    @pytest.mark.asyncio
    @pytest.mark.contract
    async def test_list_components_filter_by_quality_score(self):
        """Test filtering components by minimum quality score"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            pipeline_id = "550e8400-e29b-41d4-a716-446655440000"
            min_quality = 0.8

            response = await client.get(
                "/api/v1/migration/components",
                params={
                    "pipeline_id": pipeline_id,
                    "min_quality_score": min_quality
                }
            )

            assert response.status_code == 200

            data = response.json()
            assert isinstance(data, list)

            # Verify all returned components meet quality threshold
            for component in data:
                assert component["quality_score"] >= min_quality

    @pytest.mark.asyncio
    @pytest.mark.contract
    async def test_list_components_combined_filters(self):
        """Test using multiple filters together"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            pipeline_id = "550e8400-e29b-41d4-a716-446655440000"

            response = await client.get(
                "/api/v1/migration/components",
                params={
                    "pipeline_id": pipeline_id,
                    "status": "analyzed",
                    "min_quality_score": 0.7
                }
            )

            assert response.status_code == 200

            data = response.json()
            assert isinstance(data, list)

            # Verify components meet both criteria
            for component in data:
                assert component["analysis_status"] == "analyzed"
                assert component["quality_score"] >= 0.7

    def _validate_component_structure(self, component_data: dict):
        """Validate component data structure matches contract"""
        required_fields = [
            "component_id",
            "name",
            "file_path",
            "dependencies",
            "consciousness_functionality",
            "strategic_value",
            "quality_score",
            "analysis_status",
            "extracted_at"
        ]

        for field in required_fields:
            assert field in component_data, f"Missing required field: {field}"

        # Validate field types
        assert isinstance(component_data["dependencies"], list)
        assert isinstance(component_data["consciousness_functionality"], dict)
        assert isinstance(component_data["strategic_value"], dict)
        assert isinstance(component_data["quality_score"], (int, float))

        # Validate quality score range
        quality_score = component_data["quality_score"]
        assert 0.0 <= quality_score <= 1.0

        # Validate consciousness functionality structure
        consciousness = component_data["consciousness_functionality"]
        assert "awareness_score" in consciousness
        assert "inference_score" in consciousness
        assert "memory_score" in consciousness

        # Validate strategic value structure
        strategic = component_data["strategic_value"]
        assert "uniqueness_score" in strategic
        assert "reusability_score" in strategic
        assert "framework_alignment_score" in strategic

        # Validate status enum
        valid_statuses = [
            "pending",
            "analyzing",
            "analyzed",
            "failed"
        ]
        assert component_data["analysis_status"] in valid_statuses

    @pytest.mark.asyncio
    @pytest.mark.contract
    async def test_list_components_invalid_status_filter(self):
        """Test filtering with invalid status value"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            pipeline_id = "550e8400-e29b-41d4-a716-446655440000"

            response = await client.get(
                "/api/v1/migration/components",
                params={
                    "pipeline_id": pipeline_id,
                    "status": "invalid_status"
                }
            )

            # Should return 422 for invalid status value
            assert response.status_code == 422

    @pytest.mark.asyncio
    @pytest.mark.contract
    async def test_list_components_invalid_quality_score(self):
        """Test filtering with invalid quality score range"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            pipeline_id = "550e8400-e29b-41d4-a716-446655440000"

            # Test quality score > 1.0
            response = await client.get(
                "/api/v1/migration/components",
                params={
                    "pipeline_id": pipeline_id,
                    "min_quality_score": 1.5
                }
            )
            assert response.status_code == 422

            # Test negative quality score
            response = await client.get(
                "/api/v1/migration/components",
                params={
                    "pipeline_id": pipeline_id,
                    "min_quality_score": -0.1
                }
            )
            assert response.status_code == 422

    @pytest.mark.asyncio
    @pytest.mark.contract
    async def test_list_components_invalid_pipeline_id(self):
        """Test with invalid pipeline ID format"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get(
                "/api/v1/migration/components",
                params={"pipeline_id": "not-a-uuid"}
            )

            # Should return 422 for invalid UUID format
            assert response.status_code == 422

    @pytest.mark.asyncio
    @pytest.mark.contract
    async def test_list_components_content_type(self):
        """Test that response has correct content type"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            pipeline_id = "550e8400-e29b-41d4-a716-446655440000"

            response = await client.get(
                "/api/v1/migration/components",
                params={"pipeline_id": pipeline_id}
            )

            assert "application/json" in response.headers.get("content-type", "")