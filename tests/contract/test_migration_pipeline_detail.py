"""
Contract test for GET /api/v1/migration/pipeline/{pipeline_id} endpoint

Tests the API contract for retrieving specific migration pipeline details
according to the OpenAPI specification. This test MUST FAIL until the
endpoint is implemented.
"""

import pytest
from httpx import AsyncClient

from src.dionysus_migration.api.main import app


class TestMigrationPipelineDetail:
    """Contract tests for migration pipeline detail endpoint"""

    @pytest.mark.asyncio
    @pytest.mark.contract
    async def test_get_migration_pipeline_success(self):
        """Test successful pipeline detail retrieval"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Use a valid UUID format
            pipeline_id = "550e8400-e29b-41d4-a716-446655440000"

            response = await client.get(f"/api/v1/migration/pipeline/{pipeline_id}")

            # Expected to return pipeline details or 404 if not found
            assert response.status_code in [200, 404]

            if response.status_code == 200:
                data = response.json()
                self._validate_pipeline_structure(data)
                assert data["pipeline_id"] == pipeline_id

    def _validate_pipeline_structure(self, pipeline_data: dict):
        """Validate pipeline data structure matches contract"""
        required_fields = [
            "pipeline_id",
            "status",
            "total_components",
            "completed_components",
            "failed_components",
            "active_agents",
            "started_at",
            "estimated_completion",
            "coordinator_agent_id"
        ]

        for field in required_fields:
            assert field in pipeline_data, f"Missing required field: {field}"

        # Validate field types
        assert isinstance(pipeline_data["total_components"], int)
        assert isinstance(pipeline_data["completed_components"], int)
        assert isinstance(pipeline_data["failed_components"], int)
        assert isinstance(pipeline_data["active_agents"], list)

        # Validate status enum
        valid_statuses = [
            "initializing",
            "analyzing",
            "migrating",
            "testing",
            "completed",
            "failed"
        ]
        assert pipeline_data["status"] in valid_statuses

        # Validate UUID format for coordinator agent
        coordinator_id = pipeline_data["coordinator_agent_id"]
        assert isinstance(coordinator_id, str)
        assert len(coordinator_id) > 0

    @pytest.mark.asyncio
    @pytest.mark.contract
    async def test_get_migration_pipeline_not_found(self):
        """Test pipeline detail retrieval for non-existent pipeline"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Use a non-existent but valid UUID
            pipeline_id = "00000000-0000-0000-0000-000000000000"

            response = await client.get(f"/api/v1/migration/pipeline/{pipeline_id}")

            assert response.status_code == 404

            error_data = response.json()
            assert "detail" in error_data

    @pytest.mark.asyncio
    @pytest.mark.contract
    async def test_get_migration_pipeline_invalid_uuid(self):
        """Test pipeline detail retrieval with invalid UUID format"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            invalid_uuid = "not-a-uuid"

            response = await client.get(f"/api/v1/migration/pipeline/{invalid_uuid}")

            # Should return 422 for invalid UUID format
            assert response.status_code == 422

    @pytest.mark.asyncio
    @pytest.mark.contract
    async def test_get_migration_pipeline_empty_id(self):
        """Test pipeline detail retrieval with empty pipeline ID"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/api/v1/migration/pipeline/")

            # Should not match the route (trailing slash)
            assert response.status_code == 404

    @pytest.mark.asyncio
    @pytest.mark.contract
    async def test_get_migration_pipeline_content_type(self):
        """Test that response has correct content type"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            pipeline_id = "550e8400-e29b-41d4-a716-446655440000"

            response = await client.get(f"/api/v1/migration/pipeline/{pipeline_id}")

            # Whether 200 or 404, should return JSON
            assert "application/json" in response.headers.get("content-type", "")

    @pytest.mark.asyncio
    @pytest.mark.contract
    async def test_get_migration_pipeline_active_agents_structure(self):
        """Test that active_agents field has correct structure when present"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            pipeline_id = "550e8400-e29b-41d4-a716-446655440000"

            response = await client.get(f"/api/v1/migration/pipeline/{pipeline_id}")

            if response.status_code == 200:
                data = response.json()
                active_agents = data["active_agents"]

                assert isinstance(active_agents, list)

                # If agents exist, validate their structure
                for agent_id in active_agents:
                    assert isinstance(agent_id, str)
                    assert len(agent_id) > 0  # Should be valid UUID format

    @pytest.mark.asyncio
    @pytest.mark.contract
    async def test_get_migration_pipeline_only_get_method(self):
        """Test that endpoint only accepts GET requests"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            pipeline_id = "550e8400-e29b-41d4-a716-446655440000"

            # Test other HTTP methods are rejected
            response = await client.post(f"/api/v1/migration/pipeline/{pipeline_id}")
            assert response.status_code == 405

            response = await client.put(f"/api/v1/migration/pipeline/{pipeline_id}")
            assert response.status_code == 405

            response = await client.patch(f"/api/v1/migration/pipeline/{pipeline_id}")
            assert response.status_code == 405

    @pytest.mark.asyncio
    @pytest.mark.contract
    async def test_get_migration_pipeline_numeric_constraints(self):
        """Test numeric field constraints in response"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            pipeline_id = "550e8400-e29b-41d4-a716-446655440000"

            response = await client.get(f"/api/v1/migration/pipeline/{pipeline_id}")

            if response.status_code == 200:
                data = response.json()

                # Component counts should be non-negative
                assert data["total_components"] >= 0
                assert data["completed_components"] >= 0
                assert data["failed_components"] >= 0

                # Completed + failed should not exceed total
                total = data["total_components"]
                completed = data["completed_components"]
                failed = data["failed_components"]

                if total > 0:
                    assert completed + failed <= total