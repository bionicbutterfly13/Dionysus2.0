"""
Contract test for GET /api/v1/migration/pipeline endpoint

Tests the API contract for listing active migration pipelines according
to the OpenAPI specification. This test MUST FAIL until the endpoint
is implemented.
"""

import pytest
from httpx import AsyncClient

from src.dionysus_migration.api.main import app


class TestMigrationPipelineGet:
    """Contract tests for migration pipeline listing endpoint"""

    @pytest.mark.asyncio
    @pytest.mark.contract
    async def test_list_migration_pipelines_empty(self):
        """Test listing pipelines when none exist"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/api/v1/migration/pipeline")

            assert response.status_code == 200

            data = response.json()
            assert isinstance(data, list)
            # May be empty if no pipelines exist

    @pytest.mark.asyncio
    @pytest.mark.contract
    async def test_list_migration_pipelines_response_structure(self):
        """Test that pipeline list response has correct structure"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/api/v1/migration/pipeline")

            assert response.status_code == 200

            data = response.json()
            assert isinstance(data, list)

            # If pipelines exist, validate structure
            for pipeline in data:
                assert "pipeline_id" in pipeline
                assert "status" in pipeline
                assert "total_components" in pipeline
                assert "completed_components" in pipeline
                assert "failed_components" in pipeline
                assert "active_agents" in pipeline
                assert "started_at" in pipeline
                assert "estimated_completion" in pipeline
                assert "coordinator_agent_id" in pipeline

                # Validate field types
                assert isinstance(pipeline["total_components"], int)
                assert isinstance(pipeline["completed_components"], int)
                assert isinstance(pipeline["failed_components"], int)
                assert isinstance(pipeline["active_agents"], list)

                # Validate status enum
                valid_statuses = [
                    "initializing",
                    "analyzing",
                    "migrating",
                    "testing",
                    "completed",
                    "failed"
                ]
                assert pipeline["status"] in valid_statuses

    @pytest.mark.asyncio
    @pytest.mark.contract
    async def test_list_migration_pipelines_with_multiple_pipelines(self):
        """Test listing multiple pipelines"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # This test assumes pipelines might exist
            response = await client.get("/api/v1/migration/pipeline")

            assert response.status_code == 200

            data = response.json()
            assert isinstance(data, list)

            # If multiple pipelines exist, ensure each has unique ID
            if len(data) > 1:
                pipeline_ids = [p["pipeline_id"] for p in data]
                assert len(pipeline_ids) == len(set(pipeline_ids))  # All unique

    @pytest.mark.asyncio
    @pytest.mark.contract
    async def test_list_migration_pipelines_accepts_get_only(self):
        """Test that endpoint only accepts GET requests"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Test other HTTP methods are rejected
            response = await client.post("/api/v1/migration/pipeline")
            assert response.status_code == 405  # Method not allowed

            response = await client.put("/api/v1/migration/pipeline")
            assert response.status_code == 405  # Method not allowed

            response = await client.delete("/api/v1/migration/pipeline")
            assert response.status_code == 405  # Method not allowed

    @pytest.mark.asyncio
    @pytest.mark.contract
    async def test_list_migration_pipelines_content_type(self):
        """Test that response has correct content type"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/api/v1/migration/pipeline")

            assert response.status_code == 200
            assert "application/json" in response.headers.get("content-type", "")

    @pytest.mark.asyncio
    @pytest.mark.contract
    async def test_list_migration_pipelines_no_query_params_required(self):
        """Test that endpoint works without query parameters"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/api/v1/migration/pipeline")

            assert response.status_code == 200
            # Should work without any query parameters

    @pytest.mark.asyncio
    @pytest.mark.contract
    async def test_list_migration_pipelines_ignores_unknown_params(self):
        """Test that endpoint ignores unknown query parameters"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get(
                "/api/v1/migration/pipeline",
                params={"unknown_param": "value"}
            )

            assert response.status_code == 200
            # Should ignore unknown parameters and still work