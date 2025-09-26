"""
Contract test for POST /api/v1/migration/pipeline endpoint

Tests the API contract for creating new migration pipelines according
to the OpenAPI specification. This test MUST FAIL until the endpoint
is implemented.
"""

import pytest
from httpx import AsyncClient

from src.dionysus_migration.api.main import app


class TestMigrationPipelinePost:
    """Contract tests for migration pipeline creation endpoint"""

    @pytest.mark.asyncio
    @pytest.mark.contract
    async def test_create_migration_pipeline_success(self):
        """Test successful migration pipeline creation"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Valid request payload
            payload = {
                "legacy_codebase_path": "/path/to/legacy/dionysus",
                "migration_strategy": "complete_rewrite",
                "quality_threshold": 0.8
            }

            response = await client.post("/api/v1/migration/pipeline", json=payload)

            # Contract assertions
            assert response.status_code == 201

            data = response.json()
            assert "pipeline_id" in data
            assert "status" in data
            assert "estimated_components" in data
            assert "coordinator_agent_id" in data

            # Validate response structure
            assert data["status"] == "initializing"
            assert isinstance(data["estimated_components"], int)
            assert data["estimated_components"] >= 0

    @pytest.mark.asyncio
    @pytest.mark.contract
    async def test_create_migration_pipeline_default_strategy(self):
        """Test pipeline creation with default migration strategy"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            payload = {
                "legacy_codebase_path": "/path/to/legacy/dionysus",
                "quality_threshold": 0.7
            }

            response = await client.post("/api/v1/migration/pipeline", json=payload)

            assert response.status_code == 201
            data = response.json()

            # Should use default strategy (complete_rewrite)
            assert "pipeline_id" in data
            assert data["status"] == "initializing"

    @pytest.mark.asyncio
    @pytest.mark.contract
    async def test_create_migration_pipeline_missing_required_field(self):
        """Test pipeline creation with missing required field"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            payload = {
                "migration_strategy": "complete_rewrite",
                "quality_threshold": 0.7
                # Missing legacy_codebase_path
            }

            response = await client.post("/api/v1/migration/pipeline", json=payload)

            assert response.status_code == 422  # Validation error

            error_data = response.json()
            assert "detail" in error_data

    @pytest.mark.asyncio
    @pytest.mark.contract
    async def test_create_migration_pipeline_invalid_strategy(self):
        """Test pipeline creation with invalid migration strategy"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            payload = {
                "legacy_codebase_path": "/path/to/legacy/dionysus",
                "migration_strategy": "invalid_strategy",
                "quality_threshold": 0.7
            }

            response = await client.post("/api/v1/migration/pipeline", json=payload)

            assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    @pytest.mark.contract
    async def test_create_migration_pipeline_invalid_threshold(self):
        """Test pipeline creation with invalid quality threshold"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            payload = {
                "legacy_codebase_path": "/path/to/legacy/dionysus",
                "migration_strategy": "complete_rewrite",
                "quality_threshold": 1.5  # Invalid: > 1.0
            }

            response = await client.post("/api/v1/migration/pipeline", json=payload)

            assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    @pytest.mark.contract
    async def test_create_migration_pipeline_negative_threshold(self):
        """Test pipeline creation with negative quality threshold"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            payload = {
                "legacy_codebase_path": "/path/to/legacy/dionysus",
                "migration_strategy": "complete_rewrite",
                "quality_threshold": -0.1  # Invalid: < 0.0
            }

            response = await client.post("/api/v1/migration/pipeline", json=payload)

            assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    @pytest.mark.contract
    async def test_create_migration_pipeline_selective_enhancement(self):
        """Test pipeline creation with selective enhancement strategy"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            payload = {
                "legacy_codebase_path": "/path/to/legacy/dionysus",
                "migration_strategy": "selective_enhancement",
                "quality_threshold": 0.6
            }

            response = await client.post("/api/v1/migration/pipeline", json=payload)

            assert response.status_code == 201

            data = response.json()
            assert data["status"] == "initializing"
            assert "coordinator_agent_id" in data

    @pytest.mark.asyncio
    @pytest.mark.contract
    async def test_create_migration_pipeline_content_type_validation(self):
        """Test that endpoint requires JSON content type"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            payload = "legacy_codebase_path=/path/to/legacy"

            response = await client.post(
                "/api/v1/migration/pipeline",
                content=payload,
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )

            # Should reject non-JSON content
            assert response.status_code in [415, 422]