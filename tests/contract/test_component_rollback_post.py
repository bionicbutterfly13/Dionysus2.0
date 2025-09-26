"""
Contract test for POST /api/v1/migration/components/{component_id}/rollback endpoint
"""

import pytest
from httpx import AsyncClient

from src.dionysus_migration.api.main import app


class TestComponentRollbackPost:
    """Contract tests for component rollback endpoint"""

    @pytest.mark.asyncio
    @pytest.mark.contract
    async def test_rollback_component_success(self):
        """Test successful component rollback"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            component_id = "migrated_component"

            response = await client.post(
                f"/api/v1/migration/components/{component_id}/rollback"
            )

            assert response.status_code == 202

            data = response.json()
            assert "rollback_id" in data
            assert "estimated_completion" in data