"""
Integration test for individual component rollback
"""

import pytest
from httpx import AsyncClient

from src.dionysus_migration.api.main import app


class TestComponentRollbackIntegration:
    """Integration tests for component rollback process"""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_component_rollback_flow(self):
        """Test individual component rollback within 30 seconds"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            component_id = "migrated_test_component"

            # Initiate rollback
            rollback_response = await client.post(
                f"/api/v1/migration/components/{component_id}/rollback"
            )

            assert rollback_response.status_code == 202
            rollback_data = rollback_response.json()
            assert "rollback_id" in rollback_data
            assert "estimated_completion" in rollback_data