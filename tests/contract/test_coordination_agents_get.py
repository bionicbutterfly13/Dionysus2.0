"""
Contract test for GET /api/v1/coordination/agents endpoint
"""

import pytest
from httpx import AsyncClient

from src.dionysus_migration.api.main import app


class TestCoordinationAgentsGet:
    """Contract tests for coordination agents listing endpoint"""

    @pytest.mark.asyncio
    @pytest.mark.contract
    async def test_list_coordination_agents(self):
        """Test listing active DAEDALUS coordination agents"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/api/v1/coordination/agents")

            assert response.status_code == 200

            data = response.json()
            assert isinstance(data, list)