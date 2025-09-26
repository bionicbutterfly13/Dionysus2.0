"""
Contract test for GET /api/v1/thoughtseed/enhancements endpoint
"""

import pytest
from httpx import AsyncClient

from src.dionysus_migration.api.main import app


class TestThoughtSeedEnhancementsGet:
    """Contract tests for ThoughtSeed enhancements listing endpoint"""

    @pytest.mark.asyncio
    @pytest.mark.contract
    async def test_list_thoughtseed_enhancements(self):
        """Test listing ThoughtSeed enhancement results"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/api/v1/thoughtseed/enhancements")

            assert response.status_code == 200

            data = response.json()
            assert isinstance(data, list)