"""
Integration test for DAEDALUS coordination efficiency
"""

import pytest
from httpx import AsyncClient

from src.dionysus_migration.api.main import app


class TestDaedalusCoordinationIntegration:
    """Integration tests for DAEDALUS coordination process"""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_daedalus_coordination_efficiency(self):
        """Test DAEDALUS coordination and agent management"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Get coordination agents
            agents_response = await client.get("/api/v1/coordination/agents")
            assert agents_response.status_code == 200

            agents = agents_response.json()
            assert isinstance(agents, list)

            # Verify coordination efficiency metrics if agents exist
            for agent in agents:
                if "performance_metrics" in agent:
                    metrics = agent["performance_metrics"]
                    assert "throughput_tasks_per_hour" in metrics
                    assert "success_rate_percent" in metrics
                    assert "resource_utilization_percent" in metrics