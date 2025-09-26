"""
Integration test for ThoughtSeed component enhancement
"""

import pytest
from httpx import AsyncClient

from src.dionysus_migration.api.main import app


class TestThoughtSeedEnhancementIntegration:
    """Integration tests for ThoughtSeed enhancement process"""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_thoughtseed_enhancement_flow(self):
        """Test ThoughtSeed enhancement complete flow"""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Approve component for migration
            component_id = "test_consciousness_component"
            approval_payload = {
                "approved": True,
                "approval_notes": "High consciousness impact",
                "consciousness_impact_review": {
                    "awareness_enhancement": True,
                    "inference_improvement": True,
                    "memory_integration": True
                }
            }

            approval_response = await client.post(
                f"/api/v1/migration/components/{component_id}/approve",
                json=approval_payload
            )

            # Should schedule migration
            assert approval_response.status_code == 200

            # Check ThoughtSeed enhancements
            enhancements_response = await client.get("/api/v1/thoughtseed/enhancements")
            assert enhancements_response.status_code == 200