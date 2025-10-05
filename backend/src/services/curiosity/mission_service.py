"""
Curiosity Mission Service

Provides in-memory management of curiosity missions for Spec 029 / T030.
Designed for local development; production should swap with persistent storage.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional
import uuid

from models.curiosity_mission import (
    CuriosityMission,
    MissionStatus,
    CuriosityType,
)

logger = logging.getLogger(__name__)


class CuriosityMissionService:
    """In-memory curiosity mission manager."""

    def __init__(self) -> None:
        self._missions: Dict[str, CuriosityMission] = {}
        self._lock = asyncio.Lock()

    async def create_mission(
        self,
        user_id: str,
        mission_title: str,
        mission_description: str,
        primary_curiosity_type: CuriosityType,
        research_questions: Optional[List[str]] = None,
    ) -> CuriosityMission:
        mission_id = f"cm_{uuid.uuid4().hex[:8]}"
        mission = CuriosityMission(
            mission_id=mission_id,
            user_id=user_id,
            mission_title=mission_title,
            mission_description=mission_description,
            primary_curiosity_type=primary_curiosity_type,
            research_questions=research_questions or [],
        )

        async with self._lock:
            self._missions[mission_id] = mission

        logger.info("Curiosity mission created: %s", mission_id)
        return mission

    async def list_missions(self) -> List[CuriosityMission]:
        async with self._lock:
            return list(self._missions.values())

    async def get_mission(self, mission_id: str) -> Optional[CuriosityMission]:
        async with self._lock:
            return self._missions.get(mission_id)

    async def update_mission_status(
        self,
        mission_id: str,
        mission_status: MissionStatus,
    ) -> Optional[CuriosityMission]:
        async with self._lock:
            mission = self._missions.get(mission_id)
            if not mission:
                return None

            mission.mission_status = mission_status
            mission.updated_at = datetime.utcnow()
            self._missions[mission_id] = mission

        logger.info("Curiosity mission %s status updated to %s", mission_id, mission_status)
        return mission


_service_instance: Optional[CuriosityMissionService] = None


def get_curiosity_mission_service() -> CuriosityMissionService:
    global _service_instance
    if _service_instance is None:
        _service_instance = CuriosityMissionService()
    return _service_instance
