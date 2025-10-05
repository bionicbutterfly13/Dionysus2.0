"""
Curiosity Mission API Routes - Spec 029 / T030
Flux Self-Evolving Consciousness Emulator

Handles curiosity mission creation, management, and replay scheduling.
"""

import logging
from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from models.curiosity_mission import CuriosityMission, CuriosityType, MissionStatus
from services.curiosity.mission_service import (
    CuriosityMissionService,
    get_curiosity_mission_service,
)

logger = logging.getLogger(__name__)

router = APIRouter()


class CuriosityMissionCreateRequest(BaseModel):
    user_id: str = Field(..., description="User initiating the mission")
    mission_title: str = Field(..., description="Title of the mission")
    mission_description: str = Field("", description="Detailed mission description")
    primary_curiosity_type: CuriosityType = Field(..., description="Primary curiosity type")
    research_questions: List[str] = Field(default_factory=list, description="Guiding research questions")


class CuriosityMissionUpdateRequest(BaseModel):
    mission_status: MissionStatus = Field(..., description="Updated mission status")


class CuriosityMissionResponse(BaseModel):
    mission_id: str
    user_id: str
    mission_title: str
    mission_description: str
    mission_status: MissionStatus
    primary_curiosity_type: CuriosityType
    research_questions: List[str]
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_model(cls, mission: CuriosityMission) -> "CuriosityMissionResponse":
        return cls(
            mission_id=mission.mission_id,
            user_id=mission.user_id,
            mission_title=mission.mission_title,
            mission_description=mission.mission_description,
            mission_status=mission.mission_status,
            primary_curiosity_type=mission.primary_curiosity_type,
            research_questions=mission.research_questions,
            created_at=mission.created_at,
            updated_at=mission.updated_at,
        )


@router.post(
    "/curiosity/missions",
    response_model=CuriosityMissionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_curiosity_mission(
    request: CuriosityMissionCreateRequest,
    service: CuriosityMissionService = Depends(get_curiosity_mission_service),
):
    """Create a curiosity mission."""
    mission = await service.create_mission(
        user_id=request.user_id,
        mission_title=request.mission_title,
        mission_description=request.mission_description,
        primary_curiosity_type=request.primary_curiosity_type,
        research_questions=request.research_questions,
    )

    return CuriosityMissionResponse.from_model(mission)


@router.get(
    "/curiosity/missions/{mission_id}",
    response_model=CuriosityMissionResponse,
)
async def get_curiosity_mission(
    mission_id: str,
    service: CuriosityMissionService = Depends(get_curiosity_mission_service),
):
    """Get curiosity mission by ID."""
    mission = await service.get_mission(mission_id)
    if mission is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Mission not found")

    return CuriosityMissionResponse.from_model(mission)


@router.patch(
    "/curiosity/missions/{mission_id}",
    response_model=CuriosityMissionResponse,
)
async def update_curiosity_mission(
    mission_id: str,
    request: CuriosityMissionUpdateRequest,
    service: CuriosityMissionService = Depends(get_curiosity_mission_service),
):
    """Update curiosity mission status."""
    mission = await service.update_mission_status(mission_id, request.mission_status)
    if mission is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Mission not found")

    return CuriosityMissionResponse.from_model(mission)


@router.get("/curiosity/missions")
async def list_curiosity_missions(
    service: CuriosityMissionService = Depends(get_curiosity_mission_service),
):
    """List all curiosity missions."""
    missions = await service.list_missions()
    return {
        "missions": [CuriosityMissionResponse.from_model(m).model_dump() for m in missions]
    }
