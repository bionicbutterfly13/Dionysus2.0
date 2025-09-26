"""
DAEDALUS Coordination model

Central orchestration component managing distributed migration subagents
with independent context windows and iterative self-improvement.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List
from uuid import uuid4

from pydantic import BaseModel, Field


class CoordinatorStatus(str, Enum):
    """DAEDALUS coordinator status"""
    INITIALIZING = "initializing"
    COORDINATING = "coordinating"
    SCALING = "scaling"
    MAINTAINING = "maintaining"
    SHUTTING_DOWN = "shutting_down"


class DaedalusCoordination(BaseModel):
    """DAEDALUS coordination model for managing distributed migration agents"""

    coordination_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique coordination instance identifier"
    )
    coordinator_status: CoordinatorStatus = Field(
        default=CoordinatorStatus.INITIALIZING,
        description="Current coordinator status"
    )
    active_subagents: List[Dict[str, str]] = Field(
        default_factory=list,
        description="List of active subagents with their context info"
    )
    task_queue: List[str] = Field(
        default_factory=list,
        description="Pending migration task IDs"
    )
    completed_tasks: List[str] = Field(
        default_factory=list,
        description="Successfully completed task IDs"
    )
    failed_tasks: List[str] = Field(
        default_factory=list,
        description="Failed task IDs with error details"
    )
    performance_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Coordination performance metrics"
    )
    learning_state: Dict[str, any] = Field(
        default_factory=dict,
        description="Coordination improvement data"
    )
    last_optimization: datetime = Field(
        default_factory=datetime.utcnow,
        description="Last optimization timestamp"
    )

    class Config:
        use_enum_values = True