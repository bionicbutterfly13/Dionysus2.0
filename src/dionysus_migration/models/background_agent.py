"""
Background Migration Agent model

Independent subagents executing migration tasks without blocking
active development workflows.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class AgentStatus(str, Enum):
    """Background agent status"""
    IDLE = "idle"
    ANALYZING = "analyzing"
    REWRITING = "rewriting"
    TESTING = "testing"
    REPORTING = "reporting"


class BackgroundAgent(BaseModel):
    """Background migration agent model for independent task execution"""

    agent_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique agent identifier"
    )
    context_window_id: str = Field(
        description="Isolated context identifier"
    )
    agent_status: AgentStatus = Field(
        default=AgentStatus.IDLE,
        description="Current agent status"
    )
    current_task_id: Optional[str] = Field(
        default=None,
        description="Migration task currently being processed"
    )
    assigned_component_id: Optional[str] = Field(
        default=None,
        description="Component currently being migrated"
    )
    task_history: List[str] = Field(
        default_factory=list,
        description="List of completed task IDs"
    )
    performance_stats: Dict[str, float] = Field(
        default_factory=dict,
        description="Agent performance statistics"
    )
    context_isolation: bool = Field(
        default=True,
        description="Context window independence verification"
    )
    last_activity: datetime = Field(
        default_factory=datetime.utcnow,
        description="Last activity timestamp"
    )
    coordinator_id: str = Field(
        description="DAEDALUS coordinator managing this agent"
    )

    model_config = {"use_enum_values": True}