"""
Migration Pipeline model

Orchestrates distributed background migration process using DAEDALUS coordination.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class PipelineStatus(str, Enum):
    """Pipeline status enumeration"""
    INITIALIZING = "initializing"
    ANALYZING = "analyzing"
    MIGRATING = "migrating"
    TESTING = "testing"
    COMPLETED = "completed"
    FAILED = "failed"


class MigrationPipeline(BaseModel):
    """Migration pipeline model for orchestrating distributed background migration"""

    pipeline_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique pipeline identifier"
    )
    status: PipelineStatus = Field(
        default=PipelineStatus.INITIALIZING,
        description="Current pipeline status"
    )
    total_components: int = Field(
        ge=0,
        description="Total number of components to migrate"
    )
    completed_components: int = Field(
        default=0,
        ge=0,
        description="Number of successfully migrated components"
    )
    failed_components: int = Field(
        default=0,
        ge=0,
        description="Number of failed component migrations"
    )
    active_agents: List[str] = Field(
        default_factory=list,
        description="List of DAEDALUS subagent IDs currently processing"
    )
    started_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Pipeline start timestamp"
    )
    estimated_completion: Optional[datetime] = Field(
        default=None,
        description="Estimated completion timestamp"
    )
    coordinator_agent_id: str = Field(
        description="Primary DAEDALUS coordinator ID"
    )

    # Configuration
    legacy_codebase_path: str = Field(
        description="Path to legacy Dionysus consciousness codebase"
    )
    migration_strategy: str = Field(
        default="complete_rewrite",
        description="Migration strategy (complete_rewrite, selective_enhancement)"
    )
    quality_threshold: float = Field(
        ge=0.0, le=1.0,
        description="Minimum quality score for component migration"
    )

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }