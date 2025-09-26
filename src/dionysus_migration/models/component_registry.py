"""
Component Registry model

Catalog of all identified, extracted, and migrated components with status,
enhancements, and integration information.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class MigrationStatus(str, Enum):
    """Migration status for components"""
    IDENTIFIED = "identified"
    QUEUED = "queued"
    MIGRATING = "migrating"
    ENHANCED = "enhanced"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"


class ComponentRegistry(BaseModel):
    """Component registry model for tracking all migration components"""

    registry_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique registry entry identifier"
    )
    component_name: str = Field(
        description="Canonical component name"
    )
    legacy_component_id: str = Field(
        description="Reference to original legacy component"
    )
    enhanced_component_id: Optional[str] = Field(
        default=None,
        description="Reference to ThoughtSeed enhanced version"
    )
    migration_status: MigrationStatus = Field(
        default=MigrationStatus.IDENTIFIED,
        description="Current migration status"
    )
    deployment_environment: Optional[str] = Field(
        default=None,
        description="Environment where enhanced component is active"
    )
    backward_compatibility: bool = Field(
        default=True,
        description="Whether legacy interface is preserved"
    )
    rollback_available: bool = Field(
        default=True,
        description="Whether component can revert to legacy version"
    )
    usage_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Performance and adoption tracking metrics"
    )
    last_updated: datetime = Field(
        default_factory=datetime.utcnow,
        description="Last update timestamp"
    )

    class Config:
        use_enum_values = True