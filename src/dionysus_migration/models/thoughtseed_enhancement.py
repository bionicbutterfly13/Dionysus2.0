"""
ThoughtSeed Enhancement model

Tracks complete rewrite process using ThoughtSeed framework patterns
to rebuild legacy component functionality with enhanced consciousness capabilities.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class EnhancementType(str, Enum):
    """Types of ThoughtSeed enhancements"""
    ACTIVE_INFERENCE = "active_inference"
    CONSCIOUSNESS_DETECTION = "consciousness_detection"
    META_COGNITIVE = "meta_cognitive"


class RewriteStatus(str, Enum):
    """Status of component rewrite process"""
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    TESTED = "tested"
    APPROVED = "approved"
    DEPLOYED = "deployed"
    ROLLED_BACK = "rolled_back"


class ThoughtSeedEnhancement(BaseModel):
    """ThoughtSeed enhancement model for complete component rewrite"""

    enhancement_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique enhancement identifier"
    )
    source_component_id: str = Field(
        description="Original legacy component ID"
    )
    enhanced_component_path: Optional[str] = Field(
        default=None,
        description="Path to new ThoughtSeed component"
    )
    enhancement_type: EnhancementType = Field(
        description="Type of ThoughtSeed enhancement applied"
    )
    legacy_functionality_preserved: List[str] = Field(
        default_factory=list,
        description="List of preserved interface methods"
    )
    new_capabilities_added: List[str] = Field(
        default_factory=list,
        description="List of new ThoughtSeed capabilities"
    )
    rewrite_status: RewriteStatus = Field(
        default=RewriteStatus.PLANNED,
        description="Current rewrite status"
    )
    performance_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Before/after performance comparison"
    )
    consciousness_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Enhanced consciousness capabilities"
    )
    approved_by: Optional[str] = Field(
        default=None,
        description="User who approved migration"
    )
    approved_at: Optional[datetime] = Field(
        default=None,
        description="Approval timestamp"
    )

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }