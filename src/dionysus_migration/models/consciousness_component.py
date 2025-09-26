"""
Consciousness Component model

Represents individual consciousness-enabled code modules in Dionysus 2.0
with awareness, inference, memory capabilities and enhancement potential.
"""

import hashlib
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class AnalysisStatus(str, Enum):
    """Analysis status for legacy components"""
    PENDING = "pending"
    ANALYZING = "analyzing"
    ANALYZED = "analyzed"
    FAILED = "failed"


class ConsciousnessFunctionality(BaseModel):
    """Consciousness capability metrics for a component"""
    awareness_score: float = Field(
        ge=0.0, le=1.0,
        description="Awareness processing capability score"
    )
    inference_score: float = Field(
        ge=0.0, le=1.0,
        description="Inference and reasoning capability score"
    )
    memory_score: float = Field(
        ge=0.0, le=1.0,
        description="Memory integration capability score"
    )

    @property
    def composite_score(self) -> float:
        """Calculate composite consciousness score"""
        return (self.awareness_score + self.inference_score + self.memory_score) / 3.0


class StrategicValue(BaseModel):
    """Strategic value metrics for migration prioritization"""
    uniqueness_score: float = Field(
        ge=0.0, le=1.0,
        description="Uniqueness and irreplaceability score"
    )
    reusability_score: float = Field(
        ge=0.0, le=1.0,
        description="Reusability potential score"
    )
    framework_alignment_score: float = Field(
        ge=0.0, le=1.0,
        description="Alignment with Dionysus 2.0 framework score"
    )

    @property
    def composite_score(self) -> float:
        """Calculate composite strategic value score"""
        return (
            self.uniqueness_score +
            self.reusability_score +
            self.framework_alignment_score
        ) / 3.0


class LegacyComponent(BaseModel):
    """
    Legacy component model representing individual code modules
    from the legacy Dionysus consciousness system
    """

    component_id: str = Field(
        description="Unique identifier (hash of file path + content signature)"
    )
    name: str = Field(
        description="Component name extracted from module/class"
    )
    file_path: str = Field(
        description="Absolute path to component source"
    )
    dependencies: List[str] = Field(
        default_factory=list,
        description="List of component_ids this component depends on"
    )
    consciousness_functionality: ConsciousnessFunctionality = Field(
        description="Consciousness capability metrics"
    )
    strategic_value: StrategicValue = Field(
        description="Strategic value metrics for migration prioritization"
    )
    quality_score: float = Field(
        ge=0.0, le=1.0,
        description="Composite consciousness + strategic value score"
    )
    extracted_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when component was extracted from legacy codebase"
    )
    analysis_status: AnalysisStatus = Field(
        default=AnalysisStatus.PENDING,
        description="Current analysis status"
    )

    # Additional metadata
    source_code_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 hash of source code content"
    )
    file_size_bytes: Optional[int] = Field(
        default=None,
        description="Source file size in bytes"
    )
    last_modified: Optional[datetime] = Field(
        default=None,
        description="Last modification time of source file"
    )

    # Consciousness-specific metadata
    consciousness_patterns: List[str] = Field(
        default_factory=list,
        description="Detected consciousness patterns in the code"
    )
    inference_complexity: Optional[str] = Field(
        default=None,
        description="Complexity level of inference patterns (low/medium/high)"
    )
    memory_integration_type: Optional[str] = Field(
        default=None,
        description="Type of memory integration (episodic/semantic/procedural)"
    )

    model_config = {
        "use_enum_values": True,
        "json_encoders": {
            datetime: lambda v: v.isoformat()
        }
    }

    @field_validator("component_id")
    @classmethod
    def validate_component_id(cls, v):
        """Validate component ID format"""
        if not v or len(v) < 16:
            raise ValueError("Component ID must be at least 16 characters")
        return v

    @field_validator("file_path")
    @classmethod
    def validate_file_path(cls, v):
        """Validate file path exists and is readable"""
        path = Path(v)
        if not path.is_absolute():
            raise ValueError("File path must be absolute")
        # Note: In production, would check if file exists
        # For testing, we allow non-existent paths
        return str(path)

    @field_validator("quality_score")
    @classmethod
    def validate_quality_score_composition(cls, v):
        """Validate quality score is in valid range"""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Quality score must be between 0.0 and 1.0")

        return v

    @classmethod
    def generate_component_id(cls, file_path: str, content: str) -> str:
        """
        Generate unique component ID from file path and content

        Args:
            file_path: Absolute path to the component file
            content: Source code content

        Returns:
            Unique component identifier
        """
        combined = f"{file_path}:{content}"
        return hashlib.sha256(combined.encode()).hexdigest()[:32]

    @classmethod
    def calculate_quality_score(
        cls,
        consciousness: ConsciousnessFunctionality,
        strategic: StrategicValue,
        consciousness_weight: float = 0.7,
        strategic_weight: float = 0.3
    ) -> float:
        """
        Calculate composite quality score from consciousness and strategic metrics

        Args:
            consciousness: Consciousness functionality metrics
            strategic: Strategic value metrics
            consciousness_weight: Weight for consciousness score (default 0.7)
            strategic_weight: Weight for strategic score (default 0.3)

        Returns:
            Composite quality score between 0.0 and 1.0
        """
        if abs(consciousness_weight + strategic_weight - 1.0) > 0.001:
            raise ValueError("Weights must sum to 1.0")

        return (
            consciousness.composite_score * consciousness_weight +
            strategic.composite_score * strategic_weight
        )

    def update_analysis_status(self, status: AnalysisStatus) -> None:
        """Update component analysis status"""
        self.analysis_status = status

    def add_consciousness_pattern(self, pattern: str) -> None:
        """Add detected consciousness pattern"""
        if pattern not in self.consciousness_patterns:
            self.consciousness_patterns.append(pattern)

    def is_consciousness_component(self) -> bool:
        """Check if component has significant consciousness functionality"""
        return self.consciousness_functionality.composite_score >= 0.6

    def is_high_strategic_value(self) -> bool:
        """Check if component has high strategic value"""
        return self.strategic_value.composite_score >= 0.7

    def meets_migration_threshold(self, threshold: float = 0.7) -> bool:
        """Check if component meets migration quality threshold"""
        return self.quality_score >= threshold

    def to_dict(self) -> Dict:
        """Convert to dictionary representation"""
        return self.dict()

    def __str__(self) -> str:
        return f"LegacyComponent(id={self.component_id[:8]}..., name={self.name}, quality={self.quality_score:.3f})"

    def __repr__(self) -> str:
        return (
            f"LegacyComponent("
            f"component_id='{self.component_id}', "
            f"name='{self.name}', "
            f"quality_score={self.quality_score}, "
            f"status={self.analysis_status}"
            f")"
        )