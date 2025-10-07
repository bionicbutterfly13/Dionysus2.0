#!/usr/bin/env python3
"""
Document Node Models - Spec 054 T021

Pydantic models for Document, Concept, and ThoughtSeed nodes.
These are lightweight serialization models for Neo4j persistence.

For actual basin behavior, import from:
    from thoughtseed_active_inference_services.attractor_basin_dynamics import AttractorBasin

Author: Spec 054 Implementation
Created: 2025-10-07
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class TierEnum(str, Enum):
    """Document storage tiers."""
    WARM = "warm"
    COOL = "cool"
    COLD = "cold"


class ProcessingStatus(str, Enum):
    """Document processing status."""
    COMPLETE = "complete"
    PARTIAL = "partial"
    FAILED = "failed"


class ConceptLevel(str, Enum):
    """5-level concept hierarchy from AutoSchemaKG."""
    ATOMIC = "atomic"
    RELATIONSHIP = "relationship"
    COMPOSITE = "composite"
    CONTEXT = "context"
    NARRATIVE = "narrative"


class DocumentNode(BaseModel):
    """
    Document node for Neo4j persistence.

    From plan.md lines 165-200.
    """
    # Core metadata
    document_id: str
    filename: str
    content_hash: str
    upload_timestamp: datetime = Field(default_factory=datetime.utcnow)
    file_size: int = Field(gt=0)
    mime_type: str
    tags: List[str] = Field(default_factory=list)

    # Processing metadata
    processed_at: Optional[datetime] = None
    processing_duration_ms: Optional[int] = None
    processing_status: ProcessingStatus = ProcessingStatus.COMPLETE

    # Quality metrics (from Daedalus)
    quality_overall: float = Field(ge=0.0, le=1.0)
    quality_coherence: Optional[float] = Field(None, ge=0.0, le=1.0)
    quality_novelty: Optional[float] = Field(None, ge=0.0, le=1.0)
    quality_depth: Optional[float] = Field(None, ge=0.0, le=1.0)

    # Research metadata
    curiosity_triggers: int = Field(default=0, ge=0)
    research_questions: int = Field(default=0, ge=0)

    # Tier management
    tier: TierEnum = TierEnum.WARM
    last_accessed: datetime = Field(default_factory=datetime.utcnow)
    access_count: int = Field(default=0, ge=0)
    tier_changed_at: Optional[datetime] = None

    # Cold tier archival
    archive_location: Optional[str] = None
    archived_at: Optional[datetime] = None

    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "doc_1234567890",
                "filename": "research_paper.pdf",
                "content_hash": "sha256:abc123",
                "file_size": 1048576,
                "mime_type": "application/pdf",
                "tags": ["research", "neuroscience"],
                "quality_overall": 0.85,
                "tier": "warm"
            }
        }


class ConceptNode(BaseModel):
    """
    Concept node for Neo4j persistence.

    Supports all 5 levels: atomic, relationship, composite, context, narrative.
    From plan.md lines 203-250.
    """
    concept_id: str
    name: str
    level: ConceptLevel
    salience: float = Field(ge=0.0, le=1.0)

    # Level-specific fields (optional, depends on level)
    definition: Optional[str] = None  # Atomic
    source_concept: Optional[str] = None  # Relationship
    target_concept: Optional[str] = None  # Relationship
    components: List[str] = Field(default_factory=list)  # Composite
    domain: Optional[str] = None  # Context
    era: Optional[str] = None  # Context
    storyline: Optional[str] = None  # Narrative

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "concept_id": "concept_001",
                "name": "active_inference",
                "level": "atomic",
                "salience": 0.95,
                "definition": "A framework for understanding brain function"
            }
        }


class ThoughtSeedNode(BaseModel):
    """
    ThoughtSeed node for Neo4j persistence.

    From plan.md lines 277-295.
    """
    seed_id: str
    content: str
    germination_potential: float = Field(ge=0.0, le=1.0)
    resonance_score: float = Field(ge=0.0, le=1.0)

    # Context Engineering: Neural field resonance
    field_resonance: Optional[Dict[str, Any]] = Field(
        None,
        description="Neural field resonance data: {energy, phase, interference_pattern}"
    )

    # Metadata
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    source_stage: str = Field(default="consciousness_processing")

    class Config:
        json_schema_extra = {
            "example": {
                "seed_id": "seed_001",
                "content": "How does active inference relate to consciousness?",
                "germination_potential": 0.92,
                "resonance_score": 0.85,
                "field_resonance": {
                    "energy": 0.73,
                    "phase": 0.45,
                    "interference_pattern": "constructive"
                }
            }
        }


class AttractorBasinNode(BaseModel):
    """
    AttractorBasin node for Neo4j persistence.

    Lightweight adapter for persistence. For basin behavior, use:
        from thoughtseed_active_inference_services.attractor_basin_dynamics import AttractorBasin

    From plan.md lines 253-274.
    """
    basin_id: str
    name: str
    depth: float = Field(ge=0.0, le=1.0, description="Attraction strength")
    stability: float = Field(ge=0.0, le=1.0, description="Resistance to perturbation")
    strength: float = Field(ge=0.0, description="Overall basin strength")
    associated_concepts: List[str] = Field(default_factory=list)

    # Context Engineering: Influence history (from Redis)
    influence_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Basin evolution events: [{document_id, influence_type, strength_delta, timestamp}]"
    )

    # Evolution tracking
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_modified: Optional[datetime] = None
    modification_count: int = Field(default=0, ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "basin_id": "basin_consciousness_001",
                "name": "consciousness_dynamics",
                "depth": 0.75,
                "stability": 0.88,
                "strength": 1.5,
                "associated_concepts": ["consciousness", "emergence", "integration"]
            }
        }
