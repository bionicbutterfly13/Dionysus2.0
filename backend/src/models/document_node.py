#!/usr/bin/env python3
"""
Document Node Models - Spec 054 T021 + Spec 055 Agent 1

Pydantic models for Document, Concept, and ThoughtSeed nodes.
These are lightweight serialization models for Neo4j persistence.

SPEC 055 AGENT 1 ENHANCEMENTS:
- content_hash field is required for DocumentNode
- SHA-256 format validation (64 hex characters)

For actual basin behavior, import from:
    from thoughtseed_active_inference_services.attractor_basin_dynamics import AttractorBasin

Author: Spec 054 + Spec 055 Agent 1 Implementation
Created: 2025-10-07
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from enum import Enum
import re


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

    SPEC 055 AGENT 1: content_hash is REQUIRED and validated as SHA-256 format.
    """
    # Core metadata
    document_id: str
    filename: str
    content_hash: str = Field(
        ...,
        description="SHA-256 content hash (64 hex characters). Required for deduplication."
    )
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

    # Spec 055 Agent 3: LLM Summary fields
    summary: Optional[str] = Field(
        None,
        description="Token-budgeted LLM summary of document (max 150 tokens)"
    )
    summary_metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Summary generation metadata: {method, model, tokens_used, generated_at, error}"
    )

    # Tier management
    tier: TierEnum = TierEnum.WARM
    last_accessed: datetime = Field(default_factory=datetime.utcnow)
    access_count: int = Field(default=0, ge=0)
    tier_changed_at: Optional[datetime] = None

    # Cold tier archival
    archive_location: Optional[str] = None
    archived_at: Optional[datetime] = None

    # Spec 057: Source metadata fields
    source_type: str = Field(
        default="uploaded_file",
        description="How document was ingested: uploaded_file, url, api"
    )
    original_url: Optional[str] = Field(
        None,
        description="Original URL if document came from web"
    )
    connector_icon: Optional[str] = Field(
        None,
        description="Icon hint for UI (pdf, html, upload)"
    )
    download_metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Metadata from download: status_code, redirects, etc."
    )

    @field_validator('source_type')
    @classmethod
    def validate_source_type(cls, v: str) -> str:
        """
        Validate source_type is one of allowed values.

        SPEC 057: Must be uploaded_file, url, or api.

        Args:
            v: source_type value

        Returns:
            Validated source_type

        Raises:
            ValueError: If source_type is invalid
        """
        allowed_types = ["uploaded_file", "url", "api"]
        if v not in allowed_types:
            raise ValueError(
                f"source_type must be one of {allowed_types}. Got: {v}"
            )
        return v

    @field_validator('original_url')
    @classmethod
    def validate_original_url(cls, v: Optional[str]) -> Optional[str]:
        """
        Validate original_url is valid HTTP(S) URL if provided.

        SPEC 057: Must be valid HTTP(S) URL.

        Args:
            v: original_url value

        Returns:
            Validated URL

        Raises:
            ValueError: If URL format is invalid
        """
        if v is None:
            return None

        # Check if URL starts with http:// or https://
        if not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError(
                f"original_url must start with http:// or https://. Got: {v}"
            )

        # Basic URL validation (no spaces, reasonable length)
        if " " in v:
            raise ValueError(
                f"original_url must not contain spaces. Got: {v}"
            )

        if len(v) > 2048:
            raise ValueError(
                f"original_url must not exceed 2048 characters. Got {len(v)} characters."
            )

        return v

    @field_validator('content_hash')
    @classmethod
    def validate_content_hash_format(cls, v: str) -> str:
        """
        Validate content_hash is valid SHA-256 format.

        SPEC 055 AGENT 1: Must be exactly 64 hexadecimal characters.

        Args:
            v: content_hash value

        Returns:
            Normalized lowercase hash

        Raises:
            ValueError: If hash format is invalid
        """
        if not isinstance(v, str):
            raise ValueError("content_hash must be a string")

        # Normalize to lowercase
        normalized = v.lower()

        # Check length
        if len(normalized) != 64:
            raise ValueError(
                f"content_hash must be exactly 64 characters (SHA-256). "
                f"Got {len(normalized)} characters."
            )

        # Check hex format
        hex_pattern = re.compile(r'^[0-9a-f]{64}$')
        if not hex_pattern.match(normalized):
            raise ValueError(
                "content_hash must contain only hexadecimal characters (0-9, a-f). "
                f"Invalid hash: {v[:20]}..."
            )

        return normalized

    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "doc_1234567890",
                "filename": "research_paper.pdf",
                "content_hash": "a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a",
                "file_size": 1048576,
                "mime_type": "application/pdf",
                "tags": ["research", "neuroscience"],
                "quality_overall": 0.85,
                "tier": "warm",
                "source_type": "uploaded_file",
                "original_url": None,
                "connector_icon": "pdf"
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
