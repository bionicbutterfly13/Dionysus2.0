#!/usr/bin/env python3
"""
Document Relationship Models - Spec 054 T022

Pydantic models for relationship properties between nodes.

From plan.md lines 299-335.

Author: Spec 054 Implementation
Created: 2025-10-07
"""

from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class InfluenceType(str, Enum):
    """Basin influence types from Context Engineering."""
    REINFORCEMENT = "reinforcement"
    COMPETITION = "competition"
    SYNTHESIS = "synthesis"
    EMERGENCE = "emergence"


class DerivationType(str, Enum):
    """Concept derivation types."""
    COMPOSITION = "composition"
    ABSTRACTION = "abstraction"
    SPECIALIZATION = "specialization"


class ResonanceType(str, Enum):
    """Neural field resonance types."""
    CONSTRUCTIVE = "constructive"
    DESTRUCTIVE = "destructive"
    NEUTRAL = "neutral"


class ExtractedFromRel(BaseModel):
    """
    [:EXTRACTED_FROM] relationship properties.

    Links Concept → Document.
    From plan.md lines 300-305.
    """
    confidence: float = Field(ge=0.0, le=1.0, default=0.90)
    extraction_method: str = Field(default="AutoSchemaKG")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "confidence": 0.90,
                "extraction_method": "AutoSchemaKG",
                "timestamp": "2025-10-07T12:00:00Z"
            }
        }


class AttractedToRel(BaseModel):
    """
    [:ATTRACTED_TO] relationship properties.

    Links AttractorBasin → Document.
    From plan.md lines 307-313.
    """
    activation_strength: float = Field(ge=0.0, le=1.0)
    influence_type: InfluenceType
    strength_delta: float = Field(
        description="Change in basin strength from this document"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "activation_strength": 0.85,
                "influence_type": "reinforcement",
                "strength_delta": 0.10,
                "timestamp": "2025-10-07T12:00:00Z"
            }
        }


class GerminatedFromRel(BaseModel):
    """
    [:GERMINATED_FROM] relationship properties.

    Links ThoughtSeed → Document.
    From plan.md lines 315-320.
    """
    potential: float = Field(ge=0.0, le=1.0)
    generation_stage: str = Field(default="consciousness_processing")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "potential": 0.92,
                "generation_stage": "consciousness_processing",
                "timestamp": "2025-10-07T12:00:00Z"
            }
        }


class DerivedFromRel(BaseModel):
    """
    [:DERIVED_FROM] relationship properties.

    Links Concept → Concept (e.g., composite → atomic).
    From plan.md lines 322-327.
    """
    derivation_type: DerivationType
    confidence: float = Field(ge=0.0, le=1.0, default=0.88)

    class Config:
        json_schema_extra = {
            "example": {
                "derivation_type": "composition",
                "confidence": 0.88
            }
        }


class ResonatesWithRel(BaseModel):
    """
    [:RESONATES_WITH] relationship properties.

    Links Concept ↔ Concept (bidirectional).
    Neural field integration from Context Engineering.
    From plan.md lines 329-335.
    """
    field_score: float = Field(ge=0.0, le=1.0)
    field_energy: float = Field(ge=0.0, le=1.0)
    resonance_type: ResonanceType
    discovered_via: str = Field(default="neural_field_evolution")

    class Config:
        json_schema_extra = {
            "example": {
                "field_score": 0.73,
                "field_energy": 0.85,
                "resonance_type": "constructive",
                "discovered_via": "neural_field_evolution"
            }
        }
