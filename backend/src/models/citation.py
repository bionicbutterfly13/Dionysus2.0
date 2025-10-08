#!/usr/bin/env python3
"""
Citation Models - Spec 058

Response models for document citation endpoint.
Returns chunk, basin, and thoughtseed data for trust interaction.

Author: Agent 058-Backend
Created: 2025-10-08
"""

from typing import Optional, Dict, List
from pydantic import BaseModel, Field
from datetime import datetime


class ChunkCitationData(BaseModel):
    """Chunk data for citation display"""

    chunk_id: str = Field(..., description="Unique chunk identifier")
    chunk_text: str = Field(..., description="Text content of the chunk")
    chunk_index: int = Field(..., ge=0, description="Sequential position in document")
    start_offset: int = Field(..., ge=0, description="Character offset where chunk starts")
    end_offset: int = Field(..., gt=0, description="Character offset where chunk ends")
    created_at: datetime = Field(..., description="Timestamp when chunk was created")

    class Config:
        json_schema_extra = {
            "example": {
                "chunk_id": "chunk_doc123_0",
                "chunk_text": "This is example chunk text from the document.",
                "chunk_index": 0,
                "start_offset": 0,
                "end_offset": 46,
                "created_at": "2025-10-08T12:00:00Z"
            }
        }


class BasinCitationData(BaseModel):
    """Basin metadata for citation display"""

    basin_id: str = Field(..., description="Unique basin identifier")
    basin_name: str = Field(..., description="Human-readable basin name")
    stability: float = Field(..., ge=0.0, le=1.0, description="Basin stability score")
    attractor_strength: float = Field(..., ge=0.0, le=1.0, description="Attractor strength")
    layer_influences: Dict[str, float] = Field(
        ...,
        description="Layer-wise influence distribution (should sum to ~1.0)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "basin_id": "basin_456",
                "basin_name": "Cognitive Processing",
                "stability": 0.85,
                "attractor_strength": 0.72,
                "layer_influences": {
                    "layer1": 0.3,
                    "layer2": 0.5,
                    "layer3": 0.2
                }
            }
        }


class ThoughtseedCitationData(BaseModel):
    """Thoughtseed data for citation display"""

    thoughtseed_id: str = Field(..., description="Unique thoughtseed identifier")
    resonance_score: float = Field(..., ge=0.0, le=1.0, description="Resonance strength")
    concept_labels: List[str] = Field(..., min_length=1, description="Associated concept labels")
    emergence_timestamp: datetime = Field(..., description="When thoughtseed emerged")

    class Config:
        json_schema_extra = {
            "example": {
                "thoughtseed_id": "ts_789",
                "resonance_score": 0.91,
                "concept_labels": ["machine learning", "neural networks", "cognition"],
                "emergence_timestamp": "2025-10-08T12:00:00Z"
            }
        }


class CitationResponse(BaseModel):
    """
    Complete citation response payload.

    Includes chunk text, basin metadata, and thoughtseed data.
    Basin and thoughtseed may be null if not associated with chunk.
    """

    document_id: str = Field(..., description="Source document identifier")
    chunk: ChunkCitationData = Field(..., description="Chunk data")
    basin: Optional[BasinCitationData] = Field(None, description="Basin metadata (may be null)")
    thoughtseed: Optional[ThoughtseedCitationData] = Field(None, description="Thoughtseed data (may be null)")

    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "doc_123",
                "chunk": {
                    "chunk_id": "chunk_doc123_0",
                    "chunk_text": "Example text content.",
                    "chunk_index": 0,
                    "start_offset": 0,
                    "end_offset": 21,
                    "created_at": "2025-10-08T12:00:00Z"
                },
                "basin": {
                    "basin_id": "basin_456",
                    "basin_name": "Cognitive Processing",
                    "stability": 0.85,
                    "attractor_strength": 0.72,
                    "layer_influences": {
                        "layer1": 0.3,
                        "layer2": 0.5,
                        "layer3": 0.2
                    }
                },
                "thoughtseed": {
                    "thoughtseed_id": "ts_789",
                    "resonance_score": 0.91,
                    "concept_labels": ["machine learning", "cognition"],
                    "emergence_timestamp": "2025-10-08T12:00:00Z"
                }
            }
        }
