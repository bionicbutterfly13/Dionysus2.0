#!/usr/bin/env python3
"""
Document Citations API Routes - Spec 058

FastAPI endpoint for retrieving citation data (chunk + basin + thoughtseed).

CONSTITUTIONAL COMPLIANCE (Spec 040):
- All Neo4j access via CitationService → DaedalusGraphChannel
- NO direct neo4j imports

Author: Agent 058-Backend
Created: 2025-10-08
"""

from fastapi import APIRouter, HTTPException, status, Query
from typing import Optional
import logging

# Service imports
from ...services.citation_service import get_citation_service
from ...models.citation import CitationResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["citations"])

# Initialize service
citation_service = get_citation_service()


@router.get(
    "/documents/{document_id}/citations",
    response_model=CitationResponse,
    status_code=status.HTTP_200_OK,
    summary="Get citation data for a chunk",
    description="""
    Retrieve citation data for a specific chunk including:
    - Chunk text and metadata
    - Basin metadata (stability, attractor strength, layer influences)
    - ThoughtSeed data (resonance score, concepts, emergence time)

    Basin and ThoughtSeed may be null if not associated with the chunk.
    """,
    responses={
        200: {
            "description": "Citation data retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "document_id": "doc_123",
                        "chunk": {
                            "chunk_id": "chunk_doc123_0",
                            "chunk_text": "Example chunk text.",
                            "chunk_index": 0,
                            "start_offset": 0,
                            "end_offset": 20,
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
            }
        },
        400: {
            "description": "Chunk not found or doesn't belong to document",
            "content": {
                "application/json": {
                    "example": {"detail": "Chunk not found or doesn't belong to document: chunk_invalid"}
                }
            }
        },
        404: {
            "description": "Document not found",
            "content": {
                "application/json": {
                    "example": {"detail": "Document not found: doc_invalid"}
                }
            }
        },
        422: {
            "description": "Missing or invalid query parameters",
            "content": {
                "application/json": {
                    "example": {
                        "detail": [
                            {
                                "loc": ["query", "chunk_id"],
                                "msg": "field required",
                                "type": "value_error.missing"
                            }
                        ]
                    }
                }
            }
        }
    }
)
async def get_document_citations(
    document_id: str,
    chunk_id: str = Query(..., description="Chunk identifier to retrieve citation for")
) -> CitationResponse:
    """
    Get citation data for a specific chunk.

    Args:
        document_id: Document identifier
        chunk_id: Chunk identifier (required query parameter)

    Returns:
        CitationResponse with chunk, basin, and thoughtseed data

    Raises:
        HTTPException 404: Document not found
        HTTPException 400: Chunk not found or invalid
        HTTPException 500: Server error
    """
    logger.info(f"GET /api/documents/{document_id}/citations?chunk_id={chunk_id}")

    try:
        # Retrieve citation data from service
        citation_data = await citation_service.get_citation_data(
            document_id=document_id,
            chunk_id=chunk_id
        )

        logger.info(f"Successfully retrieved citation data for chunk {chunk_id}")
        return citation_data

    except ValueError as e:
        error_msg = str(e)

        # Determine error type based on message
        if "Document not found" in error_msg:
            logger.warning(f"Document not found: {document_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=error_msg
            )
        elif "Chunk not found" in error_msg or "doesn't belong to document" in error_msg:
            logger.warning(f"Chunk not found or invalid: {chunk_id}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_msg
            )
        else:
            # Generic error
            logger.error(f"Error retrieving citation data: {error_msg}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve citation data: {error_msg}"
            )

    except Exception as e:
        logger.error(f"Unexpected error in get_document_citations: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error while retrieving citation data"
        )
