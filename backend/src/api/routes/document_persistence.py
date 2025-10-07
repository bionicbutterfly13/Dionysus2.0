#!/usr/bin/env python3
"""
Document Persistence API Routes - Spec 054 T041-T044

FastAPI endpoints for document persistence via DocumentRepository.

CONSTITUTIONAL COMPLIANCE (Spec 040):
- All Neo4j access via DocumentRepository → DaedalusGraphChannel
- NO direct neo4j imports

Author: Spec 054 Implementation
Created: 2025-10-07
"""

from fastapi import APIRouter, HTTPException, status, Query
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import logging

# Repository imports
from ...services.document_repository import DocumentRepository
from ...services.tier_manager import TierManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["document_persistence"])

# Initialize services
document_repo = DocumentRepository()
tier_manager = TierManager()


# Request/Response models
class PersistDocumentRequest(BaseModel):
    """Request model for POST /api/documents/persist"""
    document_id: str
    filename: str
    content_hash: str
    file_size: int
    mime_type: str = "application/pdf"
    tags: List[str] = Field(default_factory=list)
    daedalus_output: Dict[str, Any]

    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "doc_123456",
                "filename": "research.pdf",
                "content_hash": "sha256:abc123",
                "file_size": 1048576,
                "mime_type": "application/pdf",
                "tags": ["research", "ai"],
                "daedalus_output": {
                    "quality": {"scores": {"overall": 0.85}},
                    "concepts": {"atomic": []},
                    "basins": [],
                    "thoughtseeds": [],
                    "research": {"curiosity_triggers": 5}
                }
            }
        }


class UpdateTierRequest(BaseModel):
    """Request model for PUT /api/documents/{id}/tier"""
    new_tier: str = Field(..., pattern="^(warm|cool|cold)$")
    reason: Optional[str] = "manual"

    class Config:
        json_schema_extra = {
            "example": {
                "new_tier": "cool",
                "reason": "manual_archival"
            }
        }


# T041: POST /api/documents/persist
@router.post("/documents/persist", status_code=status.HTTP_201_CREATED)
async def persist_document(request: PersistDocumentRequest):
    """
    T041: Persist document with all processing artifacts to Neo4j.

    From plan.md lines 1075-1106.

    Args:
        request: Document metadata + Daedalus final_output

    Returns:
        Persistence result with performance metrics
    """
    try:
        # Prepare metadata
        metadata = {
            "document_id": request.document_id,
            "filename": request.filename,
            "content_hash": request.content_hash,
            "file_size": request.file_size,
            "mime_type": request.mime_type,
            "tags": request.tags
        }

        # Persist via repository
        result = await document_repo.persist_document(
            final_output=request.daedalus_output,
            metadata=metadata
        )

        return result

    except ValueError as e:
        # Duplicate document or validation error
        if "Duplicate document" in str(e):
            # Extract existing document_id from error message
            existing_doc_id = None
            if "exists as document" in str(e):
                existing_doc_id = str(e).split("exists as document ")[-1]

            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "status": "duplicate",
                    "message": str(e),
                    "content_hash": request.content_hash,
                    "existing_document": existing_doc_id,
                    "options": ["reprocess", "skip"]
                }
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
    except Exception as e:
        logger.error(f"Document persistence failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Persistence failed: {str(e)}"
        )


# T042: GET /api/documents
@router.get("/documents")
async def list_documents(
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=100),
    tags: Optional[str] = Query(None),
    quality_min: Optional[float] = Query(None, ge=0.0, le=1.0),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    sort: str = Query("upload_date", pattern="^(upload_date|quality|curiosity)$"),
    order: str = Query("desc", pattern="^(asc|desc)$"),
    tier: Optional[str] = Query(None, pattern="^(warm|cool|cold)$")
):
    """
    T042: List documents with pagination, filtering, and sorting.

    From plan.md lines 1108-1147.

    Query Parameters:
        page: Page number (1-indexed)
        limit: Items per page (max 100)
        tags: Comma-separated tags to filter by
        quality_min: Minimum quality score (0.0-1.0)
        date_from: Start date (ISO 8601)
        date_to: End date (ISO 8601)
        sort: Sort field (upload_date, quality, curiosity)
        order: Sort order (asc, desc)
        tier: Filter by tier (warm, cool, cold)

    Returns:
        Documents list with pagination metadata
    """
    try:
        # Parse tags
        tag_list = tags.split(",") if tags else None

        # Call repository
        result = await document_repo.list_documents(
            page=page,
            limit=limit,
            tags=tag_list,
            quality_min=quality_min,
            date_from=date_from,
            date_to=date_to,
            sort=sort,
            order=order,
            tier=tier
        )

        return result

    except Exception as e:
        logger.error(f"Document listing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Listing failed: {str(e)}"
        )


# T043: GET /api/documents/{id}
@router.get("/documents/{document_id}")
async def get_document_detail(document_id: str):
    """
    T043: Get full document detail with all artifacts.

    From plan.md lines 1149-1180.

    Args:
        document_id: Document ID

    Returns:
        Complete document data with concepts, basins, thoughtseeds
    """
    try:
        result = await document_repo.get_document(document_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document retrieval failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval failed: {str(e)}"
        )


# T044: PUT /api/documents/{id}/tier
@router.put("/documents/{document_id}/tier")
async def update_document_tier(document_id: str, request: UpdateTierRequest):
    """
    T044: Update document tier (warm/cool/cold).

    From plan.md lines 1182-1205.

    Args:
        document_id: Document ID
        request: New tier and reason

    Returns:
        Tier update result with archive_location if moved to cold
    """
    try:
        result = await tier_manager.update_tier(
            document_id=document_id,
            new_tier=request.new_tier,
            reason=request.reason or "manual"
        )

        return result

    except ValueError as e:
        if "not found" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(e)
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
    except Exception as e:
        logger.error(f"Tier update failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Tier update failed: {str(e)}"
        )
