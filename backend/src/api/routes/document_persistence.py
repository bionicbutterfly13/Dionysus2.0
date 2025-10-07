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

    # Spec 057: Source metadata fields
    source_type: str = Field(default="uploaded_file")
    original_url: Optional[str] = None
    connector_icon: Optional[str] = None
    download_metadata: Optional[Dict[str, Any]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "doc_123456",
                "filename": "research.pdf",
                "content_hash": "sha256:abc123",
                "file_size": 1048576,
                "mime_type": "application/pdf",
                "tags": ["research", "ai"],
                "source_type": "uploaded_file",
                "original_url": None,
                "connector_icon": "pdf",
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

    Spec 055 Agent 2: Enhanced with structured 409 response for duplicates.

    From plan.md lines 1075-1106.

    Args:
        request: Document metadata + Daedalus final_output

    Returns:
        Persistence result with performance metrics
        409 Conflict with canonical document metadata if duplicate detected
    """
    try:
        # Spec 055 Agent 2: Check for duplicate BEFORE attempting persistence
        duplicate = await document_repo.find_duplicate_by_hash(request.content_hash)

        if duplicate:
            # Log duplicate attempt for analytics
            logger.info(
                f"Duplicate upload attempt blocked: content_hash={request.content_hash}, "
                f"canonical_document={duplicate['document_id']}, "
                f"attempted_filename={request.filename}"
            )

            # Return structured 409 response with reuse guidance
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "status": "duplicate",
                    "message": "Document with this content already exists",
                    "content_hash": request.content_hash,
                    "canonical_document": {
                        "document_id": duplicate["document_id"],
                        "filename": duplicate["filename"],
                        "upload_timestamp": duplicate["upload_timestamp"],
                        "quality_overall": duplicate["quality_overall"],
                        "tier": duplicate["tier"],
                        "tags": duplicate.get("tags", []),
                        "file_size": duplicate.get("file_size", 0),
                        "access_count": duplicate.get("access_count", 0)
                    },
                    "reuse_guidance": {
                        "action": "link_to_existing",
                        "url": f"/api/documents/{duplicate['document_id']}",
                        "message": "Consider linking to the existing document instead of re-uploading"
                    }
                }
            )

        # Prepare metadata (Spec 057: Include source metadata)
        metadata = {
            "document_id": request.document_id,
            "filename": request.filename,
            "content_hash": request.content_hash,
            "file_size": request.file_size,
            "mime_type": request.mime_type,
            "tags": request.tags,
            "source_type": request.source_type,
            "original_url": request.original_url,
            "connector_icon": request.connector_icon,
            "download_metadata": request.download_metadata
        }

        # Persist via repository
        result = await document_repo.persist_document(
            final_output=request.daedalus_output,
            metadata=metadata
        )

        return result

    except HTTPException:
        # Re-raise HTTPExceptions (like the 409 above)
        raise
    except ValueError as e:
        # Validation errors (missing fields, etc.)
        logger.warning(f"Validation error in persist_document: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
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
    tier: Optional[str] = Query(None, pattern="^(warm|cool|cold)$"),
    source_type: Optional[str] = Query(None, pattern="^(uploaded_file|url|api)$")
):
    """
    T042: List documents with pagination, filtering, and sorting.

    From plan.md lines 1108-1147.
    Spec 057: Added source_type filtering.

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
        source_type: Filter by source_type (uploaded_file, url, api)

    Returns:
        Documents list with pagination metadata
    """
    try:
        # Parse tags
        tag_list = tags.split(",") if tags else None

        # Call repository (Spec 057: Pass source_type filter)
        result = await document_repo.list_documents(
            page=page,
            limit=limit,
            tags=tag_list,
            quality_min=quality_min,
            date_from=date_from,
            date_to=date_to,
            sort=sort,
            order=order,
            tier=tier,
            source_type=source_type
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


# Spec 057: GET /api/documents/{id}/external-link
@router.get("/documents/{document_id}/external-link")
async def get_external_link(document_id: str):
    """
    Spec 057: Get external link for 'Open Original' button.

    Returns availability and URL for opening the original document source.

    Args:
        document_id: Document ID

    Returns:
        {
            "available": bool,
            "url": str | None,
            "source_type": str,
            "message": str
        }
    """
    try:
        # Get document to check source metadata
        document = await document_repo.get_document(document_id)

        if document is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )

        metadata = document.get("metadata", {})
        source_type = metadata.get("source_type", "uploaded_file")
        original_url = metadata.get("original_url")

        # Determine availability and construct response
        if source_type == "url" and original_url:
            return {
                "available": True,
                "url": original_url,
                "source_type": source_type,
                "message": "Original document available at URL"
            }
        elif source_type == "uploaded_file":
            return {
                "available": False,
                "url": None,
                "source_type": source_type,
                "message": "Document was uploaded directly (no external source)"
            }
        elif source_type == "api":
            # API ingested documents might have URLs
            if original_url:
                return {
                    "available": True,
                    "url": original_url,
                    "source_type": source_type,
                    "message": "Original document available at URL"
                }
            else:
                return {
                    "available": False,
                    "url": None,
                    "source_type": source_type,
                    "message": "Document was ingested via API (no external source URL)"
                }
        else:
            return {
                "available": False,
                "url": None,
                "source_type": source_type,
                "message": "No external source available"
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"External link retrieval failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"External link retrieval failed: {str(e)}"
        )


# Spec 056: POST /api/documents/ingest-url
class IngestURLRequest(BaseModel):
    """Request model for POST /api/documents/ingest-url"""
    url: str = Field(..., description="HTTPS URL to ingest")
    tags: List[str] = Field(default_factory=list)
    document_id: Optional[str] = Field(None, description="Optional custom document ID")

    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://arxiv.org/pdf/2301.12345.pdf",
                "tags": ["research", "ai"],
                "document_id": None
            }
        }


@router.post("/documents/ingest-url", status_code=status.HTTP_201_CREATED)
async def ingest_url(request: IngestURLRequest):
    """
    Spec 056: Ingest document from HTTPS URL.

    Downloads PDF/HTML from URL, chunks content, and persists to Neo4j.

    Args:
        request: URL and metadata

    Returns:
        Persistence result with chunk count

    Raises:
        400: Invalid URL format
        409: Duplicate content detected
        422: Unsupported MIME type
        500: Download or processing error
    """
    try:
        # Validate URL format
        from urllib.parse import urlparse
        parsed = urlparse(request.url)
        if parsed.scheme != "https":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only HTTPS URLs are supported"
            )

        if not parsed.netloc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid URL format"
            )

        # Prepare metadata
        metadata = {
            "tags": request.tags,
            "namespace": "default"
        }

        if request.document_id:
            metadata["document_id"] = request.document_id

        # Ingest via repository
        result = await document_repo.persist_document_from_url(
            url=request.url,
            metadata=metadata
        )

        return result

    except ValueError as e:
        # Duplicate detection
        if "duplicate" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(e)
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(e)
            )
    except Exception as e:
        from src.services.url_downloader import (
            DownloadError,
            UnsupportedMimeTypeError,
            NetworkError
        )

        if isinstance(e, UnsupportedMimeTypeError):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(e)
            )
        elif isinstance(e, (DownloadError, NetworkError)):
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Failed to download URL: {str(e)}"
            )
        else:
            logger.error(f"URL ingestion failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"URL ingestion failed: {str(e)}"
            )


# Spec 056: GET /api/documents/{id}/chunks
@router.get("/documents/{document_id}/chunks")
async def get_document_chunks(document_id: str):
    """
    Spec 056: Get all chunks for a document.

    Returns chunks in position order for citation highlighting.

    Args:
        document_id: Document ID

    Returns:
        List of chunks with content and position metadata
    """
    try:
        chunks = await document_repo.get_document_chunks(document_id)

        return {
            "document_id": document_id,
            "chunk_count": len(chunks),
            "chunks": chunks
        }

    except Exception as e:
        logger.error(f"Chunk retrieval failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chunk retrieval failed: {str(e)}"
        )
