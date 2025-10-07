#!/usr/bin/env python3
"""
Contract Tests for Source Metadata API - Spec 057

Tests API includes source metadata, filtering by source_type,
and external link endpoint.

Author: Spec 057 Implementation
Created: 2025-10-07
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

# Setup test client
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.api.routes.document_persistence import router
from fastapi import FastAPI

app = FastAPI()
app.include_router(router)
client = TestClient(app)


# ============================================================================
# Test POST /api/documents/persist with Source Metadata
# ============================================================================

@pytest.mark.asyncio
async def test_persist_document_with_url_source():
    """Test persisting document with url source_type."""
    with patch('src.api.routes.document_persistence.document_repo') as mock_repo:
        # Mock repository methods
        mock_repo.find_duplicate_by_hash = AsyncMock(return_value=None)
        mock_repo.persist_document = AsyncMock(return_value={
            "status": "success",
            "document_id": "doc_url_001",
            "persisted_at": "2025-10-07T10:00:00",
            "tier": "warm"
        })

        request_data = {
            "document_id": "doc_url_001",
            "filename": "arxiv_paper.pdf",
            "content_hash": "a" * 64,
            "file_size": 1536000,
            "mime_type": "application/pdf",
            "tags": ["research", "ai"],
            "source_type": "url",
            "original_url": "https://arxiv.org/pdf/2024.12345.pdf",
            "connector_icon": "pdf",
            "daedalus_output": {
                "quality": {"scores": {"overall": 0.85}},
                "concepts": {"atomic": []},
                "basins": [],
                "thoughtseeds": [],
                "research": {"curiosity_triggers": 5}
            }
        }

        response = client.post("/api/documents/persist", json=request_data)

        assert response.status_code == 201
        assert response.json()["status"] == "success"

        # Verify metadata was passed to repository
        call_args = mock_repo.persist_document.call_args
        metadata = call_args.kwargs["metadata"]

        assert metadata["source_type"] == "url"
        assert metadata["original_url"] == "https://arxiv.org/pdf/2024.12345.pdf"
        assert metadata["connector_icon"] == "pdf"


@pytest.mark.asyncio
async def test_persist_document_with_uploaded_file():
    """Test persisting document with uploaded_file source_type."""
    with patch('src.api.routes.document_persistence.document_repo') as mock_repo:
        # Mock repository methods
        mock_repo.find_duplicate_by_hash = AsyncMock(return_value=None)
        mock_repo.persist_document = AsyncMock(return_value={
            "status": "success",
            "document_id": "doc_upload_001",
            "persisted_at": "2025-10-07T10:00:00",
            "tier": "warm"
        })

        request_data = {
            "document_id": "doc_upload_001",
            "filename": "uploaded.pdf",
            "content_hash": "b" * 64,
            "file_size": 2048000,
            "mime_type": "application/pdf",
            "tags": ["personal"],
            "source_type": "uploaded_file",
            "original_url": None,
            "connector_icon": "upload",
            "daedalus_output": {
                "quality": {"scores": {"overall": 0.90}},
                "concepts": {"atomic": []},
                "basins": [],
                "thoughtseeds": [],
                "research": {"curiosity_triggers": 3}
            }
        }

        response = client.post("/api/documents/persist", json=request_data)

        assert response.status_code == 201

        # Verify metadata
        call_args = mock_repo.persist_document.call_args
        metadata = call_args.kwargs["metadata"]

        assert metadata["source_type"] == "uploaded_file"
        assert metadata["original_url"] is None


# ============================================================================
# Test GET /api/documents - Source Metadata in Response
# ============================================================================

@pytest.mark.asyncio
async def test_list_documents_includes_source_metadata():
    """Test that list_documents includes source metadata in response."""
    with patch('src.api.routes.document_persistence.document_repo') as mock_repo:
        mock_repo.list_documents = AsyncMock(return_value={
            "documents": [
                {
                    "document_id": "doc_001",
                    "filename": "test1.pdf",
                    "source_type": "uploaded_file",
                    "original_url": None,
                    "connector_icon": "pdf",
                    "quality_overall": 0.85,
                    "tags": ["test"],
                    "tier": "warm"
                },
                {
                    "document_id": "doc_002",
                    "filename": "test2.pdf",
                    "source_type": "url",
                    "original_url": "https://example.com/paper.pdf",
                    "connector_icon": "web",
                    "quality_overall": 0.90,
                    "tags": ["research"],
                    "tier": "warm"
                }
            ],
            "pagination": {
                "page": 1,
                "limit": 50,
                "total": 2,
                "total_pages": 1
            }
        })

        response = client.get("/api/documents")

        assert response.status_code == 200
        data = response.json()

        # Check first document (uploaded)
        doc1 = data["documents"][0]
        assert doc1["source_type"] == "uploaded_file"
        assert doc1["original_url"] is None
        assert doc1["connector_icon"] == "pdf"

        # Check second document (url)
        doc2 = data["documents"][1]
        assert doc2["source_type"] == "url"
        assert doc2["original_url"] == "https://example.com/paper.pdf"
        assert doc2["connector_icon"] == "web"


# ============================================================================
# Test GET /api/documents - Filter by source_type
# ============================================================================

@pytest.mark.asyncio
async def test_list_documents_filter_by_source_type_uploaded():
    """Test filtering documents by source_type=uploaded_file."""
    with patch('src.api.routes.document_persistence.document_repo') as mock_repo:
        mock_repo.list_documents = AsyncMock(return_value={
            "documents": [
                {
                    "document_id": "doc_upload_001",
                    "filename": "upload1.pdf",
                    "source_type": "uploaded_file",
                    "original_url": None,
                    "connector_icon": "upload"
                }
            ],
            "pagination": {"page": 1, "limit": 50, "total": 1, "total_pages": 1}
        })

        response = client.get("/api/documents?source_type=uploaded_file")

        assert response.status_code == 200

        # Verify repository was called with source_type filter
        call_args = mock_repo.list_documents.call_args
        assert call_args.kwargs["source_type"] == "uploaded_file"


@pytest.mark.asyncio
async def test_list_documents_filter_by_source_type_url():
    """Test filtering documents by source_type=url."""
    with patch('src.api.routes.document_persistence.document_repo') as mock_repo:
        mock_repo.list_documents = AsyncMock(return_value={
            "documents": [
                {
                    "document_id": "doc_url_001",
                    "filename": "url1.pdf",
                    "source_type": "url",
                    "original_url": "https://example.com/doc.pdf",
                    "connector_icon": "web"
                }
            ],
            "pagination": {"page": 1, "limit": 50, "total": 1, "total_pages": 1}
        })

        response = client.get("/api/documents?source_type=url")

        assert response.status_code == 200

        # Verify filter
        call_args = mock_repo.list_documents.call_args
        assert call_args.kwargs["source_type"] == "url"


# ============================================================================
# Test GET /api/documents/{id} - Source Metadata in Detail
# ============================================================================

@pytest.mark.asyncio
async def test_get_document_detail_includes_source_metadata():
    """Test that document detail includes source metadata."""
    with patch('src.api.routes.document_persistence.document_repo') as mock_repo:
        mock_repo.get_document = AsyncMock(return_value={
            "document_id": "doc_001",
            "metadata": {
                "filename": "test.pdf",
                "source_type": "url",
                "original_url": "https://arxiv.org/pdf/2024.12345.pdf",
                "connector_icon": "pdf",
                "download_metadata": {
                    "status_code": 200,
                    "content_length": 1024000
                },
                "upload_timestamp": "2025-10-07T10:00:00",
                "file_size": 1024000,
                "mime_type": "application/pdf",
                "tags": ["research"],
                "tier": "warm",
                "last_accessed": "2025-10-07T11:00:00",
                "access_count": 5
            },
            "quality": {"overall": 0.85},
            "concepts": {},
            "basins": [],
            "thoughtseeds": []
        })

        response = client.get("/api/documents/doc_001")

        assert response.status_code == 200
        data = response.json()

        metadata = data["metadata"]
        assert metadata["source_type"] == "url"
        assert metadata["original_url"] == "https://arxiv.org/pdf/2024.12345.pdf"
        assert metadata["connector_icon"] == "pdf"
        assert metadata["download_metadata"] is not None
        assert metadata["download_metadata"]["status_code"] == 200


# ============================================================================
# Test GET /api/documents/{id}/external-link
# ============================================================================

@pytest.mark.asyncio
async def test_external_link_url_source_available():
    """Test external link endpoint for URL source (available)."""
    with patch('src.api.routes.document_persistence.document_repo') as mock_repo:
        mock_repo.get_document = AsyncMock(return_value={
            "document_id": "doc_url_001",
            "metadata": {
                "filename": "arxiv.pdf",
                "source_type": "url",
                "original_url": "https://arxiv.org/pdf/2024.12345.pdf",
                "connector_icon": "pdf"
            }
        })

        response = client.get("/api/documents/doc_url_001/external-link")

        assert response.status_code == 200
        data = response.json()

        assert data["available"] is True
        assert data["url"] == "https://arxiv.org/pdf/2024.12345.pdf"
        assert data["source_type"] == "url"
        assert "Original document available" in data["message"]


@pytest.mark.asyncio
async def test_external_link_uploaded_file_not_available():
    """Test external link endpoint for uploaded file (not available)."""
    with patch('src.api.routes.document_persistence.document_repo') as mock_repo:
        mock_repo.get_document = AsyncMock(return_value={
            "document_id": "doc_upload_001",
            "metadata": {
                "filename": "uploaded.pdf",
                "source_type": "uploaded_file",
                "original_url": None,
                "connector_icon": "upload"
            }
        })

        response = client.get("/api/documents/doc_upload_001/external-link")

        assert response.status_code == 200
        data = response.json()

        assert data["available"] is False
        assert data["url"] is None
        assert data["source_type"] == "uploaded_file"
        assert "uploaded directly" in data["message"]


@pytest.mark.asyncio
async def test_external_link_api_source_with_url():
    """Test external link endpoint for API source with URL (available)."""
    with patch('src.api.routes.document_persistence.document_repo') as mock_repo:
        mock_repo.get_document = AsyncMock(return_value={
            "document_id": "doc_api_001",
            "metadata": {
                "filename": "api_doc.pdf",
                "source_type": "api",
                "original_url": "https://api.example.com/documents/123",
                "connector_icon": "api"
            }
        })

        response = client.get("/api/documents/doc_api_001/external-link")

        assert response.status_code == 200
        data = response.json()

        assert data["available"] is True
        assert data["url"] == "https://api.example.com/documents/123"
        assert data["source_type"] == "api"


@pytest.mark.asyncio
async def test_external_link_api_source_without_url():
    """Test external link endpoint for API source without URL (not available)."""
    with patch('src.api.routes.document_persistence.document_repo') as mock_repo:
        mock_repo.get_document = AsyncMock(return_value={
            "document_id": "doc_api_002",
            "metadata": {
                "filename": "api_generated.pdf",
                "source_type": "api",
                "original_url": None,
                "connector_icon": "api"
            }
        })

        response = client.get("/api/documents/doc_api_002/external-link")

        assert response.status_code == 200
        data = response.json()

        assert data["available"] is False
        assert data["url"] is None
        assert data["source_type"] == "api"
        assert "no external source URL" in data["message"]


@pytest.mark.asyncio
async def test_external_link_document_not_found():
    """Test external link endpoint for non-existent document."""
    with patch('src.api.routes.document_persistence.document_repo') as mock_repo:
        mock_repo.get_document = AsyncMock(return_value=None)

        response = client.get("/api/documents/doc_nonexistent/external-link")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
