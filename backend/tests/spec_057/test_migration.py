#!/usr/bin/env python3
"""
Test Suite for Source Metadata Migration - Spec 057

Tests migration on existing documents and idempotency.

Author: Spec 057 Implementation
Created: 2025-10-07
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

# Import migration functions
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from scripts.migrate_source_metadata import (
    infer_connector_icon_from_mime,
    check_documents_needing_migration,
    migrate_document,
    migrate_existing_documents,
    verify_migration
)


# ============================================================================
# Test Helper Functions
# ============================================================================

def test_infer_connector_icon_from_mime_pdf():
    """Test icon inference for PDF."""
    icon = infer_connector_icon_from_mime("application/pdf")
    assert icon == "pdf"


def test_infer_connector_icon_from_mime_html():
    """Test icon inference for HTML."""
    icon = infer_connector_icon_from_mime("text/html")
    assert icon == "html"


def test_infer_connector_icon_from_mime_unknown():
    """Test icon inference for unknown MIME type."""
    icon = infer_connector_icon_from_mime("application/unknown")
    assert icon == "upload"


# ============================================================================
# Test Check Documents Needing Migration
# ============================================================================

@pytest.mark.asyncio
async def test_check_documents_needing_migration_found():
    """Test finding documents that need migration."""
    # Mock graph channel
    mock_channel = AsyncMock()
    mock_channel.execute_read = AsyncMock(return_value={
        "records": [
            {
                "document_id": "doc_001",
                "mime_type": "application/pdf",
                "filename": "test1.pdf"
            },
            {
                "document_id": "doc_002",
                "mime_type": "text/html",
                "filename": "test2.html"
            }
        ]
    })

    documents = await check_documents_needing_migration(mock_channel)

    assert len(documents) == 2
    assert documents[0]["document_id"] == "doc_001"
    assert documents[0]["mime_type"] == "application/pdf"
    assert documents[1]["document_id"] == "doc_002"
    assert documents[1]["mime_type"] == "text/html"


@pytest.mark.asyncio
async def test_check_documents_needing_migration_none():
    """Test when no documents need migration."""
    # Mock graph channel
    mock_channel = AsyncMock()
    mock_channel.execute_read = AsyncMock(return_value={
        "records": []
    })

    documents = await check_documents_needing_migration(mock_channel)

    assert len(documents) == 0


# ============================================================================
# Test Migrate Single Document
# ============================================================================

@pytest.mark.asyncio
async def test_migrate_document_success():
    """Test successful migration of a single document."""
    # Mock graph channel
    mock_channel = AsyncMock()
    mock_channel.execute_write = AsyncMock(return_value={
        "records": [{"document_id": "doc_001"}]
    })

    document = {
        "document_id": "doc_001",
        "mime_type": "application/pdf",
        "filename": "test.pdf"
    }

    success = await migrate_document(mock_channel, document)

    assert success is True
    mock_channel.execute_write.assert_called_once()

    # Check that correct parameters were passed
    call_args = mock_channel.execute_write.call_args
    params = call_args.kwargs["parameters"]

    assert params["document_id"] == "doc_001"
    assert params["source_type"] == "uploaded_file"
    assert params["original_url"] is None
    assert params["connector_icon"] == "pdf"
    assert params["download_metadata"] is None


@pytest.mark.asyncio
async def test_migrate_document_html():
    """Test migration of HTML document."""
    # Mock graph channel
    mock_channel = AsyncMock()
    mock_channel.execute_write = AsyncMock(return_value={
        "records": [{"document_id": "doc_002"}]
    })

    document = {
        "document_id": "doc_002",
        "mime_type": "text/html",
        "filename": "page.html"
    }

    success = await migrate_document(mock_channel, document)

    assert success is True

    # Check connector_icon was inferred correctly
    call_args = mock_channel.execute_write.call_args
    params = call_args.kwargs["parameters"]
    assert params["connector_icon"] == "html"


@pytest.mark.asyncio
async def test_migrate_document_failure():
    """Test migration failure handling."""
    # Mock graph channel to return empty result
    mock_channel = AsyncMock()
    mock_channel.execute_write = AsyncMock(return_value={
        "records": []
    })

    document = {
        "document_id": "doc_failed",
        "mime_type": "application/pdf",
        "filename": "failed.pdf"
    }

    success = await migrate_document(mock_channel, document)

    assert success is False


@pytest.mark.asyncio
async def test_migrate_document_exception():
    """Test migration exception handling."""
    # Mock graph channel to raise exception
    mock_channel = AsyncMock()
    mock_channel.execute_write = AsyncMock(side_effect=Exception("Database error"))

    document = {
        "document_id": "doc_error",
        "mime_type": "application/pdf",
        "filename": "error.pdf"
    }

    success = await migrate_document(mock_channel, document)

    assert success is False


# ============================================================================
# Test Full Migration Process
# ============================================================================

@pytest.mark.asyncio
async def test_migrate_existing_documents_no_documents():
    """Test migration when no documents need migration."""
    with patch('scripts.migrate_source_metadata.get_graph_channel') as mock_get_channel:
        # Mock graph channel
        mock_channel = AsyncMock()
        mock_channel.execute_read = AsyncMock(return_value={"records": []})
        mock_get_channel.return_value = mock_channel

        result = await migrate_existing_documents()

        assert result["status"] == "success"
        assert result["documents_checked"] == 0
        assert result["documents_migrated"] == 0
        assert result["documents_failed"] == 0


@pytest.mark.asyncio
async def test_migrate_existing_documents_success():
    """Test successful migration of multiple documents."""
    with patch('scripts.migrate_source_metadata.get_graph_channel') as mock_get_channel:
        # Mock graph channel
        mock_channel = AsyncMock()

        # First call: check for documents needing migration
        # Subsequent calls: migrate documents
        mock_channel.execute_read = AsyncMock(return_value={
            "records": [
                {"document_id": "doc_001", "mime_type": "application/pdf", "filename": "test1.pdf"},
                {"document_id": "doc_002", "mime_type": "text/html", "filename": "test2.html"}
            ]
        })

        mock_channel.execute_write = AsyncMock(return_value={
            "records": [{"document_id": "doc_001"}]
        })

        mock_get_channel.return_value = mock_channel

        result = await migrate_existing_documents()

        assert result["status"] == "success"
        assert result["documents_checked"] == 2
        assert result["documents_migrated"] == 2
        assert result["documents_failed"] == 0
        assert "duration_seconds" in result


# ============================================================================
# Test Idempotency
# ============================================================================

@pytest.mark.asyncio
async def test_migration_idempotency():
    """Test that migration can be run multiple times safely."""
    with patch('scripts.migrate_source_metadata.get_graph_channel') as mock_get_channel:
        # Mock graph channel
        mock_channel = AsyncMock()

        # First run: documents need migration
        mock_channel.execute_read = AsyncMock(side_effect=[
            # First call: documents needing migration
            {"records": [{"document_id": "doc_001", "mime_type": "application/pdf", "filename": "test.pdf"}]},
            # Second call (after migration): no documents need migration
            {"records": []}
        ])

        mock_channel.execute_write = AsyncMock(return_value={
            "records": [{"document_id": "doc_001"}]
        })

        mock_get_channel.return_value = mock_channel

        # First run
        result1 = await migrate_existing_documents()
        assert result1["status"] == "success"
        assert result1["documents_migrated"] == 1

        # Second run (should find no documents to migrate)
        result2 = await migrate_existing_documents()
        assert result2["status"] == "success"
        assert result2["documents_checked"] == 0


# ============================================================================
# Test Verification
# ============================================================================

@pytest.mark.asyncio
async def test_verify_migration_success():
    """Test verification when all documents have source metadata."""
    with patch('scripts.migrate_source_metadata.get_graph_channel') as mock_get_channel:
        # Mock graph channel
        mock_channel = AsyncMock()
        mock_channel.execute_read = AsyncMock(return_value={
            "records": [{
                "total_documents": 10,
                "with_source_type": 10,
                "with_connector_icon": 10
            }]
        })

        mock_get_channel.return_value = mock_channel

        success = await verify_migration()

        assert success is True


@pytest.mark.asyncio
async def test_verify_migration_incomplete():
    """Test verification when some documents lack source metadata."""
    with patch('scripts.migrate_source_metadata.get_graph_channel') as mock_get_channel:
        # Mock graph channel
        mock_channel = AsyncMock()
        mock_channel.execute_read = AsyncMock(return_value={
            "records": [{
                "total_documents": 10,
                "with_source_type": 8,
                "with_connector_icon": 8
            }]
        })

        mock_get_channel.return_value = mock_channel

        success = await verify_migration()

        assert success is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
