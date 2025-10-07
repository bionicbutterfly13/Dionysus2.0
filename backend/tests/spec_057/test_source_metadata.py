#!/usr/bin/env python3
"""
Test Suite for Source Metadata - Spec 057

Tests schema validation, uploaded_file vs url distinction,
and connector_icon inference.

Author: Spec 057 Implementation
Created: 2025-10-07
"""

import pytest
from pydantic import ValidationError
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.document_node import DocumentNode
from src.services.document_repository import infer_connector_icon


# ============================================================================
# Test Schema Validation
# ============================================================================

def test_document_node_default_source_type():
    """Test that DocumentNode defaults to uploaded_file source_type."""
    doc = DocumentNode(
        document_id="doc_001",
        filename="test.pdf",
        content_hash="a" * 64,
        file_size=1024,
        mime_type="application/pdf",
        quality_overall=0.85
    )

    assert doc.source_type == "uploaded_file"
    assert doc.original_url is None
    assert doc.connector_icon is None


def test_document_node_valid_source_types():
    """Test all valid source_type values."""
    valid_types = ["uploaded_file", "url", "api"]

    for source_type in valid_types:
        doc = DocumentNode(
            document_id=f"doc_{source_type}",
            filename="test.pdf",
            content_hash="a" * 64,
            file_size=1024,
            mime_type="application/pdf",
            quality_overall=0.85,
            source_type=source_type
        )
        assert doc.source_type == source_type


def test_document_node_invalid_source_type():
    """Test that invalid source_type raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        DocumentNode(
            document_id="doc_invalid",
            filename="test.pdf",
            content_hash="a" * 64,
            file_size=1024,
            mime_type="application/pdf",
            quality_overall=0.85,
            source_type="invalid_type"
        )

    error = exc_info.value.errors()[0]
    assert "source_type must be one of" in str(error["ctx"]["error"])


def test_document_node_valid_url():
    """Test that valid HTTP(S) URLs are accepted."""
    valid_urls = [
        "https://example.com/paper.pdf",
        "http://arxiv.org/pdf/2024.12345.pdf",
        "https://www.example.com/path/to/document.pdf"
    ]

    for url in valid_urls:
        doc = DocumentNode(
            document_id="doc_url",
            filename="test.pdf",
            content_hash="a" * 64,
            file_size=1024,
            mime_type="application/pdf",
            quality_overall=0.85,
            source_type="url",
            original_url=url
        )
        assert doc.original_url == url


def test_document_node_invalid_url_no_protocol():
    """Test that URLs without http(s):// are rejected."""
    with pytest.raises(ValidationError) as exc_info:
        DocumentNode(
            document_id="doc_invalid_url",
            filename="test.pdf",
            content_hash="a" * 64,
            file_size=1024,
            mime_type="application/pdf",
            quality_overall=0.85,
            source_type="url",
            original_url="example.com/paper.pdf"
        )

    error = exc_info.value.errors()[0]
    assert "must start with http://" in str(error["ctx"]["error"])


def test_document_node_invalid_url_with_spaces():
    """Test that URLs with spaces are rejected."""
    with pytest.raises(ValidationError) as exc_info:
        DocumentNode(
            document_id="doc_invalid_url",
            filename="test.pdf",
            content_hash="a" * 64,
            file_size=1024,
            mime_type="application/pdf",
            quality_overall=0.85,
            source_type="url",
            original_url="https://example.com/paper with spaces.pdf"
        )

    error = exc_info.value.errors()[0]
    assert "must not contain spaces" in str(error["ctx"]["error"])


def test_document_node_url_too_long():
    """Test that URLs longer than 2048 characters are rejected."""
    long_url = "https://example.com/" + "a" * 2100

    with pytest.raises(ValidationError) as exc_info:
        DocumentNode(
            document_id="doc_long_url",
            filename="test.pdf",
            content_hash="a" * 64,
            file_size=1024,
            mime_type="application/pdf",
            quality_overall=0.85,
            source_type="url",
            original_url=long_url
        )

    error = exc_info.value.errors()[0]
    assert "must not exceed 2048 characters" in str(error["ctx"]["error"])


# ============================================================================
# Test Uploaded File vs URL Distinction
# ============================================================================

def test_uploaded_file_document():
    """Test document with uploaded_file source_type."""
    doc = DocumentNode(
        document_id="doc_upload",
        filename="uploaded_research.pdf",
        content_hash="a" * 64,
        file_size=2048576,
        mime_type="application/pdf",
        quality_overall=0.90,
        source_type="uploaded_file",
        original_url=None,
        connector_icon="pdf"
    )

    assert doc.source_type == "uploaded_file"
    assert doc.original_url is None
    assert doc.connector_icon == "pdf"


def test_url_document():
    """Test document with url source_type."""
    doc = DocumentNode(
        document_id="doc_url",
        filename="arxiv_paper.pdf",
        content_hash="b" * 64,
        file_size=1536000,
        mime_type="application/pdf",
        quality_overall=0.85,
        source_type="url",
        original_url="https://arxiv.org/pdf/2024.12345.pdf",
        connector_icon="web"
    )

    assert doc.source_type == "url"
    assert doc.original_url == "https://arxiv.org/pdf/2024.12345.pdf"
    assert doc.connector_icon == "web"


def test_api_document_with_url():
    """Test document with api source_type and URL."""
    doc = DocumentNode(
        document_id="doc_api",
        filename="api_fetched.pdf",
        content_hash="c" * 64,
        file_size=1024000,
        mime_type="application/pdf",
        quality_overall=0.88,
        source_type="api",
        original_url="https://api.example.com/documents/12345",
        connector_icon="api"
    )

    assert doc.source_type == "api"
    assert doc.original_url == "https://api.example.com/documents/12345"
    assert doc.connector_icon == "api"


def test_api_document_without_url():
    """Test document with api source_type but no URL."""
    doc = DocumentNode(
        document_id="doc_api_no_url",
        filename="api_generated.pdf",
        content_hash="d" * 64,
        file_size=512000,
        mime_type="application/pdf",
        quality_overall=0.82,
        source_type="api",
        original_url=None,
        connector_icon="api"
    )

    assert doc.source_type == "api"
    assert doc.original_url is None
    assert doc.connector_icon == "api"


# ============================================================================
# Test Connector Icon Inference
# ============================================================================

def test_infer_connector_icon_pdf():
    """Test connector_icon inference for PDF."""
    icon = infer_connector_icon("application/pdf", "uploaded_file")
    assert icon == "pdf"


def test_infer_connector_icon_html():
    """Test connector_icon inference for HTML."""
    icon = infer_connector_icon("text/html", "url")
    assert icon == "html"


def test_infer_connector_icon_text():
    """Test connector_icon inference for plain text."""
    icon = infer_connector_icon("text/plain", "uploaded_file")
    assert icon == "text"


def test_infer_connector_icon_doc():
    """Test connector_icon inference for Word documents."""
    icon = infer_connector_icon("application/msword", "uploaded_file")
    assert icon == "doc"

    icon = infer_connector_icon(
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "uploaded_file"
    )
    assert icon == "doc"


def test_infer_connector_icon_markdown():
    """Test connector_icon inference for Markdown."""
    icon = infer_connector_icon("text/markdown", "uploaded_file")
    assert icon == "markdown"


def test_infer_connector_icon_json():
    """Test connector_icon inference for JSON."""
    icon = infer_connector_icon("application/json", "api")
    assert icon == "json"


def test_infer_connector_icon_unknown_mime_uploaded():
    """Test connector_icon inference for unknown MIME type with uploaded_file."""
    icon = infer_connector_icon("application/unknown", "uploaded_file")
    assert icon == "upload"


def test_infer_connector_icon_unknown_mime_url():
    """Test connector_icon inference for unknown MIME type with url."""
    icon = infer_connector_icon("application/unknown", "url")
    assert icon == "web"


def test_infer_connector_icon_unknown_mime_api():
    """Test connector_icon inference for unknown MIME type with api."""
    icon = infer_connector_icon("application/unknown", "api")
    assert icon == "api"


# ============================================================================
# Test Download Metadata
# ============================================================================

def test_document_with_download_metadata():
    """Test document with download metadata."""
    download_metadata = {
        "status_code": 200,
        "content_length": 1024000,
        "content_type": "application/pdf",
        "redirects": ["https://example.com/redirect1", "https://example.com/final"],
        "fetch_timestamp": "2025-10-07T10:30:00Z"
    }

    doc = DocumentNode(
        document_id="doc_download",
        filename="downloaded.pdf",
        content_hash="e" * 64,
        file_size=1024000,
        mime_type="application/pdf",
        quality_overall=0.87,
        source_type="url",
        original_url="https://example.com/paper.pdf",
        connector_icon="pdf",
        download_metadata=download_metadata
    )

    assert doc.download_metadata is not None
    assert doc.download_metadata["status_code"] == 200
    assert len(doc.download_metadata["redirects"]) == 2


def test_document_without_download_metadata():
    """Test document without download metadata (uploaded file)."""
    doc = DocumentNode(
        document_id="doc_no_download",
        filename="uploaded.pdf",
        content_hash="f" * 64,
        file_size=2048000,
        mime_type="application/pdf",
        quality_overall=0.91,
        source_type="uploaded_file",
        download_metadata=None
    )

    assert doc.download_metadata is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
