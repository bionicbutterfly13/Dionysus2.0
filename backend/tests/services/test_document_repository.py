#!/usr/bin/env python3
"""
Test Document Repository - Spec 054 + Spec 055 Agent 1

Tests for DocumentRepository including:
- SHA-256 content hash computation
- Content hash validation
- Duplicate detection via content_hash

Author: Spec 054 + Spec 055 Agent 1 Implementation
Created: 2025-10-07
"""

import pytest
import hashlib
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

# Imports will work due to conftest.py path setup
from src.services.document_repository import (
    DocumentRepository,
    compute_content_hash,
    validate_content_hash
)
from src.models.document_node import DocumentNode


class TestContentHashComputation:
    """Test SHA-256 content hash computation (Spec 055 Agent 1)."""

    def test_compute_content_hash_basic(self):
        """Test basic SHA-256 hash computation."""
        # Test with simple content and namespace
        content = "Test document content"
        namespace = "test_namespace"

        # Expected hash
        combined = content + namespace
        expected_hash = hashlib.sha256(combined.encode('utf-8')).hexdigest()

        actual_hash = compute_content_hash(content, namespace)

        assert actual_hash == expected_hash
        assert len(actual_hash) == 64  # SHA-256 produces 64 hex characters
        assert all(c in '0123456789abcdef' for c in actual_hash)

    def test_compute_content_hash_deterministic(self):
        """Test that same input produces same hash (deterministic)."""
        content = "Identical content"
        namespace = "same_namespace"

        hash1 = compute_content_hash(content, namespace)
        hash2 = compute_content_hash(content, namespace)
        hash3 = compute_content_hash(content, namespace)

        assert hash1 == hash2 == hash3

    def test_compute_content_hash_different_content(self):
        """Test that different content produces different hashes."""
        namespace = "test_namespace"
        hash1 = compute_content_hash("Content A", namespace)
        hash2 = compute_content_hash("Content B", namespace)

        assert hash1 != hash2

    def test_compute_content_hash_different_namespace(self):
        """Test that different namespace produces different hashes."""
        content = "Same content"
        hash1 = compute_content_hash(content, "namespace_1")
        hash2 = compute_content_hash(content, "namespace_2")

        assert hash1 != hash2

    def test_compute_content_hash_empty_content(self):
        """Test hash computation with empty content."""
        # Should still produce valid hash
        hash_result = compute_content_hash("", "namespace")

        assert len(hash_result) == 64
        assert all(c in '0123456789abcdef' for c in hash_result)

    def test_compute_content_hash_unicode_content(self):
        """Test hash computation with unicode characters."""
        content = "Test with unicode: 你好世界 🚀 café"
        namespace = "unicode_test"

        hash_result = compute_content_hash(content, namespace)

        assert len(hash_result) == 64
        assert all(c in '0123456789abcdef' for c in hash_result)

        # Should be deterministic even with unicode
        hash2 = compute_content_hash(content, namespace)
        assert hash_result == hash2

    def test_compute_content_hash_large_content(self):
        """Test hash computation with large content."""
        # Simulate a large document (1MB of text)
        content = "A" * (1024 * 1024)
        namespace = "large_doc"

        hash_result = compute_content_hash(content, namespace)

        assert len(hash_result) == 64
        assert all(c in '0123456789abcdef' for c in hash_result)


class TestContentHashValidation:
    """Test content hash validation (Spec 055 Agent 1)."""

    def test_validate_content_hash_valid(self):
        """Test validation accepts valid SHA-256 hash."""
        valid_hash = "a" * 64  # 64 hex characters
        assert validate_content_hash(valid_hash) is True

    def test_validate_content_hash_invalid_length(self):
        """Test validation rejects wrong length."""
        # Too short
        assert validate_content_hash("abc123") is False

        # Too long
        assert validate_content_hash("a" * 65) is False

    def test_validate_content_hash_invalid_characters(self):
        """Test validation rejects non-hex characters."""
        # Contains non-hex characters
        invalid_hash = "g" * 64  # 'g' is not a hex character
        assert validate_content_hash(invalid_hash) is False

        # Contains spaces
        invalid_hash2 = ("a" * 60) + "    "
        assert validate_content_hash(invalid_hash2) is False

    def test_validate_content_hash_uppercase(self):
        """Test validation handles uppercase hex."""
        # Uppercase should be valid (normalized to lowercase)
        uppercase_hash = "A" * 64
        assert validate_content_hash(uppercase_hash) is True

    def test_validate_content_hash_mixed_case(self):
        """Test validation handles mixed case hex."""
        # Create exactly 64 characters of mixed case hex
        mixed_hash = ("aB" * 32)  # 64 chars
        assert len(mixed_hash) == 64
        assert validate_content_hash(mixed_hash) is True


class TestDocumentRepositoryContentHash:
    """Test DocumentRepository integration with content hash."""

    @pytest.mark.asyncio
    async def test_persist_document_computes_content_hash(self):
        """Test that persist_document computes content_hash if not provided."""
        repo = DocumentRepository()

        # Mock the graph channel
        repo.graph_channel = AsyncMock()
        repo.graph_channel.execute_read = AsyncMock(return_value={"records": []})
        repo.graph_channel.execute_write = AsyncMock(return_value={"records": []})

        final_output = {
            "quality": {"scores": {"overall": 0.85}},
            "research": {"curiosity_triggers": 3, "research_questions": 5},
            "concepts": {},
            "basins": [],
            "thoughtseeds": []
        }

        metadata = {
            "document_id": "doc_test_001",
            "filename": "test.pdf",
            "file_size": 1024,
            "mime_type": "application/pdf",
            "document_body": "Test document content for hashing"
        }

        # Should compute content_hash from document_body
        result = await repo.persist_document(final_output, metadata)

        assert "content_hash" in metadata or result.get("content_hash")

    @pytest.mark.asyncio
    async def test_persist_document_validates_content_hash(self):
        """Test that persist_document validates content_hash format."""
        repo = DocumentRepository()

        final_output = {
            "quality": {"scores": {"overall": 0.85}},
            "research": {},
            "concepts": {},
            "basins": [],
            "thoughtseeds": []
        }

        # Invalid content_hash (too short)
        metadata = {
            "document_id": "doc_test_002",
            "filename": "test.pdf",
            "content_hash": "abc123",  # Invalid
            "file_size": 1024
        }

        with pytest.raises(ValueError, match="Invalid content_hash format"):
            await repo.persist_document(final_output, metadata)

    @pytest.mark.asyncio
    async def test_duplicate_detection_via_content_hash(self):
        """Test that duplicate content is detected via content_hash."""
        # Create repository with mocked dependencies
        with patch('src.services.document_repository.get_graph_channel') as mock_get_channel:
            # Mock the graph channel
            mock_channel = AsyncMock()
            mock_get_channel.return_value = mock_channel

            repo = DocumentRepository()

            # Configure mock to return existing document for duplicate check
            mock_channel.execute_read.return_value = {
                "records": [{
                    "document_id": "doc_existing",
                    "filename": "existing.pdf",
                    "upload_timestamp": "2025-10-07T10:00:00",
                    "quality_overall": 0.85,
                    "tier": "warm",
                    "tags": ["research"],
                    "file_size": 2048,
                    "access_count": 5
                }]
            }

            final_output = {
                "quality": {"scores": {"overall": 0.85}},
                "research": {},
                "concepts": {},
                "basins": [],
                "thoughtseeds": []
            }

            metadata = {
                "document_id": "doc_duplicate",
                "filename": "duplicate.pdf",
                "content_hash": "a" * 64,  # Same as existing
                "file_size": 1024
            }

            with pytest.raises(ValueError, match="Duplicate document detected"):
                await repo.persist_document(final_output, metadata)


class TestDocumentNodeContentHashValidation:
    """Test DocumentNode model validates content_hash (Spec 055 Agent 1)."""

    def test_document_node_requires_content_hash(self):
        """Test that DocumentNode requires content_hash field."""
        # Should fail without content_hash
        with pytest.raises(ValueError):
            DocumentNode(
                document_id="doc_001",
                filename="test.pdf",
                # content_hash missing
                file_size=1024,
                mime_type="application/pdf",
                quality_overall=0.85
            )

    def test_document_node_validates_content_hash_format(self):
        """Test that DocumentNode validates content_hash is SHA-256 format."""
        # Invalid format (too short)
        with pytest.raises(ValueError):
            DocumentNode(
                document_id="doc_002",
                filename="test.pdf",
                content_hash="abc123",  # Invalid
                file_size=1024,
                mime_type="application/pdf",
                quality_overall=0.85
            )

    def test_document_node_accepts_valid_content_hash(self):
        """Test that DocumentNode accepts valid SHA-256 hash."""
        valid_hash = "a" * 64

        doc = DocumentNode(
            document_id="doc_003",
            filename="test.pdf",
            content_hash=valid_hash,
            file_size=1024,
            mime_type="application/pdf",
            quality_overall=0.85
        )

        assert doc.content_hash == valid_hash
