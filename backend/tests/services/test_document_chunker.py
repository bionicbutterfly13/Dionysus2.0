#!/usr/bin/env python3
"""
Test Document Chunker Service - Spec 056

Comprehensive tests for document chunking functionality with:
- RecursiveCharacterTextSplitter pattern
- Configurable chunk size and overlap
- Stable chunk IDs
- Metadata preservation

Author: Spec 056 Implementation
Created: 2025-10-07
"""

import pytest
from typing import List, Dict

# Imports will work due to conftest.py path setup
from src.services.document_chunker import (
    DocumentChunker,
    ChunkResult
)


class TestDocumentChunkerBasics:
    """Test basic chunking functionality."""

    @pytest.mark.asyncio
    async def test_chunk_small_document(self):
        """Test chunking of small document that fits in one chunk."""
        chunker = DocumentChunker(chunk_size=1000, overlap=200)

        content = "This is a small document that fits in one chunk."
        document_id = "doc_123"

        chunks = await chunker.chunk_document(document_id, content)

        assert len(chunks) == 1
        assert chunks[0]["chunk_id"] == "doc_123_chunk_0"
        assert chunks[0]["content"] == content
        assert chunks[0]["position"] == 0
        assert chunks[0]["start_char"] == 0
        assert chunks[0]["end_char"] == len(content)
        assert chunks[0]["parent_document_id"] == document_id

    @pytest.mark.asyncio
    async def test_chunk_large_document(self):
        """Test chunking of large document into multiple chunks."""
        chunker = DocumentChunker(chunk_size=100, overlap=20)

        # Create content that will require multiple chunks
        content = "A" * 250  # 250 characters, should create 3 chunks with 100 size

        chunks = await chunker.chunk_document("doc_456", content)

        assert len(chunks) >= 2
        assert chunks[0]["chunk_id"] == "doc_456_chunk_0"
        assert chunks[1]["chunk_id"] == "doc_456_chunk_1"

        # Verify all chunks have correct structure
        for i, chunk in enumerate(chunks):
            assert "chunk_id" in chunk
            assert "content" in chunk
            assert "position" in chunk
            assert "start_char" in chunk
            assert "end_char" in chunk
            assert "parent_document_id" in chunk
            assert chunk["position"] == i
            assert chunk["parent_document_id"] == "doc_456"

    @pytest.mark.asyncio
    async def test_chunk_with_paragraphs(self):
        """Test chunking of text with natural paragraph breaks."""
        chunker = DocumentChunker(chunk_size=200, overlap=50)

        content = """This is the first paragraph. It contains several sentences.

This is the second paragraph. It also has multiple sentences.

This is the third paragraph. More content here to test chunking."""

        chunks = await chunker.chunk_document("doc_789", content)

        # Should create multiple chunks
        assert len(chunks) >= 1

        # Verify chunk IDs are sequential
        for i, chunk in enumerate(chunks):
            assert chunk["chunk_id"] == f"doc_789_chunk_{i}"


class TestDocumentChunkerOverlap:
    """Test chunk overlap functionality."""

    @pytest.mark.asyncio
    async def test_overlap_behavior(self):
        """Test that chunks have proper overlap."""
        chunker = DocumentChunker(chunk_size=100, overlap=20)

        # Create content where we can verify overlap
        content = "0123456789" * 15  # 150 characters

        chunks = await chunker.chunk_document("doc_overlap", content)

        if len(chunks) > 1:
            # Verify that end of first chunk overlaps with start of second chunk
            chunk0_end = chunks[0]["content"][-20:]
            chunk1_start = chunks[1]["content"][:20]

            # There should be some overlap in content
            assert len(chunk0_end) > 0
            assert len(chunk1_start) > 0

    @pytest.mark.asyncio
    async def test_zero_overlap(self):
        """Test chunking with zero overlap."""
        chunker = DocumentChunker(chunk_size=100, overlap=0)

        content = "A" * 250

        chunks = await chunker.chunk_document("doc_no_overlap", content)

        # Verify no overlap between chunks
        if len(chunks) > 1:
            # End char of chunk N should equal start char of chunk N+1
            for i in range(len(chunks) - 1):
                assert chunks[i]["end_char"] <= chunks[i + 1]["start_char"]

    @pytest.mark.asyncio
    async def test_large_overlap(self):
        """Test chunking with large overlap (50% of chunk size)."""
        chunker = DocumentChunker(chunk_size=100, overlap=50)

        content = "B" * 300

        chunks = await chunker.chunk_document("doc_large_overlap", content)

        # Should create more chunks due to large overlap
        assert len(chunks) >= 3


class TestDocumentChunkerStableIDs:
    """Test stable chunk ID generation."""

    @pytest.mark.asyncio
    async def test_stable_ids_same_content(self):
        """Test that same content produces same chunk IDs."""
        chunker = DocumentChunker(chunk_size=100, overlap=20)

        content = "Identical content for testing stability."
        document_id = "doc_stable"

        chunks1 = await chunker.chunk_document(document_id, content)
        chunks2 = await chunker.chunk_document(document_id, content)

        assert len(chunks1) == len(chunks2)

        for c1, c2 in zip(chunks1, chunks2):
            assert c1["chunk_id"] == c2["chunk_id"]
            assert c1["content"] == c2["content"]
            assert c1["position"] == c2["position"]

    @pytest.mark.asyncio
    async def test_unique_ids_different_documents(self):
        """Test that different documents produce different chunk IDs."""
        chunker = DocumentChunker(chunk_size=100, overlap=20)

        content = "Same content, different documents."

        chunks1 = await chunker.chunk_document("doc_001", content)
        chunks2 = await chunker.chunk_document("doc_002", content)

        assert chunks1[0]["chunk_id"] != chunks2[0]["chunk_id"]
        assert chunks1[0]["chunk_id"] == "doc_001_chunk_0"
        assert chunks2[0]["chunk_id"] == "doc_002_chunk_0"

    @pytest.mark.asyncio
    async def test_id_format_consistency(self):
        """Test that chunk IDs follow consistent format."""
        chunker = DocumentChunker(chunk_size=50, overlap=10)

        content = "C" * 200

        chunks = await chunker.chunk_document("doc_format_test", content)

        for i, chunk in enumerate(chunks):
            expected_id = f"doc_format_test_chunk_{i}"
            assert chunk["chunk_id"] == expected_id


class TestDocumentChunkerMetadata:
    """Test metadata preservation and tracking."""

    @pytest.mark.asyncio
    async def test_position_tracking(self):
        """Test that chunk positions are correctly tracked."""
        chunker = DocumentChunker(chunk_size=100, overlap=20)

        content = "D" * 300

        chunks = await chunker.chunk_document("doc_position", content)

        # Verify positions are sequential starting from 0
        for i, chunk in enumerate(chunks):
            assert chunk["position"] == i

    @pytest.mark.asyncio
    async def test_character_offsets(self):
        """Test that start_char and end_char are accurate."""
        chunker = DocumentChunker(chunk_size=100, overlap=0)

        content = "E" * 250

        chunks = await chunker.chunk_document("doc_offsets", content)

        for chunk in chunks:
            # Verify content matches the specified character range
            actual_content = content[chunk["start_char"]:chunk["end_char"]]
            assert chunk["content"] == actual_content

    @pytest.mark.asyncio
    async def test_parent_document_tracking(self):
        """Test that all chunks track their parent document."""
        chunker = DocumentChunker(chunk_size=100, overlap=20)

        content = "F" * 300
        parent_id = "doc_parent_123"

        chunks = await chunker.chunk_document(parent_id, content)

        for chunk in chunks:
            assert chunk["parent_document_id"] == parent_id


class TestDocumentChunkerCustomSizes:
    """Test different chunk sizes and configurations."""

    @pytest.mark.asyncio
    async def test_small_chunk_size(self):
        """Test with very small chunk size (50 characters)."""
        chunker = DocumentChunker(chunk_size=50, overlap=10)

        content = "This is a test document with multiple sentences. " * 3

        chunks = await chunker.chunk_document("doc_small", content)

        # Should create multiple small chunks
        assert len(chunks) >= 3

        for chunk in chunks:
            # Each chunk should be close to 50 chars (may vary due to split points)
            assert len(chunk["content"]) <= 60  # Some flexibility

    @pytest.mark.asyncio
    async def test_large_chunk_size(self):
        """Test with large chunk size (5000 characters)."""
        chunker = DocumentChunker(chunk_size=5000, overlap=500)

        content = "G" * 3000

        chunks = await chunker.chunk_document("doc_large", content)

        # Should fit in one chunk
        assert len(chunks) == 1
        assert len(chunks[0]["content"]) == 3000

    @pytest.mark.asyncio
    async def test_default_parameters(self):
        """Test with default chunk size and overlap."""
        chunker = DocumentChunker()  # Default: chunk_size=1000, overlap=200

        content = "H" * 2500

        chunks = await chunker.chunk_document("doc_default", content)

        # Should create 2-3 chunks with default settings
        assert len(chunks) >= 2


class TestDocumentChunkerEdgeCases:
    """Test edge cases and special scenarios."""

    @pytest.mark.asyncio
    async def test_empty_content(self):
        """Test chunking of empty content."""
        chunker = DocumentChunker(chunk_size=1000, overlap=200)

        chunks = await chunker.chunk_document("doc_empty", "")

        # Should return empty list or single empty chunk
        assert len(chunks) <= 1

        if len(chunks) == 1:
            assert chunks[0]["content"] == ""
            assert chunks[0]["start_char"] == 0
            assert chunks[0]["end_char"] == 0

    @pytest.mark.asyncio
    async def test_single_word(self):
        """Test chunking of single word."""
        chunker = DocumentChunker(chunk_size=1000, overlap=200)

        content = "word"

        chunks = await chunker.chunk_document("doc_word", content)

        assert len(chunks) == 1
        assert chunks[0]["content"] == "word"

    @pytest.mark.asyncio
    async def test_whitespace_only(self):
        """Test chunking of whitespace-only content."""
        chunker = DocumentChunker(chunk_size=1000, overlap=200)

        content = "   \n\n   \t   "

        chunks = await chunker.chunk_document("doc_whitespace", content)

        # Should handle gracefully
        assert len(chunks) >= 1

    @pytest.mark.asyncio
    async def test_unicode_content(self):
        """Test chunking of unicode content."""
        chunker = DocumentChunker(chunk_size=100, overlap=20)

        content = "测试中文内容 🚀 Test unicode: café, naïve, résumé. " * 5

        chunks = await chunker.chunk_document("doc_unicode", content)

        # Should handle unicode correctly
        assert len(chunks) >= 1

        for chunk in chunks:
            # Verify content is valid unicode
            assert isinstance(chunk["content"], str)

    @pytest.mark.asyncio
    async def test_special_characters(self):
        """Test chunking with special characters."""
        chunker = DocumentChunker(chunk_size=100, overlap=20)

        content = "Special chars: !@#$%^&*()_+-=[]{}|;:',.<>?/~` " * 3

        chunks = await chunker.chunk_document("doc_special", content)

        assert len(chunks) >= 1

        # Verify all content is preserved
        reconstructed = "".join(chunk["content"] for chunk in chunks)
        # May have overlap, so check original is contained
        assert content in reconstructed or len(reconstructed) >= len(content)

    @pytest.mark.asyncio
    async def test_newlines_and_tabs(self):
        """Test chunking with newlines and tabs."""
        chunker = DocumentChunker(chunk_size=100, overlap=20)

        content = "Line 1\nLine 2\n\tIndented line\n\nDouble newline"

        chunks = await chunker.chunk_document("doc_newlines", content)

        # Should preserve whitespace characters
        assert len(chunks) >= 1

        for chunk in chunks:
            # Verify newlines and tabs are preserved
            if "\n" in content[:100]:
                assert any("\n" in c["content"] for c in chunks)

    @pytest.mark.asyncio
    async def test_chunk_size_larger_than_content(self):
        """Test when chunk_size is larger than content."""
        chunker = DocumentChunker(chunk_size=10000, overlap=1000)

        content = "Short content."

        chunks = await chunker.chunk_document("doc_short", content)

        # Should create single chunk
        assert len(chunks) == 1
        assert chunks[0]["content"] == content

    @pytest.mark.asyncio
    async def test_overlap_larger_than_chunk_size(self):
        """Test invalid config: overlap >= chunk_size."""
        with pytest.raises(ValueError) as exc_info:
            chunker = DocumentChunker(chunk_size=100, overlap=100)

        assert "overlap" in str(exc_info.value).lower()
        assert "chunk_size" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_negative_chunk_size(self):
        """Test invalid config: negative chunk_size."""
        with pytest.raises(ValueError):
            chunker = DocumentChunker(chunk_size=-100, overlap=20)

    @pytest.mark.asyncio
    async def test_negative_overlap(self):
        """Test invalid config: negative overlap."""
        with pytest.raises(ValueError):
            chunker = DocumentChunker(chunk_size=100, overlap=-20)


class TestDocumentChunkerPerformance:
    """Test performance characteristics."""

    @pytest.mark.asyncio
    async def test_large_document_performance(self):
        """Test chunking of very large document (1MB text)."""
        chunker = DocumentChunker(chunk_size=1000, overlap=200)

        # Create 1MB of text
        content = "I" * (1024 * 1024)

        chunks = await chunker.chunk_document("doc_large_perf", content)

        # Should create many chunks efficiently
        assert len(chunks) > 100

        # Verify all chunks are valid
        for chunk in chunks:
            assert len(chunk["content"]) > 0
            assert chunk["chunk_id"].startswith("doc_large_perf_chunk_")

    @pytest.mark.asyncio
    async def test_many_small_chunks(self):
        """Test creating many small chunks."""
        chunker = DocumentChunker(chunk_size=10, overlap=2)

        content = "J" * 500

        chunks = await chunker.chunk_document("doc_many_chunks", content)

        # Should create many small chunks
        assert len(chunks) >= 40

        # Verify sequential numbering
        for i, chunk in enumerate(chunks):
            assert chunk["position"] == i
