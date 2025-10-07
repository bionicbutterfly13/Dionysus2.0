#!/usr/bin/env python3
"""
Document Chunker Service - Spec 056

Standardized document chunking using RecursiveCharacterTextSplitter pattern:
- Configurable chunk size (default: 1000 characters)
- Configurable overlap (default: 200 characters)
- Stable chunk IDs: {document_id}_chunk_{index}
- Metadata preservation (position, character offsets, parent doc)

Follows Perplexica's clean chunking approach for better citation highlighting.

Constitutional Compliance:
- Pure utility service, no direct Neo4j access
- Returns structured chunk data for DocumentRepository to persist

Author: Spec 056 Implementation
Created: 2025-10-07
"""

import re
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Chunk Result Type
# ============================================================================

class ChunkResult:
    """
    Structured result for a single document chunk.

    Attributes:
        chunk_id: Stable ID in format {document_id}_chunk_{index}
        content: Chunk text content
        position: Sequential position (0-indexed)
        start_char: Start character offset in original document
        end_char: End character offset in original document
        parent_document_id: Parent document ID
    """

    def __init__(
        self,
        chunk_id: str,
        content: str,
        position: int,
        start_char: int,
        end_char: int,
        parent_document_id: str
    ):
        self.chunk_id = chunk_id
        self.content = content
        self.position = position
        self.start_char = start_char
        self.end_char = end_char
        self.parent_document_id = parent_document_id

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for persistence."""
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "position": self.position,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "parent_document_id": self.parent_document_id
        }


# ============================================================================
# Document Chunker Service
# ============================================================================

class DocumentChunker:
    """
    Standardized document chunking service.

    Spec 056: Implements RecursiveCharacterTextSplitter pattern with:
    - Configurable chunk size and overlap
    - Stable, predictable chunk IDs
    - Metadata tracking for each chunk

    Example:
        >>> chunker = DocumentChunker(chunk_size=1000, overlap=200)
        >>> chunks = await chunker.chunk_document("doc_123", document_text)
        >>> print(f"Created {len(chunks)} chunks")
        >>> print(chunks[0]["chunk_id"])
        'doc_123_chunk_0'
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 200
    ):
        """
        Initialize document chunker.

        Args:
            chunk_size: Target size for each chunk in characters (default: 1000)
            overlap: Overlap between consecutive chunks (default: 200)

        Raises:
            ValueError: If chunk_size <= 0 or overlap >= chunk_size
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        if overlap < 0:
            raise ValueError("overlap cannot be negative")

        if overlap >= chunk_size:
            raise ValueError("overlap must be less than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap

        logger.info(
            f"DocumentChunker initialized: "
            f"chunk_size={self.chunk_size}, "
            f"overlap={self.overlap}"
        )

    async def chunk_document(
        self,
        document_id: str,
        content: str,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Split document into chunks with stable IDs and metadata.

        Spec 056: Core chunking method using RecursiveCharacterTextSplitter pattern.

        Args:
            document_id: Parent document ID
            content: Document text to chunk
            chunk_size: Optional override for chunk size
            overlap: Optional override for overlap

        Returns:
            List of chunk dictionaries with structure:
                {
                    "chunk_id": "doc_123_chunk_0",
                    "content": "chunk text...",
                    "position": 0,
                    "start_char": 0,
                    "end_char": 1000,
                    "parent_document_id": "doc_123"
                }

        Example:
            >>> chunker = DocumentChunker()
            >>> chunks = await chunker.chunk_document("doc_456", "Long text...")
            >>> print(chunks[0]["chunk_id"])
            'doc_456_chunk_0'
        """
        # Use instance defaults or overrides
        size = chunk_size or self.chunk_size
        olap = overlap or self.overlap

        # Handle empty content
        if not content:
            logger.warning(f"Empty content for document {document_id}, returning empty chunk")
            return [{
                "chunk_id": f"{document_id}_chunk_0",
                "content": "",
                "position": 0,
                "start_char": 0,
                "end_char": 0,
                "parent_document_id": document_id
            }]

        # Split content into chunks
        chunks = self._split_text_recursive(content, size, olap)

        # Create chunk results with stable IDs
        chunk_results = []
        current_position = 0

        for i, (chunk_text, start_offset, end_offset) in enumerate(chunks):
            chunk_result = {
                "chunk_id": f"{document_id}_chunk_{i}",
                "content": chunk_text,
                "position": i,
                "start_char": start_offset,
                "end_char": end_offset,
                "parent_document_id": document_id
            }
            chunk_results.append(chunk_result)

        logger.info(
            f"✅ Chunked document {document_id}: "
            f"{len(content)} chars → {len(chunk_results)} chunks"
        )

        return chunk_results

    def _split_text_recursive(
        self,
        text: str,
        chunk_size: int,
        overlap: int
    ) -> List[tuple]:
        """
        Recursively split text into chunks with overlap.

        Uses a recursive strategy that respects natural breakpoints:
        1. Try to split on paragraph breaks (\n\n)
        2. Fall back to sentence breaks (. ! ?)
        3. Fall back to word breaks (spaces)
        4. Fall back to character breaks (hard split)

        Args:
            text: Text to split
            chunk_size: Target chunk size
            overlap: Overlap between chunks

        Returns:
            List of tuples: (chunk_text, start_offset, end_offset)
        """
        if len(text) <= chunk_size:
            # Text fits in one chunk
            return [(text, 0, len(text))]

        chunks = []
        position = 0

        while position < len(text):
            # Calculate chunk end position
            end_position = min(position + chunk_size, len(text))

            # Extract chunk
            chunk_text = text[position:end_position]

            # If this isn't the last chunk and we're not at a natural break,
            # try to find a better split point
            if end_position < len(text):
                # Try to split at paragraph break
                better_end = self._find_split_point(
                    chunk_text,
                    ["\n\n", "\n", ". ", "! ", "? ", " "]
                )

                if better_end > 0:
                    end_position = position + better_end
                    chunk_text = text[position:end_position]

            # Add chunk
            chunks.append((chunk_text, position, end_position))

            # Move position forward with overlap
            if end_position >= len(text):
                # Last chunk, done
                break

            position = end_position - overlap

            # Ensure we're making progress
            if position <= chunks[-1][1]:
                # Force progress if overlap logic fails
                position = end_position

        return chunks

    def _find_split_point(
        self,
        text: str,
        separators: List[str]
    ) -> int:
        """
        Find the best split point in text using separator priority.

        Args:
            text: Text to find split in
            separators: List of separators in priority order

        Returns:
            Position of best split point (from end of text), or 0 if none found
        """
        for separator in separators:
            # Look for separator in the last 20% of the text
            # (prefer splits near the end to maximize chunk size)
            search_start = int(len(text) * 0.8)
            last_occurrence = text.rfind(separator, search_start)

            if last_occurrence > 0:
                # Found a separator, split after it
                return last_occurrence + len(separator)

        # No good split found
        return 0
