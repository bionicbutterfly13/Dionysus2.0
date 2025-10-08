#!/usr/bin/env python3
"""
Contract Test: GET /api/documents/{id}/citations - Spec 058

GREEN phase: Using mocked citation service for deterministic tests.

Tests cover:
- 200 OK with chunk, basin, and thoughtseed payload
- 404 when document missing
- 400 when chunk ID not found for the document
- Proper schema validation for all citation fields

Author: Agent 058B
Created: 2025-10-08
Updated: 2025-10-08 (GREEN phase with mocks)
"""

import pytest
import pytest_asyncio
from httpx import AsyncClient
from unittest.mock import AsyncMock, patch
from datetime import datetime

# Import models for mocking
import sys
from pathlib import Path
backend_src = Path(__file__).parent.parent.parent / "src"
if str(backend_src) not in sys.path:
    sys.path.insert(0, str(backend_src))

from src.models.citation import (
    CitationResponse,
    ChunkCitationData,
    BasinCitationData,
    ThoughtseedCitationData
)


# ==============================================================================
# Mock Data Fixtures
# ==============================================================================

@pytest.fixture
def mock_chunk_data():
    """Mock chunk citation data"""
    return ChunkCitationData(
        chunk_id="chunk_123",
        chunk_text="This is the chunk text from the document. It contains important information about machine learning and neural networks.",
        chunk_index=0,
        start_offset=0,
        end_offset=115,
        created_at=datetime(2025, 10, 8, 12, 0, 0)
    )


@pytest.fixture
def mock_basin_data():
    """Mock basin citation data"""
    return BasinCitationData(
        basin_id="basin_456",
        basin_name="Cognitive Processing",
        stability=0.85,
        attractor_strength=0.72,
        layer_influences={
            "layer1": 0.3,
            "layer2": 0.5,
            "layer3": 0.2
        }
    )


@pytest.fixture
def mock_thoughtseed_data():
    """Mock thoughtseed citation data"""
    return ThoughtseedCitationData(
        thoughtseed_id="ts_789",
        resonance_score=0.91,
        concept_labels=["machine learning", "neural networks", "cognition"],
        emergence_timestamp=datetime(2025, 10, 8, 12, 0, 0)
    )


@pytest.fixture
def mock_citation_response(mock_chunk_data, mock_basin_data, mock_thoughtseed_data):
    """Mock complete citation response"""
    return CitationResponse(
        document_id="doc_test_001",
        chunk=mock_chunk_data,
        basin=mock_basin_data,
        thoughtseed=mock_thoughtseed_data
    )


@pytest.fixture
def mock_citation_null_basin(mock_chunk_data, mock_thoughtseed_data):
    """Mock citation response with null basin"""
    return CitationResponse(
        document_id="doc_test_002",
        chunk=mock_chunk_data,
        basin=None,
        thoughtseed=mock_thoughtseed_data
    )


@pytest.fixture
def mock_citation_null_thoughtseed(mock_chunk_data, mock_basin_data):
    """Mock citation response with null thoughtseed"""
    return CitationResponse(
        document_id="doc_test_003",
        chunk=mock_chunk_data,
        basin=mock_basin_data,
        thoughtseed=None
    )


# ==============================================================================
# Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_get_document_citations_success(test_client: AsyncClient, mock_citation_response):
    """
    Test successful retrieval of citation data for a specific chunk.

    Expected: 200 OK with complete citation payload including
    chunk text, basin metadata, and thoughtseed data.
    """
    document_id = "doc_test_001"
    chunk_id = "chunk_123"

    # Mock the citation service
    with patch('src.services.citation_service.CitationService.get_citation_data', new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_citation_response

        response = await test_client.get(
            f"/api/documents/{document_id}/citations",
            params={"chunk_id": chunk_id}
        )

    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}"

    data = response.json()

    # Top-level structure
    assert "document_id" in data
    assert data["document_id"] == document_id
    assert "chunk" in data
    assert "basin" in data
    assert "thoughtseed" in data

    # Chunk data section
    chunk = data["chunk"]
    assert "chunk_id" in chunk
    assert chunk["chunk_id"] == chunk_id
    assert "chunk_text" in chunk
    assert isinstance(chunk["chunk_text"], str)
    assert "chunk_index" in chunk
    assert isinstance(chunk["chunk_index"], int)
    assert "start_offset" in chunk
    assert isinstance(chunk["start_offset"], int)
    assert "end_offset" in chunk
    assert isinstance(chunk["end_offset"], int)
    assert "created_at" in chunk

    # Basin metadata section
    basin = data["basin"]
    assert "basin_id" in basin
    assert isinstance(basin["basin_id"], str)
    assert "basin_name" in basin
    assert isinstance(basin["basin_name"], str)
    assert "stability" in basin
    assert isinstance(basin["stability"], (int, float))
    assert 0 <= basin["stability"] <= 1
    assert "attractor_strength" in basin
    assert isinstance(basin["attractor_strength"], (int, float))
    assert 0 <= basin["attractor_strength"] <= 1
    assert "layer_influences" in basin
    assert isinstance(basin["layer_influences"], dict)

    # Thoughtseed data section
    thoughtseed = data["thoughtseed"]
    assert "thoughtseed_id" in thoughtseed
    assert isinstance(thoughtseed["thoughtseed_id"], str)
    assert "resonance_score" in thoughtseed
    assert isinstance(thoughtseed["resonance_score"], (int, float))
    assert 0 <= thoughtseed["resonance_score"] <= 1
    assert "concept_labels" in thoughtseed
    assert isinstance(thoughtseed["concept_labels"], list)
    assert "emergence_timestamp" in thoughtseed


@pytest.mark.asyncio
async def test_get_document_citations_document_not_found(test_client: AsyncClient):
    """
    Test requesting citations for non-existent document returns 404.

    Expected: 404 Not Found with error message.
    """
    non_existent_id = "doc_does_not_exist_99999"
    chunk_id = "chunk_123"

    # Mock service to raise ValueError for document not found
    with patch('src.services.citation_service.CitationService.get_citation_data', new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = ValueError(f"Document not found: {non_existent_id}")

        response = await test_client.get(
            f"/api/documents/{non_existent_id}/citations",
            params={"chunk_id": chunk_id}
        )

    assert response.status_code == 404, f"Expected 404 Not Found, got {response.status_code}"

    data = response.json()
    assert "detail" in data
    error_msg = data["detail"].lower()
    assert "not found" in error_msg or "does not exist" in error_msg


@pytest.mark.asyncio
async def test_get_document_citations_chunk_not_found(test_client: AsyncClient):
    """
    Test requesting citations with invalid chunk ID returns 400.

    Expected: 400 Bad Request when chunk_id doesn't belong to document.
    """
    document_id = "doc_test_001"
    invalid_chunk_id = "chunk_invalid_99999"

    # Mock service to raise ValueError for chunk not found
    with patch('src.services.citation_service.CitationService.get_citation_data', new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = ValueError(f"Chunk not found or doesn't belong to document: {invalid_chunk_id}")

        response = await test_client.get(
            f"/api/documents/{document_id}/citations",
            params={"chunk_id": invalid_chunk_id}
        )

    assert response.status_code == 400, f"Expected 400 Bad Request, got {response.status_code}"

    data = response.json()
    assert "detail" in data
    error_msg = data["detail"].lower()
    assert "chunk" in error_msg and ("not found" in error_msg or "doesn't belong" in error_msg)


@pytest.mark.asyncio
async def test_get_document_citations_missing_chunk_id(test_client: AsyncClient):
    """
    Test requesting citations without chunk_id parameter returns 422.

    Expected: 422 Unprocessable Entity for missing required parameter.
    """
    document_id = "doc_test_001"

    response = await test_client.get(
        f"/api/documents/{document_id}/citations"
        # No chunk_id parameter
    )

    assert response.status_code == 422, f"Expected 422 Unprocessable Entity, got {response.status_code}"

    data = response.json()
    assert "detail" in data


@pytest.mark.asyncio
async def test_get_document_citations_with_null_basin(test_client: AsyncClient, mock_citation_null_basin):
    """
    Test citations endpoint handles chunks with no associated basin gracefully.

    Expected: 200 OK with basin field as null.
    """
    document_id = "doc_test_002"
    chunk_id = "chunk_no_basin_456"

    with patch('src.services.citation_service.CitationService.get_citation_data', new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_citation_null_basin

        response = await test_client.get(
            f"/api/documents/{document_id}/citations",
            params={"chunk_id": chunk_id}
        )

    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}"

    data = response.json()
    assert "basin" in data
    assert data["basin"] is None


@pytest.mark.asyncio
async def test_get_document_citations_with_null_thoughtseed(test_client: AsyncClient, mock_citation_null_thoughtseed):
    """
    Test citations endpoint handles chunks with no associated thoughtseed gracefully.

    Expected: 200 OK with thoughtseed field as null.
    """
    document_id = "doc_test_003"
    chunk_id = "chunk_no_thoughtseed_789"

    with patch('src.services.citation_service.CitationService.get_citation_data', new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_citation_null_thoughtseed

        response = await test_client.get(
            f"/api/documents/{document_id}/citations",
            params={"chunk_id": chunk_id}
        )

    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}"

    data = response.json()
    assert "thoughtseed" in data
    assert data["thoughtseed"] is None


@pytest.mark.asyncio
async def test_get_document_citations_performance(test_client: AsyncClient, mock_citation_response):
    """
    Test citations endpoint responds within acceptable performance target.

    Expected: Response time < 500ms for citation retrieval.
    """
    import time

    document_id = "doc_test_001"
    chunk_id = "chunk_123"

    with patch('src.services.citation_service.CitationService.get_citation_data', new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_citation_response

        start_time = time.time()
        response = await test_client.get(
            f"/api/documents/{document_id}/citations",
            params={"chunk_id": chunk_id}
        )
        elapsed_ms = (time.time() - start_time) * 1000

    assert response.status_code == 200
    # Mocked endpoint should be very fast
    assert elapsed_ms < 500, f"Response took {elapsed_ms:.2f}ms, expected < 500ms"


@pytest.mark.asyncio
async def test_get_document_citations_schema_validation(test_client: AsyncClient, mock_citation_response):
    """
    Test that all fields conform to expected types and constraints.

    Expected: All numeric fields in valid ranges, all IDs are strings, etc.
    """
    document_id = "doc_test_001"
    chunk_id = "chunk_123"

    with patch('src.services.citation_service.CitationService.get_citation_data', new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_citation_response

        response = await test_client.get(
            f"/api/documents/{document_id}/citations",
            params={"chunk_id": chunk_id}
        )

    assert response.status_code == 200

    data = response.json()

    # Validate chunk schema
    chunk = data["chunk"]
    assert chunk["chunk_index"] >= 0, "Chunk index must be non-negative"
    assert chunk["start_offset"] >= 0, "Start offset must be non-negative"
    assert chunk["end_offset"] > chunk["start_offset"], "End offset must be > start offset"

    # Validate basin schema
    if data["basin"] is not None:
        basin = data["basin"]
        assert 0 <= basin["stability"] <= 1, "Stability must be in [0, 1]"
        assert 0 <= basin["attractor_strength"] <= 1, "Attractor strength must be in [0, 1]"
        # Layer influences should sum to approximately 1.0
        total_influence = sum(basin["layer_influences"].values())
        assert abs(total_influence - 1.0) < 0.01, f"Layer influences should sum to ~1.0, got {total_influence}"

    # Validate thoughtseed schema
    if data["thoughtseed"] is not None:
        thoughtseed = data["thoughtseed"]
        assert 0 <= thoughtseed["resonance_score"] <= 1, "Resonance score must be in [0, 1]"
        assert len(thoughtseed["concept_labels"]) > 0, "Thoughtseed must have at least one concept label"


@pytest.mark.asyncio
async def test_get_document_citations_multiple_chunks(test_client: AsyncClient, mock_chunk_data, mock_basin_data, mock_thoughtseed_data):
    """
    Test retrieving citations for different chunks from same document.

    Expected: Each chunk returns distinct citation data.
    """
    document_id = "doc_test_001"
    chunk_ids = ["chunk_123", "chunk_456", "chunk_789"]

    responses = []
    for i, chunk_id in enumerate(chunk_ids):
        # Create unique chunk data for each request
        unique_chunk = ChunkCitationData(
            chunk_id=chunk_id,
            chunk_text=f"Unique chunk text {i}",
            chunk_index=i,
            start_offset=i * 100,
            end_offset=(i * 100) + 50,
            created_at=datetime(2025, 10, 8, 12, i, 0)
        )

        mock_response = CitationResponse(
            document_id=document_id,
            chunk=unique_chunk,
            basin=mock_basin_data,
            thoughtseed=mock_thoughtseed_data
        )

        with patch('src.services.citation_service.CitationService.get_citation_data', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response

            response = await test_client.get(
                f"/api/documents/{document_id}/citations",
                params={"chunk_id": chunk_id}
            )
            assert response.status_code == 200
            responses.append(response.json())

    # Each chunk should have unique data
    chunk_texts = [r["chunk"]["chunk_text"] for r in responses]
    assert len(set(chunk_texts)) == len(chunk_texts), "Each chunk should have unique text"

    chunk_indices = [r["chunk"]["chunk_index"] for r in responses]
    assert len(set(chunk_indices)) == len(chunk_indices), "Each chunk should have unique index"
