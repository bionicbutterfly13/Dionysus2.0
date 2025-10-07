#!/usr/bin/env python3
"""
Contract Test: GET /api/documents/{id} - Spec 054 T011

Tests the document detail endpoint contract.

CRITICAL: This test MUST FAIL before implementation (TDD approach).
Only passes after T043 implementation is complete.

Author: Spec 054 Implementation
Created: 2025-10-07
"""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_get_document_detail_success(test_client: AsyncClient):
    """
    Test successful document detail retrieval.

    Expected: 200 OK with full document detail including all artifacts.
    """
    # Assume a document with this ID exists
    document_id = "doc_test_001"

    response = await test_client.get(f"/api/documents/{document_id}")

    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}"

    data = response.json()

    # Core fields
    assert data["document_id"] == document_id

    # Metadata section
    assert "metadata" in data
    metadata = data["metadata"]
    assert "filename" in metadata
    assert "upload_timestamp" in metadata
    assert "file_size" in metadata
    assert "mime_type" in metadata
    assert "tags" in metadata
    assert "tier" in metadata
    assert "last_accessed" in metadata
    assert "access_count" in metadata

    # Quality metrics
    assert "quality" in data
    quality = data["quality"]
    assert "overall" in quality
    assert "coherence" in quality
    assert "novelty" in quality
    assert "depth" in quality

    # Concepts (all 5 levels)
    assert "concepts" in data
    concepts = data["concepts"]
    assert "atomic" in concepts
    assert "relationship" in concepts
    assert "composite" in concepts
    assert "context" in concepts
    assert "narrative" in concepts
    assert isinstance(concepts["atomic"], list)

    # Attractor basins
    assert "basins" in data
    assert isinstance(data["basins"], list)
    if len(data["basins"]) > 0:
        basin = data["basins"][0]
        assert "basin_id" in basin
        assert "name" in basin
        assert "depth" in basin
        assert "stability" in basin
        assert "activation_strength" in basin

    # ThoughtSeeds
    assert "thoughtseeds" in data
    assert isinstance(data["thoughtseeds"], list)
    if len(data["thoughtseeds"]) > 0:
        seed = data["thoughtseeds"][0]
        assert "seed_id" in seed
        assert "content" in seed
        assert "germination_potential" in seed
        assert "resonance_score" in seed

    # Processing timeline
    assert "processing_timeline" in data


@pytest.mark.asyncio
async def test_get_document_not_found(test_client: AsyncClient):
    """
    Test requesting non-existent document returns 404.

    Expected: 404 Not Found.
    """
    non_existent_id = "doc_does_not_exist_12345"

    response = await test_client.get(f"/api/documents/{non_existent_id}")

    assert response.status_code == 404, f"Expected 404 Not Found, got {response.status_code}"

    data = response.json()
    assert "detail" in data or "error" in data


@pytest.mark.asyncio
async def test_get_document_access_tracking(test_client: AsyncClient):
    """
    Test that accessing a document increments access_count.

    Expected: access_count increases after each GET.
    """
    document_id = "doc_test_access_tracking"

    # First access
    response1 = await test_client.get(f"/api/documents/{document_id}")
    assert response1.status_code == 200
    access_count_1 = response1.json()["metadata"]["access_count"]

    # Second access
    response2 = await test_client.get(f"/api/documents/{document_id}")
    assert response2.status_code == 200
    access_count_2 = response2.json()["metadata"]["access_count"]

    # Access count should increase
    assert access_count_2 > access_count_1, \
        f"Access count should increase: {access_count_1} -> {access_count_2}"


@pytest.mark.asyncio
async def test_get_document_includes_all_concept_levels(test_client: AsyncClient):
    """
    Test that document detail includes concepts from all 5 levels.

    Expected: All 5 concept levels present in response.
    """
    document_id = "doc_test_concepts"

    response = await test_client.get(f"/api/documents/{document_id}")
    assert response.status_code == 200

    concepts = response.json()["concepts"]

    # Verify all 5 levels exist as keys
    required_levels = ["atomic", "relationship", "composite", "context", "narrative"]
    for level in required_levels:
        assert level in concepts, f"Missing concept level: {level}"
        assert isinstance(concepts[level], list), f"Concept level {level} should be a list"


@pytest.mark.asyncio
async def test_get_document_basin_activation_data(test_client: AsyncClient):
    """
    Test that basin data includes activation strength and influence type.

    Expected: Basin objects have activation_strength and influence_type.
    """
    document_id = "doc_test_basins"

    response = await test_client.get(f"/api/documents/{document_id}")
    assert response.status_code == 200

    basins = response.json()["basins"]

    if len(basins) > 0:
        for basin in basins:
            # From plan.md: basin activation includes these fields
            assert "activation_strength" in basin, "Missing activation_strength"
            assert "influence_type" in basin, "Missing influence_type"
            assert basin["influence_type"] in [
                "reinforcement", "competition", "synthesis", "emergence"
            ], f"Invalid influence_type: {basin['influence_type']}"


@pytest.mark.asyncio
async def test_get_document_thoughtseed_resonance(test_client: AsyncClient):
    """
    Test that thoughtseed data includes germination_potential and resonance_score.

    Expected: ThoughtSeed objects have germination_potential and resonance_score.
    """
    document_id = "doc_test_seeds"

    response = await test_client.get(f"/api/documents/{document_id}")
    assert response.status_code == 200

    thoughtseeds = response.json()["thoughtseeds"]

    if len(thoughtseeds) > 0:
        for seed in thoughtseeds:
            assert "germination_potential" in seed, "Missing germination_potential"
            assert "resonance_score" in seed, "Missing resonance_score"
            assert 0.0 <= seed["germination_potential"] <= 1.0
            assert 0.0 <= seed["resonance_score"] <= 1.0


@pytest.mark.asyncio
async def test_get_document_performance_target(test_client: AsyncClient):
    """
    Test document detail retrieval meets <200ms target.

    Expected: Response time < 200ms.
    """
    import time

    document_id = "doc_test_performance"

    start = time.time()
    response = await test_client.get(f"/api/documents/{document_id}")
    duration_ms = (time.time() - start) * 1000

    assert response.status_code == 200
    assert duration_ms < 200, f"Detail retrieval took {duration_ms:.0f}ms, target is <200ms"


@pytest.mark.asyncio
async def test_get_document_tier_information(test_client: AsyncClient):
    """
    Test that document includes tier information.

    Expected: Metadata includes tier field (warm/cool/cold).
    """
    document_id = "doc_test_tier"

    response = await test_client.get(f"/api/documents/{document_id}")
    assert response.status_code == 200

    metadata = response.json()["metadata"]
    assert "tier" in metadata
    assert metadata["tier"] in ["warm", "cool", "cold"], \
        f"Invalid tier: {metadata['tier']}"
