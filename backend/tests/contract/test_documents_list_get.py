#!/usr/bin/env python3
"""
Contract Test: GET /api/documents - Spec 054 T010

Tests the document listing endpoint contract with pagination, filtering, sorting.

CRITICAL: This test MUST FAIL before implementation (TDD approach).
Only passes after T042 implementation is complete.

Author: Spec 054 Implementation
Created: 2025-10-07
"""

import pytest
from httpx import AsyncClient
from datetime import datetime, timedelta


@pytest.mark.asyncio
async def test_list_documents_basic_pagination(test_client: AsyncClient):
    """
    Test basic document listing with pagination.

    Expected: 200 OK with documents array and pagination metadata.
    """
    # Assume some documents have been created
    response = await test_client.get("/api/documents?page=1&limit=50")

    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}"

    data = response.json()
    assert "documents" in data
    assert "pagination" in data
    assert isinstance(data["documents"], list)

    # Pagination metadata
    pagination = data["pagination"]
    assert pagination["page"] == 1
    assert pagination["limit"] == 50
    assert "total" in pagination
    assert "total_pages" in pagination

    # Performance check
    assert "performance" in data
    assert data["performance"]["query_duration_ms"] < 500  # <500ms target for 100 docs


@pytest.mark.asyncio
async def test_list_documents_filter_by_tags(test_client: AsyncClient):
    """
    Test filtering documents by tags.

    Expected: Only documents with specified tags returned.
    """
    response = await test_client.get("/api/documents?tags=research,ai&page=1&limit=50")

    assert response.status_code == 200
    data = response.json()

    # Verify all returned documents have at least one of the specified tags
    for doc in data["documents"]:
        tags = doc.get("tags", [])
        assert any(tag in ["research", "ai"] for tag in tags), \
            f"Document {doc['document_id']} missing required tags"


@pytest.mark.asyncio
async def test_list_documents_filter_by_quality(test_client: AsyncClient):
    """
    Test filtering documents by minimum quality score.

    Expected: Only documents with quality >= threshold returned.
    """
    quality_threshold = 0.8

    response = await test_client.get(f"/api/documents?quality_min={quality_threshold}")

    assert response.status_code == 200
    data = response.json()

    # Verify all returned documents meet quality threshold
    for doc in data["documents"]:
        assert doc["quality_overall"] >= quality_threshold, \
            f"Document {doc['document_id']} quality {doc['quality_overall']} below threshold {quality_threshold}"


@pytest.mark.asyncio
async def test_list_documents_filter_by_date_range(test_client: AsyncClient):
    """
    Test filtering documents by upload date range.

    Expected: Only documents within date range returned.
    """
    date_from = (datetime.now() - timedelta(days=30)).isoformat()
    date_to = datetime.now().isoformat()

    response = await test_client.get(
        f"/api/documents?date_from={date_from}&date_to={date_to}"
    )

    assert response.status_code == 200
    data = response.json()

    # Verify all returned documents are within date range
    for doc in data["documents"]:
        upload_time = datetime.fromisoformat(doc["upload_timestamp"].replace("Z", "+00:00"))
        date_from_dt = datetime.fromisoformat(date_from.replace("Z", "+00:00"))
        date_to_dt = datetime.fromisoformat(date_to.replace("Z", "+00:00"))

        assert date_from_dt <= upload_time <= date_to_dt, \
            f"Document {doc['document_id']} outside date range"


@pytest.mark.asyncio
async def test_list_documents_sort_by_quality(test_client: AsyncClient):
    """
    Test sorting documents by quality score (descending).

    Expected: Documents sorted by quality_overall in descending order.
    """
    response = await test_client.get("/api/documents?sort=quality&order=desc&limit=10")

    assert response.status_code == 200
    data = response.json()

    documents = data["documents"]
    if len(documents) > 1:
        # Verify descending order
        for i in range(len(documents) - 1):
            assert documents[i]["quality_overall"] >= documents[i + 1]["quality_overall"], \
                "Documents not sorted by quality in descending order"


@pytest.mark.asyncio
async def test_list_documents_sort_by_upload_date(test_client: AsyncClient):
    """
    Test sorting documents by upload date (most recent first).

    Expected: Documents sorted by upload_timestamp in descending order.
    """
    response = await test_client.get("/api/documents?sort=upload_date&order=desc&limit=10")

    assert response.status_code == 200
    data = response.json()

    documents = data["documents"]
    if len(documents) > 1:
        # Verify descending order
        for i in range(len(documents) - 1):
            time1 = datetime.fromisoformat(documents[i]["upload_timestamp"].replace("Z", "+00:00"))
            time2 = datetime.fromisoformat(documents[i + 1]["upload_timestamp"].replace("Z", "+00:00"))
            assert time1 >= time2, "Documents not sorted by upload date in descending order"


@pytest.mark.asyncio
async def test_list_documents_sort_by_curiosity(test_client: AsyncClient):
    """
    Test sorting documents by curiosity triggers.

    Expected: Documents sorted by curiosity_triggers count.
    """
    response = await test_client.get("/api/documents?sort=curiosity&order=desc&limit=10")

    assert response.status_code == 200
    data = response.json()

    documents = data["documents"]
    if len(documents) > 1:
        # Verify descending order
        for i in range(len(documents) - 1):
            assert documents[i]["curiosity_triggers"] >= documents[i + 1]["curiosity_triggers"], \
                "Documents not sorted by curiosity triggers in descending order"


@pytest.mark.asyncio
async def test_list_documents_filter_by_tier(test_client: AsyncClient):
    """
    Test filtering documents by storage tier.

    Expected: Only documents in specified tier returned.
    """
    response = await test_client.get("/api/documents?tier=warm")

    assert response.status_code == 200
    data = response.json()

    # Verify all returned documents are in warm tier
    for doc in data["documents"]:
        assert doc["tier"] == "warm", \
            f"Document {doc['document_id']} is in {doc['tier']} tier, expected warm"


@pytest.mark.asyncio
async def test_list_documents_combined_filters(test_client: AsyncClient):
    """
    Test combining multiple filters.

    Expected: Documents match ALL filter criteria.
    """
    response = await test_client.get(
        "/api/documents?tags=research&quality_min=0.7&tier=warm&sort=quality&order=desc"
    )

    assert response.status_code == 200
    data = response.json()

    for doc in data["documents"]:
        # Verify tags
        assert "research" in doc["tags"]
        # Verify quality
        assert doc["quality_overall"] >= 0.7
        # Verify tier
        assert doc["tier"] == "warm"


@pytest.mark.asyncio
async def test_list_documents_performance_target(test_client: AsyncClient):
    """
    Test listing performance meets <500ms target for 100 documents.

    Expected: Performance metrics show met_target=True.
    """
    import time

    start = time.time()
    response = await test_client.get("/api/documents?page=1&limit=100")
    duration_ms = (time.time() - start) * 1000

    assert response.status_code == 200
    assert duration_ms < 500, f"Listing took {duration_ms:.0f}ms, target is <500ms"

    data = response.json()
    assert data["performance"]["met_target"] is True


@pytest.mark.asyncio
async def test_list_documents_includes_counts(test_client: AsyncClient):
    """
    Test that document listing includes artifact counts.

    Expected: Each document includes concept_count, basin_count, thoughtseed_count.
    """
    response = await test_client.get("/api/documents?page=1&limit=10")

    assert response.status_code == 200
    data = response.json()

    for doc in data["documents"]:
        assert "concept_count" in doc, "Missing concept_count"
        assert "basin_count" in doc, "Missing basin_count"
        assert "thoughtseed_count" in doc, "Missing thoughtseed_count"
        assert isinstance(doc["concept_count"], int)
        assert isinstance(doc["basin_count"], int)
        assert isinstance(doc["thoughtseed_count"], int)
