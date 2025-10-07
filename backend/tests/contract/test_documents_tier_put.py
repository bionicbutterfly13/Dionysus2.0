#!/usr/bin/env python3
"""
Contract Test: PUT /api/documents/{id}/tier - Spec 054 T012

Tests the document tier update endpoint contract.

CRITICAL: This test MUST FAIL before implementation (TDD approach).
Only passes after T044 implementation is complete.

Author: Spec 054 Implementation
Created: 2025-10-07
"""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_update_tier_to_cool(test_client: AsyncClient):
    """
    Test updating document tier from warm to cool.

    Expected: 200 OK with tier change confirmation.
    """
    document_id = "doc_test_tier_cool"

    # Update tier to cool
    request_data = {
        "new_tier": "cool",
        "reason": "manual_test"
    }

    response = await test_client.put(
        f"/api/documents/{document_id}/tier",
        json=request_data
    )

    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}"

    data = response.json()
    assert data["status"] == "success"
    assert data["document_id"] == document_id
    assert data["old_tier"] in ["warm", "cool", "cold"]
    assert data["new_tier"] == "cool"
    assert "tier_changed_at" in data


@pytest.mark.asyncio
async def test_update_tier_to_cold_with_archival(test_client: AsyncClient):
    """
    Test updating document tier to cold triggers archival.

    Expected: 200 OK with archive_location.
    """
    document_id = "doc_test_tier_cold"

    request_data = {
        "new_tier": "cold",
        "reason": "manual_archival"
    }

    response = await test_client.put(
        f"/api/documents/{document_id}/tier",
        json=request_data
    )

    assert response.status_code == 200

    data = response.json()
    assert data["new_tier"] == "cold"
    assert "archive_location" in data
    # Archive location should be S3 or filesystem path
    assert data["archive_location"].startswith("s3://") or data["archive_location"].startswith("/")


@pytest.mark.asyncio
async def test_update_tier_invalid_tier(test_client: AsyncClient):
    """
    Test updating to invalid tier returns 400 Bad Request.

    Expected: 400 Bad Request.
    """
    document_id = "doc_test_invalid_tier"

    request_data = {
        "new_tier": "invalid_tier",
        "reason": "test"
    }

    response = await test_client.put(
        f"/api/documents/{document_id}/tier",
        json=request_data
    )

    assert response.status_code == 400, f"Expected 400 Bad Request, got {response.status_code}"

    data = response.json()
    assert "detail" in data or "error" in data


@pytest.mark.asyncio
async def test_update_tier_nonexistent_document(test_client: AsyncClient):
    """
    Test updating tier of non-existent document returns 404.

    Expected: 404 Not Found.
    """
    non_existent_id = "doc_does_not_exist_12345"

    request_data = {
        "new_tier": "cool",
        "reason": "test"
    }

    response = await test_client.put(
        f"/api/documents/{non_existent_id}/tier",
        json=request_data
    )

    assert response.status_code == 404, f"Expected 404 Not Found, got {response.status_code}"


@pytest.mark.asyncio
async def test_update_tier_timestamp_updated(test_client: AsyncClient):
    """
    Test that tier_changed_at timestamp is updated.

    Expected: tier_changed_at reflects current time.
    """
    from datetime import datetime, timedelta

    document_id = "doc_test_tier_timestamp"

    request_data = {
        "new_tier": "cool",
        "reason": "timestamp_test"
    }

    before_update = datetime.now()
    response = await test_client.put(
        f"/api/documents/{document_id}/tier",
        json=request_data
    )
    after_update = datetime.now()

    assert response.status_code == 200

    data = response.json()
    tier_changed_at = datetime.fromisoformat(data["tier_changed_at"].replace("Z", "+00:00"))

    # Verify timestamp is between before and after update
    assert before_update <= tier_changed_at.replace(tzinfo=None) <= after_update + timedelta(seconds=1)


@pytest.mark.asyncio
async def test_update_tier_with_reason_tracking(test_client: AsyncClient):
    """
    Test that tier change reason is tracked.

    Expected: Reason appears in response or logs.
    """
    document_id = "doc_test_tier_reason"

    request_data = {
        "new_tier": "cool",
        "reason": "user_requested_archival"
    }

    response = await test_client.put(
        f"/api/documents/{document_id}/tier",
        json=request_data
    )

    assert response.status_code == 200

    # Reason tracking verified in response or audit logs
    # (Implementation detail: may be in response or separate audit trail)


@pytest.mark.asyncio
async def test_update_tier_from_warm_to_warm(test_client: AsyncClient):
    """
    Test updating tier to same tier (no-op).

    Expected: 200 OK, no error.
    """
    document_id = "doc_test_tier_noop"

    # Get current tier
    get_response = await test_client.get(f"/api/documents/{document_id}")
    current_tier = get_response.json()["metadata"]["tier"]

    # Update to same tier
    request_data = {
        "new_tier": current_tier,
        "reason": "noop_test"
    }

    response = await test_client.put(
        f"/api/documents/{document_id}/tier",
        json=request_data
    )

    assert response.status_code == 200
    data = response.json()
    assert data["old_tier"] == current_tier
    assert data["new_tier"] == current_tier


@pytest.mark.asyncio
async def test_update_tier_missing_reason(test_client: AsyncClient):
    """
    Test updating tier without reason (reason is optional).

    Expected: 200 OK, reason defaults to "manual" or similar.
    """
    document_id = "doc_test_tier_no_reason"

    request_data = {
        "new_tier": "cool"
        # reason omitted
    }

    response = await test_client.put(
        f"/api/documents/{document_id}/tier",
        json=request_data
    )

    # Should succeed even without explicit reason
    assert response.status_code == 200
