#!/usr/bin/env python3
"""
Contract Test: POST /api/documents/persist - Spec 054 T009

Tests the document persistence endpoint contract.

CRITICAL: This test MUST FAIL before implementation (TDD approach).
Only passes after T041 implementation is complete.

Author: Spec 054 Implementation
Created: 2025-10-07
"""

import pytest
from httpx import AsyncClient
from datetime import datetime


@pytest.mark.asyncio
async def test_persist_document_success(test_client: AsyncClient):
    """
    Test successful document persistence.

    Expected: 201 Created with document_id and performance metrics.
    """
    # Prepare test data (mock Daedalus final_output)
    # Note: content_hash must be valid 64-char SHA-256 hex (no prefix)
    request_data = {
        "document_id": "doc_test_001",
        "filename": "test_research.pdf",
        "content_hash": "a" * 64,  # Valid 64-char SHA-256 format
        "file_size": 1048576,
        "mime_type": "application/pdf",
        "tags": ["test", "research"],
        "daedalus_output": {
            "quality": {
                "scores": {
                    "overall": 0.85,
                    "coherence": 0.90,
                    "novelty": 0.75,
                    "depth": 0.88
                }
            },
            "concepts": {
                "atomic": [
                    {"concept_id": "c001", "name": "active_inference", "salience": 0.95}
                ],
                "relationship": [],
                "composite": [],
                "context": [],
                "narrative": []
            },
            "basins": [
                {
                    "basin_id": "b001",
                    "name": "consciousness_dynamics",
                    "depth": 0.75,
                    "stability": 0.88,
                    "influence_type": "reinforcement",
                    "strength_delta": 0.10
                }
            ],
            "thoughtseeds": [
                {
                    "seed_id": "s001",
                    "content": "How does active inference relate to consciousness?",
                    "germination_potential": 0.92,
                    "resonance_score": 0.85
                }
            ],
            "research": {
                "curiosity_triggers": 5,
                "research_questions": 3
            }
        }
    }

    # Make request
    response = await test_client.post("/api/documents/persist", json=request_data)

    # Assert response
    assert response.status_code == 201, f"Expected 201 Created, got {response.status_code}"

    data = response.json()
    assert data["status"] == "success"
    assert data["document_id"] == "doc_test_001"
    assert "persisted_at" in data
    assert data["tier"] == "warm"
    assert data["nodes_created"] > 0
    assert data["relationships_created"] > 0

    # Performance assertion
    assert "performance" in data
    assert data["performance"]["persistence_duration_ms"] < 2000  # <2s target
    assert data["performance"]["met_target"] is True


@pytest.mark.asyncio
async def test_persist_document_duplicate_conflict(test_client: AsyncClient):
    """
    Test duplicate document detection returns 409 Conflict.

    Spec 055 Agent 2: Verifies structured 409 response with canonical document metadata.

    Expected: 409 Conflict with complete canonical document info and reuse guidance.
    """
    # First, create a document
    # Note: content_hash must be valid 64-char SHA-256 hex (no prefix)
    request_data = {
        "document_id": "doc_test_duplicate",
        "filename": "duplicate.pdf",
        "content_hash": "b" * 64,  # Valid 64-char SHA-256 format
        "file_size": 1024,
        "mime_type": "application/pdf",
        "tags": ["duplicate", "test"],
        "daedalus_output": {
            "quality": {"scores": {"overall": 0.82}},
            "concepts": {"atomic": []},
            "basins": [],
            "thoughtseeds": [],
            "research": {"curiosity_triggers": 0, "research_questions": 0}
        }
    }

    # First persist should succeed
    response1 = await test_client.post("/api/documents/persist", json=request_data)
    assert response1.status_code == 201, f"Expected 201 Created, got {response1.status_code}"

    # Second persist with same content_hash should fail with 409
    request_data["document_id"] = "doc_test_duplicate_2"  # Different doc_id, same content_hash
    request_data["filename"] = "duplicate_renamed.pdf"  # Different filename
    response2 = await test_client.post("/api/documents/persist", json=request_data)

    assert response2.status_code == 409, f"Expected 409 Conflict, got {response2.status_code}"

    response_json = response2.json()
    # FastAPI wraps custom detail in "detail" field
    data = response_json.get("detail", response_json)

    # Verify structured 409 response
    assert data["status"] == "duplicate", "Missing 'status' field"
    assert data["message"] == "Document with this content already exists", "Missing 'message' field"
    assert data["content_hash"] == "b" * 64, "Missing or incorrect 'content_hash'"

    # Verify canonical_document metadata is complete
    assert "canonical_document" in data, "Missing 'canonical_document' field"
    canonical = data["canonical_document"]
    assert canonical["document_id"] == "doc_test_duplicate", "Incorrect canonical document_id"
    assert canonical["filename"] == "duplicate.pdf", "Incorrect canonical filename"
    assert "upload_timestamp" in canonical, "Missing upload_timestamp"
    assert canonical["quality_overall"] == 0.82, "Incorrect quality_overall"
    assert canonical["tier"] == "warm", "Incorrect tier"
    assert set(canonical["tags"]) == {"duplicate", "test"}, "Incorrect tags"
    assert canonical["file_size"] == 1024, "Incorrect file_size"
    assert "access_count" in canonical, "Missing access_count"

    # Verify reuse_guidance is present
    assert "reuse_guidance" in data, "Missing 'reuse_guidance' field"
    guidance = data["reuse_guidance"]
    assert guidance["action"] == "link_to_existing", "Incorrect reuse action"
    assert guidance["url"] == "/api/documents/doc_test_duplicate", "Incorrect reuse URL"
    assert "message" in guidance, "Missing reuse guidance message"
    assert "instead of re-uploading" in guidance["message"], "Reuse message missing guidance text"


@pytest.mark.asyncio
async def test_persist_document_missing_fields(test_client: AsyncClient):
    """
    Test missing required fields returns 422 Unprocessable Entity.

    Spec 055 Agent 2: Validation errors (422) are distinct from duplicates (409).

    Expected: 422 Unprocessable Entity with validation error.
    """
    # Missing required field: content_hash
    request_data = {
        "document_id": "doc_test_invalid",
        "filename": "invalid.pdf",
        # content_hash missing!
        "file_size": 1024,
        "daedalus_output": {}
    }

    response = await test_client.post("/api/documents/persist", json=request_data)

    # FastAPI returns 422 (Unprocessable Entity) for Pydantic validation errors
    assert response.status_code == 422, f"Expected 422 Unprocessable Entity, got {response.status_code}"

    data = response.json()
    assert "detail" in data  # FastAPI validation error format


@pytest.mark.asyncio
async def test_persist_document_duplicate_with_different_filename(test_client: AsyncClient):
    """
    Test duplicate detection works even if filename differs.

    Spec 055 Agent 2: Content hash is the canonical identifier, not filename.

    Expected: 409 Conflict with original filename in canonical_document.
    """
    # First upload
    request_data_1 = {
        "document_id": "doc_filename_test_1",
        "filename": "original_name.pdf",
        "content_hash": "c" * 64,  # Valid 64-char SHA-256 format
        "file_size": 2048,
        "mime_type": "application/pdf",
        "tags": ["filename-test"],
        "daedalus_output": {
            "quality": {"scores": {"overall": 0.88}},
            "concepts": {"atomic": []},
            "basins": [],
            "thoughtseeds": [],
            "research": {"curiosity_triggers": 2, "research_questions": 1}
        }
    }

    response1 = await test_client.post("/api/documents/persist", json=request_data_1)
    assert response1.status_code == 201

    # Second upload with SAME content_hash but DIFFERENT filename
    request_data_2 = {
        "document_id": "doc_filename_test_2",
        "filename": "renamed_copy.pdf",  # Different filename!
        "content_hash": "c" * 64,  # Same hash!
        "file_size": 2048,
        "mime_type": "application/pdf",
        "tags": ["filename-test"],
        "daedalus_output": {
            "quality": {"scores": {"overall": 0.88}},
            "concepts": {"atomic": []},
            "basins": [],
            "thoughtseeds": [],
            "research": {"curiosity_triggers": 2, "research_questions": 1}
        }
    }

    response2 = await test_client.post("/api/documents/persist", json=request_data_2)
    assert response2.status_code == 409

    data = response2.json()["detail"]
    assert data["status"] == "duplicate"
    # Canonical document should have the ORIGINAL filename
    assert data["canonical_document"]["filename"] == "original_name.pdf"
    assert data["canonical_document"]["document_id"] == "doc_filename_test_1"


@pytest.mark.asyncio
async def test_persist_document_canonical_metadata_completeness(test_client: AsyncClient):
    """
    Test that canonical document metadata includes all required fields.

    Spec 055 Agent 2: Canonical document must be complete enough for reuse decisions.

    Expected: All fields present in canonical_document.
    """
    # Create original document
    request_data = {
        "document_id": "doc_metadata_test",
        "filename": "metadata_test.pdf",
        "content_hash": "d" * 64,  # Valid 64-char SHA-256 format
        "file_size": 4096,
        "mime_type": "application/pdf",
        "tags": ["metadata", "completeness"],
        "daedalus_output": {
            "quality": {"scores": {"overall": 0.91}},
            "concepts": {"atomic": []},
            "basins": [],
            "thoughtseeds": [],
            "research": {"curiosity_triggers": 7, "research_questions": 3}
        }
    }

    response1 = await test_client.post("/api/documents/persist", json=request_data)
    assert response1.status_code == 201

    # Attempt duplicate
    request_data["document_id"] = "doc_metadata_test_2"
    response2 = await test_client.post("/api/documents/persist", json=request_data)
    assert response2.status_code == 409

    canonical = response2.json()["detail"]["canonical_document"]

    # Verify all required fields are present
    required_fields = [
        "document_id", "filename", "upload_timestamp",
        "quality_overall", "tier", "tags", "file_size", "access_count"
    ]

    for field in required_fields:
        assert field in canonical, f"Missing required field: {field}"
        # Ensure field is not None (except access_count can be 0)
        if field != "access_count":
            assert canonical[field] is not None, f"Field {field} is None"


@pytest.mark.asyncio
async def test_persist_document_performance_target(test_client: AsyncClient):
    """
    Test persistence completes within 2 second target.

    Expected: Performance metrics show met_target=True.
    """
    import time

    request_data = {
        "document_id": "doc_test_performance",
        "filename": "performance_test.pdf",
        "content_hash": "e" * 64,  # Valid 64-char SHA-256 format
        "file_size": 5242880,  # 5MB document
        "mime_type": "application/pdf",
        "tags": ["performance"],
        "daedalus_output": {
            "quality": {"scores": {"overall": 0.85}},
            "concepts": {
                "atomic": [{"concept_id": f"c{i}", "name": f"concept_{i}", "salience": 0.8}
                          for i in range(25)]  # 25 concepts
            },
            "basins": [{"basin_id": f"b{i}", "name": f"basin_{i}", "depth": 0.7,
                       "stability": 0.8, "influence_type": "reinforcement", "strength_delta": 0.1}
                      for i in range(5)],  # 5 basins
            "thoughtseeds": [{"seed_id": f"s{i}", "content": f"seed {i}",
                             "germination_potential": 0.9, "resonance_score": 0.8}
                            for i in range(10)],  # 10 seeds
            "research": {"curiosity_triggers": 15, "research_questions": 8}
        }
    }

    start = time.time()
    response = await test_client.post("/api/documents/persist", json=request_data)
    duration_ms = (time.time() - start) * 1000

    assert response.status_code == 201
    assert duration_ms < 2000, f"Persistence took {duration_ms:.0f}ms, target is <2000ms"

    data = response.json()
    assert data["performance"]["met_target"] is True
