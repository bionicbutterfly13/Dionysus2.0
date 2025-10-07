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
    request_data = {
        "document_id": "doc_test_001",
        "filename": "test_research.pdf",
        "content_hash": "sha256:abc123def456",
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

    Expected: 409 Conflict with existing document info.
    """
    # First, create a document
    request_data = {
        "document_id": "doc_test_duplicate",
        "filename": "duplicate.pdf",
        "content_hash": "sha256:duplicate123",
        "file_size": 1024,
        "mime_type": "application/pdf",
        "tags": ["duplicate"],
        "daedalus_output": {
            "quality": {"scores": {"overall": 0.8}},
            "concepts": {"atomic": []},
            "basins": [],
            "thoughtseeds": [],
            "research": {"curiosity_triggers": 0, "research_questions": 0}
        }
    }

    # First persist should succeed
    response1 = await test_client.post("/api/documents/persist", json=request_data)
    assert response1.status_code == 201

    # Second persist with same content_hash should fail with 409
    request_data["document_id"] = "doc_test_duplicate_2"  # Different doc_id, same content_hash
    response2 = await test_client.post("/api/documents/persist", json=request_data)

    assert response2.status_code == 409, f"Expected 409 Conflict, got {response2.status_code}"

    response_json = response2.json()
    # FastAPI wraps custom detail in "detail" field
    data = response_json.get("detail", response_json)
    assert data["status"] == "duplicate"
    assert data["content_hash"] == "sha256:duplicate123"
    assert "existing_document" in data
    assert "options" in data


@pytest.mark.asyncio
async def test_persist_document_missing_fields(test_client: AsyncClient):
    """
    Test missing required fields returns 400 Bad Request.

    Expected: 400 Bad Request with validation error.
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
async def test_persist_document_performance_target(test_client: AsyncClient):
    """
    Test persistence completes within 2 second target.

    Expected: Performance metrics show met_target=True.
    """
    import time

    request_data = {
        "document_id": "doc_test_performance",
        "filename": "performance_test.pdf",
        "content_hash": "sha256:perf123",
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
