#!/usr/bin/env python3
"""
T008: Contract Test - POST /api/clause/curate

Tests ContextCurator API contract per Spec 035.
EXPECTED: ALL TESTS MUST FAIL (endpoint not implemented yet)

Contract: specs/035-clause-phase2-multi-agent/contracts/curator_api.yaml
Data Model: specs/035-clause-phase2-multi-agent/data-model.md
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """FastAPI test client with CLAUSE routes for contract testing"""
    from fastapi import FastAPI
    import sys
    sys.path.insert(0, '/Volumes/Asylum/dev/Dionysus-2.0/backend/src')

    # Import CLAUSE router
    from api.routes import clause

    # Create minimal app with CLAUSE routes
    app = FastAPI()
    app.include_router(clause.router)

    return TestClient(app)


class TestCuratorContract:
    """Contract tests for ContextCurator API"""

    def test_curate_endpoint_exists(self, client):
        """Verify POST /api/clause/curate endpoint exists"""
        response = client.post("/api/clause/curate", json={
            "evidence_pool": ["Greenhouse gases trap heat in the atmosphere"],
            "token_budget": 2048
        })

        # Should not be 404 when endpoint exists
        assert response.status_code != 404, "Endpoint POST /api/clause/curate does not exist"

    def test_curate_requires_evidence_pool(self, client):
        """Verify evidence_pool field is required"""
        response = client.post("/api/clause/curate", json={
            "token_budget": 2048
        })

        assert response.status_code == 422, "Should reject missing evidence_pool field"
        error_detail = response.json()
        assert "evidence_pool" in str(error_detail).lower()

    def test_curate_evidence_pool_min_length(self, client):
        """Verify evidence_pool has min_length=1 validation"""
        response = client.post("/api/clause/curate", json={
            "evidence_pool": [],  # Empty array
            "token_budget": 2048
        })

        assert response.status_code == 422, "Should reject empty evidence_pool"

    def test_curate_token_budget_range(self, client):
        """Verify token_budget range validation (100-8192)"""
        # Test below minimum
        response = client.post("/api/clause/curate", json={
            "evidence_pool": ["Greenhouse gases trap heat"],
            "token_budget": 50
        })
        assert response.status_code == 422, "Should reject token_budget < 100"

        # Test above maximum
        response = client.post("/api/clause/curate", json={
            "evidence_pool": ["Greenhouse gases trap heat"],
            "token_budget": 10000
        })
        assert response.status_code == 422, "Should reject token_budget > 8192"

    def test_curate_default_token_budget(self, client):
        """Verify token_budget defaults to 2048"""
        response = client.post("/api/clause/curate", json={
            "evidence_pool": ["Greenhouse gases trap heat"]
        })

        # Should succeed (assuming endpoint returns 200 when implemented)
        if response.status_code == 200:
            data = response.json()
            assert "metadata" in data
            assert data["metadata"]["tokens_total"] == 2048, "Default token_budget should be 2048"

    def test_curate_lambda_tok_range(self, client):
        """Verify lambda_tok range validation (0.0-1.0)"""
        response = client.post("/api/clause/curate", json={
            "evidence_pool": ["Greenhouse gases trap heat"],
            "token_budget": 2048,
            "lambda_tok": 1.5  # Above max
        })

        assert response.status_code == 422, "Should reject lambda_tok > 1.0"

        response = client.post("/api/clause/curate", json={
            "evidence_pool": ["Greenhouse gases trap heat"],
            "token_budget": 2048,
            "lambda_tok": -0.1  # Below min
        })

        assert response.status_code == 422, "Should reject lambda_tok < 0.0"

    def test_curate_enable_provenance_default(self, client):
        """Verify enable_provenance defaults to True"""
        response = client.post("/api/clause/curate", json={
            "evidence_pool": ["Greenhouse gases trap heat"],
            "token_budget": 2048
        })

        # When implemented, should use default
        if response.status_code == 200:
            data = response.json()
            # Each evidence should have provenance metadata
            if data["selected_evidence"]:
                assert "provenance" in data["selected_evidence"][0]

    def test_curate_response_schema(self, client):
        """Verify response has selected_evidence, metadata, performance fields"""
        response = client.post("/api/clause/curate", json={
            "evidence_pool": ["Greenhouse gases trap heat in the atmosphere"],
            "token_budget": 2048,
            "enable_provenance": True
        })

        if response.status_code == 200:
            data = response.json()

            # Required top-level fields
            assert "selected_evidence" in data, "Response missing 'selected_evidence' field"
            assert "metadata" in data, "Response missing 'metadata' field"
            assert "performance" in data, "Response missing 'performance' field"

            # selected_evidence is array of SelectedEvidence
            assert isinstance(data["selected_evidence"], list)

            # Metadata structure
            assert "tokens_used" in data["metadata"]
            assert "tokens_total" in data["metadata"]

            # Performance structure
            assert "latency_ms" in data["performance"]

    def test_curate_evidence_structure(self, client):
        """Verify SelectedEvidence has required fields"""
        response = client.post("/api/clause/curate", json={
            "evidence_pool": ["Greenhouse gases trap heat"],
            "token_budget": 2048,
            "enable_provenance": True
        })

        if response.status_code == 200:
            data = response.json()

            if data["selected_evidence"]:
                evidence = data["selected_evidence"][0]

                # Required fields per SelectedEvidence model
                assert "text" in evidence
                assert "tokens" in evidence
                assert "score" in evidence
                assert "shaped_utility" in evidence
                assert "provenance" in evidence

                # Provenance structure (7 fields from Spec 032)
                prov = evidence["provenance"]
                assert "source_uri" in prov
                assert "extraction_timestamp" in prov
                assert "extractor_identity" in prov
                assert "supporting_evidence" in prov
                assert "verification_status" in prov
                assert "corroboration_count" in prov
                assert "trust_signals" in prov

    def test_curate_provenance_trust_signals(self, client):
        """Verify TrustSignals structure in provenance"""
        response = client.post("/api/clause/curate", json={
            "evidence_pool": ["Greenhouse gases trap heat"],
            "token_budget": 2048,
            "enable_provenance": True
        })

        if response.status_code == 200:
            data = response.json()

            if data["selected_evidence"]:
                trust = data["selected_evidence"][0]["provenance"]["trust_signals"]

                # 3 trust signal scores per Spec 032
                assert "reputation_score" in trust
                assert "recency_score" in trust
                assert "semantic_consistency" in trust

                # All scores in [0.0, 1.0]
                assert 0.0 <= trust["reputation_score"] <= 1.0
                assert 0.0 <= trust["recency_score"] <= 1.0
                assert 0.0 <= trust["semantic_consistency"] <= 1.0

    def test_curate_budget_compliance(self, client):
        """Verify response respects token_budget"""
        response = client.post("/api/clause/curate", json={
            "evidence_pool": [
                "Greenhouse gases trap heat in the atmosphere",
                "CO2 is the primary greenhouse gas from human activity"
            ],
            "token_budget": 500
        })

        if response.status_code == 200:
            data = response.json()
            tokens_used = data["metadata"]["tokens_used"]
            tokens_total = data["metadata"]["tokens_total"]

            assert tokens_total == 500, "tokens_total should match request"
            assert tokens_used <= tokens_total, "tokens_used must not exceed tokens_total"

    def test_curate_verification_status_enum(self, client):
        """Verify verification_status is one of: verified, pending_review, unverified"""
        response = client.post("/api/clause/curate", json={
            "evidence_pool": ["Greenhouse gases trap heat"],
            "token_budget": 2048,
            "enable_provenance": True
        })

        if response.status_code == 200:
            data = response.json()

            if data["selected_evidence"]:
                status = data["selected_evidence"][0]["provenance"]["verification_status"]
                assert status in ["verified", "pending_review", "unverified"]


if __name__ == "__main__":
    print("\n=== T008: ContextCurator Contract Test ===\n")
    print("⚠️  EXPECTED: All tests MUST FAIL (endpoint not implemented)")
    print("Contract: POST /api/clause/curate\n")

    pytest.main([__file__, "-v", "--tb=short"])
