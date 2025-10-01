"""
Contract Test: POST /api/query Endpoint
Per Spec 006 FR-001, FR-002, FR-003

Tests MUST fail initially - TDD RED phase
"""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime


class TestQueryAPIContract:
    """Contract tests for query API endpoint"""

    def test_query_endpoint_exists(self, client: TestClient):
        """Test that POST /api/query endpoint exists"""
        response = client.post("/api/query", json={"question": "test question"})
        assert response.status_code != 404, "Query endpoint must exist"

    def test_query_accepts_natural_language_question(self, client: TestClient):
        """FR-001: System MUST accept natural language questions"""
        question = "What neural architectures show consciousness emergence?"

        response = client.post("/api/query", json={"question": question})

        assert response.status_code in [200, 201], "Valid question must be accepted"
        data = response.json()
        assert "response_id" in data, "Response must have ID"
        assert "answer" in data, "Response must have answer field"

    def test_query_returns_synthesized_response(self, client: TestClient):
        """FR-003: System MUST synthesize results into coherent responses"""
        question = "Explain active inference"

        response = client.post("/api/query", json={"question": question})

        assert response.status_code == 200
        data = response.json()
        assert data["answer"], "Answer must not be empty"
        assert isinstance(data["answer"], str), "Answer must be string"
        assert len(data["answer"]) > 10, "Answer must be substantive"

    def test_query_includes_sources(self, client: TestClient):
        """FR-002: System MUST search both Neo4j and vector databases"""
        question = "Find ThoughtSeed examples"

        response = client.post("/api/query", json={"question": question})

        assert response.status_code == 200
        data = response.json()
        assert "sources" in data, "Response must include sources"
        assert isinstance(data["sources"], list), "Sources must be list"

    def test_query_includes_confidence(self, client: TestClient):
        """FR-006: System MUST indicate confidence levels"""
        question = "What is consciousness?"

        response = client.post("/api/query", json={"question": question})

        assert response.status_code == 200
        data = response.json()
        assert "confidence" in data, "Response must include confidence"
        assert 0.0 <= data["confidence"] <= 1.0, "Confidence must be in [0,1]"

    def test_query_includes_processing_time(self, client: TestClient):
        """FR-012: System MUST provide response within acceptable limits"""
        question = "Simple query"

        response = client.post("/api/query", json={"question": question})

        assert response.status_code == 200
        data = response.json()
        assert "processing_time_ms" in data, "Must report processing time"
        assert isinstance(data["processing_time_ms"], (int, float))
        assert data["processing_time_ms"] < 5000, "Must respond within 5 seconds"

    def test_query_rejects_empty_question(self, client: TestClient):
        """Test validation: empty questions must be rejected"""
        response = client.post("/api/query", json={"question": ""})

        assert response.status_code == 400, "Empty question must be rejected"
        data = response.json()
        assert "detail" in data or "error" in data, "Error message required"

    def test_query_rejects_missing_question(self, client: TestClient):
        """Test validation: missing question field must be rejected"""
        response = client.post("/api/query", json={})

        assert response.status_code == 400, "Missing question must be rejected"

    def test_query_accepts_optional_context(self, client: TestClient):
        """FR-008: Users MUST be able to follow up with clarifying questions"""
        question = "Continue previous discussion"
        context = {"previous_query_id": "test-123"}

        response = client.post("/api/query", json={
            "question": question,
            "context": context
        })

        assert response.status_code == 200, "Context should be optional but accepted"

    def test_query_handles_database_unavailable(self, client: TestClient):
        """Edge case: Database connection failure"""
        # This test will pass once error handling is implemented
        # For now, it documents expected behavior
        pass  # Placeholder for database failure test

    def test_query_response_schema_complete(self, client: TestClient):
        """Test complete response schema"""
        question = "Test all fields"

        response = client.post("/api/query", json={"question": question})

        assert response.status_code == 200
        data = response.json()

        # Required fields per spec
        required_fields = ["response_id", "answer", "sources", "confidence", "processing_time_ms"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

    def test_concurrent_queries_performance(self, client: TestClient):
        """FR-009: System MUST handle concurrent user queries efficiently"""
        import concurrent.futures

        def send_query(n):
            return client.post("/api/query", json={"question": f"Query {n}"})

        # Test 10 concurrent queries per spec
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(send_query, i) for i in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All should succeed
        for result in results:
            assert result.status_code == 200, "Concurrent queries must all succeed"


@pytest.fixture
def client():
    """Test client fixture - will be implemented when app exists"""
    # This will fail initially - part of TDD RED phase
    from backend.src.app_factory import create_app

    app = create_app()
    return TestClient(app)
