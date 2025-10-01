"""
Contract Test: Query Response Schema Validation
Per Spec 006 data model specifications

Tests MUST fail initially - TDD RED phase
"""

import pytest
from pydantic import ValidationError
from datetime import datetime


class TestQueryResponseSchema:
    """Contract tests for Query and Response data models"""

    def test_query_model_exists(self):
        """Test that Query model can be imported"""
        from backend.src.models.query import Query

        assert Query is not None, "Query model must exist"

    def test_query_model_has_required_fields(self):
        """Test Query model has all required fields per spec"""
        from backend.src.models.query import Query

        # This should work with minimal required fields
        query = Query(
            query_id="test-123",
            question="What is active inference?",
            timestamp=datetime.now()
        )

        assert query.query_id == "test-123"
        assert query.question == "What is active inference?"
        assert isinstance(query.timestamp, datetime)

    def test_query_model_optional_fields(self):
        """Test Query model optional fields"""
        from backend.src.models.query import Query

        query = Query(
            query_id="test-456",
            question="Test question",
            timestamp=datetime.now(),
            user_id="user-123",
            context={"previous": "context"},
            thoughtseed_id="seed-789"
        )

        assert query.user_id == "user-123"
        assert query.context == {"previous": "context"}
        assert query.thoughtseed_id == "seed-789"

    def test_query_model_validates_empty_question(self):
        """Test that empty questions are rejected"""
        from backend.src.models.query import Query

        with pytest.raises(ValidationError):
            Query(
                query_id="test",
                question="",  # Empty should fail validation
                timestamp=datetime.now()
            )

    def test_response_model_exists(self):
        """Test that Response model can be imported"""
        from backend.src.models.response import QueryResponse

        assert QueryResponse is not None, "QueryResponse model must exist"

    def test_response_model_has_required_fields(self):
        """Test Response model has all required fields per spec"""
        from backend.src.models.response import QueryResponse

        response = QueryResponse(
            response_id="resp-123",
            query_id="query-456",
            answer="This is the synthesized answer",
            sources=[],
            confidence=0.85,
            processing_time_ms=1234
        )

        assert response.response_id == "resp-123"
        assert response.query_id == "query-456"
        assert response.answer == "This is the synthesized answer"
        assert response.sources == []
        assert response.confidence == 0.85
        assert response.processing_time_ms == 1234

    def test_search_result_model_exists(self):
        """Test that SearchResult model can be imported"""
        from backend.src.models.response import SearchResult

        assert SearchResult is not None, "SearchResult model must exist"

    def test_search_result_model_structure(self):
        """Test SearchResult model structure per spec"""
        from backend.src.models.response import SearchResult

        result = SearchResult(
            result_id="result-123",
            source="neo4j",
            content="Example content from database",
            relevance_score=0.92,
            metadata={"node_type": "Architecture"},
            relationships=["EVOLVED_FROM", "HAS_STATE"]
        )

        assert result.result_id == "result-123"
        assert result.source in ["neo4j", "qdrant"]
        assert result.relevance_score >= 0.0 and result.relevance_score <= 1.0
        assert isinstance(result.relationships, list)

    def test_response_with_multiple_sources(self):
        """Test Response with multiple SearchResults"""
        from backend.src.models.response import QueryResponse, SearchResult

        sources = [
            SearchResult(
                result_id="r1",
                source="neo4j",
                content="Graph result",
                relevance_score=0.9,
                metadata={},
                relationships=[]
            ),
            SearchResult(
                result_id="r2",
                source="qdrant",
                content="Vector result",
                relevance_score=0.85,
                metadata={},
                relationships=[]
            )
        ]

        response = QueryResponse(
            response_id="resp-789",
            query_id="query-789",
            answer="Synthesized from both sources",
            sources=sources,
            confidence=0.88,
            processing_time_ms=1500
        )

        assert len(response.sources) == 2
        assert response.sources[0].source == "neo4j"
        assert response.sources[1].source == "qdrant"

    def test_response_confidence_validation(self):
        """Test that confidence must be in [0, 1] range"""
        from backend.src.models.response import QueryResponse

        # Valid confidence
        response = QueryResponse(
            response_id="r1",
            query_id="q1",
            answer="test",
            sources=[],
            confidence=0.5,
            processing_time_ms=100
        )
        assert response.confidence == 0.5

        # Invalid confidence should raise error
        with pytest.raises(ValidationError):
            QueryResponse(
                response_id="r2",
                query_id="q2",
                answer="test",
                sources=[],
                confidence=1.5,  # Invalid: > 1.0
                processing_time_ms=100
            )

    def test_response_thoughtseed_trace_optional(self):
        """Test optional thoughtseed_trace field"""
        from backend.src.models.response import QueryResponse

        response = QueryResponse(
            response_id="r1",
            query_id="q1",
            answer="test",
            sources=[],
            confidence=0.7,
            processing_time_ms=100,
            thoughtseed_trace={"processing_layers": ["L1", "L2"]}
        )

        assert response.thoughtseed_trace is not None
        assert "processing_layers" in response.thoughtseed_trace

    def test_response_json_serialization(self):
        """Test that Response can be serialized to JSON for API"""
        from backend.src.models.response import QueryResponse

        response = QueryResponse(
            response_id="r1",
            query_id="q1",
            answer="test answer",
            sources=[],
            confidence=0.8,
            processing_time_ms=200
        )

        json_data = response.model_dump()
        assert isinstance(json_data, dict)
        assert json_data["answer"] == "test answer"
        assert json_data["confidence"] == 0.8
