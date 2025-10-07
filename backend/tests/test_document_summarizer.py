#!/usr/bin/env python3
"""
Tests for DocumentSummarizer - Spec 055 Agent 3

Tests token-budgeted LLM summarization with fallback behavior.

Author: Spec 055 Agent 3 Implementation
Created: 2025-10-07
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock

# Add backend/src to path for imports
backend_src = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(backend_src))

from services.document_summarizer import (
    DocumentSummarizer,
    SummaryMetadata,
    SummarizerConfig
)


@pytest.fixture
def summarizer():
    """Create DocumentSummarizer with test configuration."""
    config = SummarizerConfig(
        model="gpt-3.5-turbo",
        max_tokens=150,
        temperature=0.3,
        api_key="test-api-key"
    )
    return DocumentSummarizer(config)


@pytest.fixture
def sample_document():
    """Sample document content for testing."""
    return """
    Active inference is a framework for understanding brain function and behavior.
    It suggests that the brain minimizes prediction errors by updating its internal model
    of the world and by acting to confirm its predictions. This framework provides a unified
    account of perception, action, and learning. The free energy principle is the theoretical
    foundation underlying active inference. It posits that biological systems minimize
    variational free energy to maintain their integrity. This principle has been applied
    to explain various aspects of cognition, including consciousness, attention, and
    decision-making. Recent research has extended active inference to multi-agent systems
    and collective intelligence. The framework has implications for artificial intelligence
    and machine learning, particularly in developing systems that learn and adapt
    in uncertain environments.
    """


@pytest.fixture
def short_document():
    """Short document content for testing."""
    return "Active inference explains brain function through prediction error minimization."


@pytest.fixture
def long_document():
    """Very long document content that exceeds token limits."""
    # ~5000 words, should be truncated
    base_text = """Active inference is a comprehensive framework for understanding brain
    function, behavior, perception, action, and learning. """ * 500
    return base_text


class TestDocumentSummarizerInit:
    """Test DocumentSummarizer initialization."""

    def test_init_with_config(self):
        """Test initialization with custom config."""
        config = SummarizerConfig(
            model="gpt-4",
            max_tokens=200,
            temperature=0.5,
            api_key="test-key"
        )
        summarizer = DocumentSummarizer(config)

        assert summarizer.config.model == "gpt-4"
        assert summarizer.config.max_tokens == 200
        assert summarizer.config.temperature == 0.5

    def test_init_with_env_api_key(self, monkeypatch):
        """Test initialization with API key from environment."""
        monkeypatch.setenv("OPENAI_API_KEY", "env-test-key")

        config = SummarizerConfig(model="gpt-3.5-turbo", max_tokens=150, temperature=0.3)
        summarizer = DocumentSummarizer(config)

        assert summarizer.config.api_key == "env-test-key"

    def test_init_without_api_key(self):
        """Test initialization fails without API key."""
        config = SummarizerConfig(
            model="gpt-3.5-turbo",
            max_tokens=150,
            temperature=0.3,
            api_key=None
        )

        with pytest.raises(ValueError, match="OpenAI API key not found"):
            DocumentSummarizer(config)


class TestTokenCounting:
    """Test token counting functionality."""

    def test_count_tokens_short_text(self, summarizer, short_document):
        """Test token counting for short text."""
        token_count = summarizer.count_tokens(short_document)

        assert isinstance(token_count, int)
        assert token_count > 0
        assert token_count < 50  # Short sentence

    def test_count_tokens_medium_text(self, summarizer, sample_document):
        """Test token counting for medium-length text."""
        token_count = summarizer.count_tokens(sample_document)

        assert isinstance(token_count, int)
        assert 50 < token_count < 500  # Paragraph

    def test_count_tokens_empty_text(self, summarizer):
        """Test token counting for empty text."""
        token_count = summarizer.count_tokens("")

        assert token_count == 0

    def test_count_tokens_special_characters(self, summarizer):
        """Test token counting with special characters."""
        text = "Test with émojis 🧠 and spëcial çharacters!"
        token_count = summarizer.count_tokens(text)

        assert token_count > 0


class TestTextTruncation:
    """Test text truncation for token limits."""

    def test_truncate_within_limit(self, summarizer, sample_document):
        """Test no truncation when within limit."""
        max_tokens = 1000
        truncated = summarizer.truncate_to_token_limit(sample_document, max_tokens)

        assert truncated == sample_document

    def test_truncate_exceeds_limit(self, summarizer, long_document):
        """Test truncation when exceeding limit."""
        max_tokens = 500
        truncated = summarizer.truncate_to_token_limit(long_document, max_tokens)

        # Should be truncated
        assert len(truncated) < len(long_document)

        # Should be within token limit
        token_count = summarizer.count_tokens(truncated)
        assert token_count <= max_tokens

    def test_truncate_preserves_coherence(self, summarizer, long_document):
        """Test truncation preserves sentence boundaries."""
        max_tokens = 500
        truncated = summarizer.truncate_to_token_limit(long_document, max_tokens)

        # Should end with punctuation or complete word
        assert truncated[-1] in ['.', '!', '?', ' '] or truncated[-1].isalnum()

    def test_truncate_very_small_limit(self, summarizer, sample_document):
        """Test truncation with very small token limit."""
        max_tokens = 10
        truncated = summarizer.truncate_to_token_limit(sample_document, max_tokens)

        token_count = summarizer.count_tokens(truncated)
        assert token_count <= max_tokens


class TestLLMSummarization:
    """Test LLM-based summarization."""

    @pytest.mark.asyncio
    async def test_generate_llm_summary_success(self, summarizer, sample_document):
        """Test successful LLM summary generation."""
        # Mock OpenAI API response
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content="Active inference minimizes prediction errors for brain function."))
        ]
        mock_response.usage = Mock(
            prompt_tokens=150,
            completion_tokens=15,
            total_tokens=165
        )

        with patch.object(summarizer, '_call_openai_api', return_value=mock_response):
            result = await summarizer.generate_llm_summary(sample_document, max_tokens=150)

        assert result["summary"] is not None
        assert len(result["summary"]) > 0
        assert result["method"] == "llm"
        assert result["tokens_used"] == 165
        assert result["model"] == "gpt-3.5-turbo"

    @pytest.mark.asyncio
    async def test_generate_llm_summary_respects_token_limit(self, summarizer, sample_document):
        """Test LLM summary respects token budget."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Summary content."))]
        mock_response.usage = Mock(prompt_tokens=100, completion_tokens=10, total_tokens=110)

        with patch.object(summarizer, '_call_openai_api', return_value=mock_response):
            result = await summarizer.generate_llm_summary(sample_document, max_tokens=50)

        # Should have called API with max_tokens=50
        assert result["tokens_used"] <= 110

    @pytest.mark.asyncio
    async def test_generate_llm_summary_with_long_input(self, summarizer, long_document):
        """Test LLM summary with long input (should truncate)."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Truncated content summary."))]
        mock_response.usage = Mock(prompt_tokens=500, completion_tokens=15, total_tokens=515)

        with patch.object(summarizer, '_call_openai_api', return_value=mock_response):
            result = await summarizer.generate_llm_summary(long_document, max_tokens=150)

        # Should have truncated input
        assert result["summary"] is not None
        assert result["tokens_used"] > 0

    @pytest.mark.asyncio
    async def test_generate_llm_summary_api_error(self, summarizer, sample_document):
        """Test LLM summary handles API errors."""
        # Mock API error
        with patch.object(summarizer, '_call_openai_api', side_effect=Exception("API Error")):
            with pytest.raises(Exception, match="API Error"):
                await summarizer.generate_llm_summary(sample_document, max_tokens=150)

    @pytest.mark.asyncio
    async def test_generate_llm_summary_empty_response(self, summarizer, sample_document):
        """Test LLM summary handles empty response."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content=""))]
        mock_response.usage = Mock(prompt_tokens=100, completion_tokens=0, total_tokens=100)

        with patch.object(summarizer, '_call_openai_api', return_value=mock_response):
            result = await summarizer.generate_llm_summary(sample_document, max_tokens=150)

        assert result["summary"] == ""
        assert result["method"] == "llm"


class TestExtractiveSummarization:
    """Test extractive (fallback) summarization."""

    def test_generate_extractive_summary_first_sentences(self, summarizer, sample_document):
        """Test extractive summary uses first sentences."""
        result = summarizer.generate_extractive_summary(sample_document, max_tokens=50)

        assert result["summary"] is not None
        assert len(result["summary"]) > 0
        assert result["method"] == "extractive"
        assert result["tokens_used"] <= 50

    def test_generate_extractive_summary_short_document(self, summarizer, short_document):
        """Test extractive summary with short document."""
        result = summarizer.generate_extractive_summary(short_document, max_tokens=100)

        # Should return entire document if within limit
        assert result["summary"] == short_document.strip()
        assert result["method"] == "extractive"

    def test_generate_extractive_summary_respects_limit(self, summarizer, long_document):
        """Test extractive summary respects token limit."""
        max_tokens = 100
        result = summarizer.generate_extractive_summary(long_document, max_tokens=max_tokens)

        # Should be within token limit
        token_count = summarizer.count_tokens(result["summary"])
        assert token_count <= max_tokens
        assert result["tokens_used"] <= max_tokens

    def test_generate_extractive_summary_sentence_boundaries(self, summarizer, sample_document):
        """Test extractive summary preserves sentence boundaries."""
        result = summarizer.generate_extractive_summary(sample_document, max_tokens=50)

        # Should end with punctuation
        assert result["summary"][-1] in ['.', '!', '?']


class TestGenerateSummary:
    """Test main generate_summary method."""

    @pytest.mark.asyncio
    async def test_generate_summary_llm_success(self, summarizer, sample_document):
        """Test generate_summary uses LLM when available."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="LLM-generated summary."))]
        mock_response.usage = Mock(prompt_tokens=100, completion_tokens=10, total_tokens=110)

        with patch.object(summarizer, '_call_openai_api', return_value=mock_response):
            result = await summarizer.generate_summary(sample_document)

        assert result["summary"] == "LLM-generated summary."
        assert result["method"] == "llm"
        assert result["model"] == "gpt-3.5-turbo"

    @pytest.mark.asyncio
    async def test_generate_summary_fallback_on_error(self, summarizer, sample_document):
        """Test generate_summary falls back to extractive on LLM error."""
        # Mock API error
        with patch.object(summarizer, '_call_openai_api', side_effect=Exception("API Error")):
            result = await summarizer.generate_summary(sample_document)

        # Should fallback to extractive
        assert result["summary"] is not None
        assert result["method"] == "extractive"
        assert "error" in result
        assert "API Error" in result["error"]

    @pytest.mark.asyncio
    async def test_generate_summary_custom_max_tokens(self, summarizer, sample_document):
        """Test generate_summary with custom max_tokens."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Custom token limit summary."))]
        mock_response.usage = Mock(prompt_tokens=50, completion_tokens=10, total_tokens=60)

        with patch.object(summarizer, '_call_openai_api', return_value=mock_response):
            result = await summarizer.generate_summary(sample_document, max_tokens=50)

        assert result["tokens_used"] <= 60
        assert result["method"] == "llm"

    @pytest.mark.asyncio
    async def test_generate_summary_includes_metadata(self, summarizer, sample_document):
        """Test generate_summary includes complete metadata."""
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Summary with metadata."))]
        mock_response.usage = Mock(prompt_tokens=100, completion_tokens=10, total_tokens=110)

        with patch.object(summarizer, '_call_openai_api', return_value=mock_response):
            result = await summarizer.generate_summary(sample_document)

        # Check all required fields
        assert "summary" in result
        assert "method" in result
        assert "model" in result
        assert "tokens_used" in result
        assert "generated_at" in result

    @pytest.mark.asyncio
    async def test_generate_summary_empty_document(self, summarizer):
        """Test generate_summary with empty document."""
        result = await summarizer.generate_summary("")

        # Should fallback to extractive with empty result
        assert result["summary"] == ""
        assert result["method"] == "extractive"


class TestSummaryMetadata:
    """Test SummaryMetadata model."""

    def test_summary_metadata_validation(self):
        """Test SummaryMetadata validates fields."""
        metadata = SummaryMetadata(
            method="llm",
            model="gpt-3.5-turbo",
            tokens_used=150,
            generated_at="2025-10-07T10:00:00Z"
        )

        assert metadata.method == "llm"
        assert metadata.model == "gpt-3.5-turbo"
        assert metadata.tokens_used == 150

    def test_summary_metadata_invalid_method(self):
        """Test SummaryMetadata rejects invalid method."""
        with pytest.raises(ValueError):
            SummaryMetadata(
                method="invalid_method",
                model="gpt-3.5-turbo",
                tokens_used=150,
                generated_at="2025-10-07T10:00:00Z"
            )

    def test_summary_metadata_optional_fields(self):
        """Test SummaryMetadata with optional fields."""
        metadata = SummaryMetadata(
            method="extractive",
            model=None,
            tokens_used=100,
            generated_at="2025-10-07T10:00:00Z",
            error="Test error"
        )

        assert metadata.error == "Test error"
        assert metadata.model is None


class TestIntegration:
    """Integration tests for DocumentSummarizer."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_end_to_end_summarization(self, sample_document):
        """Test end-to-end summarization workflow."""
        # This test requires real OpenAI API key
        # Skip if not available
        import os
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            pytest.skip("OPENAI_API_KEY not set, skipping integration test")

        config = SummarizerConfig(
            model="gpt-3.5-turbo",
            max_tokens=150,
            temperature=0.3,
            api_key=api_key
        )
        summarizer = DocumentSummarizer(config)

        result = await summarizer.generate_summary(sample_document)

        # Verify result structure
        assert result["summary"] is not None
        assert len(result["summary"]) > 0
        assert result["method"] in ["llm", "extractive"]
        assert result["tokens_used"] > 0
        assert "generated_at" in result

    @pytest.mark.asyncio
    async def test_multiple_documents_parallel(self, summarizer):
        """Test summarizing multiple documents in parallel."""
        documents = [
            "Document 1 about active inference.",
            "Document 2 about consciousness.",
            "Document 3 about neural networks."
        ]

        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Summary"))]
        mock_response.usage = Mock(prompt_tokens=50, completion_tokens=5, total_tokens=55)

        with patch.object(summarizer, '_call_openai_api', return_value=mock_response):
            results = await asyncio.gather(
                *[summarizer.generate_summary(doc) for doc in documents]
            )

        assert len(results) == 3
        assert all(r["summary"] is not None for r in results)


# Import asyncio for parallel test
import asyncio
