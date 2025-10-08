#!/usr/bin/env python3
"""
Document Summarizer Service - Spec 055 Agent 3 (Local-First Revision)

Generates token-budgeted summaries using local Ollama models with
deterministic extractive fallback when the LLM is unavailable.

CONSTITUTIONAL COMPLIANCE (Spec 040):
- No Neo4j access required (pure summarization utility)
- Prefers local inference (Ollama) with graceful degradation

Features:
- Token-aware summarization with configurable limits
- Local Ollama integration by default (no external API dependency)
- Extractive fallback when LLM generation fails
- Comprehensive metadata tracking

Author: Spec 055 Agent 3 Implementation
Updated: 2025-10-10 (Local-first summarization)
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pydantic import BaseModel, Field
import tiktoken

try:  # pragma: no cover - allow import flexibility during testing
    from services.ollama_integration import OllamaModelManager
except ImportError:  # pragma: no cover - fallback for relative imports
    from ..services.ollama_integration import OllamaModelManager  # type: ignore

logger = logging.getLogger(__name__)


class SummarizerConfig(BaseModel):
    """Configuration for DocumentSummarizer."""
    model: str = Field(default="qwen2.5:14b", description="Local Ollama model to use")
    max_tokens: int = Field(default=150, ge=10, le=500, description="Maximum tokens for summary")
    temperature: float = Field(default=0.3, ge=0.0, le=1.0, description="Sampling temperature")

    class Config:
        json_schema_extra = {
            "example": {
                "model": "qwen2.5:14b",
                "max_tokens": 150,
                "temperature": 0.3
            }
        }


class SummaryMetadata(BaseModel):
    """Metadata for generated summary."""
    method: str = Field(..., pattern="^(llm|extractive)$")
    model: Optional[str] = None
    tokens_used: int = Field(ge=0)
    generated_at: str
    error: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "method": "llm",
                "model": "qwen2.5:14b",
                "tokens_used": 165,
                "generated_at": "2025-10-07T10:00:00Z"
            }
        }


class DocumentSummarizer:
    """
    Token-budgeted document summarizer with LLM and extractive methods.

    Workflow:
    1. Try local Ollama summarization
    2. Fallback to extractive summarization on error
    3. Return summary + comprehensive metadata
    """

    def __init__(
        self,
        config: Optional[SummarizerConfig] = None,
        model_manager: Optional[OllamaModelManager] = None
    ):
        """
        Initialize DocumentSummarizer.

        Args:
            config: Summarizer configuration
            model_manager: Optional injected Ollama model manager (primarily for testing)
        """
        self.config = config or SummarizerConfig()

        try:
            self.model_manager = model_manager or OllamaModelManager()
            self.llm_available = True
            logger.info(
                "DocumentSummarizer initialized with local Ollama model '%s'",
                self.config.model
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            self.model_manager = None
            self.llm_available = False
            logger.warning(
                "Ollama model manager unavailable (%s). Falling back to extractive summaries only.",
                exc
            )

        # Initialize tokenizer for token counting
        try:
            self.encoding = tiktoken.encoding_for_model(self.config.model)
        except KeyError:
            logger.warning(
                "Unknown model '%s', falling back to cl100k_base encoding",
                self.config.model
            )
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tiktoken.

        Args:
            text: Text to count tokens for

        Returns:
            Token count
        """
        if not text:
            return 0

        try:
            return len(self.encoding.encode(text))
        except Exception as exc:
            logger.warning("Token counting failed (%s). Approximating.", exc)
            return len(text) // 4

    def truncate_to_token_limit(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to fit within token limit while preserving coherence.

        Strategy:
        1. Count tokens in full text
        2. If within limit, return as-is
        3. Otherwise, truncate at sentence boundaries

        Args:
            text: Text to truncate
            max_tokens: Maximum token count

        Returns:
            Truncated text within token limit
        """
        current_tokens = self.count_tokens(text)

        if current_tokens <= max_tokens:
            return text

        sentences = text.split('. ')
        truncated = ""

        for sentence in sentences:
            candidate = truncated + sentence + ". "
            if self.count_tokens(candidate) <= max_tokens:
                truncated = candidate
            else:
                break

        if not truncated:
            words = text.split()
            truncated = ""
            for word in words:
                candidate = truncated + word + " "
                if self.count_tokens(candidate) <= max_tokens:
                    truncated = candidate
                else:
                    break

        return truncated.strip()

    async def generate_llm_summary(
        self,
        document_content: str,
        max_tokens: int
    ) -> Dict[str, Any]:
        """
        Generate summary via local Ollama model.

        Args:
            document_content: Document text
            max_tokens: Max tokens for completion

        Returns:
            Dictionary with summary, tokens used, metadata

        Raises:
            RuntimeError: If local LLM is not available or fails to generate a summary.
        """
        if not self.llm_available or not self.model_manager:
            raise RuntimeError("Local LLM summarization unavailable")

        prompt = (
            "You are the Dionysus Flux document summarizer. Create a concise, factual summary "
            "that captures the primary findings, key data points, and conclusions from the "
            "provided document.\n\n"
            "Requirements:\n"
            "- Keep the summary under the requested token budget.\n"
            "- Maintain objective tone (no speculation).\n"
            "- Preserve critical numbers, dates, or named entities when present.\n"
            "- Return plain text without bullet lists.\n\n"
            f"Document:\n{document_content}"
        )

        result = await self.model_manager.generate_text(
            self.config.model,
            prompt,
            max_tokens=max_tokens,
            temperature=self.config.temperature
        )

        if not result.get("success"):
            raise RuntimeError(result.get("error", "Local LLM summarization failed"))

        summary_text = (result.get("response") or "").strip()
        if not summary_text:
            raise RuntimeError("Local LLM returned empty summary")

        tokens_used = result.get("eval_count") or self.count_tokens(summary_text)

        return {
            "summary": summary_text,
            "method": "llm",
            "model": self.config.model,
            "tokens_used": tokens_used,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "error": None
        }

    def generate_extractive_summary(
        self,
        document_content: str,
        max_tokens: int
    ) -> Dict[str, Any]:
        """
        Generate extractive summary by selecting key sentences.

        Args:
            document_content: Document text
            max_tokens: Max tokens for summary (approximate)

        Returns:
            Dictionary with summary and metadata
        """
        if not document_content:
            return {
                "summary": "",
                "method": "extractive",
                "model": None,
                "tokens_used": 0,
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "error": None
            }

        sentences = document_content.split('. ')
        summary = ""

        for sentence in sentences:
            candidate = summary + sentence
            if self.count_tokens(candidate) <= max_tokens:
                summary = candidate + ". "
            else:
                break

        if not summary and sentences:
            summary = self.truncate_to_token_limit(sentences[0], max_tokens)
            if not summary.endswith('.'):
                summary += "."

        summary = summary.strip()
        tokens_used = self.count_tokens(summary)

        return {
            "summary": summary,
            "method": "extractive",
            "model": None,
            "tokens_used": tokens_used,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "error": None
        }

    async def generate_summary(
        self,
        document_content: str,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate document summary with automatic fallback.

        Args:
            document_content: Document text to summarize
            max_tokens: Maximum tokens for summary (defaults to config)

        Returns:
            Summary dictionary with method, tokens, metadata
        """
        max_tokens = max_tokens or self.config.max_tokens

        try:
            result = await self.generate_llm_summary(document_content, max_tokens)
            logger.info(
                "Local LLM summary generated using %s (%d tokens)",
                result.get("model"),
                result.get("tokens_used", 0)
            )
            return result

        except Exception as exc:
            logger.warning(
                "Local LLM summarization failed (%s). Falling back to extractive summary.",
                exc
            )
            result = self.generate_extractive_summary(document_content, max_tokens)
            result["error"] = f"llm_unavailable: {type(exc).__name__}: {exc}"
            return result


async def summarize_document(
    document_content: str,
    max_tokens: Optional[int] = None
) -> Dict[str, Any]:
    """
    Convenience wrapper for one-off summarization requests.
    """
    summarizer = DocumentSummarizer()
    return await summarizer.generate_summary(document_content, max_tokens)
