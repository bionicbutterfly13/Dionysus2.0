#!/usr/bin/env python3
"""
Document Summarizer Service - Spec 055 Agent 3

Generates token-budgeted LLM summaries with extractive fallback.

CONSTITUTIONAL COMPLIANCE (Spec 040):
- No Neo4j access required (pure LLM service)
- Integrates with DocumentRepository via persist_document()

Features:
- Token-aware summarization with configurable limits
- OpenAI API integration (gpt-3.5-turbo, gpt-4)
- Extractive fallback when LLM unavailable
- Comprehensive metadata tracking

Author: Spec 055 Agent 3 Implementation
Created: 2025-10-07
"""

import os
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pydantic import BaseModel, Field
import tiktoken
import openai
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class SummarizerConfig(BaseModel):
    """Configuration for DocumentSummarizer."""
    model: str = Field(default="gpt-3.5-turbo", description="OpenAI model to use")
    max_tokens: int = Field(default=150, ge=10, le=500, description="Maximum tokens for summary")
    temperature: float = Field(default=0.3, ge=0.0, le=1.0, description="Sampling temperature")
    api_key: Optional[str] = Field(default=None, description="OpenAI API key")

    class Config:
        json_schema_extra = {
            "example": {
                "model": "gpt-3.5-turbo",
                "max_tokens": 150,
                "temperature": 0.3,
                "api_key": "sk-..."
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
                "model": "gpt-3.5-turbo",
                "tokens_used": 165,
                "generated_at": "2025-10-07T10:00:00Z"
            }
        }


class DocumentSummarizer:
    """
    Token-budgeted document summarizer with LLM and extractive methods.

    Workflow:
    1. Try LLM summarization (OpenAI API)
    2. Fallback to extractive summarization on error
    3. Return summary + comprehensive metadata
    """

    def __init__(self, config: Optional[SummarizerConfig] = None):
        """
        Initialize DocumentSummarizer.

        Args:
            config: Summarizer configuration

        Raises:
            ValueError: If API key not found
        """
        self.config = config or SummarizerConfig()

        # Get API key from config or environment
        if self.config.api_key is None:
            self.config.api_key = os.getenv("OPENAI_API_KEY")

        if not self.config.api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or pass api_key in config."
            )

        # Initialize OpenAI client
        self.client = AsyncOpenAI(api_key=self.config.api_key)

        # Initialize tokenizer for token counting
        try:
            self.encoding = tiktoken.encoding_for_model(self.config.model)
        except KeyError:
            # Fallback to cl100k_base for unknown models
            logger.warning(f"Unknown model {self.config.model}, using cl100k_base encoding")
            self.encoding = tiktoken.get_encoding("cl100k_base")

        logger.info(
            f"DocumentSummarizer initialized: model={self.config.model}, "
            f"max_tokens={self.config.max_tokens}"
        )

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
        except Exception as e:
            logger.warning(f"Token counting failed: {e}, using character approximation")
            # Fallback: approximate as chars/4
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

        # Truncate by sentences to preserve coherence
        sentences = text.split('. ')
        truncated = ""

        for sentence in sentences:
            candidate = truncated + sentence + ". "
            if self.count_tokens(candidate) <= max_tokens:
                truncated = candidate
            else:
                break

        # If no full sentences fit, truncate by words
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

    async def _call_openai_api(
        self,
        prompt: str,
        max_completion_tokens: int
    ) -> Any:
        """
        Call OpenAI API with error handling.

        Args:
            prompt: System + user prompt
            max_completion_tokens: Max tokens for completion

        Returns:
            OpenAI API response

        Raises:
            Exception: On API error
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise document summarizer. Create concise, "
                                   "informative summaries that capture the key points."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=max_completion_tokens,
                temperature=self.config.temperature
            )
            return response

        except openai.RateLimitError as e:
            logger.error(f"OpenAI rate limit exceeded: {e}")
            raise
        except openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error calling OpenAI API: {e}")
            raise

    async def generate_llm_summary(
        self,
        document_content: str,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate summary using LLM (OpenAI API).

        Token budget allocation:
        - Input: Up to 3000 tokens (truncate if needed)
        - Output: max_tokens parameter
        - Overhead: ~50 tokens for system message

        Args:
            document_content: Document text to summarize
            max_tokens: Maximum tokens for summary (defaults to config)

        Returns:
            {
                "summary": str,
                "method": "llm",
                "model": str,
                "tokens_used": int,
                "generated_at": str (ISO 8601)
            }

        Raises:
            Exception: On API error (caller should handle)
        """
        max_tokens = max_tokens or self.config.max_tokens

        # Budget: 3000 tokens for input (leave room for system message)
        input_token_budget = 3000
        truncated_content = self.truncate_to_token_limit(document_content, input_token_budget)

        # Create prompt
        prompt = f"""Summarize the following document in {max_tokens} tokens or less.
Focus on the main ideas, key findings, and core concepts.

Document:
{truncated_content}

Summary:"""

        # Call API
        response = await self._call_openai_api(prompt, max_tokens)

        # Extract summary
        summary = response.choices[0].message.content.strip()

        # Extract token usage
        tokens_used = response.usage.total_tokens

        return {
            "summary": summary,
            "method": "llm",
            "model": self.config.model,
            "tokens_used": tokens_used,
            "generated_at": datetime.utcnow().isoformat() + "Z"
        }

    def generate_extractive_summary(
        self,
        document_content: str,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate extractive summary (fallback method).

        Strategy:
        1. Take first N sentences that fit within token budget
        2. Preserve sentence boundaries for coherence

        Args:
            document_content: Document text to summarize
            max_tokens: Maximum tokens for summary (defaults to config)

        Returns:
            {
                "summary": str,
                "method": "extractive",
                "model": None,
                "tokens_used": int,
                "generated_at": str (ISO 8601)
            }
        """
        max_tokens = max_tokens or self.config.max_tokens

        # Clean up text
        content = document_content.strip()

        if not content:
            return {
                "summary": "",
                "method": "extractive",
                "model": None,
                "tokens_used": 0,
                "generated_at": datetime.utcnow().isoformat() + "Z"
            }

        # Extract first sentences within token limit
        sentences = content.split('. ')
        summary = ""

        for sentence in sentences:
            # Check if sentence already ends with period
            if sentence.endswith('.'):
                candidate = summary + sentence + " "
            else:
                candidate = summary + sentence + ". "

            if self.count_tokens(candidate) <= max_tokens:
                summary = candidate
            else:
                break

        # If no full sentences fit, take first sentence truncated
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
            "generated_at": datetime.utcnow().isoformat() + "Z"
        }

    async def generate_summary(
        self,
        document_content: str,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate document summary with automatic fallback.

        Workflow:
        1. Try LLM summarization
        2. On error, fallback to extractive summarization
        3. Return summary + metadata

        Args:
            document_content: Document text to summarize
            max_tokens: Maximum tokens for summary (defaults to config)

        Returns:
            {
                "summary": str,
                "method": "llm" | "extractive",
                "model": str | None,
                "tokens_used": int,
                "generated_at": str (ISO 8601),
                "error": str | None  (only if fallback occurred)
            }
        """
        max_tokens = max_tokens or self.config.max_tokens

        try:
            # Try LLM summarization
            result = await self.generate_llm_summary(document_content, max_tokens)
            logger.info(
                f"LLM summary generated: {result['tokens_used']} tokens, "
                f"model={result['model']}"
            )
            return result

        except Exception as e:
            # Log error and fallback to extractive
            logger.warning(
                f"LLM summarization failed ({type(e).__name__}: {e}), "
                f"falling back to extractive method"
            )

            result = self.generate_extractive_summary(document_content, max_tokens)
            result["error"] = f"LLM failed: {type(e).__name__}: {str(e)}"

            logger.info(
                f"Extractive summary generated (fallback): {result['tokens_used']} tokens"
            )

            return result


# Convenience function for quick summarization
async def summarize_document(
    document_content: str,
    max_tokens: int = 150,
    model: str = "gpt-3.5-turbo",
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Quick document summarization with default settings.

    Args:
        document_content: Document text to summarize
        max_tokens: Maximum tokens for summary
        model: OpenAI model to use
        api_key: OpenAI API key (defaults to env var)

    Returns:
        Summary result dict

    Example:
        >>> result = await summarize_document("Long document text...", max_tokens=100)
        >>> print(result["summary"])
        "Brief summary of the document."
    """
    config = SummarizerConfig(
        model=model,
        max_tokens=max_tokens,
        temperature=0.3,
        api_key=api_key
    )

    summarizer = DocumentSummarizer(config)
    return await summarizer.generate_summary(document_content)
