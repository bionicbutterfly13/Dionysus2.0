#!/usr/bin/env python3
"""
Demo script for DocumentSummarizer - Spec 055 Agent 3

Demonstrates token-budgeted LLM summarization with extractive fallback.

Usage:
    # With OpenAI API key (for LLM summarization):
    OPENAI_API_KEY=sk-xxx python demo_summarizer.py

    # Without API key (extractive fallback):
    python demo_summarizer.py

Author: Spec 055 Agent 3 Implementation
Created: 2025-10-07
"""

import asyncio
import os
from src.services.document_summarizer import DocumentSummarizer, SummarizerConfig


async def demo_summarization():
    """Demonstrate document summarization with various configurations."""

    # Sample document about active inference
    sample_document = """
    Active inference is a comprehensive theoretical framework for understanding brain
    function, behavior, and cognition. It proposes that biological agents minimize
    prediction errors by updating their internal model of the world and by acting
    to confirm their predictions. This framework provides a unified account of
    perception, action, learning, and decision-making.

    The free energy principle is the mathematical foundation underlying active inference.
    It posits that biological systems minimize variational free energy to maintain
    their integrity and resist disorder. Free energy serves as an upper bound on
    surprise, making it a tractable objective for biological agents to minimize.

    Recent research has extended active inference to explain various aspects of
    cognition including consciousness, attention, planning, and social cognition.
    The framework has also been applied to multi-agent systems and collective
    intelligence, showing promise for understanding emergent phenomena in complex
    adaptive systems.

    From a computational perspective, active inference provides insights for
    developing artificial intelligence systems that learn and adapt in uncertain
    environments. By modeling agents as active inference machines, researchers
    are creating more robust and flexible AI systems that better handle real-world
    complexity.
    """

    print("=" * 80)
    print("DOCUMENT SUMMARIZER DEMO - Spec 055 Agent 3")
    print("=" * 80)
    print()

    # Check if OpenAI API key is available
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print("✅ OpenAI API key found - will use LLM summarization")
    else:
        print("⚠️  No OpenAI API key - will use extractive fallback")

    print()
    print("-" * 80)
    print("ORIGINAL DOCUMENT")
    print("-" * 80)
    print(sample_document)
    print()

    # Demo 1: Standard summarization (150 tokens)
    print("-" * 80)
    print("DEMO 1: Standard Summarization (150 tokens)")
    print("-" * 80)

    try:
        config = SummarizerConfig(
            model="gpt-3.5-turbo",
            max_tokens=150,
            temperature=0.3,
            api_key=api_key
        )
        summarizer = DocumentSummarizer(config)

        result = await summarizer.generate_summary(sample_document, max_tokens=150)

        print(f"Summary: {result['summary']}")
        print()
        print(f"Method: {result['method']}")
        print(f"Model: {result['model']}")
        print(f"Tokens Used: {result['tokens_used']}")
        print(f"Generated At: {result['generated_at']}")
        if result.get('error'):
            print(f"Note: {result['error']}")
        print()

    except Exception as e:
        print(f"❌ Error: {e}")
        print()

    # Demo 2: Concise summarization (50 tokens)
    print("-" * 80)
    print("DEMO 2: Concise Summarization (50 tokens)")
    print("-" * 80)

    try:
        result = await summarizer.generate_summary(sample_document, max_tokens=50)

        print(f"Summary: {result['summary']}")
        print()
        print(f"Method: {result['method']}")
        print(f"Tokens Used: {result['tokens_used']}")
        print()

    except Exception as e:
        print(f"❌ Error: {e}")
        print()

    # Demo 3: Detailed summarization (300 tokens)
    print("-" * 80)
    print("DEMO 3: Detailed Summarization (300 tokens)")
    print("-" * 80)

    try:
        result = await summarizer.generate_summary(sample_document, max_tokens=300)

        print(f"Summary: {result['summary']}")
        print()
        print(f"Method: {result['method']}")
        print(f"Tokens Used: {result['tokens_used']}")
        print()

    except Exception as e:
        print(f"❌ Error: {e}")
        print()

    # Demo 4: Extractive fallback (force error to test fallback)
    print("-" * 80)
    print("DEMO 4: Extractive Fallback (Simulated)")
    print("-" * 80)

    try:
        # Create config with invalid API key to force fallback
        fallback_config = SummarizerConfig(
            model="gpt-3.5-turbo",
            max_tokens=150,
            temperature=0.3,
            api_key="invalid-key-to-test-fallback"
        )
        fallback_summarizer = DocumentSummarizer(fallback_config)

        result = await fallback_summarizer.generate_summary(sample_document, max_tokens=150)

        print(f"Summary: {result['summary']}")
        print()
        print(f"Method: {result['method']}")
        print(f"Tokens Used: {result['tokens_used']}")
        if result.get('error'):
            print(f"Fallback Reason: {result['error']}")
        print()

    except Exception as e:
        print(f"Note: Fallback failed with: {e}")
        print()

    print("=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print()
    print("Key Features Demonstrated:")
    print("  ✅ Token-aware summarization (respects max_tokens budget)")
    print("  ✅ Configurable summary length (50, 150, 300 tokens)")
    print("  ✅ LLM-based summarization (OpenAI GPT-3.5-turbo)")
    print("  ✅ Extractive fallback (when LLM unavailable)")
    print("  ✅ Comprehensive metadata (method, model, tokens, timestamp)")
    print()
    print("Integration Points:")
    print("  📄 Document Schema: summary + summary_metadata fields added")
    print("  💾 Repository: Auto-generates summaries during persist_document()")
    print("  🔌 API: Returns summaries in GET /api/documents and GET /api/documents/{id}")
    print()


if __name__ == "__main__":
    asyncio.run(demo_summarization())
