"""
Response Synthesizer - Combine search results into coherent answer
Per Spec 006 FR-003: Synthesize results into coherent responses
"""

from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from models.response import SearchResult, QueryResponse
from models.query import Query

logger = logging.getLogger(__name__)


class ResponseSynthesizer:
    """
    Synthesize multiple search results into coherent response.

    Combines Neo4j graph results and Qdrant vector results,
    analyzes relevance, and generates natural language answer.
    """

    def __init__(self):
        """Initialize synthesizer."""
        self.min_confidence = 0.3
        self.max_sources = 10

    async def synthesize(
        self,
        query: Query,
        neo4j_results: List[SearchResult],
        qdrant_results: List[SearchResult],
        processing_time_ms: int
    ) -> QueryResponse:
        """
        Synthesize search results into final response.

        Args:
            query: Original query
            neo4j_results: Results from graph search
            qdrant_results: Results from vector search
            processing_time_ms: Total processing time

        Returns:
            Complete QueryResponse with synthesized answer
        """
        try:
            # Combine and rank all sources
            all_sources = self._combine_sources(neo4j_results, qdrant_results)

            # Calculate confidence based on result quality
            confidence = self._calculate_confidence(all_sources)

            # Generate answer from sources
            answer = await self._generate_answer(query.question, all_sources)

            # Create ThoughtSeed trace if applicable
            thoughtseed_trace = self._create_thoughtseed_trace(query, all_sources)

            return QueryResponse(
                query_id=query.query_id,
                answer=answer,
                sources=all_sources[:self.max_sources],
                confidence=confidence,
                processing_time_ms=processing_time_ms,
                thoughtseed_trace=thoughtseed_trace
            )

        except Exception as e:
            logger.error(f"Response synthesis failed: {e}")
            # Return fallback response
            return QueryResponse(
                query_id=query.query_id,
                answer="I encountered an error processing your query. Please try rephrasing your question.",
                sources=[],
                confidence=0.0,
                processing_time_ms=processing_time_ms
            )

    def _combine_sources(
        self,
        neo4j_results: List[SearchResult],
        qdrant_results: List[SearchResult]
    ) -> List[SearchResult]:
        """
        Combine and deduplicate sources from both databases.

        Graph results get slight boost for relationship richness.
        Vector results provide semantic coverage.
        """
        # Boost Neo4j results slightly for graph relationships
        for result in neo4j_results:
            relationship_boost = len(result.relationships) * 0.05
            result.relevance_score = min(result.relevance_score + relationship_boost, 1.0)

        # Combine all results
        all_results = neo4j_results + qdrant_results

        # Deduplicate by content similarity
        unique_results = self._deduplicate_by_content(all_results)

        # Sort by relevance
        sorted_results = sorted(unique_results, key=lambda r: r.relevance_score, reverse=True)

        return sorted_results

    def _deduplicate_by_content(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate or very similar content."""
        # Simple deduplication - in production, use embedding similarity
        seen_content = {}
        unique = []

        for result in results:
            # Use first 100 chars as fingerprint
            fingerprint = result.content[:100].lower().strip()

            if fingerprint not in seen_content:
                seen_content[fingerprint] = result
                unique.append(result)
            elif result.relevance_score > seen_content[fingerprint].relevance_score:
                # Replace with higher relevance version
                seen_content[fingerprint] = result
                unique = [r for r in unique if r.content[:100].lower().strip() != fingerprint]
                unique.append(result)

        return unique

    def _calculate_confidence(self, sources: List[SearchResult]) -> float:
        """
        Calculate confidence in response based on source quality.

        Factors:
        - Number of high-quality sources
        - Average relevance score
        - Source diversity (graph + vector)
        """
        if not sources:
            return 0.0

        # Average relevance of top sources
        top_sources = sources[:5]
        avg_relevance = sum(s.relevance_score for s in top_sources) / len(top_sources)

        # Source count factor (more sources = higher confidence, diminishing returns)
        count_factor = min(len(sources) / 10.0, 1.0)

        # Source diversity (both Neo4j and Qdrant present)
        has_graph = any(s.source.value == "neo4j" for s in sources)
        has_vector = any(s.source.value == "qdrant" for s in sources)
        diversity_factor = 1.0 if (has_graph and has_vector) else 0.85

        # Combine factors
        confidence = avg_relevance * 0.6 + count_factor * 0.3 * diversity_factor

        return min(max(confidence, 0.0), 1.0)

    async def _generate_answer(self, question: str, sources: List[SearchResult]) -> str:
        """
        Generate natural language answer from sources.

        In production, this would use LLM (OpenAI, Claude, etc.).
        For now, provides template-based response.
        """
        if not sources:
            return "I don't have enough information to answer this question. Please try rephrasing or providing more context."

        # Template-based answer generation for testing
        # TODO: Replace with LLM integration (OpenAI, LangGraph, etc.)

        top_source = sources[0]
        num_sources = len(sources)

        # Extract key information from top sources
        key_info = [s.content[:200] for s in sources[:3]]

        # Build answer
        answer_parts = []

        # Introduction
        if top_source.relevance_score > 0.8:
            answer_parts.append(f"Based on highly relevant information from {num_sources} source(s):")
        else:
            answer_parts.append(f"Based on {num_sources} related source(s):")

        # Key findings
        for i, info in enumerate(key_info, 1):
            answer_parts.append(f"\n{i}. {info.strip()}...")

        # Graph insights if available
        graph_sources = [s for s in sources if s.source.value == "neo4j" and s.relationships]
        if graph_sources:
            relationships = set()
            for s in graph_sources[:2]:
                relationships.update(s.relationships[:3])
            if relationships:
                answer_parts.append(f"\n\nRelated concepts: {', '.join(list(relationships)[:5])}")

        # Combine into final answer
        answer = " ".join(answer_parts)

        # Ensure minimum length
        if len(answer) < 10:
            answer = f"I found information related to your question: {top_source.content[:300]}..."

        return answer

    def _create_thoughtseed_trace(
        self,
        query: Query,
        sources: List[SearchResult]
    ) -> Optional[Dict[str, Any]]:
        """
        Create ThoughtSeed trace for consciousness tracking.

        This integrates with the extracted ThoughtSeed package
        to track cognitive processing flow.
        """
        if not query.thoughtseed_id:
            return None

        # Build trace information
        trace = {
            "thoughtseed_id": query.thoughtseed_id,
            "query_id": query.query_id,
            "timestamp": datetime.now().isoformat(),
            "sources_processed": len(sources),
            "neo4j_sources": len([s for s in sources if s.source.value == "neo4j"]),
            "qdrant_sources": len([s for s in sources if s.source.value == "qdrant"]),
            "avg_relevance": sum(s.relevance_score for s in sources) / len(sources) if sources else 0.0,
            "graph_relationships": list(set(
                rel for s in sources if s.source.value == "neo4j"
                for rel in s.relationships
            ))[:10],
            "processing_layers": ["L1_PERCEPTION", "L2_SEARCH", "L3_SYNTHESIS"],
            "consciousness_level": "active_inference"
        }

        return trace
