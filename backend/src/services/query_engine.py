"""
Query Engine - Orchestrates complete query processing pipeline
Per Spec 006: Natural language query → Search → Synthesis → Response
"""

from typing import Optional, Dict, Any
import asyncio
import time
import logging
from datetime import datetime

from src.models.query import Query
from src.models.response import QueryResponse
from src.services.neo4j_searcher import Neo4jSearcher
from src.services.vector_searcher import VectorSearcher
from src.services.response_synthesizer import ResponseSynthesizer

logger = logging.getLogger(__name__)


class QueryEngine:
    """
    Main query processing engine.

    Orchestrates parallel search across Neo4j and Qdrant,
    then synthesizes results into coherent response.

    Performance target: <2s per query (per Spec 006)
    """

    def __init__(
        self,
        neo4j_searcher: Optional[Neo4jSearcher] = None,
        vector_searcher: Optional[VectorSearcher] = None,
        response_synthesizer: Optional[ResponseSynthesizer] = None
    ):
        """Initialize query engine with searchers."""
        self.neo4j_searcher = neo4j_searcher or Neo4jSearcher()
        self.vector_searcher = vector_searcher or VectorSearcher()
        self.response_synthesizer = response_synthesizer or ResponseSynthesizer()
        self.default_result_limit = 10

    async def process_query(
        self,
        question: str,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        thoughtseed_id: Optional[str] = None
    ) -> QueryResponse:
        """
        Process natural language query end-to-end.

        Args:
            question: User's natural language question
            user_id: Optional user identifier
            context: Optional session context for follow-up questions
            thoughtseed_id: Optional ThoughtSeed ID for consciousness tracking

        Returns:
            Complete QueryResponse with synthesized answer and sources
        """
        start_time = time.time()

        try:
            # Create Query object
            query = Query(
                question=question,
                user_id=user_id,
                context=context or {},
                thoughtseed_id=thoughtseed_id,
                timestamp=datetime.now()
            )

            logger.info(f"Processing query {query.query_id}: {question[:50]}...")

            # Parallel search across both databases
            neo4j_results, qdrant_results = await self._parallel_search(query.question)

            logger.info(f"Search complete: {len(neo4j_results)} Neo4j, {len(qdrant_results)} Qdrant")

            # Calculate processing time
            processing_time_ms = int((time.time() - start_time) * 1000)

            # Synthesize final response
            response = await self.response_synthesizer.synthesize(
                query=query,
                neo4j_results=neo4j_results,
                qdrant_results=qdrant_results,
                processing_time_ms=processing_time_ms
            )

            logger.info(
                f"Query {query.query_id} completed in {processing_time_ms}ms "
                f"(confidence: {response.confidence:.2f})"
            )

            return response

        except Exception as e:
            logger.error(f"Query processing failed: {e}", exc_info=True)
            processing_time_ms = int((time.time() - start_time) * 1000)

            # Return error response
            return QueryResponse(
                query_id="error",
                answer="I encountered an error processing your query. Please try again.",
                sources=[],
                confidence=0.0,
                processing_time_ms=processing_time_ms
            )

    async def _parallel_search(self, question: str):
        """
        Execute Neo4j and Qdrant searches in parallel.

        This is critical for meeting <2s performance target.
        """
        try:
            # Run both searches concurrently
            neo4j_task = self.neo4j_searcher.search(question, self.default_result_limit)
            qdrant_task = self.vector_searcher.search(question, self.default_result_limit)

            # Wait for both to complete
            neo4j_results, qdrant_results = await asyncio.gather(
                neo4j_task,
                qdrant_task,
                return_exceptions=True
            )

            # Handle exceptions from either search
            if isinstance(neo4j_results, Exception):
                logger.error(f"Neo4j search failed: {neo4j_results}")
                neo4j_results = []

            if isinstance(qdrant_results, Exception):
                logger.error(f"Qdrant search failed: {qdrant_results}")
                qdrant_results = []

            return neo4j_results, qdrant_results

        except Exception as e:
            logger.error(f"Parallel search failed: {e}")
            return [], []

    async def process_batch_queries(self, questions: list[str]) -> list[QueryResponse]:
        """
        Process multiple queries efficiently.

        Args:
            questions: List of natural language questions

        Returns:
            List of QueryResponse objects
        """
        tasks = [self.process_query(q) for q in questions]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions in batch
        results = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                logger.error(f"Batch query {i} failed: {response}")
                results.append(QueryResponse(
                    query_id=f"batch-error-{i}",
                    answer="Query processing failed",
                    sources=[],
                    confidence=0.0,
                    processing_time_ms=0
                ))
            else:
                results.append(response)

        return results

    async def health_check(self) -> Dict[str, bool]:
        """
        Check health of all query engine components.

        Returns:
            Status of Neo4j, Qdrant, and overall system
        """
        health = {
            "neo4j": False,
            "qdrant": False,
            "overall": False
        }

        try:
            # Test Neo4j connection
            neo4j_test = await self.neo4j_searcher.search("test", limit=1)
            health["neo4j"] = True
        except Exception as e:
            logger.warning(f"Neo4j health check failed: {e}")

        try:
            # Test Qdrant connection
            qdrant_exists = await self.vector_searcher.collection_exists()
            health["qdrant"] = qdrant_exists
        except Exception as e:
            logger.warning(f"Qdrant health check failed: {e}")

        health["overall"] = health["neo4j"] or health["qdrant"]

        return health
