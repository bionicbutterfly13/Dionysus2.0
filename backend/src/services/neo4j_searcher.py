"""
Neo4j Graph Searcher - Knowledge graph search for query engine
Per Spec 006 FR-002: Search Neo4j graph database
Per Spec 040 M2: Uses Daedalus Graph Channel (no direct neo4j imports)

Constitutional Compliance:
- Section 2.1: Daedalus Gateway Requirements - ENFORCED
- Section 2.2: Database Abstraction Requirements - ENFORCED
"""

from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from models.response import SearchResult, SearchSource

try:
    from daedalus_gateway import get_graph_channel, DaedalusGraphChannel
    GRAPH_CHANNEL_AVAILABLE = True
except ImportError:
    GRAPH_CHANNEL_AVAILABLE = False

logger = logging.getLogger(__name__)


class Neo4jSearcher:
    """
    Search Neo4j graph database for relevant knowledge via Daedalus Graph Channel.

    Performs graph traversal and full-text search to find
    relevant documents, concepts, and relationships.

    Per Spec 040 M2: All Neo4j operations go through Graph Channel.
    """

    def __init__(self, graph_channel: Optional[DaedalusGraphChannel] = None):
        """
        Initialize with Daedalus Graph Channel.

        Args:
            graph_channel: Optional DaedalusGraphChannel instance (uses singleton if not provided)
        """
        self._graph_channel = graph_channel
        self._channel_initialized = False
        self.caller_service = "neo4j_searcher"

    @property
    def graph_channel(self) -> Optional[DaedalusGraphChannel]:
        """Lazy-load Graph Channel singleton."""
        if not self._channel_initialized:
            if not GRAPH_CHANNEL_AVAILABLE:
                logger.warning("Graph Channel not available - searches will return empty results")
                self._graph_channel = None
                self._channel_initialized = True
                return None

            try:
                if self._graph_channel is None:
                    self._graph_channel = get_graph_channel()
                self._channel_initialized = True
                logger.info("Connected to Neo4j via Graph Channel")
            except Exception as e:
                logger.error(f"Failed to get Graph Channel: {e}")
                self._graph_channel = None
                self._channel_initialized = True

        return self._graph_channel

    async def search(self, query: str, limit: int = 10) -> List[SearchResult]:
        """
        Search Neo4j graph for relevant results.

        Args:
            query: Natural language search query
            limit: Maximum number of results to return

        Returns:
            List of SearchResult objects from Neo4j
        """
        try:
            # Combine multiple search strategies
            results = []

            # Strategy 1: Full-text search on documents
            fulltext_results = await self._fulltext_search(query, limit)
            results.extend(fulltext_results)

            # Strategy 2: Graph pattern matching for related concepts
            graph_results = await self._graph_pattern_search(query, limit)
            results.extend(graph_results)

            # Strategy 3: Relationship traversal from found nodes
            if results:
                related_results = await self._find_related_nodes(results[:3], limit)
                results.extend(related_results)

            # Deduplicate and sort by relevance
            unique_results = self._deduplicate_results(results)
            sorted_results = sorted(unique_results, key=lambda r: r.relevance_score, reverse=True)

            return sorted_results[:limit]

        except Exception as e:
            logger.error(f"Neo4j search failed: {e}")
            return []

    async def _fulltext_search(self, query: str, limit: int) -> List[SearchResult]:
        """Full-text search across document content via Graph Channel."""
        if not self.graph_channel:
            logger.warning("Graph Channel unavailable - returning empty results")
            return []

        cypher_query = """
        CALL db.index.fulltext.queryNodes('document_content_index', $query)
        YIELD node, score
        MATCH (node:Document)
        OPTIONAL MATCH (node)-[r]->(related)
        WITH node, score, collect(type(r)) as relationships
        RETURN
            elementId(node) as result_id,
            node.extracted_text as content,
            score as relevance,
            {
                node_type: 'Document',
                filename: node.filename,
                upload_timestamp: node.upload_timestamp,
                processing_status: node.processing_status
            } as metadata,
            relationships
        ORDER BY score DESC
        LIMIT $limit
        """

        results = []
        try:
            response = await self.graph_channel.execute_read(
                query=cypher_query,
                parameters={"query": query, "limit": limit},
                caller_service=self.caller_service,
                caller_function="_fulltext_search"
            )

            if response["success"]:
                for record in response.get("records", []):
                    results.append(SearchResult(
                        result_id=record["result_id"],
                        source=SearchSource.NEO4J,
                        content=record["content"] or "",
                        relevance_score=min(float(record["relevance"]), 1.0),  # Normalize to [0,1]
                        metadata=record["metadata"],
                        relationships=record["relationships"] or []
                    ))
            else:
                logger.error(f"Fulltext search failed: {response.get('error')}")

        except Exception as e:
            logger.error(f"Error during fulltext search: {e}")

        return results

    async def _graph_pattern_search(self, query: str, limit: int) -> List[SearchResult]:
        """Search for graph patterns and concepts via Graph Channel."""
        if not self.graph_channel:
            logger.warning("Graph Channel unavailable - returning empty results")
            return []

        # Extract key terms for pattern matching
        query_terms = self._extract_key_terms(query)

        cypher_query = """
        MATCH (n)
        WHERE ANY(term IN $terms WHERE
            toLower(n.extracted_text) CONTAINS toLower(term) OR
            toLower(n.center_concept) CONTAINS toLower(term) OR
            toLower(n.type) CONTAINS toLower(term)
        )
        OPTIONAL MATCH (n)-[r]->(related)
        WITH n, count(r) as connection_count, collect(type(r)) as relationships
        RETURN
            elementId(n) as result_id,
            COALESCE(n.extracted_text, n.center_concept, n.type, '') as content,
            (toFloat(connection_count) / 10.0) as relevance,
            {
                node_type: labels(n)[0],
                properties: properties(n)
            } as metadata,
            relationships
        ORDER BY connection_count DESC
        LIMIT $limit
        """

        results = []
        try:
            response = await self.graph_channel.execute_read(
                query=cypher_query,
                parameters={"terms": query_terms, "limit": limit},
                caller_service=self.caller_service,
                caller_function="_graph_pattern_search"
            )

            if response["success"]:
                for record in response.get("records", []):
                    results.append(SearchResult(
                        result_id=record["result_id"],
                        source=SearchSource.NEO4J,
                        content=record["content"][:500],  # Limit content length
                        relevance_score=min(float(record["relevance"]), 1.0),
                        metadata=record["metadata"],
                        relationships=record["relationships"] or []
                    ))
            else:
                logger.error(f"Graph pattern search failed: {response.get('error')}")

        except Exception as e:
            logger.error(f"Error during graph pattern search: {e}")

        return results

    async def _find_related_nodes(self, seed_results: List[SearchResult], limit: int) -> List[SearchResult]:
        """Find nodes related to high-relevance results via graph traversal via Graph Channel."""
        if not seed_results:
            return []

        if not self.graph_channel:
            logger.warning("Graph Channel unavailable - returning empty results")
            return []

        # Extract node IDs from seed results
        seed_ids = [r.result_id for r in seed_results[:3]]

        cypher_query = """
        MATCH (seed)
        WHERE elementId(seed) IN $seed_ids
        MATCH (seed)-[r*1..2]-(related)
        WHERE NOT elementId(related) IN $seed_ids
        WITH DISTINCT related, count(r) as path_count, collect(type(r[0])) as relationships
        RETURN
            elementId(related) as result_id,
            COALESCE(related.extracted_text, related.center_concept, related.type, '') as content,
            (toFloat(path_count) / 5.0) as relevance,
            {
                node_type: labels(related)[0],
                properties: properties(related)
            } as metadata,
            relationships
        ORDER BY path_count DESC
        LIMIT $limit
        """

        results = []
        try:
            response = await self.graph_channel.execute_read(
                query=cypher_query,
                parameters={"seed_ids": seed_ids, "limit": limit},
                caller_service=self.caller_service,
                caller_function="_find_related_nodes"
            )

            if response["success"]:
                for record in response.get("records", []):
                    results.append(SearchResult(
                        result_id=record["result_id"],
                        source=SearchSource.NEO4J,
                        content=record["content"][:500],
                        relevance_score=min(float(record["relevance"]), 0.8),  # Cap related nodes lower
                        metadata=record["metadata"],
                        relationships=record["relationships"] or []
                    ))
            else:
                logger.error(f"Find related nodes failed: {response.get('error')}")

        except Exception as e:
            logger.error(f"Error during find related nodes: {e}")

        return results

    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract key terms from query for pattern matching."""
        # Simple term extraction - split on whitespace and filter short words
        terms = [term.strip().lower() for term in query.split() if len(term.strip()) > 3]
        return terms[:10]  # Limit to 10 most significant terms

    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate results, keeping highest relevance score."""
        seen = {}
        for result in results:
            if result.result_id not in seen or result.relevance_score > seen[result.result_id].relevance_score:
                seen[result.result_id] = result
        return list(seen.values())
