"""
CLAUSE Graph Loader - Graph Channel to NetworkX subgraph extraction

Loads k-hop subgraphs from Neo4j via Daedalus Graph Channel for CLAUSE Subgraph Architect processing.
Implements NFR-005 retry logic with exponential backoff.

CONSTITUTIONAL COMPLIANCE (Spec 040 M2):
- Uses daedalus_gateway.get_graph_channel() for ALL graph operations
- NO direct neo4j imports (constitutional violation)
- Async operations for integration with Graph Channel
"""

import asyncio
import time
from typing import Dict, List, Optional, Set, Any
import networkx as nx
import logging

try:
    from daedalus_gateway import get_graph_channel
    GRAPH_CHANNEL_AVAILABLE = True
except ImportError:
    GRAPH_CHANNEL_AVAILABLE = False
    get_graph_channel = None

logger = logging.getLogger(__name__)


class GraphLoader:
    """
    Load k-hop subgraphs from Neo4j via Daedalus Graph Channel into NetworkX.

    CONSTITUTIONAL COMPLIANCE (Spec 040 M2):
    - ALL graph operations go through DaedalusGraphChannel
    - NO direct neo4j driver usage
    - Async methods for Graph Channel integration

    Supports CLAUSE Phase 1 subgraph construction with retry logic
    for connection failures (NFR-005: 3 retries with exponential backoff).
    """

    def __init__(self):
        """Initialize with Graph Channel."""
        if not GRAPH_CHANNEL_AVAILABLE:
            raise RuntimeError(
                "daedalus_gateway not available. "
                "Install with: pip install daedalus-gateway"
            )

        self.graph_channel = get_graph_channel()
        self._connected = False

    async def ensure_connected(self):
        """Ensure Graph Channel is connected."""
        if not self._connected:
            success = await self.graph_channel.connect()
            if not success:
                raise ConnectionError("Failed to connect to Graph Channel")
            self._connected = True

    async def load_subgraph_from_neo4j(
        self,
        query: str,
        hop_distance: int = 2,
        max_seed_nodes: int = 20,
    ) -> nx.MultiDiGraph:
        """
        Load k-hop subgraph from Neo4j via Graph Channel based on query.

        CONSTITUTIONAL COMPLIANCE: Uses Graph Channel for all queries.

        Implements NFR-005 retry logic: 3 retries with exponential backoff
        (100ms, 200ms, 400ms) before raising exception.

        Args:
            query: Search query for finding seed nodes
            hop_distance: Maximum hops from seed nodes (default 2)
            max_seed_nodes: Maximum seed nodes to expand from (default 20)

        Returns:
            NetworkX MultiDiGraph with nodes and edges from Neo4j

        Raises:
            ConnectionError: After 3 failed retries (NFR-005)
        """
        await self.ensure_connected()

        retries = 3
        backoff_delays = [0.1, 0.2, 0.4]  # 100ms, 200ms, 400ms

        for attempt in range(retries):
            try:
                return await self._load_subgraph_internal(
                    query, hop_distance, max_seed_nodes
                )
            except Exception as e:
                logger.warning(
                    f"Graph subgraph load attempt {attempt+1}/{retries} failed: {e}"
                )

                if attempt < retries - 1:
                    delay = backoff_delays[attempt]
                    logger.info(f"Retrying in {delay*1000:.0f}ms...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"Graph subgraph load failed after {retries} retries"
                    )
                    raise ConnectionError(
                        f"Graph connection failed after {retries} retries "
                        f"with exponential backoff"
                    ) from e

        # Should never reach here, but satisfy type checker
        raise ConnectionError("Unexpected error in retry logic")

    async def _load_subgraph_internal(
        self,
        query: str,
        hop_distance: int,
        max_seed_nodes: int,
    ) -> nx.MultiDiGraph:
        """
        Internal subgraph loading logic (retryable).

        CONSTITUTIONAL COMPLIANCE: All queries via Graph Channel.
        """

        # Step 1: Find seed nodes via full-text search
        seed_node_ids = await self._get_seed_nodes(query, max_seed_nodes)

        if not seed_node_ids:
            logger.warning(f"No seed nodes found for query: {query}")
            return nx.MultiDiGraph()

        # Step 2: Expand k-hop neighborhood
        subgraph_nodes = await self._expand_khop_neighborhood(
            seed_node_ids, hop_distance
        )

        # Step 3: Load subgraph edges
        graph = await self._build_networkx_graph(subgraph_nodes)

        logger.info(
            f"Loaded subgraph: {len(graph.nodes)} nodes, "
            f"{len(graph.edges)} edges from {len(seed_node_ids)} seeds"
        )

        return graph

    async def _get_seed_nodes(self, query: str, limit: int) -> List[str]:
        """
        Get seed nodes using full-text search on knowledge graph.

        CONSTITUTIONAL COMPLIANCE: Uses Graph Channel execute_read().

        Uses existing knowledge_search_index on KnowledgeTriple nodes.
        """
        cypher_query = """
        CALL db.index.fulltext.queryNodes('knowledge_search_index', $query)
        YIELD node, score
        WHERE node:KnowledgeTriple
        RETURN elementId(node) as node_id,
               node.subject as subject,
               score
        ORDER BY score DESC
        LIMIT $limit
        """

        result = await self.graph_channel.execute_read(
            query=cypher_query,
            parameters={"query": query, "limit": limit},
            caller_service="clause_graph_loader",
            caller_function="_get_seed_nodes"
        )

        if result["success"]:
            seed_ids = [record["node_id"] for record in result["records"]]
            return seed_ids
        else:
            logger.error(f"Failed to get seed nodes: {result.get('error')}")
            return []

    async def _expand_khop_neighborhood(
        self, seed_ids: List[str], hop_distance: int
    ) -> Set[str]:
        """
        Expand k-hop neighborhood from seed nodes using BFS.

        CONSTITUTIONAL COMPLIANCE: Uses Graph Channel execute_read().

        Note: Using manual BFS instead of APOC for compatibility.
        APOC optimization (apoc.path.subgraphNodes) can be added in T029.
        """
        all_nodes: Set[str] = set(seed_ids)
        current_layer = set(seed_ids)

        for hop in range(hop_distance):
            cypher_query = """
            MATCH (start)
            WHERE elementId(start) IN $node_ids
            MATCH (start)-[r]-(neighbor)
            RETURN DISTINCT elementId(neighbor) as neighbor_id
            """

            result = await self.graph_channel.execute_read(
                query=cypher_query,
                parameters={"node_ids": list(current_layer)},
                caller_service="clause_graph_loader",
                caller_function="_expand_khop_neighborhood"
            )

            if result["success"]:
                next_layer = {record["neighbor_id"] for record in result["records"]}
                next_layer -= all_nodes  # Remove already visited

                all_nodes.update(next_layer)
                current_layer = next_layer

                if not next_layer:
                    logger.info(f"BFS stopped at hop {hop+1} (no new nodes)")
                    break
            else:
                logger.error(f"Failed to expand hop {hop}: {result.get('error')}")
                break

        return all_nodes

    async def _build_networkx_graph(self, node_ids: Set[str]) -> nx.MultiDiGraph:
        """
        Build NetworkX MultiDiGraph from Neo4j nodes and edges.

        CONSTITUTIONAL COMPLIANCE: Uses Graph Channel execute_read().

        Loads node attributes (concept_id, basin_id) and edge attributes
        (relation_type, weight) for CLAUSE edge scoring.
        """
        graph = nx.MultiDiGraph()

        # Load nodes with attributes
        node_query = """
        MATCH (n)
        WHERE elementId(n) IN $node_ids
        OPTIONAL MATCH (n)<-[:HAS_BASIN]-(basin:AttractorBasin)
        RETURN
            elementId(n) as node_id,
            labels(n) as labels,
            n.subject as subject,
            n.predicate as predicate,
            n.object as object,
            elementId(basin) as basin_id,
            basin.strength as basin_strength
        """

        result = await self.graph_channel.execute_read(
            query=node_query,
            parameters={"node_ids": list(node_ids)},
            caller_service="clause_graph_loader",
            caller_function="_build_networkx_graph"
        )

        if result["success"]:
            for record in result["records"]:
                node_id = record["node_id"]

                # Extract concept_id from subject/object
                concept_id = (
                    record.get("subject") or record.get("object") or node_id
                )

                graph.add_node(
                    node_id,
                    concept_id=concept_id,
                    basin_id=record.get("basin_id"),
                    basin_strength=record.get("basin_strength") or 1.0,
                    labels=record.get("labels", []),
                    subject=record.get("subject"),
                    predicate=record.get("predicate"),
                    object=record.get("object"),
                )
        else:
            logger.error(f"Failed to load nodes: {result.get('error')}")

        # Load edges with attributes
        edge_query = """
        MATCH (source)-[r]->(target)
        WHERE elementId(source) IN $node_ids
          AND elementId(target) IN $node_ids
        RETURN
            elementId(source) as source_id,
            elementId(target) as target_id,
            type(r) as relation_type,
            r.weight as weight
        """

        result = await self.graph_channel.execute_read(
            query=edge_query,
            parameters={"node_ids": list(node_ids)},
            caller_service="clause_graph_loader",
            caller_function="_build_networkx_graph"
        )

        if result["success"]:
            for record in result["records"]:
                graph.add_edge(
                    record["source_id"],
                    record["target_id"],
                    relation=record["relation_type"],
                    weight=record.get("weight") or 1.0,
                )
        else:
            logger.error(f"Failed to load edges: {result.get('error')}")

        return graph

    async def get_basin_info(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """
        Get basin information for a concept.

        CONSTITUTIONAL COMPLIANCE: Uses Graph Channel execute_read().

        Used by EdgeScorer to retrieve basin strength for edge scoring.
        """
        await self.ensure_connected()

        cypher_query = """
        MATCH (k:KnowledgeTriple)
        WHERE k.subject = $concept_id OR k.object = $concept_id
        OPTIONAL MATCH (k)<-[:HAS_BASIN]-(basin:AttractorBasin)
        RETURN
            elementId(basin) as basin_id,
            basin.strength as strength,
            basin.activation_count as activation_count,
            basin.co_occurring_concepts as co_occurring_concepts
        LIMIT 1
        """

        try:
            result = await self.graph_channel.execute_read(
                query=cypher_query,
                parameters={"concept_id": concept_id},
                caller_service="clause_graph_loader",
                caller_function="get_basin_info"
            )

            if result["success"] and result["records"]:
                record = result["records"][0]

                if record.get("basin_id"):
                    return {
                        "basin_id": record["basin_id"],
                        "strength": record.get("strength") or 1.0,
                        "activation_count": record.get("activation_count") or 0,
                        "co_occurring_concepts": (
                            record.get("co_occurring_concepts") or {}
                        ),
                    }
                else:
                    # No basin for this concept yet
                    return None
            else:
                return None

        except Exception as e:
            logger.error(f"Failed to get basin info for {concept_id}: {e}")
            return None
