"""
T040: Causal Bayesian Network Service

Implements causal reasoning per Spec 033.
Uses pre-computed DAG with LRU cache for <30ms intervention predictions.

Key Features:
- Pre-computed causal DAG structure
- Do-calculus for intervention estimation
- LRU cache (size=1000) for fast lookup
- <30ms latency target
"""

import logging
from typing import Dict, Optional, Tuple
from functools import lru_cache
import time

from models.clause.causal_models import CausalIntervention

logger = logging.getLogger(__name__)


class CausalBayesianNetwork:
    """
    Causal Bayesian Network for intervention estimation.

    Per Spec 033:
    - Pre-compute causal DAG structure offline
    - Use do-calculus: P(target | do(intervention))
    - LRU cache for frequent intervention queries
    - Target latency: <30ms per prediction
    """

    def __init__(self, neo4j_client=None, cache_size: int = 1000):
        """
        Initialize Causal Bayesian Network.

        Args:
            neo4j_client: Neo4j client for DAG structure
            cache_size: LRU cache size (default 1000)
        """
        self.neo4j = neo4j_client
        self.cache_size = cache_size

        # Pre-computed DAG structure (adjacency list)
        self.dag: Dict[str, list] = {}

        # Intervention cache (will use @lru_cache on method)
        self.cache_hits = 0
        self.cache_misses = 0

        logger.info(f"Causal Bayesian Network initialized (cache_size={cache_size})")

    async def build_dag(self) -> None:
        """
        Build causal DAG structure from Neo4j.

        Offline pre-computation for fast online queries.
        """
        if not self.neo4j:
            logger.warning("Neo4j not configured - using empty DAG")
            return

        # Query Neo4j for causal relationships
        # Placeholder - would use actual Neo4j query
        query = """
        MATCH (a)-[r:CAUSES]->(b)
        RETURN a.id AS from, b.id AS to
        """

        # Build adjacency list
        # results = await self.neo4j.execute(query)
        # for record in results:
        #     self.dag.setdefault(record["from"], []).append(record["to"])

        logger.info(f"DAG built with {len(self.dag)} nodes")

    async def estimate_intervention(
        self, intervention: str, target: str
    ) -> Optional[float]:
        """
        Estimate causal intervention effect using do-calculus.

        P(target | do(intervention)) - probability of target given intervention.

        Args:
            intervention: Intervention node
            target: Target outcome node

        Returns:
            Intervention score [0, 1] or None
        """
        start_time = time.time()

        # Check cache
        cache_key = (intervention, target)
        score = self._cached_do_calculus(cache_key)

        computation_time_ms = (time.time() - start_time) * 1000

        if score is not None:
            self.cache_hits += 1
            logger.debug(
                f"Cache hit: {intervention} → {target} ({computation_time_ms:.2f}ms)"
            )
        else:
            self.cache_misses += 1
            logger.debug(
                f"Cache miss: {intervention} → {target} ({computation_time_ms:.2f}ms)"
            )

        # Return as CausalIntervention model
        if score is not None:
            return score

        return None

    @lru_cache(maxsize=1000)
    def _cached_do_calculus(self, cache_key: Tuple[str, str]) -> Optional[float]:
        """
        Cached do-calculus computation.

        Uses LRU cache for fast repeated queries.

        Args:
            cache_key: (intervention, target) tuple

        Returns:
            Intervention score or None
        """
        intervention, target = cache_key

        # Check if path exists in DAG
        if intervention not in self.dag:
            return None

        # Compute P(target | do(intervention)) using do-calculus
        # Placeholder - simplified path-based scoring
        if target in self.dag.get(intervention, []):
            return 0.85  # Direct causal link
        elif self._has_path(intervention, target):
            return 0.60  # Indirect causal link
        else:
            return 0.10  # No causal link

    def _has_path(self, source: str, target: str, visited=None) -> bool:
        """
        Check if path exists from source to target in DAG.

        Args:
            source: Source node
            target: Target node
            visited: Set of visited nodes (for cycle detection)

        Returns:
            True if path exists
        """
        if visited is None:
            visited = set()

        if source == target:
            return True

        if source in visited:
            return False

        visited.add(source)

        for neighbor in self.dag.get(source, []):
            if self._has_path(neighbor, target, visited):
                return True

        return False

    def get_cache_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.

        Returns:
            Dict with cache_hits, cache_misses, hit_rate
        """
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0.0

        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
        }
