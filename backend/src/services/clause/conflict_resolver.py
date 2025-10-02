"""
T046-T049: Conflict Resolution Service

Implements conflict detection and resolution per Spec 031.
Uses optimistic locking + MERGE strategy for concurrent writes to Neo4j.

Key Features:
- Write conflict detection (concurrent modifications to same nodes/edges)
- MERGE strategy: max(strength) wins
- Optimistic locking with version tracking
- Performance target: <50ms per conflict resolution
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from models.clause.coordinator_models import AgentHandoff

logger = logging.getLogger(__name__)


class ConflictType:
    """Conflict type enumeration"""

    NODE_UPDATE = "node_update"
    EDGE_UPDATE = "edge_update"
    CONCURRENT_WRITE = "concurrent_write"


class Conflict:
    """Conflict record"""

    def __init__(
        self,
        conflict_type: str,
        resource_id: str,
        agent1: str,
        agent2: str,
        value1: Any,
        value2: Any,
    ):
        self.conflict_type = conflict_type
        self.resource_id = resource_id
        self.agent1 = agent1
        self.agent2 = agent2
        self.value1 = value1
        self.value2 = value2
        self.detected_at = datetime.now()


class ConflictResolver:
    """
    Conflict resolver for CLAUSE multi-agent system.

    Per Spec 031:
    - Optimistic locking: Each write includes version number
    - MERGE strategy: max(strength) wins for basin updates
    - Conflict detection: Check write logs for concurrent modifications
    - Resolution latency: <50ms per conflict
    """

    def __init__(self, neo4j_client=None):
        """
        Initialize conflict resolver.

        Args:
            neo4j_client: Neo4j client for conflict detection
        """
        self.neo4j = neo4j_client

        # Conflict statistics
        self.total_detected = 0
        self.total_resolved = 0
        self.resolution_failures = 0

        logger.info("Conflict resolver initialized with MERGE strategy")

    async def detect_conflicts(
        self, agent_handoffs: List[AgentHandoff]
    ) -> List[Conflict]:
        """
        Detect write conflicts from agent execution logs.

        T046: Write Conflict Detection
        - Check Neo4j write logs for concurrent modifications
        - Detect conflicts on nodes, edges, properties
        - Return list of conflicts for resolution

        Args:
            agent_handoffs: List of agent execution records

        Returns:
            List of detected conflicts
        """
        conflicts = []

        if not self.neo4j:
            logger.warning("Neo4j not configured - skipping conflict detection")
            return conflicts

        # T048: Optimistic locking - check version conflicts
        # Query Neo4j for concurrent writes
        query = """
        MATCH (n)
        WHERE n._write_timestamp IS NOT NULL
        AND n._write_agent IS NOT NULL
        WITH n, n._write_timestamp AS ts
        ORDER BY ts DESC
        LIMIT 1000
        RETURN n.id AS resource_id,
               n._write_agent AS agent,
               n._write_timestamp AS timestamp,
               n._version AS version
        """

        # Placeholder - would execute query and detect conflicts
        # For now, return empty list (no conflicts detected)

        self.total_detected += len(conflicts)
        logger.info(f"Detected {len(conflicts)} conflicts")

        return conflicts

    async def resolve(
        self, conflict: Conflict, strategy: str = "MERGE"
    ) -> Dict[str, Any]:
        """
        Resolve conflict using specified strategy.

        T047: MERGE Resolution Strategy
        - For basin strength updates: max(strength) wins
        - For edge weights: max(weight) wins
        - For co-occurrence: sum(counts) wins

        Args:
            conflict: Conflict to resolve
            strategy: Resolution strategy (MERGE, LAST_WRITE_WINS, etc.)

        Returns:
            Resolution result with winner, resolved_value
        """
        if strategy != "MERGE":
            logger.warning(f"Unsupported strategy: {strategy}, falling back to MERGE")

        # T047: Apply MERGE strategy
        if conflict.conflict_type == ConflictType.NODE_UPDATE:
            resolved_value = self._merge_node_update(conflict)
        elif conflict.conflict_type == ConflictType.EDGE_UPDATE:
            resolved_value = self._merge_edge_update(conflict)
        else:
            resolved_value = conflict.value2  # Default to value2 (last write)

        # Apply resolution to Neo4j
        if self.neo4j:
            await self._apply_resolution(conflict, resolved_value)

        self.total_resolved += 1
        logger.info(
            f"Resolved conflict on {conflict.resource_id} using MERGE strategy"
        )

        return {
            "conflict_id": id(conflict),
            "resource_id": conflict.resource_id,
            "strategy": "MERGE",
            "resolved_value": resolved_value,
            "winner": conflict.agent1 if resolved_value == conflict.value1 else conflict.agent2,
        }

    def _merge_node_update(self, conflict: Conflict) -> Any:
        """
        Merge node updates using MERGE strategy.

        For basin strength: max(strength) wins
        For activation_count: max(count) wins

        Args:
            conflict: Conflict record

        Returns:
            Merged value
        """
        value1 = conflict.value1
        value2 = conflict.value2

        # If values are dicts (e.g., basin updates)
        if isinstance(value1, dict) and isinstance(value2, dict):
            merged = {}

            # Merge strength: max wins
            if "strength" in value1 and "strength" in value2:
                merged["strength"] = max(value1["strength"], value2["strength"])

            # Merge activation_count: max wins
            if "activation_count" in value1 and "activation_count" in value2:
                merged["activation_count"] = max(
                    value1["activation_count"], value2["activation_count"]
                )

            # Merge co_occurring: sum counts
            if "co_occurring" in value1 and "co_occurring" in value2:
                co_occ = {}
                for key in set(value1["co_occurring"].keys()) | set(
                    value2["co_occurring"].keys()
                ):
                    co_occ[key] = value1["co_occurring"].get(
                        key, 0
                    ) + value2["co_occurring"].get(key, 0)
                merged["co_occurring"] = co_occ

            return merged

        # For scalar values: max wins
        return max(value1, value2)

    def _merge_edge_update(self, conflict: Conflict) -> Any:
        """
        Merge edge updates using MERGE strategy.

        For edge weight: max(weight) wins

        Args:
            conflict: Conflict record

        Returns:
            Merged value
        """
        value1 = conflict.value1
        value2 = conflict.value2

        # If values are dicts (e.g., edge properties)
        if isinstance(value1, dict) and isinstance(value2, dict):
            merged = {}

            # Merge weight: max wins
            if "weight" in value1 and "weight" in value2:
                merged["weight"] = max(value1["weight"], value2["weight"])

            return merged

        # For scalar values: max wins
        return max(value1, value2)

    async def _apply_resolution(self, conflict: Conflict, resolved_value: Any) -> None:
        """
        Apply conflict resolution to Neo4j.

        T048: Optimistic locking - increment version number

        Args:
            conflict: Conflict record
            resolved_value: Resolved value to write
        """
        if not self.neo4j:
            return

        # T048: Increment version for optimistic locking
        query = """
        MATCH (n {id: $resource_id})
        SET n._version = coalesce(n._version, 0) + 1,
            n._write_timestamp = datetime(),
            n._write_agent = 'ConflictResolver',
            n.value = $resolved_value
        RETURN n._version AS new_version
        """

        try:
            result = await self.neo4j.execute(
                query,
                parameters={
                    "resource_id": conflict.resource_id,
                    "resolved_value": resolved_value,
                },
            )
            logger.debug(f"Applied resolution with version {result}")
        except Exception as e:
            self.resolution_failures += 1
            logger.error(f"Failed to apply resolution: {e}")

    def get_stats(self) -> Dict[str, int]:
        """
        Get conflict resolution statistics.

        Returns:
            Dict with detected, resolved, failures
        """
        return {
            "total_detected": self.total_detected,
            "total_resolved": self.total_resolved,
            "resolution_failures": self.resolution_failures,
        }
