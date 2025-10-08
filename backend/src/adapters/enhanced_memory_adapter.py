#!/usr/bin/env python3
"""
Enhanced Memory Adapter
=======================

Bridges the enhanced episodic memory orchestrator with the core Dionysus
perceptual pipeline.  The original implementation lived inside the
`dionysus-source` submodule.  This module brings the adapter into the main
backend so services can depend on it without reaching into the submodule.

Key responsibilities:
- validate NumPy constitutional requirements (Article I, Section 1.1)
- proxy memory formation/retrieval calls to the enhanced orchestrator
- expose a lightweight API that backend services can depend on
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional
import logging
import numpy as np

# ---------------------------------------------------------------------------
# Constitutional compliance: NumPy must be < 2.0 according to Article I.
# ---------------------------------------------------------------------------
if not np.__version__.startswith("1."):
    raise RuntimeError(
        f"CONSTITUTION VIOLATION: NumPy {np.__version__} detected, required < 2.0"
    )

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Attempt to import the enhanced orchestrator from dionysus-source.  If the
# submodule is not available (developer sandbox, CI without submodule, etc.)
# fall back to a stub so the adapter remains importable.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - runtime import
    from agents.unified_memory_orchestrator import (  # type: ignore
        UnifiedMemoryOrchestrator as EnhancedMemoryOrchestrator,
    )
except Exception:  # pragma: no cover - fallback path
    logger.warning(
        "EnhancedMemoryOrchestrator not available; using stub implementation."
    )

    class EnhancedMemoryOrchestrator:  # type: ignore
        """Minimal stub that mimics the orchestrator API."""

        def __init__(self) -> None:
            self._episodes: List[EpisodicMemoryTrace] = []

        async def store_episode(self, trace: "EpisodicMemoryTrace") -> str:
            self._episodes.append(trace)
            return trace.trace_id

        async def query_similar(
            self,
            context_embedding: np.ndarray,
            max_results: int = 5,
        ) -> List["EpisodicMemoryTrace"]:
            return self._episodes[:max_results]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EpisodicMemoryTrace:
    """Simple structure representing an episodic trace."""

    trace_id: str
    context_embedding: np.ndarray
    working_memory_state: Dict[str, Any]
    attractor_state: Dict[str, Any]
    perceptual_context: Dict[str, Any]
    outcome_value: float
    surprise_level: float
    timestamp: datetime
    metadata: Dict[str, Any]


# ---------------------------------------------------------------------------
# Adapter implementation
# ---------------------------------------------------------------------------

class EnhancedMemoryAdapter:
    """
    Integrates the enhanced episodic memory system with the perceptual gateway.

    The adapter is intentionally lightweight; it simply coordinates data
    structures and delegates to the orchestrator.  The orchestration logic
    lives in the submodule so we avoid duplicating complex behaviour here.
    """

    def __init__(
        self,
        perceptual_gateway: Optional[Any] = None,
        memory_orchestrator: Optional[EnhancedMemoryOrchestrator] = None,
    ) -> None:
        self.perceptual_gateway = perceptual_gateway
        self.orchestrator = memory_orchestrator or EnhancedMemoryOrchestrator()
        self.integration_stats: Dict[str, int] = {
            "episodes_stored": 0,
            "queries_run": 0,
        }
        logger.info("EnhancedMemoryAdapter initialised.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def process_perceptual_input(
        self,
        perceptual_input: Dict[str, Any],
        processed_perception: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Persist an episodic trace based on processed perception and return
        summary metadata.  This mirrors the behaviour of the original adapter
        but keeps the implementation intentionally minimal.
        """

        trace = self._build_trace(perceptual_input, processed_perception)
        await self.orchestrator.store_episode(trace)
        self.integration_stats["episodes_stored"] += 1

        logger.debug("Stored episodic trace %s", trace.trace_id)
        return {
            "trace_id": trace.trace_id,
            "stored_at": trace.timestamp.isoformat(),
            "outcome_value": trace.outcome_value,
            "surprise_level": trace.surprise_level,
        }

    async def query_enhanced_memory(
        self,
        query_embedding: Iterable[float],
        max_results: int = 5,
    ) -> List[EpisodicMemoryTrace]:
        """
        Retrieve episodes similar to the supplied embedding.  Consumers can
        feed these results back into the perceptual pipeline for context.
        """

        embedding = np.asarray(list(query_embedding), dtype=float)
        results = await self.orchestrator.query_similar(embedding, max_results)
        self.integration_stats["queries_run"] += 1
        logger.debug("Retrieved %d episodic results", len(results))
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_trace(
        self,
        perceptual_input: Dict[str, Any],
        processed_perception: Dict[str, Any],
    ) -> EpisodicMemoryTrace:
        """
        Construct an EpisodicMemoryTrace from raw inputs.  The exact structure
        mirrors what the original adapter produced, but values are simplified
        to keep the implementation dependency-light.
        """

        context_embedding = np.asarray(
            processed_perception.get("context_embedding", [0.0, 0.0, 0.0]),
            dtype=float,
        )

        working_state = processed_perception.get("working_memory", {})
        attractor_state = processed_perception.get("attractor_state", {})

        outcome_value = float(processed_perception.get("quality", 0.0))
        surprise_level = float(processed_perception.get("surprise", 0.0))

        metadata = {
            "source": perceptual_input.get("source", "unknown"),
            "tags": perceptual_input.get("tags", []),
        }

        trace = EpisodicMemoryTrace(
            trace_id=f"trace_{datetime.utcnow().timestamp():.6f}",
            context_embedding=context_embedding,
            working_memory_state=working_state,
            attractor_state=attractor_state,
            perceptual_context=perceptual_input,
            outcome_value=outcome_value,
            surprise_level=surprise_level,
            timestamp=datetime.utcnow(),
            metadata=metadata,
        )
        return trace


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def create_enhanced_memory_adapter(
    perceptual_gateway: Optional[Any] = None,
    memory_orchestrator: Optional[EnhancedMemoryOrchestrator] = None,
) -> EnhancedMemoryAdapter:
    """
    Convenience factory used by services.  Mirrors the helper provided in the
    submodule so downstream code can keep the same import surface.
    """

    return EnhancedMemoryAdapter(
        perceptual_gateway=perceptual_gateway,
        memory_orchestrator=memory_orchestrator,
    )
