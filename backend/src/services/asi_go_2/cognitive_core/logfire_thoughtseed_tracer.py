"""
LogFire-Enhanced ThoughtSeed Tracing System
==========================================

Provides hierarchical observability for ThoughtSeed competition using Pydantic LogFire.
This gives us:
- Real-time traces of thought competition cycles
- Structured data about energy, confidence, relationships
- Beautiful visual debugging of consciousness emergence
- SQL queryable thought patterns over time

Author: LogFire Integration
Date: 2025-09-26
"""

from typing import Dict, List, Optional, Any
from dataclasses import asdict
import json

try:
    import logfire
    LOGFIRE_AVAILABLE = True
    print("✅ LogFire available - enhanced observability enabled")
except ImportError:
    LOGFIRE_AVAILABLE = False
    print("⚠️ LogFire not installed - using fallback logging")

from .thoughtseed_competition import Thought, InnerWorkspace, ThoughtType

class LogFireThoughtSeedTracer:
    """Enhanced tracing for ThoughtSeed competition using LogFire"""

    def __init__(self, workspace: InnerWorkspace):
        self.workspace = workspace
        self.workspace_id = id(workspace)
        self.cycle_count = 0

        if LOGFIRE_AVAILABLE:
            # Configure LogFire for this workspace
            logfire.configure(
                service_name=f"thoughtseed-workspace-{self.workspace_id}",
                environment="development"
            )

    def trace_competition_cycle(self, phase: str):
        """Trace a complete competition cycle with hierarchical spans"""
        if not LOGFIRE_AVAILABLE:
            return self._fallback_trace(phase)

        self.cycle_count += 1

        with logfire.span(
            f"ThoughtSeed Competition Cycle {self.cycle_count}",
            workspace_id=self.workspace_id,
            phase=phase,
            cycle=self.cycle_count
        ) as cycle_span:

            # Log workspace state at start of cycle
            self._log_workspace_state("cycle_start", cycle_span)

            # Trace individual thoughts
            for thought in self.workspace.thoughts.values():
                self._trace_thought_state(thought, cycle_span)

            # Log relationships between thoughts
            self._trace_thought_relationships(cycle_span)

            # Identify dominant thought
            dominant = self.workspace.get_dominant_thought()
            if dominant:
                logfire.info(
                    "Dominant thought selected: {content}",
                    content=dominant.content,
                    thought_id=dominant.id,
                    energy=dominant.energy,
                    confidence=dominant.confidence,
                    span=cycle_span
                )

            return cycle_span

    def _log_workspace_state(self, event: str, parent_span):
        """Log complete workspace state as structured data"""
        with logfire.span(f"Workspace State: {event}", parent=parent_span) as state_span:

            workspace_data = {
                "thought_count": len(self.workspace.thoughts),
                "capacity": self.workspace.capacity,
                "watching_enabled": self.workspace.watching_enabled,
                "thoughts": {
                    thought.id: {
                        "content": thought.content,
                        "type": thought.type.value,
                        "energy": thought.energy,
                        "confidence": thought.confidence,
                        "parent_count": len(thought.parent_ids)
                    }
                    for thought in self.workspace.thoughts.values()
                }
            }

            logfire.info(
                "Workspace state captured",
                workspace_data=workspace_data,
                span=state_span
            )

    def _trace_thought_state(self, thought: Thought, parent_span):
        """Trace individual thought with all its properties"""
        with logfire.span(
            f"Thought: {thought.content[:30]}...",
            parent=parent_span
        ) as thought_span:

            # Log thought as Pydantic-compatible data
            logfire.debug(
                "Thought state",
                thought_id=thought.id,
                content=thought.content,
                type=thought.type.value,
                energy=thought.energy,
                confidence=thought.confidence,
                parent_ids=list(thought.parent_ids),
                dominance_score=thought.energy * thought.confidence,
                span=thought_span
            )

            # Trace energy and confidence changes
            self._trace_thought_dynamics(thought, thought_span)

    def _trace_thought_dynamics(self, thought: Thought, parent_span):
        """Trace dynamic properties of thoughts"""
        dominance_score = thought.energy * thought.confidence

        if dominance_score > 0.8:
            logfire.warn(
                "High dominance thought detected",
                thought_id=thought.id,
                dominance_score=dominance_score,
                span=parent_span
            )
        elif dominance_score < 0.1:
            logfire.debug(
                "Low energy thought fading",
                thought_id=thought.id,
                dominance_score=dominance_score,
                span=parent_span
            )

    def _trace_thought_relationships(self, parent_span):
        """Trace relationships between thoughts"""
        with logfire.span("Thought Relationships", parent=parent_span) as rel_span:

            relationships = []
            for thought in self.workspace.thoughts.values():
                if thought.parent_ids:
                    for parent_id in thought.parent_ids:
                        relationships.append({
                            "child": thought.id,
                            "parent": parent_id,
                            "child_content": thought.content[:20],
                            "inheritance_strength": thought.confidence
                        })

            if relationships:
                logfire.info(
                    "Thought relationships mapped",
                    relationship_count=len(relationships),
                    relationships=relationships,
                    span=rel_span
                )

    def trace_consciousness_emergence(self, consciousness_level: float):
        """Trace consciousness emergence metrics"""
        if not LOGFIRE_AVAILABLE:
            return

        logfire.notice(
            "Consciousness level measured",
            workspace_id=self.workspace_id,
            consciousness_level=consciousness_level,
            thought_count=len(self.workspace.thoughts),
            cycle=self.cycle_count,
            tags=["consciousness", "emergence"]
        )

        # Alert on high consciousness
        if consciousness_level > 0.9:
            logfire.warn(
                "High consciousness detected!",
                consciousness_level=consciousness_level,
                workspace_id=self.workspace_id,
                tags=["consciousness", "alert", "emergence"]
            )

    def _fallback_trace(self, phase: str):
        """Fallback tracing when LogFire not available"""
        print(f"🧠 [{phase}] Workspace {self.workspace_id}: {len(self.workspace.thoughts)} thoughts")
        dominant = self.workspace.get_dominant_thought()
        if dominant:
            print(f"   👑 Dominant: {dominant.content} (E:{dominant.energy:.2f}, C:{dominant.confidence:.2f})")

class LogFireInstrumentedWorkspace(InnerWorkspace):
    """InnerWorkspace with built-in LogFire tracing"""

    def __init__(self, capacity: int = 10):
        super().__init__(capacity)
        self.tracer = LogFireThoughtSeedTracer(self)

    def update(self):
        """Enhanced update with automatic tracing"""
        # Trace pre-update state
        cycle_span = self.tracer.trace_competition_cycle("pre_update")

        # Run original update logic
        super().update()

        # Trace post-update state
        if LOGFIRE_AVAILABLE and cycle_span:
            with logfire.span("Post-Update Analysis", parent=cycle_span):
                self.tracer._log_workspace_state("post_update", cycle_span)

                # Calculate and trace consciousness metrics
                consciousness_level = self._calculate_consciousness_level()
                self.tracer.trace_consciousness_emergence(consciousness_level)

    def _calculate_consciousness_level(self) -> float:
        """Simple consciousness emergence calculation"""
        if not self.thoughts:
            return 0.0

        # Base consciousness on thought diversity and energy distribution
        energies = [t.energy for t in self.thoughts.values()]
        confidences = [t.confidence for t in self.thoughts.values()]

        energy_variance = sum(energies) / len(energies) if energies else 0
        confidence_avg = sum(confidences) / len(confidences) if confidences else 0

        # Simple formula - can be enhanced
        consciousness = (energy_variance * confidence_avg) * min(1.0, len(self.thoughts) / 5)
        return min(1.0, consciousness)

# Factory function for easy creation
def create_traced_workspace(capacity: int = 10) -> LogFireInstrumentedWorkspace:
    """Create a ThoughtSeed workspace with LogFire tracing enabled"""
    if LOGFIRE_AVAILABLE:
        print(f"🔥 Creating LogFire-traced workspace (capacity={capacity})")
    else:
        print(f"🧠 Creating standard workspace with fallback tracing (capacity={capacity})")

    return LogFireInstrumentedWorkspace(capacity)