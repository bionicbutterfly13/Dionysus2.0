#!/usr/bin/env python3
"""
🌊 Attractor Basin Dynamics for ThoughtSeed Integration
=====================================================

Implements the mechanism for altering attractor basins when new concepts
and thoughtseeds are introduced to the system. This handles the dynamic
reorganization of the cognitive landscape.

Core Functions:
- Attractor basin modification when new thoughtseeds enter
- Basin strength calculation and rebalancing
- Pattern evolution storage in knowledge graph
- Cross-basin influence and competition dynamics

Author: ASI-Arch Context Engineering
Date: 2025-09-23
Version: 1.0.0 - Real-time Basin Dynamics
"""

import asyncio
import json
import logging
import numpy as np
import redis
from datetime import datetime
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid

logger = logging.getLogger(__name__)

class BasinInfluenceType(Enum):
    """Types of influence between attractor basins"""
    REINFORCEMENT = "reinforcement"     # New thoughtseed strengthens existing basin
    COMPETITION = "competition"         # New thoughtseed competes with existing basin
    SYNTHESIS = "synthesis"             # New thoughtseed merges with existing basin
    EMERGENCE = "emergence"             # New thoughtseed creates entirely new basin

@dataclass
class AttractorBasin:
    """An attractor basin in the cognitive landscape"""
    basin_id: str
    center_concept: str                 # Central concept that defines the basin
    strength: float = 1.0               # Basin depth/strength (0.0 - 2.0)
    radius: float = 0.5                 # Basin influence radius
    thoughtseeds: Set[str] = field(default_factory=set)  # ThoughtSeeds in this basin
    related_concepts: Dict[str, float] = field(default_factory=dict)  # concept -> similarity
    formation_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    last_modification: str = field(default_factory=lambda: datetime.now().isoformat())
    activation_history: List[float] = field(default_factory=list)

    def calculate_influence_on(self, new_concept: str, concept_similarity: float) -> Tuple[BasinInfluenceType, float]:
        """Calculate how this basin would influence a new concept/thoughtseed"""

        # High similarity -> reinforcement or synthesis
        if concept_similarity > 0.8:
            if self.strength > 1.5:
                return BasinInfluenceType.REINFORCEMENT, concept_similarity * self.strength
            else:
                return BasinInfluenceType.SYNTHESIS, concept_similarity * 0.8

        # Medium similarity -> competition or synthesis
        elif concept_similarity > 0.5:
            if self.strength > 1.0:
                return BasinInfluenceType.COMPETITION, (1.0 - concept_similarity) * self.strength
            else:
                return BasinInfluenceType.SYNTHESIS, concept_similarity * 0.6

        # Low similarity -> emergence (new basin)
        else:
            return BasinInfluenceType.EMERGENCE, 1.0 - concept_similarity

@dataclass
class ThoughtSeedIntegrationEvent:
    """Event representing integration of a new thoughtseed into the basin landscape"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    thoughtseed_id: str = ""
    concept_description: str = ""
    target_basin_id: Optional[str] = None
    influence_type: BasinInfluenceType = BasinInfluenceType.EMERGENCE
    influence_strength: float = 0.0
    pre_integration_state: Dict[str, Any] = field(default_factory=dict)
    post_integration_state: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

class AttractorBasinManager:
    """Manages the dynamic attractor basin landscape for thoughtseed integration"""

    def __init__(self, redis_host='localhost', redis_port=6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.basins: Dict[str, AttractorBasin] = {}
        self.integration_events: List[ThoughtSeedIntegrationEvent] = []

        # Load existing basins from Redis
        self._load_basins_from_memory()

        logger.info(f"🌊 AttractorBasinManager initialized with {len(self.basins)} existing basins")

    def _load_basins_from_memory(self):
        """Load existing attractor basins from Redis memory"""
        try:
            basin_keys = self.redis_client.keys("attractor_basin:*")
            for key in basin_keys:
                basin_data = json.loads(self.redis_client.get(key))
                basin = AttractorBasin(
                    basin_id=basin_data['basin_id'],
                    center_concept=basin_data['center_concept'],
                    strength=basin_data['strength'],
                    radius=basin_data['radius'],
                    thoughtseeds=set(basin_data['thoughtseeds']),
                    related_concepts=basin_data['related_concepts'],
                    formation_timestamp=basin_data['formation_timestamp'],
                    last_modification=basin_data['last_modification'],
                    activation_history=basin_data['activation_history']
                )
                self.basins[basin.basin_id] = basin

        except Exception as e:
            logger.warning(f"Could not load basins from memory: {e}")
            # Create default basin if none exist
            if not self.basins:
                self._create_default_basin()

    def _create_default_basin(self):
        """Create a default attractor basin for system initialization"""
        default_basin = AttractorBasin(
            basin_id="default_cognitive_basin",
            center_concept="general_learning",
            strength=1.0,
            radius=0.7,
            thoughtseeds=set(),
            related_concepts={"learning": 0.9, "adaptation": 0.8, "pattern_recognition": 0.7}
        )
        self.basins[default_basin.basin_id] = default_basin
        self._persist_basin(default_basin)

    def _persist_basin(self, basin: AttractorBasin):
        """Persist basin to Redis with TTL"""
        basin_data = {
            'basin_id': basin.basin_id,
            'center_concept': basin.center_concept,
            'strength': basin.strength,
            'radius': basin.radius,
            'thoughtseeds': list(basin.thoughtseeds),
            'related_concepts': basin.related_concepts,
            'formation_timestamp': basin.formation_timestamp,
            'last_modification': basin.last_modification,
            'activation_history': basin.activation_history
        }

        redis_key = f"attractor_basin:{basin.basin_id}"
        self.redis_client.setex(redis_key, 86400 * 7, json.dumps(basin_data))  # 1 week TTL

    def calculate_concept_similarity(self, concept1: str, concept2: str) -> float:
        """Calculate semantic similarity between two concepts"""
        # Simple implementation - in production, use semantic embeddings
        concept1_words = set(concept1.lower().split())
        concept2_words = set(concept2.lower().split())

        if not concept1_words or not concept2_words:
            return 0.0

        intersection = len(concept1_words.intersection(concept2_words))
        union = len(concept1_words.union(concept2_words))

        jaccard_similarity = intersection / union if union > 0 else 0.0

        # Bonus for related terms (semantic enhancement)
        semantic_bonus = 0.0
        related_terms = {
            'learning': ['adaptation', 'training', 'evolution', 'improvement'],
            'consciousness': ['awareness', 'attention', 'reflection', 'cognition'],
            'pattern': ['structure', 'organization', 'template', 'framework'],
            'inference': ['prediction', 'reasoning', 'deduction', 'conclusion']
        }

        for term, related in related_terms.items():
            if term in concept1.lower() and any(r in concept2.lower() for r in related):
                semantic_bonus += 0.2
            if term in concept2.lower() and any(r in concept1.lower() for r in related):
                semantic_bonus += 0.2

        return min(1.0, jaccard_similarity + semantic_bonus)

    async def integrate_new_thoughtseed(self, thoughtseed_id: str, concept_description: str,
                                       thoughtseed_data: Dict[str, Any]) -> ThoughtSeedIntegrationEvent:
        """Integrate a new thoughtseed into the attractor basin landscape"""

        print(f"🌱 Integrating new thoughtseed: {thoughtseed_id}")
        print(f"   Concept: {concept_description}")

        # Record pre-integration state
        pre_state = {
            'basin_count': len(self.basins),
            'basin_strengths': {bid: basin.strength for bid, basin in self.basins.items()},
            'total_thoughtseeds': sum(len(basin.thoughtseeds) for basin in self.basins.values())
        }

        # Find best matching basin and influence type
        best_basin_id = None
        best_influence_type = BasinInfluenceType.EMERGENCE
        best_influence_strength = 0.0

        for basin_id, basin in self.basins.items():
            similarity = self.calculate_concept_similarity(concept_description, basin.center_concept)
            influence_type, influence_strength = basin.calculate_influence_on(concept_description, similarity)

            print(f"   Basin '{basin_id}': similarity={similarity:.3f}, influence={influence_type.value}, strength={influence_strength:.3f}")

            if influence_strength > best_influence_strength:
                best_basin_id = basin_id
                best_influence_type = influence_type
                best_influence_strength = influence_strength

        # Apply the integration based on influence type
        if best_influence_type == BasinInfluenceType.REINFORCEMENT:
            await self._reinforce_basin(best_basin_id, thoughtseed_id, concept_description)

        elif best_influence_type == BasinInfluenceType.COMPETITION:
            await self._create_competing_basin(thoughtseed_id, concept_description, best_basin_id)

        elif best_influence_type == BasinInfluenceType.SYNTHESIS:
            await self._synthesize_with_basin(best_basin_id, thoughtseed_id, concept_description)

        else:  # EMERGENCE
            await self._create_new_basin(thoughtseed_id, concept_description)

        # Record post-integration state
        post_state = {
            'basin_count': len(self.basins),
            'basin_strengths': {bid: basin.strength for bid, basin in self.basins.items()},
            'total_thoughtseeds': sum(len(basin.thoughtseeds) for basin in self.basins.values())
        }

        # Create integration event
        integration_event = ThoughtSeedIntegrationEvent(
            thoughtseed_id=thoughtseed_id,
            concept_description=concept_description,
            target_basin_id=best_basin_id,
            influence_type=best_influence_type,
            influence_strength=best_influence_strength,
            pre_integration_state=pre_state,
            post_integration_state=post_state
        )

        self.integration_events.append(integration_event)

        # Store integration event in Redis for pattern tracking
        self._persist_integration_event(integration_event)

        print(f"✅ Integration complete: {best_influence_type.value} with strength {best_influence_strength:.3f}")
        return integration_event

    async def _reinforce_basin(self, basin_id: str, thoughtseed_id: str, concept: str):
        """Reinforce an existing basin with a similar thoughtseed"""
        basin = self.basins[basin_id]
        basin.thoughtseeds.add(thoughtseed_id)
        basin.strength = min(2.0, basin.strength + 0.2)  # Increase strength, cap at 2.0
        basin.last_modification = datetime.now().isoformat()
        basin.activation_history.append(basin.strength)

        # Keep activation history reasonable size
        if len(basin.activation_history) > 50:
            basin.activation_history = basin.activation_history[-50:]

        self._persist_basin(basin)
        print(f"   🔋 Reinforced basin '{basin_id}' - new strength: {basin.strength:.3f}")

    async def _create_competing_basin(self, thoughtseed_id: str, concept: str, competing_basin_id: str):
        """Create a new basin that competes with an existing one"""
        competing_basin = self.basins[competing_basin_id]

        # Reduce competing basin strength slightly
        competing_basin.strength = max(0.1, competing_basin.strength - 0.1)
        competing_basin.last_modification = datetime.now().isoformat()

        # Create new basin with moderate strength
        new_basin = AttractorBasin(
            basin_id=f"basin_{thoughtseed_id}_{int(datetime.now().timestamp())}",
            center_concept=concept,
            strength=0.8,  # Start with moderate strength
            radius=0.4,    # Smaller radius for competing basin
            thoughtseeds={thoughtseed_id}
        )

        self.basins[new_basin.basin_id] = new_basin
        self._persist_basin(competing_basin)
        self._persist_basin(new_basin)
        print(f"   ⚔️ Created competing basin '{new_basin.basin_id}' vs '{competing_basin_id}'")

    async def _synthesize_with_basin(self, basin_id: str, thoughtseed_id: str, concept: str):
        """Synthesize new thoughtseed with existing basin - merge concepts"""
        basin = self.basins[basin_id]
        basin.thoughtseeds.add(thoughtseed_id)

        # Update basin center concept to be more general/inclusive
        similarity = self.calculate_concept_similarity(concept, basin.center_concept)
        if similarity > 0.7:
            # Merge similar concepts
            concept_words = set(concept.lower().split())
            basin_words = set(basin.center_concept.lower().split())
            merged_words = concept_words.union(basin_words)
            basin.center_concept = " ".join(sorted(merged_words))

        # Moderate strength increase and radius expansion
        basin.strength = min(2.0, basin.strength + 0.1)
        basin.radius = min(1.0, basin.radius + 0.1)
        basin.last_modification = datetime.now().isoformat()

        self._persist_basin(basin)
        print(f"   🔄 Synthesized with basin '{basin_id}' - concept: '{basin.center_concept}'")

    async def _create_new_basin(self, thoughtseed_id: str, concept: str):
        """Create entirely new basin for emergent concept"""
        new_basin = AttractorBasin(
            basin_id=f"basin_{thoughtseed_id}_{int(datetime.now().timestamp())}",
            center_concept=concept,
            strength=1.0,  # Start with neutral strength
            radius=0.5,    # Standard radius
            thoughtseeds={thoughtseed_id}
        )

        self.basins[new_basin.basin_id] = new_basin
        self._persist_basin(new_basin)
        print(f"   🌟 Created new emergent basin '{new_basin.basin_id}'")

    def _persist_integration_event(self, event: ThoughtSeedIntegrationEvent):
        """Store integration event in Redis for pattern analysis"""
        event_data = {
            'event_id': event.event_id,
            'thoughtseed_id': event.thoughtseed_id,
            'concept_description': event.concept_description,
            'target_basin_id': event.target_basin_id,
            'influence_type': event.influence_type.value,
            'influence_strength': event.influence_strength,
            'pre_integration_state': event.pre_integration_state,
            'post_integration_state': event.post_integration_state,
            'timestamp': event.timestamp
        }

        redis_key = f"integration_event:{event.event_id}"
        self.redis_client.setex(redis_key, 86400 * 30, json.dumps(event_data))  # 30 day TTL

    def get_basin_landscape_summary(self) -> Dict[str, Any]:
        """Get current state of attractor basin landscape"""
        return {
            'total_basins': len(self.basins),
            'basins': {
                basin_id: {
                    'center_concept': basin.center_concept,
                    'strength': basin.strength,
                    'radius': basin.radius,
                    'thoughtseed_count': len(basin.thoughtseeds),
                    'last_modification': basin.last_modification
                }
                for basin_id, basin in self.basins.items()
            },
            'recent_integrations': len(self.integration_events),
            'landscape_timestamp': datetime.now().isoformat()
        }

    async def evolve_basin_landscape(self):
        """Periodic evolution of the basin landscape - decay unused basins, strengthen active ones"""
        print("🌊 Evolving attractor basin landscape...")

        current_time = datetime.now()
        modified_basins = []

        for basin_id, basin in list(self.basins.items()):
            # Calculate time since last modification
            last_mod = datetime.fromisoformat(basin.last_modification)
            days_inactive = (current_time - last_mod).days

            # Decay inactive basins
            if days_inactive > 7:
                decay_rate = min(0.1, days_inactive * 0.01)
                basin.strength = max(0.1, basin.strength - decay_rate)
                modified_basins.append(basin_id)

                # Remove very weak basins
                if basin.strength < 0.2:
                    print(f"   🪦 Removing weak basin '{basin_id}' (strength: {basin.strength:.3f})")
                    del self.basins[basin_id]
                    self.redis_client.delete(f"attractor_basin:{basin_id}")
                    continue

            # Persist modified basins
            if basin_id in modified_basins:
                self._persist_basin(basin)

        print(f"✅ Basin landscape evolution complete - {len(modified_basins)} basins modified")

def main():
    """Demo of attractor basin dynamics"""
    import asyncio

    async def demo():
        print("🌊🧠 Attractor Basin Dynamics Demo")
        print("=" * 50)

        # Initialize basin manager
        manager = AttractorBasinManager()

        # Show initial landscape
        summary = manager.get_basin_landscape_summary()
        print(f"\nInitial landscape: {summary['total_basins']} basins")

        # Integrate several new thoughtseeds
        test_thoughtseeds = [
            ("ts_001", "pattern recognition in neural networks"),
            ("ts_002", "deep learning adaptation mechanisms"),
            ("ts_003", "consciousness emergence indicators"),
            ("ts_004", "memory consolidation across components"),
            ("ts_005", "attention mechanism optimization"),
            ("ts_006", "quantum computing applications")
        ]

        for ts_id, concept in test_thoughtseeds:
            event = await manager.integrate_new_thoughtseed(ts_id, concept, {})
            print(f"\nEvent: {event.influence_type.value} with strength {event.influence_strength:.3f}")

        # Show final landscape
        final_summary = manager.get_basin_landscape_summary()
        print(f"\n🎯 Final landscape summary:")
        print(f"   Total basins: {final_summary['total_basins']}")
        for basin_id, basin_info in final_summary['basins'].items():
            print(f"   {basin_id}: '{basin_info['center_concept']}' (strength: {basin_info['strength']:.2f})")

        # Test landscape evolution
        await manager.evolve_basin_landscape()

    asyncio.run(demo())

if __name__ == "__main__":
    main()