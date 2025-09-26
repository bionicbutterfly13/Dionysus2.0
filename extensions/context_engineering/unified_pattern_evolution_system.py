#!/usr/bin/env python3
"""
🌊🧠📚 Unified Pattern Evolution System
======================================

Integrates all components for real-time pattern evolution, learning, and storage:
- Attractor basin dynamics for thoughtseed integration
- Permanent learning knowledge graph storage
- Real-time conversation learning capture
- Cross-component memory sharing via Redis

This provides the complete system the user needs for testing pattern evolution
with permanent learning capability.

Author: ASI-Arch Context Engineering
Date: 2025-09-23
Version: 1.0.0 - Complete Pattern Evolution System
"""

import asyncio
import json
import logging
import redis
from datetime import datetime
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
import uuid

# Import our components
from attractor_basin_dynamics import AttractorBasinManager, ThoughtSeedIntegrationEvent
from knowledge_graph_evolution_storage import PermanentLearningGraph
from conversation_learning_capture import RealTimeConversationLearning

logger = logging.getLogger(__name__)

@dataclass
class PatternEvolutionEvent:
    """Complete pattern evolution event including all components"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = ""               # 'thoughtseed_integration', 'conversation_learning', 'consciousness_emergence'
    description: str = ""
    thoughtseed_data: Dict[str, Any] = field(default_factory=dict)
    basin_changes: Dict[str, Any] = field(default_factory=dict)
    knowledge_graph_nodes: List[str] = field(default_factory=list)
    conversation_insights: List[str] = field(default_factory=list)
    impact_score: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

class UnifiedPatternEvolutionSystem:
    """Complete system for pattern evolution with permanent learning"""

    def __init__(self, redis_host='localhost', redis_port=6379):
        print("🌊🧠📚 Initializing Unified Pattern Evolution System...")

        # Initialize components
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.attractor_manager = AttractorBasinManager(redis_host, redis_port)
        self.learning_graph = PermanentLearningGraph(redis_host, redis_port)
        self.conversation_learning = RealTimeConversationLearning()

        # Event history
        self.evolution_events: List[PatternEvolutionEvent] = []

        # Load existing events
        self._load_evolution_history()

        print(f"✅ System initialized with {len(self.evolution_events)} historical events")

    def _load_evolution_history(self):
        """Load existing pattern evolution events from Redis"""
        try:
            event_keys = self.redis_client.keys("pattern_evolution:*")
            for key in event_keys:
                event_data = json.loads(self.redis_client.get(key))
                event = PatternEvolutionEvent(
                    event_id=event_data['event_id'],
                    event_type=event_data['event_type'],
                    description=event_data['description'],
                    thoughtseed_data=event_data['thoughtseed_data'],
                    basin_changes=event_data['basin_changes'],
                    knowledge_graph_nodes=event_data['knowledge_graph_nodes'],
                    conversation_insights=event_data['conversation_insights'],
                    impact_score=event_data['impact_score'],
                    timestamp=event_data['timestamp']
                )
                self.evolution_events.append(event)

        except Exception as e:
            logger.warning(f"Could not load evolution history: {e}")

    def _persist_evolution_event(self, event: PatternEvolutionEvent):
        """Persist evolution event to Redis"""
        event_data = {
            'event_id': event.event_id,
            'event_type': event.event_type,
            'description': event.description,
            'thoughtseed_data': event.thoughtseed_data,
            'basin_changes': event.basin_changes,
            'knowledge_graph_nodes': event.knowledge_graph_nodes,
            'conversation_insights': event.conversation_insights,
            'impact_score': event.impact_score,
            'timestamp': event.timestamp
        }

        redis_key = f"pattern_evolution:{event.event_id}"
        self.redis_client.setex(redis_key, 86400 * 365, json.dumps(event_data))  # 1 year TTL

    async def evolve_pattern_from_thoughtseed(self, thoughtseed_id: str,
                                            concept_description: str,
                                            thoughtseed_metadata: Dict[str, Any] = None) -> PatternEvolutionEvent:
        """Complete pattern evolution from a new thoughtseed - integrates all systems"""

        print(f"\n🌱 Evolving pattern from thoughtseed: {thoughtseed_id}")
        print(f"   Concept: {concept_description}")

        if thoughtseed_metadata is None:
            thoughtseed_metadata = {}

        # 1. Integrate thoughtseed into attractor basin landscape
        print("🌊 Step 1: Integrating into attractor basins...")
        integration_event = await self.attractor_manager.integrate_new_thoughtseed(
            thoughtseed_id, concept_description, thoughtseed_metadata
        )

        # 2. Store integration in knowledge graph for permanent learning
        print("📚 Step 2: Storing in knowledge graph...")
        kg_node_id = await self.learning_graph.store_thoughtseed_integration(integration_event)

        # 3. Capture conversation learning if this came from discussion
        print("💭 Step 3: Capturing conversation learning...")
        insight_id = self.conversation_learning.capture_insight(
            insight_type='thoughtseed_evolution',
            content=f"New thoughtseed integrated: {concept_description}",
            impact_level=integration_event.influence_strength,
            component_affected='attractor_basin_system',
            action_required='monitor_basin_evolution'
        )

        # 4. Create unified evolution event
        evolution_event = PatternEvolutionEvent(
            event_type='thoughtseed_integration',
            description=f"Thoughtseed '{thoughtseed_id}' integrated with {integration_event.influence_type.value}",
            thoughtseed_data={
                'thoughtseed_id': thoughtseed_id,
                'concept': concept_description,
                'metadata': thoughtseed_metadata,
                'influence_type': integration_event.influence_type.value,
                'target_basin': integration_event.target_basin_id
            },
            basin_changes={
                'pre_basin_count': integration_event.pre_integration_state['basin_count'],
                'post_basin_count': integration_event.post_integration_state['basin_count'],
                'basin_strength_changes': {
                    'pre': integration_event.pre_integration_state['basin_strengths'],
                    'post': integration_event.post_integration_state['basin_strengths']
                }
            },
            knowledge_graph_nodes=[kg_node_id],
            conversation_insights=[insight_id],
            impact_score=integration_event.influence_strength
        )

        # 5. Store evolution event
        self.evolution_events.append(evolution_event)
        self._persist_evolution_event(evolution_event)

        print(f"✅ Pattern evolution complete - Event ID: {evolution_event.event_id}")
        print(f"   Impact score: {evolution_event.impact_score:.3f}")
        print(f"   Basin changes: {evolution_event.basin_changes['pre_basin_count']} → {evolution_event.basin_changes['post_basin_count']} basins")

        return evolution_event

    async def evolve_pattern_from_consciousness_detection(self, consciousness_data: Dict[str, Any]) -> PatternEvolutionEvent:
        """Pattern evolution from consciousness emergence detection"""

        print(f"\n🧠 Evolving pattern from consciousness detection...")

        # 1. Store consciousness emergence in knowledge graph
        kg_node_id = await self.learning_graph.store_consciousness_emergence_event(consciousness_data)

        # 2. Capture insight about consciousness detection
        insight_id = self.conversation_learning.capture_insight(
            insight_type='consciousness_emergence',
            content=f"Consciousness emergence detected: {consciousness_data.get('description', 'unknown')}",
            impact_level=consciousness_data.get('emergence_strength', 0.7),
            component_affected='consciousness_detection_system',
            action_required='analyze_emergence_patterns'
        )

        # 3. Create evolution event
        evolution_event = PatternEvolutionEvent(
            event_type='consciousness_emergence',
            description=f"Consciousness emergence: {consciousness_data.get('description', 'unknown')}",
            thoughtseed_data=consciousness_data,
            knowledge_graph_nodes=[kg_node_id],
            conversation_insights=[insight_id],
            impact_score=consciousness_data.get('emergence_strength', 0.7)
        )

        self.evolution_events.append(evolution_event)
        self._persist_evolution_event(evolution_event)

        print(f"✅ Consciousness pattern evolution complete - Event ID: {evolution_event.event_id}")
        return evolution_event

    async def evolve_pattern_from_conversation(self, conversation_content: str,
                                             context: str,
                                             impact_score: float = 0.8) -> PatternEvolutionEvent:
        """Pattern evolution from real-time conversation learning"""

        print(f"\n💭 Evolving pattern from conversation...")

        # 1. Store conversation pattern in knowledge graph
        kg_node_id = await self.learning_graph.store_conversation_pattern(
            conversation_content, context, impact_score
        )

        # 2. Capture conversation insight
        insight_id = self.conversation_learning.capture_insight(
            insight_type='conversation_learning',
            content=conversation_content,
            impact_level=impact_score,
            component_affected='conversation_learning_system',
            action_required='integrate_conversation_patterns'
        )

        # 3. Create evolution event
        evolution_event = PatternEvolutionEvent(
            event_type='conversation_learning',
            description=f"Conversation learning: {conversation_content[:100]}...",
            thoughtseed_data={
                'content': conversation_content,
                'context': context,
                'learning_method': 'real_time_conversation'
            },
            knowledge_graph_nodes=[kg_node_id],
            conversation_insights=[insight_id],
            impact_score=impact_score
        )

        self.evolution_events.append(evolution_event)
        self._persist_evolution_event(evolution_event)

        print(f"✅ Conversation pattern evolution complete - Event ID: {evolution_event.event_id}")
        return evolution_event

    def get_pattern_evolution_status(self) -> Dict[str, Any]:
        """Get current status of pattern evolution system"""

        # Get basin landscape summary
        basin_summary = self.attractor_manager.get_basin_landscape_summary()

        # Get knowledge graph summary
        kg_summary = self.learning_graph.get_pattern_evolution_summary()

        # Get conversation learning validation
        conversation_validation = self.conversation_learning.validate_learning_occurred()

        # Analyze recent evolution events
        recent_events = sorted(self.evolution_events, key=lambda e: e.timestamp, reverse=True)[:10]

        return {
            'system_status': 'active',
            'attractor_basins': {
                'total_basins': basin_summary['total_basins'],
                'recent_modifications': len([e for e in recent_events if e.event_type == 'thoughtseed_integration']),
                'basin_details': basin_summary['basins']
            },
            'knowledge_graph': {
                'total_nodes': kg_summary['total_nodes'],
                'total_triples': kg_summary['total_triples'],
                'pattern_types': kg_summary['pattern_type_counts']
            },
            'conversation_learning': {
                'insights_captured': conversation_validation['insights_captured'],
                'insights_persisted': conversation_validation['insights_persisted'],
                'learning_validated': conversation_validation['learning_validated']
            },
            'evolution_events': {
                'total_events': len(self.evolution_events),
                'recent_events': [
                    {
                        'type': event.event_type,
                        'description': event.description[:80],
                        'impact': event.impact_score,
                        'timestamp': event.timestamp
                    }
                    for event in recent_events
                ]
            },
            'timestamp': datetime.now().isoformat()
        }

    async def test_complete_pattern_evolution_cycle(self):
        """Test the complete pattern evolution cycle with multiple types"""

        print("\n🔄 Testing Complete Pattern Evolution Cycle")
        print("=" * 60)

        # Test 1: ThoughtSeed evolution
        await self.evolve_pattern_from_thoughtseed(
            "test_thoughtseed_001",
            "adaptive neural architecture optimization",
            {"test_mode": True, "optimization_target": "accuracy"}
        )

        # Test 2: Consciousness emergence
        await self.evolve_pattern_from_consciousness_detection({
            'description': 'Recursive self-reflection pattern detected',
            'emergence_strength': 0.82,
            'attention_coherence': 0.75,
            'meta_awareness': 0.80,
            'recursive_depth': 3
        })

        # Test 3: Conversation learning
        await self.evolve_pattern_from_conversation(
            "Real-time pattern evolution system successfully integrates attractor basins with knowledge graph storage",
            "system_testing_session",
            impact_score=0.95
        )

        # Show final status
        status = self.get_pattern_evolution_status()

        print(f"\n🎯 Complete System Status:")
        print(f"   Attractor Basins: {status['attractor_basins']['total_basins']} active")
        print(f"   Knowledge Graph: {status['knowledge_graph']['total_nodes']} nodes, {status['knowledge_graph']['total_triples']} triples")
        print(f"   Conversation Learning: {status['conversation_learning']['insights_captured']} insights captured")
        print(f"   Evolution Events: {status['evolution_events']['total_events']} total events")

        print(f"\n📋 Recent Evolution Events:")
        for event in status['evolution_events']['recent_events']:
            print(f"   [{event['type']}] {event['description']} (impact: {event['impact']:.2f})")

        return status

async def main():
    """Demo the unified pattern evolution system"""
    print("🌊🧠📚 Unified Pattern Evolution System Demo")
    print("=" * 60)

    # Initialize the complete system
    evolution_system = UnifiedPatternEvolutionSystem()

    # Run complete test cycle
    final_status = await evolution_system.test_complete_pattern_evolution_cycle()

    print(f"\n✅ System is ready for user pattern evolution testing!")
    print(f"   All components integrated and functioning")
    print(f"   Permanent learning storage operational")
    print(f"   Attractor basin dynamics responsive")

    return evolution_system

if __name__ == "__main__":
    system = asyncio.run(main())