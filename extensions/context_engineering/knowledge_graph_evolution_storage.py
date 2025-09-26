#!/usr/bin/env python3
"""
📚🧠 Knowledge Graph Evolution Storage
=====================================

Stores pattern evolution, thoughtseed integration, and attractor basin changes
in a persistent knowledge graph for permanent learning capability.

This addresses the user's urgent need for:
- Pattern evolution storage in knowledge graph
- Permanent learning from all discussions and pattern tests
- Increasingly effective knowledge graph management

Author: ASI-Arch Context Engineering
Date: 2025-09-23
Version: 1.0.0 - Permanent Learning Graph
"""

import asyncio
import json
import logging
import redis
from datetime import datetime
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
import uuid

# Import our attractor basin system
from attractor_basin_dynamics import AttractorBasinManager, ThoughtSeedIntegrationEvent, BasinInfluenceType

logger = logging.getLogger(__name__)

@dataclass
class PatternEvolutionNode:
    """A node in the knowledge graph representing pattern evolution"""
    node_id: str
    pattern_type: str                   # 'thoughtseed', 'basin_modification', 'consciousness_emergence'
    description: str
    timestamp: str
    source_event: str                   # ID of the event that created this pattern
    influence_strength: float
    related_nodes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class KnowledgeGraphTriple:
    """A triple in the knowledge graph (subject, predicate, object)"""
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    source_event: str = ""

class PermanentLearningGraph:
    """Knowledge graph system for permanent learning storage"""

    def __init__(self, redis_host='localhost', redis_port=6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.nodes: Dict[str, PatternEvolutionNode] = {}
        self.triples: List[KnowledgeGraphTriple] = []
        self.attractor_manager = AttractorBasinManager(redis_host, redis_port)

        # Initialize knowledge graph
        self._load_existing_graph()

        logger.info(f"📚 PermanentLearningGraph initialized with {len(self.nodes)} nodes and {len(self.triples)} triples")

    def _load_existing_graph(self):
        """Load existing knowledge graph from Redis"""
        try:
            # Load nodes
            node_keys = self.redis_client.keys("kg_node:*")
            for key in node_keys:
                node_data = json.loads(self.redis_client.get(key))
                node = PatternEvolutionNode(
                    node_id=node_data['node_id'],
                    pattern_type=node_data['pattern_type'],
                    description=node_data['description'],
                    timestamp=node_data['timestamp'],
                    source_event=node_data['source_event'],
                    influence_strength=node_data['influence_strength'],
                    related_nodes=node_data['related_nodes'],
                    metadata=node_data['metadata']
                )
                self.nodes[node.node_id] = node

            # Load triples
            triple_keys = self.redis_client.keys("kg_triple:*")
            for key in triple_keys:
                triple_data = json.loads(self.redis_client.get(key))
                triple = KnowledgeGraphTriple(
                    subject=triple_data['subject'],
                    predicate=triple_data['predicate'],
                    object=triple_data['object'],
                    confidence=triple_data['confidence'],
                    timestamp=triple_data['timestamp'],
                    source_event=triple_data['source_event']
                )
                self.triples.append(triple)

        except Exception as e:
            logger.warning(f"Could not load existing graph: {e}")

    def _persist_node(self, node: PatternEvolutionNode):
        """Persist a node to Redis"""
        node_data = {
            'node_id': node.node_id,
            'pattern_type': node.pattern_type,
            'description': node.description,
            'timestamp': node.timestamp,
            'source_event': node.source_event,
            'influence_strength': node.influence_strength,
            'related_nodes': node.related_nodes,
            'metadata': node.metadata
        }

        redis_key = f"kg_node:{node.node_id}"
        self.redis_client.setex(redis_key, 86400 * 365, json.dumps(node_data))  # 1 year TTL for permanent learning

    def _persist_triple(self, triple: KnowledgeGraphTriple):
        """Persist a triple to Redis"""
        triple_data = {
            'subject': triple.subject,
            'predicate': triple.predicate,
            'object': triple.object,
            'confidence': triple.confidence,
            'timestamp': triple.timestamp,
            'source_event': triple.source_event
        }

        triple_id = f"{hash(triple.subject + triple.predicate + triple.object)}"
        redis_key = f"kg_triple:{triple_id}"
        self.redis_client.setex(redis_key, 86400 * 365, json.dumps(triple_data))  # 1 year TTL

    async def store_thoughtseed_integration(self, integration_event: ThoughtSeedIntegrationEvent) -> str:
        """Store a thoughtseed integration event in the knowledge graph"""

        print(f"📚 Storing thoughtseed integration in knowledge graph...")

        # Create node for the thoughtseed itself
        thoughtseed_node = PatternEvolutionNode(
            node_id=f"thoughtseed_{integration_event.thoughtseed_id}",
            pattern_type="thoughtseed",
            description=integration_event.concept_description,
            timestamp=integration_event.timestamp,
            source_event=integration_event.event_id,
            influence_strength=integration_event.influence_strength,
            metadata={
                'influence_type': integration_event.influence_type.value,
                'target_basin': integration_event.target_basin_id,
                'pre_state': integration_event.pre_integration_state,
                'post_state': integration_event.post_integration_state
            }
        )

        self.nodes[thoughtseed_node.node_id] = thoughtseed_node
        self._persist_node(thoughtseed_node)

        # Create node for the integration event
        integration_node = PatternEvolutionNode(
            node_id=f"integration_{integration_event.event_id}",
            pattern_type="basin_modification",
            description=f"{integration_event.influence_type.value} integration event",
            timestamp=integration_event.timestamp,
            source_event=integration_event.event_id,
            influence_strength=integration_event.influence_strength,
            metadata={
                'basin_count_change': integration_event.post_integration_state['basin_count'] - integration_event.pre_integration_state['basin_count'],
                'thoughtseed_count_change': integration_event.post_integration_state['total_thoughtseeds'] - integration_event.pre_integration_state['total_thoughtseeds']
            }
        )

        self.nodes[integration_node.node_id] = integration_node
        self._persist_node(integration_node)

        # Create triples to connect the nodes
        triples_to_add = [
            KnowledgeGraphTriple(
                subject=thoughtseed_node.node_id,
                predicate="triggered_integration",
                object=integration_node.node_id,
                confidence=integration_event.influence_strength,
                source_event=integration_event.event_id
            ),
            KnowledgeGraphTriple(
                subject=integration_node.node_id,
                predicate="has_influence_type",
                object=integration_event.influence_type.value,
                confidence=1.0,
                source_event=integration_event.event_id
            )
        ]

        # If there's a target basin, connect to it
        if integration_event.target_basin_id:
            basin_node_id = f"basin_{integration_event.target_basin_id}"
            triples_to_add.append(
                KnowledgeGraphTriple(
                    subject=thoughtseed_node.node_id,
                    predicate="integrated_into_basin",
                    object=basin_node_id,
                    confidence=integration_event.influence_strength,
                    source_event=integration_event.event_id
                )
            )

        # Add triples to graph
        for triple in triples_to_add:
            self.triples.append(triple)
            self._persist_triple(triple)

        print(f"✅ Stored thoughtseed integration: {len(triples_to_add)} triples added")
        return thoughtseed_node.node_id

    async def store_conversation_pattern(self, pattern_description: str,
                                       pattern_context: str,
                                       impact_score: float = 0.8) -> str:
        """Store a pattern discovered during conversation"""

        pattern_node = PatternEvolutionNode(
            node_id=f"pattern_{int(datetime.now().timestamp())}_{hash(pattern_description) % 10000}",
            pattern_type="conversation_learning",
            description=pattern_description,
            timestamp=datetime.now().isoformat(),
            source_event="conversation_learning",
            influence_strength=impact_score,
            metadata={
                'context': pattern_context,
                'discovery_method': 'real_time_conversation'
            }
        )

        self.nodes[pattern_node.node_id] = pattern_node
        self._persist_node(pattern_node)

        # Create triple connecting to conversation learning
        learning_triple = KnowledgeGraphTriple(
            subject=pattern_node.node_id,
            predicate="discovered_during",
            object="real_time_conversation",
            confidence=impact_score,
            source_event="conversation_learning"
        )

        self.triples.append(learning_triple)
        self._persist_triple(learning_triple)

        print(f"📝 Stored conversation pattern: '{pattern_description[:50]}...'")
        return pattern_node.node_id

    async def store_consciousness_emergence_event(self, emergence_data: Dict[str, Any]) -> str:
        """Store consciousness emergence detection event"""

        consciousness_node = PatternEvolutionNode(
            node_id=f"consciousness_{int(datetime.now().timestamp())}",
            pattern_type="consciousness_emergence",
            description=f"Consciousness emergence: {emergence_data.get('description', 'unknown')}",
            timestamp=datetime.now().isoformat(),
            source_event=emergence_data.get('event_id', 'consciousness_detection'),
            influence_strength=emergence_data.get('emergence_strength', 0.7),
            metadata=emergence_data
        )

        self.nodes[consciousness_node.node_id] = consciousness_node
        self._persist_node(consciousness_node)

        # Create triples for consciousness indicators
        consciousness_triples = []
        for indicator, value in emergence_data.items():
            if isinstance(value, (int, float)) and 0 <= value <= 1:
                consciousness_triples.append(
                    KnowledgeGraphTriple(
                        subject=consciousness_node.node_id,
                        predicate=f"has_{indicator}",
                        object=str(value),
                        confidence=value,
                        source_event=emergence_data.get('event_id', 'consciousness_detection')
                    )
                )

        for triple in consciousness_triples:
            self.triples.append(triple)
            self._persist_triple(triple)

        print(f"🧠 Stored consciousness emergence: {len(consciousness_triples)} indicators")
        return consciousness_node.node_id

    def query_related_patterns(self, query_concept: str, limit: int = 10) -> List[PatternEvolutionNode]:
        """Query for patterns related to a concept"""

        # Simple text-based matching for now
        related_nodes = []
        query_words = set(query_concept.lower().split())

        for node in self.nodes.values():
            node_words = set(node.description.lower().split())
            similarity = len(query_words.intersection(node_words)) / len(query_words.union(node_words))

            if similarity > 0.1:  # Minimum similarity threshold
                related_nodes.append((node, similarity))

        # Sort by similarity and return top results
        related_nodes.sort(key=lambda x: x[1], reverse=True)
        return [node for node, similarity in related_nodes[:limit]]

    def get_pattern_evolution_summary(self) -> Dict[str, Any]:
        """Get summary of pattern evolution stored in the knowledge graph"""

        pattern_counts = {}
        for node in self.nodes.values():
            pattern_counts[node.pattern_type] = pattern_counts.get(node.pattern_type, 0) + 1

        recent_patterns = sorted(
            self.nodes.values(),
            key=lambda n: n.timestamp,
            reverse=True
        )[:10]

        return {
            'total_nodes': len(self.nodes),
            'total_triples': len(self.triples),
            'pattern_type_counts': pattern_counts,
            'recent_patterns': [
                {
                    'type': node.pattern_type,
                    'description': node.description[:100],
                    'timestamp': node.timestamp,
                    'influence': node.influence_strength
                }
                for node in recent_patterns
            ],
            'graph_timestamp': datetime.now().isoformat()
        }

    async def start_permanent_learning_monitoring(self):
        """Start monitoring for events to automatically store in knowledge graph"""
        print("🔄 Starting permanent learning monitoring...")

        # Monitor Redis for new integration events
        while True:
            try:
                # Check for new integration events
                integration_keys = self.redis_client.keys("integration_event:*")
                for key in integration_keys:
                    event_data = json.loads(self.redis_client.get(key))

                    # Check if we've already processed this event
                    event_id = event_data['event_id']
                    existing_node = f"integration_{event_id}"

                    if existing_node not in self.nodes:
                        # Create integration event from stored data
                        integration_event = ThoughtSeedIntegrationEvent(
                            event_id=event_data['event_id'],
                            thoughtseed_id=event_data['thoughtseed_id'],
                            concept_description=event_data['concept_description'],
                            target_basin_id=event_data['target_basin_id'],
                            influence_type=BasinInfluenceType(event_data['influence_type']),
                            influence_strength=event_data['influence_strength'],
                            pre_integration_state=event_data['pre_integration_state'],
                            post_integration_state=event_data['post_integration_state'],
                            timestamp=event_data['timestamp']
                        )

                        await self.store_thoughtseed_integration(integration_event)

                # Monitor for conversation insights
                insight_keys = self.redis_client.keys("episodic:insight:*")
                for key in insight_keys:
                    try:
                        insight_data = json.loads(self.redis_client.get(key))
                        insight_id = key.split(":")[-1]

                        # Check if we've processed this insight
                        pattern_node_id = f"insight_{insight_id}"
                        if pattern_node_id not in self.nodes:
                            await self.store_conversation_pattern(
                                pattern_description=insight_data['content'],
                                pattern_context=insight_data['context'],
                                impact_score=insight_data['impact_level']
                            )
                    except Exception as e:
                        logger.warning(f"Could not process insight {key}: {e}")

                # Sleep before next monitoring cycle
                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Error in permanent learning monitoring: {e}")
                await asyncio.sleep(60)  # Wait longer on error

async def main():
    """Demo of permanent learning graph"""
    print("📚🧠 Permanent Learning Graph Demo")
    print("=" * 50)

    # Initialize the learning graph
    learning_graph = PermanentLearningGraph()

    # Store some sample patterns
    await learning_graph.store_conversation_pattern(
        "Vector embedding fraud eliminated - now using honest error handling",
        "fraud_elimination_session",
        impact_score=0.95
    )

    await learning_graph.store_conversation_pattern(
        "Attractor basin dynamics implemented for thoughtseed integration",
        "pattern_evolution_implementation",
        impact_score=0.90
    )

    await learning_graph.store_consciousness_emergence_event({
        'description': 'Pattern recognition consciousness emergence detected',
        'emergence_strength': 0.85,
        'attention_coherence': 0.78,
        'recursive_processing': 0.82,
        'meta_awareness': 0.75
    })

    # Show summary
    summary = learning_graph.get_pattern_evolution_summary()
    print(f"\n🎯 Knowledge Graph Summary:")
    print(f"   Total nodes: {summary['total_nodes']}")
    print(f"   Total triples: {summary['total_triples']}")
    print(f"   Pattern types: {summary['pattern_type_counts']}")

    print("\n📋 Recent patterns:")
    for pattern in summary['recent_patterns']:
        print(f"   [{pattern['type']}] {pattern['description']} (impact: {pattern['influence']:.2f})")

    # Test pattern query
    print("\n🔍 Querying patterns related to 'consciousness':")
    related = learning_graph.query_related_patterns("consciousness", limit=5)
    for node in related:
        print(f"   - {node.description[:80]}... (strength: {node.influence_strength:.2f})")

if __name__ == "__main__":
    asyncio.run(main())