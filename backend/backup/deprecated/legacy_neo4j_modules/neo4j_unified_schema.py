#!/usr/bin/env python3
"""
🌐 Neo4j Unified Knowledge Graph Schema for ASI-Arch Context Flow
================================================================

This module defines the complete Neo4j schema that unifies:
- Neural Architecture Discovery (from ASI-Arch)
- Consciousness Detection & River Metaphor Framework
- Episodic Meta-Learning & Autobiographical Memory
- Archetypal Resonance Patterns & Active Inference

Author: ASI-Arch Context Engineering Extension
Date: 2025-09-22
Version: 1.0.0 - Unified Knowledge Graph Schema
"""

from neo4j import GraphDatabase
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

# =============================================================================
# Schema Enums and Types
# =============================================================================

class NodeType(Enum):
    """Core node types in the unified knowledge graph"""
    # Neural Architecture Nodes
    ARCHITECTURE = "Architecture"
    COMPONENT = "Component"
    LAYER = "Layer"
    OPERATION = "Operation"
    
    # Consciousness & Context Nodes
    CONSCIOUSNESS_STATE = "ConsciousnessState"
    CONTEXT_STREAM = "ContextStream"
    ATTRACTOR_BASIN = "AttractorBasin"
    NEURAL_FIELD = "NeuralField"
    
    # Episodic & Memory Nodes
    EPISODE = "Episode"
    MEMORY_TRACE = "MemoryTrace"
    TASK_CONTEXT = "TaskContext"
    AUTOBIOGRAPHICAL_EVENT = "AutobiographicalEvent"
    
    # Archetypal & Narrative Nodes
    ARCHETYPE = "Archetype"
    NARRATIVE_PATTERN = "NarrativePattern"
    RESONANCE_FIELD = "ResonanceField"
    
    # Meta-Learning Nodes
    META_STRATEGY = "MetaStrategy"
    LEARNING_TRAJECTORY = "LearningTrajectory"
    ADAPTATION_POINT = "AdaptationPoint"

class RelationType(Enum):
    """Relationship types in the unified knowledge graph"""
    # Architecture Relationships
    CONTAINS = "CONTAINS"
    EVOLVED_FROM = "EVOLVED_FROM"
    SIMILAR_TO = "SIMILAR_TO"
    OUTPERFORMS = "OUTPERFORMS"
    
    # Consciousness Relationships
    EXHIBITS_CONSCIOUSNESS = "EXHIBITS_CONSCIOUSNESS"
    FLOWS_INTO = "FLOWS_INTO"
    ATTRACTED_TO = "ATTRACTED_TO"
    RESONATES_WITH = "RESONATES_WITH"
    
    # Episodic Relationships
    OCCURRED_IN = "OCCURRED_IN"
    TRIGGERED_BY = "TRIGGERED_BY"
    REMEMBERED_AS = "REMEMBERED_AS"
    SIMILAR_EPISODE = "SIMILAR_EPISODE"
    
    # Archetypal Relationships
    EMBODIES_ARCHETYPE = "EMBODIES_ARCHETYPE"
    FOLLOWS_PATTERN = "FOLLOWS_PATTERN"
    TRANSFORMS_INTO = "TRANSFORMS_INTO"
    
    # Meta-Learning Relationships
    LEARNED_FROM = "LEARNED_FROM"
    ADAPTED_TO = "ADAPTED_TO"
    GENERALIZES_TO = "GENERALIZES_TO"

# =============================================================================
# Neo4j Schema Manager
# =============================================================================

@dataclass
class Neo4jUnifiedSchema:
    """Unified Neo4j schema for ASI-Arch Context Flow knowledge graph"""
    
    def __init__(self, uri: str = "bolt://localhost:7687", 
                 user: str = "neo4j", 
                 password: str = "password"):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None
        self.logger = logging.getLogger(__name__)
    
    def connect(self):
        """Connect to Neo4j database and verify connectivity"""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            # Verify the connection actually works
            self.driver.verify_connectivity()
            self.logger.info(f"Connected to Neo4j at {self.uri}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to Neo4j: {e}")
            self.driver = None
            return False
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            self.logger.info("Neo4j connection closed")
    
    def create_constraints_and_indexes(self):
        """Create all necessary constraints and indexes for optimal performance"""
        
        constraints_and_indexes = [
            # Unique constraints for core identifiers
            "CREATE CONSTRAINT architecture_id_unique IF NOT EXISTS FOR (a:Architecture) REQUIRE a.id IS UNIQUE",
            "CREATE CONSTRAINT episode_id_unique IF NOT EXISTS FOR (e:Episode) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT consciousness_state_id_unique IF NOT EXISTS FOR (c:ConsciousnessState) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT archetype_id_unique IF NOT EXISTS FOR (arch:Archetype) REQUIRE arch.id IS UNIQUE",
            
            # Performance indexes for frequent queries
            "CREATE INDEX architecture_performance_idx IF NOT EXISTS FOR (a:Architecture) ON (a.performance_score)",
            "CREATE INDEX consciousness_level_idx IF NOT EXISTS FOR (c:ConsciousnessState) ON (c.level, c.score)",
            "CREATE INDEX episode_timestamp_idx IF NOT EXISTS FOR (e:Episode) ON (e.timestamp)",
            "CREATE INDEX archetype_resonance_idx IF NOT EXISTS FOR (arch:Archetype) ON (arch.resonance_strength)",
            
            # Full-text search indexes for narrative content
            "CREATE FULLTEXT INDEX architecture_description_fulltext IF NOT EXISTS FOR (a:Architecture) ON EACH [a.description, a.motivation]",
            "CREATE FULLTEXT INDEX episode_narrative_fulltext IF NOT EXISTS FOR (e:Episode) ON EACH [e.title, e.narrative_summary]",
            "CREATE FULLTEXT INDEX pattern_description_fulltext IF NOT EXISTS FOR (p:NarrativePattern) ON EACH [p.description]",
            
            # Vector similarity indexes (for AutoSchemaKG integration)
            "CREATE VECTOR INDEX architecture_embedding_vector IF NOT EXISTS FOR (a:Architecture) ON (a.embedding) OPTIONS {indexConfig: {`vector.dimensions`: 512, `vector.similarity_function`: 'cosine'}}",
            "CREATE VECTOR INDEX episode_embedding_vector IF NOT EXISTS FOR (e:Episode) ON (e.embedding) OPTIONS {indexConfig: {`vector.dimensions`: 512, `vector.similarity_function`: 'cosine'}}"
        ]
        
        with self.driver.session() as session:
            for constraint_or_index in constraints_and_indexes:
                try:
                    session.run(constraint_or_index)
                    self.logger.debug(f"Created: {constraint_or_index}")
                except Exception as e:
                    self.logger.warning(f"Constraint/Index creation failed (may already exist): {e}")
        
        self.logger.info("✅ All constraints and indexes created/verified")
    
    def create_architecture_node(self, architecture_data: Dict[str, Any]) -> str:
        """Create a neural architecture node with full context"""
        
        cypher = """
        CREATE (a:Architecture {
            id: $id,
            name: $name,
            program: $program,
            result: $result,
            motivation: $motivation,
            performance_score: $performance_score,
            consciousness_level: $consciousness_level,
            consciousness_score: $consciousness_score,
            embedding: $embedding,
            created_at: $created_at,
            metadata: $metadata
        })
        RETURN a.id as architecture_id
        """
        
        with self.driver.session() as session:
            result = session.run(cypher, **architecture_data)
            architecture_id = result.single()["architecture_id"]
            self.logger.debug(f"Created architecture node: {architecture_id}")
            return architecture_id
    
    def create_episode_node(self, episode_data: Dict[str, Any]) -> str:
        """Create an episodic memory node"""
        
        cypher = """
        CREATE (e:Episode {
            id: $id,
            title: $title,
            narrative_summary: $narrative_summary,
            start_evaluation: $start_evaluation,
            end_evaluation: $end_evaluation,
            episode_duration: $episode_duration,
            exploration_phase: $exploration_phase,
            dominant_archetype: $dominant_archetype,
            narrative_coherence_score: $narrative_coherence_score,
            archetypal_resonance_strength: $archetypal_resonance_strength,
            embedding: $embedding,
            timestamp: $timestamp,
            metadata: $metadata
        })
        RETURN e.id as episode_id
        """
        
        with self.driver.session() as session:
            result = session.run(cypher, **episode_data)
            episode_id = result.single()["episode_id"]
            self.logger.debug(f"Created episode node: {episode_id}")
            return episode_id
    
    def create_consciousness_state_node(self, consciousness_data: Dict[str, Any]) -> str:
        """Create a consciousness state node"""
        
        cypher = """
        CREATE (c:ConsciousnessState {
            id: $id,
            level: $level,
            score: $score,
            emergence_indicators: $emergence_indicators,
            self_awareness_markers: $self_awareness_markers,
            meta_cognitive_depth: $meta_cognitive_depth,
            timestamp: $timestamp,
            metadata: $metadata
        })
        RETURN c.id as consciousness_id
        """
        
        with self.driver.session() as session:
            result = session.run(cypher, **consciousness_data)
            consciousness_id = result.single()["consciousness_id"]
            self.logger.debug(f"Created consciousness state node: {consciousness_id}")
            return consciousness_id
    
    def create_archetype_node(self, archetype_data: Dict[str, Any]) -> str:
        """Create an archetypal pattern node"""
        
        cypher = """
        CREATE (arch:Archetype {
            id: $id,
            pattern_type: $pattern_type,
            resonance_strength: $resonance_strength,
            psychological_criteria: $psychological_criteria,
            chaotic_dynamics: $chaotic_dynamics,
            narrative_elements: $narrative_elements,
            cross_cultural_persistence: $cross_cultural_persistence,
            metadata: $metadata
        })
        RETURN arch.id as archetype_id
        """
        
        with self.driver.session() as session:
            result = session.run(cypher, **archetype_data)
            archetype_id = result.single()["archetype_id"]
            self.logger.debug(f"Created archetype node: {archetype_id}")
            return archetype_id
    
    def create_relationship(self, from_node_id: str, to_node_id: str, 
                          relationship_type: RelationType, 
                          properties: Optional[Dict[str, Any]] = None) -> bool:
        """Create a relationship between two nodes"""
        
        properties = properties or {}
        properties["created_at"] = datetime.utcnow().isoformat()
        
        # Convert properties to Cypher-safe format
        props_str = ", ".join([f"{k}: ${k}" for k in properties.keys()])
        
        cypher = f"""
        MATCH (from_node), (to_node)
        WHERE from_node.id = $from_id AND to_node.id = $to_id
        CREATE (from_node)-[r:{relationship_type.value} {{{props_str}}}]->(to_node)
        RETURN r
        """
        
        params = {
            "from_id": from_node_id,
            "to_id": to_node_id,
            **properties
        }
        
        with self.driver.session() as session:
            result = session.run(cypher, **params)
            success = result.single() is not None
            if success:
                self.logger.debug(f"Created relationship: {from_node_id} -[{relationship_type.value}]-> {to_node_id}")
            return success
    
    def find_similar_architectures(self, embedding: List[float], 
                                 limit: int = 10, 
                                 threshold: float = 0.8) -> List[Dict[str, Any]]:
        """Find architectures similar to given embedding using vector similarity"""
        
        cypher = """
        CALL db.index.vector.queryNodes('architecture_embedding_vector', $limit, $embedding)
        YIELD node, score
        WHERE score >= $threshold
        RETURN node.id as id, node.name as name, node.performance_score as performance, score
        ORDER BY score DESC
        """
        
        with self.driver.session() as session:
            result = session.run(cypher, embedding=embedding, limit=limit, threshold=threshold)
            return [dict(record) for record in result]
    
    def get_architecture_evolution_path(self, architecture_id: str) -> List[Dict[str, Any]]:
        """Get the complete evolution path of an architecture"""
        
        cypher = """
        MATCH path = (start:Architecture {id: $arch_id})-[:EVOLVED_FROM*0..]->(ancestor:Architecture)
        RETURN [node in nodes(path) | {
            id: node.id, 
            name: node.name, 
            performance_score: node.performance_score,
            consciousness_level: node.consciousness_level
        }] as evolution_path
        ORDER BY length(path) DESC
        LIMIT 1
        """
        
        with self.driver.session() as session:
            result = session.run(cypher, arch_id=architecture_id)
            record = result.single()
            return record["evolution_path"] if record else []
    
    def get_episodic_context_for_architecture(self, architecture_id: str) -> List[Dict[str, Any]]:
        """Get all episodic memories related to an architecture"""
        
        cypher = """
        MATCH (a:Architecture {id: $arch_id})-[:OCCURRED_IN|TRIGGERED_BY*1..2]-(e:Episode)
        OPTIONAL MATCH (e)-[:EMBODIES_ARCHETYPE]->(arch:Archetype)
        RETURN DISTINCT e.id as episode_id, e.title as title, 
               e.narrative_summary as summary, e.dominant_archetype as archetype,
               arch.pattern_type as archetype_pattern
        ORDER BY e.timestamp DESC
        """
        
        with self.driver.session() as session:
            result = session.run(cypher, arch_id=architecture_id)
            return [dict(record) for record in result]
    
    def search_by_narrative(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search architectures and episodes by narrative content"""
        
        cypher = """
        CALL db.index.fulltext.queryNodes('architecture_description_fulltext', $query)
        YIELD node as arch_node, score as arch_score
        WITH arch_node, arch_score
        OPTIONAL MATCH (arch_node)-[:EXHIBITS_CONSCIOUSNESS]->(c:ConsciousnessState)
        RETURN 'Architecture' as type, arch_node.id as id, arch_node.name as name,
               arch_node.description as description, arch_score as relevance_score,
               c.level as consciousness_level
        
        UNION
        
        CALL db.index.fulltext.queryNodes('episode_narrative_fulltext', $query)
        YIELD node as ep_node, score as ep_score
        RETURN 'Episode' as type, ep_node.id as id, ep_node.title as name,
               ep_node.narrative_summary as description, ep_score as relevance_score,
               null as consciousness_level
        
        ORDER BY relevance_score DESC
        LIMIT $limit
        """
        
        with self.driver.session() as session:
            result = session.run(cypher, query=query, limit=limit)
            return [dict(record) for record in result]
    
    def get_consciousness_evolution_timeline(self) -> List[Dict[str, Any]]:
        """Get the timeline of consciousness evolution across architectures"""
        
        cypher = """
        MATCH (a:Architecture)-[:EXHIBITS_CONSCIOUSNESS]->(c:ConsciousnessState)
        RETURN a.id as architecture_id, a.name as architecture_name,
               c.level as consciousness_level, c.score as consciousness_score,
               a.created_at as timestamp
        ORDER BY a.created_at ASC
        """
        
        with self.driver.session() as session:
            result = session.run(cypher)
            return [dict(record) for record in result]
    
    def analyze_archetypal_patterns(self) -> Dict[str, Any]:
        """Analyze the distribution and relationships of archetypal patterns"""
        
        cypher = """
        MATCH (arch:Archetype)<-[:EMBODIES_ARCHETYPE]-(e:Episode)-[:OCCURRED_IN]->(a:Architecture)
        WITH arch.pattern_type as pattern, 
             count(DISTINCT e) as episode_count,
             count(DISTINCT a) as architecture_count,
             avg(a.performance_score) as avg_performance,
             avg(arch.resonance_strength) as avg_resonance
        RETURN pattern, episode_count, architecture_count, 
               avg_performance, avg_resonance
        ORDER BY episode_count DESC
        """
        
        with self.driver.session() as session:
            result = session.run(cypher)
            patterns = [dict(record) for record in result]
            
            return {
                "archetypal_distribution": patterns,
                "total_patterns": len(patterns),
                "most_common_pattern": patterns[0]["pattern"] if patterns else None
            }

# =============================================================================
# AutoSchemaKG Integration Layer
# =============================================================================

class AutoSchemaKGIntegration:
    """Integration layer for automatic knowledge graph construction using AutoSchemaKG"""
    
    def __init__(self, neo4j_schema: Neo4jUnifiedSchema):
        self.neo4j_schema = neo4j_schema
        self.logger = logging.getLogger(__name__)
    
    def auto_conceptualize_architecture(self, architecture_data: Dict[str, Any]) -> Dict[str, Any]:
        """Use AutoSchemaKG to automatically identify concepts and relationships in architecture data"""
        
        try:
            from atlas_rag import AutoSchemaKG
            
            # Initialize AutoSchemaKG
            auto_schema = AutoSchemaKG()
            
            # Create text representation of architecture for conceptualization
            architecture_text = self._architecture_to_text(architecture_data)
            
            # Extract concepts and relationships
            concepts = auto_schema.extract_concepts(architecture_text)
            relationships = auto_schema.extract_relationships(architecture_text)
            
            # Map to our unified schema
            mapped_concepts = self._map_concepts_to_schema(concepts)
            mapped_relationships = self._map_relationships_to_schema(relationships)
            
            return {
                "concepts": mapped_concepts,
                "relationships": mapped_relationships,
                "original_concepts": concepts,
                "original_relationships": relationships
            }
            
        except ImportError:
            self.logger.warning("AutoSchemaKG not available, using rule-based conceptualization")
            return self._rule_based_conceptualization(architecture_data)
    
    def _architecture_to_text(self, architecture_data: Dict[str, Any]) -> str:
        """Convert architecture data to text for AutoSchemaKG processing"""
        
        text_parts = []
        
        if "name" in architecture_data:
            text_parts.append(f"Architecture: {architecture_data['name']}")
        
        if "description" in architecture_data:
            text_parts.append(f"Description: {architecture_data['description']}")
        
        if "motivation" in architecture_data:
            text_parts.append(f"Motivation: {architecture_data['motivation']}")
        
        if "program" in architecture_data:
            text_parts.append(f"Implementation: {architecture_data['program']}")
        
        if "result" in architecture_data:
            text_parts.append(f"Results: {architecture_data['result']}")
        
        return " ".join(text_parts)
    
    def _map_concepts_to_schema(self, concepts: List[str]) -> List[Dict[str, Any]]:
        """Map AutoSchemaKG concepts to our unified schema"""
        
        mapped_concepts = []
        
        for concept in concepts:
            concept_lower = concept.lower()
            
            # Map to neural architecture concepts
            if any(term in concept_lower for term in ["layer", "conv", "attention", "transformer"]):
                mapped_concepts.append({
                    "type": NodeType.COMPONENT.value,
                    "concept": concept,
                    "category": "neural_component"
                })
            
            # Map to consciousness concepts
            elif any(term in concept_lower for term in ["consciousness", "awareness", "self", "meta"]):
                mapped_concepts.append({
                    "type": NodeType.CONSCIOUSNESS_STATE.value,
                    "concept": concept,
                    "category": "consciousness"
                })
            
            # Map to episodic concepts
            elif any(term in concept_lower for term in ["episode", "memory", "recall", "experience"]):
                mapped_concepts.append({
                    "type": NodeType.EPISODE.value,
                    "concept": concept,
                    "category": "episodic"
                })
            
            # Map to archetypal concepts
            elif any(term in concept_lower for term in ["hero", "sage", "creator", "pattern", "archetype"]):
                mapped_concepts.append({
                    "type": NodeType.ARCHETYPE.value,
                    "concept": concept,
                    "category": "archetypal"
                })
            
            else:
                # Generic concept
                mapped_concepts.append({
                    "type": "Concept",
                    "concept": concept,
                    "category": "general"
                })
        
        return mapped_concepts
    
    def _map_relationships_to_schema(self, relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Map AutoSchemaKG relationships to our unified schema"""
        
        mapped_relationships = []
        
        for rel in relationships:
            # Determine relationship type based on context
            rel_type = self._infer_relationship_type(rel)
            
            mapped_relationships.append({
                "from": rel.get("subject", ""),
                "to": rel.get("object", ""),
                "type": rel_type.value if rel_type else "RELATED_TO",
                "confidence": rel.get("confidence", 0.5),
                "original": rel
            })
        
        return mapped_relationships
    
    def _infer_relationship_type(self, relationship: Dict[str, Any]) -> Optional[RelationType]:
        """Infer our schema relationship type from AutoSchemaKG relationship"""
        
        predicate = relationship.get("predicate", "").lower()
        
        if "contain" in predicate or "include" in predicate:
            return RelationType.CONTAINS
        elif "evolve" in predicate or "derive" in predicate:
            return RelationType.EVOLVED_FROM
        elif "similar" in predicate or "like" in predicate:
            return RelationType.SIMILAR_TO
        elif "perform" in predicate or "better" in predicate:
            return RelationType.OUTPERFORMS
        elif "flow" in predicate or "stream" in predicate:
            return RelationType.FLOWS_INTO
        elif "resonate" in predicate or "match" in predicate:
            return RelationType.RESONATES_WITH
        elif "trigger" in predicate or "cause" in predicate:
            return RelationType.TRIGGERED_BY
        elif "learn" in predicate or "adapt" in predicate:
            return RelationType.LEARNED_FROM
        
        return None
    
    def _rule_based_conceptualization(self, architecture_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback rule-based conceptualization when AutoSchemaKG is not available"""
        
        concepts = []
        relationships = []
        
        # Extract basic concepts from architecture data
        if "name" in architecture_data:
            concepts.append({
                "type": NodeType.ARCHITECTURE.value,
                "concept": architecture_data["name"],
                "category": "architecture"
            })
        
        if "consciousness_level" in architecture_data:
            concepts.append({
                "type": NodeType.CONSCIOUSNESS_STATE.value,
                "concept": f"Consciousness Level: {architecture_data['consciousness_level']}",
                "category": "consciousness"
            })
        
        return {
            "concepts": concepts,
            "relationships": relationships,
            "method": "rule_based"
        }

# =============================================================================
# Example Usage and Testing
# =============================================================================

def example_usage():
    """Example of how to use the unified Neo4j schema"""
    
    # Initialize schema
    schema = Neo4jUnifiedSchema()
    
    if schema.connect():
        # Create constraints and indexes
        schema.create_constraints_and_indexes()
        
        # Create example architecture
        arch_data = {
            "id": str(uuid.uuid4()),
            "name": "Transformer-GPT-Hero",
            "program": "class TransformerGPT(nn.Module): ...",
            "result": "Achieved 95% accuracy on language modeling",
            "motivation": "Exploring transformer architectures with heroic narrative patterns",
            "performance_score": 0.95,
            "consciousness_level": "EMERGING",
            "consciousness_score": 0.7,
            "embedding": [0.1] * 512,  # Example embedding
            "created_at": datetime.utcnow().isoformat(),
            "metadata": json.dumps({"experiment_id": "exp_001"})
        }
        
        arch_id = schema.create_architecture_node(arch_data)
        
        # Create example episode
        episode_data = {
            "id": str(uuid.uuid4()),
            "title": "The Hero's Journey in Neural Architecture Search",
            "narrative_summary": "An architecture overcomes initial poor performance through iterative refinement",
            "start_evaluation": 1,
            "end_evaluation": 50,
            "episode_duration": 3600.0,
            "exploration_phase": "exploitation",
            "dominant_archetype": "HERO_DRAGON_SLAYER",
            "narrative_coherence_score": 0.85,
            "archetypal_resonance_strength": 0.9,
            "embedding": [0.2] * 512,  # Example embedding
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": json.dumps({"session_id": "session_001"})
        }
        
        episode_id = schema.create_episode_node(episode_data)
        
        # Create relationship
        schema.create_relationship(
            episode_id, arch_id, 
            RelationType.OCCURRED_IN,
            {"context": "architecture_evolution", "strength": 0.8}
        )
        
        # Test AutoSchemaKG integration
        auto_kg = AutoSchemaKGIntegration(schema)
        conceptualization = auto_kg.auto_conceptualize_architecture(arch_data)
        
        print("✅ Example Neo4j unified schema setup complete!")
        print(f"📊 Architecture ID: {arch_id}")
        print(f"📚 Episode ID: {episode_id}")
        print(f"🤖 AutoSchemaKG concepts: {len(conceptualization['concepts'])}")
        
        schema.close()

if __name__ == "__main__":
    example_usage()
