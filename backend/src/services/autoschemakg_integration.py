"""
AutoSchemaKG Integration for Ultra-Granular Document Processing
============================================================

Automatically constructs knowledge graphs from the five-level concept extraction.
Integrates with multi-tier memory system for persistent knowledge storage.
Implements Spec-022 Task 2.4 requirements.

Features:
- Automatic schema generation from extracted concepts
- Relationship inference between atomic/composite concepts
- Context-aware knowledge graph construction
- Integration with warm memory (Neo4j) for persistent storage
- Real-time graph updates and relationship discovery
"""

import asyncio
import logging
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import uuid

# Knowledge graph construction
try:
    from neo4j import AsyncGraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

# Concept extraction integration
try:
    from .five_level_concept_extraction import FiveLevelConceptExtractionService, FiveLevelExtractionResult
    from .multi_tier_memory import MultiTierMemorySystem, MemoryItem, MemoryTier
    SERVICES_AVAILABLE = True
except ImportError:
    # Fallback for testing
    SERVICES_AVAILABLE = False
    # Mock classes for testing
    class FiveLevelConceptExtractionService:
        pass
    class FiveLevelExtractionResult:
        pass
    class MultiTierMemorySystem:
        pass

import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

class NodeType(Enum):
    """Types of nodes in the knowledge graph"""
    ATOMIC_CONCEPT = "atomic_concept"
    RELATIONSHIP = "relationship"
    COMPOSITE_CONCEPT = "composite_concept"
    CONTEXT = "context"
    NARRATIVE = "narrative"
    DOCUMENT = "document"
    DOMAIN = "domain"

class RelationType(Enum):
    """Types of relationships in the knowledge graph"""
    CONTAINS = "CONTAINS"
    RELATES_TO = "RELATES_TO"
    PART_OF = "PART_OF"
    INFLUENCES = "INFLUENCES"
    OCCURS_IN = "OCCURS_IN"
    DERIVED_FROM = "DERIVED_FROM"
    CONTRADICTS = "CONTRADICTS"
    SUPPORTS = "SUPPORTS"
    TEMPORALLY_FOLLOWS = "TEMPORALLY_FOLLOWS"
    CAUSALLY_RELATED = "CAUSALLY_RELATED"

@dataclass
class KnowledgeGraphNode:
    """Represents a node in the knowledge graph"""
    id: str
    type: NodeType
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)
    embeddings: Optional[np.ndarray] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    confidence: float = 1.0
    source_document: Optional[str] = None

@dataclass
class KnowledgeGraphRelation:
    """Represents a relationship in the knowledge graph"""
    id: str
    source_id: str
    target_id: str
    type: RelationType
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)
    source_document: Optional[str] = None

@dataclass
class AutoSchemaResult:
    """Result of automatic schema generation"""
    schema_id: str
    nodes: List[KnowledgeGraphNode]
    relations: List[KnowledgeGraphRelation]
    statistics: Dict[str, Any]
    processing_time: float
    confidence_score: float

class AutoSchemaKGService:
    """AutoSchemaKG integration service for automatic knowledge graph construction"""
    
    def __init__(self, 
                 neo4j_uri: str = "bolt://localhost:7687",
                 neo4j_username: str = "neo4j", 
                 neo4j_password: str = "thoughtseed",
                 memory_system: Optional[MultiTierMemorySystem] = None):
        self.neo4j_uri = neo4j_uri
        self.neo4j_username = neo4j_username
        self.neo4j_password = neo4j_password
        self.driver = None
        self.memory_system = memory_system
        
        # Schema generation parameters
        self.confidence_threshold = 0.7
        self.max_relations_per_node = 50
        self.semantic_similarity_threshold = 0.8
        
        # Concept extraction service
        self.concept_extractor = None
        if SERVICES_AVAILABLE:
            self.concept_extractor = FiveLevelConceptExtractionService()
    
    async def initialize(self):
        """Initialize the AutoSchemaKG service"""
        if NEO4J_AVAILABLE:
            try:
                self.driver = AsyncGraphDatabase.driver(
                    self.neo4j_uri,
                    auth=(self.neo4j_username, self.neo4j_password)
                )
                
                # Test connection and create constraints
                async with self.driver.session() as session:
                    await session.run("RETURN 1")
                    await self._create_graph_constraints(session)
                
                logger.info("✅ AutoSchemaKG Neo4j connection established")
                
            except Exception as e:
                logger.error(f"❌ Failed to connect to Neo4j: {e}")
                self.driver = None
        
        # Initialize concept extractor
        if self.concept_extractor and hasattr(self.concept_extractor, 'initialize'):
            await self.concept_extractor.initialize()
        
        logger.info("🧠 AutoSchemaKG service initialized")
    
    async def _create_graph_constraints(self, session):
        """Create Neo4j constraints for knowledge graph"""
        constraints = [
            "CREATE CONSTRAINT kg_node_id IF NOT EXISTS FOR (n:KGNode) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT kg_relation_id IF NOT EXISTS FOR (r:KGRelation) REQUIRE r.id IS UNIQUE",
            "CREATE INDEX kg_node_type IF NOT EXISTS FOR (n:KGNode) ON (n.type)",
            "CREATE INDEX kg_node_name IF NOT EXISTS FOR (n:KGNode) ON (n.name)",
            "CREATE INDEX kg_relation_type IF NOT EXISTS FOR ()-[r:KGRelation]-() ON (r.type)"
        ]
        
        for constraint in constraints:
            try:
                await session.run(constraint)
            except Exception as e:
                # Constraint might already exist
                logger.debug(f"Constraint creation: {e}")
    
    async def construct_knowledge_graph(self, 
                                      document_source: str,
                                      domain_focus: List[str] = None) -> AutoSchemaResult:
        """Construct knowledge graph from document using five-level extraction"""
        start_time = datetime.now()
        
        # Step 1: Extract concepts using five-level extraction
        if not self.concept_extractor:
            raise RuntimeError("Concept extractor not available")
        
        extraction_result = await self.concept_extractor.extract_concepts_from_document(
            document_source, source_type="auto"
        )
        
        # Step 2: Build knowledge graph nodes
        nodes = []
        relations = []
        
        # Create document node
        doc_id = f"doc_{hashlib.md5(document_source.encode()).hexdigest()[:8]}"
        document_node = KnowledgeGraphNode(
            id=doc_id,
            type=NodeType.DOCUMENT,
            name=f"Document_{doc_id}",
            properties={
                "source": document_source,
                "processing_date": datetime.now().isoformat(),
                "extraction_confidence": extraction_result.confidence_score
            }
        )
        nodes.append(document_node)
        
        # Create nodes from extracted concepts
        await self._create_nodes_from_extraction(extraction_result, nodes, doc_id)
        
        # Step 3: Infer relationships between concepts
        await self._infer_concept_relationships(extraction_result, nodes, relations, doc_id)
        
        # Step 4: Store in Neo4j knowledge graph
        if self.driver:
            await self._store_knowledge_graph(nodes, relations)
        
        # Step 5: Store in multi-tier memory system
        if self.memory_system:
            await self._store_in_memory_system(nodes, relations, extraction_result)
        
        # Step 6: Generate schema statistics
        processing_time = (datetime.now() - start_time).total_seconds()
        statistics = await self._generate_schema_statistics(nodes, relations, extraction_result)
        
        return AutoSchemaResult(
            schema_id=f"schema_{uuid.uuid4().hex[:8]}",
            nodes=nodes,
            relations=relations,
            statistics=statistics,
            processing_time=processing_time,
            confidence_score=self._calculate_overall_confidence(nodes, relations)
        )
    
    async def _create_nodes_from_extraction(self, 
                                          extraction_result: FiveLevelExtractionResult,
                                          nodes: List[KnowledgeGraphNode],
                                          doc_id: str):
        """Create knowledge graph nodes from concept extraction"""
        
        # Level 1: Atomic concepts
        for atomic in extraction_result.atomic_concepts:
            node = KnowledgeGraphNode(
                id=f"atomic_{hashlib.md5(atomic['concept'].encode()).hexdigest()[:8]}",
                type=NodeType.ATOMIC_CONCEPT,
                name=atomic['concept'],
                properties={
                    "importance": atomic.get('importance', 0.5),
                    "category": atomic.get('category', 'unknown'),
                    "level": 1,
                    "definition": atomic.get('definition', ''),
                    "domain": atomic.get('domain', 'general')
                },
                confidence=atomic.get('confidence', 0.8),
                source_document=doc_id
            )
            nodes.append(node)
        
        # Level 2: Relationships
        for relation in extraction_result.relationships:
            node = KnowledgeGraphNode(
                id=f"rel_{hashlib.md5(relation['relationship'].encode()).hexdigest()[:8]}",
                type=NodeType.RELATIONSHIP,
                name=relation['relationship'],
                properties={
                    "type": relation.get('type', 'general'),
                    "strength": relation.get('strength', 0.5),
                    "level": 2,
                    "entities": relation.get('entities', []),
                    "description": relation.get('description', '')
                },
                confidence=relation.get('confidence', 0.7),
                source_document=doc_id
            )
            nodes.append(node)
        
        # Level 3: Composite concepts
        for composite in extraction_result.composite_concepts:
            node = KnowledgeGraphNode(
                id=f"comp_{hashlib.md5(composite['concept'].encode()).hexdigest()[:8]}",
                type=NodeType.COMPOSITE_CONCEPT,
                name=composite['concept'],
                properties={
                    "components": composite.get('components', []),
                    "complexity": composite.get('complexity', 0.5),
                    "level": 3,
                    "definition": composite.get('definition', ''),
                    "domain": composite.get('domain', 'general')
                },
                confidence=composite.get('confidence', 0.75),
                source_document=doc_id
            )
            nodes.append(node)
        
        # Level 4: Contexts
        for context in extraction_result.contexts:
            node = KnowledgeGraphNode(
                id=f"ctx_{hashlib.md5(context['context'].encode()).hexdigest()[:8]}",
                type=NodeType.CONTEXT,
                name=context['context'],
                properties={
                    "scope": context.get('scope', 'local'),
                    "influence": context.get('influence', 0.5),
                    "level": 4,
                    "description": context.get('description', ''),
                    "temporal_aspect": context.get('temporal_aspect', 'static')
                },
                confidence=context.get('confidence', 0.7),
                source_document=doc_id
            )
            nodes.append(node)
        
        # Level 5: Narratives
        for narrative in extraction_result.narratives:
            node = KnowledgeGraphNode(
                id=f"narr_{hashlib.md5(narrative['narrative'].encode()).hexdigest()[:8]}",
                type=NodeType.NARRATIVE,
                name=narrative['narrative'],
                properties={
                    "theme": narrative.get('theme', 'general'),
                    "coherence": narrative.get('coherence', 0.5),
                    "level": 5,
                    "summary": narrative.get('summary', ''),
                    "key_elements": narrative.get('key_elements', [])
                },
                confidence=narrative.get('confidence', 0.6),
                source_document=doc_id
            )
            nodes.append(node)
    
    async def _infer_concept_relationships(self, 
                                         extraction_result: FiveLevelExtractionResult,
                                         nodes: List[KnowledgeGraphNode],
                                         relations: List[KnowledgeGraphRelation],
                                         doc_id: str):
        """Infer relationships between concepts using semantic analysis"""
        
        # Create relation mapping for quick lookup
        node_by_name = {node.name.lower(): node for node in nodes}
        
        # Infer hierarchical relationships (level-based)
        await self._create_hierarchical_relations(nodes, relations, doc_id)
        
        # Infer semantic relationships (content-based)
        await self._create_semantic_relations(extraction_result, nodes, relations, doc_id)
        
        # Infer document containment relationships
        await self._create_containment_relations(nodes, relations, doc_id)
    
    async def _create_hierarchical_relations(self, 
                                           nodes: List[KnowledgeGraphNode],
                                           relations: List[KnowledgeGraphRelation],
                                           doc_id: str):
        """Create hierarchical relationships between concept levels"""
        
        # Group nodes by type
        nodes_by_type = {}
        for node in nodes:
            if node.type not in nodes_by_type:
                nodes_by_type[node.type] = []
            nodes_by_type[node.type].append(node)
        
        # Create PART_OF relationships: atomic -> composite
        if NodeType.ATOMIC_CONCEPT in nodes_by_type and NodeType.COMPOSITE_CONCEPT in nodes_by_type:
            for composite in nodes_by_type[NodeType.COMPOSITE_CONCEPT]:
                components = composite.properties.get('components', [])
                for atomic in nodes_by_type[NodeType.ATOMIC_CONCEPT]:
                    if any(comp.lower() in atomic.name.lower() for comp in components):
                        relation = KnowledgeGraphRelation(
                            id=f"rel_{uuid.uuid4().hex[:8]}",
                            source_id=atomic.id,
                            target_id=composite.id,
                            type=RelationType.PART_OF,
                            confidence=0.8,
                            source_document=doc_id
                        )
                        relations.append(relation)
        
        # Create OCCURS_IN relationships: concepts -> contexts
        if NodeType.CONTEXT in nodes_by_type:
            for context in nodes_by_type[NodeType.CONTEXT]:
                for node_type in [NodeType.ATOMIC_CONCEPT, NodeType.COMPOSITE_CONCEPT, NodeType.RELATIONSHIP]:
                    if node_type in nodes_by_type:
                        for concept in nodes_by_type[node_type]:
                            # Simple semantic matching
                            if self._semantic_match(concept.name, context.name):
                                relation = KnowledgeGraphRelation(
                                    id=f"rel_{uuid.uuid4().hex[:8]}",
                                    source_id=concept.id,
                                    target_id=context.id,
                                    type=RelationType.OCCURS_IN,
                                    confidence=0.7,
                                    source_document=doc_id
                                )
                                relations.append(relation)
    
    async def _create_semantic_relations(self, 
                                       extraction_result: FiveLevelExtractionResult,
                                       nodes: List[KnowledgeGraphNode],
                                       relations: List[KnowledgeGraphRelation],
                                       doc_id: str):
        """Create semantic relationships based on content analysis"""
        
        # Use extracted relationships to create graph relations
        node_by_name = {node.name.lower(): node for node in nodes}
        
        for rel_data in extraction_result.relationships:
            entities = rel_data.get('entities', [])
            rel_type = rel_data.get('type', 'general')
            
            # Map relationship types to graph relation types
            graph_rel_type = self._map_relationship_type(rel_type)
            
            # Create relations between entities
            for i, entity1 in enumerate(entities):
                for entity2 in entities[i+1:]:
                    node1 = node_by_name.get(entity1.lower())
                    node2 = node_by_name.get(entity2.lower())
                    
                    if node1 and node2:
                        relation = KnowledgeGraphRelation(
                            id=f"rel_{uuid.uuid4().hex[:8]}",
                            source_id=node1.id,
                            target_id=node2.id,
                            type=graph_rel_type,
                            properties={
                                "relationship_description": rel_data['relationship'],
                                "original_type": rel_type
                            },
                            confidence=rel_data.get('confidence', 0.7),
                            source_document=doc_id
                        )
                        relations.append(relation)
    
    async def _create_containment_relations(self, 
                                          nodes: List[KnowledgeGraphNode],
                                          relations: List[KnowledgeGraphRelation],
                                          doc_id: str):
        """Create containment relationships to document"""
        
        document_node = next((n for n in nodes if n.type == NodeType.DOCUMENT), None)
        if not document_node:
            return
        
        for node in nodes:
            if node.type != NodeType.DOCUMENT:
                relation = KnowledgeGraphRelation(
                    id=f"rel_{uuid.uuid4().hex[:8]}",
                    source_id=node.id,
                    target_id=document_node.id,
                    type=RelationType.DERIVED_FROM,
                    confidence=1.0,
                    source_document=doc_id
                )
                relations.append(relation)
    
    async def _store_knowledge_graph(self, 
                                   nodes: List[KnowledgeGraphNode],
                                   relations: List[KnowledgeGraphRelation]):
        """Store knowledge graph in Neo4j"""
        if not self.driver:
            logger.warning("Neo4j not available, skipping graph storage")
            return
        
        async with self.driver.session() as session:
            # Store nodes
            for node in nodes:
                query = """
                MERGE (n:KGNode {id: $id})
                SET n.type = $type,
                    n.name = $name,
                    n.properties = $properties,
                    n.confidence = $confidence,
                    n.created_at = $created_at,
                    n.last_updated = $last_updated,
                    n.source_document = $source_document
                """
                await session.run(query, {
                    "id": node.id,
                    "type": node.type.value,
                    "name": node.name,
                    "properties": json.dumps(node.properties),
                    "confidence": node.confidence,
                    "created_at": node.created_at.isoformat(),
                    "last_updated": node.last_updated.isoformat(),
                    "source_document": node.source_document
                })
            
            # Store relationships
            for relation in relations:
                query = """
                MATCH (a:KGNode {id: $source_id}), (b:KGNode {id: $target_id})
                MERGE (a)-[r:KGRelation {id: $id}]->(b)
                SET r.type = $type,
                    r.properties = $properties,
                    r.confidence = $confidence,
                    r.created_at = $created_at,
                    r.source_document = $source_document
                """
                await session.run(query, {
                    "id": relation.id,
                    "source_id": relation.source_id,
                    "target_id": relation.target_id,
                    "type": relation.type.value,
                    "properties": json.dumps(relation.properties),
                    "confidence": relation.confidence,
                    "created_at": relation.created_at.isoformat(),
                    "source_document": relation.source_document
                })
        
        logger.info(f"✅ Stored {len(nodes)} nodes and {len(relations)} relations in Neo4j")
    
    async def _store_in_memory_system(self, 
                                    nodes: List[KnowledgeGraphNode],
                                    relations: List[KnowledgeGraphRelation],
                                    extraction_result: FiveLevelExtractionResult):
        """Store knowledge graph in multi-tier memory system"""
        if not self.memory_system:
            return
        
        # Store as high-importance knowledge graph
        kg_data = {
            "type": "knowledge_graph",
            "nodes": [self._serialize_node(node) for node in nodes],
            "relations": [self._serialize_relation(relation) for relation in relations],
            "metadata": {
                "extraction_confidence": extraction_result.confidence_score,
                "processing_time": extraction_result.processing_time,
                "created_at": datetime.now().isoformat()
            }
        }
        
        kg_id = await self.memory_system.store_concept(kg_data, importance=0.9)
        logger.info(f"✅ Stored knowledge graph in memory system: {kg_id}")
    
    def _serialize_node(self, node: KnowledgeGraphNode) -> Dict[str, Any]:
        """Serialize node for storage"""
        return {
            "id": node.id,
            "type": node.type.value,
            "name": node.name,
            "properties": node.properties,
            "confidence": node.confidence,
            "created_at": node.created_at.isoformat(),
            "source_document": node.source_document
        }
    
    def _serialize_relation(self, relation: KnowledgeGraphRelation) -> Dict[str, Any]:
        """Serialize relation for storage"""
        return {
            "id": relation.id,
            "source_id": relation.source_id,
            "target_id": relation.target_id,
            "type": relation.type.value,
            "properties": relation.properties,
            "confidence": relation.confidence,
            "created_at": relation.created_at.isoformat(),
            "source_document": relation.source_document
        }
    
    async def _generate_schema_statistics(self, 
                                        nodes: List[KnowledgeGraphNode],
                                        relations: List[KnowledgeGraphRelation],
                                        extraction_result: FiveLevelExtractionResult) -> Dict[str, Any]:
        """Generate comprehensive schema statistics"""
        
        node_types = {}
        relation_types = {}
        confidence_scores = []
        
        for node in nodes:
            node_type = node.type.value
            node_types[node_type] = node_types.get(node_type, 0) + 1
            confidence_scores.append(node.confidence)
        
        for relation in relations:
            rel_type = relation.type.value
            relation_types[rel_type] = relation_types.get(rel_type, 0) + 1
            confidence_scores.append(relation.confidence)
        
        return {
            "total_nodes": len(nodes),
            "total_relations": len(relations),
            "node_types": node_types,
            "relation_types": relation_types,
            "average_confidence": np.mean(confidence_scores) if confidence_scores else 0,
            "concept_levels": {
                "atomic": len([n for n in nodes if n.type == NodeType.ATOMIC_CONCEPT]),
                "relationships": len([n for n in nodes if n.type == NodeType.RELATIONSHIP]),
                "composite": len([n for n in nodes if n.type == NodeType.COMPOSITE_CONCEPT]),
                "contexts": len([n for n in nodes if n.type == NodeType.CONTEXT]),
                "narratives": len([n for n in nodes if n.type == NodeType.NARRATIVE])
            },
            "extraction_metrics": {
                "original_confidence": extraction_result.confidence_score,
                "processing_time": extraction_result.processing_time,
                "consciousness_level": extraction_result.consciousness_level
            }
        }
    
    def _calculate_overall_confidence(self, 
                                    nodes: List[KnowledgeGraphNode],
                                    relations: List[KnowledgeGraphRelation]) -> float:
        """Calculate overall confidence score for the knowledge graph"""
        all_confidences = []
        
        for node in nodes:
            all_confidences.append(node.confidence)
        
        for relation in relations:
            all_confidences.append(relation.confidence)
        
        return np.mean(all_confidences) if all_confidences else 0.0
    
    def _semantic_match(self, text1: str, text2: str, threshold: float = 0.3) -> bool:
        """Simple semantic matching based on word overlap"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return False
        
        overlap = len(words1.intersection(words2))
        total = len(words1.union(words2))
        
        return (overlap / total) > threshold
    
    def _map_relationship_type(self, rel_type: str) -> RelationType:
        """Map extracted relationship type to graph relation type"""
        mapping = {
            "causal": RelationType.CAUSALLY_RELATED,
            "temporal": RelationType.TEMPORALLY_FOLLOWS,
            "hierarchical": RelationType.PART_OF,
            "similarity": RelationType.RELATES_TO,
            "contrast": RelationType.CONTRADICTS,
            "support": RelationType.SUPPORTS,
            "influence": RelationType.INFLUENCES,
            "general": RelationType.RELATES_TO
        }
        
        return mapping.get(rel_type.lower(), RelationType.RELATES_TO)
    
    async def query_knowledge_graph(self, 
                                  query_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query the knowledge graph with semantic search"""
        if not self.driver:
            logger.warning("Neo4j not available for querying")
            return []
        
        results = []
        
        async with self.driver.session() as session:
            # Basic node query
            if "node_type" in query_params:
                query = "MATCH (n:KGNode {type: $type}) RETURN n LIMIT 100"
                result = await session.run(query, {"type": query_params["node_type"]})
                
                async for record in result:
                    node = record["n"]
                    results.append({
                        "id": node["id"],
                        "type": node["type"],
                        "name": node["name"],
                        "properties": json.loads(node.get("properties", "{}")),
                        "confidence": node.get("confidence", 0)
                    })
            
            # Semantic search by name
            if "search_term" in query_params:
                query = """
                MATCH (n:KGNode) 
                WHERE n.name CONTAINS $term 
                RETURN n, 
                       [(n)-[r:KGRelation]-(connected) | {relation: r, node: connected}] as connections
                LIMIT 50
                """
                result = await session.run(query, {"term": query_params["search_term"]})
                
                async for record in result:
                    node = record["n"]
                    connections = record["connections"]
                    
                    results.append({
                        "id": node["id"],
                        "type": node["type"],
                        "name": node["name"],
                        "properties": json.loads(node.get("properties", "{}")),
                        "confidence": node.get("confidence", 0),
                        "connections": len(connections)
                    })
        
        return results
    
    async def close(self):
        """Close the AutoSchemaKG service"""
        if self.driver:
            await self.driver.close()
        
        logger.info("🔄 AutoSchemaKG service closed")

# Testing and demonstration
async def test_autoschemakg_integration():
    """Test AutoSchemaKG integration with sample document"""
    print("🧪 Testing AutoSchemaKG Integration")
    print("=" * 50)
    
    # Initialize service
    service = AutoSchemaKGService()
    await service.initialize()
    
    # Test document - neuroscience concepts
    test_document = """
    Synaptic plasticity is the ability of synapses to strengthen or weaken over time.
    This process involves long-term potentiation (LTP) and long-term depression (LTD).
    Hebbian learning states that "cells that fire together wire together".
    The hippocampus plays a crucial role in memory formation and synaptic plasticity.
    Neurotransmitters like glutamate and GABA regulate synaptic transmission.
    """
    
    print("🔄 Constructing knowledge graph...")
    result = await service.construct_knowledge_graph(
        test_document,
        domain_focus=["neuroscience", "cognition"]
    )
    
    print(f"✅ Schema generated: {result.schema_id}")
    print(f"📊 Processing time: {result.processing_time:.2f}s")
    print(f"🎯 Confidence score: {result.confidence_score:.3f}")
    print()
    
    print("📦 Knowledge Graph Statistics:")
    for key, value in result.statistics.items():
        print(f"  {key}: {value}")
    print()
    
    print(f"🔗 Created {len(result.nodes)} nodes and {len(result.relations)} relations")
    
    # Test querying
    print("🔍 Testing knowledge graph queries...")
    
    # Query by node type
    atomic_concepts = await service.query_knowledge_graph({"node_type": "atomic_concept"})
    print(f"  Found {len(atomic_concepts)} atomic concepts")
    
    # Semantic search
    plasticity_results = await service.query_knowledge_graph({"search_term": "plasticity"})
    print(f"  Found {len(plasticity_results)} nodes related to 'plasticity'")
    
    await service.close()
    print("🎉 AutoSchemaKG integration test completed!")

if __name__ == "__main__":
    asyncio.run(test_autoschemakg_integration())