"""
AutoSchemaKG Integration Demo
Test the knowledge graph construction with mock data
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass

# Import the service
from src.services.autoschemakg_integration import (
    AutoSchemaKGService, 
    KnowledgeGraphNode, 
    KnowledgeGraphRelation,
    NodeType, 
    RelationType,
    AutoSchemaResult
)

@dataclass
class MockExtractionResult:
    """Mock extraction result for testing"""
    atomic_concepts: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    composite_concepts: List[Dict[str, Any]]
    contexts: List[Dict[str, Any]]
    narratives: List[Dict[str, Any]]
    confidence_score: float
    processing_time: float
    consciousness_level: float

async def test_autoschemakg_demo():
    """Demo AutoSchemaKG with mock neuroscience data"""
    print("🧪 AutoSchemaKG Integration Demo")
    print("=" * 50)
    
    # Initialize service
    service = AutoSchemaKGService()
    await service.initialize()
    
    # Create mock extraction result
    mock_result = MockExtractionResult(
        atomic_concepts=[
            {
                "concept": "synaptic_plasticity",
                "importance": 0.9,
                "category": "mechanism",
                "confidence": 0.85,
                "domain": "neuroscience",
                "definition": "Ability of synapses to strengthen or weaken over time"
            },
            {
                "concept": "long_term_potentiation",
                "importance": 0.8,
                "category": "process",
                "confidence": 0.82,
                "domain": "neuroscience",
                "definition": "Persistent strengthening of synapses"
            },
            {
                "concept": "hippocampus",
                "importance": 0.85,
                "category": "structure",
                "confidence": 0.9,
                "domain": "neuroscience",
                "definition": "Brain region crucial for memory formation"
            },
            {
                "concept": "glutamate",
                "importance": 0.7,
                "category": "neurotransmitter",
                "confidence": 0.88,
                "domain": "neuroscience",
                "definition": "Primary excitatory neurotransmitter"
            }
        ],
        relationships=[
            {
                "relationship": "synaptic_plasticity involves long_term_potentiation",
                "type": "causal",
                "strength": 0.8,
                "confidence": 0.85,
                "entities": ["synaptic_plasticity", "long_term_potentiation"]
            },
            {
                "relationship": "hippocampus exhibits synaptic_plasticity",
                "type": "general",
                "strength": 0.9,
                "confidence": 0.87,
                "entities": ["hippocampus", "synaptic_plasticity"]
            },
            {
                "relationship": "glutamate enables long_term_potentiation",
                "type": "causal",
                "strength": 0.75,
                "confidence": 0.8,
                "entities": ["glutamate", "long_term_potentiation"]
            }
        ],
        composite_concepts=[
            {
                "concept": "memory_formation_system",
                "components": ["hippocampus", "synaptic_plasticity", "long_term_potentiation"],
                "complexity": 0.85,
                "confidence": 0.8,
                "domain": "neuroscience",
                "definition": "Integrated system for forming and storing memories"
            }
        ],
        contexts=[
            {
                "context": "learning_and_memory",
                "scope": "cognitive",
                "influence": 0.9,
                "confidence": 0.85,
                "description": "Context of learning and memory processes"
            }
        ],
        narratives=[
            {
                "narrative": "synaptic_plasticity_story",
                "theme": "adaptation",
                "coherence": 0.8,
                "confidence": 0.75,
                "summary": "How synapses adapt through experience to enable learning",
                "key_elements": ["experience", "adaptation", "memory"]
            }
        ],
        confidence_score=0.83,
        processing_time=2.5,
        consciousness_level=0.92
    )
    
    print("🔄 Creating knowledge graph from mock data...")
    
    # Create nodes and relations manually
    nodes = []
    relations = []
    doc_id = "demo_neuroscience_doc"
    
    # Document node
    document_node = KnowledgeGraphNode(
        id=doc_id,
        type=NodeType.DOCUMENT,
        name="Neuroscience Demo Document",
        properties={
            "source": "demo_data",
            "processing_date": datetime.now().isoformat(),
            "extraction_confidence": mock_result.confidence_score
        }
    )
    nodes.append(document_node)
    
    # Create nodes from mock data
    await service._create_nodes_from_extraction(mock_result, nodes, doc_id)
    
    # Create relationships
    await service._infer_concept_relationships(mock_result, nodes, relations, doc_id)
    
    print(f"✅ Created {len(nodes)} nodes and {len(relations)} relations")
    
    # Store in Neo4j if available
    if service.driver:
        await service._store_knowledge_graph(nodes, relations)
        print("📊 Stored in Neo4j knowledge graph")
    
    # Generate statistics
    statistics = await service._generate_schema_statistics(nodes, relations, mock_result)
    
    print("\n📊 Knowledge Graph Statistics:")
    print(f"  📦 Total nodes: {statistics['total_nodes']}")
    print(f"  🔗 Total relations: {statistics['total_relations']}")
    print(f"  🎯 Average confidence: {statistics['average_confidence']:.3f}")
    
    print("\n🧠 Concept Levels:")
    for level, count in statistics['concept_levels'].items():
        print(f"  {level}: {count}")
    
    print("\n🔗 Relation Types:")
    for rel_type, count in statistics['relation_types'].items():
        print(f"  {rel_type}: {count}")
    
    # Test querying
    if service.driver:
        print("\n🔍 Testing knowledge graph queries...")
        
        # Query atomic concepts
        atomic_results = await service.query_knowledge_graph({"node_type": "atomic_concept"})
        print(f"  Found {len(atomic_results)} atomic concepts")
        
        if atomic_results:
            print("  Sample atomic concepts:")
            for concept in atomic_results[:3]:
                print(f"    - {concept['name']} (confidence: {concept['confidence']:.2f})")
        
        # Search for plasticity
        plasticity_results = await service.query_knowledge_graph({"search_term": "plasticity"})
        print(f"  Found {len(plasticity_results)} nodes related to 'plasticity'")
        
        if plasticity_results:
            print("  Plasticity-related nodes:")
            for node in plasticity_results:
                print(f"    - {node['name']} ({node['type']}) - {node.get('connections', 0)} connections")
    
    # Create final result
    result = AutoSchemaResult(
        schema_id="demo_schema_001",
        nodes=nodes,
        relations=relations,
        statistics=statistics,
        processing_time=1.5,
        confidence_score=service._calculate_overall_confidence(nodes, relations)
    )
    
    print(f"\n🎉 AutoSchemaKG Demo Completed!")
    print(f"Schema ID: {result.schema_id}")
    print(f"Overall confidence: {result.confidence_score:.3f}")
    print(f"Processing time: {result.processing_time:.2f}s")
    
    await service.close()
    return result

if __name__ == "__main__":
    asyncio.run(test_autoschemakg_demo())