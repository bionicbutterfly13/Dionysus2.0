"""
Complete AutoSchemaKG Integration Test
Tests the full pipeline: Document -> Five-Level Extraction -> Knowledge Graph -> Multi-Tier Memory
"""

import asyncio
import sys
import os

# Add the src directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.services.five_level_concept_extraction import FiveLevelConceptExtractionService
from src.services.autoschemakg_integration import AutoSchemaKGService
from src.services.multi_tier_memory import MultiTierMemorySystem

async def test_complete_integration():
    """Test the complete pipeline integration"""
    print("🧪 Complete AutoSchemaKG Integration Test")
    print("=" * 60)
    
    # Initialize all services
    print("🔄 Initializing services...")
    
    concept_extractor = FiveLevelConceptExtractionService()
    memory_system = MultiTierMemorySystem()
    autoschema_service = AutoSchemaKGService(memory_system=memory_system)
    
    await concept_extractor.initialize()
    await memory_system.initialize()
    await autoschema_service.initialize()
    
    print("✅ All services initialized")
    
    # Test document - advanced neuroscience concepts
    test_document = """
    Synaptic plasticity is the biological basis of learning and memory formation in the brain.
    This fundamental mechanism involves both long-term potentiation (LTP) and long-term depression (LTD).
    
    The hippocampus, a critical brain structure for episodic memory, exhibits remarkable synaptic plasticity.
    CA1 and CA3 regions of the hippocampus show different patterns of plasticity induction.
    
    Neurotransmitters play essential roles in synaptic transmission and plasticity.
    Glutamate, the brain's primary excitatory neurotransmitter, activates NMDA and AMPA receptors.
    GABA, the main inhibitory neurotransmitter, modulates neural excitability.
    
    Hebbian learning theory states that "neurons that fire together, wire together."
    This principle underlies associative learning and memory consolidation processes.
    
    Spike-timing dependent plasticity (STDP) represents a refinement of Hebbian theory,
    where the precise timing of pre- and post-synaptic action potentials determines
    the direction and magnitude of synaptic strength changes.
    
    Memory consolidation involves the transfer of information from short-term to long-term storage,
    requiring protein synthesis and gene expression changes in neurons.
    """
    
    print("\n📊 Phase 1: Five-Level Concept Extraction")
    print("-" * 40)
    
    # Extract concepts using five-level extraction
    extraction_result = await concept_extractor.extract_concepts_from_document(test_document)
    
    print(f"✅ Extraction completed:")
    print(f"  📝 Atomic concepts: {len(extraction_result.atomic_concepts)}")
    print(f"  🔗 Relationships: {len(extraction_result.relationships)}")
    print(f"  🧩 Composite concepts: {len(extraction_result.composite_concepts)}")
    print(f"  📍 Contexts: {len(extraction_result.contexts)}")
    print(f"  📖 Narratives: {len(extraction_result.narratives)}")
    print(f"  🎯 Confidence: {extraction_result.confidence_score:.3f}")
    print(f"  🧠 Consciousness level: {extraction_result.consciousness_level:.3f}")
    
    print("\n🕸️  Phase 2: Knowledge Graph Construction")
    print("-" * 40)
    
    # Mock the construct_knowledge_graph method to use our extraction result
    start_time = asyncio.get_event_loop().time()
    
    # Create nodes and relations from extraction
    nodes = []
    relations = []
    doc_id = f"test_doc_{int(start_time)}"
    
    # Document node
    from src.services.autoschemakg_integration import KnowledgeGraphNode, NodeType
    document_node = KnowledgeGraphNode(
        id=doc_id,
        type=NodeType.DOCUMENT,
        name="Complete Integration Test Document",
        properties={
            "source": "test_document",
            "processing_date": asyncio.get_event_loop().time(),
            "extraction_confidence": extraction_result.confidence_score
        }
    )
    nodes.append(document_node)
    
    # Create nodes from extraction
    await autoschema_service._create_nodes_from_extraction(extraction_result, nodes, doc_id)
    
    # Create relationships
    await autoschema_service._infer_concept_relationships(extraction_result, nodes, relations, doc_id)
    
    # Store in Neo4j
    if autoschema_service.driver:
        await autoschema_service._store_knowledge_graph(nodes, relations)
    
    # Store in multi-tier memory
    await autoschema_service._store_in_memory_system(nodes, relations, extraction_result)
    
    processing_time = asyncio.get_event_loop().time() - start_time
    
    print(f"✅ Knowledge graph created:")
    print(f"  🔗 Nodes: {len(nodes)}")
    print(f"  ↔️  Relations: {len(relations)}")
    print(f"  ⏱️  Processing time: {processing_time:.2f}s")
    
    # Generate and display statistics
    statistics = await autoschema_service._generate_schema_statistics(nodes, relations, extraction_result)
    
    print("\n📊 Phase 3: Knowledge Graph Analysis")
    print("-" * 40)
    
    print("🧠 Concept Distribution:")
    for level, count in statistics['concept_levels'].items():
        if count > 0:
            print(f"  {level.replace('_', ' ').title()}: {count}")
    
    print("\n🔗 Relationship Types:")
    for rel_type, count in statistics['relation_types'].items():
        if count > 0:
            print(f"  {rel_type.replace('_', ' ').title()}: {count}")
    
    # Test knowledge graph queries
    if autoschema_service.driver:
        print("\n🔍 Phase 4: Knowledge Graph Queries")
        print("-" * 40)
        
        # Query atomic concepts
        atomic_results = await autoschema_service.query_knowledge_graph({"node_type": "atomic_concept"})
        print(f"📝 Atomic concepts found: {len(atomic_results)}")
        
        if atomic_results:
            print("  Top atomic concepts:")
            for concept in sorted(atomic_results, key=lambda x: x['confidence'], reverse=True)[:5]:
                props = concept.get('properties', {})
                domain = props.get('domain', 'general')
                print(f"    • {concept['name']} ({domain}) - confidence: {concept['confidence']:.2f}")
        
        # Search for specific terms
        plasticity_results = await autoschema_service.query_knowledge_graph({"search_term": "plasticity"})
        memory_results = await autoschema_service.query_knowledge_graph({"search_term": "memory"})
        neuron_results = await autoschema_service.query_knowledge_graph({"search_term": "neuron"})
        
        print(f"\n🔎 Semantic search results:")
        print(f"  'plasticity': {len(plasticity_results)} nodes")
        print(f"  'memory': {len(memory_results)} nodes")
        print(f"  'neuron': {len(neuron_results)} nodes")
    
    # Test memory system integration
    print("\n💾 Phase 5: Multi-Tier Memory Integration")
    print("-" * 40)
    
    memory_stats = await memory_system.get_system_statistics()
    print(f"📊 Memory system statistics:")
    print(f"  🔥 Hot tier: {memory_stats['hot']['items']} items")
    print(f"  🌡️  Warm tier: {memory_stats['warm']['items']} items") 
    print(f"  🧊 Cold tier: {memory_stats['cold']['items']} items")
    print(f"  💾 Total memory: {memory_stats['total_size_bytes']:,} bytes")
    
    # Query memory for knowledge graphs
    kg_items = await memory_system.query_concepts({"type": "knowledge_graph"})
    print(f"  🕸️  Knowledge graphs stored: {len(kg_items)}")
    
    print("\n🎉 Complete Integration Test Results")
    print("=" * 60)
    print(f"✅ Pipeline Status: SUCCESSFUL")
    print(f"📊 Overall confidence: {statistics['average_confidence']:.3f}")
    print(f"🧠 Consciousness emergence: {extraction_result.consciousness_level:.3f}")
    print(f"⏱️  Total processing time: {extraction_result.processing_time + processing_time:.2f}s")
    print(f"🔗 Knowledge graph completeness: {len(relations) / max(len(nodes), 1):.2f} relations/node")
    
    # Performance metrics
    concepts_per_second = len(extraction_result.atomic_concepts) / extraction_result.processing_time
    print(f"⚡ Performance: {concepts_per_second:.1f} concepts/second")
    
    # Close services
    await concept_extractor.close()
    await memory_system.close()
    await autoschema_service.close()
    
    return {
        "status": "success",
        "nodes": len(nodes),
        "relations": len(relations),
        "confidence": statistics['average_confidence'],
        "consciousness": extraction_result.consciousness_level,
        "processing_time": extraction_result.processing_time + processing_time
    }

if __name__ == "__main__":
    result = asyncio.run(test_complete_integration())
    print(f"\n🏁 Final Result: {result}")