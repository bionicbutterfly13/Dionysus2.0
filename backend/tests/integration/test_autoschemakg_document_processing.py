"""
TDD Test: AutoSchemaKG Integration in Document Processing
===========================================================

These tests expose that AutoSchemaKG is NOT integrated into the document
processing pipeline despite being implemented.

Author: Frustrated Developer
Date: 2025-10-07
Status: THESE SHOULD FAIL - That's the point
"""

import pytest
import sys
from pathlib import Path

# Add backend src to path
backend_src = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(backend_src))


class TestAutoSchemaKGIntegration:
    """Test that AutoSchemaKG is actually used in document processing"""

    def test_document_processing_graph_imports_autoschemakg(self):
        """CRITICAL: DocumentProcessingGraph must import AutoSchemaKG"""
        from services.document_processing_graph import DocumentProcessingGraph

        # This will FAIL because AutoSchemaKG is not imported
        graph = DocumentProcessingGraph(require_neo4j=False)

        # Should have AutoSchemaKG service
        assert hasattr(graph, 'autoschema_service'), \
            "DocumentProcessingGraph missing autoschema_service attribute"
        assert graph.autoschema_service is not None, \
            "AutoSchemaKG service not initialized"

    def test_document_processing_uses_five_level_extraction(self):
        """CRITICAL: Must use FiveLevelConceptExtraction not basic concept extraction"""
        from services.document_processing_graph import DocumentProcessingGraph

        graph = DocumentProcessingGraph(require_neo4j=False)

        # Should have five-level concept extractor
        assert hasattr(graph, 'concept_extractor'), \
            "DocumentProcessingGraph missing concept_extractor"

        # Check it's the right type
        from services.five_level_concept_extraction import FiveLevelConceptExtractionService
        assert isinstance(graph.concept_extractor, FiveLevelConceptExtractionService), \
            f"Wrong extractor type: {type(graph.concept_extractor)}"

    def test_document_processing_uses_multitier_memory(self):
        """CRITICAL: Must use MultiTierMemorySystem not direct Neo4j writes"""
        from services.document_processing_graph import DocumentProcessingGraph

        graph = DocumentProcessingGraph(require_neo4j=False)

        # Should have multi-tier memory system
        assert hasattr(graph, 'memory_system'), \
            "DocumentProcessingGraph missing memory_system"

        from services.multi_tier_memory import MultiTierMemorySystem
        assert isinstance(graph.memory_system, MultiTierMemorySystem), \
            f"Wrong memory type: {type(graph.memory_system)}"

    def test_autoschemakg_service_exists_and_works(self):
        """Verify AutoSchemaKG service can be imported and initialized"""
        from services.autoschemakg_integration import AutoSchemaKGService

        # Should initialize without Graph Channel for testing
        service = AutoSchemaKGService(graph_channel=None)
        assert service is not None
        assert hasattr(service, 'process_document_concepts')


class TestDocumentProcessingFlow:
    """Test the complete document processing flow uses all components"""

    def test_pdf_processing_creates_knowledge_graph_nodes(self):
        """Process PDF should create AutoSchemaKG knowledge graph nodes"""
        from services.daedalus import Daedalus
        import io

        # Create minimal PDF
        pdf_content = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        pdf_file = io.BytesIO(pdf_content)
        pdf_file.name = "test.pdf"

        daedalus = Daedalus()
        result = daedalus.receive_perceptual_information(pdf_file)

        # Should have knowledge graph structure
        assert 'knowledge_graph' in result, \
            "Processing result missing knowledge_graph key"

        kg = result['knowledge_graph']
        assert 'nodes' in kg, "Knowledge graph missing nodes"
        assert 'relationships' in kg, "Knowledge graph missing relationships"

        # Verify node types from AutoSchemaKG
        node_types = [node['type'] for node in kg['nodes']]
        assert 'atomic_concept' in node_types or 'composite_concept' in node_types, \
            f"Missing AutoSchemaKG node types. Found: {node_types}"

    def test_concept_extraction_five_levels(self):
        """Concept extraction should produce five levels not just flat list"""
        from services.consciousness_document_processor import ConsciousnessDocumentProcessor

        processor = ConsciousnessDocumentProcessor()

        # Process sample text
        content = b"Active inference and free energy principle in neural networks."
        result = processor.process_pdf(content, "test.pdf")

        # Should have five-level extraction
        assert hasattr(result, 'concept_hierarchy'), \
            "Missing concept_hierarchy from FiveLevelExtraction"

        hierarchy = result.concept_hierarchy
        assert 'atomic' in hierarchy, "Missing atomic level"
        assert 'relationship' in hierarchy, "Missing relationship level"
        assert 'composite' in hierarchy, "Missing composite level"
        assert 'context' in hierarchy, "Missing context level"
        assert 'narrative' in hierarchy, "Missing narrative level"

    def test_memory_tiers_used_for_storage(self):
        """Document storage should use memory tiers (warm/cool/cold)"""
        from services.document_processing_graph import DocumentProcessingGraph

        graph = DocumentProcessingGraph(require_neo4j=False)

        # Process minimal document
        result = graph.process_document(
            content=b"Test document about consciousness",
            filename="test.txt",
            tags=["test"]
        )

        # Should have memory tier information
        assert 'memory_storage' in result, \
            "Processing result missing memory_storage"

        storage = result['memory_storage']
        assert 'warm_tier' in storage, "Missing warm tier (Neo4j)"
        assert 'cool_tier' in storage, "Missing cool tier (Vector DB)"
        assert 'cold_tier' in storage, "Missing cold tier (Archive)"


class TestAutoSchemaKGNodeCreation:
    """Test that AutoSchemaKG creates proper knowledge graph structure"""

    async def test_atomic_concepts_extracted(self):
        """AutoSchemaKG should extract atomic concepts from text"""
        from services.autoschemakg_integration import AutoSchemaKGService

        service = AutoSchemaKGService(graph_channel=None)

        # Sample concepts from five-level extraction
        concepts = {
            'atomic': ['attention', 'memory', 'learning'],
            'relationship': ['attention affects memory'],
            'composite': ['attention-guided memory'],
            'context': ['cognitive neuroscience'],
            'narrative': ['memory formation process']
        }

        # Process concepts into knowledge graph
        kg_result = await service.process_document_concepts(
            concepts=concepts,
            document_id="test_doc_001"
        )

        # Should create nodes for each level
        assert len(kg_result['nodes']) > 0, "No nodes created"

        node_types = {node['type'] for node in kg_result['nodes']}
        assert 'atomic_concept' in node_types
        assert 'composite_concept' in node_types
        assert 'context' in node_types

    async def test_relationships_inferred(self):
        """AutoSchemaKG should infer relationships between concepts"""
        from services.autoschemakg_integration import AutoSchemaKGService

        service = AutoSchemaKGService(graph_channel=None)

        concepts = {
            'atomic': ['neuron', 'synapse'],
            'relationship': ['neuron connects via synapse'],
            'composite': ['neural_network']
        }

        kg_result = await service.process_document_concepts(
            concepts=concepts,
            document_id="test_doc_002"
        )

        # Should create relationships
        assert len(kg_result['relationships']) > 0, "No relationships created"

        rel_types = {rel['type'] for rel in kg_result['relationships']}
        assert 'RELATES_TO' in rel_types or 'DERIVED_FROM' in rel_types


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
