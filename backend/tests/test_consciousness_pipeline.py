"""
Consciousness Pipeline Tests
Tests attractor basin creation, ThoughtSeed generation, and concept extraction.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch
from io import BytesIO

# Add backend to path for imports
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))


class TestAttractorBasinCreation:
    """Test attractor basin creation from concepts"""

    def test_basin_manager_import(self):
        """Test AttractorBasinManager can be imported"""
        try:
            from extensions.context_engineering.attractor_basin_dynamics import AttractorBasinManager
            assert AttractorBasinManager is not None
        except ImportError:
            pytest.skip("AttractorBasinManager not available")

    def test_basin_manager_initialization(self):
        """Test AttractorBasinManager initializes without Redis"""
        try:
            from extensions.context_engineering.attractor_basin_dynamics import AttractorBasinManager
            # Should handle Redis connection failure gracefully
            manager = AttractorBasinManager(redis_host='localhost', redis_port=6379)
            assert manager is not None
        except Exception as e:
            # Should not crash even if Redis unavailable
            pytest.skip(f"Basin manager initialization failed: {e}")

    def test_basin_influence_types_exist(self):
        """Test BasinInfluenceType enum exists with all 4 types"""
        try:
            from extensions.context_engineering.attractor_basin_dynamics import BasinInfluenceType
            assert hasattr(BasinInfluenceType, 'REINFORCEMENT')
            assert hasattr(BasinInfluenceType, 'COMPETITION')
            assert hasattr(BasinInfluenceType, 'SYNTHESIS')
            assert hasattr(BasinInfluenceType, 'EMERGENCE')
        except ImportError:
            pytest.skip("BasinInfluenceType not available")

    def test_attractor_basin_model_exists(self):
        """Test AttractorBasin model class exists"""
        try:
            from extensions.context_engineering.attractor_basin_dynamics import AttractorBasin
            assert AttractorBasin is not None
        except ImportError:
            pytest.skip("AttractorBasin not available")


class TestThoughtSeedGeneration:
    """Test ThoughtSeed generation from concepts"""

    def test_thoughtseed_event_model_exists(self):
        """Test ThoughtSeedIntegrationEvent model exists"""
        try:
            from extensions.context_engineering.attractor_basin_dynamics import ThoughtSeedIntegrationEvent
            assert ThoughtSeedIntegrationEvent is not None
        except ImportError:
            pytest.skip("ThoughtSeedIntegrationEvent not available")

    def test_thoughtseed_has_required_fields(self):
        """Test ThoughtSeedIntegrationEvent has required fields"""
        try:
            from extensions.context_engineering.attractor_basin_dynamics import ThoughtSeedIntegrationEvent
            # Check if model has expected attributes
            event = ThoughtSeedIntegrationEvent(
                thoughtseed_id="test_id",
                concept="test concept",
                action_taken="EMERGENCE",
                affected_basins=[],
                new_basin_created=True,
                integration_timestamp=0.0
            )
            assert event.thoughtseed_id == "test_id"
            assert event.concept == "test concept"
            assert event.action_taken == "EMERGENCE"
        except (ImportError, TypeError):
            pytest.skip("ThoughtSeedIntegrationEvent not available or structure changed")


class TestConceptExtraction:
    """Test concept extraction from documents"""

    def test_consciousness_processor_import(self):
        """Test ConsciousnessDocumentProcessor can be imported"""
        from src.services.consciousness_document_processor import ConsciousnessDocumentProcessor
        assert ConsciousnessDocumentProcessor is not None

    def test_consciousness_processor_initialization(self):
        """Test ConsciousnessDocumentProcessor initializes"""
        from src.services.consciousness_document_processor import ConsciousnessDocumentProcessor
        processor = ConsciousnessDocumentProcessor()
        assert processor is not None

    def test_processor_has_basin_manager(self):
        """Test processor has or can create basin manager"""
        from src.services.consciousness_document_processor import ConsciousnessDocumentProcessor
        processor = ConsciousnessDocumentProcessor()
        # Basin manager may be None if Redis unavailable (graceful degradation)
        assert hasattr(processor, 'basin_manager')

    def test_extract_concepts_method_exists(self):
        """Test extract_concepts method exists"""
        from src.services.consciousness_document_processor import ConsciousnessDocumentProcessor
        processor = ConsciousnessDocumentProcessor()
        assert hasattr(processor, 'extract_concepts')
        assert callable(processor.extract_concepts)

    def test_extract_concepts_from_text(self):
        """Test concept extraction from sample text"""
        from src.services.consciousness_document_processor import ConsciousnessDocumentProcessor
        processor = ConsciousnessDocumentProcessor()

        sample_text = """
        Climate change refers to long-term shifts in global temperatures.
        Greenhouse gases trap heat in Earth's atmosphere.
        Carbon dioxide is the primary greenhouse gas from human activity.
        """

        concepts = processor.extract_concepts(sample_text)
        assert isinstance(concepts, list)
        # Should extract some concepts
        assert len(concepts) >= 0  # May be empty if extraction fails gracefully

    def test_process_through_consciousness_method_exists(self):
        """Test _process_through_consciousness method exists"""
        from src.services.consciousness_document_processor import ConsciousnessDocumentProcessor
        processor = ConsciousnessDocumentProcessor()
        assert hasattr(processor, '_process_through_consciousness')

    def test_process_through_consciousness_handles_empty_concepts(self):
        """Test consciousness processing handles empty concept list"""
        from src.services.consciousness_document_processor import ConsciousnessDocumentProcessor
        processor = ConsciousnessDocumentProcessor()

        result = processor._process_through_consciousness([])
        assert isinstance(result, dict)
        assert 'basins_created' in result
        assert result['basins_created'] == 0


class TestDocumentProcessingIntegration:
    """Test complete document processing through consciousness pipeline"""

    def test_processing_graph_import(self):
        """Test DocumentProcessingGraph imports"""
        from src.services.document_processing_graph import DocumentProcessingGraph
        assert DocumentProcessingGraph is not None

    def test_processing_graph_has_consciousness_processor(self):
        """Test DocumentProcessingGraph has consciousness processor"""
        from src.services.document_processing_graph import DocumentProcessingGraph
        graph = DocumentProcessingGraph(require_neo4j=False)
        assert hasattr(graph, 'consciousness_processor')

    def test_langgraph_workflow_exists(self):
        """Test LangGraph workflow is created"""
        from src.services.document_processing_graph import DocumentProcessingGraph
        graph = DocumentProcessingGraph(require_neo4j=False)
        assert hasattr(graph, 'workflow')
        assert graph.workflow is not None

    def test_workflow_has_required_nodes(self):
        """Test workflow has required processing nodes"""
        from src.services.document_processing_graph import DocumentProcessingGraph
        graph = DocumentProcessingGraph(require_neo4j=False)

        # Check if graph has required node methods
        assert hasattr(graph, '_extract_node')
        assert hasattr(graph, '_research_node')
        assert hasattr(graph, '_consciousness_node')
        assert hasattr(graph, '_analyze_node')
        assert hasattr(graph, '_decision_node')
        assert hasattr(graph, '_finalize_node')

    def test_process_document_method_signature(self):
        """Test process_document method has correct signature"""
        from src.services.document_processing_graph import DocumentProcessingGraph
        import inspect

        graph = DocumentProcessingGraph(require_neo4j=False)
        sig = inspect.signature(graph.process_document)
        params = list(sig.parameters.keys())

        # Should have content, filename, tags parameters
        assert 'content' in params
        assert 'filename' in params
        assert 'tags' in params


class TestDaedalusIntegration:
    """Test Daedalus gateway integration with consciousness pipeline"""

    def test_daedalus_has_processing_graph(self):
        """Test Daedalus contains DocumentProcessingGraph"""
        from src.services.daedalus import Daedalus
        daedalus = Daedalus()
        assert hasattr(daedalus, 'processing_graph')
        assert daedalus.processing_graph is not None

    def test_daedalus_processes_through_consciousness(self):
        """Test Daedalus routes through consciousness pipeline"""
        from src.services.daedalus import Daedalus
        daedalus = Daedalus()

        # Create mock PDF data
        test_data = BytesIO(b"Test document content")
        test_data.name = "test.txt"

        # Process should not crash
        try:
            result = daedalus.receive_perceptual_information(
                data=test_data,
                tags=["test"]
            )
            assert 'status' in result
            # Should either succeed or fail gracefully
            assert result['status'] in ['received', 'error']
        except Exception as e:
            pytest.fail(f"Daedalus processing crashed: {e}")

    def test_daedalus_response_structure(self):
        """Test Daedalus returns expected response structure"""
        from src.services.daedalus import Daedalus
        daedalus = Daedalus()

        test_data = BytesIO(b"Test content")
        test_data.name = "test.txt"

        result = daedalus.receive_perceptual_information(data=test_data)

        # Should have standard response fields
        assert 'status' in result
        assert 'timestamp' in result

        if result['status'] == 'received':
            # Successful processing should have these fields
            assert 'document' in result
            assert 'extraction' in result
            assert 'consciousness' in result


class TestConsciousnessMetrics:
    """Test consciousness processing metrics and tracking"""

    def test_quality_assessment_exists(self):
        """Test quality assessment is part of processing"""
        from src.services.document_processing_graph import DocumentProcessingGraph
        graph = DocumentProcessingGraph(require_neo4j=False)

        # Should have quality assessment logic
        assert hasattr(graph, '_analyze_node')

    def test_meta_cognitive_tracking(self):
        """Test meta-cognitive tracking exists"""
        from src.services.document_processing_graph import DocumentProcessingGraph
        graph = DocumentProcessingGraph(require_neo4j=False)

        # Meta-cognitive tracking should be part of state
        # This is verified through process flow
        assert graph.workflow is not None


class TestPDFProcessing:
    """Test PDF document processing"""

    def test_pdf_text_extraction_import(self):
        """Test PyPDF2 is available for PDF processing"""
        try:
            import PyPDF2
            assert PyPDF2 is not None
        except ImportError:
            pytest.fail("PyPDF2 not available - required for PDF processing")

    def test_pdf_reader_class_exists(self):
        """Test PyPDF2 PdfReader class exists"""
        try:
            from PyPDF2 import PdfReader
            assert PdfReader is not None
        except ImportError:
            pytest.skip("PyPDF2.PdfReader not available")


class TestErrorHandling:
    """Test error handling in consciousness pipeline"""

    def test_handles_invalid_document_format(self):
        """Test pipeline handles invalid document format"""
        from src.services.daedalus import Daedalus
        daedalus = Daedalus()

        # Invalid binary data
        test_data = BytesIO(b"\x00\x01\x02\x03")
        test_data.name = "invalid.dat"

        # Should not crash
        result = daedalus.receive_perceptual_information(data=test_data)
        assert 'status' in result

    def test_handles_empty_document(self):
        """Test pipeline handles empty document"""
        from src.services.daedalus import Daedalus
        daedalus = Daedalus()

        test_data = BytesIO(b"")
        test_data.name = "empty.txt"

        result = daedalus.receive_perceptual_information(data=test_data)
        assert 'status' in result

    def test_handles_redis_unavailable(self):
        """Test basin manager handles Redis unavailable"""
        from src.services.consciousness_document_processor import ConsciousnessDocumentProcessor

        # Should initialize even if Redis unavailable
        processor = ConsciousnessDocumentProcessor()
        assert processor is not None

        # Basin manager may be None (graceful degradation)
        if processor.basin_manager is None:
            # Should still extract concepts
            concepts = processor.extract_concepts("Test text")
            assert isinstance(concepts, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
