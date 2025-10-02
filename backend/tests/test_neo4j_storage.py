"""
Test Neo4j storage integration in DocumentProcessingGraph
Verifies that consciousness processing results are persisted to Neo4j
"""
import pytest
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from src.services.document_processing_graph import DocumentProcessingGraph, NEO4J_AVAILABLE


class TestNeo4jStorage:
    """Test Neo4j storage functionality"""

    def test_neo4j_import_available(self):
        """Verify Neo4j imports are working"""
        assert NEO4J_AVAILABLE, "Neo4j schema should be importable"

    @pytest.mark.skipif(not NEO4J_AVAILABLE, reason="Neo4j not available")
    def test_graph_initialization_requires_neo4j_by_default(self):
        """Test that DocumentProcessingGraph REQUIRES Neo4j connection by default"""
        # Use non-existent host to ensure connection fails
        with pytest.raises(RuntimeError, match="Neo4j connection failed|Neo4j initialization failed"):
            graph = DocumentProcessingGraph(
                neo4j_uri="bolt://nonexistent-host-12345:7687",
                neo4j_user="neo4j",
                neo4j_password="password",
                require_neo4j=True  # This is the default
            )

    @pytest.mark.skipif(not NEO4J_AVAILABLE, reason="Neo4j not available")
    def test_graph_can_initialize_without_neo4j_if_allowed(self):
        """Test that Neo4j requirement can be bypassed for testing"""
        # Allow initialization without Neo4j for testing purposes only
        # Use non-existent host
        graph = DocumentProcessingGraph(
            neo4j_uri="bolt://nonexistent-host-12345:7687",
            neo4j_user="neo4j",
            neo4j_password="password",
            require_neo4j=False  # Explicitly allow skipping
        )

        # Graph should initialize but NOT be connected
        assert graph is not None
        assert graph.neo4j_connected is False
        assert graph.processor is not None

        # Clean up
        graph.close()

    @pytest.mark.skipif(not NEO4J_AVAILABLE, reason="Neo4j not available")
    def test_processing_fails_without_neo4j_connection(self):
        """Test that document processing FAILS if Neo4j is not connected"""
        # Initialize with Neo4j disabled (for testing only)
        # Use non-existent host
        graph = DocumentProcessingGraph(
            neo4j_uri="bolt://nonexistent-host-12345:7687",
            neo4j_user="neo4j",
            neo4j_password="password",
            require_neo4j=False  # Allow init without connection
        )

        # Verify not connected
        assert graph.neo4j_connected is False

        # Processing should FAIL at finalize node
        test_content = b"This is a test document about consciousness and neural processing."
        with pytest.raises(RuntimeError, match="Neo4j not connected - cannot complete processing"):
            result = graph.process_document(
                content=test_content,
                filename="test.txt",
                tags=["test", "consciousness"],
                max_iterations=1,
                quality_threshold=0.5
            )

        # Clean up
        graph.close()

    @pytest.mark.skipif(not NEO4J_AVAILABLE, reason="Neo4j not available")
    def test_neo4j_storage_methods_exist(self):
        """Verify all Neo4j storage methods are defined"""
        # Allow init without connection for testing
        graph = DocumentProcessingGraph(require_neo4j=False)

        # Check storage methods exist
        assert hasattr(graph, '_store_to_neo4j')
        assert hasattr(graph, '_create_document_node')
        assert hasattr(graph, '_create_concept_node')
        assert hasattr(graph, '_create_basin_node')
        assert hasattr(graph, '_link_curiosity_trigger')
        assert hasattr(graph, 'close')

        graph.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
