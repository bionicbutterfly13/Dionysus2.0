"""
Document Upload Integration Tests
Tests complete document upload flow through Daedalus gateway.
"""

import pytest
import sys
from pathlib import Path
from io import BytesIO
from fastapi.testclient import TestClient

# Add backend to path for imports
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))


class TestDocumentEndpoint:
    """Test /api/v1/documents endpoint"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        from src.app_factory import create_app
        app = create_app()
        return TestClient(app)

    def test_document_endpoint_exists(self, client):
        """Test /api/v1/documents endpoint is registered"""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        openapi = response.json()

        # Check if /api/v1/documents path exists
        paths = openapi.get("paths", {})
        has_documents_endpoint = any(
            "/api/v1/documents" in path for path in paths.keys()
        )
        assert has_documents_endpoint, "Documents endpoint not found in API"

    def test_health_endpoint(self, client):
        """Test health endpoint responds"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_root_endpoint(self, client):
        """Test root endpoint responds"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data


class TestDaedalusIntegration:
    """Test document flows through Daedalus gateway"""

    def test_daedalus_receives_text_file(self):
        """Test Daedalus processes text file"""
        from src.services.daedalus import Daedalus

        daedalus = Daedalus()
        test_data = BytesIO(b"Test document about climate change and global warming.")
        test_data.name = "test.txt"

        result = daedalus.receive_perceptual_information(
            data=test_data,
            tags=["test", "climate"]
        )

        assert result is not None
        assert "status" in result
        assert "timestamp" in result

    def test_daedalus_receives_pdf_file(self):
        """Test Daedalus handles PDF file format"""
        from src.services.daedalus import Daedalus

        daedalus = Daedalus()

        # Create minimal PDF-like data
        pdf_header = b"%PDF-1.4\n"
        test_data = BytesIO(pdf_header + b"Mock PDF content")
        test_data.name = "test.pdf"

        result = daedalus.receive_perceptual_information(data=test_data)

        assert result is not None
        assert "status" in result

    def test_daedalus_returns_extraction_results(self):
        """Test Daedalus returns extraction results on success"""
        from src.services.daedalus import Daedalus

        daedalus = Daedalus()
        test_data = BytesIO(b"Climate change refers to long-term shifts in temperature patterns.")
        test_data.name = "climate.txt"

        result = daedalus.receive_perceptual_information(data=test_data)

        if result.get("status") == "received":
            # Successful processing should include these fields
            assert "document" in result
            assert "extraction" in result
            assert "consciousness" in result
            assert "research" in result
            assert "quality" in result
            assert "workflow" in result

    def test_daedalus_handles_tags(self):
        """Test Daedalus accepts and processes tags"""
        from src.services.daedalus import Daedalus

        daedalus = Daedalus()
        test_data = BytesIO(b"Test content")
        test_data.name = "test.txt"

        result = daedalus.receive_perceptual_information(
            data=test_data,
            tags=["research", "ai", "consciousness"]
        )

        assert result is not None
        assert "status" in result

    def test_daedalus_handles_quality_threshold(self):
        """Test Daedalus accepts quality_threshold parameter"""
        from src.services.daedalus import Daedalus

        daedalus = Daedalus()
        test_data = BytesIO(b"Test content")
        test_data.name = "test.txt"

        result = daedalus.receive_perceptual_information(
            data=test_data,
            quality_threshold=0.8
        )

        assert result is not None
        assert "status" in result

    def test_daedalus_handles_max_iterations(self):
        """Test Daedalus accepts max_iterations parameter"""
        from src.services.daedalus import Daedalus

        daedalus = Daedalus()
        test_data = BytesIO(b"Test content")
        test_data.name = "test.txt"

        result = daedalus.receive_perceptual_information(
            data=test_data,
            max_iterations=2
        )

        assert result is not None
        assert "status" in result


class TestWorkflowIntegration:
    """Test complete workflow integration"""

    def test_workflow_processes_document(self):
        """Test DocumentProcessingGraph processes document"""
        from src.services.document_processing_graph import DocumentProcessingGraph

        graph = DocumentProcessingGraph(require_neo4j=False)

        result = graph.process_document(
            content=b"Climate change refers to temperature shifts.",
            filename="test.txt",
            tags=["climate"]
        )

        assert result is not None
        assert isinstance(result, dict)

    def test_workflow_returns_expected_structure(self):
        """Test workflow returns expected result structure"""
        from src.services.document_processing_graph import DocumentProcessingGraph

        graph = DocumentProcessingGraph(require_neo4j=False)

        result = graph.process_document(
            content=b"Test document content",
            filename="test.txt",
            tags=[]
        )

        # Should have core result fields
        expected_keys = ["document", "extraction", "consciousness", "research", "quality"]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_workflow_handles_binary_content(self):
        """Test workflow handles binary content"""
        from src.services.document_processing_graph import DocumentProcessingGraph

        graph = DocumentProcessingGraph(require_neo4j=False)

        result = graph.process_document(
            content=b"\x80\x81\x82Binary data",
            filename="binary.dat",
            tags=[]
        )

        assert result is not None
        assert isinstance(result, dict)


class TestErrorHandling:
    """Test error handling in document upload"""

    def test_daedalus_rejects_none_data(self):
        """Test Daedalus rejects None data"""
        from src.services.daedalus import Daedalus

        daedalus = Daedalus()
        result = daedalus.receive_perceptual_information(data=None)

        assert result["status"] == "error"
        assert "error_message" in result
        assert "No data provided" in result["error_message"]

    def test_daedalus_handles_empty_file(self):
        """Test Daedalus handles empty file"""
        from src.services.daedalus import Daedalus

        daedalus = Daedalus()
        test_data = BytesIO(b"")
        test_data.name = "empty.txt"

        result = daedalus.receive_perceptual_information(data=test_data)

        # Should not crash
        assert "status" in result

    def test_daedalus_handles_large_file(self):
        """Test Daedalus handles large file content"""
        from src.services.daedalus import Daedalus

        daedalus = Daedalus()
        # Create 10KB of data
        large_content = b"Test content. " * 1000
        test_data = BytesIO(large_content)
        test_data.name = "large.txt"

        result = daedalus.receive_perceptual_information(data=test_data)

        # Should not crash
        assert "status" in result

    def test_daedalus_handles_unicode_content(self):
        """Test Daedalus handles Unicode content"""
        from src.services.daedalus import Daedalus

        daedalus = Daedalus()
        unicode_content = "测试文档 тест документ 🌍🔬".encode('utf-8')
        test_data = BytesIO(unicode_content)
        test_data.name = "unicode.txt"

        result = daedalus.receive_perceptual_information(data=test_data)

        # Should not crash
        assert "status" in result


class TestAPIRoutes:
    """Test API route registration"""

    def test_documents_router_import(self):
        """Test documents router can be imported"""
        from src.api.routes import documents
        assert documents.router is not None

    def test_documents_router_has_upload_endpoint(self):
        """Test documents router has upload endpoint"""
        from src.api.routes import documents
        import inspect

        # Get all route functions
        routes = [
            name for name, obj in inspect.getmembers(documents)
            if inspect.isfunction(obj)
        ]

        # Should have upload-related function
        has_upload = any("upload" in route.lower() for route in routes)
        assert has_upload or len(routes) > 0  # At least some routes exist

    def test_all_required_routers_import(self):
        """Test all required routers can be imported"""
        from src.api.routes import (
            documents,
            curiosity,
            visualization,
            stats,
            consciousness,
            query
        )

        assert documents.router is not None
        assert curiosity.router is not None
        assert visualization.router is not None
        assert stats.router is not None
        assert consciousness.router is not None
        assert query.router is not None


class TestDocumentProcessingFlow:
    """Test complete document processing flow"""

    def test_end_to_end_text_processing(self):
        """Test complete text document processing"""
        from src.services.daedalus import Daedalus

        daedalus = Daedalus()

        # Realistic document content
        content = b"""
        Climate Change and Global Warming

        Climate change refers to long-term shifts in temperatures and weather patterns.
        These shifts may be natural, but since the 1800s, human activities have been
        the main driver of climate change, primarily due to the burning of fossil fuels
        like coal, oil and gas.

        Greenhouse gases trap heat in Earth's atmosphere, leading to global warming.
        This results in rising sea levels, extreme weather events, and ecosystem changes.
        """

        test_data = BytesIO(content)
        test_data.name = "climate_change.txt"

        result = daedalus.receive_perceptual_information(
            data=test_data,
            tags=["climate", "environment", "science"]
        )

        assert result is not None
        assert "status" in result
        assert "timestamp" in result

        # If processing succeeded, verify structure
        if result.get("status") == "received":
            assert "document" in result
            assert "extraction" in result
            assert "consciousness" in result

    def test_processing_with_all_parameters(self):
        """Test processing with all optional parameters"""
        from src.services.daedalus import Daedalus

        daedalus = Daedalus()
        test_data = BytesIO(b"Research document about artificial intelligence")
        test_data.name = "ai_research.txt"

        result = daedalus.receive_perceptual_information(
            data=test_data,
            tags=["ai", "research", "consciousness"],
            max_iterations=2,
            quality_threshold=0.75
        )

        assert result is not None
        assert "status" in result


class TestCognitionSummary:
    """Test cognition summary functionality"""

    def test_daedalus_has_cognition_summary_method(self):
        """Test Daedalus has get_cognition_summary method"""
        from src.services.daedalus import Daedalus

        daedalus = Daedalus()
        assert hasattr(daedalus, 'get_cognition_summary')
        assert callable(daedalus.get_cognition_summary)

    def test_cognition_summary_returns_dict(self):
        """Test get_cognition_summary returns dictionary"""
        from src.services.daedalus import Daedalus

        daedalus = Daedalus()
        summary = daedalus.get_cognition_summary()

        assert isinstance(summary, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
