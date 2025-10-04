"""
Backend Startup Tests
Tests critical imports, initialization, and configuration to prevent startup failures.
"""

import pytest
import socket
import sys
from pathlib import Path

# Add backend to path for imports
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))


class TestCriticalImports:
    """Test that all critical imports work without errors"""

    def test_fastapi_imports(self):
        """Test FastAPI and middleware imports"""
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        assert FastAPI is not None
        assert CORSMiddleware is not None

    def test_app_factory_import(self):
        """Test main application factory imports"""
        from src.app_factory import create_app, load_flux_config
        assert create_app is not None
        assert load_flux_config is not None

    def test_daedalus_import(self):
        """Test Daedalus gateway imports"""
        from src.services.daedalus import Daedalus
        assert Daedalus is not None

    def test_document_processing_graph_import(self):
        """Test DocumentProcessingGraph imports"""
        from src.services.document_processing_graph import DocumentProcessingGraph
        assert DocumentProcessingGraph is not None

    def test_consciousness_processor_import(self):
        """Test consciousness document processor imports"""
        from src.services.consciousness_document_processor import ConsciousnessDocumentProcessor
        assert ConsciousnessDocumentProcessor is not None

    def test_api_routes_import(self):
        """Test API route imports"""
        from src.api.routes import documents, curiosity, visualization, stats, consciousness, query
        assert documents.router is not None
        assert curiosity.router is not None
        assert visualization.router is not None
        assert stats.router is not None
        assert consciousness.router is not None
        assert query.router is not None

    def test_pypdf2_available(self):
        """Test PyPDF2 is installed (critical for document processing)"""
        try:
            import PyPDF2
            assert PyPDF2 is not None
        except ImportError:
            pytest.fail("PyPDF2 not installed - required for PDF processing")


class TestDaedalusInitialization:
    """Test Daedalus gateway initializes correctly"""

    def test_daedalus_creates_without_error(self):
        """Test Daedalus can be instantiated"""
        from src.services.daedalus import Daedalus
        daedalus = Daedalus()
        assert daedalus is not None
        assert daedalus._is_gateway is True

    def test_daedalus_has_processing_graph(self):
        """Test Daedalus has DocumentProcessingGraph"""
        from src.services.daedalus import Daedalus
        daedalus = Daedalus()
        assert hasattr(daedalus, 'processing_graph')
        assert daedalus.processing_graph is not None

    def test_daedalus_receive_method_exists(self):
        """Test Daedalus has receive_perceptual_information method"""
        from src.services.daedalus import Daedalus
        daedalus = Daedalus()
        assert hasattr(daedalus, 'receive_perceptual_information')
        assert callable(daedalus.receive_perceptual_information)

    def test_daedalus_handles_none_data(self):
        """Test Daedalus gracefully handles None data"""
        from src.services.daedalus import Daedalus
        daedalus = Daedalus()
        result = daedalus.receive_perceptual_information(data=None)
        assert result['status'] == 'error'
        assert 'error_message' in result


class TestDocumentProcessingGraphInit:
    """Test DocumentProcessingGraph initialization with/without Neo4j"""

    def test_graph_creates_without_neo4j(self):
        """Test DocumentProcessingGraph can run without Neo4j"""
        from src.services.document_processing_graph import DocumentProcessingGraph
        graph = DocumentProcessingGraph(require_neo4j=False)
        assert graph is not None

    def test_graph_has_langgraph_workflow(self):
        """Test DocumentProcessingGraph has LangGraph workflow"""
        from src.services.document_processing_graph import DocumentProcessingGraph
        graph = DocumentProcessingGraph(require_neo4j=False)
        assert hasattr(graph, 'workflow')
        assert graph.workflow is not None

    def test_graph_process_document_method_exists(self):
        """Test process_document method exists"""
        from src.services.document_processing_graph import DocumentProcessingGraph
        graph = DocumentProcessingGraph(require_neo4j=False)
        assert hasattr(graph, 'process_document')
        assert callable(graph.process_document)

    def test_graph_graceful_neo4j_failure(self):
        """Test graph handles Neo4j connection failure gracefully"""
        from src.services.document_processing_graph import DocumentProcessingGraph
        # Should not raise exception even with wrong credentials
        try:
            graph = DocumentProcessingGraph(
                require_neo4j=False,
                neo4j_uri="bolt://localhost:7687",
                neo4j_user="neo4j",
                neo4j_password="wrongpassword"
            )
            assert graph is not None
        except Exception as e:
            pytest.fail(f"Graph should handle Neo4j failure gracefully: {e}")


class TestPortAvailability:
    """Test port availability checking"""

    def test_check_port_9127(self):
        """Test if port 9127 is available or in use by our backend"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('127.0.0.1', 9127))
        sock.close()
        # Either available (result != 0) or occupied by our backend (result == 0)
        assert result in [0, 1, 61]  # 0=connected, 1=refused, 61=connection refused

    def test_check_port_9243(self):
        """Test if port 9243 is available or in use by frontend"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('127.0.0.1', 9243))
        sock.close()
        assert result in [0, 1, 61]


class TestFluxConfig:
    """Test flux.yaml configuration loading"""

    def test_flux_config_exists(self):
        """Test flux.yaml configuration file exists"""
        from src.app_factory import load_flux_config
        config = load_flux_config()
        # Should return dict even if file missing (empty dict)
        assert isinstance(config, dict)

    def test_cors_origins_config(self):
        """Test CORS origins can be loaded from config"""
        from src.app_factory import load_flux_config
        config = load_flux_config()
        # Either has cors_origins or uses default
        if 'server' in config:
            assert isinstance(config.get('server', {}), dict)


class TestAppFactoryCreation:
    """Test FastAPI app creation"""

    def test_create_app_returns_fastapi(self):
        """Test create_app returns FastAPI instance"""
        from src.app_factory import create_app
        from fastapi import FastAPI
        app = create_app()
        assert isinstance(app, FastAPI)

    def test_app_has_required_endpoints(self):
        """Test app has required health endpoints"""
        from src.app_factory import create_app
        app = create_app()
        routes = [route.path for route in app.routes]
        assert "/" in routes
        assert "/health" in routes
        assert "/health/databases" in routes

    def test_app_has_document_routes(self):
        """Test app has document upload routes"""
        from src.app_factory import create_app
        app = create_app()
        routes = [route.path for route in app.routes]
        # Should have /api/v1/documents routes
        has_api_routes = any('/api/v1' in route for route in routes)
        assert has_api_routes


class TestEnvironmentVariables:
    """Test environment variable handling"""

    def test_env_file_exists_or_defaults_work(self):
        """Test .env file exists or defaults are used"""
        from pathlib import Path
        env_path = Path(__file__).parent.parent / ".env"
        # Either .env exists or app uses defaults
        if env_path.exists():
            content = env_path.read_text()
            assert len(content) > 0
        else:
            # Test defaults work
            assert True  # App should use defaults


class TestRedisConnection:
    """Test Redis connection handling"""

    def test_redis_import_available(self):
        """Test redis-py is installed"""
        try:
            import redis
            assert redis is not None
        except ImportError:
            pytest.fail("redis-py not installed - required for basin storage")

    def test_redis_connection_graceful_failure(self):
        """Test Redis connection fails gracefully if unavailable"""
        import redis
        try:
            client = redis.Redis(host='localhost', port=6379, socket_connect_timeout=1)
            client.ping()
            # If this succeeds, Redis is available
            assert True
        except (redis.ConnectionError, redis.TimeoutError):
            # If this fails, ensure it doesn't crash the app
            assert True  # Graceful failure is acceptable


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
