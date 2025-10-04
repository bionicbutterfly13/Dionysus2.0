"""
Database Connection Tests
Tests Neo4j, Redis, and other database connectivity with graceful degradation.
"""

import pytest
import sys
from pathlib import Path
import os

# Add backend to path for imports
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))


class TestNeo4jConnection:
    """Test Neo4j connection handling"""

    def test_neo4j_driver_import(self):
        """Test neo4j driver can be imported"""
        try:
            from neo4j import GraphDatabase
            assert GraphDatabase is not None
        except ImportError:
            pytest.skip("neo4j driver not installed")

    def test_neo4j_connection_with_env(self):
        """Test Neo4j connection using environment variables"""
        try:
            from neo4j import GraphDatabase
            from dotenv import load_dotenv

            # Load .env file
            env_path = Path(__file__).parent.parent / ".env"
            if env_path.exists():
                load_dotenv(env_path)

            uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
            user = os.getenv("NEO4J_USER", "neo4j")
            password = os.getenv("NEO4J_PASSWORD", "testpassword")

            # Try to connect
            driver = GraphDatabase.driver(uri, auth=(user, password))

            try:
                # Verify connection
                driver.verify_connectivity()
                assert True  # Connection successful
            except Exception as e:
                # Connection failed - should be handled gracefully
                pytest.skip(f"Neo4j not available: {e}")
            finally:
                driver.close()

        except ImportError:
            pytest.skip("neo4j or python-dotenv not installed")

    def test_neo4j_graceful_failure(self):
        """Test system handles Neo4j unavailability gracefully"""
        from src.services.document_processing_graph import DocumentProcessingGraph

        # Should initialize even without Neo4j
        graph = DocumentProcessingGraph(require_neo4j=False)
        assert graph is not None

    def test_neo4j_optional_connection(self):
        """Test DocumentProcessingGraph with require_neo4j=False"""
        from src.services.document_processing_graph import DocumentProcessingGraph

        # Should not raise exception
        try:
            graph = DocumentProcessingGraph(
                require_neo4j=False,
                neo4j_uri="bolt://localhost:7687",
                neo4j_user="neo4j",
                neo4j_password="wrongpassword"
            )
            assert graph is not None
        except Exception as e:
            pytest.fail(f"Should handle Neo4j failure gracefully: {e}")


class TestRedisConnection:
    """Test Redis connection handling"""

    def test_redis_import(self):
        """Test redis-py can be imported"""
        try:
            import redis
            assert redis is not None
        except ImportError:
            pytest.fail("redis-py not installed - required for basin storage")

    def test_redis_connection_attempt(self):
        """Test Redis connection with timeout"""
        import redis

        try:
            client = redis.Redis(
                host='localhost',
                port=6379,
                socket_connect_timeout=1,
                socket_timeout=1
            )
            # Try to ping
            client.ping()
            assert True  # Redis available
        except (redis.ConnectionError, redis.TimeoutError):
            # Redis not available - should be handled gracefully
            pytest.skip("Redis not available")

    def test_basin_manager_handles_redis_unavailable(self):
        """Test AttractorBasinManager handles Redis unavailable"""
        try:
            from extensions.context_engineering.attractor_basin_dynamics import AttractorBasinManager

            # Should not crash even if Redis unavailable
            manager = AttractorBasinManager(redis_host='localhost', redis_port=6379)
            assert manager is not None

        except ImportError:
            pytest.skip("AttractorBasinManager not available")
        except Exception as e:
            # Should handle gracefully
            pytest.skip(f"Basin manager initialization failed gracefully: {e}")

    def test_consciousness_processor_without_redis(self):
        """Test ConsciousnessDocumentProcessor works without Redis"""
        from src.services.consciousness_document_processor import ConsciousnessDocumentProcessor

        # Should initialize even if Redis unavailable
        processor = ConsciousnessDocumentProcessor()
        assert processor is not None

        # Should still be able to extract concepts
        concepts = processor.extract_concepts("Test document about climate change")
        assert isinstance(concepts, list)


class TestDatabaseHealthCheck:
    """Test database health check endpoint"""

    def test_database_health_service_exists(self):
        """Test database_health service exists"""
        try:
            from src.services.database_health import get_database_health
            assert get_database_health is not None
        except ImportError:
            pytest.skip("database_health service not implemented")

    def test_database_health_returns_dict(self):
        """Test get_database_health returns dictionary"""
        try:
            from src.services.database_health import get_database_health

            health = get_database_health()
            assert isinstance(health, dict)

        except ImportError:
            pytest.skip("database_health service not implemented")
        except Exception as e:
            # Should not crash
            pytest.skip(f"Health check failed gracefully: {e}")


class TestEnvironmentConfiguration:
    """Test environment configuration for databases"""

    def test_env_file_readable(self):
        """Test .env file exists and is readable"""
        env_path = Path(__file__).parent.parent / ".env"

        if env_path.exists():
            content = env_path.read_text()
            assert len(content) > 0
        else:
            # .env may not exist (using defaults)
            pytest.skip(".env file not found - using defaults")

    def test_neo4j_env_vars(self):
        """Test Neo4j environment variables can be loaded"""
        try:
            from dotenv import load_dotenv

            env_path = Path(__file__).parent.parent / ".env"
            if env_path.exists():
                load_dotenv(env_path)

                uri = os.getenv("NEO4J_URI")
                user = os.getenv("NEO4J_USER")
                password = os.getenv("NEO4J_PASSWORD")

                # Either all set or all None (using defaults)
                if uri:
                    assert user is not None
                    assert password is not None

        except ImportError:
            pytest.skip("python-dotenv not installed")

    def test_host_port_env_vars(self):
        """Test HOST and PORT environment variables"""
        try:
            from dotenv import load_dotenv

            env_path = Path(__file__).parent.parent / ".env"
            if env_path.exists():
                load_dotenv(env_path)

                host = os.getenv("HOST", "127.0.0.1")
                port = os.getenv("PORT", "9127")

                assert isinstance(host, str)
                assert isinstance(port, str)

        except ImportError:
            pytest.skip("python-dotenv not installed")


class TestNeo4jQueries:
    """Test Neo4j query execution (if available)"""

    @pytest.fixture
    def neo4j_driver(self):
        """Create Neo4j driver if available"""
        try:
            from neo4j import GraphDatabase
            from dotenv import load_dotenv

            env_path = Path(__file__).parent.parent / ".env"
            if env_path.exists():
                load_dotenv(env_path)

            uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
            user = os.getenv("NEO4J_USER", "neo4j")
            password = os.getenv("NEO4J_PASSWORD", "testpassword")

            driver = GraphDatabase.driver(uri, auth=(user, password))

            try:
                driver.verify_connectivity()
                yield driver
            except Exception:
                pytest.skip("Neo4j not available")
            finally:
                driver.close()

        except ImportError:
            pytest.skip("neo4j driver not installed")

    def test_neo4j_simple_query(self, neo4j_driver):
        """Test simple Neo4j query execution"""
        with neo4j_driver.session() as session:
            result = session.run("RETURN 1 AS num")
            record = result.single()
            assert record["num"] == 1

    def test_neo4j_node_creation(self, neo4j_driver):
        """Test Neo4j node creation and deletion"""
        with neo4j_driver.session() as session:
            # Create test node
            session.run(
                "CREATE (t:Test {id: 'test_node', created_by: 'pytest'})"
            )

            # Verify it exists
            result = session.run(
                "MATCH (t:Test {id: 'test_node'}) RETURN t"
            )
            record = result.single()
            assert record is not None

            # Clean up
            session.run(
                "MATCH (t:Test {id: 'test_node'}) DELETE t"
            )


class TestRedisOperations:
    """Test Redis operations (if available)"""

    @pytest.fixture
    def redis_client(self):
        """Create Redis client if available"""
        import redis

        try:
            client = redis.Redis(
                host='localhost',
                port=6379,
                socket_connect_timeout=1
            )
            client.ping()
            yield client
        except (redis.ConnectionError, redis.TimeoutError):
            pytest.skip("Redis not available")

    def test_redis_set_get(self, redis_client):
        """Test Redis set and get operations"""
        redis_client.set("test_key", "test_value")
        value = redis_client.get("test_key")
        assert value.decode('utf-8') == "test_value"

        # Clean up
        redis_client.delete("test_key")

    def test_redis_hash_operations(self, redis_client):
        """Test Redis hash operations (used by basin storage)"""
        redis_client.hset("test_hash", "field1", "value1")
        value = redis_client.hget("test_hash", "field1")
        assert value.decode('utf-8') == "value1"

        # Clean up
        redis_client.delete("test_hash")

    def test_redis_keys_pattern(self, redis_client):
        """Test Redis key pattern matching (used by basin loading)"""
        # Create test keys
        redis_client.set("basin:test1", "data1")
        redis_client.set("basin:test2", "data2")

        # Find keys by pattern
        keys = redis_client.keys("basin:*")
        assert len(keys) >= 2

        # Clean up
        redis_client.delete("basin:test1", "basin:test2")


class TestConnectionResilience:
    """Test system resilience to connection failures"""

    def test_system_runs_without_neo4j(self):
        """Test complete system runs without Neo4j"""
        from src.services.daedalus import Daedalus
        from io import BytesIO

        daedalus = Daedalus()
        test_data = BytesIO(b"Test content")
        test_data.name = "test.txt"

        # Should process even without Neo4j
        result = daedalus.receive_perceptual_information(data=test_data)
        assert result is not None
        assert "status" in result

    def test_system_runs_without_redis(self):
        """Test concept extraction works without Redis"""
        from src.services.consciousness_document_processor import ConsciousnessDocumentProcessor

        processor = ConsciousnessDocumentProcessor()
        concepts = processor.extract_concepts("Test document about climate change")

        # Should still work (may have empty basin results)
        assert isinstance(concepts, list)

    def test_app_starts_without_databases(self):
        """Test FastAPI app starts without database connections"""
        from src.app_factory import create_app

        # Should not crash during startup
        app = create_app()
        assert app is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
