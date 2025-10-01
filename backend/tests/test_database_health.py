"""
Test database connection health checks.
This test ensures we can connect to and verify the health of all databases.
"""
import pytest
from src.services.database_health import DatabaseHealthService


class TestDatabaseHealth:
    """Test suite for database health checking."""

    def test_database_health_service_exists(self):
        """Test that the DatabaseHealthService can be instantiated."""
        health_service = DatabaseHealthService()
        assert health_service is not None

    def test_neo4j_health_check(self):
        """Test Neo4j connection health check."""
        health_service = DatabaseHealthService()

        # Should return a health status dict
        neo4j_health = health_service.check_neo4j_health()

        assert isinstance(neo4j_health, dict)
        assert 'status' in neo4j_health
        assert 'message' in neo4j_health
        assert 'timestamp' in neo4j_health

        # Status should be either 'healthy', 'unhealthy', or 'unavailable'
        assert neo4j_health['status'] in ['healthy', 'unhealthy', 'unavailable']

    def test_redis_health_check(self):
        """Test Redis connection health check."""
        health_service = DatabaseHealthService()

        redis_health = health_service.check_redis_health()

        assert isinstance(redis_health, dict)
        assert 'status' in redis_health
        assert 'message' in redis_health
        assert 'timestamp' in redis_health
        assert redis_health['status'] in ['healthy', 'unhealthy', 'unavailable']

    def test_qdrant_health_check(self):
        """Test Qdrant connection health check."""
        health_service = DatabaseHealthService()

        qdrant_health = health_service.check_qdrant_health()

        assert isinstance(qdrant_health, dict)
        assert 'status' in qdrant_health
        assert 'message' in qdrant_health
        assert 'timestamp' in qdrant_health
        assert qdrant_health['status'] in ['healthy', 'unhealthy', 'unavailable']

    def test_comprehensive_health_check(self):
        """Test comprehensive health check for all databases."""
        health_service = DatabaseHealthService()

        all_health = health_service.check_all_databases()

        # Should return status for all three databases
        assert isinstance(all_health, dict)
        assert 'neo4j' in all_health
        assert 'redis' in all_health
        assert 'qdrant' in all_health
        assert 'overall_status' in all_health
        assert 'timestamp' in all_health

        # Each database should have proper health info
        for db_name in ['neo4j', 'redis', 'qdrant']:
            db_health = all_health[db_name]
            assert isinstance(db_health, dict)
            assert 'status' in db_health
            assert db_health['status'] in ['healthy', 'unhealthy', 'unavailable']

    def test_health_check_with_timeout(self):
        """Test that health checks respect timeout settings."""
        health_service = DatabaseHealthService(timeout=1)  # 1 second timeout

        # Should complete quickly and not hang
        import time
        start_time = time.time()
        health = health_service.check_all_databases()
        end_time = time.time()

        # Should complete within reasonable time (much less than 5 seconds)
        assert (end_time - start_time) < 5.0
        assert isinstance(health, dict)

    def test_health_check_graceful_failure(self):
        """Test that health checks handle connection failures gracefully."""
        # Use invalid connection details to simulate failures
        health_service = DatabaseHealthService(
            neo4j_uri="bolt://invalid:7687",
            redis_host="invalid_host",
            qdrant_host="invalid_host"
        )

        # Should not crash, should return 'unavailable' status
        health = health_service.check_all_databases()

        assert isinstance(health, dict)
        assert health['overall_status'] in ['unhealthy', 'unavailable']

        # All databases should report as unavailable
        for db_name in ['neo4j', 'redis', 'qdrant']:
            assert health[db_name]['status'] == 'unavailable'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])