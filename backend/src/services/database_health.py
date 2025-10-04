"""
Database Health Service for Flux
Handles health checks for Neo4j, Redis, and Qdrant databases.
"""
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass

# Import database clients with graceful fallback
try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    from qdrant_client import QdrantClient
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class DatabaseHealth:
    """Database health status information."""
    status: str  # 'healthy', 'unhealthy', 'unavailable'
    message: str
    timestamp: datetime
    response_time_ms: Optional[float] = None
    additional_info: Optional[Dict[str, Any]] = None


class DatabaseHealthService:
    """
    Service for checking database health and connectivity.

    Provides health checks for:
    - Neo4j (graph database)
    - Redis (caching and streams)
    - Qdrant (vector embeddings)
    """

    def __init__(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "flux_password",
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        timeout: float = 5.0
    ):
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.timeout = timeout

    def check_neo4j_health(self) -> Dict[str, Any]:
        """
        Check Neo4j database health.

        Returns:
            Dict with status, message, timestamp, and response time
        """
        start_time = time.time()
        timestamp = datetime.now()

        if not NEO4J_AVAILABLE:
            return {
                'status': 'unavailable',
                'message': 'Neo4j client not installed',
                'timestamp': timestamp.isoformat(),
                'response_time_ms': None
            }

        try:
            driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password)
            )

            # Test connection with a simple query
            with driver.session() as session:
                result = session.run("RETURN 1 as test")
                test_value = result.single()["test"]

                if test_value == 1:
                    response_time = (time.time() - start_time) * 1000
                    driver.close()

                    return {
                        'status': 'healthy',
                        'message': 'Neo4j connection successful',
                        'timestamp': timestamp.isoformat(),
                        'response_time_ms': round(response_time, 2)
                    }

        except Exception as e:
            logger.warning(f"Neo4j health check failed: {e}")

        response_time = (time.time() - start_time) * 1000
        return {
            'status': 'unavailable',
            'message': 'Neo4j connection failed',
            'timestamp': timestamp.isoformat(),
            'response_time_ms': round(response_time, 2)
        }

    def check_redis_health(self) -> Dict[str, Any]:
        """
        Check Redis database health.

        Returns:
            Dict with status, message, timestamp, and response time
        """
        start_time = time.time()
        timestamp = datetime.now()

        if not REDIS_AVAILABLE:
            return {
                'status': 'unavailable',
                'message': 'Redis client not installed',
                'timestamp': timestamp.isoformat(),
                'response_time_ms': None
            }

        try:
            client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                db=self.redis_db,
                socket_timeout=self.timeout,
                socket_connect_timeout=self.timeout
            )

            # Test connection with ping
            pong = client.ping()

            if pong:
                response_time = (time.time() - start_time) * 1000
                client.close()

                return {
                    'status': 'healthy',
                    'message': 'Redis connection successful',
                    'timestamp': timestamp.isoformat(),
                    'response_time_ms': round(response_time, 2)
                }

        except Exception as e:
            logger.warning(f"Redis health check failed: {e}")

        response_time = (time.time() - start_time) * 1000
        return {
            'status': 'unavailable',
            'message': 'Redis connection failed',
            'timestamp': timestamp.isoformat(),
            'response_time_ms': round(response_time, 2)
        }

    def check_qdrant_health(self) -> Dict[str, Any]:
        """
        Check Qdrant database health.

        Returns:
            Dict with status, message, timestamp, and response time
        """
        start_time = time.time()
        timestamp = datetime.now()

        if not QDRANT_AVAILABLE:
            return {
                'status': 'unavailable',
                'message': 'Qdrant client not installed',
                'timestamp': timestamp.isoformat(),
                'response_time_ms': None
            }

        try:
            client = QdrantClient(
                host=self.qdrant_host,
                port=self.qdrant_port,
                timeout=self.timeout
            )

            # Test connection by getting cluster info
            cluster_info = client.get_cluster_info()

            if cluster_info:
                response_time = (time.time() - start_time) * 1000

                return {
                    'status': 'healthy',
                    'message': 'Qdrant connection successful',
                    'timestamp': timestamp.isoformat(),
                    'response_time_ms': round(response_time, 2),
                    'additional_info': {
                        'peer_id': cluster_info.peer_id if hasattr(cluster_info, 'peer_id') else None
                    }
                }

        except Exception as e:
            logger.warning(f"Qdrant health check failed: {e}")

        response_time = (time.time() - start_time) * 1000
        return {
            'status': 'unavailable',
            'message': 'Qdrant connection failed',
            'timestamp': timestamp.isoformat(),
            'response_time_ms': round(response_time, 2)
        }

    def check_all_databases(self) -> Dict[str, Any]:
        """
        Check health of all databases comprehensively.

        Returns:
            Dict with status for each database and overall status
        """
        timestamp = datetime.now()

        # Check each database
        neo4j_health = self.check_neo4j_health()
        redis_health = self.check_redis_health()
        qdrant_health = self.check_qdrant_health()

        # Determine overall status
        all_statuses = [
            neo4j_health['status'],
            redis_health['status'],
            qdrant_health['status']
        ]

        if all(status == 'healthy' for status in all_statuses):
            overall_status = 'healthy'
        elif any(status == 'healthy' for status in all_statuses):
            overall_status = 'partial'
        else:
            overall_status = 'unavailable'

        return {
            'neo4j': neo4j_health,
            'redis': redis_health,
            'qdrant': qdrant_health,
            'overall_status': overall_status,
            'timestamp': timestamp.isoformat(),
            'healthy_count': len([s for s in all_statuses if s == 'healthy']),
            'total_count': len(all_statuses)
        }


# Global service instance for application use
# Use settings from config module
from ..config.settings import settings
database_health_service = DatabaseHealthService(
    neo4j_uri=settings.NEO4J_URI,
    neo4j_user=settings.NEO4J_USER,
    neo4j_password=settings.NEO4J_PASSWORD
)


def get_database_health() -> Dict[str, Any]:
    """Quick function to get comprehensive database health."""
    return database_health_service.check_all_databases()


def is_database_healthy(database_name: str) -> bool:
    """Check if a specific database is healthy."""
    if database_name == 'neo4j':
        health = database_health_service.check_neo4j_health()
    elif database_name == 'redis':
        health = database_health_service.check_redis_health()
    elif database_name == 'qdrant':
        health = database_health_service.check_qdrant_health()
    else:
        return False

    return health['status'] == 'healthy'