"""
Database Health Service for Flux
Handles health checks for Neo4j, Redis, and Qdrant databases.

Constitutional Compliance (Spec 040 M2):
- Uses DaedalusGraphChannel for Neo4j health checks
- No direct neo4j imports (constitutional violation)
- Maintains existing API for backwards compatibility

Migration Status:
- Neo4j health checks: MIGRATED to Graph Channel
- Redis health checks: No changes (not graph database)
- Qdrant health checks: No changes (not graph database)
"""
import asyncio
import logging
import time
import warnings
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass

# CONSTITUTIONAL COMPLIANCE: Use Graph Channel for Neo4j
try:
    from daedalus_gateway import get_graph_channel, GraphChannelConfig
    GRAPH_CHANNEL_AVAILABLE = True
except ImportError:
    GRAPH_CHANNEL_AVAILABLE = False
    warnings.warn(
        "daedalus_gateway not available. Neo4j health checks will fail.",
        ImportWarning
    )

# DEPRECATED: Legacy neo4j import for backwards compatibility only
try:
    from neo4j import GraphDatabase
    NEO4J_LEGACY_AVAILABLE = True
except ImportError:
    NEO4J_LEGACY_AVAILABLE = False

# Non-graph database clients (no constitutional restrictions)
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
    - Neo4j (graph database) - via DaedalusGraphChannel (CONSTITUTIONAL)
    - Redis (caching and streams) - direct connection (not graph database)
    - Qdrant (vector embeddings) - direct connection (not graph database)

    Constitutional Compliance:
    - Neo4j health checks use Graph Channel exclusively
    - Legacy direct neo4j access deprecated and will be removed
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

        # Graph Channel instance (lazy initialized)
        self._graph_channel = None

    def _get_graph_channel(self):
        """Get or create Graph Channel instance."""
        if self._graph_channel is None and GRAPH_CHANNEL_AVAILABLE:
            config = GraphChannelConfig(
                neo4j_uri=self.neo4j_uri,
                neo4j_user=self.neo4j_user,
                neo4j_password=self.neo4j_password
            )
            self._graph_channel = get_graph_channel(config)
        return self._graph_channel

    async def check_neo4j_health_async(self) -> Dict[str, Any]:
        """
        Check Neo4j database health using Graph Channel.

        RECOMMENDED: Use this async version for constitutional compliance.

        Returns:
            Dict with status, message, timestamp, and response time
        """
        start_time = time.time()
        timestamp = datetime.now()

        if not GRAPH_CHANNEL_AVAILABLE:
            return {
                'status': 'unavailable',
                'message': 'Graph Channel not available',
                'timestamp': timestamp.isoformat(),
                'response_time_ms': None
            }

        try:
            channel = self._get_graph_channel()

            # Ensure connection
            if not channel.connected:
                await channel.connect()

            # Use Graph Channel health check
            health_result = await channel.health_check()

            response_time = (time.time() - start_time) * 1000

            if health_result.get('connected'):
                return {
                    'status': 'healthy',
                    'message': 'Neo4j connection successful via Graph Channel',
                    'timestamp': timestamp.isoformat(),
                    'response_time_ms': round(response_time, 2),
                    'additional_info': {
                        'circuit_open': health_result.get('circuit_open', False),
                        'success_rate': health_result.get('success_rate', 0.0)
                    }
                }
            else:
                return {
                    'status': 'unavailable',
                    'message': 'Neo4j connection failed via Graph Channel',
                    'timestamp': timestamp.isoformat(),
                    'response_time_ms': round(response_time, 2)
                }

        except Exception as e:
            logger.warning(f"Neo4j health check failed: {e}")
            response_time = (time.time() - start_time) * 1000
            return {
                'status': 'unavailable',
                'message': f'Neo4j health check error: {str(e)}',
                'timestamp': timestamp.isoformat(),
                'response_time_ms': round(response_time, 2)
            }

    def check_neo4j_health(self) -> Dict[str, Any]:
        """
        Check Neo4j database health.

        DEPRECATED: Synchronous version. Use check_neo4j_health_async() for Graph Channel compliance.

        Returns:
            Dict with status, message, timestamp, and response time
        """
        warnings.warn(
            "check_neo4j_health() synchronous method is deprecated. Use check_neo4j_health_async() instead.",
            DeprecationWarning,
            stacklevel=2
        )

        # If Graph Channel is available, use async version
        if GRAPH_CHANNEL_AVAILABLE:
            return asyncio.run(self.check_neo4j_health_async())

        # Fallback to legacy direct driver access
        start_time = time.time()
        timestamp = datetime.now()

        if not NEO4J_LEGACY_AVAILABLE:
            return {
                'status': 'unavailable',
                'message': 'Neo4j client not installed and Graph Channel unavailable',
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
                        'message': 'Neo4j connection successful (LEGACY)',
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

    async def check_all_databases_async(self) -> Dict[str, Any]:
        """
        Check health of all databases comprehensively using async methods.

        RECOMMENDED: Use this async version for constitutional compliance.

        Returns:
            Dict with status for each database and overall status
        """
        timestamp = datetime.now()

        # Check each database (Neo4j via Graph Channel, others direct)
        neo4j_health = await self.check_neo4j_health_async()
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

    def check_all_databases(self) -> Dict[str, Any]:
        """
        Check health of all databases comprehensively.

        DEPRECATED: Synchronous version. Use check_all_databases_async() for Graph Channel compliance.

        Returns:
            Dict with status for each database and overall status
        """
        warnings.warn(
            "check_all_databases() synchronous method is deprecated. Use check_all_databases_async() instead.",
            DeprecationWarning,
            stacklevel=2
        )

        # If Graph Channel is available, use async version
        if GRAPH_CHANNEL_AVAILABLE:
            return asyncio.run(self.check_all_databases_async())

        # Fallback to legacy synchronous version
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

# Lazy initialize to avoid deprecation warnings on import
_database_health_service = None

def _get_database_health_service() -> DatabaseHealthService:
    """Get global database health service instance (lazy initialization)."""
    global _database_health_service
    if _database_health_service is None:
        _database_health_service = DatabaseHealthService(
            neo4j_uri=settings.NEO4J_URI,
            neo4j_user=settings.NEO4J_USER,
            neo4j_password=settings.NEO4J_PASSWORD
        )
    return _database_health_service

# Legacy global instance (deprecated, use _get_database_health_service() instead)
database_health_service = None


async def get_database_health_async() -> Dict[str, Any]:
    """Quick function to get comprehensive database health using Graph Channel.

    RECOMMENDED: Use this async version for constitutional compliance.
    """
    service = _get_database_health_service()
    return await service.check_all_databases_async()


def get_database_health() -> Dict[str, Any]:
    """Quick function to get comprehensive database health.

    DEPRECATED: Synchronous version. Use get_database_health_async() instead.
    """
    warnings.warn(
        "get_database_health() is deprecated. Use get_database_health_async() instead.",
        DeprecationWarning,
        stacklevel=2
    )

    if GRAPH_CHANNEL_AVAILABLE:
        return asyncio.run(get_database_health_async())

    service = _get_database_health_service()
    return service.check_all_databases()


async def is_database_healthy_async(database_name: str) -> bool:
    """Check if a specific database is healthy using Graph Channel.

    RECOMMENDED: Use this async version for constitutional compliance.
    """
    service = _get_database_health_service()

    if database_name == 'neo4j':
        health = await service.check_neo4j_health_async()
    elif database_name == 'redis':
        health = service.check_redis_health()
    elif database_name == 'qdrant':
        health = service.check_qdrant_health()
    else:
        return False

    return health['status'] == 'healthy'


def is_database_healthy(database_name: str) -> bool:
    """Check if a specific database is healthy.

    DEPRECATED: Synchronous version. Use is_database_healthy_async() instead.
    """
    warnings.warn(
        "is_database_healthy() is deprecated. Use is_database_healthy_async() instead.",
        DeprecationWarning,
        stacklevel=2
    )

    if database_name == 'neo4j' and GRAPH_CHANNEL_AVAILABLE:
        return asyncio.run(is_database_healthy_async(database_name))

    service = _get_database_health_service()

    if database_name == 'neo4j':
        health = service.check_neo4j_health()
    elif database_name == 'redis':
        health = service.check_redis_health()
    elif database_name == 'qdrant':
        health = service.check_qdrant_health()
    else:
        return False

    return health['status'] == 'healthy'


# Export both sync and async versions
__all__ = [
    'DatabaseHealth',
    'DatabaseHealthService',
    'get_database_health',  # deprecated
    'get_database_health_async',  # RECOMMENDED
    'is_database_healthy',  # deprecated
    'is_database_healthy_async'  # RECOMMENDED
]