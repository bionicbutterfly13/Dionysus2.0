"""Neo4j configuration and schema management for ThoughtSeed pipeline.

Constitutional Compliance (Spec 040 M2):
- DEPRECATED: Direct driver access is deprecated. Use get_graph_channel() instead.
- All new code should use DaedalusGraphChannel for graph operations.
- This module maintains backwards compatibility during migration period.

Migration Guide:
    OLD (deprecated):
        from backend.src.config.neo4j_config import get_neo4j_driver
        driver = get_neo4j_driver()
        with driver.session() as session:
            result = session.run("MATCH (d:Document) RETURN d LIMIT 10")

    NEW (constitutional):
        from daedalus_gateway import get_graph_channel
        channel = get_graph_channel()
        await channel.connect()
        result = await channel.execute_read(
            "MATCH (d:Document) RETURN d LIMIT 10",
            caller_service="my_service",
            caller_function="my_function"
        )
"""

import asyncio
import warnings
from typing import Optional, Dict, Any, List
import logging
from .settings import settings

# CONSTITUTIONAL COMPLIANCE: Use Graph Channel instead of direct neo4j imports
try:
    from daedalus_gateway import get_graph_channel, GraphChannelConfig
    GRAPH_CHANNEL_AVAILABLE = True
except ImportError:
    GRAPH_CHANNEL_AVAILABLE = False
    warnings.warn(
        "daedalus_gateway not available. Graph operations will fail. "
        "Install with: pip install -e /path/to/daedalus-gateway",
        ImportWarning
    )

# DEPRECATED: Legacy neo4j import for backwards compatibility only
# This will be removed in a future release
try:
    from neo4j import GraphDatabase, Driver
    NEO4J_LEGACY_AVAILABLE = True
except ImportError:
    NEO4J_LEGACY_AVAILABLE = False
    Driver = None

logger = logging.getLogger(__name__)

class Neo4jConfig:
    """Neo4j configuration and connection management.

    DEPRECATED: This class is deprecated. Use DaedalusGraphChannel instead.
    Maintained for backwards compatibility during migration period.
    """

    def __init__(self, uri: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None):
        warnings.warn(
            "Neo4jConfig is deprecated. Use DaedalusGraphChannel from daedalus_gateway instead. "
            "See module docstring for migration guide.",
            DeprecationWarning,
            stacklevel=2
        )

        self.uri = uri or settings.NEO4J_URI
        self.user = user or settings.NEO4J_USER
        self.password = password or settings.NEO4J_PASSWORD
        self._driver: Optional[Driver] = None
        self._graph_channel = None

    @property
    def driver(self) -> Driver:
        """Get Neo4j driver connection.

        DEPRECATED: Direct driver access is deprecated. Use get_graph_channel() instead.

        This property is maintained for backwards compatibility only and will be
        removed in a future release.
        """
        warnings.warn(
            "Direct driver access is deprecated. Use get_graph_channel() instead.",
            DeprecationWarning,
            stacklevel=2
        )

        if self._driver is None:
            if not NEO4J_LEGACY_AVAILABLE:
                raise RuntimeError(
                    "neo4j driver not available and direct driver access is deprecated. "
                    "Use get_graph_channel() from daedalus_gateway instead."
                )

            try:
                self._driver = GraphDatabase.driver(
                    self.uri,
                    auth=(self.user, self.password)
                )
                # Test connection only if not in test mode
                import os
                if not os.getenv("PYTEST_CURRENT_TEST"):
                    with self._driver.session() as session:
                        session.run("RETURN 1")
                logger.info("Neo4j connection established (LEGACY - consider migrating to Graph Channel)")
            except Exception as e:
                logger.error(f"Neo4j connection failed: {e}")
                # Don't raise in test mode - allow graceful degradation
                import os
                if not os.getenv("PYTEST_CURRENT_TEST"):
                    raise
        return self._driver

    def get_graph_channel(self):
        """Get Graph Channel instance (RECOMMENDED).

        This is the constitutional way to access the graph database.
        Returns a DaedalusGraphChannel instance configured with this config's settings.
        """
        if not GRAPH_CHANNEL_AVAILABLE:
            raise RuntimeError(
                "daedalus_gateway not available. Install with: pip install -e /path/to/daedalus-gateway"
            )

        if self._graph_channel is None:
            config = GraphChannelConfig(
                neo4j_uri=self.uri,
                neo4j_user=self.user,
                neo4j_password=self.password
            )
            self._graph_channel = get_graph_channel(config)

        return self._graph_channel

    async def create_schema_async(self) -> None:
        """Create the complete Neo4j schema for ThoughtSeed pipeline using Graph Channel.

        RECOMMENDED: Use this async version for constitutional compliance.
        """
        if not GRAPH_CHANNEL_AVAILABLE:
            raise RuntimeError("Graph Channel not available. Cannot create schema.")

        channel = self.get_graph_channel()
        await channel.connect()

        constraints_and_indexes = [
            # Unique constraints
            "CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT batch_id_unique IF NOT EXISTS FOR (b:ProcessingBatch) REQUIRE b.batch_id IS UNIQUE",
            "CREATE CONSTRAINT thoughtseed_id_unique IF NOT EXISTS FOR (t:ThoughtSeed) REQUIRE t.thoughtseed_id IS UNIQUE",
            "CREATE CONSTRAINT basin_id_unique IF NOT EXISTS FOR (a:AttractorBasin) REQUIRE a.basin_id IS UNIQUE",
            "CREATE CONSTRAINT field_id_unique IF NOT EXISTS FOR (f:NeuralField) REQUIRE f.field_id IS UNIQUE",

            # Vector indexes for 384-dimensional embeddings
            "CREATE VECTOR INDEX document_embedding_index IF NOT EXISTS FOR (d:Document) ON (d.embedding) OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}",
            "CREATE VECTOR INDEX basin_center_index IF NOT EXISTS FOR (a:AttractorBasin) ON (a.center_vector) OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}",

            # Full-text search indexes
            "CREATE FULLTEXT INDEX document_content_index IF NOT EXISTS FOR (d:Document) ON EACH [d.extracted_text, d.filename]",
            "CREATE FULLTEXT INDEX knowledge_search_index IF NOT EXISTS FOR (k:KnowledgeTriple) ON EACH [k.subject, k.predicate, k.object]",

            # Performance indexes
            "CREATE INDEX document_status_index IF NOT EXISTS FOR (d:Document) ON (d.processing_status)",
            "CREATE INDEX batch_status_index IF NOT EXISTS FOR (b:ProcessingBatch) ON (b.status)",
            "CREATE INDEX thoughtseed_type_index IF NOT EXISTS FOR (t:ThoughtSeed) ON (t.type)",
            "CREATE INDEX consciousness_level_index IF NOT EXISTS FOR (c:ConsciousnessState) ON (c.consciousness_level)",
            "CREATE INDEX memory_type_index IF NOT EXISTS FOR (m:MemoryFormation) ON (m.memory_type)",
            "CREATE INDEX upload_timestamp_index IF NOT EXISTS FOR (d:Document) ON (d.upload_timestamp)",

            # CLAUSE Phase 1 basin strengthening indexes (Spec 034 T026)
            "CREATE INDEX basin_strength_index IF NOT EXISTS FOR (a:AttractorBasin) ON (a.strength)",
            "CREATE INDEX basin_activation_count_index IF NOT EXISTS FOR (a:AttractorBasin) ON (a.activation_count)",
        ]

        for query in constraints_and_indexes:
            try:
                result = await channel.execute_schema(
                    operation=query,
                    caller_service="neo4j_config"
                )
                if result.get('success'):
                    logger.info(f"Schema operation executed: {query[:50]}...")
                else:
                    logger.debug(f"Schema operation note: {result.get('error', 'may already exist')}")
            except Exception as e:
                logger.warning(f"Schema query failed (may already exist): {e}")

    def create_schema(self) -> None:
        """Create the complete Neo4j schema for ThoughtSeed pipeline.

        DEPRECATED: Synchronous version. Use create_schema_async() for Graph Channel compliance.
        This method is maintained for backwards compatibility only.
        """
        warnings.warn(
            "create_schema() synchronous method is deprecated. Use create_schema_async() instead.",
            DeprecationWarning,
            stacklevel=2
        )

        # If Graph Channel is available, run async version
        if GRAPH_CHANNEL_AVAILABLE:
            logger.info("Running create_schema via Graph Channel (async)")
            asyncio.run(self.create_schema_async())
            return

        # Fallback to legacy direct driver access
        if not NEO4J_LEGACY_AVAILABLE:
            raise RuntimeError("Neither Graph Channel nor legacy neo4j driver available")

        constraints_and_indexes = [
            # Unique constraints
            "CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT batch_id_unique IF NOT EXISTS FOR (b:ProcessingBatch) REQUIRE b.batch_id IS UNIQUE",
            "CREATE CONSTRAINT thoughtseed_id_unique IF NOT EXISTS FOR (t:ThoughtSeed) REQUIRE t.thoughtseed_id IS UNIQUE",
            "CREATE CONSTRAINT basin_id_unique IF NOT EXISTS FOR (a:AttractorBasin) REQUIRE a.basin_id IS UNIQUE",
            "CREATE CONSTRAINT field_id_unique IF NOT EXISTS FOR (f:NeuralField) REQUIRE f.field_id IS UNIQUE",

            # Vector indexes for 384-dimensional embeddings
            "CREATE VECTOR INDEX document_embedding_index IF NOT EXISTS FOR (d:Document) ON (d.embedding) OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}",
            "CREATE VECTOR INDEX basin_center_index IF NOT EXISTS FOR (a:AttractorBasin) ON (a.center_vector) OPTIONS {indexConfig: {`vector.dimensions`: 384, `vector.similarity_function`: 'cosine'}}",

            # Full-text search indexes
            "CREATE FULLTEXT INDEX document_content_index IF NOT EXISTS FOR (d:Document) ON EACH [d.extracted_text, d.filename]",
            "CREATE FULLTEXT INDEX knowledge_search_index IF NOT EXISTS FOR (k:KnowledgeTriple) ON EACH [k.subject, k.predicate, k.object]",

            # Performance indexes
            "CREATE INDEX document_status_index IF NOT EXISTS FOR (d:Document) ON (d.processing_status)",
            "CREATE INDEX batch_status_index IF NOT EXISTS FOR (b:ProcessingBatch) ON (b.status)",
            "CREATE INDEX thoughtseed_type_index IF NOT EXISTS FOR (t:ThoughtSeed) ON (t.type)",
            "CREATE INDEX consciousness_level_index IF NOT EXISTS FOR (c:ConsciousnessState) ON (c.consciousness_level)",
            "CREATE INDEX memory_type_index IF NOT EXISTS FOR (m:MemoryFormation) ON (m.memory_type)",
            "CREATE INDEX upload_timestamp_index IF NOT EXISTS FOR (d:Document) ON (d.upload_timestamp)",

            # CLAUSE Phase 1 basin strengthening indexes (Spec 034 T026)
            "CREATE INDEX basin_strength_index IF NOT EXISTS FOR (a:AttractorBasin) ON (a.strength)",
            "CREATE INDEX basin_activation_count_index IF NOT EXISTS FOR (a:AttractorBasin) ON (a.activation_count)",
        ]

        with self.driver.session() as session:
            for query in constraints_and_indexes:
                try:
                    session.run(query)
                    logger.info(f"Schema query executed (LEGACY): {query[:50]}...")
                except Exception as e:
                    logger.warning(f"Schema query failed (may already exist): {e}")

    async def verify_schema_async(self) -> Dict[str, Any]:
        """Verify the schema is correctly created using Graph Channel.

        RECOMMENDED: Use this async version for constitutional compliance.
        """
        if not GRAPH_CHANNEL_AVAILABLE:
            raise RuntimeError("Graph Channel not available. Cannot verify schema.")

        channel = self.get_graph_channel()
        await channel.connect()

        # Check constraints
        constraints_result = await channel.execute_read(
            "SHOW CONSTRAINTS",
            caller_service="neo4j_config",
            caller_function="verify_schema_async"
        )
        constraints = [record.get("name") for record in constraints_result.get("records", [])]

        # Check indexes
        indexes_result = await channel.execute_read(
            "SHOW INDEXES",
            caller_service="neo4j_config",
            caller_function="verify_schema_async"
        )
        indexes = [record.get("name") for record in indexes_result.get("records", [])]

        # Check node labels
        labels_result = await channel.execute_read(
            "CALL db.labels()",
            caller_service="neo4j_config",
            caller_function="verify_schema_async"
        )
        labels = [record.get("label") for record in labels_result.get("records", [])]

        return {
            "constraints": constraints,
            "indexes": indexes,
            "labels": labels,
            "schema_ready": len(constraints) >= 5 and len(indexes) >= 8
        }

    def verify_schema(self) -> Dict[str, Any]:
        """Verify the schema is correctly created.

        DEPRECATED: Synchronous version. Use verify_schema_async() for Graph Channel compliance.
        """
        warnings.warn(
            "verify_schema() synchronous method is deprecated. Use verify_schema_async() instead.",
            DeprecationWarning,
            stacklevel=2
        )

        # If Graph Channel is available, run async version
        if GRAPH_CHANNEL_AVAILABLE:
            return asyncio.run(self.verify_schema_async())

        # Fallback to legacy direct driver access
        if not NEO4J_LEGACY_AVAILABLE:
            raise RuntimeError("Neither Graph Channel nor legacy neo4j driver available")

        with self.driver.session() as session:
            # Check constraints
            constraints_result = session.run("SHOW CONSTRAINTS")
            constraints = [record["name"] for record in constraints_result]

            # Check indexes
            indexes_result = session.run("SHOW INDEXES")
            indexes = [record["name"] for record in indexes_result]

            # Check node labels
            labels_result = session.run("CALL db.labels()")
            labels = [record["label"] for record in labels_result]

            return {
                "constraints": constraints,
                "indexes": indexes,
                "labels": labels,
                "schema_ready": len(constraints) >= 5 and len(indexes) >= 8
            }

    def create_sample_data(self) -> None:
        """Create sample data for testing (optional)."""
        sample_queries = [
            """
            MERGE (d:Document {
                id: 'sample-doc-1',
                filename: 'sample_research_paper.pdf',
                content_type: 'application/pdf',
                file_size: 1024000,
                upload_timestamp: datetime(),
                processing_status: 'COMPLETED',
                extracted_text: 'This is a sample research paper about consciousness and AI.',
                batch_id: 'sample-batch-1'
            })
            """,
            """
            MERGE (b:ProcessingBatch {
                batch_id: 'sample-batch-1',
                document_count: 1,
                total_size_bytes: 1024000,
                status: 'COMPLETED',
                created_timestamp: datetime(),
                progress_percentage: 100.0
            })
            """,
            """
            MERGE (t:ThoughtSeed {
                thoughtseed_id: 'sample-thoughtseed-1',
                document_id: 'sample-doc-1',
                type: 'CONCEPTUAL',
                layer: 3,
                activation_level: 0.75,
                consciousness_score: 0.65,
                created_timestamp: datetime()
            })
            """,
            """
            MERGE (a:AttractorBasin {
                basin_id: 'sample-basin-1',
                center_concept: 'consciousness_research',
                strength: 0.8,
                radius: 0.5,
                influence_type: 'EMERGENCE',
                formation_timestamp: datetime()
            })
            """
        ]

        with self.driver.session() as session:
            for query in sample_queries:
                try:
                    session.run(query)
                    logger.info("Sample data created")
                except Exception as e:
                    logger.error(f"Sample data creation failed: {e}")

    async def close_async(self) -> None:
        """Close the Graph Channel connection.

        RECOMMENDED: Use this async version for constitutional compliance.
        """
        if self._graph_channel:
            await self._graph_channel.disconnect()
            self._graph_channel = None

    def close(self) -> None:
        """Close the Neo4j driver connection.

        DEPRECATED: Use close_async() for Graph Channel compliance.
        """
        warnings.warn(
            "close() synchronous method is deprecated. Use close_async() instead.",
            DeprecationWarning,
            stacklevel=2
        )

        # Close Graph Channel if available
        if self._graph_channel:
            asyncio.run(self.close_async())

        # Close legacy driver if exists
        if self._driver:
            self._driver.close()
            self._driver = None

# Global Neo4j instance (DEPRECATED)
# Use get_graph_channel() from daedalus_gateway instead
neo4j_config = None

def _get_neo4j_config():
    """Get global Neo4jConfig instance (lazy initialization with deprecation warning)."""
    global neo4j_config
    if neo4j_config is None:
        neo4j_config = Neo4jConfig()
    return neo4j_config

def get_neo4j_driver() -> Driver:
    """Get the global Neo4j driver.

    DEPRECATED: Direct driver access is deprecated. Use get_graph_channel() instead.

    Migration:
        OLD: driver = get_neo4j_driver()
        NEW: channel = get_graph_channel()
    """
    warnings.warn(
        "get_neo4j_driver() is deprecated. Use get_graph_channel() from daedalus_gateway instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return _get_neo4j_config().driver

async def initialize_neo4j_schema_async() -> Dict[str, Any]:
    """Initialize the Neo4j schema and return verification results using Graph Channel.

    RECOMMENDED: Use this async version for constitutional compliance.
    """
    config = _get_neo4j_config()
    await config.create_schema_async()
    return await config.verify_schema_async()

def initialize_neo4j_schema() -> Dict[str, Any]:
    """Initialize the Neo4j schema and return verification results.

    DEPRECATED: Synchronous version. Use initialize_neo4j_schema_async() instead.
    """
    warnings.warn(
        "initialize_neo4j_schema() is deprecated. Use initialize_neo4j_schema_async() instead.",
        DeprecationWarning,
        stacklevel=2
    )

    if GRAPH_CHANNEL_AVAILABLE:
        return asyncio.run(initialize_neo4j_schema_async())

    # Fallback to legacy
    config = _get_neo4j_config()
    config.create_schema()
    return config.verify_schema()

# Re-export Graph Channel components for convenience
if GRAPH_CHANNEL_AVAILABLE:
    __all__ = [
        'Neo4jConfig',
        'get_neo4j_driver',  # deprecated
        'initialize_neo4j_schema',  # deprecated
        'initialize_neo4j_schema_async',
        'get_graph_channel',  # RECOMMENDED
        'GraphChannelConfig'
    ]
else:
    __all__ = [
        'Neo4jConfig',
        'get_neo4j_driver',
        'initialize_neo4j_schema'
    ]