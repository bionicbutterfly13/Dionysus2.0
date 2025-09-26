"""
Database Connection Management

Provides unified database connectivity supporting multiple
database backends with connection pooling and health monitoring.
"""

import asyncio
from typing import Dict, Optional, Any
from contextlib import asynccontextmanager

import aiosqlite
from ..config import get_migration_config
from ..logging_config import get_migration_logger


class DatabaseConnection:
    """Unified database connection manager"""

    def __init__(self):
        self.config = get_migration_config()
        self.logger = get_migration_logger()
        self._connection = None
        self._connection_pool = None
        self._health_status = "disconnected"

    async def initialize(self) -> None:
        """
        Initialize database connection

        Sets up connection to the configured database backend
        and performs initial health checks.
        """
        try:
            self.logger.info(
                "Initializing database connection",
                database_url=self.config.database_url
            )

            # For this implementation, using SQLite for simplicity
            # In production, would support PostgreSQL, Neo4j, etc.
            self._connection = await aiosqlite.connect(
                self.config.database_path,
                isolation_level=None  # Autocommit mode
            )

            # Enable foreign keys
            await self._connection.execute("PRAGMA foreign_keys = ON")

            # Create initial schema if not exists
            await self._create_schema()

            self._health_status = "connected"

            self.logger.info("Database connection established successfully")

        except Exception as e:
            self.logger.error(
                "Failed to initialize database connection",
                error=str(e)
            )
            self._health_status = "error"
            raise

    async def _create_schema(self) -> None:
        """Create database schema if it doesn't exist"""
        schema_sql = """
        -- Legacy Components table
        CREATE TABLE IF NOT EXISTS legacy_components (
            component_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            consciousness_awareness_score REAL NOT NULL,
            consciousness_inference_score REAL NOT NULL,
            consciousness_memory_score REAL NOT NULL,
            consciousness_composite_score REAL NOT NULL,
            strategic_uniqueness_score REAL NOT NULL,
            strategic_reusability_score REAL NOT NULL,
            strategic_framework_alignment_score REAL NOT NULL,
            strategic_composite_score REAL NOT NULL,
            quality_score REAL NOT NULL,
            analysis_status TEXT NOT NULL,
            source_code_hash TEXT,
            file_size_bytes INTEGER,
            consciousness_patterns TEXT, -- JSON array
            dependencies TEXT, -- JSON array
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        -- Migration Tasks table
        CREATE TABLE IF NOT EXISTS migration_tasks (
            task_id TEXT PRIMARY KEY,
            task_type TEXT NOT NULL,
            component_id TEXT NOT NULL,
            pipeline_id TEXT NOT NULL,
            agent_id TEXT,
            task_status TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            started_at DATETIME,
            completed_at DATETIME,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            errors TEXT, -- JSON array
            FOREIGN KEY (component_id) REFERENCES legacy_components (component_id)
        );

        -- Quality Assessments table
        CREATE TABLE IF NOT EXISTS quality_assessments (
            assessment_id TEXT PRIMARY KEY,
            component_id TEXT NOT NULL,
            consciousness_impact TEXT, -- JSON object
            strategic_value TEXT, -- JSON object
            composite_score REAL NOT NULL,
            assessment_method TEXT NOT NULL,
            assessor_agent_id TEXT NOT NULL,
            migration_recommended BOOLEAN NOT NULL,
            enhancement_opportunities TEXT, -- JSON array
            risk_factors TEXT, -- JSON array
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (component_id) REFERENCES legacy_components (component_id)
        );

        -- Enhancement Results table
        CREATE TABLE IF NOT EXISTS enhancement_results (
            enhancement_id TEXT PRIMARY KEY,
            component_id TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            status TEXT NOT NULL,
            original_component TEXT, -- JSON object
            enhanced_component TEXT, -- JSON object
            consciousness_improvements TEXT, -- JSON object
            validation_metrics TEXT, -- JSON object
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            completed_at DATETIME,
            errors TEXT, -- JSON array
            FOREIGN KEY (component_id) REFERENCES legacy_components (component_id)
        );

        -- Rollback Checkpoints table
        CREATE TABLE IF NOT EXISTS rollback_checkpoints (
            checkpoint_id TEXT PRIMARY KEY,
            component_id TEXT NOT NULL,
            migration_state TEXT, -- JSON object
            file_backups TEXT, -- JSON object
            metadata_backup TEXT,
            database_backup TEXT, -- JSON object
            status TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            retention_until DATETIME NOT NULL,
            FOREIGN KEY (component_id) REFERENCES legacy_components (component_id)
        );

        -- Background Agents table
        CREATE TABLE IF NOT EXISTS background_agents (
            agent_id TEXT PRIMARY KEY,
            context_window_id TEXT NOT NULL,
            agent_status TEXT NOT NULL,
            current_task_id TEXT,
            assigned_component_id TEXT,
            task_history TEXT, -- JSON array
            performance_stats TEXT, -- JSON object
            context_isolation BOOLEAN NOT NULL,
            last_activity DATETIME DEFAULT CURRENT_TIMESTAMP,
            coordinator_id TEXT NOT NULL
        );

        -- DAEDALUS Coordinations table
        CREATE TABLE IF NOT EXISTS daedalus_coordinations (
            coordination_id TEXT PRIMARY KEY,
            coordinator_status TEXT NOT NULL,
            active_subagents TEXT, -- JSON array
            task_queue TEXT, -- JSON array
            completed_tasks TEXT, -- JSON array
            failed_tasks TEXT, -- JSON array
            performance_metrics TEXT, -- JSON object
            learning_state TEXT, -- JSON object
            last_optimization DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        -- Migration Pipelines table
        CREATE TABLE IF NOT EXISTS migration_pipelines (
            pipeline_id TEXT PRIMARY KEY,
            codebase_path TEXT NOT NULL,
            coordinator_id TEXT NOT NULL,
            pipeline_status TEXT NOT NULL,
            discovered_components INTEGER DEFAULT 0,
            migration_candidates INTEGER DEFAULT 0,
            created_tasks INTEGER DEFAULT 0,
            options TEXT, -- JSON object
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            completed_at DATETIME,
            errors TEXT -- JSON array
        );

        -- Create indexes for better performance
        CREATE INDEX IF NOT EXISTS idx_components_quality_score
            ON legacy_components (quality_score);
        CREATE INDEX IF NOT EXISTS idx_components_consciousness_score
            ON legacy_components (consciousness_composite_score);
        CREATE INDEX IF NOT EXISTS idx_tasks_status
            ON migration_tasks (task_status);
        CREATE INDEX IF NOT EXISTS idx_tasks_component
            ON migration_tasks (component_id);
        CREATE INDEX IF NOT EXISTS idx_assessments_component
            ON quality_assessments (component_id);
        CREATE INDEX IF NOT EXISTS idx_checkpoints_component
            ON rollback_checkpoints (component_id);
        CREATE INDEX IF NOT EXISTS idx_agents_coordinator
            ON background_agents (coordinator_id);
        CREATE INDEX IF NOT EXISTS idx_agents_status
            ON background_agents (agent_status);
        """

        # Execute schema creation
        await self._connection.executescript(schema_sql)
        await self._connection.commit()

        self.logger.info("Database schema created/verified successfully")

    @asynccontextmanager
    async def transaction(self):
        """
        Context manager for database transactions

        Provides automatic transaction management with rollback on error.
        """
        if not self._connection:
            raise RuntimeError("Database not initialized")

        try:
            await self._connection.execute("BEGIN")
            yield self._connection
            await self._connection.commit()
        except Exception as e:
            await self._connection.rollback()
            self.logger.error(
                "Transaction rolled back due to error",
                error=str(e)
            )
            raise

    async def execute(self, query: str, parameters: Optional[tuple] = None) -> Any:
        """
        Execute a database query

        Args:
            query: SQL query string
            parameters: Optional query parameters

        Returns:
            Query result
        """
        if not self._connection:
            raise RuntimeError("Database not initialized")

        try:
            if parameters:
                cursor = await self._connection.execute(query, parameters)
            else:
                cursor = await self._connection.execute(query)

            return cursor

        except Exception as e:
            self.logger.error(
                "Database query execution failed",
                query=query[:100],  # Log first 100 chars
                error=str(e)
            )
            raise

    async def fetchone(self, query: str, parameters: Optional[tuple] = None) -> Optional[Dict]:
        """
        Fetch one row from query result

        Args:
            query: SQL query string
            parameters: Optional query parameters

        Returns:
            Single row as dictionary or None
        """
        cursor = await self.execute(query, parameters)
        row = await cursor.fetchone()

        if row:
            # Convert row to dictionary
            columns = [description[0] for description in cursor.description]
            return dict(zip(columns, row))

        return None

    async def fetchall(self, query: str, parameters: Optional[tuple] = None) -> list[Dict]:
        """
        Fetch all rows from query result

        Args:
            query: SQL query string
            parameters: Optional query parameters

        Returns:
            List of rows as dictionaries
        """
        cursor = await self.execute(query, parameters)
        rows = await cursor.fetchall()

        if rows:
            # Convert rows to dictionaries
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in rows]

        return []

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform database health check

        Returns:
            Health status and metrics
        """
        try:
            start_time = asyncio.get_event_loop().time()

            # Simple connectivity test
            await self.execute("SELECT 1")

            response_time = (asyncio.get_event_loop().time() - start_time) * 1000

            # Get database statistics
            stats = await self._get_database_stats()

            self._health_status = "healthy"

            return {
                "status": "healthy",
                "response_time_ms": response_time,
                "connection_status": self._health_status,
                "statistics": stats
            }

        except Exception as e:
            self._health_status = "error"
            return {
                "status": "error",
                "error": str(e),
                "connection_status": self._health_status
            }

    async def _get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            # Table row counts
            tables = [
                "legacy_components",
                "migration_tasks",
                "quality_assessments",
                "enhancement_results",
                "rollback_checkpoints",
                "background_agents",
                "daedalus_coordinations",
                "migration_pipelines"
            ]

            stats = {}
            for table in tables:
                cursor = await self.execute(f"SELECT COUNT(*) FROM {table}")
                count = (await cursor.fetchone())[0]
                stats[f"{table}_count"] = count

            # Database size (SQLite specific)
            cursor = await self.execute("PRAGMA page_count")
            page_count = (await cursor.fetchone())[0]
            cursor = await self.execute("PRAGMA page_size")
            page_size = (await cursor.fetchone())[0]
            stats["database_size_bytes"] = page_count * page_size

            return stats

        except Exception as e:
            self.logger.warning(
                "Failed to get database statistics",
                error=str(e)
            )
            return {}

    async def close(self) -> None:
        """Close database connection"""
        if self._connection:
            await self._connection.close()
            self._connection = None
            self._health_status = "disconnected"

            self.logger.info("Database connection closed")

    @property
    def is_connected(self) -> bool:
        """Check if database is connected"""
        return self._connection is not None and self._health_status == "healthy"

    @property
    def health_status(self) -> str:
        """Get current health status"""
        return self._health_status