"""
Flux Database Connections
Constitutional compliance: Neo4j as Single Source of Truth, hybrid storage architecture
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

# Database clients
from neo4j import AsyncGraphDatabase
from qdrant_client import AsyncQdrantClient
import aiosqlite
import redis.asyncio as aioredis

from .config import settings

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages all database connections with constitutional compliance"""
    
    def __init__(self):
        self.neo4j_driver: Optional[Any] = None
        self.qdrant_client: Optional[AsyncQdrantClient] = None
        self.redis_client: Optional[aioredis.Redis] = None
        self.sqlite_path: str = settings.sqlite_path
        self._connected = False
    
    async def connect_all(self) -> bool:
        """Connect to all databases with constitutional compliance"""
        try:
            # Connect to Neo4j (Knowledge Graph - SSoT)
            await self._connect_neo4j()
            
            # Connect to Qdrant (Vector Database)
            await self._connect_qdrant()
            
            # Connect to Redis (Real-time Streams)
            await self._connect_redis()
            
            # Initialize SQLite (Structured Data)
            await self._init_sqlite()
            
            self._connected = True
            logger.info("All database connections established successfully")
            return True
            
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            await self.disconnect_all()
            return False
    
    async def disconnect_all(self):
        """Disconnect from all databases"""
        if self.neo4j_driver:
            await self.neo4j_driver.close()
            self.neo4j_driver = None
        
        if self.qdrant_client:
            await self.qdrant_client.close()
            self.qdrant_client = None
        
        if self.redis_client:
            await self.redis_client.close()
            self.redis_client = None
        
        self._connected = False
        logger.info("All database connections closed")
    
    async def _connect_neo4j(self):
        """Connect to Neo4j knowledge graph"""
        try:
            self.neo4j_driver = AsyncGraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password)
            )
            
            # Test connection
            async with self.neo4j_driver.session() as session:
                await session.run("RETURN 1")
            
            logger.info(f"Connected to Neo4j at {settings.neo4j_uri}")
            
        except Exception as e:
            logger.error(f"Neo4j connection failed: {e}")
            raise
    
    async def _connect_qdrant(self):
        """Connect to Qdrant vector database"""
        try:
            self.qdrant_client = AsyncQdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port
            )
            
            # Test connection
            await self.qdrant_client.get_collections()
            
            logger.info(f"Connected to Qdrant at {settings.qdrant_host}:{settings.qdrant_port}")
            
        except Exception as e:
            logger.error(f"Qdrant connection failed: {e}")
            raise
    
    async def _connect_redis(self):
        """Connect to Redis for real-time streams"""
        try:
            self.redis_client = aioredis.from_url(
                f"redis://{settings.redis_host}:{settings.redis_port}/{settings.redis_db}",
                password=settings.redis_password,
                decode_responses=True
            )
            
            # Test connection
            await self.redis_client.ping()
            
            logger.info(f"Connected to Redis at {settings.redis_host}:{settings.redis_port}")
            
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            raise
    
    async def _init_sqlite(self):
        """Initialize SQLite database with schema"""
        try:
            async with aiosqlite.connect(self.sqlite_path) as db:
                # Create tables for structured data
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS user_profiles (
                        id TEXT PRIMARY KEY,
                        username TEXT UNIQUE NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        learning_preferences TEXT,
                        mock_data BOOLEAN DEFAULT 1
                    )
                """)
                
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS document_artifacts (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        filename TEXT NOT NULL,
                        file_size INTEGER,
                        content_hash TEXT,
                        processing_status TEXT DEFAULT 'pending',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        mock_data BOOLEAN DEFAULT 1,
                        FOREIGN KEY (user_id) REFERENCES user_profiles (id)
                    )
                """)
                
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS curiosity_missions (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        title TEXT NOT NULL,
                        description TEXT,
                        status TEXT DEFAULT 'active',
                        sources_found INTEGER DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        mock_data BOOLEAN DEFAULT 1,
                        FOREIGN KEY (user_id) REFERENCES user_profiles (id)
                    )
                """)
                
                await db.commit()
            
            logger.info(f"SQLite database initialized at {self.sqlite_path}")
            
        except Exception as e:
            logger.error(f"SQLite initialization failed: {e}")
            raise
    
    async def health_check(self) -> Dict[str, str]:
        """Check health of all database connections"""
        health_status = {}
        
        # Neo4j health check
        try:
            if self.neo4j_driver:
                async with self.neo4j_driver.session() as session:
                    await session.run("RETURN 1")
                health_status["neo4j"] = "healthy"
            else:
                health_status["neo4j"] = "not_connected"
        except Exception as e:
            health_status["neo4j"] = f"error: {str(e)}"
        
        # Qdrant health check
        try:
            if self.qdrant_client:
                await self.qdrant_client.get_collections()
                health_status["qdrant"] = "healthy"
            else:
                health_status["qdrant"] = "not_connected"
        except Exception as e:
            health_status["qdrant"] = f"error: {str(e)}"
        
        # Redis health check
        try:
            if self.redis_client:
                await self.redis_client.ping()
                health_status["redis"] = "healthy"
            else:
                health_status["redis"] = "not_connected"
        except Exception as e:
            health_status["redis"] = f"error: {str(e)}"
        
        # SQLite health check
        try:
            async with aiosqlite.connect(self.sqlite_path) as db:
                await db.execute("SELECT 1")
            health_status["sqlite"] = "healthy"
        except Exception as e:
            health_status["sqlite"] = f"error: {str(e)}"
        
        return health_status
    
    @property
    def is_connected(self) -> bool:
        """Check if all databases are connected"""
        return self._connected

# Global database manager instance
db_manager = DatabaseManager()

# Context managers for database operations
@asynccontextmanager
async def get_neo4j_session():
    """Get Neo4j session with proper cleanup"""
    if not db_manager.neo4j_driver:
        raise RuntimeError("Neo4j not connected")
    
    async with db_manager.neo4j_driver.session() as session:
        yield session

@asynccontextmanager
async def get_sqlite_connection():
    """Get SQLite connection with proper cleanup"""
    async with aiosqlite.connect(db_manager.sqlite_path) as db:
        yield db
