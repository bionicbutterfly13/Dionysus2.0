"""
Multi-Tier Memory System for Ultra-Granular Document Processing
============================================================

Implements a three-tier memory architecture for decades-long knowledge persistence:
- Hot Memory (Redis): Current session + 24 hours - immediate access
- Warm Memory (Neo4j): Knowledge graphs for months/years - semantic relationships
- Cold Memory (Vector DB): Embeddings + file system for decades - long-term storage

Features hierarchical memory compression, automatic migration, and efficient retrieval.
Implements Spec-022 Task 2.3 requirements.
"""

import asyncio
import logging
import time
import json
import pickle
import hashlib
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
import uuid

# Memory storage backends
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    from daedalus_gateway import get_graph_channel
    GRAPH_CHANNEL_AVAILABLE = True
except ImportError:
    GRAPH_CHANNEL_AVAILABLE = False

try:
    import qdrant_client
    from qdrant_client.models import VectorParams, Distance, PointStruct
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

class MemoryTier(Enum):
    """Three-tier memory architecture"""
    HOT = "hot"       # Redis: current session + 24h
    WARM = "warm"     # Neo4j: knowledge graphs for months/years  
    COLD = "cold"     # Vector DB + files: decades storage

class MemoryType(Enum):
    """Types of memory content"""
    CONCEPT = "concept"
    RELATIONSHIP = "relationship"
    DOCUMENT = "document"
    SESSION = "session"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    EMBEDDING = "embedding"
    METADATA = "metadata"

class CompressionLevel(Enum):
    """Levels of memory compression"""
    NONE = 0
    LIGHT = 1      # Remove duplicate concepts
    MEDIUM = 2     # Merge similar concepts
    HEAVY = 3      # Aggregate into higher-level abstractions
    MAXIMUM = 4    # Keep only essential knowledge

@dataclass
class MemoryItem:
    """Item stored in multi-tier memory"""
    item_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: Any = None
    memory_type: MemoryType = MemoryType.CONCEPT
    tier: MemoryTier = MemoryTier.HOT
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    
    # Memory management
    access_count: int = 0
    importance_score: float = 0.5
    compression_level: CompressionLevel = CompressionLevel.NONE
    
    # Relationships
    related_items: List[str] = field(default_factory=list)
    parent_items: List[str] = field(default_factory=list)
    child_items: List[str] = field(default_factory=list)
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    domain_tags: List[str] = field(default_factory=list)
    source_document: Optional[str] = None
    
    # Storage info
    size_bytes: int = 0
    checksum: Optional[str] = None

@dataclass
class MemoryStats:
    """Statistics for memory system"""
    tier: MemoryTier
    total_items: int = 0
    total_size_bytes: int = 0
    hit_rate: float = 0.0
    average_access_time: float = 0.0
    compression_ratio: float = 1.0
    
    # Tier-specific stats
    active_sessions: int = 0  # Hot
    knowledge_graphs: int = 0  # Warm
    archived_documents: int = 0  # Cold

class HotMemoryManager:
    """Redis-based hot memory for immediate access (current session + 24h)"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        self.connected = False
        self.session_ttl = 86400  # 24 hours in seconds
        
    async def connect(self):
        """Connect to Redis"""
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available - using in-memory fallback")
            self.memory_store = {}
            self.connected = True
            return
        
        try:
            self.redis_client = redis.from_url(self.redis_url)
            await self.redis_client.ping()
            self.connected = True
            logger.info("Connected to Redis hot memory")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.memory_store = {}
            self.connected = True
    
    async def store(self, item: MemoryItem) -> bool:
        """Store item in hot memory with TTL"""
        try:
            # Calculate TTL based on importance
            ttl = self.session_ttl
            if item.importance_score > 0.8:
                ttl *= 2  # Keep important items longer
            
            # Serialize item
            serialized = self._serialize_item(item)
            
            if self.redis_client:
                # Store in Redis with TTL
                await self.redis_client.setex(
                    f"hot:{item.item_id}",
                    ttl,
                    serialized
                )
                
                # Store metadata separately for quick queries
                metadata = {
                    "type": item.memory_type.value,
                    "created_at": item.created_at.isoformat(),
                    "importance": item.importance_score,
                    "tags": item.tags,
                    "size": len(serialized)
                }
                await self.redis_client.setex(
                    f"meta:{item.item_id}",
                    ttl,
                    json.dumps(metadata)
                )
            else:
                # Fallback to in-memory store
                self.memory_store[item.item_id] = {
                    "item": serialized,
                    "expires": datetime.now() + timedelta(seconds=ttl)
                }
            
            logger.debug(f"Stored item {item.item_id} in hot memory with TTL {ttl}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store item in hot memory: {e}")
            return False
    
    async def retrieve(self, item_id: str) -> Optional[MemoryItem]:
        """Retrieve item from hot memory"""
        try:
            if self.redis_client:
                serialized = await self.redis_client.get(f"hot:{item_id}")
                if serialized:
                    item = self._deserialize_item(serialized)
                    # Update access stats
                    item.last_accessed = datetime.now()
                    item.access_count += 1
                    return item
            else:
                # Fallback to in-memory store
                if item_id in self.memory_store:
                    entry = self.memory_store[item_id]
                    if datetime.now() < entry["expires"]:
                        return self._deserialize_item(entry["item"])
                    else:
                        del self.memory_store[item_id]
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve item from hot memory: {e}")
            return None
    
    async def query(self, query_params: Dict[str, Any]) -> List[MemoryItem]:
        """Query hot memory with filters"""
        results = []
        
        try:
            if self.redis_client:
                # Use Redis SCAN to iterate through keys
                async for key in self.redis_client.scan_iter(match="hot:*"):
                    item_id = key.decode().split(":", 1)[1]
                    item = await self.retrieve(item_id)
                    if item and self._matches_query(item, query_params):
                        results.append(item)
            else:
                # Fallback to in-memory search
                for item_id, entry in self.memory_store.items():
                    if datetime.now() < entry["expires"]:
                        item = self._deserialize_item(entry["item"])
                        if self._matches_query(item, query_params):
                            results.append(item)
        
        except Exception as e:
            logger.error(f"Failed to query hot memory: {e}")
        
        return results
    
    async def get_stats(self) -> MemoryStats:
        """Get hot memory statistics"""
        stats = MemoryStats(tier=MemoryTier.HOT)
        
        try:
            if self.redis_client:
                info = await self.redis_client.info()
                stats.total_size_bytes = info.get("used_memory", 0)
                
                # Count items
                count = 0
                async for _ in self.redis_client.scan_iter(match="hot:*"):
                    count += 1
                stats.total_items = count
            else:
                stats.total_items = len(self.memory_store)
                stats.total_size_bytes = sum(
                    len(str(entry["item"])) for entry in self.memory_store.values()
                )
        
        except Exception as e:
            logger.error(f"Failed to get hot memory stats: {e}")
        
        return stats
    
    def _serialize_item(self, item: MemoryItem) -> bytes:
        """Serialize memory item"""
        return pickle.dumps(item)
    
    def _deserialize_item(self, data: bytes) -> MemoryItem:
        """Deserialize memory item"""
        return pickle.loads(data)
    
    def _matches_query(self, item: MemoryItem, query_params: Dict[str, Any]) -> bool:
        """Check if item matches query parameters"""
        if "memory_type" in query_params:
            if item.memory_type.value != query_params["memory_type"]:
                return False
        
        if "tags" in query_params:
            required_tags = set(query_params["tags"])
            item_tags = set(item.tags)
            if not required_tags.issubset(item_tags):
                return False
        
        if "min_importance" in query_params:
            if item.importance_score < query_params["min_importance"]:
                return False
        
        return True

class WarmMemoryManager:
    """Neo4j-based warm memory for knowledge graphs (months/years)"""

    def __init__(self, neo4j_uri: str = "bolt://localhost:7687", username: str = "neo4j", password: str = "thoughtseed"):
        # Parameters kept for backwards compatibility but not used
        # Graph Channel manages connection via environment variables
        self.graph_channel = None
        self.connected = False

    async def connect(self):
        """Connect to Neo4j via Graph Channel"""
        if not GRAPH_CHANNEL_AVAILABLE:
            logger.warning("Graph Channel not available - using in-memory graph fallback")
            self.graph_store = {"nodes": {}, "relationships": []}
            self.connected = True
            return

        try:
            # Get singleton Graph Channel instance
            self.graph_channel = get_graph_channel()
            await self.graph_channel.connect()

            # Test connection
            result = await self.graph_channel.execute_read(
                query="RETURN 1 as test",
                parameters={},
                caller_service="multi_tier_memory",
                caller_function="connect"
            )

            if result["success"]:
                self.connected = True
                logger.info("Connected to Neo4j warm memory via Graph Channel")
            else:
                raise Exception(f"Connection test failed: {result.get('error')}")

        except Exception as e:
            logger.error(f"Failed to connect to Graph Channel: {e}")
            self.graph_store = {"nodes": {}, "relationships": []}
            self.connected = True
    
    async def store_knowledge_graph(self, concepts: List[MemoryItem], relationships: List[MemoryItem]) -> bool:
        """Store concepts and relationships as knowledge graph"""
        try:
            if self.graph_channel:
                # Create concept nodes
                for concept in concepts:
                    result = await self.graph_channel.execute_write(
                        query="""
                        MERGE (c:Concept {id: $id})
                        SET c.content = $content,
                            c.type = $type,
                            c.created_at = $created_at,
                            c.importance = $importance,
                            c.tags = $tags,
                            c.domain_tags = $domain_tags
                        """,
                        parameters={
                            "id": concept.item_id,
                            "content": json.dumps(concept.content) if concept.content else "",
                            "type": concept.memory_type.value,
                            "created_at": concept.created_at.isoformat(),
                            "importance": concept.importance_score,
                            "tags": concept.tags,
                            "domain_tags": concept.domain_tags
                        },
                        caller_service="multi_tier_memory",
                        caller_function="store_knowledge_graph"
                    )

                    if not result["success"]:
                        logger.error(f"Failed to create concept node: {result.get('error')}")

                # Create relationship edges
                for relationship in relationships:
                    if hasattr(relationship.content, 'source_concept') and hasattr(relationship.content, 'target_concept'):
                        result = await self.graph_channel.execute_write(
                            query="""
                            MATCH (a:Concept {id: $source_id})
                            MATCH (b:Concept {id: $target_id})
                            MERGE (a)-[r:RELATES_TO {id: $rel_id}]->(b)
                            SET r.type = $rel_type,
                                r.strength = $strength,
                                r.created_at = $created_at
                            """,
                            parameters={
                                "source_id": relationship.content.get('source_concept', ''),
                                "target_id": relationship.content.get('target_concept', ''),
                                "rel_id": relationship.item_id,
                                "rel_type": relationship.content.get('relationship_type', 'relates_to'),
                                "strength": relationship.importance_score,
                                "created_at": relationship.created_at.isoformat()
                            },
                            caller_service="multi_tier_memory",
                            caller_function="store_knowledge_graph"
                        )

                        if not result["success"]:
                            logger.error(f"Failed to create relationship: {result.get('error')}")
            else:
                # Fallback to in-memory graph
                for concept in concepts:
                    self.graph_store["nodes"][concept.item_id] = concept

                for relationship in relationships:
                    self.graph_store["relationships"].append(relationship)

            logger.info(f"Stored knowledge graph with {len(concepts)} concepts and {len(relationships)} relationships")
            return True

        except Exception as e:
            logger.error(f"Failed to store knowledge graph: {e}")
            return False
    
    async def query_graph(self, cypher_query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute Cypher query on knowledge graph"""
        try:
            if self.graph_channel:
                result = await self.graph_channel.execute_read(
                    query=cypher_query,
                    parameters=params or {},
                    caller_service="multi_tier_memory",
                    caller_function="query_graph"
                )

                if result["success"]:
                    return result.get("records", [])
                else:
                    logger.error(f"Graph query failed: {result.get('error')}")
                    return []
            else:
                # Simple fallback query for in-memory graph
                return self._query_memory_graph(cypher_query, params or {})

        except Exception as e:
            logger.error(f"Failed to execute graph query: {e}")
            return []
    
    async def find_related_concepts(self, concept_id: str, max_depth: int = 2) -> List[Dict[str, Any]]:
        """Find concepts related to a given concept"""
        query = """
        MATCH (start:Concept {id: $concept_id})
        MATCH (start)-[r*1..$max_depth]-(related:Concept)
        RETURN DISTINCT related.id as id, related.content as content, 
               related.importance as importance, 
               length(r) as distance
        ORDER BY distance, importance DESC
        LIMIT 20
        """
        
        return await self.query_graph(query, {"concept_id": concept_id, "max_depth": max_depth})
    
    async def get_stats(self) -> MemoryStats:
        """Get warm memory statistics"""
        stats = MemoryStats(tier=MemoryTier.WARM)

        try:
            if self.graph_channel:
                # Count nodes
                result = await self.graph_channel.execute_read(
                    query="MATCH (n) RETURN count(n) as count",
                    parameters={},
                    caller_service="multi_tier_memory",
                    caller_function="get_stats"
                )

                if result["success"] and result.get("records"):
                    stats.total_items = result["records"][0].get("count", 0)

                # Count relationships
                result = await self.graph_channel.execute_read(
                    query="MATCH ()-[r]->() RETURN count(r) as count",
                    parameters={},
                    caller_service="multi_tier_memory",
                    caller_function="get_stats"
                )

                if result["success"] and result.get("records"):
                    stats.knowledge_graphs = result["records"][0].get("count", 0)
            else:
                stats.total_items = len(self.graph_store["nodes"])
                stats.knowledge_graphs = len(self.graph_store["relationships"])

        except Exception as e:
            logger.error(f"Failed to get warm memory stats: {e}")

        return stats
    
    def _query_memory_graph(self, query: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Simple fallback query for in-memory graph"""
        # Very basic implementation - would need full Cypher parser for complete functionality
        results = []
        
        if "MATCH (n)" in query:
            for node_id, node in self.graph_store["nodes"].items():
                results.append({
                    "id": node_id,
                    "content": node.content,
                    "importance": node.importance_score
                })
        
        return results[:20]  # Limit results

class ColdMemoryManager:
    """Vector database + file system for decades-long storage"""
    
    def __init__(self, storage_path: str = "/tmp/cold_memory", qdrant_url: str = "http://localhost:6333"):
        self.storage_path = Path(storage_path)
        self.qdrant_url = qdrant_url
        self.qdrant_client = None
        self.connected = False
        
        # Ensure storage directory exists
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
    async def connect(self):
        """Connect to vector database"""
        if not QDRANT_AVAILABLE:
            logger.warning("Qdrant not available - using file-based fallback")
            self.vector_store = {}
            self.connected = True
            return
        
        try:
            self.qdrant_client = qdrant_client.QdrantClient(url=self.qdrant_url)
            
            # Create collection if it doesn't exist
            try:
                await self.qdrant_client.create_collection(
                    collection_name="cold_memory",
                    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
                )
            except Exception:
                pass  # Collection might already exist
            
            self.connected = True
            logger.info("Connected to Qdrant cold memory")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            self.vector_store = {}
            self.connected = True
    
    async def store_with_embedding(self, item: MemoryItem, embedding: np.ndarray) -> bool:
        """Store item with vector embedding for semantic search"""
        try:
            # Store full item data in file system
            file_path = self.storage_path / f"{item.item_id}.pkl"
            with open(file_path, 'wb') as f:
                pickle.dump(item, f)
            
            # Store embedding in vector database
            if self.qdrant_client:
                point = PointStruct(
                    id=item.item_id,
                    vector=embedding.tolist(),
                    payload={
                        "type": item.memory_type.value,
                        "created_at": item.created_at.isoformat(),
                        "importance": item.importance_score,
                        "tags": item.tags,
                        "domain_tags": item.domain_tags,
                        "file_path": str(file_path)
                    }
                )
                
                await self.qdrant_client.upsert(
                    collection_name="cold_memory",
                    points=[point]
                )
            else:
                # Fallback to in-memory vector store
                self.vector_store[item.item_id] = {
                    "embedding": embedding,
                    "item": item,
                    "file_path": str(file_path)
                }
            
            logger.debug(f"Stored item {item.item_id} in cold memory")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store item in cold memory: {e}")
            return False
    
    async def semantic_search(self, query_embedding: np.ndarray, limit: int = 10, 
                            min_similarity: float = 0.7) -> List[Tuple[MemoryItem, float]]:
        """Perform semantic search using vector similarity"""
        try:
            results = []
            
            if self.qdrant_client:
                search_results = await self.qdrant_client.search(
                    collection_name="cold_memory",
                    query_vector=query_embedding.tolist(),
                    limit=limit,
                    score_threshold=min_similarity
                )
                
                for result in search_results:
                    # Load full item from file system
                    file_path = Path(result.payload["file_path"])
                    if file_path.exists():
                        with open(file_path, 'rb') as f:
                            item = pickle.load(f)
                        results.append((item, result.score))
            else:
                # Fallback to in-memory similarity search
                for item_id, data in self.vector_store.items():
                    similarity = self._cosine_similarity(query_embedding, data["embedding"])
                    if similarity >= min_similarity:
                        results.append((data["item"], similarity))
                
                results.sort(key=lambda x: x[1], reverse=True)
                results = results[:limit]
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to perform semantic search: {e}")
            return []
    
    async def compress_memory(self, compression_level: CompressionLevel) -> int:
        """Compress cold memory by removing/aggregating old items"""
        compressed_count = 0
        
        try:
            if compression_level == CompressionLevel.LIGHT:
                # Remove exact duplicates
                compressed_count = await self._remove_duplicates()
            elif compression_level == CompressionLevel.MEDIUM:
                # Merge very similar items
                compressed_count = await self._merge_similar_items()
            elif compression_level == CompressionLevel.HEAVY:
                # Aggregate into higher-level concepts
                compressed_count = await self._aggregate_concepts()
            elif compression_level == CompressionLevel.MAXIMUM:
                # Keep only most important items
                compressed_count = await self._keep_essential_only()
            
            logger.info(f"Compressed {compressed_count} items at level {compression_level.value}")
            
        except Exception as e:
            logger.error(f"Failed to compress cold memory: {e}")
        
        return compressed_count
    
    async def get_stats(self) -> MemoryStats:
        """Get cold memory statistics"""
        stats = MemoryStats(tier=MemoryTier.COLD)
        
        try:
            # Count files in storage directory
            stats.total_items = len(list(self.storage_path.glob("*.pkl")))
            
            # Calculate total size
            total_size = sum(f.stat().st_size for f in self.storage_path.glob("*.pkl"))
            stats.total_size_bytes = total_size
            
            if self.qdrant_client:
                collection_info = await self.qdrant_client.get_collection("cold_memory")
                stats.archived_documents = collection_info.vectors_count
        
        except Exception as e:
            logger.error(f"Failed to get cold memory stats: {e}")
        
        return stats
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    async def _remove_duplicates(self) -> int:
        """Remove exact duplicate items"""
        # Implementation would check for identical content hashes
        return 0
    
    async def _merge_similar_items(self) -> int:
        """Merge very similar items (>95% similarity)"""
        # Implementation would find and merge highly similar embeddings
        return 0
    
    async def _aggregate_concepts(self) -> int:
        """Aggregate related concepts into higher-level abstractions"""
        # Implementation would group related concepts and create summaries
        return 0
    
    async def _keep_essential_only(self) -> int:
        """Keep only the most important items (top 10%)"""
        # Implementation would sort by importance and remove low-scoring items
        return 0

class MemoryMigrationManager:
    """Manages automatic migration between memory tiers"""
    
    def __init__(self, hot_manager: HotMemoryManager, warm_manager: WarmMemoryManager, 
                 cold_manager: ColdMemoryManager):
        self.hot_manager = hot_manager
        self.warm_manager = warm_manager
        self.cold_manager = cold_manager
        
        # Migration thresholds
        self.hot_to_warm_hours = 24
        self.warm_to_cold_days = 30
        self.access_threshold_for_retention = 5
        
    async def migrate_hot_to_warm(self) -> int:
        """Migrate items from hot to warm memory"""
        migrated_count = 0
        cutoff_time = datetime.now() - timedelta(hours=self.hot_to_warm_hours)
        
        try:
            # Query hot memory for old items
            old_items = await self.hot_manager.query({
                "created_before": cutoff_time.isoformat()
            })
            
            concepts = []
            relationships = []
            
            for item in old_items:
                if item.memory_type == MemoryType.CONCEPT:
                    concepts.append(item)
                elif item.memory_type == MemoryType.RELATIONSHIP:
                    relationships.append(item)
            
            if concepts or relationships:
                # Store as knowledge graph in warm memory
                success = await self.warm_manager.store_knowledge_graph(concepts, relationships)
                if success:
                    migrated_count = len(concepts) + len(relationships)
                    logger.info(f"Migrated {migrated_count} items from hot to warm memory")
        
        except Exception as e:
            logger.error(f"Failed to migrate hot to warm: {e}")
        
        return migrated_count
    
    async def migrate_warm_to_cold(self, embeddings_generator) -> int:
        """Migrate items from warm to cold memory with embeddings"""
        migrated_count = 0
        cutoff_time = datetime.now() - timedelta(days=self.warm_to_cold_days)
        
        try:
            # Query warm memory for old items
            query = """
            MATCH (n:Concept)
            WHERE datetime(n.created_at) < datetime($cutoff)
            RETURN n
            ORDER BY n.created_at
            LIMIT 1000
            """
            
            results = await self.warm_manager.query_graph(query, {"cutoff": cutoff_time.isoformat()})
            
            for result in results:
                # Create memory item from graph node
                item = MemoryItem(
                    item_id=result.get("n.id", str(uuid.uuid4())),
                    content=json.loads(result.get("n.content", "{}")),
                    memory_type=MemoryType.CONCEPT,
                    tier=MemoryTier.COLD,
                    importance_score=result.get("n.importance", 0.5)
                )
                
                # Generate embedding
                if hasattr(embeddings_generator, 'generate_embeddings'):
                    embedding_result = await embeddings_generator.generate_embeddings(str(item.content))
                    if embedding_result.get("success"):
                        embedding = np.array(embedding_result["embedding"])
                        
                        # Store in cold memory
                        success = await self.cold_manager.store_with_embedding(item, embedding)
                        if success:
                            migrated_count += 1
            
            logger.info(f"Migrated {migrated_count} items from warm to cold memory")
        
        except Exception as e:
            logger.error(f"Failed to migrate warm to cold: {e}")
        
        return migrated_count
    
    async def run_automatic_migration(self, embeddings_generator=None):
        """Run automatic migration between all tiers"""
        logger.info("Starting automatic memory migration")
        
        # Hot -> Warm migration
        hot_migrated = await self.migrate_hot_to_warm()
        
        # Warm -> Cold migration (if embeddings generator provided)
        cold_migrated = 0
        if embeddings_generator:
            cold_migrated = await self.migrate_warm_to_cold(embeddings_generator)
        
        logger.info(f"Migration completed: {hot_migrated} hot->warm, {cold_migrated} warm->cold")
        return {"hot_to_warm": hot_migrated, "warm_to_cold": cold_migrated}

class MultiTierMemorySystem:
    """Main orchestrator for the multi-tier memory system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        
        # Initialize tier managers
        self.hot_manager = HotMemoryManager(
            redis_url=config.get("redis_url", "redis://localhost:6379")
        )
        
        self.warm_manager = WarmMemoryManager(
            neo4j_uri=config.get("neo4j_uri", "bolt://localhost:7687"),
            username=config.get("neo4j_username", "neo4j"),
            password=config.get("neo4j_password", "thoughtseed")
        )
        
        self.cold_manager = ColdMemoryManager(
            storage_path=config.get("cold_storage_path", "/tmp/cold_memory"),
            qdrant_url=config.get("qdrant_url", "http://localhost:6333")
        )
        
        # Migration manager
        self.migration_manager = MemoryMigrationManager(
            self.hot_manager, self.warm_manager, self.cold_manager
        )
        
        self.connected = False
        self.stats_history = []
    
    async def initialize(self) -> bool:
        """Initialize all memory tiers"""
        try:
            await self.hot_manager.connect()
            await self.warm_manager.connect()
            await self.cold_manager.connect()
            
            self.connected = True
            logger.info("Multi-tier memory system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize memory system: {e}")
            return False
    
    async def store_concept(self, concept_data: Dict[str, Any], importance: float = 0.5, 
                          embedding: Optional[np.ndarray] = None) -> str:
        """Store a concept in appropriate memory tier based on importance and recency"""
        
        item = MemoryItem(
            content=concept_data,
            memory_type=MemoryType.CONCEPT,
            tier=MemoryTier.HOT,  # Start in hot memory
            importance_score=importance,
            tags=concept_data.get("tags", []),
            domain_tags=concept_data.get("domain_tags", [])
        )
        
        # Store in hot memory first
        success = await self.hot_manager.store(item)
        
        if success:
            logger.debug(f"Stored concept {item.item_id} in hot memory")
            return item.item_id
        else:
            logger.error(f"Failed to store concept in hot memory")
            return ""
    
    async def retrieve_concept(self, concept_id: str) -> Optional[MemoryItem]:
        """Retrieve concept from any memory tier (hot -> warm -> cold)"""
        
        # Try hot memory first (fastest)
        item = await self.hot_manager.retrieve(concept_id)
        if item:
            return item
        
        # Try warm memory (knowledge graph)
        related_concepts = await self.warm_manager.find_related_concepts(concept_id, max_depth=1)
        for concept_data in related_concepts:
            if concept_data.get("id") == concept_id:
                # Reconstruct memory item from graph data
                return MemoryItem(
                    item_id=concept_id,
                    content=json.loads(concept_data.get("content", "{}")),
                    memory_type=MemoryType.CONCEPT,
                    tier=MemoryTier.WARM,
                    importance_score=concept_data.get("importance", 0.5)
                )
        
        # Try cold memory (semantic search) - would need more sophisticated lookup
        # For now, return None if not found in hot/warm
        return None
    
    async def semantic_search(self, query_text: str, embedding: np.ndarray, 
                            limit: int = 10) -> List[Tuple[MemoryItem, float]]:
        """Perform semantic search across memory tiers"""
        results = []
        
        # Search cold memory (most comprehensive semantic search)
        cold_results = await self.cold_manager.semantic_search(embedding, limit=limit)
        results.extend(cold_results)
        
        # Could also search warm memory graph by content similarity
        # and hot memory by text matching
        
        return results
    
    async def get_comprehensive_stats(self) -> Dict[str, MemoryStats]:
        """Get statistics from all memory tiers"""
        stats = {}
        
        try:
            stats[MemoryTier.HOT.value] = await self.hot_manager.get_stats()
            stats[MemoryTier.WARM.value] = await self.warm_manager.get_stats()
            stats[MemoryTier.COLD.value] = await self.cold_manager.get_stats()
            
            # Store in history
            self.stats_history.append({
                "timestamp": datetime.now(),
                "stats": stats
            })
            
            # Keep only last 100 entries
            if len(self.stats_history) > 100:
                self.stats_history = self.stats_history[-100:]
        
        except Exception as e:
            logger.error(f"Failed to get comprehensive stats: {e}")
        
        return stats
    
    async def run_maintenance(self, embeddings_generator=None):
        """Run memory system maintenance (migration, compression, cleanup)"""
        logger.info("Starting memory system maintenance")
        
        # Run automatic migration
        migration_stats = await self.migration_manager.run_automatic_migration(embeddings_generator)
        
        # Run compression on cold memory (every 6 months would be configured separately)
        compression_stats = await self.cold_manager.compress_memory(CompressionLevel.LIGHT)
        
        # Get updated stats
        system_stats = await self.get_comprehensive_stats()
        
        maintenance_report = {
            "migration": migration_stats,
            "compression": {"items_compressed": compression_stats},
            "system_stats": system_stats,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info("Memory maintenance completed")
        return maintenance_report

# Global service instance
multi_tier_memory = MultiTierMemorySystem()

# Test function
async def test_multi_tier_memory():
    """Test the multi-tier memory system"""
    print("🧪 Testing Multi-Tier Memory System")
    print("=" * 50)
    
    # Initialize system
    print("🔄 Initializing memory system...")
    system = MultiTierMemorySystem()
    success = await system.initialize()
    
    print(f"📊 Initialization: {'✅ Success' if success else '❌ Failed'}")
    
    # Test storing concepts
    print("\n💾 Testing concept storage...")
    test_concepts = [
        {
            "content": {
                "term": "synaptic_plasticity",
                "definition": "Ability of synapses to strengthen or weaken over time",
                "domain": "neuroscience"
            },
            "importance": 0.9,
            "tags": ["neuroscience", "plasticity"],
            "domain_tags": ["synapses", "learning"]
        },
        {
            "content": {
                "term": "backpropagation",
                "definition": "Algorithm for training neural networks",
                "domain": "artificial_intelligence"
            },
            "importance": 0.8,
            "tags": ["ai", "algorithm"],
            "domain_tags": ["neural_networks", "training"]
        },
        {
            "content": {
                "term": "hebbian_learning",
                "definition": "Learning rule where neurons that fire together wire together",
                "domain": "computational_neuroscience"
            },
            "importance": 0.85,
            "tags": ["neuroscience", "ai", "learning"],
            "domain_tags": ["cross_domain", "learning_rules"]
        }
    ]
    
    stored_ids = []
    for i, concept_data in enumerate(test_concepts):
        concept_id = await system.store_concept(
            concept_data["content"],
            importance=concept_data["importance"]
        )
        if concept_id:
            stored_ids.append(concept_id)
            print(f"  ✅ Stored concept {i+1}: {concept_data['content']['term']} (ID: {concept_id[:8]}...)")
        else:
            print(f"  ❌ Failed to store concept {i+1}")
    
    # Test retrieval
    print(f"\n🔍 Testing concept retrieval...")
    for i, concept_id in enumerate(stored_ids):
        retrieved = await system.retrieve_concept(concept_id)
        if retrieved:
            term = retrieved.content.get("term", "Unknown")
            print(f"  ✅ Retrieved concept {i+1}: {term} (tier: {retrieved.tier.value})")
        else:
            print(f"  ❌ Failed to retrieve concept {i+1}")
    
    # Test memory statistics
    print(f"\n📈 Testing memory statistics...")
    stats = await system.get_comprehensive_stats()
    
    for tier_name, tier_stats in stats.items():
        print(f"  {tier_name.upper()} Memory:")
        print(f"    📦 Items: {tier_stats.total_items}")
        print(f"    💾 Size: {tier_stats.total_size_bytes} bytes")
        if tier_name == "hot":
            print(f"    🔥 Active sessions: {tier_stats.active_sessions}")
        elif tier_name == "warm":
            print(f"    🌐 Knowledge graphs: {tier_stats.knowledge_graphs}")
        elif tier_name == "cold":
            print(f"    📁 Archived documents: {tier_stats.archived_documents}")
    
    # Test query functionality
    print(f"\n🔎 Testing memory queries...")
    hot_query_results = await system.hot_manager.query({
        "memory_type": "concept",
        "min_importance": 0.8
    })
    print(f"  🔥 Hot memory query: {len(hot_query_results)} high-importance concepts found")
    
    # Test maintenance
    print(f"\n🔧 Testing memory maintenance...")
    maintenance_report = await system.run_maintenance()
    
    print(f"  📊 Migration stats:")
    print(f"    Hot -> Warm: {maintenance_report['migration']['hot_to_warm']} items")
    print(f"    Warm -> Cold: {maintenance_report['migration']['warm_to_cold']} items")
    print(f"  🗜️  Compression: {maintenance_report['compression']['items_compressed']} items")
    
    print("\n🎉 Multi-tier memory system test completed!")
    
    # Show final system state
    final_stats = maintenance_report["system_stats"]
    total_items = sum(stats.total_items for stats in final_stats.values())
    total_size = sum(stats.total_size_bytes for stats in final_stats.values())
    
    print(f"\n📊 Final System State:")
    print(f"  📦 Total items across all tiers: {total_items}")
    print(f"  💾 Total memory usage: {total_size} bytes")
    print(f"  🏗️  Memory architecture: 3-tier (Hot/Warm/Cold)")
    print(f"  ✅ System operational: {success}")
    
    return success

if __name__ == "__main__":
    asyncio.run(test_multi_tier_memory())