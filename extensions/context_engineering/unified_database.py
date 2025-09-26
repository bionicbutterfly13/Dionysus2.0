#!/usr/bin/env python3
"""
🗄️ Unified ASI-Arch Database System
=====================================

Unified database combining Neo4j knowledge graphs + Vector embeddings + AutoSchemaKG
for complete ASI-Arch neural architecture storage and consciousness analysis.

This system follows the spec-driven approach:
- Neo4j: Graph relationships (architecture evolution, consciousness networks)
- Vector Database: Similarity search (architecture embeddings, episode retrieval)  
- AutoSchemaKG: Automatic knowledge graph construction from development data
- SQLite: Lightweight metadata and configuration

Author: ASI-Arch Context Engineering Extension
Date: 2025-09-22
Version: 1.0.0 - Unified Database Migration
"""

import asyncio
import json
import sqlite3
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple, Union
import logging
import hashlib
import pickle
import gzip
from contextlib import contextmanager, asynccontextmanager

# Neo4j imports
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError

# AutoSchemaKG imports with fallback
try:
    from atlas_rag.kg_construction.triple_extraction import KnowledgeGraphExtractor
    from atlas_rag.kg_construction.triple_config import ProcessingConfig
    from atlas_rag.llm_generator import LLMGenerator
    ATLAS_RAG_AVAILABLE = True
except ImportError:
    print("⚠️ Atlas RAG not available, using fallback implementations")
    ATLAS_RAG_AVAILABLE = False

    # Fallback implementations
    class KnowledgeGraphExtractor:
        def __init__(self, *args, **kwargs):
            pass
        def extract_triples(self, text):
            return []

    class ProcessingConfig:
        def __init__(self, *args, **kwargs):
            pass

    class LLMGenerator:
        def __init__(self, *args, **kwargs):
            pass
        def generate(self, prompt):
            return "Fallback response - Atlas RAG not available"

# Vector similarity imports with fallback
import numpy as np
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("⚠️ SentenceTransformers not available - vector embeddings disabled")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

    # Honest fallback implementation - no fraud
    class SentenceTransformer:
        def __init__(self, *args, **kwargs):
            self.model_name = args[0] if args else "sentence-transformers/all-MiniLM-L6-v2"
            print(f"⚠️ WARNING: SentenceTransformers not available for model {self.model_name}")
            print("Install with: pip install sentence-transformers")

        def encode(self, sentences, **kwargs):
            # HONEST ERROR - No fraudulent random vectors
            raise NotImplementedError(
                "Vector embeddings require sentence-transformers package. "
                "Install with: pip install sentence-transformers"
            )

# Local imports
try:
    from .hybrid_database import HybridContextDatabase, VectorIndex
except ImportError:
    # For standalone execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from hybrid_database import HybridContextDatabase, VectorIndex

logger = logging.getLogger(__name__)

# =============================================================================
# Neo4j Knowledge Graph Manager
# =============================================================================

class Neo4jKnowledgeGraph:
    """Neo4j-based knowledge graph for ASI-Arch architectures"""
    
    def __init__(self, uri: str = "bolt://localhost:7687",
                 user: str = "neo4j", password: str = "thoughtseed123"):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None
        self._connected = False
        
    async def connect(self):
        """Connect to Neo4j database"""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            # Test connection
            await self.verify_connectivity()
            self._connected = True
            logger.info(f"✅ Connected to Neo4j at {self.uri}")
            await self._create_constraints_and_indexes()
        except (ServiceUnavailable, AuthError) as e:
            logger.warning(f"⚠️  Neo4j not available: {e}. Using fallback mode.")
            self._connected = False
    
    async def verify_connectivity(self):
        """Verify Neo4j connection"""
        if not self.driver:
            return False
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                return result.single()["test"] == 1
        except Exception as e:
            logger.error(f"Neo4j connectivity test failed: {e}")
            return False
    
    async def _create_constraints_and_indexes(self):
        """Create Neo4j constraints and indexes"""
        if not self._connected:
            return
            
        constraints_and_indexes = [
            # Unique constraints
            "CREATE CONSTRAINT architecture_id_unique IF NOT EXISTS FOR (a:Architecture) REQUIRE a.id IS UNIQUE",
            "CREATE CONSTRAINT episode_id_unique IF NOT EXISTS FOR (e:Episode) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT consciousness_id_unique IF NOT EXISTS FOR (c:ConsciousnessState) REQUIRE c.id IS UNIQUE",
            
            # Indexes for performance
            "CREATE INDEX architecture_performance_idx IF NOT EXISTS FOR (a:Architecture) ON (a.performance_score)",
            "CREATE INDEX architecture_consciousness_idx IF NOT EXISTS FOR (a:Architecture) ON (a.consciousness_level)",
            "CREATE INDEX episode_type_idx IF NOT EXISTS FOR (e:Episode) ON (e.episode_type)",
            "CREATE INDEX consciousness_level_idx IF NOT EXISTS FOR (c:ConsciousnessState) ON (c.level)"
        ]
        
        try:
            with self.driver.session() as session:
                for statement in constraints_and_indexes:
                    try:
                        session.run(statement)
                        logger.debug(f"Executed: {statement}")
                    except Exception as e:
                        logger.debug(f"Constraint/index already exists or failed: {e}")
        except Exception as e:
            logger.error(f"Failed to create constraints and indexes: {e}")
    
    async def create_architecture_node(self, arch_data: Dict[str, Any]) -> bool:
        """Create architecture node in Neo4j"""
        if not self._connected:
            return False
            
        try:
            with self.driver.session() as session:
                query = """
                MERGE (a:Architecture {id: $id})
                SET a.name = $name,
                    a.program = $program,
                    a.result = $result,
                    a.motivation = $motivation,
                    a.analysis = $analysis,
                    a.performance_score = $performance_score,
                    a.consciousness_level = $consciousness_level,
                    a.consciousness_score = $consciousness_score,
                    a.created_at = $created_at,
                    a.updated_at = datetime()
                RETURN a.id as id
                """
                
                result = session.run(query, {
                    'id': arch_data.get('id', arch_data.get('name', str(hash(str(arch_data))))),
                    'name': arch_data.get('name', ''),
                    'program': arch_data.get('program', ''),
                    'result': json.dumps(arch_data.get('result', {})),
                    'motivation': arch_data.get('motivation', ''),
                    'analysis': arch_data.get('analysis', ''),
                    'performance_score': self._extract_performance_score(arch_data),
                    'consciousness_level': arch_data.get('consciousness_level', 'UNKNOWN'),
                    'consciousness_score': arch_data.get('consciousness_score', 0.0),
                    'created_at': arch_data.get('created_at', datetime.now().isoformat())
                })
                
                return result.single() is not None
                
        except Exception as e:
            logger.error(f"Failed to create architecture node: {e}")
            return False
    
    async def create_evolution_relationship(self, parent_id: str, child_id: str, 
                                          evolution_data: Dict[str, Any] = None) -> bool:
        """Create evolution relationship between architectures"""
        if not self._connected:
            return False
            
        try:
            with self.driver.session() as session:
                query = """
                MATCH (parent:Architecture {id: $parent_id})
                MATCH (child:Architecture {id: $child_id})
                MERGE (parent)-[r:EVOLVED_TO]->(child)
                SET r.evolution_strategy = $evolution_strategy,
                    r.performance_improvement = $performance_improvement,
                    r.created_at = datetime()
                RETURN r
                """
                
                evolution_data = evolution_data or {}
                result = session.run(query, {
                    'parent_id': parent_id,
                    'child_id': child_id,
                    'evolution_strategy': evolution_data.get('strategy', 'unknown'),
                    'performance_improvement': evolution_data.get('performance_improvement', 0.0)
                })
                
                return result.single() is not None
                
        except Exception as e:
            logger.error(f"Failed to create evolution relationship: {e}")
            return False
    
    async def find_evolution_path(self, start_id: str, end_id: str, max_depth: int = 5) -> List[str]:
        """Find evolution path between architectures"""
        if not self._connected:
            return []
            
        try:
            with self.driver.session() as session:
                query = """
                MATCH path = shortestPath((start:Architecture {id: $start_id})-[:EVOLVED_TO*1..$max_depth]->(end:Architecture {id: $end_id}))
                RETURN [node in nodes(path) | node.id] as path
                """
                
                result = session.run(query, {
                    'start_id': start_id,
                    'end_id': end_id,
                    'max_depth': max_depth
                })
                
                record = result.single()
                return record['path'] if record else []
                
        except Exception as e:
            logger.error(f"Failed to find evolution path: {e}")
            return []
    
    async def get_architectures_by_consciousness(self, consciousness_level: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get architectures by consciousness level"""
        if not self._connected:
            return []
            
        try:
            with self.driver.session() as session:
                query = """
                MATCH (a:Architecture {consciousness_level: $consciousness_level})
                RETURN a.id as id, a.name as name, a.performance_score as performance_score,
                       a.consciousness_score as consciousness_score, a.created_at as created_at
                ORDER BY a.performance_score DESC
                LIMIT $limit
                """
                
                result = session.run(query, {
                    'consciousness_level': consciousness_level,
                    'limit': limit
                })
                
                return [dict(record) for record in result]
                
        except Exception as e:
            logger.error(f"Failed to get architectures by consciousness: {e}")
            return []
    
    def _extract_performance_score(self, arch_data: Dict[str, Any]) -> float:
        """Extract performance score from architecture data"""
        result = arch_data.get('result', {})
        if isinstance(result, dict):
            test_result = result.get('test', '')
            if 'acc=' in test_result:
                try:
                    return float(test_result.split('acc=')[1].split(',')[0])
                except:
                    pass
        return 0.0
    
    async def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")

# =============================================================================
# Enhanced Vector Database
# =============================================================================

class EnhancedVectorDatabase:
    """Enhanced vector database with sentence transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dimension: int = 384):
        self.model_name = model_name
        self.dimension = dimension
        self.encoder = None
        self.vectors: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.RLock()
        self._load_encoder()
    
    def _load_encoder(self):
        """Load sentence transformer model"""
        try:
            self.encoder = SentenceTransformer(self.model_name)
            logger.info(f"✅ Loaded sentence transformer: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load sentence transformer: {e}")
            self.encoder = None
    
    async def add_architecture_embedding(self, arch_id: str, arch_data: Dict[str, Any]):
        """Add architecture embedding"""
        if not self.encoder:
            return False
            
        try:
            # Create text representation for embedding
            text_repr = self._create_architecture_text(arch_data)
            
            # Generate embedding
            embedding = self.encoder.encode([text_repr])[0]
            
            with self.lock:
                self.vectors[arch_id] = embedding
                self.metadata[arch_id] = {
                    'type': 'architecture',
                    'name': arch_data.get('name', ''),
                    'consciousness_level': arch_data.get('consciousness_level', ''),
                    'performance_score': self._extract_performance_score(arch_data),
                    'created_at': arch_data.get('created_at', datetime.now().isoformat())
                }
            
            logger.debug(f"Added embedding for architecture: {arch_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add architecture embedding: {e}")
            return False
    
    def _create_architecture_text(self, arch_data: Dict[str, Any]) -> str:
        """Create text representation of architecture for embedding"""
        parts = []
        
        if arch_data.get('name'):
            parts.append(f"Architecture: {arch_data['name']}")
        
        if arch_data.get('motivation'):
            parts.append(f"Motivation: {arch_data['motivation']}")
        
        if arch_data.get('analysis'):
            parts.append(f"Analysis: {arch_data['analysis']}")
        
        if arch_data.get('consciousness_level'):
            parts.append(f"Consciousness: {arch_data['consciousness_level']}")
        
        return " | ".join(parts)
    
    async def similarity_search(self, query_text: str, top_k: int = 10) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Find similar architectures using text query"""
        if not self.encoder or not self.vectors:
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.encoder.encode([query_text])[0]
            
            # Calculate similarities
            similarities = []
            for arch_id, vector in self.vectors.items():
                similarity = np.dot(query_embedding, vector) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(vector)
                )
                similarities.append((arch_id, float(similarity), self.metadata[arch_id]))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Failed to perform similarity search: {e}")
            return []
    
    def _extract_performance_score(self, arch_data: Dict[str, Any]) -> float:
        """Extract performance score from architecture data"""
        result = arch_data.get('result', {})
        if isinstance(result, dict):
            test_result = result.get('test', '')
            if 'acc=' in test_result:
                try:
                    return float(test_result.split('acc=')[1].split(',')[0])
                except:
                    pass
        return 0.0

# =============================================================================
# AutoSchemaKG Integration
# =============================================================================

class AutoSchemaKGIntegration:
    """AutoSchemaKG integration for automatic knowledge graph construction"""
    
    def __init__(self, data_directory: str = "development_conversations",
                 output_directory: str = "knowledge_graphs/asi_arch"):
        self.data_directory = Path(data_directory)
        self.output_directory = Path(output_directory)
        self.kg_extractor = None
        self._setup_directories()
    
    def _setup_directories(self):
        """Setup required directories"""
        self.data_directory.mkdir(parents=True, exist_ok=True)
        self.output_directory.mkdir(parents=True, exist_ok=True)
    
    async def setup_kg_extractor(self, openai_client=None):
        """Setup AutoSchemaKG extractor"""
        try:
            config = ProcessingConfig(
                data_directory=str(self.data_directory),
                output_directory=str(self.output_directory),
                batch_size_triple=3,
                batch_size_concept=16,
                remove_doc_spaces=True
            )
            
            if openai_client:
                llm_generator = LLMGenerator(openai_client, model_name="gpt-4")
                self.kg_extractor = KnowledgeGraphExtractor(
                    model=llm_generator,
                    config=config
                )
                logger.info("✅ AutoSchemaKG extractor setup complete")
            else:
                logger.warning("⚠️  No OpenAI client provided, AutoSchemaKG disabled")
                
        except Exception as e:
            logger.error(f"Failed to setup AutoSchemaKG: {e}")
    
    async def extract_development_knowledge(self) -> bool:
        """Extract knowledge from development conversations"""
        if not self.kg_extractor:
            logger.warning("AutoSchemaKG not available")
            return False
        
        try:
            # Run extraction
            await self.kg_extractor.run_extraction()
            
            # Generate concepts
            await self.kg_extractor.generate_concept_csv()
            
            logger.info("✅ Development knowledge extraction complete")
            return True
            
        except Exception as e:
            logger.error(f"Failed to extract development knowledge: {e}")
            return False

# =============================================================================
# Unified Database System
# =============================================================================

class UnifiedASIArchDatabase:
    """Unified database system combining Neo4j + Vector + AutoSchemaKG + SQLite"""
    
    def __init__(self, base_path: str = "extensions/context_engineering/data",
                 neo4j_uri: str = "bolt://localhost:7687",
                 neo4j_user: str = "neo4j",
                 neo4j_password: str = "password"):
        
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.neo4j_graph = Neo4jKnowledgeGraph(neo4j_uri, neo4j_user, neo4j_password)
        self.vector_db = EnhancedVectorDatabase()
        self.autoschema_kg = AutoSchemaKGIntegration()
        
        # Fallback hybrid database for when Neo4j is not available
        self.fallback_db = HybridContextDatabase(str(self.base_path))
        
        # SQLite for metadata
        self.sqlite_db = str(self.base_path / "unified_asi_arch.db")
        
        # State tracking
        self._neo4j_available = False
        self._initialized = False
        
        logger.info(f"Unified ASI-Arch database initialized at {self.base_path}")
    
    async def initialize(self, openai_client=None):
        """Initialize all database components"""
        try:
            # Try to connect to Neo4j
            await self.neo4j_graph.connect()
            self._neo4j_available = self.neo4j_graph._connected
            
            # Setup AutoSchemaKG if OpenAI client provided
            if openai_client:
                await self.autoschema_kg.setup_kg_extractor(openai_client)
            
            # Initialize SQLite schema
            self._init_sqlite_schema()
            
            self._initialized = True
            
            if self._neo4j_available:
                logger.info("✅ Unified database fully initialized with Neo4j")
            else:
                logger.info("✅ Unified database initialized in fallback mode (no Neo4j)")
                
        except Exception as e:
            logger.error(f"Failed to initialize unified database: {e}")
            raise
    
    def _init_sqlite_schema(self):
        """Initialize SQLite schema for metadata"""
        with sqlite3.connect(self.sqlite_db) as conn:
            # Migration tracking
            conn.execute("""
                CREATE TABLE IF NOT EXISTS migration_status (
                    id INTEGER PRIMARY KEY,
                    component TEXT NOT NULL,
                    status TEXT NOT NULL,
                    migrated_count INTEGER DEFAULT 0,
                    last_migrated_at TEXT,
                    notes TEXT
                )
            """)
            
            # System configuration
            conn.execute("""
                CREATE TABLE IF NOT EXISTS system_config (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
    
    async def store_architecture(self, arch_data: Dict[str, Any], 
                                consciousness_level: str = None, 
                                consciousness_score: float = None) -> bool:
        """Store architecture in unified system"""
        if not self._initialized:
            await self.initialize()
        
        # Enhance arch_data with consciousness info
        if consciousness_level:
            arch_data['consciousness_level'] = consciousness_level
        if consciousness_score is not None:
            arch_data['consciousness_score'] = consciousness_score
        
        arch_id = arch_data.get('id', arch_data.get('name', str(hash(str(arch_data)))))
        arch_data['id'] = arch_id
        
        success_count = 0
        
        # Store in Neo4j if available
        if self._neo4j_available:
            if await self.neo4j_graph.create_architecture_node(arch_data):
                success_count += 1
                logger.debug(f"Stored in Neo4j: {arch_id}")
        else:
            # Store in fallback hybrid database
            self.fallback_db.store_architecture(
                arch_data, 
                consciousness_level or 'UNKNOWN',
                consciousness_score or 0.0
            )
            success_count += 1
            logger.debug(f"Stored in fallback DB: {arch_id}")
        
        # Store in vector database
        if await self.vector_db.add_architecture_embedding(arch_id, arch_data):
            success_count += 1
            logger.debug(f"Stored embedding: {arch_id}")
        
        logger.info(f"Architecture {arch_id} stored in {success_count} systems")
        return success_count > 0
    
    async def create_evolution_relationship(self, parent_id: str, child_id: str,
                                          evolution_data: Dict[str, Any] = None) -> bool:
        """Create evolution relationship between architectures"""
        if not self._initialized:
            await self.initialize()
        
        if self._neo4j_available:
            return await self.neo4j_graph.create_evolution_relationship(
                parent_id, child_id, evolution_data
            )
        else:
            # Use fallback database
            self.fallback_db.create_architecture_relationship(
                parent_id, child_id, 'evolved_from', 
                evolution_data.get('strength', 0.8) if evolution_data else 0.8
            )
            return True
    
    async def find_similar_architectures(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Find similar architectures using vector search"""
        if not self._initialized:
            await self.initialize()
        
        results = await self.vector_db.similarity_search(query, top_k)
        return [
            {
                'id': arch_id,
                'similarity': similarity,
                'metadata': metadata
            }
            for arch_id, similarity, metadata in results
        ]
    
    async def get_consciousness_evolution_path(self, start_id: str, target_consciousness: str) -> List[str]:
        """Find evolution path to target consciousness level"""
        if not self._initialized:
            await self.initialize()
        
        if self._neo4j_available:
            # Find architectures with target consciousness
            target_archs = await self.neo4j_graph.get_architectures_by_consciousness(target_consciousness, 5)
            
            for target_arch in target_archs:
                path = await self.neo4j_graph.find_evolution_path(start_id, target_arch['id'])
                if path:
                    return path
            return []
        else:
            # Use fallback database
            return self.fallback_db.find_consciousness_evolution_path(start_id, target_consciousness)
    
    async def extract_development_knowledge(self) -> bool:
        """Extract knowledge from development data using AutoSchemaKG"""
        if not self._initialized:
            await self.initialize()
        
        return await self.autoschema_kg.extract_development_knowledge()
    
    async def get_system_statistics(self) -> Dict[str, Any]:
        """Get unified system statistics"""
        if not self._initialized:
            await self.initialize()
        
        stats = {
            'neo4j_available': self._neo4j_available,
            'vector_embeddings_count': len(self.vector_db.vectors),
            'autoschema_kg_available': self.autoschema_kg.kg_extractor is not None,
            'fallback_db_stats': {}
        }
        
        if not self._neo4j_available:
            stats['fallback_db_stats'] = {
                'consciousness_distribution': self.fallback_db.get_consciousness_distribution(),
                'graph_nodes': len(self.fallback_db.graph_db.nodes),
                'graph_edges': len(self.fallback_db.graph_db.edges)
            }
        
        return stats
    
    async def close(self):
        """Close all database connections"""
        await self.neo4j_graph.close()
        logger.info("Unified database connections closed")

# =============================================================================
# Factory and Testing
# =============================================================================

async def create_unified_database(base_path: str = "extensions/context_engineering/data",
                                 neo4j_uri: str = "bolt://localhost:7687",
                                 neo4j_user: str = "neo4j",
                                 neo4j_password: str = "password",
                                 openai_client=None) -> UnifiedASIArchDatabase:
    """Factory function to create unified database"""
    db = UnifiedASIArchDatabase(base_path, neo4j_uri, neo4j_user, neo4j_password)
    await db.initialize(openai_client)
    return db

async def test_unified_database():
    """Test the unified database system"""
    print("🧪 Testing Unified ASI-Arch Database System")
    
    # Create database
    db = await create_unified_database()
    
    try:
        # Test architecture storage
        print("\n1. Testing architecture storage...")
        test_arch = {
            'name': 'unified_test_transformer',
            'program': 'class UnifiedTestTransformer(nn.Module): pass',
            'result': {'test': 'acc=0.92'},
            'motivation': 'testing unified database system',
            'analysis': 'demonstrates integration across Neo4j, vector DB, and fallback systems'
        }
        
        success = await db.store_architecture(test_arch, 'SELF_AWARE', 0.75)
        print(f"✅ Architecture stored: {success}")
        
        # Test similarity search
        print("\n2. Testing similarity search...")
        similar = await db.find_similar_architectures("transformer architecture for testing", top_k=3)
        print(f"Found {len(similar)} similar architectures")
        for arch in similar:
            print(f"  - {arch['id']}: {arch['similarity']:.3f}")
        
        # Test evolution relationships
        print("\n3. Testing evolution relationships...")
        await db.store_architecture({
            'name': 'parent_unified_arch',
            'program': 'class ParentArch(nn.Module): pass',
            'result': {'test': 'acc=0.80'}
        }, 'ACTIVE', 0.60)
        
        success = await db.create_evolution_relationship(
            'parent_unified_arch', 'unified_test_transformer',
            {'strategy': 'consciousness_enhancement', 'performance_improvement': 0.12}
        )
        print(f"✅ Evolution relationship created: {success}")
        
        # Test consciousness evolution path
        print("\n4. Testing consciousness evolution path...")
        path = await db.get_consciousness_evolution_path('parent_unified_arch', 'SELF_AWARE')
        print(f"Evolution path: {path}")
        
        # Test system statistics
        print("\n5. Testing system statistics...")
        stats = await db.get_system_statistics()
        print(f"System stats: {json.dumps(stats, indent=2)}")
        
        print("\n✅ Unified database test completed successfully!")
        
    finally:
        await db.close()

if __name__ == "__main__":
    asyncio.run(test_unified_database())
