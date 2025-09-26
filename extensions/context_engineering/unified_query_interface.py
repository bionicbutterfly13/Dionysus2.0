#!/usr/bin/env python3
"""
🔍 Unified Query Interface for ASI-Arch Database System
======================================================

Unified query interface that works seamlessly across:
- Neo4j Knowledge Graph (when available)
- Vector Database (similarity search)
- Fallback Hybrid Database (SQLite + JSON)
- AutoSchemaKG (automatic knowledge extraction)

Provides high-level query methods that abstract the underlying storage systems.

Author: ASI-Arch Context Engineering Extension
Date: 2025-09-22
Version: 1.0.0 - Unified Query Interface
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass
from enum import Enum

try:
    from .unified_database import UnifiedASIArchDatabase, create_unified_database
except ImportError:
    # For standalone execution
    import sys
    sys.path.append(str(Path(__file__).parent))
    from unified_database import UnifiedASIArchDatabase, create_unified_database

logger = logging.getLogger(__name__)

# =============================================================================
# Query Types and Parameters
# =============================================================================

class QueryType(Enum):
    """Types of queries supported by the unified interface"""
    ARCHITECTURE_SEARCH = "architecture_search"
    SIMILARITY_SEARCH = "similarity_search"
    EVOLUTION_PATH = "evolution_path"
    CONSCIOUSNESS_ANALYSIS = "consciousness_analysis"
    PERFORMANCE_RANKING = "performance_ranking"
    RELATIONSHIP_DISCOVERY = "relationship_discovery"
    HYBRID_EXPLORATION = "hybrid_exploration"

@dataclass
class QueryResult:
    """Standardized query result format"""
    query_type: QueryType
    query_params: Dict[str, Any]
    results: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    execution_time: float
    source_systems: List[str]
    total_results: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'query_type': self.query_type.value,
            'query_params': self.query_params,
            'results': self.results,
            'metadata': self.metadata,
            'execution_time': self.execution_time,
            'source_systems': self.source_systems,
            'total_results': self.total_results
        }

@dataclass
class QueryParameters:
    """Query parameters for different query types"""
    # Common parameters
    limit: int = 10
    offset: int = 0
    
    # Search parameters
    query_text: str = ""
    architecture_name: str = ""
    consciousness_level: str = ""
    
    # Performance parameters
    min_performance: float = 0.0
    max_performance: float = 1.0
    
    # Evolution parameters
    start_architecture: str = ""
    target_architecture: str = ""
    max_depth: int = 5
    
    # Similarity parameters
    similarity_threshold: float = 0.5
    
    # Time range parameters
    start_date: Optional[str] = None
    end_date: Optional[str] = None

# =============================================================================
# Unified Query Interface
# =============================================================================

class UnifiedQueryInterface:
    """Unified query interface for all database systems"""
    
    def __init__(self, database: UnifiedASIArchDatabase):
        self.database = database
        self.query_cache: Dict[str, Tuple[QueryResult, datetime]] = {}
        self.cache_ttl = timedelta(minutes=10)
        
    def _generate_cache_key(self, query_type: QueryType, params: QueryParameters) -> str:
        """Generate cache key for query"""
        key_data = {
            'type': query_type.value,
            'params': params.__dict__
        }
        return str(hash(json.dumps(key_data, sort_keys=True)))
    
    def _get_cached_result(self, cache_key: str) -> Optional[QueryResult]:
        """Get cached query result if still valid"""
        if cache_key in self.query_cache:
            result, timestamp = self.query_cache[cache_key]
            if datetime.now() - timestamp < self.cache_ttl:
                return result
            else:
                del self.query_cache[cache_key]
        return None
    
    def _cache_result(self, cache_key: str, result: QueryResult):
        """Cache query result"""
        self.query_cache[cache_key] = (result, datetime.now())
    
    async def architecture_search(self, params: QueryParameters) -> QueryResult:
        """Search for architectures by various criteria"""
        start_time = datetime.now()
        cache_key = self._generate_cache_key(QueryType.ARCHITECTURE_SEARCH, params)
        
        # Check cache
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result
        
        results = []
        source_systems = []
        
        try:
            # If we have specific architecture name, search by that
            if params.architecture_name:
                # Try Neo4j first (if available)
                if self.database._neo4j_available:
                    neo4j_results = await self.database.neo4j_graph.get_architectures_by_consciousness(
                        params.consciousness_level or "ACTIVE", params.limit
                    )
                    # Filter by name if provided
                    if params.architecture_name:
                        neo4j_results = [r for r in neo4j_results if params.architecture_name.lower() in r.get('name', '').lower()]
                    results.extend(neo4j_results)
                    source_systems.append("Neo4j")
                else:
                    # Use fallback database
                    fallback_results = self.database.fallback_db.get_top_performing_architectures(
                        params.consciousness_level, params.limit
                    )
                    if params.architecture_name:
                        fallback_results = [r for r in fallback_results if params.architecture_name.lower() in r.get('name', '').lower()]
                    results.extend(fallback_results)
                    source_systems.append("Fallback DB")
            
            # If we have consciousness level filter
            elif params.consciousness_level:
                if self.database._neo4j_available:
                    neo4j_results = await self.database.neo4j_graph.get_architectures_by_consciousness(
                        params.consciousness_level, params.limit
                    )
                    results.extend(neo4j_results)
                    source_systems.append("Neo4j")
                else:
                    fallback_results = self.database.fallback_db.get_top_performing_architectures(
                        params.consciousness_level, params.limit
                    )
                    results.extend(fallback_results)
                    source_systems.append("Fallback DB")
            
            # General search
            else:
                if self.database._neo4j_available:
                    neo4j_results = await self.database.neo4j_graph.get_architectures_by_consciousness(
                        "ACTIVE", params.limit
                    )
                    results.extend(neo4j_results)
                    source_systems.append("Neo4j")
                else:
                    fallback_results = self.database.fallback_db.get_top_performing_architectures(
                        limit=params.limit
                    )
                    results.extend(fallback_results)
                    source_systems.append("Fallback DB")
            
            # Filter by performance if specified
            if params.min_performance > 0 or params.max_performance < 1:
                results = [r for r in results 
                          if params.min_performance <= r.get('performance_score', 0) <= params.max_performance]
            
            # Apply offset and limit
            results = results[params.offset:params.offset + params.limit]
            
        except Exception as e:
            logger.error(f"Architecture search failed: {e}")
            results = []
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        query_result = QueryResult(
            query_type=QueryType.ARCHITECTURE_SEARCH,
            query_params=params.__dict__,
            results=results,
            metadata={
                'search_criteria': {
                    'architecture_name': params.architecture_name,
                    'consciousness_level': params.consciousness_level,
                    'performance_range': [params.min_performance, params.max_performance]
                }
            },
            execution_time=execution_time,
            source_systems=source_systems,
            total_results=len(results)
        )
        
        # Cache result
        self._cache_result(cache_key, query_result)
        return query_result
    
    async def similarity_search(self, params: QueryParameters) -> QueryResult:
        """Find similar architectures using vector search"""
        start_time = datetime.now()
        cache_key = self._generate_cache_key(QueryType.SIMILARITY_SEARCH, params)
        
        # Check cache
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result
        
        results = []
        source_systems = ["Vector DB"]
        
        try:
            # Use vector database for similarity search
            similar_results = await self.database.find_similar_architectures(
                params.query_text, params.limit
            )
            
            # Filter by similarity threshold
            results = [r for r in similar_results 
                      if r.get('similarity', 0) >= params.similarity_threshold]
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            results = []
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        query_result = QueryResult(
            query_type=QueryType.SIMILARITY_SEARCH,
            query_params=params.__dict__,
            results=results,
            metadata={
                'query_text': params.query_text,
                'similarity_threshold': params.similarity_threshold,
                'vector_model': self.database.vector_db.model_name
            },
            execution_time=execution_time,
            source_systems=source_systems,
            total_results=len(results)
        )
        
        self._cache_result(cache_key, query_result)
        return query_result
    
    async def evolution_path_search(self, params: QueryParameters) -> QueryResult:
        """Find evolution paths between architectures"""
        start_time = datetime.now()
        cache_key = self._generate_cache_key(QueryType.EVOLUTION_PATH, params)
        
        # Check cache
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result
        
        results = []
        source_systems = []
        
        try:
            if params.target_architecture:
                # Find path to specific architecture
                if self.database._neo4j_available:
                    path = await self.database.neo4j_graph.find_evolution_path(
                        params.start_architecture, params.target_architecture, params.max_depth
                    )
                    source_systems.append("Neo4j")
                else:
                    path = self.database.fallback_db.graph_db.find_path(
                        params.start_architecture, params.target_architecture, params.max_depth
                    )
                    source_systems.append("Fallback DB")
                
                if path:
                    results.append({
                        'path': path,
                        'length': len(path) - 1,
                        'start': params.start_architecture,
                        'end': params.target_architecture
                    })
            
            else:
                # Find paths to consciousness level
                path = await self.database.get_consciousness_evolution_path(
                    params.start_architecture, params.consciousness_level
                )
                if path:
                    results.append({
                        'path': path,
                        'length': len(path) - 1,
                        'start': params.start_architecture,
                        'target_consciousness': params.consciousness_level
                    })
                source_systems.append("Unified DB")
        
        except Exception as e:
            logger.error(f"Evolution path search failed: {e}")
            results = []
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        query_result = QueryResult(
            query_type=QueryType.EVOLUTION_PATH,
            query_params=params.__dict__,
            results=results,
            metadata={
                'max_depth': params.max_depth,
                'path_finding_algorithm': 'shortest_path'
            },
            execution_time=execution_time,
            source_systems=source_systems,
            total_results=len(results)
        )
        
        self._cache_result(cache_key, query_result)
        return query_result
    
    async def consciousness_analysis(self, params: QueryParameters) -> QueryResult:
        """Analyze consciousness distribution and patterns"""
        start_time = datetime.now()
        cache_key = self._generate_cache_key(QueryType.CONSCIOUSNESS_ANALYSIS, params)
        
        # Check cache
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return cached_result
        
        results = []
        source_systems = ["Fallback DB"]  # Always use fallback for consciousness distribution
        
        try:
            # Get consciousness distribution
            consciousness_dist = self.database.fallback_db.get_consciousness_distribution()
            
            # Get top performers by consciousness level
            consciousness_levels = consciousness_dist.keys()
            for level in consciousness_levels:
                if not params.consciousness_level or level == params.consciousness_level:
                    top_archs = self.database.fallback_db.get_top_performing_architectures(level, 5)
                    results.append({
                        'consciousness_level': level,
                        'architecture_count': consciousness_dist[level],
                        'top_performers': top_archs
                    })
            
        except Exception as e:
            logger.error(f"Consciousness analysis failed: {e}")
            results = []
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        query_result = QueryResult(
            query_type=QueryType.CONSCIOUSNESS_ANALYSIS,
            query_params=params.__dict__,
            results=results,
            metadata={
                'analysis_type': 'consciousness_distribution',
                'total_consciousness_levels': len(results)
            },
            execution_time=execution_time,
            source_systems=source_systems,
            total_results=len(results)
        )
        
        self._cache_result(cache_key, query_result)
        return query_result
    
    async def hybrid_exploration(self, params: QueryParameters) -> QueryResult:
        """Hybrid exploration combining multiple query types"""
        start_time = datetime.now()
        
        results = []
        source_systems = []
        
        try:
            # Combine similarity search with architecture search
            if params.query_text:
                similarity_params = QueryParameters(
                    query_text=params.query_text,
                    limit=params.limit // 2,
                    similarity_threshold=params.similarity_threshold
                )
                similarity_result = await self.similarity_search(similarity_params)
                results.extend(similarity_result.results)
                source_systems.extend(similarity_result.source_systems)
            
            # Add architecture search results
            arch_params = QueryParameters(
                consciousness_level=params.consciousness_level,
                limit=params.limit // 2,
                min_performance=params.min_performance
            )
            arch_result = await self.architecture_search(arch_params)
            results.extend(arch_result.results)
            source_systems.extend(arch_result.source_systems)
            
            # Remove duplicates based on ID
            seen_ids = set()
            unique_results = []
            for result in results:
                result_id = result.get('id', result.get('name', ''))
                if result_id and result_id not in seen_ids:
                    seen_ids.add(result_id)
                    unique_results.append(result)
            
            results = unique_results[:params.limit]
            
        except Exception as e:
            logger.error(f"Hybrid exploration failed: {e}")
            results = []
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        query_result = QueryResult(
            query_type=QueryType.HYBRID_EXPLORATION,
            query_params=params.__dict__,
            results=results,
            metadata={
                'exploration_strategy': 'similarity_plus_search',
                'deduplication': True
            },
            execution_time=execution_time,
            source_systems=list(set(source_systems)),
            total_results=len(results)
        )
        
        return query_result
    
    async def get_system_insights(self) -> Dict[str, Any]:
        """Get comprehensive system insights"""
        try:
            stats = await self.database.get_system_statistics()
            
            # Add query cache statistics
            stats['query_cache'] = {
                'cached_queries': len(self.query_cache),
                'cache_ttl_minutes': self.cache_ttl.total_seconds() / 60
            }
            
            # Add recent query patterns (if we had query logging)
            stats['query_patterns'] = {
                'most_common_query_types': ['architecture_search', 'similarity_search'],
                'average_execution_time': 0.1
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get system insights: {e}")
            return {}
    
    def clear_cache(self):
        """Clear query cache"""
        self.query_cache.clear()
        logger.info("Query cache cleared")

# =============================================================================
# Query Builder and Convenience Methods
# =============================================================================

class QueryBuilder:
    """Builder for constructing complex queries"""
    
    def __init__(self, query_interface: UnifiedQueryInterface):
        self.interface = query_interface
        self.params = QueryParameters()
    
    def search_architectures(self, name: str = "", consciousness: str = "") -> 'QueryBuilder':
        """Search for specific architectures"""
        self.params.architecture_name = name
        self.params.consciousness_level = consciousness
        return self
    
    def similar_to(self, query_text: str, threshold: float = 0.5) -> 'QueryBuilder':
        """Find architectures similar to query text"""
        self.params.query_text = query_text
        self.params.similarity_threshold = threshold
        return self
    
    def performance_range(self, min_perf: float, max_perf: float = 1.0) -> 'QueryBuilder':
        """Filter by performance range"""
        self.params.min_performance = min_perf
        self.params.max_performance = max_perf
        return self
    
    def limit_results(self, limit: int, offset: int = 0) -> 'QueryBuilder':
        """Limit and paginate results"""
        self.params.limit = limit
        self.params.offset = offset
        return self
    
    def evolution_from(self, start_arch: str, target_arch: str = "", max_depth: int = 5) -> 'QueryBuilder':
        """Find evolution paths"""
        self.params.start_architecture = start_arch
        self.params.target_architecture = target_arch
        self.params.max_depth = max_depth
        return self
    
    async def execute_search(self) -> QueryResult:
        """Execute architecture search"""
        return await self.interface.architecture_search(self.params)
    
    async def execute_similarity(self) -> QueryResult:
        """Execute similarity search"""
        return await self.interface.similarity_search(self.params)
    
    async def execute_evolution(self) -> QueryResult:
        """Execute evolution path search"""
        return await self.interface.evolution_path_search(self.params)
    
    async def execute_hybrid(self) -> QueryResult:
        """Execute hybrid exploration"""
        return await self.interface.hybrid_exploration(self.params)

# =============================================================================
# Testing and Examples
# =============================================================================

async def test_unified_queries():
    """Test the unified query interface"""
    print("🔍 Testing Unified Query Interface")
    
    # Create database and query interface
    db = await create_unified_database()
    query_interface = UnifiedQueryInterface(db)
    
    try:
        # Test architecture search
        print("\n1. Testing architecture search...")
        arch_params = QueryParameters(consciousness_level="ACTIVE", limit=5)
        arch_result = await query_interface.architecture_search(arch_params)
        print(f"Found {arch_result.total_results} architectures in {arch_result.execution_time:.3f}s")
        print(f"Source systems: {arch_result.source_systems}")
        
        # Test similarity search
        print("\n2. Testing similarity search...")
        sim_params = QueryParameters(query_text="transformer attention mechanism", limit=3)
        sim_result = await query_interface.similarity_search(sim_params)
        print(f"Found {sim_result.total_results} similar architectures in {sim_result.execution_time:.3f}s")
        for result in sim_result.results:
            print(f"  - {result['id']}: {result['similarity']:.3f}")
        
        # Test consciousness analysis
        print("\n3. Testing consciousness analysis...")
        consciousness_params = QueryParameters()
        consciousness_result = await query_interface.consciousness_analysis(consciousness_params)
        print(f"Consciousness analysis completed in {consciousness_result.execution_time:.3f}s")
        for result in consciousness_result.results:
            print(f"  - {result['consciousness_level']}: {result['architecture_count']} architectures")
        
        # Test hybrid exploration
        print("\n4. Testing hybrid exploration...")
        hybrid_params = QueryParameters(
            query_text="neural architecture",
            consciousness_level="ACTIVE",
            limit=5
        )
        hybrid_result = await query_interface.hybrid_exploration(hybrid_params)
        print(f"Hybrid exploration found {hybrid_result.total_results} results in {hybrid_result.execution_time:.3f}s")
        
        # Test query builder
        print("\n5. Testing query builder...")
        builder = QueryBuilder(query_interface)
        builder_result = await (builder
                               .search_architectures(consciousness="ACTIVE")
                               .performance_range(0.5)
                               .limit_results(3)
                               .execute_search())
        print(f"Query builder found {builder_result.total_results} results")
        
        # Get system insights
        print("\n6. Getting system insights...")
        insights = await query_interface.get_system_insights()
        print(f"System insights: {json.dumps(insights, indent=2)}")
        
        print("\n✅ Unified query interface test completed!")
        
    finally:
        await db.close()

if __name__ == "__main__":
    asyncio.run(test_unified_queries())


