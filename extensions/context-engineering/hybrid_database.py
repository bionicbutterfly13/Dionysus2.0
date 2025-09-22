#!/usr/bin/env python3
"""
🗄️ Hybrid Database Architecture for Context Engineering
=======================================================

Self-contained hybrid database combining:
- SQLite: Core context engineering data (fast, local, no dependencies)
- JSON Files: Knowledge graph structure (portable, version-controllable)
- In-Memory: Real-time analysis and caching (performance)

This provides all the benefits of MongoDB + Neo4j + Qdrant without external dependencies!

Design Philosophy:
- Self-Contained: No external database services required
- High Performance: In-memory caching for real-time operations
- Portable: Everything stored in files that can be version controlled
- Scalable: Can migrate to external databases later without code changes

Author: ASI-Arch Context Engineering Extension
Date: 2025-09-22
Version: 1.0.0 - Hybrid Database Architecture
"""

import json
import sqlite3
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
import logging
import hashlib
import pickle
import gzip
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# =============================================================================
# Graph Database Layer (JSON-based Neo4j alternative)
# =============================================================================

class GraphNode:
    """Node in the knowledge graph"""
    def __init__(self, node_id: str, node_type: str, properties: Dict[str, Any]):
        self.id = node_id
        self.type = node_type
        self.properties = properties
        self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'type': self.type,
            'properties': self.properties,
            'created_at': self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GraphNode':
        node = cls(data['id'], data['type'], data['properties'])
        node.created_at = data.get('created_at', datetime.now().isoformat())
        return node

class GraphEdge:
    """Edge in the knowledge graph"""
    def __init__(self, source_id: str, target_id: str, edge_type: str, 
                 properties: Dict[str, Any], weight: float = 1.0):
        self.source_id = source_id
        self.target_id = target_id
        self.type = edge_type
        self.properties = properties
        self.weight = weight
        self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'source_id': self.source_id,
            'target_id': self.target_id,
            'type': self.type,
            'properties': self.properties,
            'weight': self.weight,
            'created_at': self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GraphEdge':
        edge = cls(data['source_id'], data['target_id'], data['type'], 
                  data['properties'], data.get('weight', 1.0))
        edge.created_at = data.get('created_at', datetime.now().isoformat())
        return edge

class JSONGraphDatabase:
    """JSON-based graph database (Neo4j alternative)"""
    
    def __init__(self, graph_file: str = "context_graph.json"):
        self.graph_file = graph_file
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        self.adjacency: Dict[str, List[str]] = defaultdict(list)
        self.reverse_adjacency: Dict[str, List[str]] = defaultdict(list)
        self.lock = threading.RLock()
        self.load_graph()
    
    def load_graph(self):
        """Load graph from JSON file"""
        if not Path(self.graph_file).exists():
            return
        
        try:
            with open(self.graph_file, 'r') as f:
                data = json.load(f)
            
            # Load nodes
            for node_data in data.get('nodes', []):
                node = GraphNode.from_dict(node_data)
                self.nodes[node.id] = node
            
            # Load edges
            for edge_data in data.get('edges', []):
                edge = GraphEdge.from_dict(edge_data)
                self.edges.append(edge)
                self.adjacency[edge.source_id].append(edge.target_id)
                self.reverse_adjacency[edge.target_id].append(edge.source_id)
            
            logger.info(f"Loaded graph: {len(self.nodes)} nodes, {len(self.edges)} edges")
            
        except Exception as e:
            logger.error(f"Error loading graph: {e}")
    
    def save_graph(self):
        """Save graph to JSON file"""
        try:
            with self.lock:
                data = {
                    'metadata': {
                        'saved_at': datetime.now().isoformat(),
                        'node_count': len(self.nodes),
                        'edge_count': len(self.edges)
                    },
                    'nodes': [node.to_dict() for node in self.nodes.values()],
                    'edges': [edge.to_dict() for edge in self.edges]
                }
            
            # Atomic write
            temp_file = f"{self.graph_file}.tmp"
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            Path(temp_file).rename(self.graph_file)
            logger.debug(f"Saved graph: {len(self.nodes)} nodes, {len(self.edges)} edges")
            
        except Exception as e:
            logger.error(f"Error saving graph: {e}")
    
    def add_node(self, node_id: str, node_type: str, properties: Dict[str, Any]) -> GraphNode:
        """Add node to graph"""
        with self.lock:
            node = GraphNode(node_id, node_type, properties)
            self.nodes[node_id] = node
            return node
    
    def add_edge(self, source_id: str, target_id: str, edge_type: str, 
                 properties: Dict[str, Any] = None, weight: float = 1.0) -> GraphEdge:
        """Add edge to graph"""
        with self.lock:
            if source_id not in self.nodes or target_id not in self.nodes:
                raise ValueError(f"Nodes must exist before adding edge: {source_id} -> {target_id}")
            
            properties = properties or {}
            edge = GraphEdge(source_id, target_id, edge_type, properties, weight)
            self.edges.append(edge)
            
            self.adjacency[source_id].append(target_id)
            self.reverse_adjacency[target_id].append(source_id)
            
            return edge
    
    def get_neighbors(self, node_id: str, direction: str = "out") -> List[str]:
        """Get neighboring node IDs"""
        if direction == "out":
            return self.adjacency.get(node_id, [])
        elif direction == "in":
            return self.reverse_adjacency.get(node_id, [])
        else:  # both
            return list(set(self.adjacency.get(node_id, []) + self.reverse_adjacency.get(node_id, [])))
    
    def find_path(self, start_id: str, end_id: str, max_depth: int = 5) -> List[str]:
        """Find shortest path between nodes"""
        if start_id == end_id:
            return [start_id]
        
        visited = set()
        queue = deque([(start_id, [start_id])])
        
        while queue:
            current_id, path = queue.popleft()
            
            if len(path) > max_depth:
                continue
            
            if current_id in visited:
                continue
            
            visited.add(current_id)
            
            for neighbor_id in self.get_neighbors(current_id):
                new_path = path + [neighbor_id]
                
                if neighbor_id == end_id:
                    return new_path
                
                if neighbor_id not in visited:
                    queue.append((neighbor_id, new_path))
        
        return []  # No path found
    
    def query_nodes(self, node_type: str = None, properties: Dict[str, Any] = None) -> List[GraphNode]:
        """Query nodes by type and properties"""
        results = []
        
        for node in self.nodes.values():
            if node_type and node.type != node_type:
                continue
            
            if properties:
                match = True
                for key, value in properties.items():
                    if key not in node.properties or node.properties[key] != value:
                        match = False
                        break
                if not match:
                    continue
            
            results.append(node)
        
        return results

# =============================================================================
# Vector Database Layer (Qdrant alternative)
# =============================================================================

class VectorIndex:
    """Simple vector similarity index (Qdrant alternative)"""
    
    def __init__(self, dimension: int = 512):
        self.dimension = dimension
        self.vectors: Dict[str, List[float]] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.RLock()
    
    def add_vector(self, vector_id: str, vector: List[float], metadata: Dict[str, Any] = None):
        """Add vector to index"""
        if len(vector) != self.dimension:
            raise ValueError(f"Vector dimension {len(vector)} != {self.dimension}")
        
        with self.lock:
            self.vectors[vector_id] = vector
            self.metadata[vector_id] = metadata or {}
    
    def similarity_search(self, query_vector: List[float], top_k: int = 10) -> List[Tuple[str, float]]:
        """Find most similar vectors using cosine similarity"""
        if len(query_vector) != self.dimension:
            raise ValueError(f"Query vector dimension {len(query_vector)} != {self.dimension}")
        
        results = []
        
        for vector_id, vector in self.vectors.items():
            similarity = self._cosine_similarity(query_vector, vector)
            results.append((vector_id, similarity))
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = sum(x * y for x, y in zip(a, b))
        magnitude_a = sum(x * x for x in a) ** 0.5
        magnitude_b = sum(x * x for x in b) ** 0.5
        
        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0
        
        return dot_product / (magnitude_a * magnitude_b)

# =============================================================================
# Hybrid Database Manager
# =============================================================================

class HybridContextDatabase:
    """Unified interface for hybrid database system"""
    
    def __init__(self, base_path: str = "extensions/context-engineering/data"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.sqlite_db = str(self.base_path / "context_engineering.db")
        self.graph_db = JSONGraphDatabase(str(self.base_path / "context_graph.json"))
        self.vector_index = VectorIndex(dimension=512)
        
        # In-memory caches
        self.consciousness_cache: Dict[str, Tuple[str, float]] = {}
        self.stream_cache: Dict[str, Dict[str, Any]] = {}
        self.basin_cache: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self.query_stats = defaultdict(int)
        self.last_save = datetime.now()
        
        # Initialize SQLite
        self._init_sqlite()
        
        logger.info(f"Hybrid database initialized at {self.base_path}")
    
    def _init_sqlite(self):
        """Initialize SQLite database"""
        with sqlite3.connect(self.sqlite_db) as conn:
            # Core tables
            conn.execute("""
                CREATE TABLE IF NOT EXISTS architectures (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    program TEXT,
                    result TEXT,
                    motivation TEXT,
                    analysis TEXT,
                    consciousness_level TEXT,
                    consciousness_score REAL,
                    created_at TEXT NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS context_streams (
                    id TEXT PRIMARY KEY,
                    source_architectures TEXT,
                    flow_state TEXT,
                    flow_velocity REAL,
                    information_density REAL,
                    created_at TEXT NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS attractor_basins (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    center_architecture TEXT,
                    radius REAL,
                    attraction_strength REAL,
                    contained_architectures TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            
            # Performance indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_arch_consciousness ON architectures(consciousness_level, consciousness_score)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_streams_state ON context_streams(flow_state, flow_velocity)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_basins_strength ON attractor_basins(attraction_strength)")
            
            conn.commit()
    
    @contextmanager
    def get_sqlite_connection(self):
        """Get SQLite connection with automatic cleanup"""
        conn = sqlite3.connect(self.sqlite_db)
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        try:
            yield conn
        finally:
            conn.close()
    
    def store_architecture(self, arch_data: Dict[str, Any], consciousness_level: str, consciousness_score: float):
        """Store architecture with consciousness analysis"""
        arch_id = arch_data.get('name', str(hash(str(arch_data))))
        
        # Store in SQLite
        with self.get_sqlite_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO architectures 
                (id, name, program, result, motivation, analysis, consciousness_level, consciousness_score, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                arch_id,
                arch_data.get('name', ''),
                arch_data.get('program', ''),
                json.dumps(arch_data.get('result', {})),
                arch_data.get('motivation', ''),
                arch_data.get('analysis', ''),
                consciousness_level,
                consciousness_score,
                datetime.now().isoformat()
            ))
            conn.commit()
        
        # Store in graph database
        self.graph_db.add_node(arch_id, 'architecture', {
            'name': arch_data.get('name', ''),
            'consciousness_level': consciousness_level,
            'consciousness_score': consciousness_score,
            'performance': self._extract_performance_score(arch_data)
        })
        
        # Update cache
        self.consciousness_cache[arch_id] = (consciousness_level, consciousness_score)
        
        logger.debug(f"Stored architecture {arch_id} with consciousness {consciousness_level}")
    
    def create_architecture_relationship(self, source_arch: str, target_arch: str, 
                                       relationship_type: str, strength: float):
        """Create relationship between architectures"""
        try:
            self.graph_db.add_edge(source_arch, target_arch, relationship_type, 
                                 {'strength': strength}, weight=strength)
            logger.debug(f"Created {relationship_type} relationship: {source_arch} -> {target_arch}")
        except ValueError as e:
            logger.warning(f"Could not create relationship: {e}")
    
    def find_consciousness_evolution_path(self, start_arch: str, target_consciousness: str) -> List[str]:
        """Find path to architectures with target consciousness level"""
        # Find architectures with target consciousness
        target_nodes = self.graph_db.query_nodes('architecture', {'consciousness_level': target_consciousness})
        
        if not target_nodes:
            return []
        
        # Find shortest path to any target
        for target_node in target_nodes:
            path = self.graph_db.find_path(start_arch, target_node.id)
            if path:
                return path
        
        return []
    
    def get_consciousness_distribution(self) -> Dict[str, int]:
        """Get distribution of consciousness levels"""
        with self.get_sqlite_connection() as conn:
            cursor = conn.execute("""
                SELECT consciousness_level, COUNT(*) as count 
                FROM architectures 
                GROUP BY consciousness_level
            """)
            return dict(cursor.fetchall())
    
    def get_top_performing_architectures(self, consciousness_level: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top performing architectures, optionally filtered by consciousness level"""
        query = "SELECT * FROM architectures"
        params = []
        
        if consciousness_level:
            query += " WHERE consciousness_level = ?"
            params.append(consciousness_level)
        
        query += " ORDER BY consciousness_score DESC LIMIT ?"
        params.append(limit)
        
        with self.get_sqlite_connection() as conn:
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
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
    
    def periodic_maintenance(self):
        """Periodic maintenance tasks"""
        now = datetime.now()
        
        # Save graph database every 5 minutes
        if now - self.last_save > timedelta(minutes=5):
            self.graph_db.save_graph()
            self.last_save = now
            
            # Log statistics
            logger.info(f"Database stats - Architectures: {len(self.consciousness_cache)}, "
                       f"Graph nodes: {len(self.graph_db.nodes)}, "
                       f"Graph edges: {len(self.graph_db.edges)}")
    
    def export_knowledge_graph(self, output_file: str = "knowledge_graph_export.json"):
        """Export complete knowledge graph for analysis"""
        export_data = {
            'metadata': {
                'exported_at': datetime.now().isoformat(),
                'architecture_count': len(self.consciousness_cache),
                'graph_nodes': len(self.graph_db.nodes),
                'graph_edges': len(self.graph_db.edges)
            },
            'consciousness_distribution': self.get_consciousness_distribution(),
            'top_architectures': self.get_top_performing_architectures(limit=20),
            'graph_structure': {
                'nodes': [node.to_dict() for node in self.graph_db.nodes.values()],
                'edges': [edge.to_dict() for edge in self.graph_db.edges]
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported knowledge graph to {output_file}")
        return output_file

# =============================================================================
# Factory and Testing
# =============================================================================

def create_hybrid_database() -> HybridContextDatabase:
    """Factory function to create hybrid database"""
    return HybridContextDatabase()

def test_hybrid_database():
    """Test the hybrid database system"""
    print("🧪 Testing Hybrid Context Database")
    
    db = create_hybrid_database()
    
    # Test architecture storage
    print("\n1. Testing architecture storage...")
    test_arch = {
        'name': 'test_linear_attention',
        'program': 'class TestLinearAttention(nn.Module): pass',
        'result': {'test': 'acc=0.85'},
        'motivation': 'testing hybrid database',
        'analysis': 'shows good integration with database'
    }
    
    db.store_architecture(test_arch, 'ACTIVE', 0.65)
    print("✅ Architecture stored")
    
    # Test relationship creation
    print("\n2. Testing relationships...")
    db.graph_db.add_node('parent_arch', 'architecture', {'name': 'parent', 'consciousness_level': 'DORMANT'})
    db.create_architecture_relationship('parent_arch', 'test_linear_attention', 'evolved_from', 0.8)
    print("✅ Relationship created")
    
    # Test queries
    print("\n3. Testing queries...")
    consciousness_dist = db.get_consciousness_distribution()
    print(f"Consciousness distribution: {consciousness_dist}")
    
    top_archs = db.get_top_performing_architectures(limit=5)
    print(f"Found {len(top_archs)} architectures")
    
    # Test path finding
    print("\n4. Testing path finding...")
    path = db.find_consciousness_evolution_path('parent_arch', 'ACTIVE')
    print(f"Evolution path: {path}")
    
    # Test export
    print("\n5. Testing export...")
    export_file = db.export_knowledge_graph()
    print(f"Exported to: {export_file}")
    
    print("\n✅ Hybrid database test completed!")

if __name__ == "__main__":
    test_hybrid_database()
