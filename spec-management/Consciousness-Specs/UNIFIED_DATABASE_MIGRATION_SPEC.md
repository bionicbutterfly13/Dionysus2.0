# Unified Database Migration Specification

**Version**: 1.0.0  
**Status**: ARCHITECTURAL DECISION - UNIFIED APPROACH  
**Last Updated**: 2025-09-22  
**Specification Type**: Database Architecture Migration  
**Development Methodology**: Spec-Driven Development

## 🎯 Executive Summary

**User Decision**: Migrate everything to our unified hybrid system to eliminate redundancy and create a single relational graph database for all concepts, contexts, and learnings.

**Rationale**: No data duplication, no redundancy, unified knowledge base accessible without crossing multiple systems.

## 🗄️ Current Database Reality Check

### **What We Actually Have**
1. **FAISS** (ASI-Arch) - Vector indexing ✅ **Keep for performance**
2. **OpenSearch** (Cognition Base) - RAG/search service ✅ **Different purpose, keep**
3. **Our Hybrid System** - SQLite + JSON graph + Vector ✅ **Keep and expand**

### **What We DON'T Have** (User was thinking of different project)
- ❌ No Redis messaging
- ❌ No PostgreSQL  
- ❌ No separate graph database (we built JSON-based)

## 🎯 Migration Strategy: Extend Our Hybrid System

### **Phase 1: Extend Hybrid System for ASI-Arch Data**

Our hybrid system currently handles:
- ✅ Consciousness detection data
- ✅ Episodic memory 
- ✅ Autobiographical events
- ✅ Context streams and attractor basins

**Need to add:**
- ❌ Neural architecture storage (currently in separate system)
- ❌ Architecture evolution sequences
- ❌ Performance metrics and evaluation data
- ❌ Candidate management

### **Enhanced Hybrid Architecture**

```python
class UnifiedASIArchDatabase(HybridContextDatabase):
    """Unified database for ALL ASI-Arch data - no redundancy"""
    
    def __init__(self, base_path: str = "data/unified_asi_arch"):
        super().__init__(base_path)
        
        # Extend SQLite schema for ASI-Arch data
        self._add_architecture_tables()
        
        # Enhance vector index for architecture similarity
        self._enhance_vector_index()
        
        # Extend graph database for architecture relationships
        self._add_architecture_graph_schema()
    
    def _add_architecture_tables(self):
        """Add ASI-Arch specific tables to SQLite"""
        with sqlite3.connect(self.sqlite_db) as conn:
            # Neural architectures table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS neural_architectures (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    program TEXT,
                    result TEXT,
                    motivation TEXT,
                    analysis TEXT,
                    performance_score REAL,
                    evaluation_metrics TEXT,
                    created_at TEXT NOT NULL,
                    
                    -- Context Engineering Integration
                    consciousness_level TEXT,
                    consciousness_score REAL,
                    episodic_memory_id TEXT,
                    attractor_basin_id TEXT,
                    
                    -- Relational connections
                    parent_architecture_ids TEXT,  -- JSON array
                    child_architecture_ids TEXT,   -- JSON array
                    similar_architecture_ids TEXT  -- JSON array
                )
            """)
            
            # Architecture evolution sequences
            conn.execute("""
                CREATE TABLE IF NOT EXISTS evolution_sequences (
                    id TEXT PRIMARY KEY,
                    sequence_name TEXT,
                    architecture_ids TEXT,  -- JSON array of architecture IDs
                    evolution_strategy TEXT,
                    performance_trajectory TEXT,  -- JSON array of performance over time
                    episode_boundaries TEXT,  -- JSON array of episode boundary positions
                    narrative_summary TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            
            # Unified evaluation metrics
            conn.execute("""
                CREATE TABLE IF NOT EXISTS evaluation_metrics (
                    id TEXT PRIMARY KEY,
                    architecture_id TEXT,
                    metric_name TEXT,
                    metric_value REAL,
                    evaluation_context TEXT,
                    timestamp TEXT,
                    FOREIGN KEY (architecture_id) REFERENCES neural_architectures (id)
                )
            """)
            
            # Architecture relationships (many-to-many)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS architecture_relationships (
                    id TEXT PRIMARY KEY,
                    source_architecture_id TEXT,
                    target_architecture_id TEXT,
                    relationship_type TEXT,  -- 'parent', 'child', 'similar', 'evolved_from'
                    relationship_strength REAL,
                    created_at TEXT,
                    FOREIGN KEY (source_architecture_id) REFERENCES neural_architectures (id),
                    FOREIGN KEY (target_architecture_id) REFERENCES neural_architectures (id)
                )
            """)
```

## 🔄 Migration Implementation

### **Step 1: Schema Extension**
```python
async def migrate_mongodb_to_unified_system():
    """Migrate existing MongoDB data to unified system"""
    
    # Initialize unified database
    unified_db = UnifiedASIArchDatabase()
    
    # Connect to existing MongoDB (if available)
    try:
        mongo_db = MongoDatabase()
        existing_architectures = mongo_db.get_all_architectures()
        
        # Migrate each architecture
        for arch_data in existing_architectures:
            await unified_db.store_architecture_with_context(
                architecture_data=arch_data,
                consciousness_analysis=await analyze_consciousness(arch_data),
                episodic_context=await generate_episodic_context(arch_data)
            )
        
        print(f"✅ Migrated {len(existing_architectures)} architectures")
        
    except Exception as e:
        print(f"⚠️ No existing MongoDB data found: {e}")
        print("✅ Starting with clean unified system")
```

### **Step 2: Unified Data Model**
```python
@dataclass
class UnifiedArchitecture:
    """Single unified representation of neural architecture + context"""
    
    # Core Architecture Data (from MongoDB)
    id: str
    name: str
    program: str
    result: str
    motivation: str
    analysis: str
    performance_score: float
    
    # Context Engineering Data (from our system)
    consciousness_level: ConsciousnessLevel
    consciousness_score: float
    episodic_memories: List[ArchitectureEpisode]
    attractor_basin: Optional[AttractorBasin]
    context_streams: List[ContextStream]
    
    # Relational Graph Data
    parent_architectures: List[str]  # Architecture IDs
    child_architectures: List[str]
    similar_architectures: List[str]
    evolution_path: List[str]  # Sequence of evolution
    
    # Unified Vector Representation
    embedding_vector: List[float]  # Combined architecture + context embedding
```

### **Step 3: Unified Query Interface**
```python
class UnifiedQueryInterface:
    """Single interface for all ASI-Arch queries"""
    
    async def find_architectures_by_consciousness(self, level: ConsciousnessLevel):
        """Find architectures by consciousness level"""
        # Single query across unified data
    
    async def get_evolution_episode(self, episode_id: str):
        """Get complete evolution episode with all context"""
        # Returns: architectures + consciousness + episodic narrative
    
    async def find_similar_architectures(self, query_vector: List[float]):
        """Find similar architectures using unified vector index"""
        # Searches both architecture similarity AND consciousness similarity
    
    async def get_architecture_genealogy(self, arch_id: str):
        """Get complete family tree of architecture evolution"""
        # Returns: parents, children, evolution path, episodic story
```

## 🎯 Benefits of Unified Approach

### **1. No Data Duplication**
- ✅ Single storage for each architecture
- ✅ Single vector representation (architecture + context)
- ✅ Single source of truth for all relationships

### **2. Unified Knowledge Base**
- ✅ All learnings in one relational graph
- ✅ Architecture evolution + consciousness + episodes in same query
- ✅ No crossing multiple systems

### **3. Clean Architecture**
- ✅ One database system to maintain
- ✅ One query interface
- ✅ One data model

### **4. Enhanced Capabilities**
- ✅ Architecture similarity includes consciousness similarity
- ✅ Evolution episodes include full context
- ✅ Genealogy tracking across all dimensions

## 🧪 Migration Testing Strategy

### **Test 1: Data Integrity**
```python
async def test_migration_integrity():
    """Ensure no data loss during migration"""
    
    # Count records before migration
    mongo_count = mongo_db.count_architectures()
    
    # Perform migration
    await migrate_mongodb_to_unified_system()
    
    # Count records after migration
    unified_count = unified_db.count_architectures()
    
    assert mongo_count == unified_count, "Data loss during migration"
```

### **Test 2: Query Equivalence**
```python
async def test_query_equivalence():
    """Ensure unified queries return same results"""
    
    # Test architecture retrieval
    mongo_arch = mongo_db.get_architecture(arch_id)
    unified_arch = unified_db.get_architecture(arch_id)
    
    assert mongo_arch.program == unified_arch.program
    assert mongo_arch.performance_score == unified_arch.performance_score
    
    # Plus: unified system has additional context data
    assert unified_arch.consciousness_level is not None
    assert len(unified_arch.episodic_memories) >= 0
```

## 🚀 Implementation Timeline

### **Week 1: Schema Extension**
- [ ] Extend hybrid database with ASI-Arch tables
- [ ] Create unified data models
- [ ] Test schema with synthetic data

### **Week 2: Migration Implementation**
- [ ] Build migration scripts
- [ ] Test with existing data (if any)
- [ ] Verify data integrity

### **Week 3: Unified Query Interface**
- [ ] Implement unified query methods
- [ ] Test cross-system queries
- [ ] Performance optimization

### **Week 4: Integration Testing**
- [ ] Full pipeline testing
- [ ] Performance benchmarking
- [ ] Documentation and deployment

## ✅ Success Criteria

### **Migration Success**
- [ ] All ASI-Arch data in unified system
- [ ] No data duplication or redundancy
- [ ] Single query interface for all data
- [ ] Enhanced capabilities (consciousness + architecture)

### **Performance Success**
- [ ] Query performance ≥ original systems
- [ ] Memory usage reasonable for development
- [ ] Scalable architecture for future growth

---

**Status**: ✅ **UNIFIED APPROACH APPROVED**  
**Next Action**: Begin schema extension implementation  
**Timeline**: 4 weeks to complete migration  
**Benefit**: Single relational graph database for all concepts and learnings
