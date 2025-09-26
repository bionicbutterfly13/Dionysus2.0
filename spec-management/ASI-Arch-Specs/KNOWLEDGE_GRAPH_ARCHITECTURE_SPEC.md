# Knowledge Graph Architecture Specification - AutoSchemaKG Integration

**Version**: 1.0.0  
**Status**: ARCHITECTURAL REVISION - NEO4J + VECTOR APPROACH  
**Last Updated**: 2025-09-22  
**Specification Type**: Knowledge Graph Architecture  
**Development Methodology**: Spec-Driven Development  
**Reference**: [AutoSchemaKG Framework](https://github.com/HKUST-KnowComp/AutoSchemaKG.git)

## 🎯 Executive Summary

**User Insight**: "I thought we had a Neo4j database. I definitely trust the vector database and the Neo4j database for their ability to contain relational information, for the graph's ability to do graph things."

**Revised Architecture**: Use **Neo4j for knowledge graphs** + **Vector database for embeddings** + **AutoSchemaKG for automatic schema construction**, following proven patterns from the AutoSchemaKG project.

## 🏗️ Recommended Architecture: Neo4j + Vector Hybrid

### **Why This Is The Right Approach**

Based on AutoSchemaKG's success and your intuition:

1. **Neo4j excels at graph relationships** - Architecture evolution, consciousness patterns, episodic connections
2. **Vector databases excel at similarity** - Architecture embeddings, consciousness similarity, episodic retrieval  
3. **AutoSchemaKG provides proven framework** - Automatic knowledge graph construction from our development data
4. **Specialized tools for specialized purposes** - Each system does what it's best at

### **Architecture Components**

```
┌─────────────────────────────────────────────────────────────┐
│                ASI-Arch Context Flow                        │
│                Knowledge Graph System                       │
├─────────────────────────────────────────────────────────────┤
│  Neo4j Knowledge Graph                                      │
│  ├─ Architecture Evolution Graphs                          │
│  ├─ Consciousness Development Networks                      │
│  ├─- Episodic Memory Connections                           │
│  └─- Research Paper Relationship Maps                      │
├─────────────────────────────────────────────────────────────┤
│  Vector Database (Qdrant/FAISS)                            │
│  ├─ Architecture Embeddings                                │
│  ├─ Consciousness State Vectors                            │
│  ├─ Episode Similarity Search                              │
│  └─ Research Paper Embeddings                              │
├─────────────────────────────────────────────────────────────┤
│  AutoSchemaKG Integration                                   │
│  ├─ Automatic Schema Generation                            │
│  ├─ Triple Extraction from Development Data                │
│  ├─ Concept Generation                                     │
│  └─ Knowledge Graph Construction Pipeline                  │
├─────────────────────────────────────────────────────────────┤
│  SQLite (Lightweight Data)                                 │
│  ├─ System Metadata                                        │
│  ├─ Configuration                                          │
│  └─ Development Logs                                       │
└─────────────────────────────────────────────────────────────┘
```

## 🔄 Integration with AutoSchemaKG

### **AutoSchemaKG Framework Benefits**

From the [AutoSchemaKG repository](https://github.com/HKUST-KnowComp/AutoSchemaKG.git):

1. **Automatic Knowledge Graph Construction** - No manual schema design needed
2. **Triple Extraction from Text** - Converts our development conversations into graph triples
3. **Dynamic Schema Induction** - Evolves schema as our system grows
4. **Neo4j Integration** - Direct support for Neo4j graph databases
5. **RAG Support** - Built-in retrieval augmented generation

### **Implementation Plan**

```python
# Based on AutoSchemaKG patterns
from atlas_rag.kg_construction.triple_extraction import KnowledgeGraphExtractor
from atlas_rag.kg_construction.triple_config import ProcessingConfig
from atlas_rag.llm_generator import LLMGenerator

class ASIArchKnowledgeGraph:
    """ASI-Arch knowledge graph using AutoSchemaKG framework"""
    
    def __init__(self):
        # Neo4j for graph relationships
        self.neo4j_client = Neo4jClient()
        
        # Vector database for similarity search
        self.vector_db = QdrantClient()  # or FAISS
        
        # AutoSchemaKG for automatic construction
        self.kg_extractor = self._setup_kg_extractor()
        
        # Our existing systems for specialized data
        self.autobiographical_memory = AutobiographicalMemorySystem()
    
    def _setup_kg_extractor(self):
        """Setup AutoSchemaKG extractor for our development data"""
        config = ProcessingConfig(
            data_directory="development_conversations",
            output_directory="knowledge_graphs/asi_arch",
            batch_size_triple=3,
            batch_size_concept=16,
            remove_doc_spaces=True
        )
        
        return KnowledgeGraphExtractor(
            model=LLMGenerator(client, model_name="gpt-4"),
            config=config
        )
    
    async def construct_development_knowledge_graph(self):
        """Build knowledge graph from our development process"""
        
        # 1. Extract triples from development conversations
        await self.kg_extractor.run_extraction()
        
        # 2. Generate concepts automatically
        await self.kg_extractor.generate_concept_csv()
        
        # 3. Load into Neo4j
        await self._load_into_neo4j()
        
        # 4. Create vector embeddings
        await self._create_vector_embeddings()
    
    async def store_architecture_with_context(self, architecture_data):
        """Store architecture in both Neo4j and vector database"""
        
        # Store in Neo4j as graph nodes/relationships
        await self.neo4j_client.create_architecture_node(architecture_data)
        
        # Store embeddings in vector database
        embedding = await self._generate_embedding(architecture_data)
        await self.vector_db.upsert_vector(architecture_data.id, embedding)
        
        # Update autobiographical memory
        await self.autobiographical_memory.capture_architecture_event(architecture_data)
```

## 📊 Data Model Design

### **Neo4j Graph Schema**

```cypher
// Architecture nodes
CREATE (a:Architecture {
    id: "arch_001",
    name: "transformer_variant_1", 
    performance_score: 0.85,
    consciousness_level: "SELF_AWARE",
    created_at: datetime()
})

// Evolution relationships  
CREATE (parent:Architecture)-[:EVOLVED_TO]->(child:Architecture)

// Consciousness relationships
CREATE (arch:Architecture)-[:HAS_CONSCIOUSNESS]->(consciousness:ConsciousnessState)

// Episodic memory relationships
CREATE (arch:Architecture)-[:PART_OF_EPISODE]->(episode:Episode)

// Research paper relationships
CREATE (arch:Architecture)-[:INSPIRED_BY]->(paper:ResearchPaper)

// Development event relationships  
CREATE (arch:Architecture)-[:CREATED_DURING]->(event:DevelopmentEvent)
```

### **Vector Database Schema**

```python
# Architecture embeddings
{
    "id": "arch_001",
    "vector": [0.1, 0.2, ...],  # 1536-dimensional embedding
    "metadata": {
        "architecture_type": "transformer",
        "consciousness_level": "SELF_AWARE", 
        "performance_score": 0.85,
        "episode_id": "episode_001"
    }
}

# Episode embeddings
{
    "id": "episode_001", 
    "vector": [0.3, 0.4, ...],
    "metadata": {
        "episode_type": "breakthrough_moment",
        "architectures_involved": ["arch_001", "arch_002"],
        "development_phase": "consciousness_emergence"
    }
}
```

## 🚀 Implementation Phases

### **Phase 1: Neo4j + Vector Setup (Week 1)**
- [ ] Install Neo4j and Qdrant/FAISS
- [ ] Install AutoSchemaKG framework (`pip install atlas-rag`)
- [ ] Create basic graph schema
- [ ] Test with synthetic data

### **Phase 2: AutoSchemaKG Integration (Week 2)**  
- [ ] Configure AutoSchemaKG for our development data
- [ ] Extract triples from our conversations and specs
- [ ] Generate automatic schema from our domain
- [ ] Load into Neo4j

### **Phase 3: Hybrid Queries (Week 3)**
- [ ] Implement graph + vector hybrid queries
- [ ] Create unified query interface
- [ ] Test cross-system relationships

### **Phase 4: Development Data Migration (Week 4)**
- [ ] Migrate autobiographical memory to graph format
- [ ] Create architecture evolution graphs
- [ ] Build consciousness development networks

## 🎯 Benefits of This Architecture

### **Neo4j Strengths**
- ✅ **Excellent at relationships** - Architecture evolution paths, consciousness networks
- ✅ **Graph traversal** - Find related architectures, trace development paths
- ✅ **Complex queries** - Multi-hop relationships, pattern matching
- ✅ **Visualization** - Built-in graph visualization tools

### **Vector Database Strengths**  
- ✅ **Similarity search** - Find similar architectures, episodes, consciousness states
- ✅ **High performance** - Fast nearest neighbor search
- ✅ **Scalability** - Handle millions of embeddings efficiently

### **AutoSchemaKG Benefits**
- ✅ **Automatic construction** - No manual schema design
- ✅ **Dynamic evolution** - Schema grows with our system
- ✅ **Proven framework** - 492 stars, actively maintained
- ✅ **RAG integration** - Built-in retrieval capabilities

## 🧪 Testing Strategy

### **Graph + Vector Integration Tests**
```python
async def test_hybrid_architecture_query():
    """Test querying across Neo4j and vector database"""
    
    # Find similar architectures using vector search
    similar_vectors = await vector_db.search(query_vector, top_k=10)
    arch_ids = [v.metadata['id'] for v in similar_vectors]
    
    # Get evolution paths using Neo4j
    cypher = """
    MATCH (a:Architecture)-[:EVOLVED_TO*1..3]->(descendant)
    WHERE a.id IN $arch_ids
    RETURN a, descendant
    """
    evolution_paths = await neo4j_client.run(cypher, arch_ids=arch_ids)
    
    # Combine results
    return {
        'similar_architectures': similar_vectors,
        'evolution_paths': evolution_paths
    }
```

## ✅ Success Criteria

### **Integration Success**
- [ ] Neo4j stores architecture relationships correctly
- [ ] Vector database enables fast similarity search
- [ ] AutoSchemaKG automatically constructs domain schema
- [ ] Hybrid queries work across both systems
- [ ] Development data flows into knowledge graph

### **Performance Success**
- [ ] Graph queries < 100ms for typical relationships
- [ ] Vector similarity search < 50ms 
- [ ] Knowledge graph construction automated
- [ ] System scales to 10,000+ architectures

---

**Status**: ✅ **REVISED ARCHITECTURE APPROVED**  
**Approach**: Neo4j + Vector + AutoSchemaKG  
**Reference**: [AutoSchemaKG Framework](https://github.com/HKUST-KnowComp/AutoSchemaKG.git)  
**Rationale**: Specialized tools for specialized purposes, proven framework  
**Next Action**: Begin Neo4j + AutoSchemaKG setup
