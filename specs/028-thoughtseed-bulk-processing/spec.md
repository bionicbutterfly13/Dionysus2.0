# Spec 028: ThoughtSeed Generation During Bulk Document Upload

**Status**: DRAFT
**Priority**: HIGH
**Dependencies**: 021 (Daedalus), 005 (ThoughtSeeds), 027 (Basin Strengthening)
**Created**: 2025-10-01

## Overview

Implement automatic ThoughtSeed generation during bulk document processing where each concept creates a ThoughtSeed that propagates through the consciousness system. ThoughtSeeds enable cross-document concept linking, pattern discovery, and emergent knowledge synthesis.

## Problem Statement

Currently, document processing extracts concepts and relationships but doesn't create ThoughtSeeds for cross-document propagation. When processing 100+ research papers:

1. **No cross-document linking**: Paper #15's concepts don't connect to Paper #3's related concepts
2. **No emergent patterns**: System can't discover that 10 papers all explore the same underlying principle
3. **No concept evolution tracking**: Can't trace how "neural architecture search" evolved from "AutoML"
4. **No consciousness integration**: Concepts stay isolated in Neo4j without entering active inference loops

ThoughtSeeds solve this by:
- Propagating concepts through the consciousness system
- Creating cross-basin connections
- Triggering pattern emergence detection
- Enabling concept evolution tracking

## Agent Role Definition (Following User Guidelines)

### Specialized Agent Roles for ThoughtSeed Processing

#### 1. **ThoughtSeed Generation Agent**
**Responsibility**: Create ThoughtSeeds from extracted concepts
**Input**: Concepts from document processing
**Output**: ThoughtSeed objects with propagation parameters
**Tools**: Semantic encoding, basin context lookup

#### 2. **ThoughtSeed Propagation Agent**
**Responsibility**: Route ThoughtSeeds through consciousness system
**Input**: Generated ThoughtSeeds
**Output**: Propagation paths and basin assignments
**Tools**: Redis state management, active inference engine

#### 3. **Basin Integration Agent**
**Responsibility**: Integrate ThoughtSeeds into attractor basins
**Input**: ThoughtSeeds with target basins
**Output**: Updated basins, strengthening events
**Tools**: `AttractorBasinManager`, frequency tracking

#### 4. **Cross-Document Linking Agent**
**Responsibility**: Find connections between ThoughtSeeds from different documents
**Input**: Active ThoughtSeeds in Redis
**Output**: RELATED_THOUGHTSEED relationships in Neo4j
**Tools**: Semantic similarity, temporal proximity analysis

#### 5. **Pattern Emergence Agent**
**Responsibility**: Detect when multiple ThoughtSeeds form coherent patterns
**Input**: ThoughtSeed clusters in basins
**Output**: Emergent pattern records in Neo4j
**Tools**: Clustering algorithms, narrative extraction

#### 6. **ThoughtSeed Validation Agent**
**Responsibility**: Quality-check ThoughtSeeds before propagation
**Input**: Raw ThoughtSeeds
**Output**: Validated, enriched ThoughtSeeds
**Tools**: Coherence scoring, domain validation

**Best Practices Applied**:
- ✅ **Specialization**: Each agent has one clear task
- ✅ **Modularity**: Agents can be replaced/upgraded independently
- ✅ **Bidirectional Graph Interaction**: All agents read/write Neo4j
- ✅ **Monitoring**: Each agent logs performance metrics
- ✅ **Integration Layer**: LangGraph orchestrates agent workflow

## Requirements

### Functional Requirements

#### FR1: ThoughtSeed Generation from Concepts
**Description**: Every extracted concept generates a ThoughtSeed
**Acceptance Criteria**:
- [ ] ThoughtSeed created for each concept with confidence >0.7
- [ ] ThoughtSeed includes: concept text, source document, timestamp, semantic encoding
- [ ] ThoughtSeed stored in Redis with 24-hour TTL
- [ ] ThoughtSeed stored in Neo4j with GENERATED_THOUGHTSEED relationship

#### FR2: ThoughtSeed Propagation Through Basins
**Description**: ThoughtSeeds propagate to related basins
**Acceptance Criteria**:
- [ ] ThoughtSeed integrated into matching basin (>0.7 similarity)
- [ ] ThoughtSeed triggers basin strengthening
- [ ] Propagation path recorded: document → concept → thoughtseed → basin
- [ ] Multiple basins can attract same ThoughtSeed (max 3)

#### FR3: Cross-Document Concept Linking
**Description**: Link related concepts from different papers via ThoughtSeeds
**Acceptance Criteria**:
- [ ] ThoughtSeeds from different documents with >0.8 similarity get RELATED_THOUGHTSEED link
- [ ] Links stored in Neo4j with similarity score and discovery timestamp
- [ ] Cross-document queries: "Find all papers discussing concept X"
- [ ] Temporal tracking: "How did this concept evolve across papers?"

#### FR4: Emergent Pattern Detection
**Description**: Detect when multiple papers converge on same idea
**Acceptance Criteria**:
- [ ] When 5+ ThoughtSeeds cluster in same basin → create EMERGENT_PATTERN node
- [ ] Pattern node links to all contributing documents
- [ ] Pattern node includes: central concept, strength, contributing papers, discovery date
- [ ] Patterns trigger curiosity agents for deeper investigation

#### FR5: ThoughtSeed Lifecycle Management
**Description**: Manage ThoughtSeed creation, propagation, decay
**Acceptance Criteria**:
- [ ] Active ThoughtSeeds in Redis (24-hour TTL)
- [ ] Archived ThoughtSeeds in Neo4j (permanent)
- [ ] Propagation metrics logged: hops, basin visits, connection count
- [ ] Dead ThoughtSeeds (no basin match) archived with reason

### Non-Functional Requirements

#### NFR1: Performance
- ThoughtSeed generation: <100ms per concept
- Bulk processing: 100 papers should generate 500-1000 ThoughtSeeds in <5 minutes
- Cross-document linking: Process 1000 ThoughtSeeds in <30 seconds

#### NFR2: Scalability
- Support 10,000+ active ThoughtSeeds in Redis
- Neo4j should handle 100,000+ ThoughtSeed nodes
- Linking algorithm should scale O(n log n)

#### NFR3: Quality
- >90% of ThoughtSeeds should match at least one basin
- Cross-document links should have >0.8 similarity
- Emergent patterns should be validated by human review (optional)

## Technical Design

### Architecture

```
Bulk Document Upload with ThoughtSeed Processing:
┌───────────────────────────────────────────────────────┐
│  Upload 100 Research Papers                           │
└───────────────┬───────────────────────────────────────┘
                │
                ▼
    ┌──────────────────────────────┐
    │  Daedalus Gateway            │
    │  - Receive perceptual info   │
    └──────────┬───────────────────┘
               │
               ▼
    ┌──────────────────────────────────────┐
    │  Document Processing Graph           │
    │  (LangGraph Workflow)                │
    │                                      │
    │  1. Extract Concepts                 │
    │  2. Extract Relationships (LLM)      │
    │  3. Generate ThoughtSeeds ← NEW      │
    │  4. Propagate ThoughtSeeds ← NEW     │
    │  5. Store to Neo4j                   │
    └──────────┬───────────────────────────┘
               │
               ▼
    ┌──────────────────────────────────────┐
    │  ThoughtSeed Generation Agent        │
    │  For each concept:                   │
    │  - Create ThoughtSeed object         │
    │  - Semantic encoding                 │
    │  - Assign propagation params         │
    └──────────┬───────────────────────────┘
               │
               ▼
    ┌──────────────────────────────────────┐
    │  Redis (Active ThoughtSeeds)         │
    │  Key: thoughtseed:{ts_id}            │
    │  TTL: 24 hours                       │
    │                                      │
    │  {                                   │
    │    "ts_id": "ts_001",                │
    │    "concept": "neural arch search",  │
    │    "source_doc": "doc_123",          │
    │    "encoding": [0.1, 0.5, ...],      │
    │    "basins": ["basin_A", "basin_B"], │
    │    "created": "2025-10-01T10:00"     │
    │  }                                   │
    └──────────┬───────────────────────────┘
               │
               ▼
    ┌──────────────────────────────────────┐
    │  Basin Integration Agent             │
    │  - Find matching basins (>0.7 sim)   │
    │  - Strengthen basins                 │
    │  - Record basin assignments          │
    └──────────┬───────────────────────────┘
               │
               ▼
    ┌──────────────────────────────────────┐
    │  AttractorBasinManager               │
    │  Basin: "neural_arch_search"         │
    │  - ThoughtSeeds: [ts_001, ts_045...] │
    │  - Strength: 1.8 (strengthened)      │
    │  - Activation: 9                     │
    └──────────┬───────────────────────────┘
               │
               ▼
    ┌──────────────────────────────────────┐
    │  Cross-Document Linking Agent        │
    │  - Compare ThoughtSeeds from diff docs│
    │  - Create RELATED_THOUGHTSEED links  │
    │  - Detect temporal evolution         │
    └──────────┬───────────────────────────┘
               │
               ▼
    ┌──────────────────────────────────────┐
    │  Neo4j Knowledge Graph               │
    │                                      │
    │  (Document)-[:GENERATED_THOUGHTSEED]→(ThoughtSeed)
    │  (ThoughtSeed)-[:RELATED_THOUGHTSEED]→(ThoughtSeed)
    │  (ThoughtSeed)-[:ATTRACTED_TO]→(Basin)
    │  (Basin)-[:FORMS_PATTERN]→(EmergentPattern)
    │  (EmergentPattern)-[:DISCOVERED_IN]→(Document)
    └──────────────────────────────────────┘
```

### Data Model

#### ThoughtSeed Object (Redis)
```python
@dataclass
class ThoughtSeed:
    ts_id: str                              # Unique ID
    concept: str                            # Original concept text
    source_document: str                    # Document ID
    semantic_encoding: List[float]          # Vector embedding
    target_basins: List[str] = []           # Assigned basins
    propagation_hops: int = 0               # How many basins visited
    created_at: str                         # Timestamp
    ttl: int = 86400                        # 24 hours in seconds
    status: str = "active"                  # active/propagating/archived/dead
    related_thoughtseeds: List[str] = []    # Cross-document links
    emergence_patterns: List[str] = []      # Patterns this contributes to
```

#### Neo4j Schema Extensions
```cypher
// ThoughtSeed node
CREATE (ts:ThoughtSeed {
  ts_id: "ts_12345",
  concept: "neural architecture search",
  source_document: "doc_nas_2024.pdf",
  created_at: "2025-10-01T10:30:00",
  status: "archived",
  propagation_hops: 3,
  basin_count: 2
})

// Document generated ThoughtSeed
CREATE (doc:Document)-[:GENERATED_THOUGHTSEED {
  timestamp: "2025-10-01T10:30:00",
  concept_confidence: 0.95
}]->(ts:ThoughtSeed)

// ThoughtSeed attracted to Basin
CREATE (ts:ThoughtSeed)-[:ATTRACTED_TO {
  similarity: 0.87,
  strength_contribution: 0.2,
  timestamp: "2025-10-01T10:31:00"
}]->(basin:AttractorBasin)

// Cross-document ThoughtSeed linking
CREATE (ts1:ThoughtSeed)-[:RELATED_THOUGHTSEED {
  similarity: 0.92,
  temporal_distance_days: 15,
  discovered_at: "2025-10-01T11:00:00"
}]->(ts2:ThoughtSeed)

// Emergent Pattern
CREATE (pattern:EmergentPattern {
  pattern_id: "pattern_nas_optimization",
  central_concept: "neural architecture optimization",
  strength: 1.8,
  contributing_documents: 8,
  discovered_at: "2025-10-01T12:00:00"
})

CREATE (basin:AttractorBasin)-[:FORMS_PATTERN]->(pattern:EmergentPattern)
CREATE (pattern:EmergentPattern)-[:DISCOVERED_IN]->(doc:Document)
```

### Agent Implementation

#### ThoughtSeed Generation Agent
```python
class ThoughtSeedGenerationAgent:
    """Generate ThoughtSeeds from extracted concepts"""

    def __init__(self, redis_client, neo4j_schema):
        self.redis = redis_client
        self.neo4j = neo4j_schema

    def generate_thoughtseeds(self, concepts: List[str], document_id: str) -> List[ThoughtSeed]:
        """Create ThoughtSeeds for all concepts"""
        thoughtseeds = []

        for concept in concepts:
            # Create ThoughtSeed
            ts = ThoughtSeed(
                ts_id=f"ts_{uuid.uuid4().hex[:12]}",
                concept=concept,
                source_document=document_id,
                semantic_encoding=self._encode_concept(concept),
                created_at=datetime.now().isoformat(),
                status="active"
            )

            # Store in Redis (24-hour TTL)
            self.redis.setex(
                f"thoughtseed:{ts.ts_id}",
                ts.ttl,
                json.dumps(asdict(ts))
            )

            # Store in Neo4j (permanent archive)
            self.neo4j.create_thoughtseed_node(ts, document_id)

            thoughtseeds.append(ts)
            logger.info(f"🌱 Generated ThoughtSeed: {ts.ts_id} for '{concept}'")

        return thoughtseeds

    def _encode_concept(self, concept: str) -> List[float]:
        """Create semantic embedding for concept"""
        # Use simple hash-based encoding for now
        # In production, use sentence-transformers or OpenAI embeddings
        import hashlib
        hash_bytes = hashlib.sha256(concept.encode()).digest()
        encoding = [b / 255.0 for b in hash_bytes[:128]]  # 128-dim vector
        return encoding
```

#### Cross-Document Linking Agent
```python
class CrossDocumentLinkingAgent:
    """Link related ThoughtSeeds from different documents"""

    def __init__(self, redis_client, neo4j_schema):
        self.redis = redis_client
        self.neo4j = neo4j_schema

    def link_thoughtseeds(self, new_thoughtseed: ThoughtSeed) -> List[Dict[str, Any]]:
        """Find and link related ThoughtSeeds from other documents"""
        links = []

        # Get all active ThoughtSeeds
        active_ts_keys = self.redis.keys("thoughtseed:*")

        for key in active_ts_keys:
            ts_data = json.loads(self.redis.get(key))

            # Skip same document
            if ts_data["source_document"] == new_thoughtseed.source_document:
                continue

            # Calculate similarity
            similarity = self._calculate_similarity(
                new_thoughtseed.semantic_encoding,
                ts_data["semantic_encoding"]
            )

            # Create link if high similarity
            if similarity > 0.8:
                # Create relationship in Neo4j
                self.neo4j.create_thoughtseed_relationship(
                    source_id=new_thoughtseed.ts_id,
                    target_id=ts_data["ts_id"],
                    relationship_type="RELATED_THOUGHTSEED",
                    properties={"similarity": similarity}
                )

                links.append({
                    "target_ts": ts_data["ts_id"],
                    "target_concept": ts_data["concept"],
                    "target_document": ts_data["source_document"],
                    "similarity": similarity
                })

                logger.info(f"🔗 Linked {new_thoughtseed.ts_id} → {ts_data['ts_id']} "
                           f"(similarity: {similarity:.3f})")

        return links

    def _calculate_similarity(self, encoding1: List[float], encoding2: List[float]) -> float:
        """Calculate cosine similarity between encodings"""
        import numpy as np
        vec1 = np.array(encoding1)
        vec2 = np.array(encoding2)
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
```

#### Pattern Emergence Agent
```python
class PatternEmergenceAgent:
    """Detect emergent patterns from ThoughtSeed clusters"""

    def __init__(self, neo4j_schema, cognition_base):
        self.neo4j = neo4j_schema
        self.cognition = cognition_base

    def detect_emergent_patterns(self, basin_id: str, min_thoughtseeds: int = 5) -> Optional[Dict[str, Any]]:
        """Detect if basin has formed emergent pattern"""

        # Query Neo4j for ThoughtSeeds in basin
        query = """
        MATCH (basin:AttractorBasin {basin_id: $basin_id})
              <-[:ATTRACTED_TO]-(ts:ThoughtSeed)
              <-[:GENERATED_THOUGHTSEED]-(doc:Document)
        RETURN basin, collect(ts) as thoughtseeds, collect(doc) as documents
        """
        result = self.neo4j.driver.session().run(query, basin_id=basin_id).single()

        if not result:
            return None

        thoughtseeds = result["thoughtseeds"]
        documents = result["documents"]

        # Check if pattern emerged
        if len(thoughtseeds) >= min_thoughtseeds:
            # Create emergent pattern node
            pattern = {
                "pattern_id": f"pattern_{basin_id}_{int(datetime.now().timestamp())}",
                "central_concept": result["basin"]["center_concept"],
                "strength": result["basin"]["strength"],
                "contributing_documents": len(documents),
                "thoughtseed_count": len(thoughtseeds),
                "discovered_at": datetime.now().isoformat()
            }

            # Store in Neo4j
            self.neo4j.create_emergent_pattern_node(pattern, basin_id, documents)

            # Log to Cognition Base
            self.cognition.record_successful_pattern("emergent_patterns", pattern)

            logger.info(f"🌟 EMERGENT PATTERN DETECTED: '{pattern['central_concept']}' "
                       f"from {len(documents)} documents")

            return pattern

        return None
```

### Integration with Document Processing Graph

```python
def _generate_and_propagate_thoughtseeds(self, state: DocumentProcessingState):
    """New node in LangGraph workflow"""

    concepts = state["concepts"]
    document_id = state["metadata"]["document_id"]

    # 1. Generate ThoughtSeeds
    thoughtseed_agent = ThoughtSeedGenerationAgent(self.redis, self.neo4j)
    thoughtseeds = thoughtseed_agent.generate_thoughtseeds(concepts, document_id)

    # 2. Link cross-document
    linking_agent = CrossDocumentLinkingAgent(self.redis, self.neo4j)
    for ts in thoughtseeds:
        links = linking_agent.link_thoughtseeds(ts)
        ts.related_thoughtseeds = [link["target_ts"] for link in links]

    # 3. Integrate into basins
    basin_agent = BasinIntegrationAgent(self.basin_manager)
    for ts in thoughtseeds:
        basins = basin_agent.integrate_thoughtseed(ts)
        ts.target_basins = basins

        # 4. Check for emergent patterns
        pattern_agent = PatternEmergenceAgent(self.neo4j, self.cognition_base)
        for basin_id in basins:
            pattern = pattern_agent.detect_emergent_patterns(basin_id)
            if pattern:
                ts.emergence_patterns.append(pattern["pattern_id"])

    # Update state
    state["thoughtseeds_generated"] = [asdict(ts) for ts in thoughtseeds]
    state["messages"].append(f"Generated {len(thoughtseeds)} ThoughtSeeds with "
                            f"{sum(len(ts.related_thoughtseeds) for ts in thoughtseeds)} cross-document links")

    return state
```

## Test Strategy

### Unit Tests

```python
def test_thoughtseed_generation():
    """Test ThoughtSeed creation from concepts"""
    agent = ThoughtSeedGenerationAgent(redis_client, neo4j_schema)

    concepts = ["neural architecture search", "AutoML", "meta-learning"]
    thoughtseeds = agent.generate_thoughtseeds(concepts, "doc_123")

    assert len(thoughtseeds) == 3
    assert all(ts.status == "active" for ts in thoughtseeds)
    assert all(len(ts.semantic_encoding) == 128 for ts in thoughtseeds)

def test_cross_document_linking():
    """Test ThoughtSeeds link across documents"""
    agent = CrossDocumentLinkingAgent(redis_client, neo4j_schema)

    # Create two similar ThoughtSeeds from different docs
    ts1 = create_thoughtseed("neural networks", "doc_1")
    ts2 = create_thoughtseed("neural network optimization", "doc_2")

    links = agent.link_thoughtseeds(ts1)

    assert len(links) > 0
    assert links[0]["target_ts"] == ts2.ts_id
    assert links[0]["similarity"] > 0.8

def test_emergent_pattern_detection():
    """Test pattern emergence from ThoughtSeed clusters"""
    agent = PatternEmergenceAgent(neo4j_schema, cognition_base)

    # Create basin with 6 ThoughtSeeds
    basin_id = create_basin_with_thoughtseeds(count=6)

    pattern = agent.detect_emergent_patterns(basin_id, min_thoughtseeds=5)

    assert pattern is not None
    assert pattern["thoughtseed_count"] == 6
    assert pattern["pattern_id"].startswith("pattern_")
```

### Integration Tests

```python
def test_bulk_upload_thoughtseed_generation():
    """Test ThoughtSeed generation during bulk upload"""
    graph = DocumentProcessingGraph()

    # Upload 10 papers
    results = []
    for i in range(10):
        result = graph.process_document(
            content=sample_papers[i],
            filename=f"paper_{i}.pdf"
        )
        results.append(result)

    # Verify ThoughtSeeds generated
    total_thoughtseeds = sum(len(r["thoughtseeds_generated"]) for r in results)
    assert total_thoughtseeds > 50  # ~5+ concepts per paper

    # Verify cross-document links
    total_links = sum(
        len(ts["related_thoughtseeds"])
        for r in results
        for ts in r["thoughtseeds_generated"]
    )
    assert total_links > 20  # Some concepts should link

def test_pattern_emergence_across_papers():
    """Test emergent patterns form across multiple papers"""
    graph = DocumentProcessingGraph()

    # Upload 6 papers on same topic (NAS)
    for i in range(6):
        graph.process_document(
            content=nas_papers[i],
            filename=f"nas_paper_{i}.pdf"
        )

    # Check if pattern emerged
    patterns = neo4j_schema.query_emergent_patterns()

    assert len(patterns) > 0
    assert any("neural architecture" in p["central_concept"].lower() for p in patterns)
```

## Implementation Plan

### Phase 1: ThoughtSeed Generation (3-4 hours)
1. Implement `ThoughtSeedGenerationAgent`
2. Add semantic encoding (hash-based initially)
3. Redis storage with TTL
4. Neo4j archival

### Phase 2: Cross-Document Linking (3-4 hours)
1. Implement `CrossDocumentLinkingAgent`
2. Similarity calculation
3. Neo4j relationship creation
4. Temporal tracking

### Phase 3: Pattern Emergence (3-4 hours)
1. Implement `PatternEmergenceAgent`
2. Clustering detection
3. Emergent pattern node creation
4. Cognition Base integration

### Phase 4: Integration (2-3 hours)
1. Add `_generate_and_propagate_thoughtseeds` node to LangGraph
2. Wire up all agents
3. Update document processing workflow
4. Test end-to-end

### Phase 5: Testing & Documentation (2-3 hours)
1. Unit tests for all agents
2. Integration tests
3. Update documentation
4. Performance validation

**Total Estimated Time**: 13-18 hours

## Success Criteria

- [ ] ThoughtSeeds generated for all concepts (>0.7 confidence)
- [ ] Cross-document links created (>0.8 similarity)
- [ ] Emergent patterns detected (5+ ThoughtSeeds in basin)
- [ ] Redis storage with 24-hour TTL functional
- [ ] Neo4j stores all ThoughtSeed nodes and relationships
- [ ] All tests passing (unit + integration)
- [ ] Performance: 100 papers → 500-1000 ThoughtSeeds in <5 min

## References

- Spec 005: ThoughtSeed Active Inference System
- Spec 021: Daedalus Perceptual Information Gateway
- Spec 027: Basin Frequency Strengthening
- `attractor_basin_dynamics.py`: Basin integration
- `document_processing_graph.py`: Main workflow
