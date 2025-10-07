# Implementation Plan: Document Persistence & Repository

**Feature Branch**: `054-document-persistence-repository`
**Created**: 2025-10-07
**Status**: Planning Complete
**Input Spec**: [spec.md](./spec.md)

---

## Executive Summary

This plan implements a document persistence and repository system that stores Daedalus LangGraph output to Neo4j via constitutional Graph Channel. The system persists document metadata, 5-level concepts, attractor basins, and thoughtseeds while providing REST APIs for document listing and retrieval.

**Key Technical Decisions**:
- **Graph Channel Only**: All Neo4j access via DaedalusGraphChannel (Spec 040 compliance)
- **Hybrid Tier Management**: Age + access patterns determine tier transitions
- **Cold Tier Archival**: S3/filesystem for cold documents, metadata in Neo4j
- **Context Engineering**: Basin evolution tracking, field resonance storage

---

## Context Engineering Integration Strategy

### Attractor Basin Integration

**Basins Created/Modified by This Feature**:
1. **DocumentProcessing Basin**: Tracks all document processing events
2. **ConceptExtraction Basin**: Strengthens with each 5-level concept extraction
3. **ConsciousnessAnalysis Basin**: Grows with thoughtseed/basin persistence
4. **TierManagement Basin**: New basin for warm/cool/cold lifecycle

**Basin Modifications**:
- Each document persistence triggers basin influence events
- Influence types: REINFORCEMENT (similar concepts), SYNTHESIS (cross-domain), EMERGENCE (novel patterns)
- Basin strength stored in Redis with document_id references
- Historical basin evolution tracked for /basins/{id}/evolution API

### Neural Field Resonance

**Resonance Patterns Expected**:
- **Concept-to-Concept**: Field resonance scores stored as relationship properties
- **Basin-to-Document**: Activation strength from field interference patterns
- **Cross-Basin**: Emergent connections via field evolution trajectories

**Storage Schema**:
```cypher
// Resonance scores from neural field integration
(:Concept)-[:RESONATES_WITH {field_score: 0.73, field_energy: 0.85}]->(:Concept)
(:Basin)-[:FIELD_ACTIVATES {interference_pattern: "constructive", energy_delta: 0.12}]->(:Document)
```

---

## Technical Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                  Daedalus LangGraph Workflow                 │
│         (External - produces final_output JSON)              │
└─────────────────────────────────┬───────────────────────────┘
                                  │
                                  ↓ final_output
┌─────────────────────────────────────────────────────────────┐
│              DocumentRepository Service                      │
│                                                               │
│  Core Methods:                                               │
│  - persist_document(final_output, metadata)                 │
│  - get_document(document_id)                                │
│  - list_documents(page, limit, filters, sort)               │
│  - update_tier(document_id, new_tier)                       │
│                                                               │
│  Dependencies:                                               │
│  - DaedalusGraphChannel (constitutional interface)          │
│  - MultiTierMemorySystem (warm/cool/cold management)        │
│  - AttractorBasinManager (basin evolution)                  │
│  - AutoSchemaKGService (5-level concepts)                   │
└─────────────────────────────────┬───────────────────────────┘
                                  │
                                  ↓ Graph Channel operations
┌─────────────────────────────────────────────────────────────┐
│            DaedalusGraphChannel                              │
│         (from daedalus-gateway package)                      │
│                                                               │
│  Operations:                                                 │
│  - execute_write(query, params, caller_service, caller_fn)  │
│  - execute_read(query, params, caller_service, caller_fn)   │
│  - health_check()                                            │
│                                                               │
│  Features:                                                   │
│  - Automatic retry (3 attempts, exponential backoff)        │
│  - Circuit breaker (opens after 5 failures)                 │
│  - Telemetry (timing, query patterns)                       │
│  - Audit trail (caller tracking)                            │
└─────────────────────────────────┬───────────────────────────┘
                                  │
                                  ↓ Cypher queries
┌─────────────────────────────────────────────────────────────┐
│                    Neo4j Knowledge Graph                     │
│                                                               │
│  Schema:                                                     │
│  - :Document nodes (metadata + tier)                        │
│  - :Concept nodes (5 levels: atomic → narrative)            │
│  - :AttractorBasin nodes (depth, stability, strength)       │
│  - :ThoughtSeed nodes (germination_potential, resonance)    │
│                                                               │
│  Relationships:                                              │
│  - [:EXTRACTED_FROM] - Concept → Document                   │
│  - [:ATTRACTED_TO] - Basin → Document                       │
│  - [:GERMINATED_FROM] - ThoughtSeed → Document              │
│  - [:DERIVED_FROM] - Concept → Concept                      │
│  - [:RESONATES_WITH] - Concept ↔ Concept                    │
└─────────────────────────────────────────────────────────────┘
                                  ‖
                  ┌───────────────┴───────────────┐
                  ↓                               ↓
┌────────────────────────────┐  ┌─────────────────────────────┐
│  Warm Tier (Neo4j)         │  │  Cold Tier (S3/Filesystem)  │
│  - Recent documents         │  │  - Archived documents        │
│  - Frequently accessed      │  │  - Metadata-only in Neo4j    │
│  - Full graph in memory     │  │  - Slower retrieval          │
└────────────────────────────┘  └─────────────────────────────┘
```

### Data Flow: Document Upload → Persistence

```
1. User uploads PDF
   ↓
2. Daedalus workflow processes document
   ├─ Extract & Process (SurfSense patterns)
   ├─ Generate Research Plan (ASI-GO-2 + R-Zero)
   ├─ Consciousness Processing (Basins + ThoughtSeeds)
   ├─ Analyze Results (Quality + Insights)
   ├─ Refine Processing
   └─ Finalize Output → final_output JSON
   ↓
3. DocumentRepository.persist_document(final_output, metadata)
   ├─ Validate required fields (document_id, content_hash, filename)
   ├─ Check for duplicates (content_hash lookup)
   ├─ Start transaction (atomic)
   ├─ Create :Document node (metadata + tier="warm")
   ├─ Process 5-level concepts → create :Concept nodes
   ├─ Process basins → create/update :AttractorBasin nodes
   ├─ Process thoughtseeds → create :ThoughtSeed nodes
   ├─ Create all relationships (EXTRACTED_FROM, ATTRACTED_TO, etc.)
   ├─ Store basin evolution events (influence type, strength delta)
   ├─ Commit transaction
   └─ Return persistence result
   ↓
4. Document available via:
   - GET /api/documents (listing with pagination)
   - GET /api/documents/{id} (detail view)
```

---

## Neo4j Schema Design

### Node Types

#### Document Node
```cypher
CREATE (d:Document {
  // Core metadata
  document_id: "doc_1234567890",
  filename: "research_paper.pdf",
  content_hash: "sha256:abc123...",
  upload_timestamp: datetime("2025-10-07T12:00:00Z"),
  file_size: 1048576,  // bytes
  mime_type: "application/pdf",
  tags: ["research", "neuroscience", "consciousness"],

  // Processing metadata
  processed_at: datetime("2025-10-07T12:00:05Z"),
  processing_duration_ms: 1500,
  processing_status: "complete",  // complete | partial | failed

  // Quality metrics
  quality_overall: 0.85,
  quality_coherence: 0.90,
  quality_novelty: 0.75,
  quality_depth: 0.88,

  // Research metadata
  curiosity_triggers: 5,
  research_questions: 3,

  // Tier management (clarification: hybrid age + access)
  tier: "warm",  // warm | cool | cold
  last_accessed: datetime("2025-10-07T15:30:00Z"),
  access_count: 12,
  tier_changed_at: datetime("2025-10-07T12:00:05Z"),

  // Cold tier archival (clarification: S3/filesystem)
  archive_location: null,  // s3://bucket/path or /archive/path when cold
  archived_at: null
})
```

#### Concept Node (5 Levels)
```cypher
// Atomic Concept
CREATE (c:Concept:AtomicConcept {
  concept_id: "concept_001",
  name: "active_inference",
  level: "atomic",
  salience: 0.95,
  definition: "A framework for understanding brain function..."
})

// Relationship Concept
CREATE (c:Concept:RelationshipConcept {
  concept_id: "concept_002",
  name: "inference_enables_prediction",
  level: "relationship",
  salience: 0.82,
  source_concept: "active_inference",
  target_concept: "prediction_error"
})

// Composite Concept
CREATE (c:Concept:CompositeConcept {
  concept_id: "concept_003",
  name: "consciousness_framework",
  level: "composite",
  salience: 0.88,
  components: ["active_inference", "free_energy", "prediction_error"]
})

// Context Concept
CREATE (c:Concept:ContextConcept {
  concept_id: "concept_004",
  name: "neuroscience_paradigm_shift",
  level: "context",
  salience: 0.79,
  domain: "neuroscience",
  era: "21st_century"
})

// Narrative Concept
CREATE (c:Concept:NarrativeConcept {
  concept_id: "concept_005",
  name: "emerging_consciousness_theory",
  level: "narrative",
  salience: 0.91,
  storyline: "How active inference unifies consciousness studies"
})
```

#### Attractor Basin Node
```cypher
CREATE (b:AttractorBasin {
  basin_id: "basin_001",
  name: "consciousness_dynamics",
  depth: 0.75,  // Attraction strength
  stability: 0.88,  // Resistance to perturbation
  strength: 1.5,  // Overall basin strength (from Context Engineering)
  associated_concepts: ["consciousness", "emergence", "integration"],

  // Context Engineering integration
  influence_history: [
    {document_id: "doc_123", influence_type: "reinforcement", strength_delta: +0.1},
    {document_id: "doc_456", influence_type: "competition", strength_delta: -0.05},
    {document_id: "doc_789", influence_type: "synthesis", strength_delta: +0.15}
  ],

  // Evolution tracking
  created_at: datetime("2025-10-01T10:00:00Z"),
  last_modified: datetime("2025-10-07T12:00:05Z"),
  modification_count: 15
})
```

#### ThoughtSeed Node
```cypher
CREATE (t:ThoughtSeed {
  seed_id: "seed_001",
  content: "How does active inference relate to consciousness emergence?",
  germination_potential: 0.92,  // Likelihood of generating insights
  resonance_score: 0.85,  // Connection to existing knowledge

  // Context Engineering: Neural field resonance
  field_resonance: {
    energy: 0.73,
    phase: 0.45,
    interference_pattern: "constructive"
  },

  // Metadata
  generated_at: datetime("2025-10-07T12:00:04Z"),
  source_stage: "consciousness_processing"
})
```

### Relationships

```cypher
// Concept extracted from document
(:Concept)-[:EXTRACTED_FROM {
  confidence: 0.90,
  extraction_method: "AutoSchemaKG",
  timestamp: datetime()
}]->(:Document)

// Basin attracted to document (activated during processing)
(:AttractorBasin)-[:ATTRACTED_TO {
  activation_strength: 0.85,
  influence_type: "reinforcement",  // reinforcement | competition | synthesis | emergence
  strength_delta: +0.10,
  timestamp: datetime()
}]->(:Document)

// ThoughtSeed germinated from document
(:ThoughtSeed)-[:GERMINATED_FROM {
  potential: 0.92,
  generation_stage: "consciousness_processing",
  timestamp: datetime()
}]->(:Document)

// Concept derived from another concept
(:Concept)-[:DERIVED_FROM {
  derivation_type: "composition",  // composition | abstraction | specialization
  confidence: 0.88
}]->(:Concept)

// Concepts resonate with each other (neural field integration)
(:Concept)-[:RESONATES_WITH {
  field_score: 0.73,
  field_energy: 0.85,
  resonance_type: "constructive",  // constructive | destructive | neutral
  discovered_via: "neural_field_evolution"
}]-(:Concept)
```

### Indexes & Constraints

```cypher
// Uniqueness constraints
CREATE CONSTRAINT document_id_unique IF NOT EXISTS
FOR (d:Document) REQUIRE d.document_id IS UNIQUE;

CREATE CONSTRAINT content_hash_unique IF NOT EXISTS
FOR (d:Document) REQUIRE d.content_hash IS UNIQUE;

CREATE CONSTRAINT concept_id_unique IF NOT EXISTS
FOR (c:Concept) REQUIRE c.concept_id IS UNIQUE;

CREATE CONSTRAINT basin_id_unique IF NOT EXISTS
FOR (b:AttractorBasin) REQUIRE b.basin_id IS UNIQUE;

CREATE CONSTRAINT seed_id_unique IF NOT EXISTS
FOR (t:ThoughtSeed) REQUIRE t.seed_id IS UNIQUE;

// Performance indexes (for listing API filters/sorts)
CREATE INDEX document_upload_timestamp IF NOT EXISTS
FOR (d:Document) ON (d.upload_timestamp);

CREATE INDEX document_quality IF NOT EXISTS
FOR (d:Document) ON (d.quality_overall);

CREATE INDEX document_tier IF NOT EXISTS
FOR (d:Document) ON (d.tier);

CREATE INDEX document_tags IF NOT EXISTS
FOR (d:Document) ON (d.tags);

// Full-text search (for future enhancements)
CREATE FULLTEXT INDEX document_content IF NOT EXISTS
FOR (d:Document) ON EACH [d.filename, d.tags];

CREATE FULLTEXT INDEX concept_names IF NOT EXISTS
FOR (c:Concept) ON EACH [c.name, c.definition];
```

---

## API Contract Design

### POST /api/documents/persist

**Purpose**: Persist Daedalus final_output to Neo4j

**Request Body**:
```json
{
  "document_id": "doc_1234567890",
  "filename": "research.pdf",
  "content_hash": "sha256:abc123...",
  "file_size": 1048576,
  "mime_type": "application/pdf",
  "tags": ["research", "ai"],
  "daedalus_output": {
    "quality": {
      "scores": {
        "overall": 0.85,
        "coherence": 0.90,
        "novelty": 0.75,
        "depth": 0.88
      }
    },
    "concepts": {
      "atomic": [
        {"concept_id": "c001", "name": "active_inference", "salience": 0.95}
      ],
      "relationship": [...],
      "composite": [...],
      "context": [...],
      "narrative": [...]
    },
    "basins": [
      {
        "basin_id": "b001",
        "name": "consciousness_dynamics",
        "depth": 0.75,
        "stability": 0.88,
        "influence_type": "reinforcement",
        "strength_delta": 0.10
      }
    ],
    "thoughtseeds": [
      {
        "seed_id": "s001",
        "content": "How does active inference relate to consciousness?",
        "germination_potential": 0.92,
        "resonance_score": 0.85
      }
    ],
    "research": {
      "curiosity_triggers": 5,
      "research_questions": 3
    },
    "processing_timeline": [
      {"stage": "extract", "duration_ms": 300},
      {"stage": "consciousness", "duration_ms": 800},
      {"stage": "analyze", "duration_ms": 400}
    ]
  }
}
```

**Response** (200 OK):
```json
{
  "status": "success",
  "document_id": "doc_1234567890",
  "persisted_at": "2025-10-07T12:00:05Z",
  "tier": "warm",
  "nodes_created": 45,
  "relationships_created": 78,
  "performance": {
    "persistence_duration_ms": 1500,
    "met_target": true  // target: <2000ms
  }
}
```

**Response** (409 Conflict - Duplicate):
```json
{
  "status": "duplicate",
  "document_id": "doc_0987654321",  // Existing document ID
  "content_hash": "sha256:abc123...",
  "message": "Document with this content already exists",
  "existing_document": {
    "document_id": "doc_0987654321",
    "filename": "research.pdf",
    "uploaded_at": "2025-10-05T10:00:00Z"
  },
  "options": ["view_existing", "reprocess"]
}
```

### GET /api/documents

**Purpose**: List documents with pagination, filtering, sorting

**Query Parameters**:
- `page` (default: 1)
- `limit` (default: 50, max: 200)
- `tags` (comma-separated, e.g., "research,ai")
- `quality_min` (0.0-1.0, e.g., 0.8)
- `date_from` (ISO 8601, e.g., "2025-10-01")
- `date_to` (ISO 8601)
- `sort` (upload_date | quality | curiosity, default: upload_date)
- `order` (asc | desc, default: desc)
- `tier` (warm | cool | cold, filter by tier)

**Example Request**:
```
GET /api/documents?page=1&limit=50&tags=research&quality_min=0.8&sort=quality&order=desc
```

**Response** (200 OK):
```json
{
  "documents": [
    {
      "document_id": "doc_1234567890",
      "filename": "research.pdf",
      "upload_timestamp": "2025-10-07T12:00:00Z",
      "tags": ["research", "ai"],
      "quality_overall": 0.85,
      "tier": "warm",
      "concept_count": 25,
      "basin_count": 3,
      "thoughtseed_count": 5,
      "curiosity_triggers": 5
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 50,
    "total": 127,
    "total_pages": 3
  },
  "performance": {
    "query_duration_ms": 350,
    "met_target": true  // target: <500ms for 100 docs
  }
}
```

### GET /api/documents/{id}

**Purpose**: Get document detail with all processing artifacts

**Path Parameters**:
- `id`: document_id

**Response** (200 OK):
```json
{
  "document_id": "doc_1234567890",
  "metadata": {
    "filename": "research.pdf",
    "upload_timestamp": "2025-10-07T12:00:00Z",
    "file_size": 1048576,
    "mime_type": "application/pdf",
    "tags": ["research", "ai"],
    "tier": "warm",
    "last_accessed": "2025-10-07T15:30:00Z",
    "access_count": 12
  },
  "quality": {
    "overall": 0.85,
    "coherence": 0.90,
    "novelty": 0.75,
    "depth": 0.88
  },
  "concepts": {
    "atomic": [{"concept_id": "c001", "name": "active_inference", "salience": 0.95}],
    "relationship": [...],
    "composite": [...],
    "context": [...],
    "narrative": [...]
  },
  "basins": [
    {
      "basin_id": "b001",
      "name": "consciousness_dynamics",
      "depth": 0.75,
      "stability": 0.88,
      "activation_strength": 0.85
    }
  ],
  "thoughtseeds": [
    {
      "seed_id": "s001",
      "content": "How does active inference relate to consciousness?",
      "germination_potential": 0.92,
      "resonance_score": 0.85
    }
  ],
  "processing_timeline": [
    {"stage": "extract", "duration_ms": 300},
    {"stage": "consciousness", "duration_ms": 800},
    {"stage": "analyze", "duration_ms": 400}
  ]
}
```

### PUT /api/documents/{id}/tier

**Purpose**: Update document tier (for manual tier management)

**Path Parameters**:
- `id`: document_id

**Request Body**:
```json
{
  "new_tier": "cold",
  "reason": "manual_archival"
}
```

**Response** (200 OK):
```json
{
  "status": "success",
  "document_id": "doc_1234567890",
  "old_tier": "warm",
  "new_tier": "cold",
  "tier_changed_at": "2025-10-07T16:00:00Z",
  "archive_location": "s3://dionysus-cold/doc_1234567890.json"
}
```

---

## Implementation Phases

### Phase 1: Core Repository Service (Week 1)

**Tasks**:
1. Create `DocumentRepository` class in `backend/src/services/document_repository.py`
2. Implement `persist_document()` with atomic transaction
3. Implement duplicate detection (content_hash lookup)
4. Add validation for required fields
5. Integrate DaedalusGraphChannel for all Neo4j operations
6. Add audit trail (caller_service="document_repository", caller_function)

**Constitutional Compliance**:
- ✅ Import only `from daedalus_gateway import get_graph_channel`
- ✅ NO direct `from neo4j import` statements
- ✅ All operations via `channel.execute_write()` / `channel.execute_read()`

**Deliverables**:
- `document_repository.py` (service implementation)
- Unit tests for persist_document with mocked Graph Channel
- Constitutional compliance test (no neo4j imports)

### Phase 2: Graph Schema & Document Nodes (Week 1)

**Tasks**:
1. Create schema initialization script
2. Define all node types (Document, Concept, Basin, ThoughtSeed)
3. Define all relationship types
4. Create uniqueness constraints
5. Create performance indexes
6. Implement document node creation logic
7. Add tier classification (initial tier="warm")

**Cypher Operations** (via Graph Channel):
```python
# Create document node
query = """
CREATE (d:Document {
  document_id: $document_id,
  filename: $filename,
  content_hash: $content_hash,
  upload_timestamp: datetime($upload_timestamp),
  file_size: $file_size,
  mime_type: $mime_type,
  tags: $tags,
  quality_overall: $quality_overall,
  tier: "warm",
  processed_at: datetime(),
  processing_duration_ms: $processing_duration_ms
})
RETURN d.document_id as document_id
"""

result = await graph_channel.execute_write(
    query=query,
    parameters=params,
    caller_service="document_repository",
    caller_function="create_document_node"
)
```

**Deliverables**:
- Schema initialization script
- Document node creation logic
- Integration test: persist document → verify node in Neo4j

### Phase 3: 5-Level Concept Storage (Week 2)

**Tasks**:
1. Parse Daedalus concepts output (atomic → narrative levels)
2. Create concept nodes with level-specific labels
3. Create EXTRACTED_FROM relationships
4. Create DERIVED_FROM relationships (composite → atomic)
5. Store salience scores
6. Integrate AutoSchemaKGService for concept processing

**Concept Processing Logic**:
```python
async def _persist_concepts(self, concepts_data, document_id):
    """Persist 5-level concepts from Daedalus output."""
    for level in ["atomic", "relationship", "composite", "context", "narrative"]:
        concepts = concepts_data.get(level, [])
        for concept in concepts:
            # Create concept node
            query = f"""
            MERGE (c:Concept:{level.capitalize()}Concept {{concept_id: $concept_id}})
            ON CREATE SET
              c.name = $name,
              c.level = $level,
              c.salience = $salience,
              c.created_at = datetime()
            RETURN c.concept_id
            """

            await self.graph_channel.execute_write(
                query=query,
                parameters={
                    "concept_id": concept["concept_id"],
                    "name": concept["name"],
                    "level": level,
                    "salience": concept["salience"]
                },
                caller_service="document_repository",
                caller_function="_persist_concepts"
            )

            # Link to document
            link_query = """
            MATCH (c:Concept {concept_id: $concept_id})
            MATCH (d:Document {document_id: $document_id})
            CREATE (c)-[:EXTRACTED_FROM {
              confidence: $confidence,
              extraction_method: "AutoSchemaKG",
              timestamp: datetime()
            }]->(d)
            """

            await self.graph_channel.execute_write(
                query=link_query,
                parameters={
                    "concept_id": concept["concept_id"],
                    "document_id": document_id,
                    "confidence": concept.get("confidence", 0.90)
                },
                caller_service="document_repository",
                caller_function="_persist_concepts"
            )
```

**Deliverables**:
- Concept persistence logic (all 5 levels)
- Relationship creation (EXTRACTED_FROM, DERIVED_FROM)
- Integration test: persist document with concepts → verify concept nodes + relationships

### Phase 4: Attractor Basin & ThoughtSeed Storage (Week 2)

**Tasks**:
1. Parse basin data from Daedalus output
2. Create/update AttractorBasin nodes
3. Store basin influence events (reinforcement/competition/synthesis/emergence)
4. Create ATTRACTED_TO relationships with activation strength
5. Parse thoughtseed data
6. Create ThoughtSeed nodes with germination potential
7. Create GERMINATED_FROM relationships
8. Integrate AttractorBasinManager for basin evolution tracking

**Basin Persistence with Context Engineering**:
```python
async def _persist_basins(self, basins_data, document_id):
    """Persist attractor basins with Context Engineering integration."""
    for basin in basins_data:
        # Create/update basin node
        query = """
        MERGE (b:AttractorBasin {basin_id: $basin_id})
        ON CREATE SET
          b.name = $name,
          b.depth = $depth,
          b.stability = $stability,
          b.strength = $strength,
          b.created_at = datetime()
        ON MATCH SET
          b.depth = $depth,
          b.stability = $stability,
          b.strength = b.strength + $strength_delta,
          b.last_modified = datetime(),
          b.modification_count = b.modification_count + 1
        RETURN b.basin_id
        """

        await self.graph_channel.execute_write(
            query=query,
            parameters={
                "basin_id": basin["basin_id"],
                "name": basin["name"],
                "depth": basin["depth"],
                "stability": basin["stability"],
                "strength": basin.get("strength", 1.0),
                "strength_delta": basin.get("strength_delta", 0)
            },
            caller_service="document_repository",
            caller_function="_persist_basins"
        )

        # Link basin to document with influence type
        link_query = """
        MATCH (b:AttractorBasin {basin_id: $basin_id})
        MATCH (d:Document {document_id: $document_id})
        CREATE (b)-[:ATTRACTED_TO {
          activation_strength: $activation_strength,
          influence_type: $influence_type,
          strength_delta: $strength_delta,
          timestamp: datetime()
        }]->(d)
        """

        await self.graph_channel.execute_write(
            query=link_query,
            parameters={
                "basin_id": basin["basin_id"],
                "document_id": document_id,
                "activation_strength": basin.get("activation_strength", 0.85),
                "influence_type": basin.get("influence_type", "reinforcement"),
                "strength_delta": basin.get("strength_delta", 0.10)
            },
            caller_service="document_repository",
            caller_function="_persist_basins"
        )

        # Update AttractorBasinManager in Redis
        await self._update_basin_manager(basin, document_id)
```

**Deliverables**:
- Basin persistence with evolution tracking
- ThoughtSeed persistence with resonance scores
- Context Engineering integration (basin manager updates)
- Integration test: persist document with basins/seeds → verify nodes + basin evolution

### Phase 5: Document Listing API (Week 3)

**Tasks**:
1. Implement `list_documents()` method
2. Add pagination logic (page, limit)
3. Add filtering (tags, date_range, quality_min, tier)
4. Add sorting (upload_date, quality, curiosity_triggers)
5. Optimize queries with indexes
6. Add performance monitoring (<500ms target)

**Listing Query Implementation**:
```python
async def list_documents(
    self,
    page: int = 1,
    limit: int = 50,
    tags: Optional[List[str]] = None,
    quality_min: Optional[float] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    sort: str = "upload_date",
    order: str = "desc",
    tier: Optional[str] = None
):
    """List documents with pagination, filtering, sorting."""

    # Build dynamic Cypher query
    where_clauses = []
    if tags:
        where_clauses.append("ANY(tag IN $tags WHERE tag IN d.tags)")
    if quality_min:
        where_clauses.append("d.quality_overall >= $quality_min")
    if date_from:
        where_clauses.append("d.upload_timestamp >= datetime($date_from)")
    if date_to:
        where_clauses.append("d.upload_timestamp <= datetime($date_to)")
    if tier:
        where_clauses.append("d.tier = $tier")

    where_clause = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

    # Sort field mapping
    sort_fields = {
        "upload_date": "d.upload_timestamp",
        "quality": "d.quality_overall",
        "curiosity": "d.curiosity_triggers"
    }
    sort_field = sort_fields.get(sort, "d.upload_timestamp")
    order_clause = f"{sort_field} {order.upper()}"

    # Count total (for pagination)
    count_query = f"""
    MATCH (d:Document)
    {where_clause}
    RETURN count(d) as total
    """

    count_result = await self.graph_channel.execute_read(
        query=count_query,
        parameters={
            "tags": tags,
            "quality_min": quality_min,
            "date_from": date_from,
            "date_to": date_to,
            "tier": tier
        },
        caller_service="document_repository",
        caller_function="list_documents_count"
    )

    total = count_result["records"][0]["total"]

    # Get paginated documents with counts
    skip = (page - 1) * limit

    list_query = f"""
    MATCH (d:Document)
    {where_clause}
    OPTIONAL MATCH (c:Concept)-[:EXTRACTED_FROM]->(d)
    OPTIONAL MATCH (b:AttractorBasin)-[:ATTRACTED_TO]->(d)
    OPTIONAL MATCH (t:ThoughtSeed)-[:GERMINATED_FROM]->(d)
    WITH d,
         count(DISTINCT c) as concept_count,
         count(DISTINCT b) as basin_count,
         count(DISTINCT t) as thoughtseed_count
    ORDER BY {order_clause}
    SKIP $skip
    LIMIT $limit
    RETURN d, concept_count, basin_count, thoughtseed_count
    """

    result = await self.graph_channel.execute_read(
        query=list_query,
        parameters={
            "tags": tags,
            "quality_min": quality_min,
            "date_from": date_from,
            "date_to": date_to,
            "tier": tier,
            "skip": skip,
            "limit": limit
        },
        caller_service="document_repository",
        caller_function="list_documents"
    )

    # Format response
    documents = []
    for record in result["records"]:
        doc = record["d"]
        documents.append({
            "document_id": doc["document_id"],
            "filename": doc["filename"],
            "upload_timestamp": doc["upload_timestamp"],
            "tags": doc["tags"],
            "quality_overall": doc["quality_overall"],
            "tier": doc["tier"],
            "concept_count": record["concept_count"],
            "basin_count": record["basin_count"],
            "thoughtseed_count": record["thoughtseed_count"],
            "curiosity_triggers": doc["curiosity_triggers"]
        })

    return {
        "documents": documents,
        "pagination": {
            "page": page,
            "limit": limit,
            "total": total,
            "total_pages": (total + limit - 1) // limit
        }
    }
```

**Deliverables**:
- Document listing API with all filters/sorts
- Performance optimization (query caching, index usage)
- Integration test: list 100 documents in <500ms

### Phase 6: Document Detail API (Week 3)

**Tasks**:
1. Implement `get_document()` method
2. Fetch document node + all linked artifacts
3. Fetch concepts (all 5 levels)
4. Fetch basins with activation strengths
5. Fetch thoughtseeds with resonance scores
6. Include processing timeline
7. Update access tracking (last_accessed, access_count)

**Detail Query Implementation**:
```python
async def get_document(self, document_id: str):
    """Get document detail with all processing artifacts."""

    # Single comprehensive query to minimize round trips
    query = """
    MATCH (d:Document {document_id: $document_id})

    // Get all concepts by level
    OPTIONAL MATCH (c:Concept)-[r_concept:EXTRACTED_FROM]->(d)
    WITH d, collect({
      concept_id: c.concept_id,
      name: c.name,
      level: c.level,
      salience: c.salience
    }) as concepts

    // Get all basins
    OPTIONAL MATCH (b:AttractorBasin)-[r_basin:ATTRACTED_TO]->(d)
    WITH d, concepts, collect({
      basin_id: b.basin_id,
      name: b.name,
      depth: b.depth,
      stability: b.stability,
      activation_strength: r_basin.activation_strength,
      influence_type: r_basin.influence_type
    }) as basins

    // Get all thoughtseeds
    OPTIONAL MATCH (t:ThoughtSeed)-[r_seed:GERMINATED_FROM]->(d)
    WITH d, concepts, basins, collect({
      seed_id: t.seed_id,
      content: t.content,
      germination_potential: t.germination_potential,
      resonance_score: t.resonance_score
    }) as thoughtseeds

    // Update access tracking
    SET d.last_accessed = datetime(),
        d.access_count = d.access_count + 1

    RETURN d, concepts, basins, thoughtseeds
    """

    result = await self.graph_channel.execute_write(  # Write to update access tracking
        query=query,
        parameters={"document_id": document_id},
        caller_service="document_repository",
        caller_function="get_document"
    )

    if not result["records"]:
        return None  # Document not found

    record = result["records"][0]
    doc = record["d"]

    # Organize concepts by level
    concepts_by_level = {
        "atomic": [],
        "relationship": [],
        "composite": [],
        "context": [],
        "narrative": []
    }
    for concept in record["concepts"]:
        level = concept["level"]
        if level in concepts_by_level:
            concepts_by_level[level].append(concept)

    return {
        "document_id": doc["document_id"],
        "metadata": {
            "filename": doc["filename"],
            "upload_timestamp": doc["upload_timestamp"],
            "file_size": doc["file_size"],
            "mime_type": doc["mime_type"],
            "tags": doc["tags"],
            "tier": doc["tier"],
            "last_accessed": doc["last_accessed"],
            "access_count": doc["access_count"]
        },
        "quality": {
            "overall": doc["quality_overall"],
            "coherence": doc.get("quality_coherence"),
            "novelty": doc.get("quality_novelty"),
            "depth": doc.get("quality_depth")
        },
        "concepts": concepts_by_level,
        "basins": record["basins"],
        "thoughtseeds": record["thoughtseeds"],
        "processing_timeline": doc.get("processing_timeline", [])
    }
```

**Deliverables**:
- Document detail API with complete data
- Access tracking (last_accessed, access_count updates)
- Integration test: get document → verify all artifacts present

### Phase 7: Tier Management (Week 4)

**Tasks**:
1. Implement hybrid tier migration logic (age + access patterns)
2. Create background job for tier evaluation
3. Implement cold tier archival (S3/filesystem)
4. Update document metadata on tier change
5. Add manual tier update API
6. Integrate MultiTierMemorySystem

**Tier Migration Logic** (Clarification: Hybrid age + access):
```python
async def evaluate_tier_migrations(self):
    """Background job to evaluate and migrate document tiers."""

    # Tier migration rules (hybrid: age + access)
    rules = {
        "warm_to_cool": {
            "min_age_days": 30,
            "max_access_count": 5,  # If accessed >5 times, stays warm
            "max_days_since_access": 14
        },
        "cool_to_cold": {
            "min_age_days": 90,
            "max_access_count": 2,
            "max_days_since_access": 60
        }
    }

    # Find warm documents eligible for cool tier
    query = """
    MATCH (d:Document {tier: "warm"})
    WHERE duration.between(d.upload_timestamp, datetime()).days >= $min_age_days
      AND d.access_count <= $max_access_count
      AND duration.between(d.last_accessed, datetime()).days >= $max_days_since_access
    RETURN d.document_id as document_id, d.filename as filename
    """

    warm_to_cool = await self.graph_channel.execute_read(
        query=query,
        parameters=rules["warm_to_cool"],
        caller_service="document_repository",
        caller_function="evaluate_warm_to_cool"
    )

    for record in warm_to_cool["records"]:
        await self.update_tier(record["document_id"], "cool", "automatic_age_access_hybrid")

    # Find cool documents eligible for cold tier (archival)
    query = """
    MATCH (d:Document {tier: "cool"})
    WHERE duration.between(d.upload_timestamp, datetime()).days >= $min_age_days
      AND d.access_count <= $max_access_count
      AND duration.between(d.last_accessed, datetime()).days >= $max_days_since_access
    RETURN d.document_id as document_id, d.filename as filename
    """

    cool_to_cold = await self.graph_channel.execute_read(
        query=query,
        parameters=rules["cool_to_cold"],
        caller_service="document_repository",
        caller_function="evaluate_cool_to_cold"
    )

    for record in cool_to_cold["records"]:
        await self.archive_to_cold_tier(record["document_id"])

async def archive_to_cold_tier(self, document_id: str):
    """Archive document to cold tier storage (S3/filesystem)."""

    # Fetch complete document data
    doc_data = await self.get_document(document_id)

    # Archive to S3/filesystem (clarification: separate storage)
    archive_location = await self._write_to_archive(document_id, doc_data)

    # Update document node (keep metadata, mark archived)
    query = """
    MATCH (d:Document {document_id: $document_id})
    SET d.tier = "cold",
        d.archived_at = datetime(),
        d.archive_location = $archive_location,
        d.tier_changed_at = datetime()
    RETURN d.document_id
    """

    await self.graph_channel.execute_write(
        query=query,
        parameters={
            "document_id": document_id,
            "archive_location": archive_location
        },
        caller_service="document_repository",
        caller_function="archive_to_cold_tier"
    )

    # Optional: Remove full concept/basin/seed nodes to save space
    # (Keep references, remove detailed content)
    # This is a design decision for Phase 7 implementation

async def _write_to_archive(self, document_id: str, doc_data: dict) -> str:
    """Write document data to cold tier storage."""
    # Implementation depends on chosen storage:
    # Option A: S3 (boto3)
    # Option B: Filesystem (/archive/)
    # Return: archive_location (s3://bucket/key or /path/to/file)
    pass  # Implement based on infrastructure
```

**Deliverables**:
- Tier migration background job
- Cold tier archival to S3/filesystem
- Manual tier update API (PUT /api/documents/{id}/tier)
- Integration test: tier migration rules + archival

### Phase 8: Performance Optimization & Testing (Week 4)

**Tasks**:
1. Add query caching for frequently accessed documents
2. Optimize pagination queries (index usage)
3. Add performance monitoring (timing metrics)
4. Load testing: 10,000 documents
5. Verify persistence <2s target
6. Verify listing <500ms target
7. Add circuit breaker monitoring

**Performance Monitoring**:
```python
import time
from typing import Callable

async def _with_performance_monitoring(
    self,
    operation: str,
    target_ms: int,
    func: Callable,
    *args,
    **kwargs
):
    """Wrapper to monitor operation performance."""
    start = time.time()
    result = await func(*args, **kwargs)
    duration_ms = (time.time() - start) * 1000

    met_target = duration_ms < target_ms

    # Emit metrics (Prometheus, CloudWatch, etc.)
    self.metrics.record_operation(
        operation=operation,
        duration_ms=duration_ms,
        met_target=met_target
    )

    if not met_target:
        self.logger.warning(
            f"Performance target missed: {operation} took {duration_ms:.0f}ms (target: {target_ms}ms)"
        )

    return result, {"duration_ms": duration_ms, "met_target": met_target}

# Usage
async def persist_document(self, final_output, metadata):
    result, perf = await self._with_performance_monitoring(
        operation="persist_document",
        target_ms=2000,  # <2s target
        func=self._do_persist_document,
        final_output=final_output,
        metadata=metadata
    )
    return {**result, "performance": perf}
```

**Load Testing Script**:
```python
import asyncio
import time

async def load_test_persistence():
    """Load test document persistence with 1000 documents."""
    repository = DocumentRepository()

    durations = []
    for i in range(1000):
        start = time.time()

        # Generate test document
        final_output = generate_test_document(f"doc_{i}")
        metadata = {"filename": f"test_{i}.pdf", "tags": ["test"]}

        # Persist
        result = await repository.persist_document(final_output, metadata)

        duration = (time.time() - start) * 1000
        durations.append(duration)

        if i % 100 == 0:
            print(f"Progress: {i}/1000, Avg: {sum(durations)/len(durations):.0f}ms")

    # Statistics
    avg = sum(durations) / len(durations)
    p50 = sorted(durations)[len(durations) // 2]
    p95 = sorted(durations)[int(len(durations) * 0.95)]
    p99 = sorted(durations)[int(len(durations) * 0.99)]

    print(f"\nPersistence Performance:")
    print(f"  Avg: {avg:.0f}ms")
    print(f"  P50: {p50:.0f}ms")
    print(f"  P95: {p95:.0f}ms")
    print(f"  P99: {p99:.0f}ms")
    print(f"  Target (<2000ms): {'✅ PASS' if p95 < 2000 else '❌ FAIL'}")

async def load_test_listing():
    """Load test document listing with 10,000 documents."""
    repository = DocumentRepository()

    # Pre-populate 10,000 documents
    print("Pre-populating 10,000 documents...")
    # ... populate logic

    # Test listing performance
    durations = []
    for i in range(100):
        start = time.time()
        result = await repository.list_documents(page=1, limit=50)
        duration = (time.time() - start) * 1000
        durations.append(duration)

    avg = sum(durations) / len(durations)
    p95 = sorted(durations)[int(len(durations) * 0.95)]

    print(f"\nListing Performance (10k docs):")
    print(f"  Avg: {avg:.0f}ms")
    print(f"  P95: {p95:.0f}ms")
    print(f"  Target (<500ms): {'✅ PASS' if p95 < 500 else '❌ FAIL'}")
```

**Deliverables**:
- Performance monitoring wrapper
- Load testing scripts
- Performance validation report
- Optimization recommendations

---

## Testing Strategy

### Unit Tests

**Test Files**:
- `test_document_repository.py` - Core repository methods
- `test_concept_persistence.py` - 5-level concept storage
- `test_basin_persistence.py` - Basin evolution tracking
- `test_tier_management.py` - Tier migration logic

**Test Coverage Targets**:
- Repository methods: 90%+
- Constitutional compliance: 100% (no neo4j imports)
- Error handling: 80%+

### Integration Tests

**Test Scenarios**:
1. **End-to-End Persistence**: Upload → Process → Persist → Retrieve
2. **Duplicate Detection**: Upload same document twice → 409 Conflict
3. **Graph Channel Compliance**: All operations via channel, audit trail present
4. **Tier Migration**: Document ages → moves to cool → moves to cold
5. **Performance**: Persistence <2s, listing <500ms

**Integration Test Example**:
```python
@pytest.mark.integration
async def test_end_to_end_persistence():
    """Test complete document lifecycle: persist → list → retrieve."""
    repository = DocumentRepository()

    # Step 1: Persist document
    final_output = load_test_daedalus_output("test_research.pdf")
    metadata = {
        "document_id": "test_doc_001",
        "filename": "test_research.pdf",
        "content_hash": "sha256:test123",
        "file_size": 1048576,
        "mime_type": "application/pdf",
        "tags": ["test", "research"]
    }

    persist_result = await repository.persist_document(final_output, metadata)
    assert persist_result["status"] == "success"
    assert persist_result["document_id"] == "test_doc_001"
    assert persist_result["performance"]["met_target"] is True  # <2s

    # Step 2: Verify appears in listing
    list_result = await repository.list_documents(page=1, limit=50)
    doc_ids = [d["document_id"] for d in list_result["documents"]]
    assert "test_doc_001" in doc_ids

    # Step 3: Retrieve document detail
    detail_result = await repository.get_document("test_doc_001")
    assert detail_result["document_id"] == "test_doc_001"
    assert detail_result["metadata"]["filename"] == "test_research.pdf"
    assert len(detail_result["concepts"]["atomic"]) > 0
    assert len(detail_result["basins"]) > 0
    assert len(detail_result["thoughtseeds"]) > 0

    # Step 4: Verify Graph Channel compliance
    # Check that all operations used Graph Channel (audit trail)
    audit_logs = get_graph_channel_audit_logs()
    operations = [log["caller_function"] for log in audit_logs]
    assert "create_document_node" in operations
    assert "persist_concepts" in operations
    assert "persist_basins" in operations
```

### Constitutional Compliance Tests

**Test File**: `test_constitutional_compliance.py`

```python
def test_no_direct_neo4j_imports():
    """Verify no direct neo4j imports in repository code."""
    import ast

    # Read repository source
    with open("backend/src/services/document_repository.py") as f:
        tree = ast.parse(f.read())

    # Check all imports
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert not node.module.startswith("neo4j"), \
                f"Constitutional violation: direct neo4j import at line {node.lineno}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("neo4j"), \
                    f"Constitutional violation: direct neo4j import at line {node.lineno}"

async def test_all_operations_via_graph_channel():
    """Verify all Neo4j operations use Graph Channel."""
    repository = DocumentRepository()

    # Mock Graph Channel to capture calls
    calls = []
    original_execute_write = repository.graph_channel.execute_write

    async def mock_execute_write(query, parameters, caller_service, caller_function):
        calls.append({
            "operation": "execute_write",
            "caller_service": caller_service,
            "caller_function": caller_function
        })
        return await original_execute_write(query, parameters, caller_service, caller_function)

    repository.graph_channel.execute_write = mock_execute_write

    # Perform operation
    final_output = load_test_daedalus_output("test.pdf")
    await repository.persist_document(final_output, test_metadata())

    # Verify all calls went through Graph Channel
    assert len(calls) > 0
    for call in calls:
        assert call["caller_service"] == "document_repository"
        assert call["caller_function"] in [
            "create_document_node",
            "persist_concepts",
            "persist_basins",
            "persist_thoughtseeds"
        ]
```

---

## Risk Mitigation

### Risk 1: Performance Degradation at Scale

**Risk**: Document persistence or listing exceeds performance targets (>2s, >500ms) at 10,000+ documents

**Mitigation**:
- **Phase 2**: Create indexes early (upload_timestamp, quality, tier, tags)
- **Phase 5**: Optimize pagination queries (SKIP/LIMIT with ORDER BY indexed fields)
- **Phase 8**: Load testing to validate performance before deployment
- **Fallback**: Add Redis caching layer for frequently accessed documents

**Monitoring**:
- Prometheus metrics: `document_persistence_duration_ms`, `document_listing_duration_ms`
- Alert if P95 > target thresholds

### Risk 2: Cold Tier Archival Complexity

**Risk**: S3/filesystem archival implementation delays or complicates deployment

**Mitigation**:
- **Phase 7**: Start with simple filesystem archival (`/archive/` directory)
- **Phase 7**: Add S3 support as enhancement (boto3 integration)
- **Incremental**: Deploy with filesystem first, migrate to S3 post-launch
- **Fallback**: Keep cold tier in Neo4j with `tier="cold"` flag, defer archival

### Risk 3: Duplicate Document Handling Ambiguity

**Risk**: Unclear user behavior on duplicate uploads (show existing? re-process? both?)

**Mitigation**:
- **Implementation**: Return 409 Conflict with existing document info + options
- **Frontend**: Spec 056 will handle user choice (view existing OR trigger re-process)
- **API**: Provide both behaviors: `POST /persist` (409 on duplicate) + `POST /reprocess` (force)

### Risk 4: Graph Channel Unavailable

**Risk**: DaedalusGraphChannel from daedalus-gateway not installed or configured

**Mitigation**:
- **Phase 1**: Add dependency check in repository `__init__`
- **Error Handling**: Raise clear error message with installation instructions
- **Testing**: Mock Graph Channel in unit tests, use real channel in integration tests
- **Documentation**: Update README with daedalus-gateway installation steps

---

## Deployment Checklist

### Prerequisites
- [ ] Daedalus Gateway installed (`pip install -e /path/to/daedalus-gateway`)
- [ ] Neo4j running and accessible
- [ ] Redis running (for basin persistence)
- [ ] S3 bucket created (for cold tier archival) OR `/archive/` directory permissions

### Database Setup
- [ ] Run schema initialization script (constraints + indexes)
- [ ] Verify Graph Channel connection (`channel.health_check()`)
- [ ] Test write operation (create test document node)

### Service Deployment
- [ ] Deploy `DocumentRepository` service
- [ ] Deploy API endpoints (FastAPI routes)
- [ ] Configure environment variables (NEO4J_URI, REDIS_URL, ARCHIVE_LOCATION)
- [ ] Start tier migration background job

### Testing
- [ ] Run unit tests (90%+ coverage)
- [ ] Run integration tests (all passing)
- [ ] Run constitutional compliance tests (100% passing)
- [ ] Run performance tests (load testing with 10k documents)
- [ ] Validate performance targets (<2s persistence, <500ms listing)

### Monitoring
- [ ] Configure Prometheus metrics
- [ ] Set up performance alerts (P95 > target)
- [ ] Configure Graph Channel telemetry dashboard
- [ ] Enable audit trail logging

### Documentation
- [ ] API documentation (OpenAPI/Swagger)
- [ ] Schema diagram (Neo4j graph structure)
- [ ] Deployment guide (README updates)
- [ ] Tier migration runbook (operations manual)

---

## Progress Tracking

- [x] Phase 0: Context Engineering validation
- [ ] Phase 1: Core repository service (Week 1)
- [ ] Phase 2: Graph schema & document nodes (Week 1)
- [ ] Phase 3: 5-level concept storage (Week 2)
- [ ] Phase 4: Basin & thoughtseed storage (Week 2)
- [ ] Phase 5: Document listing API (Week 3)
- [ ] Phase 6: Document detail API (Week 3)
- [ ] Phase 7: Tier management (Week 4)
- [ ] Phase 8: Performance optimization & testing (Week 4)

**Estimated Timeline**: 4 weeks
**Next Step**: Begin Phase 1 implementation

---

## Next Command

**Suggested**: `/tasks` - Generate actionable work items for Phase 1 implementation
