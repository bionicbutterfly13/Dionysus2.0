# Spec 050: Document Persistence & Repository

**Status**: Draft
**Created**: 2025-10-07
**Dependencies**: Spec 040 (Graph Channel compliance), Daedalus LangGraph integration
**Blocks**: Spec 051 (Knowledge Graph APIs)

---

## Problem Statement

Currently, document processing results are lost after upload. The Daedalus LangGraph workflow produces rich outputs (concepts, basins, thoughtseeds, quality metrics, research plans) but these are not persisted to any database. The frontend has no way to retrieve historical documents or display their processing results.

**Current State**:
- ✅ Daedalus receives uploads and processes through LangGraph workflow
- ✅ AutoSchemaKG extracts 5-level concepts
- ✅ Consciousness processing creates basins and thoughtseeds
- ❌ All output is returned to client but NOT stored
- ❌ No document repository or listing API
- ❌ Frontend cannot display real processed documents

**Desired State**:
- ✅ All Daedalus `final_output` persisted to Neo4j via Graph Channel
- ✅ Document metadata (ID, filename, upload_date, tags) stored
- ✅ Tier metadata (warm/cool/cold) tracked
- ✅ Concepts, basins, thoughtseeds linked to documents in graph
- ✅ API endpoint for document listing (`GET /api/documents`)
- ✅ API endpoint for document detail (`GET /api/documents/{id}`)

---

## Requirements

### Functional Requirements

**FR-001**: Document metadata persistence
- MUST store: document_id, filename, upload_timestamp, content_hash, file_size, mime_type, tags
- MUST use Neo4j `:Document` nodes via Graph Channel
- MUST include tier classification (warm/cool/cold)
- MUST be atomic (all-or-nothing)

**FR-002**: Processing output persistence
- MUST store complete Daedalus `final_output` structure
- MUST link concepts to document via `:EXTRACTED_FROM` relationships
- MUST link basins to document via `:ATTRACTED_TO` relationships
- MUST link thoughtseeds to document via `:GERMINATED_FROM` relationships
- MUST store quality metrics as document properties

**FR-003**: Document repository API
- MUST provide `GET /api/documents` - list all documents with pagination
- MUST provide `GET /api/documents/{id}` - get document detail with full output
- MUST support filtering by tags, date range, quality score
- MUST support sorting by upload_date, quality, curiosity_triggers

**FR-004**: Constitutional compliance
- MUST use DaedalusGraphChannel for ALL Neo4j writes (Spec 040 §2.2)
- MUST include `caller_service="document_repository"` on all operations
- MUST use MultiTierMemorySystem for tier management
- MUST NOT import neo4j directly

**FR-005**: Idempotency
- MUST handle duplicate uploads gracefully (same content_hash)
- MUST support re-processing existing documents
- MUST update existing nodes rather than create duplicates

### Non-Functional Requirements

**NFR-001**: Performance
- Document persistence MUST complete in <2 seconds for typical uploads
- Document listing MUST return in <500ms for 100 documents
- MUST use Graph Channel connection pooling

**NFR-002**: Scalability
- MUST support 10,000+ documents without degradation
- MUST use pagination for large result sets (default: 50 items/page)
- MUST use Neo4j indexes on frequently queried fields

**NFR-003**: Reliability
- MUST use transactions for atomic writes
- MUST retry transient failures (via Graph Channel)
- MUST log all persistence operations for audit

**NFR-004**: Observability
- MUST emit metrics: documents_persisted_total, persistence_duration_seconds
- MUST log document_id on all operations
- MUST track tier distribution (warm/cool/cold counts)

---

## Architecture

### Component Diagram

```
Daedalus LangGraph Workflow
         ↓
   final_output
         ↓
┌─────────────────────────────┐
│  DocumentRepository         │
│  (new service)              │
│                             │
│  - persist_document()       │
│  - get_document()           │
│  - list_documents()         │
│  - update_tier()            │
└─────────────────────────────┘
         ↓
┌─────────────────────────────┐
│  DaedalusGraphChannel       │ ← Constitutional compliance (Spec 040)
│  (from daedalus-gateway)    │
└─────────────────────────────┘
         ↓
┌─────────────────────────────┐
│  Neo4j Knowledge Graph      │
│                             │
│  Nodes:                     │
│  - :Document                │
│  - :Concept (5 levels)      │
│  - :AttractorBasin          │
│  - :ThoughtSeed             │
│                             │
│  Relationships:             │
│  - :EXTRACTED_FROM          │
│  - :ATTRACTED_TO            │
│  - :GERMINATED_FROM         │
│  - :DERIVED_FROM            │
│  - :RESONATES_WITH          │
└─────────────────────────────┘
         ↓
┌─────────────────────────────┐
│  MultiTierMemorySystem      │
│  - Warm tier (Neo4j)        │
│  - Cool tier (Vector)       │
│  - Cold tier (Archive)      │
└─────────────────────────────┘
```

### Data Model

**Neo4j Schema**:

```cypher
// Document node
CREATE (d:Document {
  document_id: "doc_12345",
  filename: "research_paper.pdf",
  upload_timestamp: datetime(),
  content_hash: "sha256:abcdef...",
  file_size: 1048576,
  mime_type: "application/pdf",
  tags: ["research", "ai"],

  // Quality metrics
  quality_overall: 0.85,
  quality_coherence: 0.90,
  quality_novelty: 0.75,
  quality_depth: 0.88,

  // Processing metadata
  processed_at: datetime(),
  processing_duration_ms: 1500,
  tier: "warm",

  // Curiosity
  curiosity_triggers: 5,
  research_questions: 3
})

// Concept nodes (5 levels)
CREATE (c:Concept:AtomicConcept {
  concept_id: "concept_001",
  name: "active_inference",
  level: "atomic",
  salience: 0.95
})

CREATE (c:Concept:CompositeConcept {
  concept_id: "concept_002",
  name: "consciousness_framework",
  level: "composite",
  components: ["active_inference", "free_energy", "prediction_error"]
})

// Attractor Basin
CREATE (b:AttractorBasin {
  basin_id: "basin_001",
  name: "consciousness_dynamics",
  depth: 0.75,
  stability: 0.88,
  concepts: ["consciousness", "emergence", "integration"]
})

// ThoughtSeed
CREATE (t:ThoughtSeed {
  seed_id: "seed_001",
  content: "How does active inference relate to consciousness?",
  germination_potential: 0.92,
  resonance: 0.85
})

// Relationships
CREATE (c)-[:EXTRACTED_FROM {confidence: 0.90}]->(d)
CREATE (b)-[:ATTRACTED_TO {strength: 0.85}]->(d)
CREATE (t)-[:GERMINATED_FROM {potential: 0.92}]->(d)
CREATE (c2)-[:DERIVED_FROM]->(c1)
CREATE (t)-[:RESONATES_WITH {score: 0.88}]->(c)
```

### API Contract

**Endpoint**: `POST /api/documents/persist`
```json
// Request
{
  "document_id": "doc_12345",
  "filename": "research.pdf",
  "content_hash": "sha256:abc...",
  "file_size": 1048576,
  "mime_type": "application/pdf",
  "tags": ["research", "ai"],
  "daedalus_output": {
    // Full final_output from Daedalus workflow
    "quality": { ... },
    "concepts": { ... },
    "basins": [ ... ],
    "thoughtseeds": [ ... ],
    "research": { ... }
  }
}

// Response
{
  "status": "success",
  "document_id": "doc_12345",
  "persisted_at": "2025-10-07T12:34:56Z",
  "tier": "warm",
  "nodes_created": 45,
  "relationships_created": 78
}
```

**Endpoint**: `GET /api/documents`
```json
// Query params: ?page=1&limit=50&tags=research&sort=upload_date desc

// Response
{
  "documents": [
    {
      "document_id": "doc_12345",
      "filename": "research.pdf",
      "upload_timestamp": "2025-10-07T12:00:00Z",
      "tags": ["research", "ai"],
      "quality_overall": 0.85,
      "tier": "warm",
      "concept_count": 25,
      "basin_count": 3,
      "thoughtseed_count": 5
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 50,
    "total": 127,
    "total_pages": 3
  }
}
```

**Endpoint**: `GET /api/documents/{id}`
```json
// Response
{
  "document_id": "doc_12345",
  "metadata": { ... },
  "quality": { ... },
  "concepts": {
    "atomic": [ ... ],
    "relationship": [ ... ],
    "composite": [ ... ],
    "context": [ ... ],
    "narrative": [ ... ]
  },
  "basins": [ ... ],
  "thoughtseeds": [ ... ],
  "research": { ... },
  "processing_timeline": [
    {"stage": "extract", "duration_ms": 300},
    {"stage": "consciousness", "duration_ms": 800},
    {"stage": "analyze", "duration_ms": 400}
  ]
}
```

---

## Implementation Plan

### Phase 1: Core Repository Service
- Create `DocumentRepository` class in `backend/src/services/`
- Implement `persist_document()` method
- Use Graph Channel for all Neo4j writes
- Create document node + metadata properties
- Implement idempotency (check content_hash)

### Phase 2: Concept Graph Persistence
- Implement 5-level concept storage
- Create `:EXTRACTED_FROM` relationships
- Store concept salience and level metadata
- Link composite concepts to atomic components

### Phase 3: Consciousness Artifacts
- Persist attractor basins with depth/stability
- Store thoughtseeds with germination potential
- Create `:ATTRACTED_TO` and `:GERMINATED_FROM` relationships
- Link basins/seeds to relevant concepts

### Phase 4: Document Listing API
- Implement `GET /api/documents` endpoint
- Add pagination (default 50, max 200)
- Support filtering by tags, date, quality
- Support sorting by multiple fields

### Phase 5: Document Detail API
- Implement `GET /api/documents/{id}` endpoint
- Fetch complete document with all linked nodes
- Include processing timeline
- Return tier metadata

### Phase 6: Tier Management
- Integrate MultiTierMemorySystem
- Track warm/cool/cold tier for each document
- Implement tier update API
- Add tier distribution metrics

---

## Acceptance Criteria

- [ ] **AC-001**: Document metadata persisted to Neo4j via Graph Channel
- [ ] **AC-002**: All Daedalus output (concepts, basins, seeds) stored in graph
- [ ] **AC-003**: `GET /api/documents` returns paginated list
- [ ] **AC-004**: `GET /api/documents/{id}` returns full document detail
- [ ] **AC-005**: Duplicate uploads handled gracefully (same content_hash)
- [ ] **AC-006**: Constitutional compliance verified (no direct neo4j imports)
- [ ] **AC-007**: Integration tests pass (upload → persist → retrieve)
- [ ] **AC-008**: Performance: persistence <2s, listing <500ms
- [ ] **AC-009**: Metrics emitted (documents_persisted_total, tier_distribution)
- [ ] **AC-010**: Documentation complete (API docs, schema diagram)

---

## Dependencies

**Blockers** (must be complete first):
- ✅ Spec 040 M3: Graph Channel enforcement active
- ✅ Daedalus LangGraph integration complete
- ✅ AutoSchemaKG 5-level concept extraction working

**Blocks** (waiting on this spec):
- Spec 051: Knowledge Graph APIs (needs document repository)
- Spec 052: Frontend live data (needs listing API)

**Parallel Work** (can proceed independently):
- None

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Large documents create too many nodes | HIGH | Implement node limit (e.g., top 100 concepts by salience) |
| Graph writes too slow | MEDIUM | Use batched writes, Graph Channel connection pooling |
| Duplicate content_hash handling fails | MEDIUM | Add unique constraint on content_hash, implement MERGE logic |
| Tier classification inconsistent | LOW | Use MultiTierMemorySystem tier logic, add tests |

---

## Open Questions

1. **Q**: Should we store raw file content in Neo4j or just metadata?
   **A**: Metadata only. Raw content in filesystem/S3, referenced by document_id.

2. **Q**: How to handle document updates (same filename, different content)?
   **A**: New document_id, new content_hash. Original document archived.

3. **Q**: What's the tier migration policy (warm → cool → cold)?
   **A**: Use MultiTierMemorySystem rules: warm=recent, cool=older, cold=archived.

4. **Q**: Should we version the Daedalus output schema?
   **A**: Yes. Add `output_schema_version: "1.0"` to document node.

---

## References

- **Spec 040**: Daedalus Graph Hardening (Graph Channel compliance)
- **CLAUDE.md**: Daedalus LangGraph synthesis architecture
- **AutoSchemaKG**: 5-level concept extraction
- **MultiTierMemorySystem**: Warm/cool/cold tier management
- **AGENT_CONSTITUTION §2.2**: Database abstraction requirements
