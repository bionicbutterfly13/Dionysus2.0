# Ingest OpenSpec Specifications into Neo4j Knowledge Graph

## What

Create an automated pipeline to ingest OpenSpec specification documents (spec.md, design.md) into Neo4j through the existing Daedalus → LangGraph → DocumentRepository consciousness pipeline, enabling semantic search, consciousness-enhanced analysis, and knowledge graph connections across specifications.

## Why

**Current Gap**:
- OpenSpec specs are isolated markdown files
- No semantic search across specifications
- No consciousness-enhanced pattern detection
- No cross-spec relationship discovery (ThoughtSeeds)
- No evolution tracking (AttractorBasins)

**Value of Neo4j Integration**:
1. **Semantic Search**: "Find all specs related to authentication"
2. **Cross-Spec Connections**: Discover specs that share concepts via ThoughtSeeds
3. **Evolution Tracking**: Track how specifications evolve through AttractorBasins
4. **Consciousness Analysis**: Apply 5-level concept extraction to spec documents
5. **Knowledge Graph Visualization**: See specifications in 3D graph with relationships

## How

### High-Level Flow

```
OpenSpec Specs (Markdown)
├── openspec/specs/document-processing/spec.md
├── openspec/specs/clause-multi-agent/spec.md
└── openspec/specs/knowledge-graph/spec.md
         │
         │ (Scan & Upload)
         ▼
POST /api/documents (with metadata: source_type="openspec")
         │
         ▼
Daedalus Gateway
├── receive_perceptual_information()
└── Forward to LangGraph
         │
         ▼
DocumentProcessingGraph (6 nodes)
├── Extract & Process
├── Research (ASI-GO-2)
├── Consciousness (5-level concepts, basins, thoughtseeds)
├── Analyze Results
├── Refine Processing
└── Finalize Output
         │
         ▼
DocumentRepository → Neo4j
├── Document nodes (label: Specification)
├── Concept nodes (extracted requirements)
├── ThoughtSeed nodes (cross-spec connections)
└── AttractorBasin nodes (spec domains)
```

### Implementation Approach

**Option A: CLI Command** (Recommended for MVP)
```bash
# Ingest all specs
python backend/scripts/ingest_openspec_specs.py

# Ingest specific capability
python backend/scripts/ingest_openspec_specs.py --capability document-processing
```

**Option B: Automated Watch** (Future Enhancement)
- File watcher on `openspec/specs/`
- Auto-ingest on spec.md changes
- Triggered via git hooks or CI/CD

**Option C: API Endpoint**
```bash
curl -X POST http://localhost:9127/api/admin/ingest-specs
```

### Data Enrichment

**Spec-Specific Metadata**:
```python
{
    "source_type": "openspec",
    "capability": "document-processing",
    "spec_type": "spec" | "design",
    "change_id": "add-feature-xyz",  # If from change proposal
    "version": "1.0",
    "requirements_count": 8
}
```

**Neo4j Node Labels**:
- `Document:Specification` (for spec.md)
- `Document:DesignDocument` (for design.md)
- `Concept:Requirement` (extracted from ### Requirement:)
- `Concept:Scenario` (extracted from #### Scenario:)

**Relationships**:
```cypher
(:Specification)-[:DEFINES_CAPABILITY]->(:Capability {name: "document-processing"})
(:Specification)-[:HAS_REQUIREMENT]->(:Requirement {title: "Import OpenSpec..."})
(:Requirement)-[:HAS_SCENARIO]->(:Scenario {given: "...", when: "...", then: "..."})
(:Specification)-[:SIMILAR_TO]->(:Specification)  # Via ThoughtSeeds
```

## Impact

### Use Cases Enabled

1. **Spec Discovery**
   ```
   Query: "Find all specs that deal with authentication"
   Result: Returns document-processing, clause-multi-agent specs (both mention auth)
   ```

2. **Requirement Traceability**
   ```
   Query: "Which specs have requirements about Neo4j?"
   Result: knowledge-graph/spec.md, document-processing/spec.md (persistence)
   ```

3. **Design Pattern Mining**
   ```
   Query: "Show all design patterns related to LangGraph"
   Result: Document-processing design.md, consciousness design.md
   ```

4. **Spec Evolution**
   ```
   Track: How "document-processing" spec evolved from version 1.0 → 2.0
   Result: Show concept drift, new requirements added, deprecated features
   ```

5. **Cross-Spec Consistency**
   ```
   Detect: Multiple specs define "authentication" differently
   Result: Flag potential conflicts, suggest harmonization
   ```

### Developer Workflow

**Before** (Specs isolated):
```bash
# Search specs manually
grep -r "authentication" openspec/specs/

# No semantic search
# No cross-spec relationships
# No consciousness analysis
```

**After** (Specs in knowledge graph):
```bash
# Ingest specs once
python backend/scripts/ingest_openspec_specs.py

# Search semantically via API
curl http://localhost:9127/api/query \
  -d '{"query": "authentication patterns across specs"}'

# View in 3D graph
# See ThoughtSeed connections between specs
# Track AttractorBasins (spec domains)
```

## Risks & Mitigations

**Risk 1: Spec File Duplication**
- Scenario: Same spec ingested multiple times
- Mitigation: Use content hash (SHA-256) for deduplication, return 409 on duplicate

**Risk 2: Large Spec Documents**
- Scenario: design.md with 1000+ lines exceeds processing limits
- Mitigation: Chunk large documents, process in segments

**Risk 3: OpenSpec Format Changes**
- Scenario: OpenSpec updates format, breaks parser
- Mitigation: Version metadata field, support multiple format versions

**Risk 4: Stale Data**
- Scenario: Spec.md updated, Neo4j has old version
- Mitigation: Re-ingest on change (watch mode), version tracking

## Success Criteria

- [ ] Ingest script processes all specs in `openspec/specs/`
- [ ] Each spec.md creates Document:Specification node in Neo4j
- [ ] Metadata includes source_type="openspec", capability name
- [ ] Requirements and scenarios extracted as Concept nodes
- [ ] ThoughtSeeds connect related specifications
- [ ] Semantic search returns relevant specs via `/api/query`
- [ ] Deduplication via content hash prevents duplicate ingestion
- [ ] Documentation added to backend/README.md
