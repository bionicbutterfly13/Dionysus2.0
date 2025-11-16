# Spec Ingestion Pipeline

## Overview

Automated pipeline to ingest OpenSpec specification documents (spec.md, design.md) into Neo4j through the existing Daedalus → LangGraph → DocumentRepository consciousness pipeline, enabling semantic search, cross-spec relationship discovery, and knowledge graph visualization of specifications.

## ADDED Requirements

### Requirement: Scan OpenSpec Specs for Ingestion
The system SHALL scan the `openspec/specs/` directory to discover all spec.md and design.md files for ingestion into the knowledge graph.

#### Scenario: Scan all capabilities
**Given** `openspec/specs/` contains 3 capability directories (document-processing, clause-multi-agent, knowledge-graph)
**And** each capability has spec.md and design.md files (6 files total)
**When** ingestion script runs with `--all` flag
**Then** the system discovers all 6 specification files with correct metadata (capability name, spec type)

#### Scenario: Scan specific capability
**Given** `openspec/specs/document-processing/` contains spec.md and design.md
**When** ingestion script runs with `--capability document-processing`
**Then** the system discovers only 2 files from document-processing capability

#### Scenario: Handle empty specs directory
**Given** `openspec/specs/` exists but contains no capability directories
**When** ingestion script runs
**Then** the system reports "0 spec files found" without errors

---

### Requirement: Ingest Specs via Document Processing Pipeline
The system SHALL ingest OpenSpec specification files through the existing POST /api/documents endpoint, processing them via Daedalus → LangGraph → DocumentRepository pipeline.

#### Scenario: Ingest spec.md as Document
**Given** `openspec/specs/document-processing/spec.md` contains specification content
**When** ingestion script POSTs file to /api/documents with metadata `{source_type: "openspec", capability: "document-processing", spec_type: "spec"}`
**Then** the system:
- Processes file through Daedalus Gateway
- Runs DocumentProcessingGraph (6 nodes: extract, research, consciousness, analyze, refine, finalize)
- Creates Document:Specification node in Neo4j
- Returns 200 OK with document_id

#### Scenario: Ingest design.md as DesignDocument
**Given** `openspec/specs/clause-multi-agent/design.md` contains design patterns
**When** ingestion script POSTs file with metadata `{spec_type: "design"}`
**Then** the system creates Document:DesignDocument node with design-specific metadata

#### Scenario: API endpoint rejects invalid file
**Given** ingestion script POSTs empty file
**When** /api/documents receives request
**Then** the system returns 400 Bad Request with error "Empty file content"

---

### Requirement: Deduplicate Specs by Content Hash
The system SHALL prevent duplicate ingestion of identical specification files using SHA-256 content hashing.

#### Scenario: Detect duplicate spec
**Given** spec.md with content "# Document Processing..." has been ingested (content_hash: "abc123...")
**When** same file is ingested again with identical content
**Then** the system:
- Calculates SHA-256 hash "abc123..."
- Finds existing Document node with same content_hash
- Returns 409 Conflict with message "Spec already ingested"
- Does not create duplicate node

#### Scenario: Ingest updated spec as new version
**Given** spec.md v1.0 has content_hash "abc123..."
**When** updated spec.md with modified content (content_hash "def456...") is ingested
**Then** the system:
- Recognizes different content_hash
- Creates new Document node with version "1.1"
- Links versions: (v1.0)-[:NEXT_VERSION]->(v1.1)

---

### Requirement: Extract Requirements and Scenarios from Specs
The system SHALL parse OpenSpec spec.md files to extract requirements and scenarios as structured Concept nodes in the knowledge graph.

#### Scenario: Extract requirement with scenarios
**Given** spec.md contains:
```markdown
### Requirement: Import OpenSpec Change to Archon
The system SHALL import...

#### Scenario: Import change with multiple tasks
**Given** OpenSpec change exists
**When** developer runs command
**Then** system creates Archon project
```
**When** DocumentProcessingGraph processes the spec
**Then** the system creates:
- Concept:Requirement node with title "Import OpenSpec Change to Archon"
- Concept:Scenario node with title "Import change with multiple tasks"
- Relationship: (Specification)-[:HAS_REQUIREMENT]->(Requirement)
- Relationship: (Requirement)-[:HAS_SCENARIO]->(Scenario)

#### Scenario: Handle spec without requirements
**Given** design.md contains only architecture descriptions (no ### Requirement: headers)
**When** system processes design.md
**Then** no Requirement nodes are created (only Document:DesignDocument node)

---

### Requirement: Enable Semantic Search Across Specifications
The system SHALL support semantic search queries that return relevant OpenSpec specifications from the knowledge graph.

#### Scenario: Search for specs by keyword
**Given** 3 specifications ingested: document-processing, clause-multi-agent, knowledge-graph
**And** document-processing spec mentions "LangGraph workflow"
**When** user queries `/api/query` with "LangGraph patterns"
**Then** the system returns document-processing spec with relevance score > 0.8

#### Scenario: Filter search by source type
**Given** Neo4j contains both user documents and OpenSpec specifications
**When** user queries with filter `{source_type: "openspec"}`
**Then** the system returns only specification documents, excluding user documents

#### Scenario: Search across requirements
**Given** clause-multi-agent spec has requirement "Multi-Agent Coordination"
**When** user queries "agent coordination patterns"
**Then** the system returns clause-multi-agent spec with matched concept highlighted

---

### Requirement: Discover Cross-Spec Relationships via ThoughtSeeds
The system SHALL generate ThoughtSeed nodes that connect related specifications based on shared concepts and patterns.

#### Scenario: ThoughtSeeds link related specs
**Given** document-processing spec mentions "Neo4j persistence"
**And** knowledge-graph spec mentions "Neo4j architecture"
**When** consciousness processing runs on both specs
**Then** the system creates ThoughtSeed connecting both specs with concept "Neo4j integration patterns"

#### Scenario: Visualize spec relationships
**Given** ThoughtSeeds link document-processing ↔ clause-multi-agent ↔ knowledge-graph
**When** user views knowledge graph visualization
**Then** the system displays 3 Specification nodes connected via ThoughtSeed relationships

---

### Requirement: Store OpenSpec-Specific Metadata
The system SHALL enrich specification Document nodes with OpenSpec-specific metadata fields for filtering and traceability.

#### Scenario: Store capability metadata
**Given** ingesting spec from `openspec/specs/document-processing/spec.md`
**When** Document node is created
**Then** node properties include:
- `source_type: "openspec"`
- `capability: "document-processing"`
- `spec_type: "spec"`
- `version: "1.0"`
- `content_hash: "abc123..."`

#### Scenario: Query specs by capability
**Given** 3 specs ingested with different capabilities
**When** Cypher query `MATCH (s:Specification {capability: "clause-multi-agent"}) RETURN s`
**Then** the system returns only clause-multi-agent spec

---

## Non-Functional Requirements

### Performance
- Ingestion throughput: < 10 seconds for 6 spec files (3 capabilities × 2 files)
- Semantic search latency: < 500ms for search across all specs
- Deduplication check: < 100ms hash comparison query

### Reliability
- Atomic ingestion: File either fully ingested or transaction rolled back
- Idempotent deduplication: Multiple runs with same files produce same graph state
- Error recovery: API failures don't corrupt existing graph data

### Usability
- CLI progress reporting: "Ingesting X/Y specs... ✓ Success"
- Clear error messages: Duplicate detection shows existing document_id
- Dry-run mode: Preview files without actual ingestion

### Compatibility
- Backward compatible: Existing document processing pipeline unaffected
- Forward compatible: Additional metadata fields supported via flexible schema
- Git-friendly: No modification of source spec files

---

## Acceptance Criteria

- [ ] Ingestion script processes all specs in `openspec/specs/` directory
- [ ] Each spec.md creates Document:Specification node with correct labels
- [ ] Metadata includes source_type="openspec", capability, spec_type, version
- [ ] Requirements extracted as Concept:Requirement nodes with HAS_REQUIREMENT relationships
- [ ] Scenarios extracted as Concept:Scenario nodes with HAS_SCENARIO relationships
- [ ] Content hash deduplication prevents duplicate ingestion (returns 409)
- [ ] Semantic search via /api/query returns relevant specifications
- [ ] ThoughtSeeds connect related specs in knowledge graph
- [ ] CLI supports --all, --capability, --dry-run flags
- [ ] Integration test validates end-to-end ingestion → search workflow
- [ ] Documentation added to backend/README.md with usage examples
