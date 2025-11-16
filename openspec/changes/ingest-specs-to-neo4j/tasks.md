# Implementation Tasks

## Phase 1: Ingestion Script (Core Pipeline)
- [ ] Create `backend/scripts/ingest_openspec_specs.py` script
- [ ] Implement spec file scanner (find all spec.md, design.md in openspec/specs/)
- [ ] Add file reader with metadata extraction (capability name, spec type)
- [ ] Integrate with existing document upload endpoint (POST /api/documents)
- [ ] Add source_type="openspec" metadata to requests
- [ ] Test ingestion with 3 existing capability specs
- [ ] Verify Document nodes created in Neo4j with correct labels

## Phase 2: Enhanced Metadata & Schema
- [ ] Define OpenSpec-specific metadata schema (capability, spec_type, version)
- [ ] Add Neo4j node labels: Document:Specification, Document:DesignDocument
- [ ] Extract requirements from ### Requirement: headers as Concept nodes
- [ ] Extract scenarios from #### Scenario: blocks with Given/When/Then
- [ ] Create relationships: HAS_REQUIREMENT, HAS_SCENARIO, DEFINES_CAPABILITY
- [ ] Test requirement/scenario extraction with clause-multi-agent spec
- [ ] Validate graph structure with Cypher queries

## Phase 3: Deduplication & Versioning
- [ ] Implement content hash (SHA-256) calculation for spec files
- [ ] Check for existing Document with same content_hash before ingest
- [ ] Return 409 Conflict on duplicate with message "Spec already ingested"
- [ ] Add version field to metadata (default: "1.0")
- [ ] Support re-ingestion with version increment (1.0 → 1.1)
- [ ] Test deduplication: ingest same spec twice, verify single node
- [ ] Test versioning: modify spec, re-ingest, verify version 1.1

## Phase 4: CLI & Automation
- [ ] Add CLI argument parsing (--capability, --all, --watch)
- [ ] Implement selective ingestion: `--capability document-processing`
- [ ] Add progress reporting (X/Y specs processed)
- [ ] Create `--watch` mode for file monitoring (future enhancement)
- [ ] Add `--dry-run` flag to preview without ingestion
- [ ] Test CLI: `python ingest_openspec_specs.py --all`
- [ ] Test selective: `python ingest_openspec_specs.py --capability knowledge-graph`

## Phase 5: Search & Query Integration
- [ ] Verify semantic search returns OpenSpec specs via /api/query
- [ ] Test query: "Find specs about authentication"
- [ ] Verify ThoughtSeeds connect related specifications
- [ ] Test cross-spec relationships in graph visualization
- [ ] Add spec-specific filters to query API (source_type="openspec")
- [ ] Test hybrid search: combine full-text + vector search for specs
- [ ] Validate attractor basins group spec domains correctly

## Phase 6: Documentation & Maintenance
- [ ] Document ingestion script in backend/README.md
- [ ] Add usage examples for CLI flags
- [ ] Create troubleshooting guide (common errors, fixes)
- [ ] Add integration test for spec ingestion pipeline
- [ ] Document metadata schema in openspec/AGENTS.md
- [ ] Create runbook for re-ingesting updated specs

## Dependencies
- Daedalus Gateway operational (backend/src/services/daedalus.py)
- DocumentProcessingGraph working (LangGraph 6-node pipeline)
- DocumentRepository connected to Neo4j
- OpenSpec specs migrated (document-processing, clause-multi-agent, knowledge-graph)

## Validation
- Ingest all 3 capability specs
- Query Neo4j: `MATCH (s:Specification) RETURN count(s)` → expect 3
- Search semantically: "multi-agent coordination" → returns clause-multi-agent spec
- Verify ThoughtSeeds created between related specs
