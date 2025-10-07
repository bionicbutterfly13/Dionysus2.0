# Tasks: Document Persistence & Repository

**Input**: Design documents from `/Volumes/Asylum/dev/Dionysus-2.0/specs/054-document-persistence-repository/`
**Prerequisites**: [plan.md](./plan.md) (complete), [spec.md](./spec.md) (complete)

## Execution Flow (main)
```
1. Load plan.md from feature directory ✅
   → Extract: tech stack (Python, Neo4j, Graph Channel), structure (backend/src/)
2. Load spec.md ✅
   → Extract: 30 functional requirements, 6 key entities
3. Generate tasks by category:
   → Setup: Graph Channel validation, constitutional compliance
   → Context Engineering: Basin/field integration (MANDATORY FIRST)
   → Tests: Contract tests for APIs, constitutional compliance tests
   → Core: DocumentRepository service, schema, persistence logic
   → Integration: Tier management, archival, performance
   → Polish: Load testing, monitoring, optimization
4. Apply task rules:
   → Context Engineering validation BEFORE core work
   → Tests before implementation (TDD)
   → Different files = mark [P] for parallel
   → Same file = sequential (no [P])
5. Number tasks sequentially (T001-T057)
6. Return: SUCCESS (tasks ready for execution)
```

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths in descriptions

## Phase 1: Setup & Context Engineering Validation (MANDATORY FIRST)

### Context Engineering Validation (T001-T005)
⚠️ **CRITICAL**: These tasks MUST complete successfully before any core implementation

- [ ] **T001** [P] Verify attractor basin integration available
  - File: `extensions/context_engineering/attractor_basin_dynamics.py`
  - Validate: `AttractorBasinDynamics` class can be imported and instantiated
  - Test: Create test basin, verify strength calculations work
  - Blocker: If unavailable, create stub implementation for Spec 054

- [ ] **T002** [P] Verify neural field system integration available
  - File: `dionysus-source/agents/neural_field_graph_dynamics.py`
  - Validate: Neural field resonance scoring available
  - Test: Calculate field resonance between two test concepts
  - Blocker: If unavailable, use fallback resonance scoring (cosine similarity)

- [ ] **T003** [P] Validate Redis persistence for basin state storage
  - Connect to Redis on localhost:6379
  - Test: Write basin state, read back, verify persistence
  - Verify: Basin evolution history can be appended incrementally
  - Blocker: If Redis unavailable, use in-memory fallback with warning

- [ ] **T004** [P] Validate DaedalusGraphChannel availability
  - File: Check `daedalus-gateway` package or `backend/src/services/graph_channel.py`
  - Test: `from daedalus_gateway import get_graph_channel`
  - Verify: `execute_write()` and `execute_read()` methods available
  - Document: Graph Channel connection params (NEO4J_URI, auth)

- [ ] **T005** Create DocumentRepository project structure
  - Create: `backend/src/services/document_repository.py` (empty scaffold)
  - Create: `backend/src/models/attractor_basin.py` (model definitions)
  - Create: `backend/tests/services/test_document_repository.py` (test scaffold)
  - Create: `backend/tests/integration/test_document_persistence.py` (integration scaffold)
  - Create: `backend/src/services/tier_manager.py` (tier management scaffold)

### Constitutional Compliance Setup (T006-T008)

- [ ] **T006** [P] Configure constitutional compliance linter for new files
  - File: `backend/.ruff_constitutional_plugin.py` (already exists, verify active)
  - Test: Run linter on empty `document_repository.py` → should pass
  - Verify: `CONST001` and `CONST002` rules active

- [ ] **T007** [P] Create constitutional compliance test for DocumentRepository
  - File: `backend/tests/test_constitutional_compliance_spec054.py`
  - Test: Scan all Spec 054 files for banned neo4j imports
  - Assert: Only `from daedalus_gateway import get_graph_channel` allowed
  - Assert: No `from neo4j import` or `import neo4j` statements

- [ ] **T008** Initialize Python dependencies for Spec 054
  - Add to `backend/requirements.txt`:
    - `daedalus-gateway` (Graph Channel)
    - `redis` (basin persistence)
    - `boto3` (S3 archival, optional)
  - Run: `pip install -r backend/requirements.txt`
  - Verify: All imports successful

## Phase 2: Tests First (TDD) - MUST FAIL BEFORE IMPLEMENTATION

### API Contract Tests (T009-T012)
⚠️ **CRITICAL**: Write tests, verify they FAIL, then implement

- [ ] **T009** [P] Contract test POST /api/documents/persist
  - File: `backend/tests/contract/test_documents_persist_post.py`
  - Test: POST with Daedalus final_output → 201 Created
  - Test: POST with duplicate content_hash → 409 Conflict
  - Test: POST missing required fields → 400 Bad Request
  - Assert: Response includes document_id, persistence metrics
  - **VERIFY FAILS** before implementing endpoint

- [ ] **T010** [P] Contract test GET /api/documents
  - File: `backend/tests/contract/test_documents_list_get.py`
  - Test: GET with pagination → 200 OK with documents array
  - Test: GET with filters (tags, quality_min, date_range) → filtered results
  - Test: GET with sort (upload_date, quality, curiosity) → sorted results
  - Test: Performance <500ms for 100 documents
  - **VERIFY FAILS** before implementing endpoint

- [ ] **T011** [P] Contract test GET /api/documents/{id}
  - File: `backend/tests/contract/test_documents_detail_get.py`
  - Test: GET existing document → 200 OK with full detail
  - Test: GET includes concepts (all 5 levels), basins, thoughtseeds
  - Test: GET non-existent document → 404 Not Found
  - Test: Verify access tracking (access_count increments)
  - **VERIFY FAILS** before implementing endpoint

- [ ] **T012** [P] Contract test PUT /api/documents/{id}/tier
  - File: `backend/tests/contract/test_documents_tier_put.py`
  - Test: PUT valid tier (warm/cool/cold) → 200 OK
  - Test: PUT invalid tier → 400 Bad Request
  - Test: PUT to cold tier → archives to S3/filesystem
  - Test: Verify tier_changed_at timestamp updated
  - **VERIFY FAILS** before implementing endpoint

### Integration Tests (T013-T018)

- [ ] **T013** [P] Integration test: Full document persistence flow
  - File: `backend/tests/integration/test_document_persistence.py`
  - Test: Daedalus final_output → persist_document() → verify Neo4j nodes created
  - Assert: Document node, 5-level concepts, basins, thoughtseeds all present
  - Assert: Relationships (EXTRACTED_FROM, ATTRACTED_TO, GERMINATED_FROM) created
  - Assert: Persistence completes in <2s
  - **VERIFY FAILS** before implementing DocumentRepository

- [ ] **T014** [P] Integration test: Context Engineering basin evolution
  - File: `backend/tests/integration/test_basin_evolution.py`
  - Test: Persist document with basin influence → verify basin strength updated
  - Assert: Basin influence_history appended to Redis
  - Assert: ATTRACTED_TO relationship includes influence_type and strength_delta
  - Assert: Subsequent documents modify existing basin strength
  - **VERIFY FAILS** before implementing basin persistence

- [ ] **T015** [P] Integration test: Tier migration (hybrid age + access)
  - File: `backend/tests/integration/test_tier_migration.py`
  - Test: Create warm document, age 30 days, access ≤5 times → migrates to cool
  - Test: Create cool document, age 90 days, access ≤2 times → migrates to cold
  - Test: Frequently accessed old document stays warm
  - Assert: Tier migration rules follow hybrid criteria
  - **VERIFY FAILS** before implementing tier management

- [ ] **T016** [P] Integration test: Cold tier archival
  - File: `backend/tests/integration/test_cold_tier_archival.py`
  - Test: Archive document to cold tier → full data written to S3/filesystem
  - Assert: Document metadata remains in Neo4j
  - Assert: archive_location and archived_at fields set
  - Test: Retrieval from cold tier slower but functional
  - **VERIFY FAILS** before implementing archival

- [ ] **T017** [P] Integration test: Constitutional compliance (Graph Channel only)
  - File: `backend/tests/integration/test_graph_channel_compliance.py`
  - Test: All Neo4j operations route through DaedalusGraphChannel
  - Assert: Audit trail (caller_service, caller_function) on every operation
  - Assert: Circuit breaker triggers after 5 consecutive failures
  - Assert: Retry logic (3 attempts, exponential backoff) active
  - **VERIFY FAILS** before implementing Graph Channel integration

- [ ] **T018** [P] Integration test: Performance targets
  - File: `backend/tests/integration/test_performance_targets.py`
  - Test: Persist 100 documents → average <2s per document
  - Test: List 100 documents → <500ms response time
  - Test: Get document detail → <200ms response time
  - Load test: 10,000 documents, verify performance holds
  - **VERIFY FAILS** before implementing performance optimizations

## Phase 3: Core Implementation (ONLY AFTER TESTS FAIL)

### Neo4j Schema (T019-T022)

- [ ] **T019** Create Neo4j schema initialization script
  - File: `backend/src/services/neo4j_schema_init.py`
  - Create uniqueness constraints:
    - `CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.document_id IS UNIQUE`
    - `CREATE CONSTRAINT concept_id_unique IF NOT EXISTS FOR (c:Concept) REQUIRE c.concept_id IS UNIQUE`
    - `CREATE CONSTRAINT basin_id_unique IF NOT EXISTS FOR (b:AttractorBasin) REQUIRE b.basin_id IS UNIQUE`
    - `CREATE CONSTRAINT seed_id_unique IF NOT EXISTS FOR (t:ThoughtSeed) REQUIRE t.seed_id IS UNIQUE`
  - All operations via Graph Channel (constitutional compliance)

- [ ] **T020** Create Neo4j performance indexes
  - File: `backend/src/services/neo4j_schema_init.py` (extend T019)
  - Create indexes:
    - `CREATE INDEX document_upload_timestamp IF NOT EXISTS FOR (d:Document) ON (d.upload_timestamp)`
    - `CREATE INDEX document_quality IF NOT EXISTS FOR (d:Document) ON (d.quality_overall)`
    - `CREATE INDEX document_tier IF NOT EXISTS FOR (d:Document) ON (d.tier)`
    - `CREATE INDEX document_tags IF NOT EXISTS FOR (d:Document) ON (d.tags)`
    - `CREATE INDEX concept_level IF NOT EXISTS FOR (c:Concept) ON (c.level)`
    - `CREATE INDEX concept_salience IF NOT EXISTS FOR (c:Concept) ON (c.salience)`
  - Run via Graph Channel execute_write()

- [ ] **T021** Define node type models
  - File: `backend/src/models/document_node.py`
  - Pydantic models: DocumentNode, ConceptNode, AttractorBasinNode, ThoughtSeedNode
  - Include all properties from plan.md schema (metadata, quality, tier, etc.)
  - Validation: Required fields, type checks, range constraints

- [ ] **T022** [P] Define relationship models
  - File: `backend/src/models/document_relationships.py`
  - Pydantic models: ExtractedFromRel, AttractedToRel, GerminatedFromRel, DerivedFromRel, ResonatesWithRel
  - Include relationship properties (confidence, activation_strength, field_score, etc.)

### DocumentRepository Core Service (T023-T030)

- [ ] **T023** Create DocumentRepository class scaffold
  - File: `backend/src/services/document_repository.py`
  - Initialize with DaedalusGraphChannel dependency injection
  - Methods: persist_document(), get_document(), list_documents(), update_tier()
  - Constitutional compliance: ONLY `from daedalus_gateway import get_graph_channel`
  - NO direct neo4j imports

- [ ] **T024** Implement persist_document() - validation and transaction start
  - File: `backend/src/services/document_repository.py` (extend T023)
  - Validate required fields (document_id, content_hash, filename)
  - Check for duplicates (content_hash lookup via Graph Channel)
  - Return 409 if duplicate found
  - Start transaction context

- [ ] **T025** Implement persist_document() - Document node creation
  - File: `backend/src/services/document_repository.py` (extend T024)
  - Create :Document node via Graph Channel execute_write()
  - Set all metadata properties (filename, upload_timestamp, file_size, mime_type, tags)
  - Set processing metadata (processed_at, processing_duration_ms, processing_status)
  - Set quality metrics (quality_overall, quality_coherence, quality_novelty, quality_depth)
  - Set initial tier="warm", last_accessed=now(), access_count=0
  - Cypher query from plan.md lines 165-200

- [ ] **T026** Implement persist_document() - 5-level concept persistence
  - File: `backend/src/services/document_repository.py` (extend T025)
  - Method: `_persist_concepts(concepts_data, document_id)` (private helper)
  - For each level (atomic, relationship, composite, context, narrative):
    - Create :Concept nodes with level-specific labels (AtomicConcept, etc.)
    - Set properties: concept_id, name, level, salience, definition/components/etc.
    - Create [:EXTRACTED_FROM] relationships → Document
    - All via Graph Channel with caller_service="document_repository", caller_function="_persist_concepts"
  - Reference plan.md lines 693-739

- [ ] **T027** Implement persist_document() - Attractor basin persistence with Context Engineering
  - File: `backend/src/services/document_repository.py` (extend T026)
  - Method: `_persist_basins(basins_data, document_id)` (private helper)
  - For each basin:
    - MERGE :AttractorBasin node (create or update existing)
    - ON CREATE: Set name, depth, stability, strength, created_at
    - ON MATCH: Update depth, stability, increment strength by strength_delta
    - Create [:ATTRACTED_TO] relationship → Document with activation_strength, influence_type, strength_delta
    - Update AttractorBasinManager in Redis (basin evolution history)
  - Reference plan.md lines 760-823
  - Dependencies: T001 (basin integration validated)

- [ ] **T028** Implement persist_document() - ThoughtSeed persistence
  - File: `backend/src/services/document_repository.py` (extend T027)
  - Method: `_persist_thoughtseeds(thoughtseeds_data, document_id)` (private helper)
  - For each thoughtseed:
    - Create :ThoughtSeed node with seed_id, content, germination_potential, resonance_score
    - Store field_resonance data (energy, phase, interference_pattern) from Neural Field System
    - Create [:GERMINATED_FROM] relationship → Document
    - All via Graph Channel
  - Dependencies: T002 (neural field integration validated)

- [ ] **T029** Implement persist_document() - Transaction commit and performance monitoring
  - File: `backend/src/services/document_repository.py` (extend T028)
  - Commit transaction (all or nothing)
  - Add performance monitoring wrapper (plan.md lines 1209-1249)
  - Verify <2s performance target
  - Return persistence result with metrics (duration_ms, met_target)
  - Log warning if performance target missed

- [ ] **T030** Implement get_document() - Full detail retrieval
  - File: `backend/src/services/document_repository.py` (extend T023)
  - Single comprehensive Cypher query to minimize round trips
  - Fetch: Document node, all concepts (grouped by level), all basins, all thoughtseeds
  - Update access tracking: last_accessed=now(), increment access_count
  - Return: Structured response matching API contract (plan.md lines 533-582)
  - Reference plan.md lines 983-1075

### Document Listing API (T031-T033)

- [ ] **T031** Implement list_documents() - Query building
  - File: `backend/src/services/document_repository.py` (extend T023)
  - Build dynamic Cypher query with WHERE clauses for filters:
    - tags (array intersection)
    - quality_min (>=)
    - date_from/date_to (datetime range)
    - tier (exact match)
  - Build ORDER BY clause for sorting (upload_date, quality, curiosity_triggers)
  - Reference plan.md lines 842-936

- [ ] **T032** Implement list_documents() - Pagination logic
  - File: `backend/src/services/document_repository.py` (extend T031)
  - Count total matching documents (separate query)
  - Calculate skip/limit for pagination (page, limit params)
  - Include counts (concept_count, basin_count, thoughtseed_count) in listing
  - Return pagination metadata (page, limit, total, total_pages)

- [ ] **T033** Implement list_documents() - Performance optimization
  - File: `backend/src/services/document_repository.py` (extend T032)
  - Use indexes for filtering (document_upload_timestamp, document_quality, document_tier, document_tags)
  - Add query caching for common filters (via Redis)
  - Add performance monitoring (<500ms target for 100 documents)
  - Optimize: Fetch minimal data in listing (no full concept/basin details)

## Phase 4: Tier Management & Archival (T034-T040)

### Tier Migration Logic (T034-T037)

- [ ] **T034** Create TierManager class
  - File: `backend/src/services/tier_manager.py`
  - Dependencies: DaedalusGraphChannel, DocumentRepository
  - Methods: evaluate_tier_migrations(), update_tier(), archive_to_cold_tier()
  - Hybrid tier rules (age + access patterns) from clarifications

- [ ] **T035** Implement evaluate_tier_migrations() - Warm to Cool
  - File: `backend/src/services/tier_manager.py` (extend T034)
  - Cypher query: Find warm documents where:
    - age >= 30 days AND
    - access_count <= 5 AND
    - days_since_last_access >= 14
  - For each eligible document: call update_tier(document_id, "cool", "automatic_age_access_hybrid")
  - Reference plan.md lines 1094-1128

- [ ] **T036** Implement evaluate_tier_migrations() - Cool to Cold
  - File: `backend/src/services/tier_manager.py` (extend T035)
  - Cypher query: Find cool documents where:
    - age >= 90 days AND
    - access_count <= 2 AND
    - days_since_last_access >= 60
  - For each eligible document: call archive_to_cold_tier(document_id)
  - Reference plan.md lines 1130-1147

- [ ] **T037** Implement update_tier() - Tier update logic
  - File: `backend/src/services/tier_manager.py` (extend T034)
  - Cypher UPDATE: Set tier, tier_changed_at timestamp
  - Emit metrics (tier transition counts)
  - Log tier change with reason (automatic vs manual)

### Cold Tier Archival (T038-T040)

- [ ] **T038** Implement archive_to_cold_tier() - Fetch and archive
  - File: `backend/src/services/tier_manager.py` (extend T034)
  - Fetch complete document data via get_document()
  - Call _write_to_archive() to store in S3/filesystem
  - Return archive_location (s3://bucket/key or /path/to/file)
  - Reference plan.md lines 1149-1180

- [ ] **T039** Implement _write_to_archive() - S3/filesystem storage
  - File: `backend/src/services/tier_manager.py` (extend T038)
  - Option A: S3 via boto3 (if S3 configured)
  - Option B: Filesystem (/archive/ directory)
  - Write full document JSON to archive
  - Return archive_location string

- [ ] **T040** Implement archive_to_cold_tier() - Update metadata
  - File: `backend/src/services/tier_manager.py` (extend T038)
  - Cypher UPDATE: Set tier="cold", archived_at=now(), archive_location
  - Optional: Remove full concept/basin/seed content (keep references only)
  - Verify: Metadata remains in Neo4j for discovery

### API Endpoints (T041-T044)

- [ ] **T041** [P] Create POST /api/documents/persist endpoint
  - File: `backend/src/api/routes/documents.py`
  - FastAPI route: POST /api/documents/persist
  - Request body: {document_id, filename, content_hash, daedalus_output}
  - Call: DocumentRepository.persist_document()
  - Response: 201 Created with document_id, performance metrics
  - Error handling: 409 Conflict (duplicate), 400 Bad Request (validation)

- [ ] **T042** [P] Create GET /api/documents endpoint
  - File: `backend/src/api/routes/documents.py`
  - FastAPI route: GET /api/documents
  - Query params: page, limit, tags, quality_min, date_from, date_to, sort, order, tier
  - Call: DocumentRepository.list_documents()
  - Response: 200 OK with documents array, pagination metadata
  - Verify: <500ms performance for 100 documents

- [ ] **T043** [P] Create GET /api/documents/{id} endpoint
  - File: `backend/src/api/routes/documents.py`
  - FastAPI route: GET /api/documents/{id}
  - Path param: document_id
  - Call: DocumentRepository.get_document()
  - Response: 200 OK with full document detail
  - Error handling: 404 Not Found

- [ ] **T044** [P] Create PUT /api/documents/{id}/tier endpoint
  - File: `backend/src/api/routes/documents.py`
  - FastAPI route: PUT /api/documents/{id}/tier
  - Request body: {new_tier, reason}
  - Call: TierManager.update_tier() or archive_to_cold_tier()
  - Response: 200 OK with tier change confirmation
  - Error handling: 400 Bad Request (invalid tier)

## Phase 5: Background Jobs & Integration (T045-T048)

- [ ] **T045** Create background tier migration job
  - File: `backend/src/jobs/tier_migration_job.py`
  - Schedule: Run every 6 hours (configurable)
  - Call: TierManager.evaluate_tier_migrations()
  - Logging: Report tier transitions (warm→cool, cool→cold counts)
  - Error handling: Circuit breaker on repeated failures

- [ ] **T046** Integrate AttractorBasinManager with Redis
  - File: `backend/src/services/document_repository.py` (extend T027)
  - Method: `_update_basin_manager(basin, document_id)` (private helper)
  - Store basin evolution history in Redis:
    - Key: `basin:evolution:{basin_id}`
    - Value: Append influence event {document_id, influence_type, strength_delta, timestamp}
  - TTL: 90 days (configurable)
  - Dependencies: T003 (Redis validated)

- [ ] **T047** [P] Add circuit breaker monitoring for Graph Channel
  - File: `backend/src/services/document_repository.py` (extend T023)
  - Monitor: DaedalusGraphChannel circuit breaker state
  - Emit metrics: Circuit open/closed events
  - Fallback: Return degraded response when circuit open

- [ ] **T048** [P] Add audit trail logging for all operations
  - File: `backend/src/services/document_repository.py` (extend T023)
  - Log every Graph Channel operation:
    - caller_service="document_repository"
    - caller_function (persist_document, get_document, etc.)
    - query (sanitized, no sensitive data)
    - duration_ms
  - Store in structured logs (JSON format)

## Phase 6: Polish & Optimization (T049-T057)

### Unit Tests (T049-T052)

- [ ] **T049** [P] Unit test: Document validation logic
  - File: `backend/tests/unit/test_document_validation.py`
  - Test: Required fields validation (document_id, content_hash, filename)
  - Test: Type validation (file_size int, quality float, etc.)
  - Test: Range validation (quality 0.0-1.0, file_size > 0)
  - Mock: No Graph Channel calls

- [ ] **T050** [P] Unit test: Tier migration rule calculations
  - File: `backend/tests/unit/test_tier_rules.py`
  - Test: Hybrid age + access patterns
  - Test: Warm→Cool eligibility (30 days, ≤5 accesses, 14 days idle)
  - Test: Cool→Cold eligibility (90 days, ≤2 accesses, 60 days idle)
  - Test: Frequently accessed documents stay warm
  - Mock: No Graph Channel calls

- [ ] **T051** [P] Unit test: Query building logic
  - File: `backend/tests/unit/test_query_builder.py`
  - Test: Dynamic WHERE clause generation
  - Test: Sort field mapping (upload_date, quality, curiosity)
  - Test: Pagination skip/limit calculation
  - Mock: No Graph Channel calls

- [ ] **T052** [P] Unit test: Performance monitoring wrapper
  - File: `backend/tests/unit/test_performance_monitoring.py`
  - Test: Duration calculation accurate
  - Test: met_target flag correct (<2s for persist, <500ms for list)
  - Test: Warning logged when target missed
  - Mock: Time module

### Performance Optimization (T053-T055)

- [ ] **T053** Add query caching for frequently accessed documents
  - File: `backend/src/services/document_repository.py` (extend T030)
  - Cache: get_document() results in Redis
  - TTL: 5 minutes (configurable)
  - Cache key: `document:detail:{document_id}`
  - Invalidate: On tier change, on access_count update

- [ ] **T054** Optimize pagination queries with index hints
  - File: `backend/src/services/document_repository.py` (extend T031)
  - Add USING INDEX hints to Cypher queries
  - Verify: Query planner uses document_upload_timestamp index for date sorting
  - Verify: Query planner uses document_tier index for tier filtering

- [ ] **T055** Load test with 10,000 documents
  - File: `backend/tests/load/test_load_10k_documents.py`
  - Generate: 10,000 test documents with concepts/basins/seeds
  - Persist all via persist_document()
  - Measure: Average persistence time (target <2s)
  - Measure: List 100 documents time (target <500ms)
  - Measure: Get document detail time (target <200ms)
  - Report: Performance metrics, bottlenecks identified

### Documentation & Final Validation (T056-T057)

- [ ] **T056** [P] Update API documentation
  - File: `backend/docs/api_document_persistence.md`
  - Document: All 4 endpoints (POST persist, GET list, GET detail, PUT tier)
  - Include: Request/response examples from plan.md
  - Include: Performance targets, error codes
  - Include: Constitutional compliance notes

- [ ] **T057** Run complete test suite and verify all pass
  - Run: All contract tests (T009-T012) → PASS
  - Run: All integration tests (T013-T018) → PASS
  - Run: All unit tests (T049-T052) → PASS
  - Run: Constitutional compliance test (T007) → PASS
  - Run: Load test (T055) → Performance targets met
  - Verify: No direct neo4j imports (only Graph Channel)
  - Verify: All operations have audit trail

## Dependencies

### Critical Path
```
Context Engineering Validation (T001-T003)
    ↓
Constitutional Compliance (T006-T008)
    ↓
Tests Written & Failing (T009-T018)
    ↓
Schema & Models (T019-T022)
    ↓
DocumentRepository Core (T023-T030)
    ↓
Listing API (T031-T033)
    ↓
Tier Management (T034-T040)
    ↓
API Endpoints (T041-T044)
    ↓
Integration & Jobs (T045-T048)
    ↓
Optimization & Polish (T049-T057)
```

### Detailed Dependencies
- **T001-T003** (Context Engineering) block **T027** (basin persistence), **T028** (thoughtseed field resonance)
- **T004** (Graph Channel) blocks **T019-T020** (schema init), **T023-T030** (all repository methods)
- **T007** (compliance test) blocks **T023** (DocumentRepository implementation)
- **T009-T018** (all tests) MUST FAIL before **T023-T044** (implementation)
- **T019-T020** (schema) block **T025** (document node creation)
- **T021-T022** (models) block **T025-T028** (node/relationship creation)
- **T023** (repository scaffold) blocks **T024-T030** (all repository methods)
- **T024-T028** (persistence methods) must complete before **T029** (transaction commit)
- **T030** (get_document) blocks **T038** (archive requires get_document)
- **T034** (TierManager) blocks **T035-T040** (all tier methods)
- **T041-T044** (endpoints) depend on **T023-T033** (repository methods)
- **T045** (background job) depends on **T034-T036** (tier migration logic)
- **T046** (basin manager) depends on **T001** (basin integration), **T003** (Redis)
- **T053** (caching) depends on **T003** (Redis), **T030** (get_document)

### Parallel Execution Groups
Group tasks by [P] marker - these can run in parallel:

**Group 1 (Context Engineering Validation)**:
```
T001, T002, T003, T004 (all can run in parallel)
```

**Group 2 (Setup)**:
```
T006, T007, T008 (all can run in parallel)
```

**Group 3 (Contract Tests - AFTER T005 scaffold)**:
```
T009, T010, T011, T012 (all different files)
```

**Group 4 (Integration Tests - AFTER T005 scaffold)**:
```
T013, T014, T015, T016, T017, T018 (all different files)
```

**Group 5 (Models - AFTER T019-T020 schema)**:
```
T021, T022 (different files)
```

**Group 6 (API Endpoints - AFTER T023-T033 repository complete)**:
```
T041, T042, T043, T044 (all different route functions)
```

**Group 7 (Integration & Monitoring - AFTER T034-T040 tier management)**:
```
T047, T048 (different concerns)
```

**Group 8 (Unit Tests - AFTER implementation complete)**:
```
T049, T050, T051, T052 (all different files)
```

**Group 9 (Final Polish)**:
```
T056 (docs can run while T055 load test runs)
```

## Parallel Execution Examples

### Example 1: Context Engineering Validation (Phase 1 Start)
Launch all validation tasks in parallel:
```
Task: "Verify attractor basin integration available in extensions/context_engineering/attractor_basin_dynamics.py"
Task: "Verify neural field system integration available in dionysus-source/agents/neural_field_graph_dynamics.py"
Task: "Validate Redis persistence for basin state storage"
Task: "Validate DaedalusGraphChannel availability from daedalus-gateway package"
```

### Example 2: Contract Tests (Phase 2)
After T005 scaffold created, launch all contract tests in parallel:
```
Task: "Contract test POST /api/documents/persist in backend/tests/contract/test_documents_persist_post.py"
Task: "Contract test GET /api/documents in backend/tests/contract/test_documents_list_get.py"
Task: "Contract test GET /api/documents/{id} in backend/tests/contract/test_documents_detail_get.py"
Task: "Contract test PUT /api/documents/{id}/tier in backend/tests/contract/test_documents_tier_put.py"
```

### Example 3: API Endpoints (Phase 4)
After repository methods complete (T023-T033), launch all endpoints in parallel:
```
Task: "Create POST /api/documents/persist endpoint in backend/src/api/routes/documents.py"
Task: "Create GET /api/documents endpoint in backend/src/api/routes/documents.py"
Task: "Create GET /api/documents/{id} endpoint in backend/src/api/routes/documents.py"
Task: "Create PUT /api/documents/{id}/tier endpoint in backend/src/api/routes/documents.py"
```
Note: These modify same file but different functions, so FastAPI router allows parallel implementation.

## Notes

### Constitutional Compliance
- **EVERY** task interacting with Neo4j MUST use Graph Channel
- **NO** direct `from neo4j import` or `import neo4j` statements
- **ALL** operations include audit trail (caller_service, caller_function)
- T007 compliance test MUST pass before final delivery

### Context Engineering Integration
- T001-T003 are **MANDATORY** and **MUST SUCCEED** before core implementation
- Basin evolution tracking is a core feature, not optional
- Neural field resonance scores enhance concept relationships
- Redis persistence enables basin history API (Spec 055 dependency)

### Test-Driven Development
- Tests (T009-T018) MUST be written first
- Tests MUST fail before implementation starts
- Verify failures before proceeding to T023-T044
- Tests passing = implementation complete

### Performance Targets
- Persistence: <2s per document
- Listing: <500ms for 100 documents
- Detail: <200ms per document
- Load test (T055) validates these under 10k document load

### Commit Strategy
- Commit after each task completion
- Use conventional commit messages:
  - `feat(repo): implement persist_document core logic (T024-T025)`
  - `test(contract): add POST /api/documents/persist test (T009)`
  - `perf(query): optimize pagination with index hints (T054)`
- Link to task number in commit message

## Validation Checklist

Before marking Spec 054 complete, verify:
- [x] All 57 tasks completed
- [ ] All contract tests (T009-T012) passing
- [ ] All integration tests (T013-T018) passing
- [ ] All unit tests (T049-T052) passing
- [ ] Constitutional compliance test (T007) passing (no neo4j imports)
- [ ] Context Engineering validation (T001-T003) passing
- [ ] Performance targets met (T018, T055)
- [ ] All endpoints functional (T041-T044)
- [ ] Tier migration working (T034-T040)
- [ ] Documentation complete (T056)
- [ ] Load test passed (T055)
- [ ] No direct database driver usage (only Graph Channel)
- [ ] Audit trail on every operation

## Next Steps After Tasks Complete

1. **Run /analyze** to verify spec-plan-tasks consistency
2. **Begin T001** (Context Engineering validation)
3. **Sequential execution** through critical path
4. **Parallel execution** where marked [P]
5. **Commit after each task**
6. **Update REAL_DATA_FRONTEND_STATUS.md** when Spec 054 complete
7. **Proceed to Spec 055** (Knowledge Graph APIs) clarification and planning
