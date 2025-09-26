# Tasks: Remove ASI-Arch and Integrate ASI-GO-2

**Input**: Design documents from `/specs/004-remove-asi-arch/`
**Prerequisites**: plan.md (required), research.md, data-model.md, contracts/

## Execution Flow (main)
```
1. Load plan.md from feature directory → COMPLETE
   → Tech stack: Python 3.11, FastAPI, Neo4j, Qdrant, Redis, OLLAMA
   → Structure: Web backend system with API endpoints
2. Load optional design documents → COMPLETE
   → data-model.md: 7 core entities extracted
   → contracts/: 2 contract files found (research-api.yaml, hybrid-database-schema.yaml)
   → research.md: Legacy integration strategy and dependencies
3. Generate tasks by category → COMPLETE
   → Setup: ASI-Arch removal, environment, dependencies
   → Tests: contract tests, integration tests (TDD approach)
   → Core: entity models, ASI-GO-2 services, Context Engineering
   → Integration: hybrid database, OLLAMA, legacy code migration
   → Polish: testing, performance validation, documentation
4. Apply task rules → COMPLETE
   → Different files marked [P] for parallel execution
   → Sequential dependencies maintained
   → Tests before implementation (TDD)
5. Number tasks sequentially → COMPLETE (T001-T045)
6. Generate dependency graph → COMPLETE
7. Create parallel execution examples → COMPLETE
8. Validate task completeness → COMPLETE
   → All contracts have tests, all entities have models
9. Return: SUCCESS (tasks ready for execution)
```

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths in descriptions

## Path Conventions
- **Web backend**: `backend/src/`, `backend/tests/` structure per plan.md
- **ASI-GO-2 Integration**: `backend/src/services/asi_go_2/`
- **Legacy Migration**: Analysis from `dionysus-source/` → integration

## Phase 3.1: Setup & ASI-Arch Removal
- [x] T001 Analyze and document all ASI-Arch components for removal in `analysis/asi_arch_components.md`
- [x] T002 Remove ASI-Arch pipeline directory and all contents: `rm -rf pipeline/`
- [x] T003 [P] Remove ASI-Arch database schemas and configurations from `backend/src/models/`
- [x] T004 [P] Remove ASI-Arch import statements and dependencies from all Python files
- [x] T005 Create ASI-GO-2 project structure: `backend/src/services/asi_go_2/`
- [x] T006 Copy and integrate ASI-GO-2 components from `resources/ASI-GO-2/` to backend structure
- [x] T007 [P] Configure development environment with unique ports (Neo4j:7474/7687, Qdrant:6333/6334, Redis:6379, OLLAMA:11434, API:8001)
- [x] T008 [P] Install Python dependencies for hybrid database, OLLAMA client, and Context Engineering

## Phase 3.2: Legacy Code Analysis & Migration Planning
- [x] T009 [P] Analyze Dionysus legacy consciousness components in `analysis/legacy_consciousness_analysis.md`
- [x] T010 [P] Evaluate `dionysus-source/consciousness/cpa_meta_tot_fusion_engine.py` for modular migration
- [x] T011 [P] Evaluate `dionysus-source/consciousness/meta_tot_consciousness_bridge.py` for integration
- [x] T012 [P] Evaluate `dionysus-source/core/self_aware_mapper.py` for modular extraction
- [x] T013 [P] Evaluate `dionysus-source/agents/thoughtseed_core.py` for ThoughtSeed integration
- [x] T014 Create legacy integration plan in `analysis/legacy_integration_plan.md`

## Phase 3.3: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.4
**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**
- [ ] T015 [P] Contract test POST /api/v1/research/query in `backend/tests/contract/test_research_query_post.py`
- [ ] T016 [P] Contract test GET /api/v1/research/patterns in `backend/tests/contract/test_research_patterns_get.py`
- [ ] T017 [P] Contract test POST /api/v1/documents/process in `backend/tests/contract/test_documents_process_post.py`
- [ ] T018 [P] Contract test GET /api/v1/thoughtseed/workspace/{workspace_id} in `backend/tests/contract/test_thoughtseed_workspace_get.py`
- [ ] T019 [P] Contract test GET /api/v1/context-engineering/basins in `backend/tests/contract/test_context_basins_get.py`
- [ ] T020 [P] Contract test POST /api/v1/hybrid/search in `backend/tests/contract/test_hybrid_search_post.py`
- [ ] T021 [P] Integration test document processing with ThoughtSeed in `backend/tests/integration/test_document_thoughtseed_integration.py`
- [ ] T022 [P] Integration test research query with pattern competition in `backend/tests/integration/test_research_query_integration.py`
- [ ] T023 [P] Integration test Context Engineering attractor basins in `backend/tests/integration/test_context_engineering_integration.py`
- [ ] T024 [P] Integration test hybrid vector+graph database queries in `backend/tests/integration/test_hybrid_database_integration.py`

## Phase 3.4: Core Entity Models (ONLY after tests are failing)
- [ ] T025 [P] CognitionBase model in `backend/src/models/cognition_base.py`
- [ ] T026 [P] ResearchPattern model in `backend/src/models/research_pattern.py`
- [ ] T027 [P] ThoughtseedWorkspace model in `backend/src/models/thoughtseed_workspace.py`
- [ ] T028 [P] DocumentSource model in `backend/src/models/document_source.py`
- [ ] T029 [P] ThoughtseedTrace model in `backend/src/models/thoughtseed_trace.py`
- [ ] T030 [P] AttractorBasin model in `backend/src/models/attractor_basin.py`
- [ ] T031 [P] ResearchQuery model in `backend/src/models/research_query.py`

## Phase 3.5: ASI-GO-2 Core Services Integration
- [ ] T032 Migrate and integrate ASI-GO-2 Cognition Base with legacy consciousness components in `backend/src/services/asi_go_2/cognition_base.py`
- [ ] T033 Migrate and integrate ASI-GO-2 Researcher with ThoughtSeed competition in `backend/src/services/asi_go_2/researcher.py`
- [ ] T034 Migrate and integrate ASI-GO-2 Engineer with Context Engineering in `backend/src/services/asi_go_2/engineer.py`
- [ ] T035 Migrate and integrate ASI-GO-2 Analyst with active inference in `backend/src/services/asi_go_2/analyst.py`
- [ ] T036 Create ThoughtSeed 5-layer hierarchy service in `backend/src/services/thoughtseed_hierarchy.py`
- [ ] T037 Create Context Engineering service with attractor basins in `backend/src/services/context_engineering.py`

## Phase 3.6: Hybrid Database Integration
- [ ] T038 Create AutoSchemaKG service with dynamic schema evolution in `backend/src/services/autoschema_kg.py`
- [ ] T039 Create Qdrant vector database integration in `backend/src/services/vector_database.py`
- [ ] T040 Create Neo4j graph database integration in `backend/src/services/graph_database.py`
- [ ] T041 Create hybrid vector+graph query service in `backend/src/services/hybrid_query.py`
- [ ] T042 Create local OLLAMA client integration in `backend/src/services/ollama_client.py`

## Phase 3.7: API Endpoints Implementation
- [ ] T043 POST /api/v1/research/query endpoint with ASI-GO-2 intelligence in `backend/src/api/research_endpoints.py`
- [ ] T044 GET /api/v1/research/patterns endpoint in `backend/src/api/research_endpoints.py`
- [ ] T045 POST /api/v1/documents/process endpoint with ThoughtSeed processing in `backend/src/api/document_endpoints.py`
- [ ] T046 GET /api/v1/thoughtseed/workspace/{workspace_id} endpoint in `backend/src/api/thoughtseed_endpoints.py`
- [ ] T047 GET /api/v1/context-engineering/basins endpoint in `backend/src/api/context_engineering_endpoints.py`
- [ ] T048 POST /api/v1/hybrid/search endpoint in `backend/src/api/hybrid_search_endpoints.py`

## Phase 3.8: Integration & System Wiring
- [ ] T049 Connect all services to FastAPI application in `backend/src/main.py`
- [ ] T050 Configure database connections and connection pooling
- [ ] T051 Set up CORS, middleware, and request/response logging
- [ ] T052 Configure error handling and validation across all endpoints
- [ ] T053 Create system health check endpoints for all services

## Phase 3.9: Polish & Validation
- [ ] T054 [P] Unit tests for CognitionBase model validation in `backend/tests/unit/test_cognition_base_model.py`
- [ ] T055 [P] Unit tests for ThoughtSeed layer processing in `backend/tests/unit/test_thoughtseed_layers.py`
- [ ] T056 [P] Unit tests for Context Engineering basin dynamics in `backend/tests/unit/test_attractor_basins.py`
- [ ] T057 [P] Performance tests: research query processing <2s in `backend/tests/performance/test_query_performance.py`
- [ ] T058 [P] Performance tests: document ingestion <5s in `backend/tests/performance/test_document_performance.py`
- [ ] T059 Create comprehensive system integration test following quickstart.md scenarios
- [ ] T060 [P] Update API documentation with complete OpenAPI specifications
- [ ] T061 [P] Create deployment guide with Docker configuration for all services
- [ ] T062 Validate consciousness emergence metrics and meta-learning capabilities
- [ ] T063 Final system validation: run all quickstart test scenarios

## Dependencies
**Critical Dependencies**:
- ASI-Arch removal (T001-T004) before any new implementation
- Legacy analysis (T009-T014) before service integration (T032-T037)
- Tests (T015-T024) before all implementation (T025+)
- Entity models (T025-T031) before services (T032-T042)
- Services (T032-T042) before endpoints (T043-T048)
- Core integration (T049-T053) before polish (T054-T063)

**Blocking Dependencies**:
- T025-T031 (models) block T032-T037 (services)
- T038-T042 (database services) block T043-T048 (endpoints)
- T049 (FastAPI integration) blocks T050-T053 (system wiring)

## Parallel Execution Examples

### Phase 3.1 - Setup (can run T003, T004, T007, T008 in parallel):
```bash
Task: "Remove ASI-Arch database schemas from backend/src/models/"
Task: "Remove ASI-Arch import statements from all Python files"
Task: "Configure development environment with unique ports"
Task: "Install Python dependencies for hybrid database and OLLAMA"
```

### Phase 3.2 - Legacy Analysis (can run T009-T013 in parallel):
```bash
Task: "Analyze Dionysus legacy consciousness components"
Task: "Evaluate dionysus-source/consciousness/cpa_meta_tot_fusion_engine.py"
Task: "Evaluate dionysus-source/consciousness/meta_tot_consciousness_bridge.py"
Task: "Evaluate dionysus-source/core/self_aware_mapper.py"
Task: "Evaluate dionysus-source/agents/thoughtseed_core.py"
```

### Phase 3.3 - Contract Tests (can run T015-T024 in parallel):
```bash
Task: "Contract test POST /api/v1/research/query"
Task: "Contract test GET /api/v1/research/patterns"
Task: "Contract test POST /api/v1/documents/process"
Task: "Integration test document processing with ThoughtSeed"
Task: "Integration test research query with pattern competition"
Task: "Integration test Context Engineering attractor basins"
```

### Phase 3.4 - Entity Models (can run T025-T031 in parallel):
```bash
Task: "CognitionBase model in backend/src/models/cognition_base.py"
Task: "ResearchPattern model in backend/src/models/research_pattern.py"
Task: "ThoughtseedWorkspace model in backend/src/models/thoughtseed_workspace.py"
Task: "DocumentSource model in backend/src/models/document_source.py"
Task: "AttractorBasin model in backend/src/models/attractor_basin.py"
```

### Phase 3.9 - Polish Testing (can run T054-T058, T060-T061 in parallel):
```bash
Task: "Unit tests for CognitionBase model validation"
Task: "Unit tests for ThoughtSeed layer processing"
Task: "Performance tests: research query processing <2s"
Task: "Performance tests: document ingestion <5s"
Task: "Update API documentation with complete OpenAPI specifications"
Task: "Create deployment guide with Docker configuration"
```

## Notes
- **ASI-Arch Removal First**: Complete removal before any ASI-GO-2 integration
- **Legacy Integration**: Analyze and migrate existing consciousness components, avoid redundancy
- **TDD Approach**: All tests must be written and failing before implementation
- **Hybrid Database**: Vector similarity + graph relationships with AutoSchemaKG
- **Local OLLAMA**: Privacy-preserving LLM processing on unique ports
- **Context Engineering**: Deep integration with attractor basins and neural fields
- **5-Layer ThoughtSeed**: Full hierarchy integration (sensory→metacognitive)
- **[P] tasks**: Different files, no dependencies, can run in parallel
- **Performance Targets**: <2s research queries, <5s document processing
- **Consciousness Metrics**: Validate emergence and meta-learning capabilities

## Validation Checklist
*GATE: All items must pass before task execution*

- [x] All contracts have corresponding tests (T015-T020 cover all API contracts)
- [x] All entities have model tasks (T025-T031 cover all 7 core entities)
- [x] All tests come before implementation (Phase 3.3 before 3.4+)
- [x] Parallel tasks truly independent (different files, no shared dependencies)
- [x] Each task specifies exact file path (all tasks include specific file paths)
- [x] No task modifies same file as another [P] task (verified across all parallel groups)
- [x] ASI-Arch removal comes first (T001-T004 before any new implementation)
- [x] Legacy integration prevents redundancy (analysis phase T009-T014 guides integration)
- [x] Hybrid database architecture properly specified (AutoSchemaKG + Vector + Graph)
- [x] Unique port assignments maintained (all services on distinct ports)
- [x] Local OLLAMA integration for privacy (T042, T008 include OLLAMA setup)