# Tasks: Complete ThoughtSeed Pipeline Implementation

**Input**: Design documents from `/specs/010-the-entire-pipeline/`
**Prerequisites**: plan.md (required), research.md, data-model.md, contracts/

## Execution Flow (main)
```
1. Load plan.md from feature directory
   → Tech stack: Python 3.11+ (backend), TypeScript/React (frontend)
   → Libraries: FastAPI, Redis, Neo4j, NumPy 2.0, react-dropzone, three.js
   → Structure: Web application (frontend + backend)
2. Load design documents:
   → data-model.md: 10 core entities extracted
   → contracts/api-spec.yaml: 8 endpoints identified
   → research.md: Technology decisions documented
3. Generate tasks by category:
   → Setup: project init, dependencies, services
   → Tests: 18 contract/integration tests
   → Core: 35 implementation tasks
   → Integration: 8 system integration tasks
   → Polish: 6 validation and performance tasks
4. Apply task rules:
   → Different files = mark [P] for parallel
   → Tests before implementation (TDD)
   → Models before services before endpoints
5. Number tasks sequentially (T001-T067)
6. Generate dependency graph and execution groups
7. Validate: All contracts tested, all entities modeled
8. Return: SUCCESS (67 tasks ready for execution)
```

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths in descriptions

## Path Conventions
- **Backend**: `backend/src/`, `backend/tests/`
- **Frontend**: `frontend/src/`, `frontend/tests/`

## Phase 3.1: Setup & Infrastructure

- [x] T001 Create project structure for web application (backend/src/, frontend/src/, backend/tests/, frontend/tests/)
- [x] T002 Initialize Python backend with FastAPI, Redis, Neo4j, NumPy 2.0 dependencies in backend/requirements.txt
- [x] T003 Initialize React frontend with TypeScript, react-dropzone, three.js dependencies in frontend/package.json
- [x] T004 [P] Configure backend linting (black, flake8, mypy) in backend/.pre-commit-config.yaml
- [x] T005 [P] Configure frontend linting (ESLint, Prettier) in frontend/.eslintrc.js
- [x] T006 Set up Redis server configuration with TTL settings in backend/src/config/redis_config.py
- [x] T007 Set up Neo4j database configuration and schema in backend/src/config/neo4j_config.py
- [x] T008 Set up Docker Compose for development services in docker-compose.dev.yml
- [x] T009 Create environment configuration management in backend/src/config/settings.py
- [x] T010 [P] Set up Playwright test configuration in frontend/playwright.config.ts

## Phase 3.2: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3
**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**

### Contract Tests (Backend)
- [x] T011 [P] Contract test POST /api/v1/documents/bulk in backend/tests/contract/test_documents_bulk_post.py
- [x] T012 [P] Contract test GET /api/v1/documents/batch/{batch_id}/status in backend/tests/contract/test_batch_status_get.py
- [x] T013 [P] Contract test GET /api/v1/documents/batch/{batch_id}/results in backend/tests/contract/test_batch_results_get.py
- [x] T014 [P] Contract test GET /api/v1/thoughtseeds/{thoughtseed_id} in backend/tests/contract/test_thoughtseeds_get.py
- [x] T015 [P] Contract test GET /api/v1/attractors in backend/tests/contract/test_attractors_get.py
- [x] T016 [P] Contract test GET /api/v1/neural-fields/{field_id}/state in backend/tests/contract/test_neural_fields_get.py
- [x] T017 [P] Contract test POST /api/v1/knowledge/search in backend/tests/contract/test_knowledge_search_post.py
- [x] T018 [P] Contract test WebSocket /ws/batch/{batch_id}/progress in backend/tests/contract/test_websocket_progress.py

### Integration Tests
- [x] T019 [P] Integration test complete document upload flow in backend/tests/integration/test_document_processing_flow.py
- [x] T020 [P] Integration test ThoughtSeed 5-layer processing in backend/tests/integration/test_thoughtseed_layers.py
- [x] T021 [P] Integration test attractor basin modification in backend/tests/integration/test_attractor_dynamics.py
- [x] T022 [P] Integration test neural field evolution in backend/tests/integration/test_neural_field_dynamics.py
- [x] T023 [P] Integration test consciousness detection in backend/tests/integration/test_consciousness_detection.py
- [x] T024 [P] Integration test WebSocket real-time updates in backend/tests/integration/test_websocket_realtime.py
- [x] T025 [P] Integration test capacity management and queuing in backend/tests/integration/test_capacity_management.py
- [x] T026 [P] Integration test knowledge graph search in backend/tests/integration/test_knowledge_search.py

### Frontend Tests
- [x] T027 [P] Frontend integration test document upload UI in frontend/tests/integration/test_document_upload.spec.ts
- [x] T028 [P] Frontend integration test 3D neural field visualization in frontend/tests/integration/test_neural_field_viz.spec.ts

## Phase 3.3: Core Implementation (ONLY after tests are failing)

### Data Models
- [ ] T029 [P] Document model in backend/src/models/document.py
- [ ] T030 [P] ProcessingBatch model in backend/src/models/processing_batch.py
- [ ] T031 [P] ThoughtSeed model in backend/src/models/thoughtseed.py
- [ ] T032 [P] NeuronalPacket model in backend/src/models/neuronal_packet.py
- [ ] T033 [P] AttractorBasin model in backend/src/models/attractor_basin.py
- [ ] T034 [P] NeuralField model in backend/src/models/neural_field.py
- [ ] T035 [P] ConsciousnessState model in backend/src/models/consciousness_state.py
- [ ] T036 [P] MemoryFormation model in backend/src/models/memory_formation.py
- [ ] T037 [P] KnowledgeTriple model in backend/src/models/knowledge_triple.py
- [ ] T038 [P] EvolutionaryPrior model in backend/src/models/evolutionary_prior.py

### Core Services
- [ ] T039 [P] DocumentProcessingService in backend/src/services/document_processing_service.py
- [ ] T040 [P] ThoughtSeedProcessingService (5-layer implementation) in backend/src/services/thoughtseed_processing_service.py
- [ ] T041 [P] AttractorBasinManager with mathematical foundation in backend/src/services/attractor_basin_manager.py
- [ ] T042 [P] NeuralFieldDynamics with PDE solver in backend/src/services/neural_field_dynamics.py
- [ ] T043 [P] ConsciousnessDetectionService in backend/src/services/consciousness_detection_service.py
- [ ] T044 [P] MemoryIntegrationService (multi-timescale) in backend/src/services/memory_integration_service.py
- [ ] T045 [P] AutoSchemaKGService for knowledge extraction in backend/src/services/autoschema_kg_service.py
- [ ] T046 [P] CapacityManagementService with queuing in backend/src/services/capacity_management_service.py

### Database Layer
- [ ] T047 [P] Redis cache manager with TTL enforcement in backend/src/db/redis_manager.py
- [ ] T048 [P] Neo4j graph database manager in backend/src/db/neo4j_manager.py
- [ ] T049 [P] Vector database integration in backend/src/db/vector_db_manager.py
- [ ] T050 [P] Unified query interface in backend/src/db/unified_query_manager.py

### API Endpoints
- [ ] T051 POST /api/v1/documents/bulk endpoint in backend/src/api/routes/documents.py
- [ ] T052 GET /api/v1/documents/batch/{batch_id}/status endpoint in backend/src/api/routes/documents.py
- [ ] T053 GET /api/v1/documents/batch/{batch_id}/results endpoint in backend/src/api/routes/documents.py
- [ ] T054 GET /api/v1/thoughtseeds/{thoughtseed_id} endpoint in backend/src/api/routes/thoughtseeds.py
- [ ] T055 GET /api/v1/attractors endpoint in backend/src/api/routes/attractors.py
- [ ] T056 GET /api/v1/neural-fields/{field_id}/state endpoint in backend/src/api/routes/neural_fields.py
- [ ] T057 POST /api/v1/knowledge/search endpoint in backend/src/api/routes/knowledge.py
- [ ] T058 WebSocket /ws/batch/{batch_id}/progress endpoint in backend/src/api/websocket/progress.py

### Frontend Implementation
- [ ] T059 Replace mock DocumentUpload component with real API calls in frontend/src/pages/DocumentUpload.tsx
- [ ] T060 [P] Create DocumentService API client in frontend/src/services/documentService.ts
- [ ] T061 [P] Create WebSocketService for real-time updates in frontend/src/services/websocketService.ts
- [ ] T062 [P] Create 3D neural field visualization component in frontend/src/components/NeuralFieldVisualization.tsx
- [ ] T063 [P] Create ThoughtSeed progress display component in frontend/src/components/ThoughtSeedProgress.tsx
- [ ] T064 [P] Create attractor basin visualization component in frontend/src/components/AttractorBasinDisplay.tsx

## Phase 3.4: Integration & Orchestration

- [ ] T065 Connect ThoughtSeedProcessingService to AttractorBasinManager in backend/src/services/pipeline_orchestrator.py
- [ ] T066 Integrate ConsciousnessDetectionService with NeuralFieldDynamics in backend/src/services/consciousness_integration.py
- [ ] T067 Connect all services to unified database layer in backend/src/services/database_integration.py
- [ ] T068 Implement WebSocket broadcast system for real-time updates in backend/src/websocket/broadcast_manager.py
- [ ] T069 Add constitutional gateway validation to document processing in backend/src/middleware/constitutional_gateway.py
- [ ] T070 Implement MIT/IBM/Shanghai research integration points in backend/src/services/research_integration_service.py
- [ ] T071 Configure FastAPI middleware for CORS, logging, error handling in backend/src/main.py
- [ ] T072 Frontend WebSocket connection and state management in frontend/src/hooks/useWebSocket.ts

## Phase 3.5: Polish & Validation

- [ ] T073 [P] Unit tests for mathematical foundations (attractor basin φ function) in backend/tests/unit/test_attractor_math.py
- [ ] T074 [P] Unit tests for neural field PDE solver in backend/tests/unit/test_neural_field_pde.py
- [ ] T075 [P] Unit tests for consciousness detection thresholds in backend/tests/unit/test_consciousness_thresholds.py
- [ ] T076 [P] Performance tests for batch processing (1000 files, 500MB each) in backend/tests/performance/test_batch_performance.py
- [ ] T077 [P] Frontend unit tests for 3D visualization performance in frontend/tests/unit/test_visualization_performance.spec.ts
- [ ] T078 Execute quickstart.md validation scenarios in backend/tests/validation/test_quickstart_scenarios.py
- [ ] T079 [P] Update CLAUDE.md with implementation details
- [ ] T080 Clean up any remaining mock/simulation code
- [ ] T081 Verify Redis TTL enforcement (24h packets, 7d basins, 30d results)

## Dependencies

### Critical Path
1. **Setup (T001-T010)** → All subsequent tasks
2. **Tests (T011-T028)** → Implementation tasks (T029+)
3. **Models (T029-T038)** → Services (T039-T050)
4. **Services (T039-T050)** → API Endpoints (T051-T058)
5. **Backend Core (T029-T058)** → Integration (T065-T072)
6. **Integration (T065-T072)** → Polish (T073-T081)

### Blocking Dependencies
- T040 (ThoughtSeed) blocks T041 (Attractor), T042 (Neural Field)
- T051-T053 (Documents API) share same file - cannot be parallel
- T059 (Frontend mock removal) blocks T060-T064 (Frontend services)
- T065-T067 (Integration) require all core services complete

## Parallel Execution Examples

### Phase 3.2 - All Tests in Parallel
```bash
# Launch T011-T018 (Contract Tests) together:
Task: "Contract test POST /api/v1/documents/bulk in backend/tests/contract/test_documents_bulk_post.py"
Task: "Contract test GET /api/v1/documents/batch/{batch_id}/status in backend/tests/contract/test_batch_status_get.py"
Task: "Contract test GET /api/v1/documents/batch/{batch_id}/results in backend/tests/contract/test_batch_results_get.py"
Task: "Contract test GET /api/v1/thoughtseeds/{thoughtseed_id} in backend/tests/contract/test_thoughtseeds_get.py"
Task: "Contract test GET /api/v1/attractors in backend/tests/contract/test_attractors_get.py"
Task: "Contract test GET /api/v1/neural-fields/{field_id}/state in backend/tests/contract/test_neural_fields_get.py"
Task: "Contract test POST /api/v1/knowledge/search in backend/tests/contract/test_knowledge_search_post.py"
Task: "Contract test WebSocket /ws/batch/{batch_id}/progress in backend/tests/contract/test_websocket_progress.py"

# Launch T019-T028 (Integration Tests) together:
Task: "Integration test complete document upload flow in backend/tests/integration/test_document_processing_flow.py"
Task: "Integration test ThoughtSeed 5-layer processing in backend/tests/integration/test_thoughtseed_layers.py"
Task: "Integration test attractor basin modification in backend/tests/integration/test_attractor_dynamics.py"
Task: "Integration test neural field evolution in backend/tests/integration/test_neural_field_dynamics.py"
```

### Phase 3.3 - Models in Parallel
```bash
# Launch T029-T038 (Data Models) together:
Task: "Document model in backend/src/models/document.py"
Task: "ProcessingBatch model in backend/src/models/processing_batch.py"
Task: "ThoughtSeed model in backend/src/models/thoughtseed.py"
Task: "NeuronalPacket model in backend/src/models/neuronal_packet.py"
Task: "AttractorBasin model in backend/src/models/attractor_basin.py"
Task: "NeuralField model in backend/src/models/neural_field.py"
Task: "ConsciousnessState model in backend/src/models/consciousness_state.py"
Task: "MemoryFormation model in backend/src/models/memory_formation.py"
Task: "KnowledgeTriple model in backend/src/models/knowledge_triple.py"
Task: "EvolutionaryPrior model in backend/src/models/evolutionary_prior.py"
```

### Phase 3.3 - Core Services in Parallel
```bash
# Launch T039-T050 (Core Services) together:
Task: "DocumentProcessingService in backend/src/services/document_processing_service.py"
Task: "ThoughtSeedProcessingService (5-layer implementation) in backend/src/services/thoughtseed_processing_service.py"
Task: "AttractorBasinManager with mathematical foundation in backend/src/services/attractor_basin_manager.py"
Task: "NeuralFieldDynamics with PDE solver in backend/src/services/neural_field_dynamics.py"
Task: "ConsciousnessDetectionService in backend/src/services/consciousness_detection_service.py"
Task: "MemoryIntegrationService (multi-timescale) in backend/src/services/memory_integration_service.py"
```

## Notes
- [P] tasks = different files, no dependencies - can run simultaneously
- **TDD CRITICAL**: All test tasks (T011-T028) MUST be completed and failing before any implementation
- Verify tests fail before implementing corresponding functionality
- Each task should result in a focused, reviewable commit
- Mock/simulation code replacement is the primary goal
- Consciousness processing must integrate MIT/IBM/Shanghai research findings

## Task Generation Rules Applied

1. **From Contracts**: 8 API endpoints → 8 contract tests (T011-T018)
2. **From Data Model**: 10 entities → 10 model tasks (T029-T038)
3. **From User Stories**: Core scenarios → 8 integration tests (T019-T026)
4. **From Research**: Technology decisions → setup tasks (T001-T010)
5. **From Quickstart**: Validation scenarios → polish tasks (T073-T081)

## Validation Checklist

- [x] All contracts have corresponding tests (T011-T018)
- [x] All entities have model tasks (T029-T038)
- [x] All tests come before implementation (Phase 3.2 → 3.3)
- [x] Parallel tasks truly independent (different files, [P] marked)
- [x] Each task specifies exact file path
- [x] No task modifies same file as another [P] task
- [x] TDD enforced: tests must fail before implementation
- [x] Complete replacement of mock/simulation code addressed (T059, T080)
- [x] ThoughtSeed 5-layer processing covered (T020, T040)
- [x] Mathematical foundations included (T041, T042, T073, T074)
- [x] Real-time features and 3D visualization addressed (T058, T062, T072)
- [x] All clarified requirements covered (TTL, file limits, capacity management)