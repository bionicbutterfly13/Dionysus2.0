# Tasks: Research Engine Query Interface

**Input**: Design documents from `/specs/006-ability-to-ask/`
**Prerequisites**: plan.md (✅ complete), spec.md (✅ complete)

## Phase 3.1: Setup
- [ ] T001 Verify Neo4j and Qdrant connections in backend
- [ ] T002 Create query module structure (backend/src/services/query_engine.py)
- [ ] T003 [P] Setup pytest fixtures for query testing

## Phase 3.2: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3
**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**
- [ ] T004 [P] Contract test: POST /api/query endpoint (tests/contract/test_query_api_post.py)
- [ ] T005 [P] Contract test: Response schema validation (tests/contract/test_query_response_schema.py)
- [ ] T006 [P] Integration test: Neo4j search (tests/integration/test_neo4j_search.py)
- [ ] T007 [P] Integration test: Vector search (tests/integration/test_vector_search.py)
- [ ] T008 [P] Integration test: Response synthesis (tests/integration/test_response_synthesis.py)
- [ ] T009 [P] Integration test: End-to-end query flow (tests/integration/test_end_to_end_query.py)

## Phase 3.3: Core Implementation (ONLY after tests are failing)
- [ ] T010 [P] Implement Query model (backend/src/models/query.py)
- [ ] T011 [P] Implement Response model (backend/src/models/response.py)
- [ ] T012 [P] Implement Neo4j searcher (backend/src/services/neo4j_searcher.py)
- [ ] T013 [P] Implement Vector searcher (backend/src/services/vector_searcher.py)
- [ ] T014 Implement Response synthesizer (backend/src/services/response_synthesizer.py)
- [ ] T015 Implement Query engine orchestrator (backend/src/services/query_engine.py)

## Phase 3.4: Integration
- [ ] T016 Create query API endpoint (backend/src/api/routes/query.py)
- [ ] T017 Integrate ThoughtSeed tracking for queries
- [ ] T018 Add Redis caching for frequent queries
- [ ] T019 Connect to existing attractor basins
- [ ] T020 Test concurrent query handling (10 simultaneous)

## Phase 3.5: Polish
- [ ] T021 [P] Performance optimization (<2s target)
- [ ] T022 [P] Create data-model.md documentation
- [ ] T023 [P] Create API contract documentation
- [ ] T024 Add query logging and analytics
- [ ] T025 Run full test suite validation

## Dependencies
```
Setup (T001-T003) → Tests (T004-T009) → Implementation (T010-T015) → Integration (T016-T020) → Polish (T021-T025)

Parallel Tasks:
- T004-T009: All test files (independent)
- T010-T013: Models and searchers (independent)
- T021-T023: Documentation tasks (independent)
```

## Notes
- Query creates ThoughtSeed that flows through consciousness processing
- Parallel search across Neo4j + Qdrant for speed
- Meta-ToT active inference for response synthesis
- Redis caching for performance
