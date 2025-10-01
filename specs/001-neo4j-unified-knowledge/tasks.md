# Tasks: Neo4j Unified Knowledge Graph Schema Implementation

**Input**: Design documents from `/specs/001-neo4j-unified-knowledge/`
**Prerequisites**: plan.md (✅ complete), spec.md (✅ complete)

## Phase 3.1: Setup
- [ ] T001 Create Neo4j test container configuration in docker-compose-neo4j.yml
- [ ] T002 Initialize Python project with dependencies (neo4j-driver, sentence-transformers, networkx)
- [ ] T003 [P] Configure pytest with Neo4j test fixtures

## Phase 3.2: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3
**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**
- [ ] T004 [P] Contract test: Architecture node creation (tests/contract/test_architecture_create.py)
- [ ] T005 [P] Contract test: Episode node creation (tests/contract/test_episode_create.py)
- [ ] T006 [P] Contract test: Relationship creation (tests/contract/test_relationships.py)
- [ ] T007 [P] Contract test: Vector similarity search (tests/contract/test_vector_search.py)
- [ ] T008 [P] Integration test: Complete migration workflow (tests/integration/test_migration.py)
- [ ] T009 [P] Integration test: Data integrity validation (tests/integration/test_data_integrity.py)

## Phase 3.3: Core Implementation (ONLY after tests are failing)
- [ ] T010 [P] Implement Neo4j schema initialization (extensions/context_engineering/neo4j_unified_schema.py)
- [ ] T011 [P] Implement Architecture node model with vector properties
- [ ] T012 [P] Implement Episode, ConsciousnessState, Archetype node models
- [ ] T013 [P] Implement relationship creation logic (EVOLVED_FROM, EXHIBITS, HAS_STATE)
- [ ] T014 Implement vector indexing integration
- [ ] T015 Implement unified query API (extensions/context_engineering/unified_api.py)

## Phase 3.4: Migration Implementation
- [ ] T016 Implement migration manager (extensions/context_engineering/migration_manager.py)
- [ ] T017 Extract data from MongoDB (legacy architectures)
- [ ] T018 Extract data from FAISS (vector embeddings)
- [ ] T019 Extract data from SQLite+JSON hybrid
- [ ] T020 Transform extracted data to Neo4j format
- [ ] T021 Load data into Neo4j with relationships
- [ ] T022 Validate data integrity (100% preservation check)

## Phase 3.5: Polish & Validation
- [ ] T023 [P] Create migration rollback mechanism
- [ ] T024 [P] Implement legacy compatibility layer
- [ ] T025 Performance testing with 10k+ architectures
- [ ] T026 Create data-model.md documentation
- [ ] T027 Create migration quickstart guide
- [ ] T028 Run full test suite validation

## Dependencies
```
Setup (T001-T003) → Tests (T004-T009) → Implementation (T010-T015) → Migration (T016-T022) → Polish (T023-T028)

Parallel Tasks:
- T004-T009: All tests can run in parallel (independent files)
- T010-T013: All model implementations (independent)
- T017-T019: All data extraction tasks (independent sources)
- T023-T024: Rollback and compatibility (independent)
```

## Notes
- Zero data loss requirement - all migration must be validated
- Lifelong accumulation system - design for 10,000+ architectures
- AutoSchemaKG integration for automatic relationship discovery
- Vector similarity critical for solution mutation feature
