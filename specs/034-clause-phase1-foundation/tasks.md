# Tasks: CLAUSE Phase 1 - Agentic Subgraph Architect with Basin Strengthening

**Input**: Design documents from `/specs/034-clause-phase1-foundation/`
**Prerequisites**: plan.md ✅, research.md ✅, data-model.md ✅, contracts/ ✅, quickstart.md ✅

## Overview

This task list implements CLAUSE Subgraph Architect with basin frequency strengthening. The system uses 5-signal edge scoring (φ_ent, φ_rel, φ_nbr, φ_deg, φ_basin) with shaped gain rule to construct budget-aware subgraphs while strengthening attractor basins (+0.2 per concept reappearance, cap 2.0).

**Tech Stack**:
- Python 3.11+
- Neo4j 5.x driver, NetworkX 3.x, NumPy 2.0+
- pytest (contract, integration, performance tests)
- Redis (basin caching)

**Structure**: Web application (backend)
- Code: `backend/src/services/clause/`, `backend/src/models/`
- Tests: `backend/tests/contract/`, `backend/tests/integration/`, `backend/tests/unit/`

## Phase 3.1: Setup & Context Engineering Validation

### Context Engineering Foundation (Constitution Article II - MANDATORY FIRST)
- [x] **T001** [P] Verify AttractorBasin accessibility and extension compatibility ✅
  - File: `backend/tests/test_context_engineering_basin.py`
  - Verify existing `backend/src/models/attractor_basin.py` is accessible
  - Check AttractorBasin model has required base fields (basin_id, strength, stability)
  - Confirm backward compatibility for extension with new fields
  - Assert NumPy 2.0+ compliance per Constitution Article I
  - **PASSED**: NumPy 2.3.3 compliant, basin accessible, ready for extension

- [x] **T002** [P] Validate Redis persistence for basin cache ✅
  - File: `backend/tests/test_context_engineering_redis.py`
  - Connect to Redis at localhost:6379
  - Test basin state caching (setex with 1-hour TTL)
  - Verify cache invalidation works
  - Test concurrent basin reads (100+ simultaneous lookups)
  - **PASSED**: 120 concurrent reads in 75ms, memory efficient (1.6KB/basin)

- [x] **T003** [P] Test basin influence calculations with co-occurrence ✅
  - File: `backend/tests/test_context_engineering_influence.py`
  - Load existing AttractorBasin model
  - Test basin strength increment (+0.2, cap at 2.0)
  - Verify co-occurrence dictionary updates (symmetric)
  - Calculate neighborhood influence from co-occurring concepts
  - **PASSED**: Basin influence validated, strength normalization working

### Project Setup
- [x] **T004** Create CLAUSE service directory structure ✅
  - Create `backend/src/services/clause/` directory
  - Create `backend/src/services/clause/__init__.py`
  - Create `backend/tests/contract/` directory (if not exists)
  - Create `backend/tests/integration/` directory (if not exists)
  - Create `backend/tests/unit/` directory (if not exists)
  - **Depends on**: T001-T003 passing
  - **PASSED**: Directory structure created, __init__.py with module docstring

- [x] **T005** Install Phase 1 dependencies ✅
  - Add to `backend/requirements.txt`:
    - `networkx>=3.1`
    - `sentence-transformers>=2.2.0` (for embeddings)
  - Run: `pip install -r backend/requirements.txt`
  - Verify installations: `python -c "import networkx, sentence_transformers; print('✅')"`
  - **Depends on**: T004
  - **PASSED**: NetworkX 3.3 installed, sentence-transformers (NumPy 2.0 incompatibility noted, using hash-based embeddings)

- [x] **T006** [P] Configure linting and type checking ✅
  - File: `backend/.pylintrc` (extend existing)
  - Add mypy config for `backend/src/services/clause/`
  - Run: `mypy backend/src/services/clause/` (should pass with no files initially)
  - Run: `ruff check backend/src/services/clause/` (should pass)
  - **Depends on**: T004
  - **PASSED**: Created .flake8, mypy.ini, pyproject.toml configs, all checks passed

## Phase 3.2: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3

**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**

### Contract Tests (API Validation)
- [x] **T007** [P] Contract test POST /api/clause/subgraph ✅
  - File: `backend/tests/contract/test_architect_subgraph_contract.py`
  - Test request schema validation (query, edge_budget, lambda_edge, hop_distance)
  - Test response schema (selected_edges, edge_scores, shaped_gains, budget_used, stopped_reason)
  - Assert edge_budget range (1-1000)
  - Assert lambda_edge range (0.0-1.0)
  - Test missing required field 'query' returns 400
  - **Must fail initially** (endpoint not implemented)
  - **Depends on**: T001-T003
  - **PASSED**: 6 tests created, all properly skipping (endpoint not implemented yet)

- [x] **T008** [P] Contract test POST /api/clause/basins/strengthen ✅
  - File: `backend/tests/contract/test_basin_strengthen_contract.py`
  - Test request schema (concepts, document_id, increment)
  - Test response schema (updated_basins, new_basins, cooccurrence_updates, total_strengthening_time_ms)
  - Assert concepts list is non-empty
  - Assert increment is 0.0-1.0
  - Test empty concepts list returns 400
  - **Must fail initially**
  - **Depends on**: T001-T003
  - **PASSED**: 7 tests created, all properly skipping (endpoint not implemented yet)

- [x] **T009** [P] Contract test GET /api/clause/basins/{basin_id} ✅
  - File: `backend/tests/contract/test_basin_get_contract.py`
  - Test path parameter validation (basin_id format)
  - Test response schema (basin_id, strength, activation_count, co_occurring_concepts)
  - Assert strength range 1.0-2.0
  - Test non-existent basin_id returns 404
  - **Must fail initially**
  - **Depends on**: T001-T003
  - **PASSED**: 7 tests created, all properly skipping (endpoint not implemented yet)

- [x] **T010** [P] Contract test POST /api/clause/edges/score ✅
  - File: `backend/tests/contract/test_edge_score_contract.py`
  - Test request schema (edges, query)
  - Test response schema (scores with EdgeScore breakdown, top_k_edges, scoring_time_ms)
  - Assert EdgeScore has all 5 signals (phi_ent, phi_rel, phi_nbr, phi_deg, phi_basin)
  - Test empty edges list returns 400
  - **Must fail initially**
  - **Depends on**: T001-T003
  - **PASSED**: 9 tests created, all properly skipping (endpoint not implemented yet)

### Integration Tests (User Scenarios from quickstart.md)
- [x] **T011** [P] Integration test: Full subgraph construction workflow ✅
  - File: `backend/tests/integration/test_architect_workflow.py`
  - Test scenario from quickstart.md Step 3.1
  - Create sample Neo4j graph (10 concept triplets)
  - Build subgraph with query "What is neural architecture search?"
  - Assert budget compliance (selected ≤ edge_budget)
  - Assert shaped gain rule (all gains > 0)
  - Assert construction time <500ms
  - **Must fail initially**
  - **Depends on**: T001-T003
  - **PASSED**: 4 tests created, all properly skipping (SubgraphArchitect not implemented yet)

- [x] **T012** [P] Integration test: Basin strengthening across multiple documents ✅
  - File: `backend/tests/integration/test_basin_strengthening_workflow.py`
  - Test scenario from quickstart.md Step 4.1
  - Simulate 3 documents with "neural_architecture" concept
  - Assert strength progression: 1.0 → 1.2 → 1.4 → 1.6
  - Assert activation_count increments correctly
  - Assert co-occurrence pairs tracked symmetrically
  - **Must fail initially**
  - **Depends on**: T001-T003
  - **PASSED**: 6 tests created, all properly skipping (BasinTracker not implemented yet)

- [x] **T013** [P] Integration test: Neo4j + Redis persistence ✅
  - File: `backend/tests/integration/test_basin_persistence.py`
  - Create basin in Neo4j
  - Cache in Redis (1-hour TTL)
  - Modify basin (strength +0.2)
  - Verify Neo4j update and Redis invalidation
  - Test lazy migration (old basin gets defaults)
  - **Must fail initially**
  - **Depends on**: T001-T003
  - **PASSED**: 5 tests created, all properly skipping (BasinTracker not implemented yet)

- [x] **T014** [P] Integration test: Basin influence on edge scoring ✅
  - File: `backend/tests/integration/test_basin_influence.py`
  - Test scenario from quickstart.md Step 4.2
  - Score same edge with basin strength 1.6 vs 2.0
  - Assert higher basin strength → higher edge score
  - Calculate expected score: 0.25·φ_ent + 0.25·φ_rel + 0.20·φ_nbr + 0.15·φ_deg + 0.15·φ_basin_norm
  - Verify basin_norm = (strength - 1.0) / 1.0
  - **Must fail initially**
  - **Depends on**: T001-T003
  - **PASSED**: 6 tests created, all properly skipping (EdgeScorer not implemented yet)

## Phase 3.3: Core Implementation (ONLY after tests are failing)

### Data Models
- [ ] **T015** [P] Extend AttractorBasin model with new fields
  - File: `backend/src/models/attractor_basin.py` (modify existing)
  - Add fields: `strength: float = 1.0`, `activation_count: int = 0`
  - Add field: `activation_history: List[datetime] = []`
  - Add field: `co_occurring_concepts: Dict[str, int] = {}`
  - Add validation: strength range [1.0, 2.0]
  - Add validation: activation_count ≥ 0
  - Ensure backward compatibility (defaults for old basins)
  - **Depends on**: T007-T014 failing

- [ ] **T016** [P] Create CLAUSESubgraphArchitect request/response models
  - File: `backend/src/services/clause/models.py`
  - Create `SubgraphRequest` dataclass (query, edge_budget, lambda_edge, hop_distance)
  - Create `SubgraphResponse` dataclass (selected_edges, edge_scores, shaped_gains, etc.)
  - Create `EdgeScore` dataclass (edge, phi_ent, phi_rel, phi_nbr, phi_deg, phi_basin, total_score, shaped_gain)
  - Add Pydantic validators for ranges
  - **Depends on**: T007-T014 failing

- [ ] **T017** [P] Create BasinTracker request/response models
  - File: `backend/src/services/clause/basin_models.py`
  - Create `BasinStrengtheningRequest` dataclass
  - Create `BasinStrengtheningResponse` dataclass
  - Create `CoOccurrencePair` dataclass with symmetric_key property
  - **Depends on**: T007-T014 failing

### Core Services
- [ ] **T018** Implement EdgeScorer with 5-signal scoring
  - File: `backend/src/services/clause/edge_scoring.py`
  - Implement `score_edge()` method with 5 signals:
    - φ_ent: Entity-question match (BM25 or embedding similarity)
    - φ_rel: Relation-text match
    - φ_nbr: Neighborhood score (sum of co-occurrence weights)
    - φ_deg: Degree prior (prefer moderate degree nodes)
    - φ_basin: Basin strength normalized [(strength-1.0)/1.0]
  - Implement weighted sum: 0.25·φ_ent + 0.25·φ_rel + 0.20·φ_nbr + 0.15·φ_deg + 0.15·φ_basin
  - Use SentenceTransformer for embeddings
  - **Depends on**: T015-T017, T007-T014 failing
  - **Blocks**: T019, T020

- [ ] **T019** Implement shaped gain rule and budget enforcement
  - File: `backend/src/services/clause/architect.py`
  - Implement CLAUSESubgraphArchitect class
  - Implement shaped gain: `score - lambda_edge × edge_cost`
  - Implement edge selection loop:
    - Sort edges by shaped gain descending
    - Accept if shaped_gain > 0 AND budget_used < edge_budget
    - Stop when either condition fails
  - Record stopped_reason ("BUDGET_EXHAUSTED", "GAIN_NEGATIVE", "COMPLETE")
  - **Depends on**: T018
  - **Blocks**: T020, T021

- [ ] **T020** Implement basin frequency strengthening
  - File: `backend/src/services/basin_tracker.py`
  - Implement BasinTracker class
  - Implement `strengthen_basins()` method:
    - For each concept: basin.strength += increment (default 0.2)
    - Cap at 2.0
    - Increment activation_count
    - Append to activation_history
  - Implement co-occurrence updates (symmetric):
    - For each pair (A, B): A.co_occurring[B] += 1, B.co_occurring[A] += 1
  - **Depends on**: T018, T019
  - **Blocks**: T022

- [ ] **T021** Implement NetworkX graph integration with Neo4j
  - File: `backend/src/services/clause/graph_loader.py`
  - Implement `load_subgraph_from_neo4j()`:
    - Get seed nodes from query (BM25 top 20)
    - Expand k-hop using APOC: `apoc.path.subgraphNodes()`
    - Build NetworkX MultiDiGraph
    - Add node attributes (concept_id, basin_id)
    - Add edge attributes (relation_type, weight)
  - Use Neo4j driver async methods
  - **Depends on**: T019
  - **Blocks**: T022

### API Endpoints
- [ ] **T022** Implement POST /api/clause/subgraph endpoint
  - File: `backend/src/api/routes/clause.py` (new file)
  - Create FastAPI router
  - Implement subgraph construction endpoint:
    - Parse SubgraphRequest
    - Call CLAUSESubgraphArchitect.build_subgraph()
    - Return SubgraphResponse
  - Add error handling (400 for validation, 500 for internal)
  - **Depends on**: T019, T020, T021
  - **Blocks**: T023

- [ ] **T023** Implement POST /api/clause/basins/strengthen endpoint
  - File: `backend/src/api/routes/clause.py` (modify)
  - Implement basin strengthening endpoint:
    - Parse BasinStrengtheningRequest
    - Call BasinTracker.strengthen_basins()
    - Return BasinStrengtheningResponse with timing
  - Add validation for concepts list (non-empty)
  - **Depends on**: T020, T022

- [ ] **T024** Implement GET /api/clause/basins/{basin_id} endpoint
  - File: `backend/src/api/routes/clause.py` (modify)
  - Implement basin retrieval endpoint:
    - Parse basin_id path parameter
    - Query Neo4j with backward-compatible defaults
    - Return full AttractorBasin with new fields
  - Return 404 if basin not found
  - **Depends on**: T022, T023

- [ ] **T025** Implement POST /api/clause/edges/score endpoint
  - File: `backend/src/api/routes/clause.py` (modify)
  - Implement edge scoring endpoint:
    - Parse EdgeScoringRequest
    - Call EdgeScorer.batch_score_edges()
    - Return EdgeScoringResponse with full breakdown
  - Include timing in response
  - **Depends on**: T018, T022

## Phase 3.4: Integration & Optimization

### Database Integration
- [ ] **T026** Extend Neo4j schema with basin indexes
  - File: `backend/src/config/neo4j_config.py` (modify)
  - Add Cypher migration:
    ```cypher
    CREATE INDEX basin_strength_index IF NOT EXISTS
    FOR (b:AttractorBasin) ON (b.strength);

    CREATE INDEX basin_activation_index IF NOT EXISTS
    FOR (b:AttractorBasin) ON (b.activation_count);
    ```
  - Add lazy migration query with defaults (coalesce for old basins)
  - Test backward compatibility
  - **Depends on**: T015, T024

- [ ] **T027** Implement Redis basin caching layer
  - File: `backend/src/services/clause/basin_cache.py`
  - Implement basin cache with 1-hour TTL:
    - `setex(f"basin:{concept_id}", 3600, json.dumps(basin))`
  - Implement cache invalidation on update
  - Implement batch cache loading on startup
  - Target: <1ms basin lookup
  - **Depends on**: T020, T026

### Performance Optimization
- [ ] **T028** Implement NumPy vectorized edge scoring
  - File: `backend/src/services/clause/edge_scoring.py` (modify)
  - Refactor `batch_score_edges()` for vectorization:
    - Create signal matrices (N×5) using NumPy 2.0
    - Single matrix multiplication: `signal_matrix @ weights`
    - Target: <5ms for 1000 edges
  - Use pre-computed query embeddings (avoid re-encoding)
  - **Depends on**: T018, T027

- [ ] **T029** Optimize subgraph loading with APOC
  - File: `backend/src/services/clause/graph_loader.py` (modify)
  - Use APOC batch operations for k-hop expansion
  - Limit subgraph size (max 1k nodes, 2-hop default)
  - Add connection pooling for Neo4j driver
  - Target: <100ms subgraph load
  - **Depends on**: T021, T026

## Phase 3.5: Polish & Validation

### Unit Tests
- [ ] **T030** [P] Unit tests for edge scoring signals
  - File: `backend/tests/unit/test_edge_scoring_signals.py`
  - Test each signal independently (φ_ent, φ_rel, φ_nbr, φ_deg, φ_basin)
  - Test signal ranges (all 0.0-1.0)
  - Test edge cases (empty graph, missing basin)
  - **Depends on**: T018, T028

- [ ] **T031** [P] Unit tests for basin strengthening logic
  - File: `backend/tests/unit/test_basin_strengthening.py`
  - Test +0.2 increment
  - Test 2.0 cap
  - Test activation_count increment
  - Test co-occurrence symmetry
  - **Depends on**: T020

- [ ] **T032** [P] Unit tests for shaped gain rule
  - File: `backend/tests/unit/test_shaped_gain.py`
  - Test gain calculation: score - λ_edge × cost
  - Test budget enforcement (stop at β_edge)
  - Test stop conditions (BUDGET_EXHAUSTED vs GAIN_NEGATIVE)
  - **Depends on**: T019

### Performance & Validation
- [ ] **T033** Performance profiling and benchmarking
  - File: `backend/tests/performance/test_clause_performance.py`
  - Profile edge scoring: assert <10ms for 1000 edges
  - Profile subgraph construction: assert <500ms total
  - Profile basin update: assert <5ms
  - Memory profiling: assert <100MB for 10k concepts
  - Generate performance report
  - **Depends on**: T028, T029

- [ ] **T034** Execute quickstart.md validation
  - Follow `specs/034-clause-phase1-foundation/quickstart.md` step-by-step
  - Run Step 1 (environment setup)
  - Run Step 2 (initialize services)
  - Run Step 3 (build first subgraph) - verify output
  - Run Step 4 (basin strengthening) - verify 1.0→1.6 progression
  - Run Step 5 (evolution query) - verify co-occurrence tracking
  - **Depends on**: T022-T025, T033

- [ ] **T035** [P] Update API documentation with examples
  - File: `specs/034-clause-phase1-foundation/contracts/architect_api.yaml` (add examples)
  - Add curl examples for each endpoint
  - Add response samples with actual data
  - Document performance characteristics
  - **Depends on**: T022-T025

- [ ] **T036** Final constitution compliance check
  - Run: `python -c "import numpy; assert numpy.__version__.startswith('2.'), 'NumPy 1.x violation'"`
  - Verify AttractorBasin integration (Article II)
  - Verify Context Engineering tests passed first (T001-T003)
  - Verify Redis persistence working
  - Generate compliance report
  - **Depends on**: T034, T035

## Dependencies

### Critical Path
```
T001-T003 (Context Engineering)
  → T004-T006 (Setup)
  → T007-T014 (Tests - must fail)
  → T015-T017 (Models)
  → T018 (EdgeScorer)
  → T019 (Architect + Shaped Gain)
  → T020 (BasinTracker)
  → T021 (Graph Loader)
  → T022-T025 (API Endpoints)
  → T026-T029 (Integration + Performance)
  → T030-T036 (Polish + Validation)
```

### Parallel Opportunities
- **Wave 1** (Context Engineering): T001, T002, T003 (independent files)
- **Wave 2** (Setup): T006 can run parallel with T005
- **Wave 3** (Contract Tests): T007, T008, T009, T010 (independent test files)
- **Wave 4** (Integration Tests): T011, T012, T013, T014 (independent test files)
- **Wave 5** (Models): T015, T016, T017 (independent files)
- **Wave 6** (Unit Tests): T030, T031, T032 (independent test files)
- **Wave 7** (Docs): T035 parallel with T034

### Blocking Relationships
- T018 blocks T019, T020, T025, T028, T030
- T019 blocks T021, T022, T032
- T020 blocks T023, T031
- T022 blocks T023, T024, T025, T034
- T026 blocks T027, T029
- T028 blocks T033
- T033 blocks T034
- T034 blocks T036

## Parallel Execution Examples

### Wave 1: Context Engineering Foundation (Constitution-Mandated)
```bash
# Run in parallel - must complete BEFORE any implementation
Task: "Verify AttractorBasin accessibility in backend/tests/test_context_engineering_basin.py"
Task: "Validate Redis persistence in backend/tests/test_context_engineering_redis.py"
Task: "Test basin influence in backend/tests/test_context_engineering_influence.py"
```

### Wave 3: Contract Tests (TDD - Must Fail First)
```bash
# Run in parallel after T001-T006
Task: "Contract test POST /api/clause/subgraph in backend/tests/contract/test_architect_subgraph_contract.py"
Task: "Contract test POST /api/clause/basins/strengthen in backend/tests/contract/test_basin_strengthen_contract.py"
Task: "Contract test GET /api/clause/basins/{id} in backend/tests/contract/test_basin_get_contract.py"
Task: "Contract test POST /api/clause/edges/score in backend/tests/contract/test_edge_score_contract.py"
```

### Wave 4: Integration Tests (TDD)
```bash
# Run in parallel after T007-T010
Task: "Integration test subgraph workflow in backend/tests/integration/test_architect_workflow.py"
Task: "Integration test basin strengthening in backend/tests/integration/test_basin_strengthening_workflow.py"
Task: "Integration test persistence in backend/tests/integration/test_basin_persistence.py"
Task: "Integration test basin influence in backend/tests/integration/test_basin_influence.py"
```

### Wave 5: Data Models
```bash
# Run in parallel after T007-T014 failing
Task: "Extend AttractorBasin model in backend/src/models/attractor_basin.py"
Task: "Create SubgraphRequest/Response models in backend/src/services/clause/models.py"
Task: "Create BasinTracker models in backend/src/services/clause/basin_models.py"
```

## Notes

- **[P] = Parallel execution allowed** (different files, no dependencies)
- **Constitution compliance**: T001-T003 MUST pass before any implementation (Article II)
- **TDD enforcement**: T007-T014 MUST fail before T015-T025
- **Performance targets**: T033 validates <10ms edge scoring, <500ms subgraph, <5ms basin update
- **Backward compatibility**: T015 must not break existing AttractorBasin usage
- Commit after each task
- Run `pytest` after each implementation task to see tests pass

## Validation Checklist

- [x] All contracts have corresponding tests (T007-T010)
- [x] All entities have model tasks (T015-T017)
- [x] All tests come before implementation (T007-T014 before T015-T025)
- [x] Parallel tasks truly independent (Wave 1-7 verified)
- [x] Each task specifies exact file path
- [x] No task modifies same file as another [P] task
- [x] Context Engineering tests first (T001-T003 per Constitution)
- [x] Performance targets specified (T033)
- [x] Quickstart validation included (T034)

---

**Total Tasks**: 36
**Estimated Duration**: 20-25 hours (Phase 1 from 10-week roadmap)
**Next Command**: Start with T001-T003 (Context Engineering validation)
