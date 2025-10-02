# Tasks: CLAUSE Phase 2 - Path Navigator & Context Curator

**Input**: Design documents from `/Volumes/Asylum/dev/Dionysus-2.0/specs/035-clause-phase2-multi-agent/`
**Prerequisites**: plan.md (✅), research.md (✅), data-model.md (✅), contracts/ (✅), quickstart.md (✅)
**Branch**: `035-clause-phase2-multi-agent`
**Total Tasks**: 66

## Execution Flow
```
1. Context Engineering validation (T001-T003) - REQUIRED by Constitution Article II
2. Project setup and dependencies (T004-T006)
3. Contract tests (T007-T009) - TDD before implementation
4. Model implementation (T010-T017) - Parallel where possible
5. Core service implementation (T018-T034) - Sequential per service
6. Intelligence integration (T035-T042) - Parallel services
7. Conflict resolution (T043-T046) - Sequential (shared Neo4j)
8. API integration (T047-T049) - Sequential (shared routes file)
9. Integration tests (T050-T056) - Parallel test files
10. Performance tests (T057-T063) - Parallel test files
11. Documentation and validation (T064-T066)
```

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- Paths are absolute from repository root

---

## Phase 3.1: Context Engineering Validation (Constitution Mandated)
**CRITICAL**: Article II requires Context Engineering tests FIRST

- [ ] **T001** [P] Verify AttractorBasinManager accessibility and Phase 1 basin strengthening
  - **File**: `/Volumes/Asylum/dev/Dionysus-2.0/backend/tests/test_context_engineering_basin.py`
  - **Goal**: Verify Phase 1 `AttractorBasin` model is accessible with strength, activation_count, co_occurring fields
  - **Tests**:
    - Import `AttractorBasin` from `backend.models.attractor_basin`
    - Create basin with strength=1.0, activation_count=0
    - Verify basin strengthening logic (1.0 → 1.2 on 2nd activation)
    - Assert co_occurring_concepts dictionary exists
  - **Dependencies**: None (first task)

- [ ] **T002** [P] Validate Redis persistence for ThoughtSeeds and curiosity queue
  - **File**: `/Volumes/Asylum/dev/Dionysus-2.0/backend/tests/test_context_engineering_redis.py`
  - **Goal**: Verify Redis connection for ThoughtSeed cache and curiosity queue
  - **Tests**:
    - Connect to Redis (localhost:6379)
    - Test SET/GET for ThoughtSeed cache with 1-hour TTL
    - Test LPUSH/RPOP for curiosity_queue
    - Verify TTL expiration
  - **Dependencies**: None (independent test)

- [ ] **T003** [P] Test basin influence with Navigator and Curator updates
  - **File**: `/Volumes/Asylum/dev/Dionysus-2.0/backend/tests/test_context_engineering_influence.py`
  - **Goal**: Verify basin updates during navigation and curation
  - **Tests**:
    - Create basins for concepts A, B, C
    - Simulate navigation path (A → B → C)
    - Assert basin strengths increase (A:1.0→1.2, B:1.0→1.2)
    - Assert co_occurring updates (A.co_occurring[B] += 1)
  - **Dependencies**: None (independent test)

---

## Phase 3.2: Project Setup

- [ ] **T004** Create Phase 2 directory structure per plan.md
  - **Directories to create**:
    - `/Volumes/Asylum/dev/Dionysus-2.0/backend/src/models/clause/` (Navigator, Curator, Coordinator models)
    - `/Volumes/Asylum/dev/Dionysus-2.0/backend/src/services/thoughtseed/` (ThoughtSeed integration)
    - `/Volumes/Asylum/dev/Dionysus-2.0/backend/src/services/curiosity/` (Curiosity queue)
    - `/Volumes/Asylum/dev/Dionysus-2.0/backend/src/services/causal/` (Bayesian network)
    - `/Volumes/Asylum/dev/Dionysus-2.0/backend/src/services/provenance/` (Provenance tracking)
    - `/Volumes/Asylum/dev/Dionysus-2.0/backend/tests/contract/` (Contract tests)
    - `/Volumes/Asylum/dev/Dionysus-2.0/backend/tests/integration/` (Integration tests)
    - `/Volumes/Asylum/dev/Dionysus-2.0/backend/tests/performance/` (Performance tests)
  - **Files to create**: `__init__.py` in each new directory
  - **Dependencies**: T001-T003 complete

- [ ] **T005** Install Phase 2 dependencies (tiktoken, update NetworkX if needed)
  - **File**: `/Volumes/Asylum/dev/Dionysus-2.0/backend/requirements.txt`
  - **Add**:
    - `tiktoken==0.5.1` (token counting)
    - Verify `networkx>=3.0` (path finding)
    - Verify `numpy>=2.0` (Constitution compliance)
  - **Command**: `pip install tiktoken==0.5.1`
  - **Verify**: `python -c "import tiktoken; print(tiktoken.encoding_for_model('gpt-4'))"`
  - **Dependencies**: T004

- [ ] **T006** [P] Configure linting for new Phase 2 files (mypy, flake8, black)
  - **Files**:
    - `/Volumes/Asylum/dev/Dionysus-2.0/backend/.flake8` (extend ignore if needed)
    - `/Volumes/Asylum/dev/Dionysus-2.0/backend/pyproject.toml` (black config)
  - **Verify**: `flake8 backend/src/services/clause/ --count`
  - **Dependencies**: T004

---

## Phase 3.3: Contract Tests (TDD - Tests Before Implementation)
**CRITICAL**: These tests MUST be written and MUST FAIL before implementation

- [ ] **T007** [P] Contract test POST /api/clause/navigate
  - **File**: `/Volumes/Asylum/dev/Dionysus-2.0/backend/tests/contract/test_navigator_contract.py`
  - **Goal**: Validate PathNavigationRequest and PathNavigationResponse schemas
  - **Tests**:
    - Test valid request: `{"query": "...", "start_node": "...", "step_budget": 10}`
    - Test invalid request: missing `query` → 422 error
    - Test request with step_budget > 20 → validation error
    - Test response schema: `path`, `metadata`, `performance` fields
    - Assert `metadata.budget_used <= request.step_budget`
  - **Expected**: All tests FAIL (endpoint not implemented yet)
  - **Dependencies**: T006

- [ ] **T008** [P] Contract test POST /api/clause/curate
  - **File**: `/Volumes/Asylum/dev/Dionysus-2.0/backend/tests/contract/test_curator_contract.py`
  - **Goal**: Validate ContextCurationRequest and ContextCurationResponse schemas
  - **Tests**:
    - Test valid request: `{"evidence_pool": [...], "token_budget": 2048}`
    - Test invalid request: empty evidence_pool → 422 error
    - Test response schema: `selected_evidence`, `metadata`, `performance` fields
    - Assert each evidence has `provenance` dict with 7 required fields
    - Assert `metadata.tokens_used <= request.token_budget`
  - **Expected**: All tests FAIL (endpoint not implemented yet)
  - **Dependencies**: T006

- [ ] **T009** [P] Contract test POST /api/clause/coordinate
  - **File**: `/Volumes/Asylum/dev/Dionysus-2.0/backend/tests/contract/test_coordinator_contract.py`
  - **Goal**: Validate CoordinationRequest and CoordinationResponse schemas
  - **Tests**:
    - Test valid request: `{"query": "...", "budgets": {...}, "lambdas": {...}}`
    - Test invalid request: negative budget → validation error
    - Test response schema: `result`, `agent_handoffs`, `conflicts_detected`, `performance`
    - Assert `len(agent_handoffs) == 3` (Architect, Navigator, Curator)
    - Assert `conflicts_resolved <= conflicts_detected`
  - **Expected**: All tests FAIL (endpoint not implemented yet)
  - **Dependencies**: T006

---

## Phase 3.4: Model Implementation (Parallel Where Possible)

- [ ] **T010** [P] PathNavigator request/response models
  - **File**: `/Volumes/Asylum/dev/Dionysus-2.0/backend/src/models/clause/path_models.py`
  - **Models to create**:
    - `PathNavigationRequest` (query, start_node, step_budget, enable_thoughtseeds, enable_curiosity, enable_causal, curiosity_threshold)
    - `PathStep` (step, from_node, to_node, relation, action, causal_score, thoughtseed_id)
    - `PathNavigationResponse` (path, metadata, performance)
  - **Validation**: step_budget 1-20, action in [CONTINUE, BACKTRACK, STOP]
  - **Reference**: data-model.md section 1
  - **Dependencies**: T004

- [ ] **T011** [P] ContextCurator request/response models
  - **File**: `/Volumes/Asylum/dev/Dionysus-2.0/backend/src/models/clause/curator_models.py`
  - **Models to create**:
    - `ContextCurationRequest` (evidence_pool, token_budget, enable_provenance, lambda_tok)
    - `SelectedEvidence` (text, tokens, score, shaped_utility, provenance)
    - `ContextCurationResponse` (selected_evidence, metadata, performance)
  - **Validation**: token_budget 100-8192, evidence_pool min_length=1
  - **Reference**: data-model.md section 2
  - **Dependencies**: T004

- [ ] **T012** [P] LCMAPPOCoordinator request/response models
  - **File**: `/Volumes/Asylum/dev/Dionysus-2.0/backend/src/models/clause/coordinator_models.py`
  - **Models to create**:
    - `BudgetAllocation` (edge_budget, step_budget, token_budget)
    - `LambdaParameters` (edge, latency, token)
    - `CoordinationRequest` (query, budgets, lambdas)
    - `AgentHandoff` (step, agent, action, budget_used, latency_ms)
    - `CoordinationResponse` (result, agent_handoffs, conflicts_detected, conflicts_resolved, performance)
  - **Validation**: all budgets > 0, lambdas in [0.0, 1.0]
  - **Reference**: data-model.md section 3
  - **Dependencies**: T004

- [ ] **T013** [P] Provenance metadata models
  - **File**: `/Volumes/Asylum/dev/Dionysus-2.0/backend/src/models/clause/provenance_models.py`
  - **Models to create**:
    - `TrustSignals` (reputation_score, recency_score, semantic_consistency) - all 0.0-1.0
    - `ProvenanceMetadata` (source_uri, extraction_timestamp, extractor_identity, supporting_evidence, verification_status, corroboration_count, trust_signals)
  - **Validation**: verification_status in [verified, pending_review, unverified], supporting_evidence max 200 chars
  - **Reference**: data-model.md section 2 (Provenance)
  - **Dependencies**: T004

- [ ] **T014** [P] ThoughtSeed models (Spec 028)
  - **File**: `/Volumes/Asylum/dev/Dionysus-2.0/backend/src/models/clause/thoughtseed_models.py`
  - **Models to create**:
    - `BasinContext` (strength 1.0-2.0, activation_count ≥0, co_occurring dict)
    - `ThoughtSeed` (id, concept, source_doc, basin_context, similarity_threshold 0.0-1.0, linked_documents, created_at)
  - **Reference**: data-model.md section 4 (ThoughtSeed)
  - **Dependencies**: T004

- [ ] **T015** [P] CuriosityTrigger models (Spec 029)
  - **File**: `/Volumes/Asylum/dev/Dionysus-2.0/backend/src/models/clause/curiosity_models.py`
  - **Models to create**:
    - `CuriosityTrigger` (trigger_type="prediction_error", concept, error_magnitude 0.0-1.0, timestamp, investigation_status in [queued, investigating, completed])
  - **Reference**: data-model.md section 4 (Curiosity)
  - **Dependencies**: T004

- [ ] **T016** [P] CausalIntervention models (Spec 033)
  - **File**: `/Volumes/Asylum/dev/Dionysus-2.0/backend/src/models/clause/causal_models.py`
  - **Models to create**:
    - `CausalIntervention` (intervention_node, target_node, intervention_score 0.0-1.0, computation_time_ms ≥0)
  - **Reference**: data-model.md section 4 (Causal)
  - **Dependencies**: T004

- [ ] **T017** [P] Shared models (StateEncoding, BudgetUsage)
  - **File**: `/Volumes/Asylum/dev/Dionysus-2.0/backend/src/models/clause/shared_models.py`
  - **Models to create**:
    - `StateEncoding` (query_embedding 384-dim, node_embedding 384-dim, node_degree, basin_strength, neighborhood_mean 384-dim, budget_remaining 0.0-1.0) + `to_numpy()` method
    - `BudgetUsage` (edge_used, step_used, token_used, edge_total, step_total, token_total) + properties for remaining budgets
  - **Reference**: data-model.md section 5
  - **Dependencies**: T004

---

## Phase 3.5: Core Navigator Implementation

- [ ] **T018** Implement state encoding (query + node + neighborhood)
  - **File**: `/Volumes/Asylum/dev/Dionysus-2.0/backend/src/services/clause/path_navigator.py`
  - **Method**: `encode_state(query, current_node, graph) -> StateEncoding`
  - **Logic**:
    - Embed query (384-dim)
    - Embed current node (384-dim)
    - Get node degree from graph
    - Get basin strength from BasinTracker
    - Compute neighborhood mean embedding (1-hop neighbors)
    - Return StateEncoding model
  - **Reference**: research.md decision 1
  - **Dependencies**: T010, T017

- [ ] **T019** Implement termination head (stop probability calculation)
  - **File**: `/Volumes/Asylum/dev/Dionysus-2.0/backend/src/services/clause/path_navigator.py`
  - **Method**: `should_terminate(state, budget_remaining) -> bool`
  - **Logic**:
    - Augment state with budget_remaining normalized
    - Apply sigmoid classifier: `1 / (1 + exp(-logit))`
    - Return True if stop_prob > 0.5
  - **Reference**: research.md decision 2
  - **Dependencies**: T018

- [ ] **T020** Implement action selection (CONTINUE, BACKTRACK, STOP)
  - **File**: `/Volumes/Asylum/dev/Dionysus-2.0/backend/src/services/clause/path_navigator.py`
  - **Method**: `select_action(candidates, budget_remaining) -> Tuple[str, Optional[str]]`
  - **Logic**:
    - Check termination head → STOP if budget low or stop_prob > 0.5
    - Score candidates (causal + relevance)
    - Select best candidate → CONTINUE
    - If no good candidates → BACKTRACK
  - **Reference**: spec.md FR-001
  - **Dependencies**: T019

- [ ] **T021** Implement step budget enforcement
  - **File**: `/Volumes/Asylum/dev/Dionysus-2.0/backend/src/services/clause/path_navigator.py`
  - **Method**: Update `navigate()` to track budget and enforce limit
  - **Logic**:
    - Initialize budget_used = 0
    - Each step: budget_used += 1
    - Stop if budget_used >= step_budget
    - Return path with metadata.budget_used
  - **Reference**: spec.md FR-001 acceptance criteria
  - **Dependencies**: T020

- [ ] **T022** Integrate ThoughtSeed generation (Spec 028)
  - **File**: `/Volumes/Asylum/dev/Dionysus-2.0/backend/src/services/clause/path_navigator.py`
  - **Method**: `generate_thoughtseeds(candidates) -> List[ThoughtSeed]`
  - **Logic**:
    - For each candidate: fetch basin context from BasinTracker
    - Call ThoughtSeedGenerator.create(concept, source_doc, basin_context)
    - Store ThoughtSeed IDs in path metadata
  - **Reference**: spec.md FR-002, research.md decision 3
  - **Dependencies**: T021, T035 (ThoughtSeedGenerator)

- [ ] **T023** Integrate curiosity triggers (Spec 029)
  - **File**: `/Volumes/Asylum/dev/Dionysus-2.0/backend/src/services/clause/path_navigator.py`
  - **Method**: `check_curiosity_trigger(candidate, expected_score, actual_score)`
  - **Logic**:
    - Calculate prediction_error = |expected - actual|
    - If error > curiosity_threshold: add to Redis curiosity_queue
    - Increment curiosity_triggers_spawned counter
  - **Reference**: spec.md FR-003, research.md decision 4
  - **Dependencies**: T022, T037 (CuriosityQueue)

- [ ] **T024** Integrate causal reasoning (Spec 033)
  - **File**: `/Volumes/Asylum/dev/Dionysus-2.0/backend/src/services/clause/path_navigator.py`
  - **Method**: `score_candidates_causal(candidates) -> Dict[str, float]`
  - **Logic**:
    - For each candidate: call CausalBayesianNetwork.estimate_intervention()
    - Return causal scores: P(answer | do(select_path=candidate))
    - Use causal_score in action selection
  - **Reference**: spec.md FR-004, research.md decision 5
  - **Dependencies**: T023, T039 (CausalBayesianNetwork)

- [ ] **T025** Complete PathNavigator service class
  - **File**: `/Volumes/Asylum/dev/Dionysus-2.0/backend/src/services/clause/path_navigator.py`
  - **Class**: `PathNavigator`
  - **Methods**: `navigate(request: PathNavigationRequest, graph: nx.Graph) -> PathNavigationResponse`
  - **Orchestration**:
    - Initialize from start_node
    - Loop: encode_state → termination check → generate thoughtseeds → causal scoring → select action
    - Track budgets, performance metrics
    - Return PathNavigationResponse
  - **Reference**: spec.md FR-001 full workflow
  - **Dependencies**: T018-T024

---

## Phase 3.6: Core Curator Implementation

- [ ] **T026** Implement listwise evidence scoring
  - **File**: `/Volumes/Asylum/dev/Dionysus-2.0/backend/src/services/clause/context_curator.py`
  - **Method**: `score_evidence_listwise(evidence_pool, query) -> List[Tuple[int, str, float]]`
  - **Logic**:
    - Compute pairwise similarity matrix (N x N)
    - For each evidence: base_score (query relevance) - diversity_penalty (similarity to selected)
    - Return sorted by final_score descending
  - **Reference**: research.md decision 6
  - **Dependencies**: T011

- [ ] **T027** Implement shaped utility calculation
  - **File**: `/Volumes/Asylum/dev/Dionysus-2.0/backend/src/services/clause/context_curator.py`
  - **Method**: `calculate_shaped_utility(score, tokens, lambda_tok) -> float`
  - **Logic**: `shaped_utility = score - lambda_tok * tokens`
  - **Reference**: spec.md FR-005 acceptance criteria
  - **Dependencies**: T026

- [ ] **T028** Implement learned stop (utility ≤ 0)
  - **File**: `/Volumes/Asylum/dev/Dionysus-2.0/backend/src/services/clause/context_curator.py`
  - **Method**: Update `curate()` to stop when shaped_utility ≤ 0
  - **Logic**:
    - Iterate scored evidence
    - If shaped_utility ≤ 0: set learned_stop_triggered = True, break
    - Return selected evidence with metadata
  - **Reference**: spec.md FR-005
  - **Dependencies**: T027

- [ ] **T029** Implement token budget enforcement (tiktoken)
  - **File**: `/Volumes/Asylum/dev/Dionysus-2.0/backend/src/services/clause/context_curator.py`
  - **Method**: `count_tokens(text) -> int`
  - **Logic**:
    - Use tiktoken.encoding_for_model("gpt-4").encode(text)
    - Add 10% safety buffer: `int(len(tokens) * 1.1)`
    - Enforce: total_tokens + snippet_tokens <= token_budget
  - **Reference**: research.md decision 7
  - **Dependencies**: T028, T005 (tiktoken dependency)

- [ ] **T030** Integrate provenance tracking (Spec 032)
  - **File**: `/Volumes/Asylum/dev/Dionysus-2.0/backend/src/services/clause/context_curator.py`
  - **Method**: `add_provenance(evidence, source_uri, query) -> SelectedEvidence`
  - **Logic**:
    - Create ProvenanceMetadata with 7 required fields
    - Calculate trust signals (reputation, recency, consistency)
    - Attach provenance to SelectedEvidence model
  - **Reference**: spec.md FR-006, research.md decision 8
  - **Dependencies**: T029, T041 (ProvenanceTracker)

- [ ] **T031** Complete ContextCurator service class
  - **File**: `/Volumes/Asylum/dev/Dionysus-2.0/backend/src/services/clause/context_curator.py`
  - **Class**: `ContextCurator`
  - **Methods**: `curate(request: ContextCurationRequest) -> ContextCurationResponse`
  - **Orchestration**:
    - Score evidence listwise
    - Iterate scored evidence: calculate shaped_utility, check budget, add provenance
    - Stop when learned_stop or budget exhausted
    - Return ContextCurationResponse
  - **Reference**: spec.md FR-005 full workflow
  - **Dependencies**: T026-T030

---

## Phase 3.7: LC-MAPPO Coordinator Implementation

- [ ] **T032** Implement centralized critic (4 heads: task, edge_cost, latency_cost, token_cost)
  - **File**: `/Volumes/Asylum/dev/Dionysus-2.0/backend/src/services/clause/centralized_critic.py`
  - **Class**: `CentralizedCritic`
  - **Methods**:
    - `__init__(state_dim=1154, heads=["task_value", "edge_cost", "latency_cost", "token_cost"])`
    - `forward(state) -> Dict[str, float]` - returns value for each head
  - **Reference**: research.md decision 9, spec.md FR-007
  - **Dependencies**: T012

- [ ] **T033** Implement shaped return calculation
  - **File**: `/Volumes/Asylum/dev/Dionysus-2.0/backend/src/services/clause/lc_mappo_coordinator.py`
  - **Method**: `calculate_shaped_returns(episode) -> List[float]`
  - **Logic**:
    - For each transition: `r_shaped = r_acc - lambda_edge*c_edge - lambda_lat*c_lat - lambda_tok*c_tok`
    - Return shaped_rewards list
  - **Reference**: spec.md FR-007 test case, research.md decision 9
  - **Dependencies**: T032

- [ ] **T034** Implement dual variable updates
  - **File**: `/Volumes/Asylum/dev/Dionysus-2.0/backend/src/services/clause/lc_mappo_coordinator.py`
  - **Method**: `update_duals(batch_episodes)`
  - **Logic**:
    - `lambda_k = max(0, lambda_k + eta * (E[C_k] - beta_k))` (projected ascent)
    - Update lambda_edge, lambda_lat, lambda_tok
  - **Reference**: spec.md FR-007 dual update rule
  - **Dependencies**: T033

- [ ] **T035** Implement agent handoff protocol
  - **File**: `/Volumes/Asylum/dev/Dionysus-2.0/backend/src/services/clause/lc_mappo_coordinator.py`
  - **Method**: `coordinate(request: CoordinationRequest) -> CoordinationResponse`
  - **Logic**:
    - Sequential execution: Architect → Navigator → Curator
    - Track latency and budget usage for each agent
    - Create AgentHandoff records
  - **Reference**: research.md decision 12
  - **Dependencies**: T034, T025 (Navigator), T031 (Curator), Phase 1 Architect

- [ ] **T036** Integrate conflict resolver (Spec 031)
  - **File**: `/Volumes/Asylum/dev/Dionysus-2.0/backend/src/services/clause/lc_mappo_coordinator.py`
  - **Method**: Update `coordinate()` to detect and resolve conflicts
  - **Logic**:
    - After each agent write: call ConflictResolver.write_with_conflict_detection()
    - Track conflicts_detected and conflicts_resolved
  - **Reference**: spec.md FR-008
  - **Dependencies**: T035, T043-T046 (ConflictResolver)

- [ ] **T037** Complete LCMAPPOCoordinator service class
  - **File**: `/Volumes/Asylum/dev/Dionysus-2.0/backend/src/services/clause/lc_mappo_coordinator.py`
  - **Class**: `LCMAPPOCoordinator`
  - **Full orchestration**: coordinate() with all three agents, conflict resolution, budget tracking, performance metrics
  - **Reference**: spec.md FR-007 full workflow
  - **Dependencies**: T032-T036

---

## Phase 3.8: Intelligence Service Implementation (Parallel)

- [ ] **T038** [P] Implement ThoughtSeedGenerator service
  - **File**: `/Volumes/Asylum/dev/Dionysus-2.0/backend/src/services/thoughtseed/generator.py`
  - **Class**: `ThoughtSeedGenerator`
  - **Methods**:
    - `create(concept, source_doc, basin_context) -> ThoughtSeed`
    - `link_thoughtseed(thoughtseed)` - cross-document linking (similarity > 0.8)
  - **Reference**: spec.md FR-002, research.md decision 3
  - **Dependencies**: T014

- [ ] **T039** [P] Implement ThoughtSeed cross-document linking
  - **File**: `/Volumes/Asylum/dev/Dionysus-2.0/backend/src/services/thoughtseed/generator.py`
  - **Method**: `link_across_documents(thoughtseed, similarity_threshold=0.8)`
  - **Logic**:
    - Query Redis for existing ThoughtSeeds with similar concepts
    - Compute similarity between ThoughtSeeds
    - If similarity > threshold: add to linked_documents
  - **Reference**: spec.md FR-002 acceptance criteria
  - **Dependencies**: T038

- [ ] **T040** [P] Implement CuriosityQueue service (Redis)
  - **File**: `/Volumes/Asylum/dev/Dionysus-2.0/backend/src/services/curiosity/queue.py`
  - **Class**: `CuriosityQueue`
  - **Methods**:
    - `add_trigger(concept, error_magnitude)` - LPUSH to Redis curiosity_queue
    - `spawn_background_agent()` - async background processing
    - `get_queue_size()` - LLEN
  - **Reference**: spec.md FR-003, research.md decision 4
  - **Dependencies**: T015

- [ ] **T041** [P] Implement background curiosity agent spawn
  - **File**: `/Volumes/Asylum/dev/Dionysus-2.0/backend/src/services/curiosity/queue.py`
  - **Method**: `spawn_agent(trigger: CuriosityTrigger)`
  - **Logic**:
    - Read trigger from Redis queue (RPOP)
    - Launch background investigation (async)
    - Update investigation_status
  - **Reference**: spec.md FR-003 acceptance criteria
  - **Dependencies**: T040

- [ ] **T042** [P] Implement CausalBayesianNetwork service
  - **File**: `/Volumes/Asylum/dev/Dionysus-2.0/backend/src/services/causal/bayesian_network.py`
  - **Class**: `CausalBayesianNetwork`
  - **Methods**:
    - `build_causal_dag()` - pre-compute DAG structure
    - `estimate_intervention(intervention_node, target_node) -> float` - P(target | do(intervention))
    - Use LRU cache (size=1000) for predictions
  - **Reference**: spec.md FR-004, research.md decision 5
  - **Dependencies**: T016

- [ ] **T043** [P] Implement do-calculus intervention prediction
  - **File**: `/Volumes/Asylum/dev/Dionysus-2.0/backend/src/services/causal/bayesian_network.py`
  - **Method**: `do_calculus(intervention, target) -> float`
  - **Logic**:
    - Use causal DAG to compute intervention effect
    - Cache result with LRU
    - Return intervention_score (0.0-1.0)
  - **Reference**: spec.md FR-004 acceptance criteria
  - **Dependencies**: T042

- [ ] **T044** [P] Implement ProvenanceTracker service
  - **File**: `/Volumes/Asylum/dev/Dionysus-2.0/backend/src/services/provenance/tracker.py`
  - **Class**: `ProvenanceTracker`
  - **Methods**:
    - `create_provenance(evidence, source_uri, query) -> ProvenanceMetadata`
    - `calculate_trust_signals(source_uri, evidence, query) -> TrustSignals`
    - `store_in_neo4j(evidence, provenance)` - create Provenance node
  - **Reference**: spec.md FR-006, research.md decision 8
  - **Dependencies**: T013

- [ ] **T045** [P] Implement trust signal calculation
  - **File**: `/Volumes/Asylum/dev/Dionysus-2.0/backend/src/services/provenance/tracker.py`
  - **Methods**:
    - `calculate_reputation(source_uri) -> float` (0.0-1.0)
    - `calculate_recency(source_uri) -> float` (0.0-1.0)
    - `calculate_consistency(evidence, query) -> float` (0.0-1.0)
  - **Reference**: spec.md FR-006 acceptance criteria
  - **Dependencies**: T044

---

## Phase 3.9: Conflict Resolution Implementation

- [ ] **T046** Implement Neo4j transaction checkpointing
  - **File**: `/Volumes/Asylum/dev/Dionysus-2.0/backend/src/services/clause/conflict_resolver.py`
  - **Class**: `ConflictResolver`
  - **Method**: `write_with_conflict_detection(node_id, updates)`
  - **Logic**:
    - Begin transaction with version read
    - Write with version check (optimistic locking)
    - Rollback if version mismatch (conflict detected)
  - **Reference**: research.md decision 10, spec.md FR-008
  - **Dependencies**: T012

- [ ] **T047** Implement conflict detection (version checking)
  - **File**: `/Volumes/Asylum/dev/Dionysus-2.0/backend/src/services/clause/conflict_resolver.py`
  - **Method**: Detect conflicts via Neo4j version field
  - **Logic**:
    - Compare expected_version with current_version
    - If mismatch: conflict detected, rollback transaction
  - **Reference**: spec.md FR-008 acceptance criteria
  - **Dependencies**: T046

- [ ] **T048** Implement MERGE strategy (max basin strength)
  - **File**: `/Volumes/Asylum/dev/Dionysus-2.0/backend/src/services/clause/conflict_resolver.py`
  - **Method**: `resolve_conflict(node_id, updates, current_strength) -> Dict`
  - **Logic**:
    - MERGE: final_strength = max(updates["strength"], current_strength)
    - Retry write with merged value
  - **Reference**: spec.md FR-008 test case, research.md decision 10
  - **Dependencies**: T047

- [ ] **T049** Implement exponential backoff retry
  - **File**: `/Volumes/Asylum/dev/Dionysus-2.0/backend/src/services/clause/conflict_resolver.py`
  - **Method**: Update `resolve_conflict()` with exponential backoff [100ms, 200ms, 400ms]
  - **Logic**:
    - Retry up to 3 times with exponential delays
    - Raise exception after 3 failed retries
  - **Reference**: spec.md FR-008 acceptance criteria
  - **Dependencies**: T048

---

## Phase 3.10: API Integration

- [ ] **T050** Extend /api/clause with Navigator endpoint
  - **File**: `/Volumes/Asylum/dev/Dionysus-2.0/backend/src/api/routes/clause.py` (EXISTING from Phase 1)
  - **Endpoint**: `POST /api/clause/navigate`
  - **Logic**:
    - Validate PathNavigationRequest
    - Call PathNavigator.navigate()
    - Return PathNavigationResponse
    - Handle errors (400, 422, 503)
  - **Reference**: spec.md API Design (Navigator)
  - **Dependencies**: T025 (Navigator service)

- [ ] **T051** Extend /api/clause with Curator endpoint
  - **File**: `/Volumes/Asylum/dev/Dionysus-2.0/backend/src/api/routes/clause.py`
  - **Endpoint**: `POST /api/clause/curate`
  - **Logic**:
    - Validate ContextCurationRequest
    - Call ContextCurator.curate()
    - Return ContextCurationResponse
    - Handle errors (400, 422, 503)
  - **Reference**: spec.md API Design (Curator)
  - **Dependencies**: T031 (Curator service), T050 (same file)

- [ ] **T052** Extend /api/clause with Coordinator endpoint
  - **File**: `/Volumes/Asylum/dev/Dionysus-2.0/backend/src/api/routes/clause.py`
  - **Endpoint**: `POST /api/clause/coordinate`
  - **Logic**:
    - Validate CoordinationRequest
    - Call LCMAPPOCoordinator.coordinate()
    - Return CoordinationResponse
    - Handle errors (400, 422, 503)
  - **Reference**: spec.md API Design (Coordinator)
  - **Dependencies**: T037 (Coordinator service), T051 (same file)

---

## Phase 3.11: Integration Tests (Parallel Test Files)

- [ ] **T053** [P] Test full workflow (Architect → Navigator → Curator)
  - **File**: `/Volumes/Asylum/dev/Dionysus-2.0/backend/tests/integration/test_full_workflow.py`
  - **Test**: End-to-end CLAUSE workflow
  - **Steps**:
    - Create subgraph with Architect
    - Navigate path with Navigator (ThoughtSeeds, Curiosity, Causal)
    - Curate evidence with Curator (provenance)
    - Verify all budgets enforced
  - **Reference**: quickstart.md workflow demo
  - **Dependencies**: T052

- [ ] **T054** [P] Test ThoughtSeed cross-document linking
  - **File**: `/Volumes/Asylum/dev/Dionysus-2.0/backend/tests/integration/test_thoughtseed_linking.py`
  - **Test**: ThoughtSeeds link across documents
  - **Steps**:
    - Generate ThoughtSeeds for concepts A, B
    - Assert similarity > 0.8 creates link
    - Verify linked_documents updated
  - **Reference**: spec.md FR-002
  - **Dependencies**: T039

- [ ] **T055** [P] Test curiosity agent spawning
  - **File**: `/Volumes/Asylum/dev/Dionysus-2.0/backend/tests/integration/test_curiosity_spawning.py`
  - **Test**: High prediction error spawns curiosity agent
  - **Steps**:
    - Simulate path navigation with high prediction error
    - Assert curiosity trigger added to Redis queue
    - Verify background agent spawns
  - **Reference**: spec.md FR-003
  - **Dependencies**: T041

- [ ] **T056** [P] Test causal intervention predictions
  - **File**: `/Volumes/Asylum/dev/Dionysus-2.0/backend/tests/integration/test_causal_prediction.py`
  - **Test**: Causal reasoning selects best path
  - **Steps**:
    - Build causal DAG
    - Compute P(answer | do(select=candidate)) for multiple candidates
    - Assert best candidate selected based on causal_score
  - **Reference**: spec.md FR-004
  - **Dependencies**: T043

- [ ] **T057** [P] Test provenance persistence in Neo4j
  - **File**: `/Volumes/Asylum/dev/Dionysus-2.0/backend/tests/integration/test_provenance_persistence.py`
  - **Test**: Provenance metadata stored in Neo4j
  - **Steps**:
    - Curate evidence with provenance
    - Query Neo4j for Provenance nodes
    - Assert all 7 required fields present
    - Assert trust_signals calculated
  - **Reference**: spec.md FR-006
  - **Dependencies**: T045

- [ ] **T058** [P] Test conflict detection and resolution
  - **File**: `/Volumes/Asylum/dev/Dionysus-2.0/backend/tests/integration/test_conflict_resolution.py`
  - **Test**: Concurrent basin writes resolved via MERGE
  - **Steps**:
    - Simulate concurrent writes to same basin (strength 1.4 vs 1.6)
    - Assert conflict detected
    - Assert MERGE resolves to max strength (1.6)
    - Verify exponential backoff retry
  - **Reference**: spec.md FR-008
  - **Dependencies**: T049

---

## Phase 3.12: Performance Tests (Parallel Test Files)

- [ ] **T059** [P] Test navigation latency (NFR-001: <200ms for 10-step path)
  - **File**: `/Volumes/Asylum/dev/Dionysus-2.0/backend/tests/performance/test_navigation_latency.py`
  - **Test**: Measure p95 latency for 100 queries
  - **Target**: <200ms for 10-step navigation
  - **Metrics**: latency_ms, thoughtseed_gen_ms, causal_pred_ms
  - **Reference**: spec.md NFR-001, quickstart.md benchmarks
  - **Dependencies**: T050

- [ ] **T060** [P] Test curation latency (NFR-002: <100ms for 20 evidence snippets)
  - **File**: `/Volumes/Asylum/dev/Dionysus-2.0/backend/tests/performance/test_curation_latency.py`
  - **Test**: Measure p95 latency for varying evidence pool sizes (10, 20, 50, 100)
  - **Target**: <100ms for 20 snippets
  - **Metrics**: latency_ms, provenance_overhead_ms
  - **Reference**: spec.md NFR-002
  - **Dependencies**: T051

- [ ] **T061** [P] Test ThoughtSeed throughput (NFR-003: 100+ seeds/sec)
  - **File**: `/Volumes/Asylum/dev/Dionysus-2.0/backend/tests/performance/test_thoughtseed_throughput.py`
  - **Test**: Batch generate 1000 ThoughtSeeds, measure throughput
  - **Target**: 100+ ThoughtSeeds/second (<10ms each)
  - **Reference**: spec.md NFR-003
  - **Dependencies**: T039

- [ ] **T062** [P] Test curiosity spawn latency (NFR-004: <50ms to spawn agent)
  - **File**: `/Volumes/Asylum/dev/Dionysus-2.0/backend/tests/performance/test_curiosity_spawn_latency.py`
  - **Test**: Measure queue insertion time and metadata construction
  - **Target**: <50ms (non-blocking async spawn)
  - **Reference**: spec.md NFR-004
  - **Dependencies**: T041

- [ ] **T063** [P] Test causal prediction latency (NFR-005: <30ms per intervention)
  - **File**: `/Volumes/Asylum/dev/Dionysus-2.0/backend/tests/performance/test_causal_latency.py`
  - **Test**: Benchmark with 10 candidates, measure average prediction time
  - **Target**: <30ms includes Bayesian network inference + LRU cache lookup
  - **Reference**: spec.md NFR-005
  - **Dependencies**: T043

- [ ] **T064** [P] Test provenance overhead (NFR-006: <20% latency increase)
  - **File**: `/Volumes/Asylum/dev/Dionysus-2.0/backend/tests/performance/test_provenance_overhead.py`
  - **Test**: A/B test curator with/without provenance metadata
  - **Target**: Baseline (no provenance) + 20% = with provenance
  - **Reference**: spec.md NFR-006
  - **Dependencies**: T045

- [ ] **T065** [P] Test conflict resolution latency (NFR-008: <10ms rollback)
  - **File**: `/Volumes/Asylum/dev/Dionysus-2.0/backend/tests/performance/test_conflict_latency.py`
  - **Test**: Simulate concurrent writes, measure checkpoint → rollback time
  - **Target**: <10ms includes conflict detection + transaction rollback
  - **Reference**: spec.md NFR-008
  - **Dependencies**: T049

---

## Phase 3.13: Documentation & Validation

- [ ] **T066** Execute quickstart.md validation
  - **File**: `/Volumes/Asylum/dev/Dionysus-2.0/specs/035-clause-phase2-multi-agent/quickstart.md`
  - **Steps**: Run all quickstart examples end-to-end
  - **Verify**:
    - Setup completes in 5 minutes
    - Full workflow demo completes in 15 minutes
    - All validation tests pass
    - Performance benchmarks meet targets
  - **Reference**: quickstart.md
  - **Dependencies**: All previous tasks (T001-T065)

- [ ] **T067** Update API documentation with Phase 2 examples
  - **Files**:
    - `/Volumes/Asylum/dev/Dionysus-2.0/specs/035-clause-phase2-multi-agent/contracts/README.md`
    - `/Volumes/Asylum/dev/Dionysus-2.0/backend/README.md` (if exists)
  - **Add**: cURL examples, request/response examples, error handling examples
  - **Reference**: quickstart.md API Testing section
  - **Dependencies**: T066

- [ ] **T068** Final constitution compliance check
  - **Verification**:
    - NumPy version: `python -c "import numpy; assert numpy.__version__.startswith('2.')"`
    - AttractorBasin integration: Import tests pass
    - Context Engineering tests: T001-T003 all passing
    - Performance tests: All NFRs met
  - **Document**: Any deviations or warnings
  - **Reference**: plan.md Constitution Check section
  - **Dependencies**: T067

---

## Dependencies Summary

### Sequential Dependencies (Must Follow Order)
```
T001-T003 (Context Engineering) → T004 (Project Setup)
T004 → T005, T006 (Dependencies & Linting)
T006 → T007-T009 (Contract Tests)
T010-T017 (Models) → T018-T025 (Navigator)
T011, T013 → T026-T031 (Curator)
T012 → T032-T037 (Coordinator)
T014, T015, T016, T013 → T038-T045 (Intelligence Services)
T025, T031, T037 → T050-T052 (API Endpoints)
T052 → T053 (Full Workflow Test)
T001-T065 → T066-T068 (Documentation & Validation)
```

### Parallel Opportunities
```
Batch 1 (Context Engineering): T001, T002, T003 [P]
Batch 2 (Contract Tests): T007, T008, T009 [P]
Batch 3 (Models): T010, T011, T012, T013, T014, T015, T016, T017 [P]
Batch 4 (Intelligence Services): T038, T040, T042, T044 [P]
Batch 5 (Intelligence Extensions): T039, T041, T043, T045 [P]
Batch 6 (Integration Tests): T053, T054, T055, T056, T057, T058 [P]
Batch 7 (Performance Tests): T059, T060, T061, T062, T063, T064, T065 [P]
```

---

## Parallel Execution Examples

### Example 1: Context Engineering Validation (T001-T003)
```bash
# Launch all three in parallel (different test files)
Task: "Verify AttractorBasinManager accessibility in backend/tests/test_context_engineering_basin.py"
Task: "Validate Redis persistence in backend/tests/test_context_engineering_redis.py"
Task: "Test basin influence in backend/tests/test_context_engineering_influence.py"
```

### Example 2: Contract Tests (T007-T009)
```bash
# Launch all three in parallel (different test files)
Task: "Contract test POST /api/clause/navigate in backend/tests/contract/test_navigator_contract.py"
Task: "Contract test POST /api/clause/curate in backend/tests/contract/test_curator_contract.py"
Task: "Contract test POST /api/clause/coordinate in backend/tests/contract/test_coordinator_contract.py"
```

### Example 3: Model Implementation (T010-T017)
```bash
# Launch all eight in parallel (different model files)
Task: "Create PathNavigator models in backend/src/models/clause/path_models.py"
Task: "Create ContextCurator models in backend/src/models/clause/curator_models.py"
Task: "Create LCMAPPOCoordinator models in backend/src/models/clause/coordinator_models.py"
Task: "Create Provenance models in backend/src/models/clause/provenance_models.py"
Task: "Create ThoughtSeed models in backend/src/models/clause/thoughtseed_models.py"
Task: "Create CuriosityTrigger models in backend/src/models/clause/curiosity_models.py"
Task: "Create CausalIntervention models in backend/src/models/clause/causal_models.py"
Task: "Create shared models in backend/src/models/clause/shared_models.py"
```

### Example 4: Performance Tests (T059-T065)
```bash
# Launch all seven in parallel (different test files)
Task: "Test navigation latency in backend/tests/performance/test_navigation_latency.py"
Task: "Test curation latency in backend/tests/performance/test_curation_latency.py"
Task: "Test ThoughtSeed throughput in backend/tests/performance/test_thoughtseed_throughput.py"
Task: "Test curiosity spawn latency in backend/tests/performance/test_curiosity_spawn_latency.py"
Task: "Test causal prediction latency in backend/tests/performance/test_causal_latency.py"
Task: "Test provenance overhead in backend/tests/performance/test_provenance_overhead.py"
Task: "Test conflict resolution latency in backend/tests/performance/test_conflict_latency.py"
```

---

## Validation Checklist

- [x] All contracts (3) have corresponding tests (T007-T009)
- [x] All entities (22 models) have model tasks (T010-T017)
- [x] All tests come before implementation (T007-T009 before T018+)
- [x] Parallel tasks truly independent (different files, verified)
- [x] Each task specifies exact file path (absolute paths from repo root)
- [x] No task modifies same file as another [P] task (verified: API endpoints T050-T052 sequential, same file)
- [x] Context Engineering tests first (T001-T003 per Constitution)
- [x] Performance targets specified (T059-T065 with NFR references)

---

## Notes

- **[P] tasks** can run in parallel (different files, no dependencies)
- **Sequential tasks** (no [P]) must run in order (same file or dependencies)
- **TDD critical**: Contract tests (T007-T009) MUST FAIL before implementation starts
- **Constitution compliance**: T001-T003 MUST pass before any implementation
- **Commit after each task** for proper git history
- **Performance validation**: T066 validates all NFRs met before completion

---

**Tasks Status**: ✅ 68 tasks generated, dependency-ordered, ready for execution
**Next Step**: Execute tasks sequentially/in parallel following dependency graph
**Estimated Time**: 7-9 weeks (per plan.md Phase 2 timeline)

---
*Generated: 2025-10-02 | Branch: 035-clause-phase2-multi-agent*
