# Feature Specification: CLAUSE Phase 2 - Path Navigator & Context Curator

**Spec ID**: 035
**Feature Name**: CLAUSE Phase 2 Multi-Agent Coordination
**Status**: Draft
**Created**: 2025-10-02
**Dependencies**: Spec 034 (CLAUSE Phase 1 - Subgraph Architect)

## Overview

Implement CLAUSE Path Navigator and Context Curator agents to complete the three-agent CLAUSE architecture. The Navigator performs budget-aware path exploration with ThoughtSeed generation, curiosity triggers, and causal reasoning. The Curator performs listwise evidence selection with token budget enforcement and full provenance tracking. Both agents integrate with the existing Subgraph Architect from Phase 1 and coordinate via LC-MAPPO training.

## Motivation

Phase 1 delivered basin-aware subgraph construction. Phase 2 adds:
1. **Path Navigation**: Budget-aware traversal with continue/backtrack/stop decisions
2. **Evidence Curation**: Listwise selection with redundancy detection and provenance
3. **Multi-Agent Coordination**: LC-MAPPO training with conflict resolution
4. **Intelligence Features**: ThoughtSeeds (028), Curiosity (029), Causal reasoning (033), Provenance (032)

This completes the CLAUSE foundation for production knowledge graph reasoning with deployment budget controls.

## Goals

### Primary Goals
- Implement Path Navigator with step budget (β_step=10) and termination head
- Implement Context Curator with token budget (β_tok=2048) and learned stop
- Integrate ThoughtSeed generation during path exploration (Spec 028)
- Integrate curiosity triggers for high prediction error paths (Spec 029)
- Integrate causal intervention prediction for path selection (Spec 033)
- Implement full provenance tracking for curated evidence (Spec 032)
- Implement LC-MAPPO trainer with centralized critic and dual variable updates
- Implement Neo4j write conflict resolution (Spec 031)

### Success Metrics
- **Navigation Latency**: <200ms for 10-step path (p95)
- **Curation Latency**: <100ms for 20 evidence snippets (p95)
- **ThoughtSeed Generation**: 100+ seeds/sec
- **Curiosity Trigger Latency**: <50ms background spawn
- **Causal Prediction**: <30ms per candidate intervention
- **Provenance Overhead**: <20% latency increase
- **Budget Compliance**: 100% enforcement (never exceed β_step or β_tok)
- **Conflict Resolution**: <10ms rollback on detection

## Non-Goals

- **LC-MAPPO Training Loop**: Full RL training deferred to Phase 3 (implement trainer interface only)
- **Production Model Weights**: Use random policy for Phase 2 (training in Phase 3)
- **Frontend Visualization**: Visual interface deferred to Spec 030 (Phase 4)
- **APOC Optimization**: Advanced Neo4j procedures deferred to Phase 3

## Functional Requirements

### FR-001: Path Navigator Core
**Priority**: P0 (Critical)
**Description**: Implement budget-aware path navigation with termination head

**Acceptance Criteria**:
- Navigator accepts (query, start_node, graph, step_budget) input
- At each step: encode state → termination check → candidate scoring → action selection
- Actions: CONTINUE (select next hop), BACKTRACK (return to previous node), STOP (terminate)
- Step budget strictly enforced (stop when budget_used >= β_step)
- Return complete path with metadata (nodes, edges, actions, budget_used)

**Test Cases**:
```python
def test_navigator_budget_enforcement():
    navigator = CLAUSEPathNavigator(step_budget=5)
    path = navigator.navigate(query="...", start_node="A", graph=kg)
    assert len(path.steps) <= 5  # Never exceeds budget
    assert path.budget_used == len(path.steps)
    assert path.final_action in ["STOP", "BACKTRACK"]
```

### FR-002: ThoughtSeed Integration
**Priority**: P0 (Critical)
**Description**: Generate ThoughtSeeds during path exploration for cross-document linking (Spec 028)

**Acceptance Criteria**:
- For each candidate next hop, generate ThoughtSeed with basin context
- ThoughtSeed contains: concept, source_doc (query), basin_context, similarity_threshold
- Link ThoughtSeeds across documents when similarity > 0.8
- Store ThoughtSeed IDs in path metadata for provenance

**Test Cases**:
```python
def test_thoughtseed_generation_during_navigation():
    navigator = CLAUSEPathNavigator()
    path = navigator.navigate(query="...", start_node="A", graph=kg)
    assert len(path.thoughtseeds_generated) > 0
    for ts in path.thoughtseeds_generated:
        assert ts.concept in path.nodes
        assert ts.basin_context is not None
```

### FR-003: Curiosity Trigger Integration
**Priority**: P1 (High)
**Description**: Spawn curiosity agents when prediction error exceeds threshold (Spec 029)

**Acceptance Criteria**:
- Calculate prediction error: |expected_score - actual_score|
- If prediction_error > 0.7: add to curiosity_queue with metadata
- Curiosity metadata: trigger_type, concept, error_magnitude, timestamp
- Background queue processing (non-blocking)
- Return curiosity_triggers_spawned count in path metadata

**Test Cases**:
```python
def test_curiosity_trigger_on_high_prediction_error():
    navigator = CLAUSEPathNavigator(curiosity_threshold=0.7)
    path = navigator.navigate(query="...", start_node="A", graph=kg)
    # Simulate high prediction error
    assert path.curiosity_triggers_spawned >= 1
    assert curiosity_queue.size() > 0
```

### FR-004: Causal Path Selection
**Priority**: P1 (High)
**Description**: Use causal intervention prediction for path selection (Spec 033)

**Acceptance Criteria**:
- For each candidate next hop: estimate P(answer | do(select_path=candidate))
- Causal reasoner implements do-calculus intervention
- Select candidate with highest causal_score - λ_step × step_cost
- Store causal_scores in path metadata for analysis

**Test Cases**:
```python
def test_causal_path_selection():
    navigator = CLAUSEPathNavigator()
    path = navigator.navigate(query="...", start_node="A", graph=kg)
    assert "causal_scores" in path.metadata
    # Verify selection uses causal reasoning
    selected_hops = [step.to_node for step in path.steps]
    causal_scores = path.metadata["causal_scores"]
    # Selected hop should have high causal score
    assert all(causal_scores[hop] > 0.5 for hop in selected_hops)
```

### FR-005: Context Curator Core
**Priority**: P0 (Critical)
**Description**: Implement listwise evidence selection with token budget and learned stop

**Acceptance Criteria**:
- Curator accepts (evidence_pool, token_budget) input
- Score evidence listwise (anti-redundancy, query relevance)
- For each candidate: shaped_utility = score - λ_tok × snippet_tokens
- Select if shaped_utility > 0 AND total_tokens + snippet_tokens <= β_tok
- Stop when shaped_utility ≤ 0 (learned stop)
- Return selected_evidence list with total_tokens_used

**Test Cases**:
```python
def test_curator_token_budget_enforcement():
    curator = CLAUSEContextCurator(token_budget=1000)
    selected = curator.curate(evidence_pool=[...])
    total_tokens = sum(curator.count_tokens(e) for e in selected)
    assert total_tokens <= 1000  # Never exceeds budget
    assert curator.learned_stop_triggered is True
```

### FR-006: Provenance Tracking
**Priority**: P0 (Critical)
**Description**: Add full provenance metadata to each curated evidence snippet (Spec 032)

**Acceptance Criteria**:
- Each selected snippet has provenance dict with 7 required fields:
  - source_uri, extraction_timestamp, extractor_identity
  - supporting_evidence (snippet), verification_status, corroboration_count
  - trust_signals (reputation, recency, semantic_consistency)
- Provenance metadata stored in Neo4j with evidence nodes
- API returns evidence + provenance in structured format

**Test Cases**:
```python
def test_provenance_metadata_completeness():
    curator = CLAUSEContextCurator()
    selected = curator.curate(evidence_pool=[...])
    for evidence in selected:
        assert "provenance" in evidence
        prov = evidence["provenance"]
        assert all(k in prov for k in [
            "source_uri", "extraction_timestamp", "extractor_identity",
            "supporting_evidence", "verification_status", "corroboration_count"
        ])
```

### FR-007: LC-MAPPO Coordinator
**Priority**: P1 (High)
**Description**: Implement multi-agent coordinator with centralized critic and dual updates

**Acceptance Criteria**:
- Centralized critic with 4 heads: task_value, edge_cost, latency_cost, token_cost
- Three dual variables: λ_edge, λ_lat, λ_tok (initialized to 0.01)
- Shaped return calculation: r'_t = r_acc - Σ λ_k × c_k
- Dual update rule: λ_k ← max(0, λ_k + η × (E[C_k] - β_k))
- Conflict resolver integration (Neo4j transaction management)

**Test Cases**:
```python
def test_lcmappo_shaped_return_calculation():
    coordinator = LCMAPPOCoordinator()
    episode = generate_test_episode()
    shaped_rewards = coordinator.calculate_shaped_returns(episode)
    # Verify shaped return formula
    for t, r_shaped in enumerate(shaped_rewards):
        r_acc = episode.rewards[t]
        costs = episode.costs[t]
        expected = r_acc - sum(
            coordinator.lambdas[k] * costs[k]
            for k in ["edge", "latency", "token"]
        )
        assert abs(r_shaped - expected) < 1e-6
```

### FR-008: Write Conflict Resolution
**Priority**: P1 (High)
**Description**: Detect and resolve Neo4j write conflicts during multi-agent operations (Spec 031)

**Acceptance Criteria**:
- Checkpoint before each agent write operation
- Detect conflicts: multiple agents writing to same node simultaneously
- Conflict resolution strategies: MERGE (take max basin strength), ROLLBACK, RETRY
- Exponential backoff on retry (100ms, 200ms, 400ms)
- Log conflict events with metadata (agents, nodes, resolution)

**Test Cases**:
```python
def test_write_conflict_detection_and_resolution():
    coord = LCMAPPOCoordinator()
    # Simulate concurrent writes to same basin
    conflict = coord.detect_conflicts([
        {"agent": "navigator", "node": "A", "strength": 1.4},
        {"agent": "curator", "node": "A", "strength": 1.6}
    ])
    assert conflict.detected is True
    resolved = coord.resolve_conflict(conflict, strategy="MERGE")
    assert resolved.final_strength == 1.6  # Max of 1.4, 1.6
```

## Non-Functional Requirements

### NFR-001: Navigation Performance
- **Metric**: p95 latency < 200ms for 10-step path navigation
- **Validation**: Performance test with 100 queries, measure 95th percentile
- **Target**: <200ms includes state encoding, candidate scoring, action selection

### NFR-002: Curation Performance
- **Metric**: p95 latency < 100ms for 20 evidence snippets
- **Validation**: Benchmark with varying evidence pool sizes (10, 20, 50, 100)
- **Target**: <100ms includes listwise scoring and learned stop

### NFR-003: ThoughtSeed Throughput
- **Metric**: 100+ ThoughtSeeds generated per second
- **Validation**: Batch generation test with 1000 concepts
- **Target**: <10ms per ThoughtSeed (generation + basin context fetch)

### NFR-004: Curiosity Spawn Latency
- **Metric**: <50ms to spawn background curiosity agent
- **Validation**: Measure queue insertion time and metadata construction
- **Target**: Non-blocking (async spawn)

### NFR-005: Causal Prediction Latency
- **Metric**: <30ms per candidate intervention prediction
- **Validation**: Benchmark with 10 candidates, measure average prediction time
- **Target**: <30ms includes Bayesian network inference

### NFR-006: Provenance Overhead
- **Metric**: <20% latency increase with provenance tracking vs. without
- **Validation**: A/B test curator with/without provenance metadata
- **Target**: Baseline (no provenance) + 20% = with provenance

### NFR-007: Budget Compliance
- **Metric**: 100% enforcement (zero budget violations)
- **Validation**: Fuzz testing with random queries and varying budgets
- **Target**: Never exceed β_step or β_tok in any test case

### NFR-008: Conflict Resolution Latency
- **Metric**: <10ms to detect and rollback on conflict
- **Validation**: Simulate concurrent writes, measure checkpoint → rollback time
- **Target**: <10ms includes conflict detection + transaction rollback

## API Design

### Path Navigator API

**Endpoint**: `POST /api/clause/navigate`

**Request**:
```json
{
  "query": "What causes climate change?",
  "start_node": "climate_change",
  "step_budget": 10,
  "enable_thoughtseeds": true,
  "enable_curiosity": true,
  "enable_causal": true,
  "curiosity_threshold": 0.7
}
```

**Response**:
```json
{
  "path": {
    "nodes": ["climate_change", "greenhouse_gases", "CO2_emissions", "fossil_fuels"],
    "edges": [
      {"from": "climate_change", "relation": "caused_by", "to": "greenhouse_gases"},
      {"from": "greenhouse_gases", "relation": "includes", "to": "CO2_emissions"},
      {"from": "CO2_emissions", "relation": "from", "to": "fossil_fuels"}
    ],
    "steps": [
      {"step": 1, "from": "climate_change", "to": "greenhouse_gases", "action": "CONTINUE", "causal_score": 0.85},
      {"step": 2, "from": "greenhouse_gases", "to": "CO2_emissions", "action": "CONTINUE", "causal_score": 0.78},
      {"step": 3, "from": "CO2_emissions", "to": "fossil_fuels", "action": "STOP", "causal_score": 0.92}
    ]
  },
  "metadata": {
    "budget_used": 3,
    "budget_total": 10,
    "final_action": "STOP",
    "thoughtseeds_generated": 12,
    "curiosity_triggers_spawned": 2,
    "causal_predictions": 15
  },
  "performance": {
    "latency_ms": 145,
    "thoughtseed_gen_ms": 23,
    "causal_pred_ms": 87
  }
}
```

### Context Curator API

**Endpoint**: `POST /api/clause/curate`

**Request**:
```json
{
  "evidence_pool": [
    "Greenhouse gases trap heat in the atmosphere...",
    "CO2 is the primary greenhouse gas from human activity...",
    "Fossil fuel combustion releases CO2..."
  ],
  "token_budget": 2048,
  "enable_provenance": true,
  "lambda_tok": 0.01
}
```

**Response**:
```json
{
  "selected_evidence": [
    {
      "text": "Greenhouse gases trap heat in the atmosphere...",
      "tokens": 156,
      "score": 0.92,
      "shaped_utility": 0.904,
      "provenance": {
        "source_uri": "neo4j://concept/greenhouse_gases",
        "extraction_timestamp": "2025-10-02T10:30:15Z",
        "extractor_identity": "ContextCurator-v2.0",
        "supporting_evidence": "Greenhouse gases trap heat...",
        "verification_status": "verified",
        "corroboration_count": 5,
        "trust_signals": {
          "reputation_score": 0.95,
          "recency_score": 0.88,
          "semantic_consistency": 0.91
        }
      }
    }
  ],
  "metadata": {
    "tokens_used": 428,
    "tokens_total": 2048,
    "learned_stop_triggered": true,
    "evidence_pool_size": 3,
    "selected_count": 1
  },
  "performance": {
    "latency_ms": 78,
    "provenance_overhead_ms": 12
  }
}
```

### LC-MAPPO Coordinator API

**Endpoint**: `POST /api/clause/coordinate`

**Request**:
```json
{
  "query": "What causes climate change?",
  "budgets": {
    "edge_budget": 50,
    "step_budget": 10,
    "token_budget": 2048
  },
  "lambdas": {
    "edge": 0.01,
    "latency": 0.01,
    "token": 0.01
  }
}
```

**Response**:
```json
{
  "result": {
    "subgraph": { "nodes": [...], "edges": [...] },
    "path": { "nodes": [...], "edges": [...] },
    "evidence": [ {...}, {...} ]
  },
  "agent_handoffs": [
    {"step": 1, "agent": "SubgraphArchitect", "action": "built_subgraph", "budget_used": {"edges": 35}},
    {"step": 2, "agent": "PathNavigator", "action": "explored_paths", "budget_used": {"steps": 7}},
    {"step": 3, "agent": "ContextCurator", "action": "selected_evidence", "budget_used": {"tokens": 1024}}
  ],
  "conflicts_detected": 0,
  "conflicts_resolved": 0,
  "performance": {
    "total_latency_ms": 542,
    "architect_ms": 287,
    "navigator_ms": 145,
    "curator_ms": 78,
    "coordination_overhead_ms": 32
  }
}
```

## Implementation Plan

### Phase 1: Path Navigator Foundation (Week 1)
- Implement CLAUSEPathNavigator class with state encoding
- Implement termination head (budget check)
- Implement action selection (CONTINUE, BACKTRACK, STOP)
- Contract tests for navigation API

### Phase 2: Navigator Intelligence (Week 2)
- Integrate ThoughtSeed generation (Spec 028)
- Integrate curiosity triggers (Spec 029)
- Integrate causal reasoning (Spec 033)
- Integration tests for intelligence features

### Phase 3: Context Curator Foundation (Week 3)
- Implement CLAUSEContextCurator class with listwise scoring
- Implement learned stop (shaped utility)
- Implement token budget enforcement
- Contract tests for curation API

### Phase 4: Curator Provenance (Week 4)
- Integrate provenance tracking (Spec 032)
- Implement trust signal calculation
- Neo4j provenance node storage
- Integration tests for provenance

### Phase 5: LC-MAPPO Coordination (Week 5)
- Implement LCMAPPOCoordinator class
- Implement centralized critic (4 heads)
- Implement dual variable updates
- Shaped return calculation

### Phase 6: Conflict Resolution (Week 6)
- Implement Neo4j transaction manager
- Implement conflict detection
- Implement MERGE/ROLLBACK/RETRY strategies
- Integration tests for conflict scenarios

### Phase 7: Integration & Testing (Week 7)
- End-to-end workflow tests (Architect → Navigator → Curator)
- Performance benchmarks (all NFRs)
- API documentation
- Quickstart guide

## Testing Strategy

### Contract Tests
- Navigator API contract (request/response schema validation)
- Curator API contract (request/response schema validation)
- Coordinator API contract (agent handoff structure)

### Integration Tests
- Full workflow: Architect → Navigator → Curator
- ThoughtSeed cross-document linking
- Curiosity agent spawning and execution
- Causal intervention predictions
- Provenance metadata persistence
- Conflict detection and resolution

### Performance Tests
- Navigation latency (NFR-001)
- Curation latency (NFR-002)
- ThoughtSeed throughput (NFR-003)
- Curiosity spawn latency (NFR-004)
- Causal prediction latency (NFR-005)
- Provenance overhead (NFR-006)
- Conflict resolution latency (NFR-008)

### Fuzz Tests
- Budget compliance (random queries, varying budgets)
- Conflict scenarios (concurrent agent writes)

## Dependencies

### Internal Dependencies
- **Spec 034**: CLAUSE Phase 1 (SubgraphArchitect, BasinTracker, EdgeScorer)
- **Spec 028**: ThoughtSeed Bulk Processing (generation and linking)
- **Spec 029**: Curiosity-Driven Background Agents (queue and spawning)
- **Spec 031**: Write Conflict Resolution (Neo4j transaction manager)
- **Spec 032**: Emergent Pattern Detection (provenance metadata)
- **Spec 033**: Causal Reasoning (Bayesian network, do-calculus)

### External Dependencies
- NetworkX 3.x (path finding, graph traversal)
- NumPy 2.0+ (vectorized operations)
- Neo4j 5.x driver (graph storage, transactions)
- Redis 7.x (caching, queue management)

## Risks and Mitigations

### Risk 1: Causal Inference Latency
- **Risk**: Bayesian network inference may exceed 30ms for complex graphs
- **Mitigation**: Pre-compute causal DAG structure, cache intervention predictions
- **Fallback**: Simple heuristic scoring if causal unavailable

### Risk 2: Conflict Resolution Overhead
- **Risk**: Frequent conflicts may degrade throughput
- **Mitigation**: Partition basin writes by concept hash (reduce collision probability)
- **Fallback**: Read-only mode for high-conflict scenarios

### Risk 3: ThoughtSeed Storage Growth
- **Risk**: Generating 100+ seeds/sec may overwhelm Neo4j write capacity
- **Mitigation**: Batch writes (100 seeds per transaction), async persistence
- **Fallback**: In-memory ThoughtSeed cache with periodic flush

### Risk 4: Token Budget Estimation
- **Risk**: Token counting may not match LLM tokenizer exactly
- **Mitigation**: Use tiktoken library (matches GPT tokenizer)
- **Fallback**: Conservative estimation (add 10% buffer)

## Success Criteria

### Must Have (Phase 2 Complete)
- ✅ Path Navigator with step budget enforcement
- ✅ Context Curator with token budget enforcement
- ✅ ThoughtSeed generation during navigation
- ✅ Curiosity triggers for high prediction error
- ✅ Causal reasoning for path selection
- ✅ Provenance tracking for curated evidence
- ✅ LC-MAPPO coordinator with shaped returns
- ✅ Conflict resolution with MERGE strategy
- ✅ All NFRs met (latency, throughput, budget compliance)
- ✅ 100% test coverage (contract, integration, performance)

### Nice to Have (Future Enhancements)
- LC-MAPPO training loop (full RL training)
- Advanced APOC procedures for Neo4j optimization
- Visual interface for agent handoffs (Spec 030)
- Multi-query batching for throughput

## References

- **CLAUSE Paper**: arXiv:2509.21035v1 [cs.AI] 25 Sep 2025
- **CLAUSE Integration Analysis**: `/Volumes/Asylum/dev/Dionysus-2.0/CLAUSE_INTEGRATION_ANALYSIS.md`
- **Phase 1 Completion**: `/Volumes/Asylum/dev/Dionysus-2.0/specs/034-clause-phase1-foundation/COMPLETION_REPORT.md`
- **Spec 028**: ThoughtSeed Bulk Processing
- **Spec 029**: Curiosity-Driven Background Agents
- **Spec 031**: Write Conflict Resolution
- **Spec 032**: Emergent Pattern Detection
- **Spec 033**: Causal Reasoning and Counterfactual
