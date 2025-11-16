# CLAUSE Multi-Agent System

## Overview

The CLAUSE Multi-Agent System provides consciousness-enhanced knowledge graph reasoning through coordinated agent collaboration. It implements budget-aware subgraph construction, path exploration, and evidence curation using reinforcement learning coordination (LC-MAPPO). The system processes documents to build intelligent knowledge graphs that strengthen over time through basin frequency tracking and cross-document linking.

This capability consists of three specialized agents working in concert:
- **Subgraph Architect**: Constructs compact, query-relevant subgraphs using budget-aware edge selection
- **Path Navigator**: Explores knowledge graph paths with curiosity-driven expansion and causal reasoning
- **Context Curator**: Selects evidence with provenance tracking and redundancy detection

## Purpose

Enable consciousness-enhanced knowledge graph reasoning through coordinated multi-agent collaboration, providing budget-aware subgraph construction, path exploration, and evidence curation that strengthens over time through reinforcement learning and basin frequency tracking.

## Requirements

### Requirement: Subgraph Construction with Budget Control

The system SHALL construct compact, query-specific subgraphs from the knowledge graph using budget-aware edge selection. Edge selection uses a 5-signal scoring system (entity match, relation match, neighborhood, degree, basin strength) combined with shaped gain optimization to enforce strict edge budgets while maximizing query relevance.

#### Scenario: Budget-Enforced Edge Selection
**Given** a research paper with 50 unique concepts is uploaded
**When** the Subgraph Architect processes it with edge_budget=50
**Then** exactly 50 edges are selected using shaped gain rule (score - λ_edge × cost > 0), never exceeding the budget

#### Scenario: Basin Strength Prioritization
**Given** a concept "neural architecture search" has appeared in 5 previous documents
**When** basin strength reaches 2.0 (1.0 base + 5×0.2 increments)
**Then** edges involving this concept receive higher scores in future selections, creating a self-improving system

#### Scenario: Co-occurrence Context Tracking
**Given** two concepts co-occur in 3 documents
**When** neighborhood scoring evaluates candidate edges
**Then** their co-occurrence count of 3 influences edge relevance, strengthening semantic relationships

### Requirement: Basin Frequency Strengthening

Attractor basins SHALL strengthen over time as concepts reappear across documents. Each concept appearance increases basin strength by +0.2 (capped at 2.0), making frequently-used concepts more influential in future edge selection. This creates an adaptive knowledge graph that prioritizes well-established concepts.

#### Scenario: Basin Strength Increment
**Given** a concept has base basin strength of 1.0
**When** the concept appears in a new document
**Then** basin strength increases by exactly +0.2 to 1.2, and activation_history records the timestamp

#### Scenario: Basin Strength Cap
**Given** a concept has appeared in 10+ documents (strength would exceed 2.0)
**When** basin strengthening calculates new strength
**Then** strength is capped at exactly 2.0 to prevent unbounded growth

#### Scenario: New Concept Initialization
**Given** a completely new concept with no prior basin
**When** the concept is first encountered
**Then** basin strength initializes to 0.0, and edge scoring uses the remaining 4 signals only

### Requirement: Path Navigation with Step Budget

The Path Navigator SHALL explore knowledge graph paths using a budget-aware termination head. At each step, it encodes state, checks termination conditions, scores candidate next hops, and selects an action (CONTINUE, BACKTRACK, STOP). Navigation terminates when step budget is exhausted or shaped gain becomes non-positive.

#### Scenario: Step Budget Enforcement
**Given** path navigation starts with step_budget=10
**When** the navigator explores the graph
**Then** path contains at most 10 steps, and final_action is STOP or BACKTRACK when budget exhausted

#### Scenario: Shaped Gain Termination
**Given** path navigation is in progress
**When** shaped_gain (score - λ_step × step_cost) becomes ≤ 0 for all candidates
**Then** navigator executes STOP action immediately, even if step budget remains

#### Scenario: Backtracking on Dead Ends
**Given** path navigation reaches a node with no promising candidates
**When** all candidate scores are below threshold
**Then** navigator executes BACKTRACK action to return to previous promising node

### Requirement: ThoughtSeed Cross-Document Linking

During path exploration, the system SHALL generate ThoughtSeeds for each candidate next hop. ThoughtSeeds contain concept, source document, basin context, and similarity threshold. When ThoughtSeeds from different documents have similarity > 0.8, they are linked, enabling cross-document knowledge integration.

#### Scenario: ThoughtSeed Generation During Navigation
**Given** path navigator evaluates 5 candidate next hops
**When** navigation step completes
**Then** 5 ThoughtSeeds are generated, each with concept, source_doc, basin_context, and similarity_threshold fields

#### Scenario: Cross-Document ThoughtSeed Linking
**Given** two documents generate ThoughtSeeds for similar concepts
**When** similarity score exceeds 0.8
**Then** ThoughtSeeds are linked via SIMILAR_TO relationship in Neo4j, with similarity score stored

#### Scenario: ThoughtSeed Provenance Tracking
**Given** path navigation generates 12 ThoughtSeeds
**When** path metadata is returned
**Then** metadata includes thoughtseeds_generated=12 and list of ThoughtSeed IDs for full provenance

### Requirement: Curiosity-Driven Exploration

The system SHALL spawn curiosity agents when prediction error exceeds threshold (default 0.7), the navigator spawns background curiosity agents. Prediction error is calculated as |expected_score - actual_score| for candidate hops. High prediction error indicates unexpected patterns worth investigating.

#### Scenario: Curiosity Trigger on High Prediction Error
**Given** path navigator expected score=0.9 but observed score=0.15
**When** prediction_error (|0.9 - 0.15| = 0.75) exceeds curiosity_threshold=0.7
**Then** curiosity agent is spawned with metadata (trigger_type, concept, error_magnitude, timestamp)

#### Scenario: Non-Blocking Curiosity Spawn
**Given** curiosity trigger is activated during navigation
**When** curiosity agent is added to background queue
**Then** navigation continues immediately without blocking, and curiosity_triggers_spawned count increments

#### Scenario: Curiosity Queue Processing
**Given** 3 curiosity triggers were spawned during navigation
**When** background workers process the curiosity queue
**Then** each trigger executes exploration asynchronously, and results update knowledge graph separately

### Requirement: Causal Path Selection

The Path Navigator SHALL use causal intervention prediction for selecting next hops. For each candidate, it estimates P(answer | do(select_path=candidate)) using do-calculus. If computation exceeds 30ms timeout, the system queues it for background processing and uses semantic similarity heuristic immediately to avoid blocking.

#### Scenario: Causal Score Influences Path Selection
**Given** three candidate next hops with causal scores [0.85, 0.62, 0.45]
**When** path selection occurs
**Then** candidate with causal_score=0.85 is selected (highest causal impact on answer)

#### Scenario: Causal Computation Timeout Fallback
**Given** causal inference for a candidate exceeds 30ms timeout
**When** timeout is detected
**Then** computation is queued for background processing, semantic similarity heuristic is used immediately, and fallback_used flag is set in metadata

#### Scenario: Background Causal Result Integration
**Given** causal computation was queued in step 1 and completes during step 3
**When** step 4 evaluates candidates
**Then** cached causal result is used for scoring if available, improving future hop selection

### Requirement: Evidence Curation with Token Budget

The Context Curator SHALL perform listwise evidence selection with token budget enforcement. It scores evidence snippets using anti-redundancy and query relevance signals, then applies shaped utility rule (score - λ_tok × snippet_tokens). Selection stops when shaped utility ≤ 0 (learned stop) or token budget exhausted.

#### Scenario: Token Budget Enforcement
**Given** curator starts with token_budget=1000
**When** evidence selection completes
**Then** total tokens used ≤ 1000, and no snippet selection violates the budget

#### Scenario: Learned Stop Activation
**Given** curator is evaluating evidence snippets
**When** shaped_utility (score - λ_tok × tokens) becomes ≤ 0 for next candidate
**Then** curator stops immediately, and learned_stop_triggered flag is set to true

#### Scenario: Anti-Redundancy Scoring
**Given** two evidence snippets with 85% semantic overlap
**When** listwise scoring evaluates the second snippet
**Then** redundancy penalty reduces its score, preventing duplicate information selection

### Requirement: Provenance Tracking

Each curated evidence snippet SHALL include full provenance metadata with 7 required fields: source_uri, extraction_timestamp, extractor_identity, supporting_evidence, verification_status, corroboration_count, and trust_signals (reputation, recency, semantic_consistency). Provenance enables evidence verification and trust assessment.

#### Scenario: Provenance Metadata Completeness
**Given** curator selects 3 evidence snippets
**When** response is returned
**Then** each snippet has provenance dict with all 7 required fields populated

#### Scenario: Trust Signal Calculation
**Given** an evidence snippet from a high-reputation source (0.95) published recently (0.88) with semantic consistency (0.91)
**When** provenance is generated
**Then** trust_signals contains reputation_score=0.95, recency_score=0.88, semantic_consistency=0.91

#### Scenario: Corroboration Tracking
**Given** evidence snippet appears in 5 different documents
**When** provenance metadata is created
**Then** corroboration_count=5, indicating multi-source verification

### Requirement: Multi-Agent Coordination via LC-MAPPO

The LC-MAPPO coordinator SHALL manage three agents using centralized critic with 4 value heads (task_value, edge_cost, latency_cost, token_cost) and three dual variables (λ_edge, λ_lat, λ_tok). Shaped return calculation penalizes budget violations: r'_t = r_acc - Σ λ_k × c_k. Dual variables update based on expected vs. actual budget usage.

#### Scenario: Shaped Return Calculation
**Given** episode with accuracy_reward=0.85 and costs {edge: 45, latency: 230ms, tokens: 1100}
**When** coordinator calculates shaped returns with lambdas {edge: 0.01, latency: 0.01, token: 0.01}
**Then** shaped_return = 0.85 - (0.01×45 + 0.01×230 + 0.01×1100) = 0.85 - 13.75 = -12.90

#### Scenario: Dual Variable Update
**Given** expected edge cost E[C_edge]=50 with budget β_edge=50, but actual cost=45
**When** dual update occurs with learning rate η=0.001
**Then** λ_edge updates: λ_edge ← max(0, λ_edge + 0.001 × (45 - 50)) = λ_edge - 0.005

#### Scenario: Agent Handoff Sequencing
**Given** query "What causes climate change?"
**When** coordinator processes the query
**Then** agent handoffs execute in sequence: SubgraphArchitect → PathNavigator → ContextCurator, with each agent receiving output from previous

### Requirement: Write Conflict Resolution

The system SHALL detect and resolve conflicts when multiple agents write to the same Neo4j node simultaneously via transaction checkpoints. Conflict resolution strategies include MERGE (take max basin strength), ROLLBACK (discard conflicting write), and RETRY (exponential backoff: 100ms, 200ms, 400ms).

#### Scenario: Conflict Detection on Concurrent Writes
**Given** navigator and curator both update basin strength for node "A" simultaneously
**When** transaction conflict is detected
**Then** conflict metadata is logged (agents, node, attempted strengths) and resolution strategy activates

#### Scenario: MERGE Conflict Resolution
**Given** navigator writes basin_strength=1.4 and curator writes basin_strength=1.6 to same node
**When** MERGE strategy resolves the conflict
**Then** final basin_strength=1.6 (maximum of conflicting values)

#### Scenario: Retry with Exponential Backoff
**Given** write conflict occurs on first attempt
**When** RETRY strategy is selected
**Then** system retries with delays [100ms, 200ms, 400ms], abandoning after 3 failures with error

#### Scenario: Conflict Rate Monitoring
**Given** system is processing documents with multiple agents
**When** conflicts occur
**Then** conflict rate is continuously monitored and logged with metadata for threshold analysis

### Requirement: Performance Guarantees

The system SHALL enforce strict performance SLAs across all operations to ensure production viability at scale.

#### Scenario: Navigation Latency SLA
**Given** 100 navigation requests with step_budget=10
**When** p95 latency is measured
**Then** p95 latency < 200ms for 10-step path navigation

#### Scenario: Curation Latency SLA
**Given** 100 curation requests with 20 evidence snippets each
**When** p95 latency is measured
**Then** p95 latency < 100ms for 20 evidence snippets

#### Scenario: ThoughtSeed Generation Throughput
**Given** batch generation of 1000 ThoughtSeeds
**When** throughput is measured
**Then** system generates 100+ ThoughtSeeds per second (< 10ms per seed)

#### Scenario: Curiosity Spawn Non-Blocking
**Given** path navigator triggers curiosity agent
**When** spawn latency is measured
**Then** spawn completes in < 50ms and does not block navigation

#### Scenario: Causal Prediction Performance
**Given** 10 candidate interventions require causal prediction
**When** average prediction time is measured
**Then** each prediction completes in < 30ms via Bayesian network inference

#### Scenario: Provenance Overhead Limit
**Given** curator runs with and without provenance tracking
**When** latency overhead is calculated
**Then** provenance tracking adds < 20% latency vs. baseline

#### Scenario: Budget Compliance Enforcement
**Given** 1000 random queries with varying budgets (edge, step, token)
**When** fuzz testing is performed
**Then** 100% of requests enforce budgets with zero violations

#### Scenario: Conflict Resolution Speed
**Given** concurrent writes trigger conflict detection
**When** conflict resolution executes
**Then** detect + rollback completes in < 10ms
