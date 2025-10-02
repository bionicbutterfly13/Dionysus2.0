# Phase 0: Research - CLAUSE Phase 2 Multi-Agent

**Date**: 2025-10-02
**Feature**: CLAUSE Phase 2 - Path Navigator & Context Curator
**Research Status**: ✅ Complete - All technical decisions resolved

## Research Questions

###  1. Path Navigation Strategy
**Question**: How should we encode state for path navigation decisions?

**Decision**: Encode state as (query_embedding, current_node_features, local_neighborhood_structure)

**Rationale**:
- CLAUSE paper specifies state encoding must capture query relevance AND local topology
- Combines semantic similarity (query vs node) with structural information (neighborhood)
- Enables learned termination head to balance exploration vs budget conservation

**Alternatives Considered**:
- ❌ Simple embedding concatenation - loses neighborhood structure information
- ❌ Full graph encoding - too computationally expensive (exceeds <200ms budget)

**Implementation Approach**:
```python
def encode_state(query, current_node, graph):
    # Query embedding (384-dim from sentence-transformers)
    query_emb = self.embed(query)

    # Current node features
    node_emb = self.embed(current_node.text)
    node_degree = graph.degree(current_node)
    basin_strength = self.basin_tracker.get(current_node).strength

    # Local neighborhood structure (1-hop)
    neighbors = list(graph.neighbors(current_node))
    neighbor_embs = [self.embed(n.text) for n in neighbors]
    neighborhood_mean = np.mean(neighbor_embs, axis=0)

    # Concatenate features
    state = np.concatenate([
        query_emb,              # 384
        node_emb,               # 384
        [node_degree],          # 1
        [basin_strength],       # 1
        neighborhood_mean       # 384
    ])  # Total: 1154 features

    return state
```

**References**: CLAUSE paper Section 3.2 (State Encoding)

---

### 2. Termination Head Design
**Question**: How should the navigator decide when to stop exploration?

**Decision**: Binary classifier (stop_probability > 0.5) based on state encoding + budget remaining

**Rationale**:
- CLAUSE termination head learns optimal stopping point (not just hard budget cutoff)
- Balances task completion (reaching answer) vs budget conservation
- Simple sigmoid head over state features is fast (<10ms inference)

**Alternatives Considered**:
- ❌ Hard budget cutoff only - doesn't learn when exploration is productive
- ❌ Multi-class (CONTINUE/BACKTRACK/STOP) - adds complexity without clear benefit

**Implementation Approach**:
```python
def should_terminate(state, budget_remaining):
    # Augment state with budget signal
    features = np.concatenate([state, [budget_remaining / self.total_budget]])

    # Sigmoid classifier
    logit = self.termination_head(features)  # Linear layer
    stop_prob = 1 / (1 + np.exp(-logit))

    return stop_prob > 0.5
```

**References**: CLAUSE paper Section 3.2 (Termination Head)

---

### 3. ThoughtSeed Generation Integration
**Question**: When and how should ThoughtSeeds be generated during navigation?

**Decision**: Generate ThoughtSeed for each candidate next hop with basin context from BasinTracker

**Rationale**:
- Spec 028 requires bulk generation during exploration (not after)
- Basin context enables similarity matching for cross-document linking
- Incremental generation allows linking as exploration proceeds

**Alternatives Considered**:
- ❌ Generate only for selected hops - misses potential cross-document links
- ❌ Generate after navigation complete - loses incremental linking benefit

**Implementation Approach**:
```python
def select_next_hop(candidates):
    thoughtseeds = []

    for candidate in candidates:
        # Fetch basin context
        basin = self.basin_tracker.get(candidate)

        # Generate ThoughtSeed
        ts = self.thoughtseed_gen.create(
            concept=candidate,
            source_doc=self.query,
            basin_context={
                "strength": basin.strength,
                "activation_count": basin.activation_count,
                "co_occurring": basin.co_occurring_concepts
            },
            similarity_threshold=0.8
        )

        # Link across documents
        self.link_thoughtseed(ts)
        thoughtseeds.append(ts)

    # Select best candidate
    best = max(candidates, key=lambda c: self.score_hop(c))
    return best, thoughtseeds
```

**References**: Spec 028 (ThoughtSeed Bulk Processing)

---

### 4. Curiosity Trigger Mechanism
**Question**: How should high prediction error trigger curiosity agents?

**Decision**: Redis queue with prediction_error > threshold triggers background agent spawn

**Rationale**:
- Spec 029 requires non-blocking background investigation
- Redis provides async queue with persistence
- Prediction error = |expected_score - actual_score| measures surprise

**Alternatives Considered**:
- ❌ Synchronous curiosity agent - blocks navigation (violates <200ms budget)
- ❌ Database queue - Redis faster for high-throughput queue operations

**Implementation Approach**:
```python
def check_curiosity_trigger(candidate, expected_score, actual_score):
    prediction_error = abs(expected_score - actual_score)

    if prediction_error > self.curiosity_threshold:
        # Add to Redis queue (non-blocking)
        trigger_data = {
            "trigger_type": "prediction_error",
            "concept": candidate,
            "error_magnitude": prediction_error,
            "timestamp": datetime.now().isoformat()
        }
        self.redis_client.lpush("curiosity_queue", json.dumps(trigger_data))
        self.curiosity_triggers_spawned += 1
```

**References**: Spec 029 (Curiosity-Driven Background Agents)

---

### 5. Causal Reasoning Integration
**Question**: How can we achieve <30ms causal intervention predictions?

**Decision**: Pre-compute causal DAG structure, cache intervention predictions for frequent paths

**Rationale**:
- Spec 033 requires <30ms prediction latency
- On-the-fly causal inference too slow for real-time navigation
- Pre-computation enables fast lookup during path selection

**Alternatives Considered**:
- ❌ On-the-fly causal inference - exceeds latency budget (100-500ms typical)
- ❌ Simple heuristic scoring - loses causal semantics and intervention benefits

**Implementation Approach**:
```python
class CausalBayesianNetwork:
    def __init__(self):
        # Pre-compute causal DAG offline
        self.dag = self.build_causal_dag()
        self.intervention_cache = {}  # LRU cache (size=1000)

    def estimate_intervention(self, intervention, target):
        cache_key = (intervention, target)

        if cache_key in self.intervention_cache:
            return self.intervention_cache[cache_key]

        # Do-calculus: P(target | do(intervention))
        score = self.do_calculus(intervention, target)

        self.intervention_cache[cache_key] = score
        return score
```

**References**: Spec 033 (Causal Reasoning and Counterfactual)

---

### 6. Listwise Evidence Scoring
**Question**: How should the curator avoid selecting redundant evidence?

**Decision**: Pairwise redundancy matrix + greedy selection with diversity penalty

**Rationale**:
- CLAUSE curator uses listwise scoring to detect redundancy
- Pairwise similarity matrix captures evidence overlap
- Greedy selection with diversity penalty is fast and effective

**Alternatives Considered**:
- ❌ Independent scoring - selects redundant snippets (high query relevance but duplicate info)
- ❌ Global optimization (ILP) - computationally expensive (exceeds <100ms budget)

**Implementation Approach**:
```python
def score_evidence_listwise(evidence_pool, query):
    # Compute pairwise similarity matrix
    n = len(evidence_pool)
    similarity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            sim = self.compute_similarity(evidence_pool[i], evidence_pool[j])
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim

    # Score each evidence with diversity penalty
    scores = []
    selected_indices = []

    for i, evidence in enumerate(evidence_pool):
        # Base score: query relevance
        base_score = self.query_relevance(evidence, query)

        # Diversity penalty: similarity to already selected
        diversity_penalty = 0
        for selected_idx in selected_indices:
            diversity_penalty += similarity_matrix[i, selected_idx]

        final_score = base_score - 0.3 * diversity_penalty
        scores.append((i, evidence, final_score))

    return sorted(scores, key=lambda x: x[2], reverse=True)
```

**References**: CLAUSE paper Section 3.3 (Context Curator)

---

### 7. Token Budget Enforcement
**Question**: How can we accurately count tokens to match LLM tokenization?

**Decision**: Use tiktoken library (GPT tokenizer) with 10% safety buffer

**Rationale**:
- tiktoken matches GPT tokenization exactly
- Safety buffer prevents budget violations due to tokenization edge cases
- Fast (<1ms for typical evidence snippets)

**Alternatives Considered**:
- ❌ Simple whitespace tokenization - inaccurate (20-30% error vs GPT)
- ❌ Exact LLM tokenization without buffer - risks budget violations

**Implementation Approach**:
```python
import tiktoken

class ContextCurator:
    def __init__(self, token_budget=2048):
        self.token_budget = token_budget
        self.encoder = tiktoken.encoding_for_model("gpt-4")

    def count_tokens(self, text):
        tokens = self.encoder.encode(text)
        # Add 10% safety buffer
        return int(len(tokens) * 1.1)

    def curate(self, evidence_pool):
        selected = []
        total_tokens = 0

        for evidence, score in self.score_evidence_listwise(evidence_pool):
            snippet_tokens = self.count_tokens(evidence)
            shaped_utility = score - self.lambda_tok * snippet_tokens

            if shaped_utility > 0 and total_tokens + snippet_tokens <= self.token_budget:
                selected.append(evidence)
                total_tokens += snippet_tokens
            else:
                self.learned_stop_triggered = True
                break

        return selected, total_tokens
```

**Dependencies**: tiktoken==0.5.1

---

### 8. Provenance Metadata Structure
**Question**: How should provenance metadata be stored and queried?

**Decision**: Nested dict with 7 required fields + 3 trust signals, stored as Neo4j node properties

**Rationale**:
- Spec 032 specifies exact provenance schema (7 fields + trust signals)
- Neo4j properties enable querying by provenance fields (source_uri, verification_status, etc.)
- Nested structure preserves hierarchy while remaining queryable

**Alternatives Considered**:
- ❌ Separate provenance table - duplicates data and complicates queries
- ❌ JSON string storage - can't query provenance fields efficiently

**Implementation Approach**:
```python
def add_provenance(evidence, source_uri, query):
    provenance = {
        "source_uri": source_uri,
        "extraction_timestamp": datetime.now().isoformat(),
        "extractor_identity": "ContextCurator-v2.0",
        "supporting_evidence": evidence[:200],  # Snippet
        "verification_status": "pending_review",
        "corroboration_count": self.count_corroborations(evidence),
        "trust_signals": {
            "reputation_score": self.calculate_reputation(source_uri),
            "recency_score": self.calculate_recency(source_uri),
            "semantic_consistency": self.calculate_consistency(evidence, query)
        }
    }

    # Store in Neo4j as node properties
    self.neo4j.run("""
        MERGE (e:Evidence {text: $text})
        SET e.provenance = $provenance
    """, text=evidence, provenance=json.dumps(provenance))

    return {"evidence": evidence, "provenance": provenance}
```

**References**: Spec 032 (Emergent Pattern Detection)

---

### 9. LC-MAPPO Architecture
**Question**: How should the multi-agent coordinator be structured?

**Decision**: Centralized critic with 4 heads (task, edge_cost, latency_cost, token_cost), decentralized actors

**Rationale**:
- CLAUSE uses CTDE (centralized training, decentralized execution)
- Centralized critic enables coordination via counterfactual advantages
- 4 heads separate task reward from 3 cost types

**Alternatives Considered**:
- ❌ Independent critics per agent - loses coordination benefits
- ❌ Fully centralized actors - loses agent autonomy and modularity

**Implementation Approach**:
```python
class LCMAPPOCoordinator:
    def __init__(self):
        # Centralized critic (4 heads)
        self.critic = CentralizedCritic(
            state_dim=1154,  # From state encoding
            heads=["task_value", "edge_cost", "latency_cost", "token_cost"]
        )

        # Decentralized actors
        self.architect = SubgraphArchitect()  # Phase 1
        self.navigator = PathNavigator()      # Phase 2
        self.curator = ContextCurator()       # Phase 2

        # Dual variables
        self.lambda_edge = 0.01
        self.lambda_lat = 0.01
        self.lambda_tok = 0.01

    def calculate_shaped_returns(self, episode):
        shaped_rewards = []

        for t, transition in enumerate(episode):
            r_acc = transition.reward
            c_edge = transition.edge_cost
            c_lat = transition.latency_cost
            c_tok = transition.token_cost

            r_shaped = (
                r_acc -
                self.lambda_edge * c_edge -
                self.lambda_lat * c_lat -
                self.lambda_tok * c_tok
            )
            shaped_rewards.append(r_shaped)

        return shaped_rewards
```

**References**: CLAUSE paper Section 3.4 (LC-MAPPO)

---

### 10. Conflict Resolution Strategy
**Question**: How can we detect and resolve concurrent basin writes with <10ms latency?

**Decision**: Optimistic locking with Neo4j transaction versioning, MERGE takes max basin strength

**Rationale**:
- Spec 031 requires <10ms rollback latency
- Optimistic locking avoids pessimistic locks (better throughput)
- MERGE strategy preserves strongest basin update

**Alternatives Considered**:
- ❌ Pessimistic locking - degrades throughput (locks held during agent execution)
- ❌ Last-write-wins - loses basin strength updates (violates Phase 1 semantics)

**Implementation Approach**:
```python
class ConflictResolver:
    def __init__(self, neo4j_driver):
        self.driver = neo4j_driver

    def write_with_conflict_detection(self, node_id, updates):
        with self.driver.session() as session:
            # Start transaction with checkpoint
            tx = session.begin_transaction()

            try:
                # Read current version
                result = tx.run("""
                    MATCH (n {id: $node_id})
                    RETURN n.version as version, n.strength as strength
                """, node_id=node_id)

                current_version = result.single()["version"]
                current_strength = result.single()["strength"]

                # Write with version check
                update_result = tx.run("""
                    MATCH (n {id: $node_id, version: $version})
                    SET n.strength = $new_strength, n.version = $version + 1
                    RETURN n
                """, node_id=node_id, version=current_version, new_strength=updates["strength"])

                if not update_result.single():
                    # Conflict detected (version mismatch)
                    tx.rollback()
                    return self.resolve_conflict(node_id, updates, current_strength)

                tx.commit()
                return {"status": "success", "final_strength": updates["strength"]}

            except Exception as e:
                tx.rollback()
                raise

    def resolve_conflict(self, node_id, updates, current_strength):
        # MERGE strategy: take max
        final_strength = max(updates["strength"], current_strength)

        # Retry with exponential backoff
        for delay in [0.1, 0.2, 0.4]:
            time.sleep(delay)
            result = self.write_with_conflict_detection(node_id, {"strength": final_strength})
            if result["status"] == "success":
                return result

        raise Exception("Conflict resolution failed after 3 retries")
```

**References**: Spec 031 (Write Conflict Resolution)

---

### 11. Performance Optimization Strategy
**Question**: How can we meet aggressive latency targets (<200ms navigation, <100ms curation)?

**Decision**: Multi-level caching (Redis for ThoughtSeeds, in-memory for causal, NumPy for batch ops)

**Rationale**:
- NFRs require aggressive latency targets
- Redis TTL=1h for ThoughtSeeds (cross-session reuse)
- LRU cache (size=1000) for causal predictions (hot paths)
- NumPy vectorization for batch similarity computations

**Alternatives Considered**:
- ❌ Single-level cache only - insufficient speedup for all operations
- ❌ No caching - misses latency targets (baseline 500ms+ for navigation)

**Implementation Approach**:
```python
# ThoughtSeed caching (Redis)
def get_thoughtseed_cached(concept):
    cache_key = f"ts:{concept}"
    cached = self.redis_client.get(cache_key)

    if cached:
        return json.loads(cached)

    # Generate and cache
    ts = self.thoughtseed_gen.create(concept)
    self.redis_client.setex(cache_key, 3600, json.dumps(ts))  # 1-hour TTL
    return ts

# Causal prediction caching (in-memory LRU)
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_causal_score_cached(candidate, target):
    return self.causal_reasoner.estimate_intervention(candidate, target)

# Batch similarity (NumPy vectorization)
def compute_similarity_batch(query_emb, candidate_embs):
    # Vectorized cosine similarity
    query_norm = query_emb / np.linalg.norm(query_emb)
    candidate_norms = candidate_embs / np.linalg.norm(candidate_embs, axis=1, keepdims=True)
    similarities = candidate_norms @ query_norm
    return similarities
```

---

### 12. Agent Handoff Protocol
**Question**: How should the coordinator orchestrate the three agents?

**Decision**: Sequential execution (Architect → Navigator → Curator) with budget tracking between agents

**Rationale**:
- CLAUSE paper shows sequential handoff preserves budget coordination
- Each agent receives remaining budgets from previous agent
- Sequential execution simpler than parallel (no conflict resolution needed)

**Alternatives Considered**:
- ❌ Parallel agent execution - requires complex conflict resolution, no clear benefit
- ❌ Iterative refinement - exceeds latency budget (3x agent executions)

**Implementation Approach**:
```python
def coordinate(query, budgets):
    # Agent 1: Subgraph Architect
    start = time.time()
    subgraph_result = self.architect.build_subgraph(
        query=query,
        edge_budget=budgets["edge_budget"]
    )
    architect_latency = time.time() - start

    # Agent 2: Path Navigator
    start = time.time()
    path_result = self.navigator.navigate(
        query=query,
        graph=subgraph_result["graph"],
        step_budget=budgets["step_budget"]
    )
    navigator_latency = time.time() - start

    # Agent 3: Context Curator
    start = time.time()
    evidence_pool = self.extract_evidence_from_path(path_result["path"])
    curator_result = self.curator.curate(
        evidence_pool=evidence_pool,
        token_budget=budgets["token_budget"]
    )
    curator_latency = time.time() - start

    return {
        "result": {
            "subgraph": subgraph_result,
            "path": path_result,
            "evidence": curator_result
        },
        "agent_handoffs": [
            {"step": 1, "agent": "SubgraphArchitect", "latency_ms": architect_latency * 1000},
            {"step": 2, "agent": "PathNavigator", "latency_ms": navigator_latency * 1000},
            {"step": 3, "agent": "ContextCurator", "latency_ms": curator_latency * 1000}
        ]
    }
```

---

## Research Summary

**Total Research Items**: 12
**Status**: ✅ All resolved - No NEEDS CLARIFICATION remaining

**Key Technologies Selected**:
1. NetworkX 3.x - Path finding and graph traversal
2. tiktoken - Token counting (GPT tokenizer)
3. NumPy 2.0+ - Vectorized operations
4. Redis 7.x - ThoughtSeed cache + curiosity queue
5. Neo4j 5.x - Knowledge graph + provenance storage

**Performance Optimizations**:
- Redis caching (ThoughtSeed, 1-hour TTL)
- LRU caching (Causal predictions, size=1000)
- NumPy vectorization (Batch similarity)
- Pre-computation (Causal DAG structure)

**Next Phase**: Generate data models and API contracts from research decisions

---
*Research complete: 2025-10-02*
