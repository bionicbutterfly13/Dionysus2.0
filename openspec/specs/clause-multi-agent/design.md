# CLAUSE Multi-Agent System - Design Document

## Architecture Overview

The CLAUSE Multi-Agent System implements consciousness-enhanced knowledge graph reasoning through three specialized agents coordinated by an LC-MAPPO reinforcement learning framework. The architecture follows a pipeline pattern where each agent produces output consumed by the next agent.

### Agent Pipeline

```
Query → SubgraphArchitect → PathNavigator → ContextCurator → Response
          (subgraph)         (paths)        (evidence)
```

**Data Flow**:
1. **Input**: User query + knowledge graph
2. **SubgraphArchitect**: Constructs compact, query-relevant subgraph (budget: β_edge)
3. **PathNavigator**: Explores paths through subgraph (budget: β_step)
4. **ContextCurator**: Selects evidence from path (budget: β_tok)
5. **Output**: Curated evidence with full provenance

### Component Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   LC-MAPPO Coordinator                       │
│  - Centralized Critic (4 heads)                             │
│  - Dual Variables (λ_edge, λ_lat, λ_tok)                    │
│  - Shaped Return Calculator                                  │
│  - Conflict Resolver                                         │
└─────────────────────────────────────────────────────────────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
           ▼               ▼               ▼
┌──────────────────┐ ┌──────────────┐ ┌──────────────────┐
│ SubgraphArchitect│ │PathNavigator │ │ContextCurator    │
│ - 5-Signal Scorer│ │- Termination │ │- Listwise Scorer │
│ - Basin Tracker  │ │  Head        │ │- Token Counter   │
│ - Edge Selector  │ │- ThoughtSeed │ │- Provenance Gen  │
│                  │ │  Generator   │ │                  │
└──────────────────┘ └──────────────┘ └──────────────────┘
           │               │               │
           └───────────────┼───────────────┘
                           ▼
                    ┌──────────────┐
                    │   Neo4j DB   │
                    │ - Graph      │
                    │ - Vector     │
                    │ - Full-text  │
                    └──────────────┘
```

## Subgraph Architect

### 5-Signal Edge Scoring

Edges are scored using weighted combination of 5 signals:

**Formula**:
```
score(e|q,G) = 0.25×φ_ent + 0.25×φ_rel + 0.20×φ_nbr + 0.15×φ_deg + 0.15×φ_basin
```

**Signal Definitions**:

1. **Entity Match (φ_ent)**: Cosine similarity between edge nodes and query entities
   - Range: [0, 1]
   - Calculation: `max(sim(e.source, q), sim(e.target, q))`

2. **Relation Match (φ_rel)**: Semantic similarity between edge relation and query intent
   - Range: [0, 1]
   - Calculation: `sim(e.relation_type, query_intent_embedding)`

3. **Neighborhood (φ_nbr)**: Co-occurrence frequency of edge nodes in attractor basins
   - Range: [0, 1]
   - Calculation: `basin.co_occurring_concepts[target_concept] / max_cooccurrence`

4. **Degree (φ_deg)**: Normalized node degree (prefers hub nodes)
   - Range: [0, 1]
   - Calculation: `log(degree(node) + 1) / log(max_degree + 1)`

5. **Basin Strength (φ_basin)**: Frequency-based concept importance
   - Range: [0, 1]
   - Calculation: `(basin.strength - 1.0) / 1.0` (normalized to 0-1)

### Shaped Gain Rule

Edge selection uses budget-aware shaped gain:

**Formula**:
```
shaped_gain(e) = score(e) - λ_edge × c_edge
```

**Selection Rule**:
- Accept edge if `shaped_gain(e) > 0`
- Stop when budget exhausted OR all candidates have `shaped_gain ≤ 0`

**Dual Variable Update**:
```
λ_edge ← max(0, λ_edge + η × (E[C_edge] - β_edge))
```

Where:
- `E[C_edge]`: Expected edge cost from current policy
- `β_edge`: Edge budget constraint
- `η`: Learning rate (default 0.001)

### Basin Frequency Strengthening

**Strength Update Rule**:
```
strength_new = min(2.0, strength_old + 0.2)
```

**Properties**:
- Base strength: 1.0 (first appearance)
- Increment: +0.2 per reappearance
- Cap: 2.0 maximum
- Storage: Neo4j `AttractorBasin.strength` property

**Co-occurrence Tracking**:
```python
# Dictionary structure in AttractorBasin node
co_occurring_concepts = {
    "concept_A": 5,  # Co-occurred 5 times
    "concept_B": 3,  # Co-occurred 3 times
    ...
}
```

**Activation History**:
```python
activation_history = [
    {"timestamp": "2025-10-01T10:30:00Z", "strength": 1.2},
    {"timestamp": "2025-10-01T14:45:00Z", "strength": 1.4},
    ...
]
```

### Neo4j Integration

**Cypher Query Pattern**:
```cypher
// Fetch basin with co-occurrence context
MATCH (b:AttractorBasin {concept: $concept})
RETURN b.strength, b.co_occurring_concepts, b.activation_history

// Update basin strength
MATCH (b:AttractorBasin {concept: $concept})
SET b.strength = $new_strength,
    b.activation_history = b.activation_history + [{
        timestamp: datetime(),
        strength: $new_strength
    }]
```

### Performance Optimization

**Edge Scoring Cache**:
- LRU cache for entity embeddings (10,000 entries)
- TTL: 1 hour
- Cache key: `(concept, query_embedding_hash)`

**Batch Processing**:
- Score edges in batches of 100
- Parallel processing via `ThreadPoolExecutor` (8 workers)
- Target: < 10ms per edge

## Path Navigator

### State Encoding

At each navigation step, encode state vector:

**State Vector Components**:
```python
state = [
    query_embedding,           # 384-dim
    current_node_embedding,    # 384-dim
    path_history_embedding,    # 384-dim (LSTM-encoded)
    budget_remaining,          # scalar (normalized)
    basin_strength_current,    # scalar
    step_count                 # scalar (normalized)
]
# Total: 1152 + 3 dimensions
```

### Termination Head

**Decision Rule**:
```python
def should_terminate(state, candidates, budget_used, budget_total):
    if budget_used >= budget_total:
        return True, "BUDGET_EXHAUSTED"

    # Calculate shaped gain for all candidates
    shaped_gains = [
        score(c) - lambda_step * step_cost
        for c in candidates
    ]

    if max(shaped_gains) <= 0:
        return True, "NO_POSITIVE_GAIN"

    return False, None
```

### Action Selection

**Action Space**:
1. **CONTINUE**: Select next hop from candidates
   - Choose candidate with highest `shaped_gain`
   - Update path history
   - Decrement budget

2. **BACKTRACK**: Return to previous node
   - Pop from path stack
   - Restore previous state
   - No budget cost

3. **STOP**: Terminate navigation
   - Return current path
   - Final action marked as STOP

**Selection Policy**:
```python
if should_terminate(state, candidates, budget_used, budget_total):
    action = STOP
elif best_candidate.shaped_gain > backtrack_threshold:
    action = CONTINUE(best_candidate)
else:
    action = BACKTRACK
```

### ThoughtSeed Generation

**Generation Timing**: For each candidate next hop during action selection

**ThoughtSeed Structure**:
```python
@dataclass
class ThoughtSeed:
    id: UUID
    concept: str
    source_doc: str  # Query or document ID
    basin_context: Dict[str, float]  # {concept: co-occurrence_score}
    similarity_threshold: float = 0.8
    embedding: np.ndarray  # 384-dim
    created_at: datetime
```

**Cross-Document Linking**:
```cypher
// Find similar ThoughtSeeds
MATCH (ts1:ThoughtSeed {concept: $concept})
MATCH (ts2:ThoughtSeed)
WHERE ts1.id <> ts2.id
  AND vector.similarity.cosine(ts1.embedding, ts2.embedding) > $threshold
CREATE (ts1)-[r:SIMILAR_TO {similarity: $score}]->(ts2)
```

**Performance Target**: 100+ seeds/sec (< 10ms per seed)

### Curiosity Integration

**Prediction Error Calculation**:
```python
prediction_error = abs(expected_score - actual_score)
```

**Trigger Condition**:
```python
if prediction_error > curiosity_threshold:  # Default: 0.7
    spawn_curiosity_agent(
        trigger_type="HIGH_PREDICTION_ERROR",
        concept=candidate.concept,
        error_magnitude=prediction_error,
        timestamp=datetime.now()
    )
```

**Non-Blocking Spawn**:
```python
# Add to Redis queue for background processing
curiosity_queue.push({
    "trigger_type": "HIGH_PREDICTION_ERROR",
    "concept": concept,
    "error_magnitude": error_magnitude,
    "query_context": query_embedding,
    "timestamp": timestamp
})
```

**Queue Processing**:
- Background worker pool (4 workers)
- Process curiosity triggers asynchronously
- Update knowledge graph with exploration results
- No impact on navigation latency

### Causal Path Selection

**Causal Intervention Prediction**:

For each candidate next hop, estimate:
```
P(answer | do(select_path = candidate))
```

**Do-Calculus Implementation**:
```python
def estimate_causal_impact(candidate, query, graph):
    # Build Bayesian network from graph structure
    bn = build_bayesian_network(graph, query)

    # Perform intervention: do(select_path = candidate)
    intervened_bn = bn.intervene(variable="path_selection", value=candidate)

    # Estimate posterior probability
    posterior = intervened_bn.infer(variable="answer_correctness")

    return posterior.probability(True)
```

**Timeout Handling**:
```python
try:
    with timeout(30):  # 30ms timeout
        causal_score = estimate_causal_impact(candidate, query, graph)
except TimeoutError:
    # Queue for background processing
    causal_queue.push({
        "candidate": candidate,
        "query": query,
        "timestamp": datetime.now()
    })

    # Use semantic similarity heuristic immediately
    causal_score = semantic_similarity(candidate.embedding, query_embedding)
    fallback_used = True
```

**Caching**:
- LRU cache for causal DAG structures (1,000 entries)
- Cache key: `(query_embedding_hash, graph_structure_hash)`
- TTL: 1 hour

## Context Curator

### Listwise Evidence Scoring

**Anti-Redundancy Scoring**:
```python
def score_evidence_listwise(candidate, already_selected):
    # Base query relevance
    relevance_score = cosine_similarity(candidate.embedding, query_embedding)

    # Redundancy penalty
    max_overlap = max([
        semantic_overlap(candidate, selected)
        for selected in already_selected
    ] or [0])

    redundancy_penalty = max_overlap * redundancy_weight  # redundancy_weight=0.5

    return relevance_score - redundancy_penalty
```

**Shaped Utility Rule**:
```python
def select_evidence(evidence_pool, token_budget):
    selected = []
    tokens_used = 0

    for candidate in sorted(evidence_pool, key=lambda e: e.score, reverse=True):
        candidate_tokens = count_tokens(candidate.text)
        shaped_utility = candidate.score - lambda_tok * candidate_tokens

        if shaped_utility <= 0:
            # Learned stop triggered
            break

        if tokens_used + candidate_tokens > token_budget:
            # Budget exhausted
            break

        selected.append(candidate)
        tokens_used += candidate_tokens

    return selected, tokens_used
```

### Token Counting

**Implementation**: Use `tiktoken` library (matches GPT tokenizer)

```python
import tiktoken

encoder = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer

def count_tokens(text: str) -> int:
    return len(encoder.encode(text))
```

**Conservative Buffer**: Add 10% buffer to handle tokenizer discrepancies

### Provenance Metadata Generation

**Required Fields**:
```python
@dataclass
class ProvenanceMetadata:
    source_uri: str  # Neo4j URI or document ID
    extraction_timestamp: datetime
    extractor_identity: str  # "ContextCurator-v2.0"
    supporting_evidence: str  # The evidence text itself
    verification_status: str  # "verified" | "unverified" | "disputed"
    corroboration_count: int  # Number of sources with same evidence
    trust_signals: TrustSignals
```

**Trust Signals Structure**:
```python
@dataclass
class TrustSignals:
    reputation_score: float  # Source reputation (0-1)
    recency_score: float     # Time-decay score (0-1)
    semantic_consistency: float  # Consistency with graph (0-1)
```

**Reputation Calculation**:
```python
def calculate_reputation_score(source_uri):
    # Fetch historical accuracy from Neo4j
    accuracy_history = fetch_source_accuracy(source_uri)

    # Exponential moving average
    reputation = sum([
        accuracy * (decay_factor ** age_days)
        for accuracy, age_days in accuracy_history
    ]) / len(accuracy_history)

    return min(1.0, max(0.0, reputation))
```

**Recency Score**:
```python
def calculate_recency_score(extraction_timestamp):
    age_days = (datetime.now() - extraction_timestamp).days
    decay_rate = 0.1  # 10% decay per day

    return math.exp(-decay_rate * age_days)
```

**Semantic Consistency**:
```python
def calculate_semantic_consistency(evidence, graph_context):
    # Compare evidence embedding with graph neighborhood embeddings
    similarities = [
        cosine_similarity(evidence.embedding, neighbor.embedding)
        for neighbor in graph_context.neighbors
    ]

    return np.mean(similarities)
```

### Neo4j Provenance Storage

**Cypher Pattern**:
```cypher
CREATE (e:Evidence {
    text: $evidence_text,
    tokens: $token_count,
    score: $relevance_score
})
CREATE (p:Provenance {
    source_uri: $source_uri,
    extraction_timestamp: datetime(),
    extractor_identity: "ContextCurator-v2.0",
    verification_status: $verification_status,
    corroboration_count: $corroboration_count,
    reputation_score: $reputation_score,
    recency_score: $recency_score,
    semantic_consistency: $semantic_consistency
})
CREATE (e)-[:HAS_PROVENANCE]->(p)
```

## LC-MAPPO Coordinator

### Centralized Critic Architecture

**4-Head Value Network**:
```python
class CentralizedCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_encoder = nn.Linear(state_dim, 512)

        # 4 value heads
        self.task_value_head = nn.Linear(512, 1)
        self.edge_cost_head = nn.Linear(512, 1)
        self.latency_cost_head = nn.Linear(512, 1)
        self.token_cost_head = nn.Linear(512, 1)

    def forward(self, state):
        h = F.relu(self.shared_encoder(state))

        return {
            "task_value": self.task_value_head(h),
            "edge_cost": self.edge_cost_head(h),
            "latency_cost": self.latency_cost_head(h),
            "token_cost": self.token_cost_head(h)
        }
```

### Dual Variable Management

**Initialization**:
```python
lambdas = {
    "edge": 0.01,
    "latency": 0.01,
    "token": 0.01
}
```

**Update Rule** (after each episode):
```python
def update_dual_variables(episode_costs, budgets, learning_rate=0.001):
    for constraint in ["edge", "latency", "token"]:
        expected_cost = episode_costs[constraint]
        budget = budgets[constraint]

        # Gradient step
        lambdas[constraint] = max(0,
            lambdas[constraint] + learning_rate * (expected_cost - budget)
        )
```

**Interpretation**:
- If `expected_cost > budget`: Increase λ (penalize constraint violation more)
- If `expected_cost < budget`: Decrease λ (relax constraint)
- Clamp at 0 (never reward exceeding budget)

### Shaped Return Calculation

**Formula**:
```python
def calculate_shaped_return(reward, costs, lambdas):
    shaped_reward = reward

    for constraint, cost in costs.items():
        shaped_reward -= lambdas[constraint] * cost

    return shaped_reward
```

**Example**:
```python
# Episode results
reward = 0.85  # Accuracy
costs = {
    "edge": 45,      # 45 edges used (budget: 50)
    "latency": 230,  # 230ms (budget: 200ms)
    "token": 1100    # 1100 tokens (budget: 2048)
}
lambdas = {
    "edge": 0.01,
    "latency": 0.01,
    "token": 0.01
}

shaped_return = 0.85 - (0.01*45 + 0.01*230 + 0.01*1100)
              = 0.85 - 13.75
              = -12.90
```

**Interpretation**: High penalty from latency violation drives policy to reduce latency

### Agent Handoff Coordination

**Sequential Pipeline**:
```python
def coordinate_agents(query, budgets, lambdas):
    # Step 1: Subgraph construction
    subgraph = subgraph_architect.construct(
        query=query,
        edge_budget=budgets["edge"],
        lambda_edge=lambdas["edge"]
    )

    # Step 2: Path navigation
    path = path_navigator.navigate(
        query=query,
        graph=subgraph,
        step_budget=budgets["step"],
        lambda_step=lambdas["latency"]
    )

    # Step 3: Evidence curation
    evidence = context_curator.curate(
        evidence_pool=path.evidence_pool,
        token_budget=budgets["token"],
        lambda_tok=lambdas["token"]
    )

    return {
        "subgraph": subgraph,
        "path": path,
        "evidence": evidence
    }
```

**Handoff Metadata**:
```python
agent_handoffs = [
    {
        "step": 1,
        "agent": "SubgraphArchitect",
        "action": "built_subgraph",
        "budget_used": {"edges": 35},
        "latency_ms": 287
    },
    {
        "step": 2,
        "agent": "PathNavigator",
        "action": "explored_paths",
        "budget_used": {"steps": 7},
        "latency_ms": 145
    },
    {
        "step": 3,
        "agent": "ContextCurator",
        "action": "selected_evidence",
        "budget_used": {"tokens": 1024},
        "latency_ms": 78
    }
]
```

## Write Conflict Resolution

### Conflict Detection

**Transaction Checkpoint Pattern**:
```python
def write_with_conflict_detection(node_id, updates):
    # Begin transaction and record checkpoint
    tx = neo4j_driver.session().begin_transaction()

    try:
        # Read current version
        current_version = tx.run(
            "MATCH (n) WHERE id(n) = $node_id RETURN n.version",
            node_id=node_id
        ).single()["n.version"]

        # Apply updates with version check
        result = tx.run(
            """
            MATCH (n) WHERE id(n) = $node_id AND n.version = $expected_version
            SET n += $updates, n.version = $expected_version + 1
            RETURN n.version
            """,
            node_id=node_id,
            expected_version=current_version,
            updates=updates
        )

        if result.single() is None:
            # Version mismatch = conflict detected
            raise WriteConflictError(f"Conflict detected on node {node_id}")

        tx.commit()

    except WriteConflictError as e:
        tx.rollback()
        raise e
```

### Resolution Strategies

**1. MERGE Strategy** (for basin strength):
```python
def resolve_conflict_merge(node_id, agent_updates):
    # Take maximum basin strength across conflicting writes
    max_strength = max([
        update["basin_strength"]
        for update in agent_updates
    ])

    # Apply merged update
    tx.run(
        "MATCH (n) WHERE id(n) = $node_id SET n.basin_strength = $max_strength",
        node_id=node_id,
        max_strength=max_strength
    )
```

**2. ROLLBACK Strategy**:
```python
def resolve_conflict_rollback(tx):
    tx.rollback()
    # Discard conflicting write, revert to checkpoint
```

**3. RETRY Strategy**:
```python
def resolve_conflict_retry(write_fn, max_retries=3):
    backoff_delays = [100, 200, 400]  # ms

    for attempt, delay in enumerate(backoff_delays):
        try:
            return write_fn()
        except WriteConflictError:
            if attempt == max_retries - 1:
                raise
            time.sleep(delay / 1000)  # Convert to seconds
```

### Conflict Monitoring

**Metrics Collection**:
```python
@dataclass
class ConflictEvent:
    timestamp: datetime
    agents: List[str]  # Agents involved in conflict
    node_id: int
    resolution_strategy: str  # "MERGE" | "ROLLBACK" | "RETRY"
    resolution_time_ms: float

# Continuous monitoring
conflict_rate = len(conflict_events) / total_writes
```

**Threshold Analysis**:
- Monitor conflict rate continuously during production
- Research distributed systems best practices for acceptable rates
- Establish baseline threshold during implementation
- Implement read-only fallback when threshold exceeded (initially disabled)

## Performance Optimization

### Caching Strategy

**Multi-Level Cache**:
1. **L1 (In-Memory)**: Entity embeddings, query embeddings
   - Size: 10,000 entries
   - TTL: 1 hour
   - Eviction: LRU

2. **L2 (Redis)**: Causal DAG structures, basin contexts
   - Size: 100,000 entries
   - TTL: 24 hours
   - Eviction: TTL-based

3. **L3 (Neo4j)**: Persistent graph data
   - Size: Unlimited
   - TTL: Permanent

### Batch Processing

**Edge Scoring Batches**:
```python
def score_edges_batch(edges, query, batch_size=100):
    results = []

    with ThreadPoolExecutor(max_workers=8) as executor:
        for i in range(0, len(edges), batch_size):
            batch = edges[i:i+batch_size]
            futures = [
                executor.submit(score_edge, edge, query)
                for edge in batch
            ]
            results.extend([f.result() for f in futures])

    return results
```

**ThoughtSeed Persistence Batches**:
```python
def persist_thoughtseeds_batch(thoughtseeds, batch_size=100):
    with neo4j_driver.session() as session:
        for i in range(0, len(thoughtseeds), batch_size):
            batch = thoughtseeds[i:i+batch_size]

            session.run(
                """
                UNWIND $seeds AS seed
                CREATE (ts:ThoughtSeed {
                    id: seed.id,
                    concept: seed.concept,
                    embedding: seed.embedding,
                    ...
                })
                """,
                seeds=[asdict(ts) for ts in batch]
            )
```

### Async Background Processing

**Curiosity Queue Workers**:
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def process_curiosity_queue():
    executor = ThreadPoolExecutor(max_workers=4)

    while True:
        trigger = await curiosity_queue.pop()

        if trigger:
            # Non-blocking spawn
            executor.submit(
                explore_curiosity_trigger,
                trigger["concept"],
                trigger["error_magnitude"]
            )
        else:
            await asyncio.sleep(0.1)  # Polling interval
```

**Causal Computation Workers**:
```python
async def process_causal_queue():
    executor = ThreadPoolExecutor(max_workers=4)

    while True:
        task = await causal_queue.pop()

        if task:
            future = executor.submit(
                estimate_causal_impact,
                task["candidate"],
                task["query"],
                task["graph"]
            )

            # Cache result when complete
            result = future.result()
            causal_cache.set(task["cache_key"], result)
        else:
            await asyncio.sleep(0.1)
```

## API Endpoints

### POST /api/clause/navigate

**Request Body**:
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
    "nodes": ["climate_change", "greenhouse_gases", "CO2_emissions"],
    "edges": [
      {"from": "climate_change", "relation": "caused_by", "to": "greenhouse_gases"},
      {"from": "greenhouse_gases", "relation": "includes", "to": "CO2_emissions"}
    ],
    "steps": [
      {"step": 1, "from": "climate_change", "to": "greenhouse_gases", "action": "CONTINUE", "causal_score": 0.85},
      {"step": 2, "from": "greenhouse_gases", "to": "CO2_emissions", "action": "STOP", "causal_score": 0.92}
    ]
  },
  "metadata": {
    "budget_used": 2,
    "budget_total": 10,
    "final_action": "STOP",
    "thoughtseeds_generated": 8,
    "curiosity_triggers_spawned": 1,
    "causal_predictions": 6
  },
  "performance": {
    "latency_ms": 145,
    "thoughtseed_gen_ms": 15,
    "causal_pred_ms": 72
  }
}
```

### POST /api/clause/curate

**Request Body**:
```json
{
  "evidence_pool": [
    "Greenhouse gases trap heat in the atmosphere...",
    "CO2 is the primary greenhouse gas from human activity..."
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
    "learned_stop_triggered": true
  },
  "performance": {
    "latency_ms": 78,
    "provenance_overhead_ms": 12
  }
}
```

### POST /api/clause/coordinate

**Request Body**:
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
    "subgraph": {"nodes": [...], "edges": [...]},
    "path": {"nodes": [...], "edges": [...]},
    "evidence": [{"text": "...", "provenance": {...}}]
  },
  "agent_handoffs": [
    {"step": 1, "agent": "SubgraphArchitect", "budget_used": {"edges": 35}, "latency_ms": 287},
    {"step": 2, "agent": "PathNavigator", "budget_used": {"steps": 7}, "latency_ms": 145},
    {"step": 3, "agent": "ContextCurator", "budget_used": {"tokens": 1024}, "latency_ms": 78}
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

## Testing Strategy

### Unit Tests

**Subgraph Architect**:
- 5-signal scoring correctness
- Basin strength updates
- Budget enforcement
- Shaped gain calculation

**Path Navigator**:
- State encoding
- Termination head logic
- Action selection
- ThoughtSeed generation
- Curiosity trigger spawning
- Causal prediction timeout

**Context Curator**:
- Listwise scoring
- Token counting
- Learned stop
- Provenance generation

### Integration Tests

**Full Pipeline**:
```python
def test_full_clause_pipeline():
    query = "What causes climate change?"

    coordinator = LCMAPPOCoordinator()
    result = coordinator.coordinate_agents(
        query=query,
        budgets={"edge": 50, "step": 10, "token": 2048}
    )

    assert len(result["subgraph"]["edges"]) <= 50
    assert len(result["path"]["steps"]) <= 10
    assert result["metadata"]["tokens_used"] <= 2048
    assert all("provenance" in e for e in result["evidence"])
```

**Conflict Resolution**:
```python
def test_concurrent_basin_writes():
    node_id = 12345

    # Simulate concurrent writes from 2 agents
    with ThreadPoolExecutor(max_workers=2) as executor:
        future1 = executor.submit(
            write_basin_strength, node_id, agent="navigator", strength=1.4
        )
        future2 = executor.submit(
            write_basin_strength, node_id, agent="curator", strength=1.6
        )

        results = [future1.result(), future2.result()]

    # Verify MERGE strategy took max
    final_strength = fetch_basin_strength(node_id)
    assert final_strength == 1.6
```

### Performance Tests

**Latency Benchmarks**:
```python
def test_navigation_latency_p95():
    queries = generate_test_queries(100)
    latencies = []

    for query in queries:
        start = time.time()
        path = path_navigator.navigate(query, step_budget=10)
        latencies.append((time.time() - start) * 1000)

    p95_latency = np.percentile(latencies, 95)
    assert p95_latency < 200, f"p95 latency {p95_latency}ms exceeds 200ms SLA"
```

**Throughput Tests**:
```python
def test_thoughtseed_throughput():
    concepts = generate_test_concepts(1000)

    start = time.time()
    thoughtseeds = [
        generate_thoughtseed(concept)
        for concept in concepts
    ]
    duration = time.time() - start

    throughput = len(thoughtseeds) / duration
    assert throughput >= 100, f"Throughput {throughput} seeds/sec < 100 seeds/sec SLA"
```

## Deployment Considerations

### Environment Variables

```bash
# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# Redis (caching + queues)
REDIS_HOST=localhost
REDIS_PORT=6379

# Budget defaults
DEFAULT_EDGE_BUDGET=50
DEFAULT_STEP_BUDGET=10
DEFAULT_TOKEN_BUDGET=2048

# Performance tuning
BATCH_SIZE=100
THREAD_POOL_WORKERS=8
CACHE_SIZE=10000
CACHE_TTL_HOURS=1

# Curiosity
CURIOSITY_THRESHOLD=0.7
CURIOSITY_WORKERS=4

# Causal reasoning
CAUSAL_TIMEOUT_MS=30
CAUSAL_WORKERS=4
```

### Resource Requirements

**CPU**: 8+ cores (parallel edge scoring, background workers)
**Memory**: 16GB+ (embedding cache, graph structures)
**Storage**: 100GB+ (Neo4j graph database)
**Network**: Low latency to Neo4j (< 1ms preferred)

### Monitoring Metrics

**System Health**:
- Agent handoff latencies
- Budget compliance rate
- Conflict detection rate
- Conflict resolution latency

**Cache Performance**:
- Hit rate (target: > 80%)
- Eviction rate
- Memory usage

**Queue Health**:
- Curiosity queue depth
- Causal queue depth
- Worker utilization

**Database**:
- Neo4j query latency
- Transaction rollback rate
- Write throughput
