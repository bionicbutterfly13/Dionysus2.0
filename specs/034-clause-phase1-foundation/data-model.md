# Data Model: CLAUSE Architect + Basin Strengthening

**Date**: 2025-10-01
**Feature**: CLAUSE Phase 1 Foundation

## Core Entities

### 1. AttractorBasin (Extended)
**Purpose**: Represents stable cognitive states with frequency strengthening

**Fields**:
| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| basin_id | str | Primary Key, UUID | Unique basin identifier |
| basin_name | str | Required, 1-200 chars | Human-readable concept name |
| basin_type | BasinType | Enum | Type of basin (CONCEPTUAL, SEMANTIC, etc.) |
| strength | float | 1.0-2.0 | **NEW**: Frequency-based strength (starts 1.0, +0.2 per appear, cap 2.0) |
| activation_count | int | ≥0 | **NEW**: Total number of activations |
| activation_history | List[datetime] | - | **NEW**: Timestamps of each activation |
| co_occurring_concepts | Dict[str, int] | - | **NEW**: {concept_id: co-occurrence_count} |
| stability | float | 0.0-1.0 | Basin stability measure |
| depth | float | >0 | Energy well depth |
| current_activation | float | 0.0-1.0 | Current activation level |
| created_at | datetime | Auto | Creation timestamp |
| last_updated | datetime | Auto | Last update timestamp |

**Validation Rules**:
```python
# FR-003: Basin strength increases by +0.2 per reappearance, cap at 2.0
assert 1.0 <= basin.strength <= 2.0, "Strength must be in range [1.0, 2.0]"

# Activation count must match history length
assert basin.activation_count == len(basin.activation_history), \
    "Activation count mismatch"

# Co-occurrence values must be positive
assert all(count > 0 for count in basin.co_occurring_concepts.values()), \
    "Co-occurrence counts must be positive"
```

**State Transitions**:
```
DORMANT → ACTIVATING → ACTIVE → SATURATED
                ↓
           DECAYING → DORMANT
```

**Neo4j Schema**:
```cypher
// Node constraints
CREATE CONSTRAINT basin_id_unique IF NOT EXISTS
FOR (b:AttractorBasin) REQUIRE b.basin_id IS UNIQUE;

// Indexes for performance
CREATE INDEX basin_strength_index IF NOT EXISTS
FOR (b:AttractorBasin) ON (b.strength);

CREATE INDEX basin_activation_index IF NOT EXISTS
FOR (b:AttractorBasin) ON (b.activation_count);

// Example node
CREATE (b:AttractorBasin {
    basin_id: "basin_uuid_123",
    basin_name: "neural architecture search",
    basin_type: "CONCEPTUAL",
    strength: 1.4,  // Appeared 2 times (1.0 + 0.2 + 0.2)
    activation_count: 2,
    activation_history: ["2025-10-01T10:00:00", "2025-10-01T15:30:00"],
    co_occurring_concepts: {"differentiable_nas": 2, "nas_bench": 1},
    stability: 0.85,
    depth: 1.5,
    current_activation: 0.0,
    created_at: "2025-10-01T10:00:00",
    last_updated: "2025-10-01T15:30:00"
})
```

---

### 2. CLAUSESubgraphArchitect (Service)
**Purpose**: Constructs budget-aware query-specific subgraphs

**Input Model**:
```python
@dataclass
class SubgraphRequest:
    query: str                    # User query text
    edge_budget: int = 50         # Maximum edges (β_edge)
    lambda_edge: float = 0.01     # Dual variable (from LC-MAPPO)
    hop_distance: int = 2         # k-hop neighborhood
    min_edge_score: float = 0.3   # Minimum edge quality threshold
```

**Output Model**:
```python
@dataclass
class SubgraphResponse:
    graph: nx.MultiDiGraph        # Constructed subgraph
    selected_edges: List[Edge]    # Edges selected (≤ edge_budget)
    edge_scores: Dict[Edge, float] # Score for each selected edge
    shaped_gains: Dict[Edge, float] # Shaped gain for each edge
    budget_used: int              # Actual edges added
    stopped_reason: str           # "BUDGET_EXHAUSTED" | "GAIN_NEGATIVE" | "COMPLETE"
    construction_time_ms: float   # Total time taken
    basins_strengthened: List[str] # Basin IDs that were strengthened
```

**Internal Models**:
```python
@dataclass
class EdgeScore:
    edge: Tuple[str, str, str]    # (subject, relation, object)
    phi_ent: float                # Entity-question match (0-1)
    phi_rel: float                # Relation-text match (0-1)
    phi_nbr: float                # Neighborhood score (0-1)
    phi_deg: float                # Degree prior (0-1)
    phi_basin: float              # Basin strength normalized (0-1)
    total_score: float            # Weighted sum
    shaped_gain: float            # total_score - λ_edge × cost
```

**Validation Rules**:
```python
# FR-001: Budget must be enforced
assert len(response.selected_edges) <= request.edge_budget, \
    "Budget violation: selected edges exceed edge_budget"

# FR-005: Stop when shaped gain ≤ 0
for edge in response.selected_edges:
    assert response.shaped_gains[edge] > 0, \
        "Invalid selection: edge has non-positive shaped gain"

# Performance constraint (NFR-001)
assert response.construction_time_ms < 500, \
    "Performance violation: construction took >500ms"
```

---

### 3. BasinTracker (Service)
**Purpose**: Manages basin frequency strengthening and co-occurrence tracking

**Input Model**:
```python
@dataclass
class BasinStrengtheningRequest:
    concepts: List[str]           # Concepts from document
    document_id: str              # Source document identifier
    increment: float = 0.2        # Strength increment per appearance
```

**Output Model**:
```python
@dataclass
class BasinStrengtheningResponse:
    updated_basins: List[AttractorBasin]  # Basins that were updated
    new_basins: List[AttractorBasin]      # Newly created basins
    cooccurrence_updates: Dict[Tuple[str, str], int]  # Pair → count delta
    total_strengthening_time_ms: float
```

**Internal Models**:
```python
@dataclass
class CoOccurrencePair:
    concept_a: str
    concept_b: str
    count: int                    # Number of co-occurrences
    first_seen: datetime          # When pair first observed
    last_seen: datetime           # Most recent co-occurrence

    @property
    def symmetric_key(self) -> Tuple[str, str]:
        """Returns (a,b) if a<b else (b,a) for deduplication"""
        return tuple(sorted([self.concept_a, self.concept_b]))
```

**Validation Rules**:
```python
# FR-003: Strength cap at 2.0
for basin in response.updated_basins:
    assert basin.strength <= 2.0, \
        f"Basin {basin.basin_id} exceeded max strength 2.0"

# FR-004: Co-occurrence must be symmetric
for (a, b), count in response.cooccurrence_updates.items():
    basin_a = get_basin(a)
    basin_b = get_basin(b)
    assert basin_a.co_occurring_concepts[b] == count, \
        "Asymmetric co-occurrence: A→B count mismatch"
    assert basin_b.co_occurring_concepts[a] == count, \
        "Asymmetric co-occurrence: B→A count mismatch"
```

---

### 4. EdgeScorer (Service)
**Purpose**: Computes 5-signal edge scores for CLAUSE architect

**Input Model**:
```python
@dataclass
class EdgeScoringRequest:
    edges: List[Tuple[str, str, str]]  # List of (subj, rel, obj) edges
    query: str                          # Query text for relevance
    query_embedding: np.ndarray         # Pre-computed query embedding (384-dim)
    graph: nx.MultiDiGraph              # Current graph state
    basin_tracker: BasinTracker         # For basin strength lookup
```

**Output Model**:
```python
@dataclass
class EdgeScoringResponse:
    scores: Dict[Tuple, EdgeScore]      # Edge → full score breakdown
    top_k_edges: List[Tuple]            # Edges sorted by score descending
    scoring_time_ms: float              # Total scoring duration
```

**Signal Computation Models**:
```python
@dataclass
class SignalWeights:
    w_ent: float = 0.25    # Entity match weight
    w_rel: float = 0.25    # Relation match weight
    w_nbr: float = 0.20    # Neighborhood weight
    w_deg: float = 0.15    # Degree prior weight
    w_basin: float = 0.15  # Basin strength weight

    def __post_init__(self):
        assert abs(sum([self.w_ent, self.w_rel, self.w_nbr,
                        self.w_deg, self.w_basin]) - 1.0) < 1e-6, \
            "Weights must sum to 1.0"
```

**Validation Rules**:
```python
# FR-002: 5-signal scoring with specified weights
weights = SignalWeights()
for edge, score in response.scores.items():
    expected = (
        weights.w_ent * score.phi_ent +
        weights.w_rel * score.phi_rel +
        weights.w_nbr * score.phi_nbr +
        weights.w_deg * score.phi_deg +
        weights.w_basin * score.phi_basin
    )
    assert abs(score.total_score - expected) < 1e-6, \
        "Score calculation mismatch"

# NFR-001: Edge scoring <10ms
assert response.scoring_time_ms < 10, \
    f"Scoring too slow: {response.scoring_time_ms}ms > 10ms"
```

---

## Entity Relationships

### Neo4j Graph Schema
```cypher
// Concept nodes (existing)
(c:Concept {concept_id, name, embedding})

// Basin nodes (extended with new fields)
(b:AttractorBasin {
    basin_id, basin_name, basin_type,
    strength, activation_count, activation_history,
    co_occurring_concepts, ...
})

// Relationships
(c)-[:HAS_BASIN]->(b)          // Concept to its attractor basin
(c)-[:RELATES_TO {              // Concept relationships (edges in subgraph)
    relation_type,
    weight,
    selected_in_query: [query_ids]  // Track which queries selected this edge
}]->(c)

(b)-[:CO_OCCURS_WITH {          // Basin co-occurrence (explicit edges)
    count,
    first_seen,
    last_seen
}]->(b)
```

### Memory Layout (In-Process)
```python
class ArchitectState:
    # Level 1: Basin cache (Redis → memory on startup)
    basin_cache: Dict[str, AttractorBasin]  # {concept_id: basin}

    # Level 2: Query-specific subgraph (NetworkX)
    current_subgraph: nx.MultiDiGraph  # Loaded from Neo4j per query

    # Level 3: Scoring cache (per-query TTL)
    edge_score_cache: Dict[Tuple, EdgeScore]  # {edge: score}

    # Level 4: Performance metrics
    metrics: Dict[str, float]  # {metric_name: value}
```

---

## Data Flow

### 1. Document Processing → Basin Strengthening
```
Document Upload
    ↓
Extract Concepts [concept1, concept2, concept3]
    ↓
BasinTracker.strengthen_basins(concepts)
    ↓
For each concept:
    - Get/Create AttractorBasin
    - strength += 0.2 (cap at 2.0)
    - activation_count += 1
    - activation_history.append(now)
    ↓
For each pair (ci, cj):
    - ci.co_occurring_concepts[cj] += 1
    - cj.co_occurring_concepts[ci] += 1
    ↓
Persist to Neo4j (atomic transaction)
```

### 2. Query → Subgraph Construction
```
User Query "What is neural architecture search?"
    ↓
CLAUSESubgraphArchitect.build_subgraph(query, edge_budget=50)
    ↓
1. Embed query → query_embedding (384-dim)
2. Load k-hop subgraph from Neo4j → NetworkX graph
3. Get candidate edges from graph
    ↓
EdgeScorer.score_edges(candidates, query_embedding, graph, basins)
    ↓
For each edge (u, r, v):
    - φ_ent = entity_match(v, query_embedding)
    - φ_rel = relation_match(r, query_embedding)
    - φ_nbr = neighborhood_score(v, graph)
    - φ_deg = degree_prior(v, graph)
    - φ_basin = (basin[v].strength - 1.0) / 1.0
    - score = Σ wi × φi
    - shaped_gain = score - λ_edge × 1.0
    ↓
Sort edges by shaped_gain descending
    ↓
Select top edges while:
    - shaped_gain > 0 AND
    - count < edge_budget
    ↓
Return SubgraphResponse (selected edges, scores, metrics)
```

---

## Storage Strategy

### Redis (Basin Cache)
**Purpose**: Fast in-memory basin lookups

**Key Format**: `basin:{concept_id}`
**Value**: JSON-serialized AttractorBasin
**TTL**: 1 hour (refresh from Neo4j hourly)

```python
# Write
redis.setex(
    f"basin:{concept_id}",
    3600,  # 1 hour TTL
    json.dumps(basin.dict())
)

# Read
basin_json = redis.get(f"basin:{concept_id}")
if basin_json:
    basin = AttractorBasin(**json.loads(basin_json))
else:
    basin = load_from_neo4j(concept_id)
    redis.setex(f"basin:{concept_id}", 3600, json.dumps(basin.dict()))
```

### Neo4j (Source of Truth)
**Purpose**: Persistent graph storage with ACID guarantees

**Write Pattern**: Atomic transactions for basin updates
```python
with neo4j.session() as session:
    with session.begin_transaction() as tx:
        # Update basin strength
        tx.run("""
            MATCH (b:AttractorBasin {basin_id: $basin_id})
            SET b.strength = $strength,
                b.activation_count = $count,
                b.activation_history = b.activation_history + [$timestamp]
        """, params)

        # Update co-occurrence
        tx.run("""
            MATCH (b1:AttractorBasin {basin_id: $id1})
            MATCH (b2:AttractorBasin {basin_id: $id2})
            SET b1.co_occurring_concepts[$id2] = $count,
                b2.co_occurring_concepts[$id1] = $count
        """, params)

        tx.commit()
```

**Read Pattern**: APOC subgraph procedures for fast k-hop queries
```cypher
MATCH (start:Concept) WHERE start.concept_id IN $seed_ids
CALL apoc.path.subgraphNodes(start, {
    maxLevel: 2,
    relationshipFilter: ">"
}) YIELD node
RETURN node
```

---

## Performance Characteristics

| Operation | Target | Strategy |
|-----------|--------|----------|
| Basin lookup | <1ms | Redis cache (O(1)) |
| Basin update | <5ms | Neo4j indexed write |
| Edge scoring (1000 edges) | <5ms | NumPy vectorization |
| Subgraph construction | <500ms | APOC + NetworkX |
| Co-occurrence update | <10ms | Batch symmetric writes |

**Memory Budget**:
- Basin cache: ~10MB (10k concepts × 1KB each)
- NetworkX subgraph: ~5MB (1k nodes × 10 edges × 500B)
- Score cache: ~1MB (1k edges × 1KB score object)
- **Total**: ~16MB (well under 100MB limit, NFR-004)

---

## Migration Strategy

### Backward Compatibility
**Approach**: Lazy migration with default values

```python
def get_basin_with_defaults(concept_id: str) -> AttractorBasin:
    """Loads basin from Neo4j with backward-compatible defaults."""
    result = neo4j.run("""
        MATCH (b:AttractorBasin {basin_id: $basin_id})
        RETURN b
    """, basin_id=basin_id).single()

    if result:
        node = result['b']
        return AttractorBasin(
            basin_id=node['basin_id'],
            basin_name=node['basin_name'],
            # NEW FIELDS with defaults
            strength=node.get('strength', 1.0),  # Default: not yet strengthened
            activation_count=node.get('activation_count', 0),
            activation_history=node.get('activation_history', []),
            co_occurring_concepts=node.get('co_occurring_concepts', {}),
            # EXISTING FIELDS
            stability=node['stability'],
            depth=node['depth'],
            ...
        )
```

**Migration Timeline**:
1. **Phase 1**: Code supports both old and new basins (default values)
2. **Phase 2**: Gradual conversion as basins are accessed
3. **Phase 3**: All active basins converted (passive basins remain old format)

**No Downtime**: Old format basins work immediately with defaults
