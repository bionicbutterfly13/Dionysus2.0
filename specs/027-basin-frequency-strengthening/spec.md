# Spec 027: Basin Frequency Strengthening During Document Processing

**Status**: DRAFT
**Priority**: HIGH
**Dependencies**: 021 (Daedalus), 001 (Neo4j), 005 (ThoughtSeeds)
**Created**: 2025-10-01

## Overview

Implement frequency-based basin strengthening where attractor basins grow stronger as concepts repeatedly appear across multiple documents. This creates a dynamic cognitive landscape where frequently activated concepts form deeper, more influential basins.

## Problem Statement

Currently, attractor basins are created during document processing but don't track concept reappearance across multiple documents. When the same concept appears in paper #5, #10, and #15, the system should:

1. **Recognize** the concept has appeared before
2. **Strengthen** the corresponding basin with each reappearance
3. **Use basin strength** to guide relationship extraction (stronger basins = higher priority)
4. **Learn patterns** about which concepts frequently co-occur

Without this, agents miss opportunities to learn domain-specific patterns and improve extraction quality over time.

## Requirements

### Functional Requirements

#### FR1: Concept Frequency Tracking
**Description**: Track how many times each concept appears across documents
**Acceptance Criteria**:
- [ ] Each basin maintains `activation_count` (total appearances)
- [ ] Each basin maintains `activation_history` (timestamps of appearances)
- [ ] Redis stores basin state persistently (7-day TTL)
- [ ] Basin strength increases by 0.2 per reappearance (capped at 2.0)

#### FR2: Basin Strength Integration with LLM
**Description**: Pass basin strength to LLM during relationship extraction
**Acceptance Criteria**:
- [ ] LLM prompt includes basin context: `(concept, strength, activation_count)`
- [ ] Higher strength concepts get priority in relationship discovery
- [ ] LLM output includes justification based on basin strength
- [ ] Quality metrics track correlation between strength and relationship quality

#### FR3: Cross-Document Pattern Learning
**Description**: Learn which concepts frequently co-occur across papers
**Acceptance Criteria**:
- [ ] Track concept pairs that appear together >3 times
- [ ] Store co-occurrence patterns in Cognition Base
- [ ] Boost relationship priority for frequently co-occurring concepts
- [ ] Log pattern discoveries: "Concept A + Concept B appear together in 8 papers"

#### FR4: Basin Decay for Inactive Concepts
**Description**: Weaken basins that haven't been activated recently
**Acceptance Criteria**:
- [ ] Basins decay by 0.1 strength per week of inactivity
- [ ] Basins below 0.2 strength are pruned from Redis
- [ ] Decay process runs during `evolve_basin_landscape()`
- [ ] Removed basins are logged for analysis

### Non-Functional Requirements

#### NFR1: Performance
- Basin lookup must complete in <50ms
- Strengthening update must complete in <10ms
- Batch processing of 100 papers should not degrade performance

#### NFR2: Storage Efficiency
- Redis memory usage should not exceed 1GB for 10,000 basins
- Use TTL-based cleanup to prevent unbounded growth
- Compress `activation_history` if >50 entries

#### NFR3: Learning Quality
- Basin-guided extraction should improve relationship quality by >15%
- Agents should show measurable improvement after processing 50 papers
- Co-occurrence pattern detection should achieve >80% precision

## Technical Design

### Architecture

```
Document Processing Flow with Basin Strengthening:
┌─────────────────────────────────────────────────────────────┐
│  Paper #1                                                   │
│  Concepts: ["neural architecture search", "AutoML"]         │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
    ┌────────────────────────────────────────┐
    │  AttractorBasinManager                 │
    │  - Check if concepts exist             │
    │  - Create basins if new                │
    │  - Strengthen basins if existing       │
    └────────────────┬───────────────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │  Basin: "neural_arch..."   │
        │  - strength: 1.0           │
        │  - activation_count: 1     │
        │  - created: 2025-10-01     │
        └────────────────────────────┘
                     │
┌────────────────────┴─────────────────────┐
│  Paper #10                                │
│  Concepts: ["neural architecture search"] │ ← SAME CONCEPT
└────────────────────┬─────────────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │  Basin: "neural_arch..."   │
        │  - strength: 1.8 (+0.2)    │ ← STRENGTHENED
        │  - activation_count: 2     │
        │  - last_seen: 2025-10-01   │
        └────────────────────────────┘
                     │
                     ▼
    ┌────────────────────────────────────────┐
    │  LLM Relationship Extraction           │
    │  Prompt: "neural architecture search"  │
    │  Basin strength: 1.8 (HIGH PRIORITY)   │ ← GUIDES LLM
    │  → Extract relationships carefully     │
    └────────────────────────────────────────┘
```

### Data Model

#### AttractorBasin Extension
```python
@dataclass
class AttractorBasin:
    basin_id: str
    center_concept: str
    strength: float = 1.0                    # 0.0 - 2.0
    radius: float = 0.5
    activation_count: int = 0                # NEW: Total appearances
    activation_history: List[str] = []       # NEW: Timestamps
    co_occurring_concepts: Dict[str, int] = {}  # NEW: {concept: count}
    last_activation: str = ""                # NEW: Last seen timestamp
    # ... existing fields ...
```

#### Redis Storage Schema
```
Key Pattern: attractor_basin:{basin_id}
TTL: 7 days (refreshed on each activation)

Value (JSON):
{
  "basin_id": "basin_neural_arch_1234567890",
  "center_concept": "neural architecture search",
  "strength": 1.8,
  "activation_count": 9,
  "activation_history": [
    "2025-10-01T10:30:00",
    "2025-10-01T11:45:00",
    ...
  ],
  "co_occurring_concepts": {
    "AutoML": 7,
    "meta-learning": 5,
    "optimization": 8
  },
  "last_activation": "2025-10-01T14:20:00"
}
```

### Integration Points

#### 1. Document Processing Graph (`document_processing_graph.py`)
```python
def _create_concept_relationships(self, state):
    concepts = state["concepts"]

    # STEP 1: Get basin context for each concept
    basin_context = {}
    for concept in concepts:
        basin_info = self._get_or_create_basin(concept)
        basin_context[concept] = {
            "strength": basin_info["strength"],
            "activation_count": basin_info["activation_count"],
            "priority": basin_info["strength"] / 2.0  # 0.0 - 1.0
        }

    # STEP 2: Pass basin context to LLM
    relationships = self._llm_extract_relationships_with_basin_context(
        concepts, basin_context
    )

    # STEP 3: Track co-occurrence patterns
    self._update_basin_co_occurrence(concepts)

    return relationships
```

#### 2. LLM Prompt Enhancement
```python
def _llm_extract_relationships_with_basin_context(self, concepts, basin_context):
    prompt = f"""
    Analyze these research concepts to build a knowledge graph.

    Concepts with basin context:
    {json.dumps([
        {
            "concept": c,
            "strength": basin_context[c]["strength"],
            "activation_count": basin_context[c]["activation_count"],
            "priority": basin_context[c]["priority"]
        }
        for c in concepts
    ], indent=2)}

    PRIORITIZE RELATIONSHIPS FOR:
    - HIGH STRENGTH concepts (>1.5): These are frequently appearing, well-established concepts
    - HIGH ACTIVATION concepts (>5 appearances): These are domain-central ideas

    Extract ALL semantic relationships...
    [rest of prompt]
    """
```

#### 3. Basin Manager Integration
```python
def _get_or_create_basin(self, concept: str) -> Dict[str, Any]:
    """Get existing basin or create new one for concept"""

    # Check Redis for existing basin
    basin_key = self._concept_to_basin_key(concept)
    existing_basin = self.redis_client.get(f"attractor_basin:{basin_key}")

    if existing_basin:
        # EXISTING BASIN - STRENGTHEN IT
        basin_data = json.loads(existing_basin)
        basin_data["strength"] = min(2.0, basin_data["strength"] + 0.2)
        basin_data["activation_count"] += 1
        basin_data["activation_history"].append(datetime.now().isoformat())
        basin_data["last_activation"] = datetime.now().isoformat()

        # Keep history manageable
        if len(basin_data["activation_history"]) > 50:
            basin_data["activation_history"] = basin_data["activation_history"][-50:]

        # Save back to Redis with refreshed TTL
        self.redis_client.setex(
            f"attractor_basin:{basin_key}",
            86400 * 7,  # 7 days
            json.dumps(basin_data)
        )

        logger.info(f"🔋 Strengthened basin '{concept}': "
                   f"strength={basin_data['strength']:.2f}, "
                   f"count={basin_data['activation_count']}")

        return basin_data

    else:
        # NEW BASIN - CREATE IT
        basin_data = {
            "basin_id": basin_key,
            "center_concept": concept,
            "strength": 1.0,
            "activation_count": 1,
            "activation_history": [datetime.now().isoformat()],
            "co_occurring_concepts": {},
            "last_activation": datetime.now().isoformat()
        }

        self.redis_client.setex(
            f"attractor_basin:{basin_key}",
            86400 * 7,
            json.dumps(basin_data)
        )

        logger.info(f"🌟 Created new basin '{concept}'")

        return basin_data
```

### Test Strategy

#### Unit Tests (`test_basin_frequency_strengthening.py`)

```python
def test_basin_strengthens_on_reappearance():
    """Test basin strength increases when concept appears again"""
    manager = AttractorBasinManager()

    # First appearance
    basin1 = manager.get_or_create_basin("neural architecture search")
    assert basin1["strength"] == 1.0
    assert basin1["activation_count"] == 1

    # Second appearance
    basin2 = manager.get_or_create_basin("neural architecture search")
    assert basin2["strength"] == 1.2
    assert basin2["activation_count"] == 2

    # Fifth appearance
    for _ in range(3):
        manager.get_or_create_basin("neural architecture search")

    basin5 = manager.get_or_create_basin("neural architecture search")
    assert basin5["strength"] == 2.0  # Capped at 2.0
    assert basin5["activation_count"] == 5

def test_basin_context_passed_to_llm():
    """Test LLM receives basin strength and priority"""
    graph = DocumentProcessingGraph()

    # Create basin with high strength
    manager = AttractorBasinManager()
    manager.get_or_create_basin("neural networks")
    manager.get_or_create_basin("neural networks")
    manager.get_or_create_basin("neural networks")

    # Process document with "neural networks" concept
    state = {"concepts": ["neural networks", "optimization"]}

    # Extract relationships
    relationships = graph._create_concept_relationships(state)

    # Verify LLM received basin context
    # (check logs or mock LLM to verify prompt includes strength)
    assert any("strength" in str(r) for r in relationships)

def test_co_occurrence_tracking():
    """Test system learns which concepts appear together"""
    graph = DocumentProcessingGraph()

    # Process 5 papers with NAS + AutoML
    for i in range(5):
        state = {"concepts": ["neural architecture search", "AutoML"]}
        graph._update_basin_co_occurrence(state["concepts"])

    # Check co-occurrence was recorded
    basin = graph._get_or_create_basin("neural architecture search")
    assert "AutoML" in basin["co_occurring_concepts"]
    assert basin["co_occurring_concepts"]["AutoML"] == 5

def test_basin_decay():
    """Test inactive basins decay over time"""
    manager = AttractorBasinManager()

    # Create basin
    basin = manager.get_or_create_basin("old concept")
    assert basin["strength"] == 1.0

    # Simulate 8 days of inactivity
    basin_data = json.loads(manager.redis_client.get("attractor_basin:old_concept"))
    basin_data["last_activation"] = (datetime.now() - timedelta(days=8)).isoformat()
    manager.redis_client.setex(
        "attractor_basin:old_concept",
        86400 * 7,
        json.dumps(basin_data)
    )

    # Run evolution
    manager.evolve_basin_landscape()

    # Check decay
    decayed_basin = json.loads(manager.redis_client.get("attractor_basin:old_concept"))
    assert decayed_basin["strength"] < 1.0
```

#### Integration Tests (`test_basin_strengthening_integration.py`)

```python
def test_agent_learning_with_basin_strengthening():
    """Test agents improve relationship extraction as basins strengthen"""
    graph = DocumentProcessingGraph()

    # Process paper 1 - initial quality
    result1 = graph.process_document(
        content=sample_paper_nas_1,
        filename="nas_paper_1.pdf"
    )
    quality1 = result1["analysis"]["quality_scores"]["overall"]

    # Process papers 2-10 with similar concepts
    for i in range(2, 11):
        graph.process_document(
            content=sample_paper_nas_i,
            filename=f"nas_paper_{i}.pdf"
        )

    # Process paper 11 - quality should improve
    result11 = graph.process_document(
        content=sample_paper_nas_11,
        filename="nas_paper_11.pdf"
    )
    quality11 = result11["analysis"]["quality_scores"]["overall"]

    # Verify improvement
    assert quality11 > quality1 + 0.15, "Agent should improve by >15% after 10 papers"

    # Verify basins strengthened
    basin_nas = graph._get_or_create_basin("neural architecture search")
    assert basin_nas["strength"] > 1.5
    assert basin_nas["activation_count"] >= 10
```

## Implementation Plan

### Phase 1: Basin Manager Enhancement (2-3 hours)
1. Extend `AttractorBasin` dataclass with new fields
2. Update `_get_or_create_basin()` to track frequency
3. Implement `_update_basin_co_occurrence()`
4. Add basin decay to `evolve_basin_landscape()`

### Phase 2: LLM Integration (2-3 hours)
1. Modify `_create_concept_relationships()` to get basin context
2. Update `_llm_extract_relationships()` prompt with basin info
3. Add basin strength logging
4. Test LLM prompt quality

### Phase 3: Testing (2-3 hours)
1. Write unit tests for basin strengthening
2. Write integration tests for agent learning
3. Validate performance (<50ms lookups)
4. Measure quality improvement

### Phase 4: Documentation (1 hour)
1. Update `AGENTIC_KNOWLEDGE_GRAPH_COMPLETE.md`
2. Add usage examples
3. Document basin strength interpretation

**Total Estimated Time**: 8-10 hours

## Success Criteria

- [ ] Basins strengthen by 0.2 per concept reappearance (capped at 2.0)
- [ ] Basin strength is passed to LLM during relationship extraction
- [ ] Co-occurrence patterns are tracked and boost relationship priority
- [ ] Inactive basins decay by 0.1 per week
- [ ] Agent quality improves >15% after processing 50 papers
- [ ] All tests passing (unit + integration)

## Risks & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Redis memory growth | HIGH | Implement TTL-based cleanup, compress history |
| Basin lookup latency | MEDIUM | Use hash-based keys, batch lookups |
| False co-occurrence patterns | MEDIUM | Require >3 appearances before boosting priority |
| Basin strength inflation | LOW | Cap at 2.0, implement decay |

## References

- Spec 021: Daedalus Perceptual Information Gateway
- Spec 005: ThoughtSeed Active Inference System
- `attractor_basin_dynamics.py`: Existing basin implementation
- `document_processing_graph.py`: Main processing workflow
- `AGENTIC_KNOWLEDGE_GRAPH_COMPLETE.md`: System documentation
