# Spec 031: Write Conflict Resolution - Summary

**Created**: 2025-10-01
**Status**: ✅ COMPLETE - Ready for implementation

## What This Spec Adds

Following your research on **LangGraph write conflict handling**, I've created a comprehensive specification for managing concurrent agent write conflicts in the agentic knowledge graph system.

## The Problem

With multiple agents running concurrently:
- **Foreground**: Sequential pipeline (Daedalus → Extractor → Analyst)
- **Background**: 5 concurrent curiosity agents

**Conflicts occur when**:
1. **Basin Strength Race**: Two agents strengthen same basin simultaneously
2. **Duplicate Node Creation**: Two agents create same concept node
3. **Contradictory Relationships**: Agents create conflicting relationship types
4. **ThoughtSeed Assignment**: Multiple basin assignments for same ThoughtSeed

## The Solution (Following LangGraph Best Practices)

### 1. Conflict Detection (<50ms overhead)
```python
ConflictDetectionNode:
  - Groups pending updates by target entity
  - Classifies conflict type (BASIN_RACE, DUPLICATE_NODE, etc.)
  - Sets conflict flag in LangGraph state
```

### 2. Atomic Transactions with Checkpointing
```python
Neo4jTransactionManager:
  - Creates state checkpoint before risky operations
  - Wraps all writes in Neo4j transactions
  - Rolls back to checkpoint on failure
  - Keeps last 10 checkpoints per workflow
```

### 3. Retry Policy with Exponential Backoff
```python
RetryableNeo4jOperation:
  - Max 5 retry attempts
  - Exponential backoff: 1s, 2s, 4s, 8s, 16s
  - Only retries transient errors (DB locks, network)
  - No retry on logical conflicts (requires resolution)
```

### 4. Resolution Strategies

**MERGE Strategy** (Basin Strength Race):
```python
# Two agents both strengthen basin from 1.6 → 1.8
# Resolution: Take max(1.8, 1.8) = 1.8
# But track that TWO updates occurred → final should be 2.0
```

**VOTE Strategy** (Contradictory Relationships):
```python
# Agent A: (DARTS)-[:EXTENDS]->(gradient_opt)
# Agent B: (DARTS)-[:REPLACES]->(gradient_opt)
# Resolution: QA agent votes based on confidence scores
```

**DIALOGUE Strategy** (Complex Conflicts):
```python
# Agents A and B negotiate via LangGraph dialogue node
# Exchange reasoning until consensus or timeout (max 5 turns)
```

**VERSION Strategy** (Unresolvable):
```python
# Create versioned nodes for manual review
# Flag for human intervention
```

### 5. Compensating Transactions

Instead of full rollback, undo specific operations:
```python
CompensatingTransaction:
  - Delete created node (inverse of CREATE)
  - Revert property update (restore old value)
  - Remove relationship (inverse of relationship creation)
```

## LangGraph Integration

```
Workflow:
  extract_concepts
       ↓
  extract_relationships
       ↓
  generate_thoughtseeds
       ↓
  [NEW] detect_conflicts ← Checks for overlapping updates
       ↓
  ┌────┴────┐
  │         │
CONFLICT   NO CONFLICT
  │         │
  ↓         ↓
resolve   execute_updates (atomic)
conflicts
  ↓
execute_updates (atomic)
```

## Key Features

### ✅ Atomic Operations
- All Neo4j writes wrapped in transactions
- Auto-rollback on failure
- Checkpoint-based recovery

### ✅ Conflict Detection
- Node-level (duplicate creation)
- Property-level (concurrent updates)
- Relationship-level (contradictions)
- Basin-level (strength races)

### ✅ Smart Resolution
- MERGE: Combine updates (take max/min/sum)
- VOTE: QA agent decides winner
- DIALOGUE: Agents negotiate consensus
- VERSION: Create alternate versions

### ✅ Retry Logic
- Exponential backoff (1s → 16s)
- Only transient errors retried
- Max 5 attempts before escalation

### ✅ Auditing
- All conflicts logged to Neo4j
- Resolution rationale recorded
- Metrics tracked (conflict rate, resolution time)

## Test Strategy

### Unit Tests
```python
test_conflict_detection_basin_strength_race()
test_merge_resolution_strategy()
test_atomic_transaction_rollback()
test_retry_with_exponential_backoff()
```

### Integration Tests
```python
test_concurrent_agent_basin_updates()
  - Spawn 2 agents processing same concept
  - Verify conflict detected and resolved
  - Confirm final basin strength correct (2.0, not 1.8)
```

## Implementation Timeline

**Total**: 16-22 hours

- Phase 1: Conflict Detection (2-3 hours)
- Phase 2: Transaction Management (3-4 hours)
- Phase 3: Retry Logic (2-3 hours)
- Phase 4: Resolution Strategies (4-5 hours)
- Phase 5: LangGraph Integration (3-4 hours)
- Phase 6: Testing & Docs (2-3 hours)

## Success Criteria

- [ ] Conflict detection <50ms overhead
- [ ] Atomic transactions functional
- [ ] Retry with exponential backoff working
- [ ] All resolution strategies implemented
- [ ] Zero data loss during resolution
- [ ] Concurrent agent tests passing
- [ ] Metrics: conflict rate, resolution time

## How This Fits Into The Roadmap

This spec is **CRITICAL INFRASTRUCTURE** for Specs 027-029:

**Spec 027** (Basin Strengthening): Needs conflict resolution for concurrent basin updates
**Spec 028** (ThoughtSeeds): Needs atomic operations for ThoughtSeed-Basin assignments
**Spec 029** (Curiosity Agents): Needs conflict handling for 5 concurrent background agents

**Recommendation**: Implement Spec 031 **FIRST** before or alongside Spec 027, since basin strengthening immediately creates race conditions.

## Updated Implementation Order

### Option 1: Infrastructure First (Safest)
1. **Week 1**: Spec 031 (Conflict Resolution) - Foundation
2. **Week 2**: Spec 027 (Basin Strengthening) - Builds on conflict handling
3. **Week 3**: Spec 028 (ThoughtSeeds) - Uses atomic transactions
4. **Week 4**: Spec 029 (Curiosity Agents) - Concurrent agents safe
5. **Week 5**: Spec 030 (Visual Interface) - Everything working

### Option 2: Parallel Development
1. **Developer 1**: Spec 031 (Conflict) + Spec 027 (Basin)
2. **Developer 2**: Spec 030 (Visual Interface)
3. **Week 3**: Integrate, add Spec 028 (ThoughtSeeds)
4. **Week 4**: Spec 029 (Curiosity Agents)
5. **Week 5**: Testing

## Files Created

```
specs/031-write-conflict-resolution/spec.md (35 KB)
SPEC_031_CONFLICT_RESOLUTION_SUMMARY.md (this file)
```

## References

- [Spec 031 Full Specification](specs/031-write-conflict-resolution/spec.md)
- LangGraph Conflict Handling Documentation
- Neo4j Transaction Management
- Tenacity Retry Library

---

**Status**: ✅ Spec complete, ready for implementation
**Priority**: CRITICAL (implement before or with Spec 027)
**Estimated Time**: 16-22 hours
**Dependencies**: Neo4j, LangGraph, Tenacity

**Your conflict resolution research was excellent - this spec implements all the patterns you described: atomic transactions, checkpointing, retry with backoff, and multiple resolution strategies.**
