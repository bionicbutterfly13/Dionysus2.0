# Spec 031: Write Conflict Resolution for Multi-Agent Knowledge Graph

**Status**: DRAFT
**Priority**: CRITICAL
**Dependencies**: 027 (Basin Strengthening), 028 (ThoughtSeeds), 029 (Curiosity Agents)
**Created**: 2025-10-01

## Overview

Implement robust write conflict detection, rollback, and resolution mechanisms for the agentic knowledge graph when multiple agents attempt to modify the same Neo4j nodes/relationships simultaneously.

## Problem Statement

In the agentic knowledge graph system, multiple agents operate concurrently:
- **Foreground agents**: Daedalus → Extractor → Analyst (sequential handoff)
- **Background agents**: 5 concurrent curiosity agents (asynchronous)

**Conflict Scenarios**:

### Scenario 1: Concept Extraction Collision
```
Agent A (Paper 1): Extracts "neural architecture search" → Creates concept node
Agent B (Paper 2): Extracts "neural architecture search" → Attempts to create same node
  → CONFLICT: Duplicate node creation
```

### Scenario 2: Basin Strength Update Race
```
Agent A: Strengthen basin "neural_arch" (1.6 → 1.8)
Agent B: Strengthen basin "neural_arch" (1.6 → 1.8) [both read 1.6 before updating]
  → CONFLICT: Lost update (final should be 2.0, but becomes 1.8)
```

### Scenario 3: Relationship Contradiction
```
Agent A: Creates (DARTS)-[:EXTENDS]->(gradient_optimization)
Agent B: Creates (DARTS)-[:REPLACES]->(gradient_optimization)
  → CONFLICT: Contradictory relationships
```

### Scenario 4: ThoughtSeed Basin Assignment Race
```
Agent A: Assign ThoughtSeed TS_001 to Basin_NAS
Agent B: Assign ThoughtSeed TS_001 to Basin_MetaLearning
  → CONFLICT: Multiple basin assignments (should be max 3, need atomic check)
```

## Requirements

### Functional Requirements

#### FR1: Conflict Detection
**Description**: Detect write conflicts at multiple granularities
**Acceptance Criteria**:
- [ ] Node-level conflicts: Duplicate node creation attempts
- [ ] Property-level conflicts: Concurrent updates to same property
- [ ] Relationship-level conflicts: Contradictory or duplicate relationships
- [ ] Basin-level conflicts: Concurrent basin strength updates
- [ ] Conflicts detected within <50ms of occurrence

#### FR2: Atomic Transactions with Checkpointing
**Description**: All Neo4j writes are atomic with rollback capability
**Acceptance Criteria**:
- [ ] Each agent operation wrapped in Neo4j transaction
- [ ] State checkpoints created before multi-step operations
- [ ] Rollback to checkpoint on conflict detection
- [ ] Checkpoint history retained (last 10 checkpoints per workflow)

#### FR3: Retry Policy with Exponential Backoff
**Description**: Configurable retry behavior for transient conflicts
**Acceptance Criteria**:
- [ ] Max retry attempts: 5 (configurable)
- [ ] Exponential backoff: 1s, 2s, 4s, 8s, 16s
- [ ] Retry only on transient errors (DB locks, network issues)
- [ ] No retry on logical conflicts (contradictions require resolution)

#### FR4: Conflict Resolution Strategies
**Description**: Multiple strategies for resolving conflicts
**Acceptance Criteria**:
- [ ] **MERGE**: Combine updates (e.g., basin strength: max(A, B))
- [ ] **VOTE**: QA agent decides winner based on confidence scores
- [ ] **DIALOGUE**: Agents negotiate via LangGraph dialogue node
- [ ] **VERSION**: Create versioned nodes for manual review
- [ ] **TIMESTAMP**: Last-write-wins (with audit trail)

#### FR5: Compensating Transactions
**Description**: Undo specific operations without full rollback
**Acceptance Criteria**:
- [ ] Delete created nodes (inverse of CREATE)
- [ ] Revert property updates (restore previous value)
- [ ] Remove relationships (inverse of relationship creation)
- [ ] Selective rollback: Only affected operations, not entire workflow

#### FR6: Escalation and Auditing
**Description**: Log all conflicts and escalate unresolved cases
**Acceptance Criteria**:
- [ ] All conflicts logged to Neo4j with timestamp, agents involved, resolution
- [ ] Unresolved conflicts flagged for manual review
- [ ] Metrics tracked: conflict rate, resolution time, retry success rate
- [ ] Alerts triggered for conflict rate >10%

### Non-Functional Requirements

#### NFR1: Performance
- Conflict detection: <50ms overhead per operation
- Retry with backoff: Max 31s total delay (5 retries)
- Rollback: <100ms for typical checkpoint

#### NFR2: Consistency
- Eventual consistency guaranteed within 60 seconds
- Strong consistency for critical operations (basin strength updates)
- No data loss during rollback

#### NFR3: Scalability
- Support 10+ concurrent agents without deadlock
- Handle 100+ conflicts per hour
- Checkpoint storage: <10MB per workflow

## Technical Design

### Architecture

```
Multi-Agent Write Conflict Resolution:

┌─────────────────────────────────────────────────────────┐
│  Agent A (Foreground)          Agent B (Background)     │
│  Processing Paper 1            Curiosity exploration    │
└────────┬───────────────────────────────┬────────────────┘
         │                               │
         ▼                               ▼
    ┌────────────────────────────────────────────┐
    │  LangGraph State (Shared)                  │
    │  - Current workflow state                  │
    │  - Pending updates queue                   │
    │  - Conflict flags                          │
    └────────┬───────────────────────┬───────────┘
             │                       │
             ▼                       ▼
    ┌─────────────────┐    ┌─────────────────┐
    │ Update Queue A  │    │ Update Queue B  │
    │ - Strengthen    │    │ - Strengthen    │
    │   basin "NAS"   │    │   basin "NAS"   │
    │   1.6 → 1.8     │    │   1.6 → 1.8     │
    └────────┬────────┘    └────────┬────────┘
             │                       │
             └───────────┬───────────┘
                         ▼
              ┌──────────────────────────────┐
              │  Conflict Detection Node     │
              │  - Check for overlapping ops │
              │  - Detect race conditions    │
              │  - Flag conflicts            │
              └──────────┬───────────────────┘
                         │
              ┌──────────┴──────────┐
              │                     │
         CONFLICT              NO CONFLICT
              │                     │
              ▼                     ▼
    ┌──────────────────┐   ┌──────────────────┐
    │ Conflict         │   │ Execute Updates  │
    │ Resolution Node  │   │ (Atomic Neo4j)   │
    │                  │   └──────────────────┘
    │ 1. Checkpoint    │
    │ 2. Choose        │
    │    strategy      │
    │ 3. Resolve       │
    │ 4. Retry or      │
    │    Escalate      │
    └──────────┬───────┘
               │
      ┌────────┴────────┬─────────────┐
      │                 │             │
   MERGE             VOTE          DIALOGUE
      │                 │             │
      ▼                 ▼             ▼
┌──────────┐   ┌────────────┐  ┌──────────────┐
│ Take max │   │ QA Agent   │  │ Agent A ⇄ B  │
│ strength │   │ decides    │  │ Negotiate    │
│ 2.0      │   │ winner     │  │ Consensus    │
└────┬─────┘   └─────┬──────┘  └──────┬───────┘
     │               │                │
     └───────────────┴────────────────┘
                     │
                     ▼
          ┌──────────────────────────┐
          │  Update Neo4j (Atomic)   │
          │  - Apply resolution      │
          │  - Log conflict event    │
          │  - Clear conflict flag   │
          └──────────────────────────┘
```

### Conflict Detection Implementation

#### Node-Level Conflict Detection
```python
class ConflictDetectionNode:
    """LangGraph node for detecting write conflicts"""

    def __init__(self, neo4j_driver):
        self.driver = neo4j_driver

    def detect_conflicts(self, state: DocumentProcessingState) -> DocumentProcessingState:
        """Check for conflicts in pending updates"""
        pending_updates = state.get("pending_updates", [])
        conflicts = []

        # Group updates by target entity
        entity_updates = defaultdict(list)
        for update in pending_updates:
            entity_key = (update["entity_type"], update["entity_id"])
            entity_updates[entity_key].append(update)

        # Detect conflicts
        for entity_key, updates in entity_updates.items():
            if len(updates) > 1:
                conflict = self._classify_conflict(entity_key, updates)
                if conflict:
                    conflicts.append(conflict)

        # Update state
        if conflicts:
            state["conflicts"] = conflicts
            state["conflict_detected"] = True
            logger.warning(f"🚨 Detected {len(conflicts)} write conflicts")
        else:
            state["conflict_detected"] = False

        return state

    def _classify_conflict(self, entity_key, updates):
        """Classify type of conflict"""
        entity_type, entity_id = entity_key

        # Property update conflicts
        if entity_type == "basin_strength":
            return {
                "type": "BASIN_STRENGTH_RACE",
                "entity": entity_id,
                "updates": updates,
                "resolution_strategy": "MERGE",
                "severity": "HIGH"
            }

        # Relationship conflicts
        elif entity_type == "relationship":
            rel_types = [u["relationship_type"] for u in updates]
            if len(set(rel_types)) > 1:
                return {
                    "type": "CONTRADICTORY_RELATIONSHIP",
                    "entity": entity_id,
                    "updates": updates,
                    "resolution_strategy": "VOTE",
                    "severity": "CRITICAL"
                }

        # Node creation conflicts
        elif entity_type == "concept_node":
            return {
                "type": "DUPLICATE_NODE_CREATION",
                "entity": entity_id,
                "updates": updates,
                "resolution_strategy": "MERGE",
                "severity": "MEDIUM"
            }

        return None
```

### Atomic Transactions with Checkpointing

```python
class Neo4jTransactionManager:
    """Manage atomic Neo4j transactions with checkpointing"""

    def __init__(self, driver):
        self.driver = driver
        self.checkpoints = {}

    def create_checkpoint(self, workflow_id: str) -> str:
        """Create state checkpoint before risky operations"""
        checkpoint_id = f"checkpoint_{workflow_id}_{int(time.time())}"

        with self.driver.session() as session:
            # Save current state snapshot
            snapshot = session.run("""
                MATCH (n)
                WHERE n.workflow_id = $workflow_id
                RETURN n, labels(n), properties(n)
            """, workflow_id=workflow_id).data()

            self.checkpoints[checkpoint_id] = {
                "workflow_id": workflow_id,
                "timestamp": datetime.now().isoformat(),
                "snapshot": snapshot
            }

        logger.info(f"📸 Created checkpoint: {checkpoint_id}")
        return checkpoint_id

    def rollback_to_checkpoint(self, checkpoint_id: str):
        """Rollback to previous checkpoint"""
        if checkpoint_id not in self.checkpoints:
            raise ValueError(f"Checkpoint {checkpoint_id} not found")

        checkpoint = self.checkpoints[checkpoint_id]
        snapshot = checkpoint["snapshot"]

        with self.driver.session() as session:
            # Delete all nodes created after checkpoint
            session.run("""
                MATCH (n)
                WHERE n.workflow_id = $workflow_id
                  AND n.created_at > $checkpoint_time
                DELETE n
            """, workflow_id=checkpoint["workflow_id"],
                checkpoint_time=checkpoint["timestamp"])

            # Restore node properties from snapshot
            for node_data in snapshot:
                session.run("""
                    MATCH (n)
                    WHERE id(n) = $node_id
                    SET n = $properties
                """, node_id=node_data["n"].id,
                    properties=node_data["properties(n)"])

        logger.info(f"⏪ Rolled back to checkpoint: {checkpoint_id}")

    @contextmanager
    def atomic_transaction(self, workflow_id: str):
        """Context manager for atomic Neo4j operations with auto-rollback"""
        checkpoint_id = self.create_checkpoint(workflow_id)

        try:
            with self.driver.session() as session:
                tx = session.begin_transaction()
                try:
                    yield tx
                    tx.commit()
                    logger.info("✅ Transaction committed successfully")
                except Exception as e:
                    tx.rollback()
                    self.rollback_to_checkpoint(checkpoint_id)
                    logger.error(f"❌ Transaction failed, rolled back: {e}")
                    raise
        finally:
            # Keep last 10 checkpoints
            if len(self.checkpoints) > 10:
                oldest = min(self.checkpoints.keys())
                del self.checkpoints[oldest]
```

### Retry Policy with Exponential Backoff

```python
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

class RetryableNeo4jOperation:
    """Neo4j operations with configurable retry policy"""

    def __init__(self, driver, max_attempts=5):
        self.driver = driver
        self.max_attempts = max_attempts

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=16),
        retry=retry_if_exception_type((TransientError, ServiceUnavailable)),
        reraise=True
    )
    def execute_with_retry(self, operation_fn, *args, **kwargs):
        """Execute Neo4j operation with retry logic"""
        try:
            with self.driver.session() as session:
                result = operation_fn(session, *args, **kwargs)
                logger.info(f"✅ Operation succeeded")
                return result

        except TransientError as e:
            logger.warning(f"⚠️  Transient error, will retry: {e}")
            raise

        except ServiceUnavailable as e:
            logger.warning(f"⚠️  Service unavailable, will retry: {e}")
            raise

        except Exception as e:
            # Non-retryable errors
            logger.error(f"❌ Non-retryable error: {e}")
            raise

# Usage
retryable_op = RetryableNeo4jOperation(driver)

def strengthen_basin(session, basin_id, increment):
    """Operation to strengthen basin (may conflict)"""
    result = session.run("""
        MATCH (b:AttractorBasin {basin_id: $basin_id})
        SET b.strength = b.strength + $increment
        RETURN b.strength as new_strength
    """, basin_id=basin_id, increment=increment)
    return result.single()["new_strength"]

# Execute with retry
new_strength = retryable_op.execute_with_retry(
    strengthen_basin,
    basin_id="basin_nas_12345",
    increment=0.2
)
```

### Conflict Resolution Strategies

#### Strategy 1: MERGE (Basin Strength)
```python
class MergeResolutionStrategy:
    """Merge conflicting updates by taking max/min/sum"""

    def resolve(self, conflict):
        """Resolve basin strength race condition by taking max"""
        updates = conflict["updates"]

        # Extract all proposed strengths
        proposed_strengths = [u["new_value"] for u in updates]
        merged_strength = max(proposed_strengths)

        logger.info(f"🔀 MERGE: Basin strength resolved to max({proposed_strengths}) = {merged_strength}")

        return {
            "resolution": "MERGE",
            "final_value": merged_strength,
            "rationale": "Took maximum of conflicting updates to preserve strongest signal"
        }
```

#### Strategy 2: VOTE (Contradictory Relationships)
```python
class VoteResolutionStrategy:
    """Use QA agent to vote on conflicting relationships"""

    def __init__(self, qa_agent):
        self.qa_agent = qa_agent

    def resolve(self, conflict):
        """QA agent votes on which relationship is correct"""
        updates = conflict["updates"]

        # Present conflict to QA agent
        vote_result = self.qa_agent.vote_on_relationship(
            source=conflict["entity"]["source"],
            target=conflict["entity"]["target"],
            options=[
                {"type": u["relationship_type"], "confidence": u["confidence"]}
                for u in updates
            ]
        )

        logger.info(f"🗳️  VOTE: QA agent chose {vote_result['winner']}")

        return {
            "resolution": "VOTE",
            "final_value": vote_result["winner"],
            "rationale": vote_result["reasoning"]
        }
```

#### Strategy 3: DIALOGUE (Agent Negotiation)
```python
class DialogueResolutionStrategy:
    """Agents negotiate via LangGraph dialogue node"""

    def resolve(self, conflict):
        """Spawn dialogue between conflicting agents"""
        updates = conflict["updates"]
        agent_a_id = updates[0]["agent_id"]
        agent_b_id = updates[1]["agent_id"]

        # Create dialogue state
        dialogue_state = {
            "participants": [agent_a_id, agent_b_id],
            "topic": conflict["entity"],
            "positions": [u["rationale"] for u in updates],
            "max_turns": 5,
            "consensus_threshold": 0.8
        }

        # Run dialogue workflow
        dialogue_result = self._run_dialogue(dialogue_state)

        logger.info(f"💬 DIALOGUE: Consensus reached after {dialogue_result['turns']} turns")

        return {
            "resolution": "DIALOGUE",
            "final_value": dialogue_result["consensus"],
            "rationale": dialogue_result["reasoning"]
        }

    def _run_dialogue(self, dialogue_state):
        """Execute dialogue workflow between agents"""
        # LangGraph dialogue node implementation
        # Agents exchange reasoning until consensus or timeout
        pass
```

### Compensating Transactions

```python
class CompensatingTransaction:
    """Undo specific operations without full rollback"""

    def __init__(self, driver):
        self.driver = driver

    def compensate_node_creation(self, node_id):
        """Delete created node (inverse of CREATE)"""
        with self.driver.session() as session:
            session.run("""
                MATCH (n)
                WHERE id(n) = $node_id
                DELETE n
            """, node_id=node_id)

        logger.info(f"↩️  Compensated: Deleted node {node_id}")

    def compensate_property_update(self, node_id, property_name, old_value):
        """Revert property to previous value"""
        with self.driver.session() as session:
            session.run(f"""
                MATCH (n)
                WHERE id(n) = $node_id
                SET n.{property_name} = $old_value
            """, node_id=node_id, old_value=old_value)

        logger.info(f"↩️  Compensated: Reverted {property_name} to {old_value}")

    def compensate_relationship_creation(self, rel_id):
        """Delete created relationship"""
        with self.driver.session() as session:
            session.run("""
                MATCH ()-[r]->()
                WHERE id(r) = $rel_id
                DELETE r
            """, rel_id=rel_id)

        logger.info(f"↩️  Compensated: Deleted relationship {rel_id}")
```

### LangGraph Integration

```python
class ConflictAwareDocumentProcessingGraph(DocumentProcessingGraph):
    """Extended graph with conflict handling"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tx_manager = Neo4jTransactionManager(self.neo4j.driver)
        self.conflict_resolver = ConflictResolver()

    def _build_graph(self):
        """Build LangGraph with conflict detection/resolution nodes"""
        workflow = StateGraph(DocumentProcessingState)

        # Existing nodes
        workflow.add_node("extract_concepts", self._concept_extraction_node)
        workflow.add_node("extract_relationships", self._relationship_extraction_node)
        workflow.add_node("generate_thoughtseeds", self._thoughtseed_generation_node)

        # NEW: Conflict handling nodes
        workflow.add_node("detect_conflicts", self._conflict_detection_node)
        workflow.add_node("resolve_conflicts", self._conflict_resolution_node)
        workflow.add_node("execute_updates", self._atomic_update_node)

        # Edges with conflict handling
        workflow.add_edge("generate_thoughtseeds", "detect_conflicts")
        workflow.add_conditional_edges(
            "detect_conflicts",
            lambda state: "resolve" if state.get("conflict_detected") else "execute",
            {
                "resolve": "resolve_conflicts",
                "execute": "execute_updates"
            }
        )
        workflow.add_edge("resolve_conflicts", "execute_updates")

        return workflow.compile()

    def _conflict_detection_node(self, state):
        """Detect conflicts in pending updates"""
        detector = ConflictDetectionNode(self.neo4j.driver)
        return detector.detect_conflicts(state)

    def _conflict_resolution_node(self, state):
        """Resolve detected conflicts"""
        conflicts = state.get("conflicts", [])
        resolutions = []

        for conflict in conflicts:
            strategy = self._choose_resolution_strategy(conflict)
            resolution = strategy.resolve(conflict)
            resolutions.append(resolution)

        state["conflict_resolutions"] = resolutions
        state["conflicts"] = []  # Clear after resolution
        return state

    def _atomic_update_node(self, state):
        """Execute updates atomically with checkpoint"""
        workflow_id = state["metadata"]["workflow_id"]

        with self.tx_manager.atomic_transaction(workflow_id) as tx:
            # Apply all resolved updates
            for update in state.get("pending_updates", []):
                self._apply_update(tx, update)

            # Clear pending updates
            state["pending_updates"] = []

        return state

    def _choose_resolution_strategy(self, conflict):
        """Choose appropriate resolution strategy"""
        strategy_map = {
            "BASIN_STRENGTH_RACE": MergeResolutionStrategy(),
            "CONTRADICTORY_RELATIONSHIP": VoteResolutionStrategy(self.qa_agent),
            "DUPLICATE_NODE_CREATION": MergeResolutionStrategy(),
        }
        return strategy_map.get(conflict["type"], MergeResolutionStrategy())
```

## Test Strategy

### Unit Tests

```python
def test_conflict_detection_basin_strength_race():
    """Test detection of concurrent basin strength updates"""
    detector = ConflictDetectionNode(neo4j_driver)

    state = {
        "pending_updates": [
            {
                "agent_id": "agent_a",
                "entity_type": "basin_strength",
                "entity_id": "basin_nas",
                "old_value": 1.6,
                "new_value": 1.8
            },
            {
                "agent_id": "agent_b",
                "entity_type": "basin_strength",
                "entity_id": "basin_nas",
                "old_value": 1.6,
                "new_value": 1.8
            }
        ]
    }

    result = detector.detect_conflicts(state)

    assert result["conflict_detected"] is True
    assert len(result["conflicts"]) == 1
    assert result["conflicts"][0]["type"] == "BASIN_STRENGTH_RACE"

def test_merge_resolution_strategy():
    """Test MERGE strategy takes maximum value"""
    strategy = MergeResolutionStrategy()

    conflict = {
        "type": "BASIN_STRENGTH_RACE",
        "updates": [
            {"new_value": 1.8},
            {"new_value": 1.7},
            {"new_value": 1.9}
        ]
    }

    resolution = strategy.resolve(conflict)

    assert resolution["resolution"] == "MERGE"
    assert resolution["final_value"] == 1.9

def test_atomic_transaction_rollback():
    """Test rollback on transaction failure"""
    tx_manager = Neo4jTransactionManager(driver)

    workflow_id = "test_workflow"
    checkpoint_id = tx_manager.create_checkpoint(workflow_id)

    try:
        with tx_manager.atomic_transaction(workflow_id) as tx:
            # Create node
            tx.run("CREATE (n:TestNode {name: 'test'})")
            # Simulate error
            raise Exception("Simulated error")
    except:
        pass

    # Verify rollback occurred
    with driver.session() as session:
        result = session.run("MATCH (n:TestNode {name: 'test'}) RETURN count(n) as count")
        assert result.single()["count"] == 0

def test_retry_with_exponential_backoff():
    """Test retry policy with backoff"""
    retryable_op = RetryableNeo4jOperation(driver, max_attempts=3)

    call_count = 0

    def flaky_operation(session):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise TransientError("Temporary failure")
        return "success"

    result = retryable_op.execute_with_retry(flaky_operation)

    assert result == "success"
    assert call_count == 3  # Failed twice, succeeded on third attempt
```

### Integration Tests

```python
def test_concurrent_agent_basin_updates():
    """Test two agents updating same basin concurrently"""
    graph = ConflictAwareDocumentProcessingGraph()

    # Simulate two agents processing different papers with same concept
    import threading

    def agent_a():
        graph.process_document(
            content=paper_with_nas_concept,
            filename="paper_a.pdf"
        )

    def agent_b():
        graph.process_document(
            content=paper_with_nas_concept,
            filename="paper_b.pdf"
        )

    thread_a = threading.Thread(target=agent_a)
    thread_b = threading.Thread(target=agent_b)

    thread_a.start()
    thread_b.start()
    thread_a.join()
    thread_b.join()

    # Verify final basin strength is correct (should be 2.0, not 1.8)
    with driver.session() as session:
        result = session.run("""
            MATCH (b:AttractorBasin {center_concept: 'neural architecture search'})
            RETURN b.strength as strength
        """)
        final_strength = result.single()["strength"]

    assert final_strength == 2.0, "Conflict resolution should merge to max strength"
```

## Implementation Plan

### Phase 1: Conflict Detection (2-3 hours)
1. Implement `ConflictDetectionNode`
2. Add conflict classification logic
3. Test detection for all conflict types

### Phase 2: Transaction Management (3-4 hours)
1. Implement `Neo4jTransactionManager`
2. Add checkpointing mechanism
3. Implement rollback logic
4. Test atomic transactions

### Phase 3: Retry Logic (2-3 hours)
1. Implement `RetryableNeo4jOperation`
2. Add exponential backoff
3. Test retry behavior

### Phase 4: Resolution Strategies (4-5 hours)
1. Implement MERGE strategy
2. Implement VOTE strategy
3. Implement DIALOGUE strategy
4. Test each strategy

### Phase 5: LangGraph Integration (3-4 hours)
1. Extend `DocumentProcessingGraph` with conflict nodes
2. Wire conflict detection → resolution → execution
3. Test end-to-end workflow

### Phase 6: Testing & Documentation (2-3 hours)
1. Integration tests
2. Concurrent agent tests
3. Documentation
4. Performance validation

**Total Estimated Time**: 16-22 hours

## Success Criteria

- [ ] All conflict types detected within <50ms
- [ ] Atomic transactions with rollback functional
- [ ] Retry logic with exponential backoff working
- [ ] All resolution strategies implemented (MERGE, VOTE, DIALOGUE)
- [ ] Compensating transactions functional
- [ ] Concurrent agent updates resolved correctly
- [ ] Zero data loss during conflict resolution
- [ ] All tests passing (unit + integration)

## References

- LangGraph Conflict Handling Documentation
- Neo4j Transaction Management
- Tenacity Retry Library
- Spec 027: Basin Frequency Strengthening
- Spec 028: ThoughtSeed Generation
- Spec 029: Curiosity-Driven Background Agents
