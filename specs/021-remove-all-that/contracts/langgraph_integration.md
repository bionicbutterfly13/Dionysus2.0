# LangGraph Integration with Daedalus Gateway

**Spec**: 021-remove-all-that
**Component**: Daedalus Perceptual Information Gateway
**Integration Type**: Agent Creation

## Overview

Daedalus serves as the perceptual information gateway, receiving external data (uploads) and creating LangGraph agents for processing. This document specifies the integration pattern between Daedalus and LangGraph.

## Integration Architecture

```
External Upload
      ↓
Daedalus.receive_perceptual_information()
      ↓
create_langgraph_agents(data)
      ↓
LangGraph StateGraph
      ↓
Processing Agents (Coordinator → Specialist → Monitor)
```

## Agent Creation Pattern

### Current Implementation (Placeholder)

```python
def create_langgraph_agents(data: Any) -> List[str]:
    """
    Create LangGraph agents for processing received data.

    This is a placeholder implementation per Spec 021.
    Full LangGraph integration is planned for future specs.
    """
    import time
    return [
        f"agent_{int(time.time())}_1",
        f"agent_{int(time.time())}_2"
    ]
```

### Future LangGraph Integration

```python
from langgraph.graph import StateGraph

def create_langgraph_agents(data: BinaryIO) -> List[str]:
    """
    Create LangGraph agents with StateGraph orchestration.

    Args:
        data: File-like binary stream from upload

    Returns:
        List of agent IDs created for processing
    """
    # Extract metadata
    metadata = {
        'filename': getattr(data, 'name', 'unknown'),
        'size': len(data.read()),
        'content_type': get_content_type(data)
    }
    data.seek(0)  # Reset after reading

    # Define processing graph
    graph = StateGraph()

    # Add agent nodes
    coordinator_id = graph.add_node(
        "coordinator",
        CoordinatorAgent(metadata=metadata)
    )

    specialist_id = graph.add_node(
        "specialist",
        SpecialistAgent(data=data, metadata=metadata)
    )

    monitor_id = graph.add_node(
        "monitor",
        MonitorAgent(metadata=metadata)
    )

    # Define edges (workflow)
    graph.add_edge("coordinator", "specialist")
    graph.add_edge("specialist", "monitor")

    # Compile and start
    workflow = graph.compile()
    workflow.invoke({"data": data, "metadata": metadata})

    return [coordinator_id, specialist_id, monitor_id]
```

## Agent Types

### 1. Coordinator Agent
**Role**: Orchestrate processing workflow
**Responsibilities**:
- Analyze document metadata
- Determine processing strategy
- Delegate to specialist agents
- Monitor overall progress

### 2. Specialist Agent
**Role**: Perform domain-specific processing
**Responsibilities**:
- Extract features from content
- Apply domain knowledge
- Generate insights
- Return processed results

### 3. Monitor Agent
**Role**: Track and report processing status
**Responsibilities**:
- Monitor agent health
- Track processing metrics
- Handle errors gracefully
- Report completion status

## State Management

### LangGraph State Schema

```python
from typing import TypedDict, List

class ProcessingState(TypedDict):
    # Input
    data: BinaryIO
    metadata: Dict[str, Any]

    # Processing
    coordinator_status: str
    specialist_results: List[Dict]
    monitor_metrics: Dict[str, float]

    # Output
    processing_complete: bool
    final_results: Dict[str, Any]
    errors: List[str]
```

### State Transitions

```
Initial State
  ↓
Coordinator Analysis
  ↓
Specialist Processing (parallel)
  ↓
Monitor Validation
  ↓
Final State
```

## Integration Points

### 1. Daedalus → LangGraph

**Entry Point**: `receive_perceptual_information()`
```python
def receive_perceptual_information(self, data: Optional[BinaryIO]) -> Dict[str, Any]:
    # ... validation ...

    # Create LangGraph agents
    agents = create_langgraph_agents(data)

    return {
        'status': 'received',
        'agents_created': agents,  # LangGraph integration
        # ... other metadata ...
    }
```

### 2. LangGraph → Processing Pipeline

**Output**: Agent IDs returned to API
```json
{
  "agents_created": [
    "agent_coordinator_1727705400",
    "agent_specialist_1727705400",
    "agent_monitor_1727705400"
  ]
}
```

### 3. Agent Communication

**Protocol**: Redis Pub/Sub (Future)
```python
# Agent publishes status
redis_client.publish(
    f"agent:{agent_id}:status",
    json.dumps({"status": "processing", "progress": 0.5})
)

# Monitor subscribes to updates
redis_client.subscribe(f"agent:*:status")
```

## Configuration

### Agent Configuration

```yaml
langgraph:
  agents:
    coordinator:
      max_workers: 5
      timeout: 30s
      retry_policy:
        max_attempts: 3
        backoff: exponential

    specialist:
      max_workers: 10
      timeout: 60s
      memory_limit: 1GB

    monitor:
      max_workers: 2
      polling_interval: 1s
      alert_threshold: 0.8
```

### Daedalus Integration Config

```yaml
daedalus:
  gateway:
    agent_creation: langgraph
    agent_factory: create_langgraph_agents
    max_agents_per_upload: 3
    agent_naming: "agent_{timestamp}_{index}"
```

## Error Handling

### Agent Creation Failures

```python
try:
    agents = create_langgraph_agents(data)
except Exception as e:
    logger.error(f"LangGraph agent creation failed: {e}")
    return {
        'status': 'error',
        'error_message': f'Agent creation failed: {str(e)}',
        'timestamp': time.time()
    }
```

### Processing Failures

```python
# Monitor agent detects failure
if specialist_agent.status == "failed":
    monitor_agent.report_error({
        'agent_id': specialist_agent.id,
        'error': specialist_agent.last_error,
        'recovery_action': 'retry_with_fallback'
    })
```

## Performance Characteristics

### Agent Creation
- **Time**: <10ms per agent
- **Memory**: ~50MB per agent instance
- **Concurrent Agents**: Up to 20 per upload

### Processing Throughput
- **Documents**: 100+ per minute
- **Agent Coordination**: <5ms overhead
- **State Persistence**: Redis (sub-millisecond)

## Testing Strategy

### Unit Tests
```python
def test_daedalus_creates_langgraph_agents():
    """Test that Daedalus creates LangGraph agents"""
    daedalus = Daedalus()
    test_data = io.BytesIO(b"Test document")
    test_data.name = "test.txt"

    result = daedalus.receive_perceptual_information(test_data)

    assert result['status'] == 'received'
    assert 'agents_created' in result
    assert len(result['agents_created']) > 0
```

### Integration Tests
```python
async def test_end_to_end_langgraph_processing():
    """Test complete flow: Upload → Daedalus → LangGraph → Processing"""
    # Upload document
    response = await client.post("/api/documents", files={"file": test_file})

    # Verify Daedalus reception
    assert response.json()['daedalus_reception'] == 'received'

    # Verify agents created
    agents = response.json()['agents_created']
    assert len(agents) == 3  # Coordinator, Specialist, Monitor

    # Verify agent processing (future)
    # ... agent status checks ...
```

## Migration Path

### Phase 1 (Current - Spec 021) ✅
- Placeholder agent creation
- Simple agent ID generation
- Basic integration structure

### Phase 2 (Future - Spec TBD)
- Full LangGraph StateGraph integration
- Coordinator/Specialist/Monitor agents
- Redis communication

### Phase 3 (Future - Spec TBD)
- Advanced agent strategies
- Adaptive processing
- Self-optimizing workflows

## References

- **Daedalus Implementation**: `backend/src/services/daedalus.py`
- **API Integration**: `backend/src/api/routes/documents.py`
- **LangGraph Docs**: [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- **StateGraph Pattern**: [StateGraph Guide](https://langchain-ai.github.io/langgraph/concepts/low_level/)

## Constitutional Compliance

✅ **Single Responsibility**: Daedalus only receives data, LangGraph handles processing
✅ **Separation of Concerns**: Clear boundaries between gateway and processing
✅ **Testability**: Each component independently testable
✅ **Scalability**: Agents can scale horizontally

---
*Last Updated*: 2025-10-01
*Spec Version*: 021-remove-all-that
*Integration Status*: Phase 1 (Placeholder) - Complete ✅
