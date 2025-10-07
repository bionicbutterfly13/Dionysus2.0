# Graph Channel Migration Quick Reference
**For Dionysus Service Authors**

## Quick Migration Patterns

### Pattern 1: Simple Query Execution

**BEFORE (Deprecated)**:
```python
from backend.src.config.neo4j_config import get_neo4j_driver

driver = get_neo4j_driver()
with driver.session() as session:
    result = session.run("MATCH (d:Document) RETURN d.filename LIMIT 10")
    filenames = [record["d"]["filename"] for record in result]
```

**AFTER (Constitutional)**:
```python
from daedalus_gateway import get_graph_channel

channel = get_graph_channel()
await channel.connect()

result = await channel.execute_read(
    query="MATCH (d:Document) RETURN d.filename LIMIT 10",
    caller_service="document_processor",
    caller_function="get_recent_documents"
)

filenames = [record["d"]["filename"] for record in result['records']]
```

### Pattern 2: Write Operations

**BEFORE (Deprecated)**:
```python
from backend.src.config.neo4j_config import get_neo4j_driver

driver = get_neo4j_driver()
with driver.session() as session:
    session.run(
        "CREATE (d:Document {id: $id, filename: $filename})",
        id=doc_id,
        filename=filename
    )
```

**AFTER (Constitutional)**:
```python
from daedalus_gateway import get_graph_channel

channel = get_graph_channel()
await channel.connect()

result = await channel.execute_write(
    query="CREATE (d:Document {id: $id, filename: $filename})",
    parameters={"id": doc_id, "filename": filename},
    caller_service="document_processor",
    caller_function="create_document_node"
)
```

### Pattern 3: Schema Operations

**BEFORE (Deprecated)**:
```python
from backend.src.config.neo4j_config import initialize_neo4j_schema

schema_result = initialize_neo4j_schema()
if schema_result['schema_ready']:
    print("Schema is ready")
```

**AFTER (Constitutional)**:
```python
from backend.src.config.neo4j_config import initialize_neo4j_schema_async

schema_result = await initialize_neo4j_schema_async()
if schema_result['schema_ready']:
    print("Schema is ready")
```

### Pattern 4: Health Checks

**BEFORE (Deprecated)**:
```python
from backend.src.services.database_health import is_database_healthy

if is_database_healthy('neo4j'):
    # proceed with operation
    pass
```

**AFTER (Constitutional)**:
```python
from backend.src.services.database_health import is_database_healthy_async

if await is_database_healthy_async('neo4j'):
    # proceed with operation
    pass
```

### Pattern 5: Batch Operations

**NEW Capability** (Not available in legacy driver approach):
```python
from daedalus_gateway import get_graph_channel

channel = get_graph_channel()
await channel.connect()

# Batch create multiple documents
documents = [
    {"id": "doc1", "filename": "file1.pdf"},
    {"id": "doc2", "filename": "file2.pdf"},
    # ... up to 1000 items
]

result = await channel.execute_write(
    query="""
        UNWIND $batch as doc
        CREATE (d:Document {id: doc.id, filename: doc.filename})
    """,
    parameters={"batch": documents},
    caller_service="document_processor",
    caller_function="batch_create_documents"
)
```

## Common Conversion Patterns

### Converting Sync to Async

**If your function is already async**:
```python
async def my_service_function():
    channel = get_graph_channel()
    await channel.connect()
    result = await channel.execute_read(...)
    # process result
```

**If your function is sync** (two options):

Option A - Temporary asyncio.run() wrapper:
```python
import asyncio

def my_service_function():
    async def _async_impl():
        channel = get_graph_channel()
        await channel.connect()
        return await channel.execute_read(...)

    result = asyncio.run(_async_impl())
    # process result
```

Option B - Convert to async (RECOMMENDED):
```python
async def my_service_function():  # Changed from def to async def
    channel = get_graph_channel()
    await channel.connect()
    result = await channel.execute_read(...)
    # process result
```

### Error Handling

**Graph Channel provides structured error responses**:
```python
result = await channel.execute_read(
    query="MATCH (d:Document) WHERE d.id = $id RETURN d",
    parameters={"id": document_id},
    caller_service="my_service",
    caller_function="get_document"
)

if result['success']:
    records = result['records']
    # process records
else:
    error = result['error']
    logger.error(f"Query failed: {error}")
    # handle error
```

### Health Check with Details

**Get detailed health information**:
```python
channel = get_graph_channel()
await channel.connect()

health = await channel.health_check()

print(f"Connected: {health['connected']}")
print(f"Circuit open: {health['circuit_open']}")
print(f"Success rate: {health['success_rate']:.2%}")
print(f"Latency: {health['latency_ms']:.2f}ms")
```

## Parameter Naming Convention

All Graph Channel operations require these parameters for constitutional compliance:

```python
await channel.execute_read(
    query="...",                          # Your Cypher query
    parameters={...},                     # Query parameters (optional)
    caller_service="service_name",        # REQUIRED: Your service name
    caller_function="function_name"       # REQUIRED: Your function name
)
```

**Why caller_service and caller_function?**
- Enables telemetry and audit trails
- Tracks which services use which graph operations
- Helps identify slow queries by caller
- Constitutional compliance requirement

## Common Mistakes

### ❌ Mistake 1: Forgetting caller_service
```python
# WRONG - Missing caller information
result = await channel.execute_read(query="MATCH ...")
```

```python
# CORRECT
result = await channel.execute_read(
    query="MATCH ...",
    caller_service="my_service",
    caller_function="my_function"
)
```

### ❌ Mistake 2: Not awaiting async calls
```python
# WRONG - Not awaiting
result = channel.execute_read(...)  # Returns coroutine, not result!
```

```python
# CORRECT
result = await channel.execute_read(...)
```

### ❌ Mistake 3: Creating new channel instances
```python
# WRONG - Don't create multiple instances
channel1 = DaedalusGraphChannel()
channel2 = DaedalusGraphChannel()
```

```python
# CORRECT - Use singleton
channel = get_graph_channel()  # Always returns same instance
```

### ❌ Mistake 4: Accessing result['records'] without checking success
```python
# WRONG - Assumes success
records = result['records']  # May fail if result['success'] is False
```

```python
# CORRECT - Check success first
if result['success']:
    records = result['records']
else:
    logger.error(f"Query failed: {result['error']}")
```

## Configuration

### Using Custom Neo4j Connection

**Default** (uses settings from environment):
```python
from daedalus_gateway import get_graph_channel

channel = get_graph_channel()  # Uses NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD from settings
```

**Custom Configuration**:
```python
from daedalus_gateway import get_graph_channel, GraphChannelConfig

config = GraphChannelConfig(
    neo4j_uri="bolt://custom-host:7687",
    neo4j_user="custom_user",
    neo4j_password="custom_password",
    max_connection_pool_size=100,
    default_query_timeout_ms=60000
)

channel = get_graph_channel(config)
```

## Testing with Graph Channel

### Unit Tests

```python
import pytest
from daedalus_gateway import get_graph_channel

@pytest.mark.asyncio
async def test_document_query():
    channel = get_graph_channel()
    await channel.connect()

    result = await channel.execute_read(
        query="MATCH (d:Document) RETURN count(d) as total",
        caller_service="test",
        caller_function="test_document_query"
    )

    assert result['success']
    assert result['records'][0]['total'] >= 0
```

### Mocking Graph Channel

```python
from unittest.mock import AsyncMock, MagicMock

# Create mock channel
mock_channel = MagicMock()
mock_channel.execute_read = AsyncMock(return_value={
    'success': True,
    'records': [{'d': {'filename': 'test.pdf'}}],
    'rows_affected': 1
})

# Use in test
result = await mock_channel.execute_read(
    query="...",
    caller_service="test",
    caller_function="test"
)
assert result['success']
```

## Getting Help

### Enable Debug Logging
```python
import logging

logging.getLogger("daedalus.graph_channel").setLevel(logging.DEBUG)
```

### Check Telemetry
```python
channel = get_graph_channel()
telemetry = channel.get_telemetry_report()

print(f"Total operations: {telemetry['total_operations']}")
print(f"Success rate: {telemetry['success_rate']:.2%}")
print(f"Average execution time: {telemetry['average_execution_time_ms']:.2f}ms")
```

### Circuit Breaker Status
```python
channel = get_graph_channel()
health = await channel.health_check()

if health['circuit_open']:
    print("⚠️ Circuit breaker is OPEN - too many failures")
    print(f"Consecutive failures: {health['consecutive_failures']}")
```

## Migration Checklist

- [ ] Replace `get_neo4j_driver()` with `get_graph_channel()`
- [ ] Convert sync functions to async
- [ ] Add `caller_service` and `caller_function` parameters
- [ ] Replace `driver.session()` context managers with `channel.execute_*()`
- [ ] Update error handling to check `result['success']`
- [ ] Update tests to async patterns
- [ ] Remove `with driver.session() as session:` blocks
- [ ] Update health checks to `*_async()` versions

## Resources

- Full specification: `/Volumes/Asylum/dev/Dionysus-2.0/SPEC_040_M2_INFRASTRUCTURE_MIGRATION_REPORT.md`
- Graph Channel implementation: `/Volumes/Asylum/dev/daedalus-gateway/src/daedalus_gateway/graph_channel.py`
- Constitutional requirements: `AGENT_CONSTITUTION.md` Section 2.1-2.2

---

**Quick Reference Version**: 1.0
**Last Updated**: 2025-10-07
