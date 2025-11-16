# Knowledge Graph Design

## Architecture Overview

The Knowledge Graph uses Neo4j 5.x as the unified storage layer, consolidating:
- Graph relationships (evolution paths, archetypal connections, episodic links)
- Vector embeddings (512-dimensional architecture representations)
- Full-text indexes (narrative content, descriptions, summaries)

All Neo4j access flows through the **Daedalus Graph Channel**, which orchestrates operations via LangGraph River tasks. Direct Neo4j driver imports are prohibited in application code.

---

## Neo4j-Only Unified Storage

### Decision: Native Vector Indexing

**Rationale**: Neo4j 5.0+ includes native vector similarity search with HNSW indexing, eliminating the need for separate vector databases (Qdrant, FAISS) and reducing system complexity.

**Implementation**:
```cypher
-- Create vector index for architecture embeddings
CREATE VECTOR INDEX architecture_embeddings IF NOT EXISTS
FOR (a:Architecture)
ON a.embedding
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 512,
    `vector.similarity_function`: 'cosine'
  }
}
```

**Query Pattern**:
```cypher
-- Vector similarity search within Cypher
CALL db.index.vector.queryNodes('architecture_embeddings', 10, $query_embedding)
YIELD node, score
MATCH (node)-[:EVOLVED_FROM*0..3]->(ancestor)
RETURN node, score, collect(ancestor) as lineage
ORDER BY score DESC
```

### Full-Text Search Integration

**Implementation**:
```cypher
-- Create full-text index across multiple properties
CREATE FULLTEXT INDEX document_content_index IF NOT EXISTS
FOR (d:Document|Episode|Archetype)
ON EACH [d.extracted_text, d.narrative, d.description]
```

**Query Pattern**:
```cypher
-- Full-text search with relationship context
CALL db.index.fulltext.queryNodes('document_content_index', $query)
YIELD node, score
OPTIONAL MATCH (node)-[r]->(related)
RETURN node, score, collect({type: type(r), target: related})
ORDER BY score DESC
LIMIT 10
```

---

## Schema and Constraints

### Uniqueness Constraints

Prevent duplicate entities and ensure data integrity:

```cypher
-- Document constraints
CREATE CONSTRAINT document_id_unique IF NOT EXISTS
FOR (d:Document) REQUIRE d.document_id IS UNIQUE;

CREATE CONSTRAINT content_hash_unique IF NOT EXISTS
FOR (d:Document) REQUIRE d.content_hash IS UNIQUE;

-- Concept constraints
CREATE CONSTRAINT concept_id_unique IF NOT EXISTS
FOR (c:Concept) REQUIRE c.concept_id IS UNIQUE;

-- AttractorBasin constraints
CREATE CONSTRAINT basin_id_unique IF NOT EXISTS
FOR (b:AttractorBasin) REQUIRE b.basin_id IS UNIQUE;

-- Thoughtseed constraints
CREATE CONSTRAINT seed_id_unique IF NOT EXISTS
FOR (s:Thoughtseed) REQUIRE s.seed_id IS UNIQUE;

-- Architecture constraints
CREATE CONSTRAINT architecture_id_unique IF NOT EXISTS
FOR (a:Architecture) REQUIRE a.architecture_id IS UNIQUE;
```

### Performance Indexes

Optimize frequent query patterns:

```cypher
-- Temporal queries
CREATE INDEX document_upload_timestamp IF NOT EXISTS
FOR (d:Document) ON d.upload_timestamp;

-- Quality filtering
CREATE INDEX concept_quality IF NOT EXISTS
FOR (c:Concept) ON c.quality;

-- Tier-based queries
CREATE INDEX concept_tier IF NOT EXISTS
FOR (c:Concept) ON c.tier;

-- Consciousness depth queries
CREATE INDEX consciousness_depth IF NOT EXISTS
FOR (cs:ConsciousnessState) ON cs.meta_cognitive_depth;

-- Composite indexes for common filter combinations
CREATE INDEX document_status_timestamp IF NOT EXISTS
FOR (d:Document) ON (d.processing_status, d.upload_timestamp);
```

---

## Daedalus Graph Channel Architecture

### Constitutional Compliance

All Neo4j operations flow through the Graph Channel facade. Direct driver imports are blocked by CI checks.

**Allowed Pattern**:
```python
from daedalus_gateway import get_graph_channel

class Neo4jSearcher:
    def __init__(self):
        self.graph_channel = get_graph_channel()

    async def search(self, query: str):
        cypher_query = "MATCH (n) WHERE ... RETURN n"
        response = await self.graph_channel.execute_read(
            query=cypher_query,
            parameters={"query": query},
            caller_service="neo4j_searcher",
            caller_function="search"
        )
        return response["records"]
```

**Prohibited Pattern**:
```python
# BLOCKED BY CI - Direct neo4j import
from neo4j import GraphDatabase  # ❌ VIOLATION

driver = GraphDatabase.driver(uri, auth=auth)  # ❌ VIOLATION
```

### Graph Channel API

The DaedalusGraphChannel exposes these operations:

```python
# Read operations (via LangGraph River)
await graph_channel.execute_read(
    query: str,              # Cypher query
    parameters: dict,        # Query parameters
    caller_service: str,     # Service identifier
    caller_function: str     # Function name for auditing
)

# Write operations (via LangGraph River)
await graph_channel.execute_write(
    query: str,
    parameters: dict,
    caller_service: str,
    caller_function: str
)

# Schema operations
await graph_channel.execute_schema(
    query: str,              # DDL statements (CREATE INDEX, etc.)
    caller_service: str,
    caller_function: str
)

# Health checks
await graph_channel.health_check()
```

### LangGraph River Orchestration

Each graph operation creates a LangGraph River task:

1. **Request arrives** → `execute_read()` called
2. **Task created** → LangGraph River task with operation details
3. **Task queued** → Added to River processing queue
4. **Task executed** → Neo4j driver executes Cypher in River context
5. **Response returned** → Results flow back through channel

**Benefits**:
- Centralized logging and monitoring
- Consistent error handling
- Failure recovery without driver bypass
- Constitutional enforcement at single point

---

## Search Patterns

### Multi-Strategy Search

Combine full-text, graph pattern, vector similarity, and relationship traversal:

```python
async def search(self, query: str, limit: int = 10):
    # Strategy 1: Full-text search
    fulltext_results = await self._fulltext_search(query, limit)

    # Strategy 2: Graph pattern matching
    graph_results = await self._graph_pattern_search(query, limit)

    # Strategy 3: Relationship traversal from high-relevance results
    related_results = await self._find_related_nodes(
        seed_results=fulltext_results[:3],
        limit=limit
    )

    # Deduplicate and rank by relevance
    all_results = fulltext_results + graph_results + related_results
    unique_results = self._deduplicate_results(all_results)

    return sorted(unique_results, key=lambda r: r.relevance_score, reverse=True)[:limit]
```

### Hybrid Vector + Graph Queries

Single Cypher query combining vector similarity with graph constraints:

```cypher
-- Find similar architectures that evolved from transformers
CALL db.index.vector.queryNodes('architecture_embeddings', 20, $query_embedding)
YIELD node, score
WHERE (node)-[:EVOLVED_FROM*1..5]->(:Architecture {pattern: 'transformer'})
MATCH (node)-[:HAS_CONSCIOUSNESS]->(cs:ConsciousnessState)
RETURN
    node.architecture_id as id,
    node.description as description,
    score as similarity,
    cs.meta_cognitive_depth as consciousness_level
ORDER BY score DESC
LIMIT 10
```

### Relevance Scoring Normalization

Results from different strategies use different scoring ranges. Normalize to [0, 1]:

```python
def _normalize_scores(results: List[SearchResult]) -> List[SearchResult]:
    for result in results:
        if result.source == SearchSource.FULLTEXT:
            # Full-text scores from Neo4j are already normalized
            pass
        elif result.source == SearchSource.VECTOR:
            # Cosine similarity already in [-1, 1], shift to [0, 1]
            result.relevance_score = (result.relevance_score + 1) / 2
        elif result.source == SearchSource.GRAPH_PATTERN:
            # Connection count / max_connections
            result.relevance_score = min(result.relevance_score / 10.0, 1.0)

    return results
```

---

## Migration Strategy

### Three-Tier Validation

Ensure zero data loss when migrating from MongoDB, FAISS, SQLite:

**Tier 1: Count Validation**
```python
source_count = mongodb.collection.count_documents({})
target_count = await graph_channel.execute_read(
    query="MATCH (n:Architecture) RETURN count(n) as count",
    parameters={},
    caller_service="migration",
    caller_function="validate_count"
)
assert source_count == target_count["records"][0]["count"]
```

**Tier 2: Content Validation (10% Sample)**
```python
sample_ids = random.sample(all_ids, len(all_ids) // 10)
for arch_id in sample_ids:
    source_data = mongodb.find_one({"_id": arch_id})
    target_data = await graph_channel.execute_read(
        query="MATCH (a:Architecture {architecture_id: $id}) RETURN a",
        parameters={"id": arch_id},
        caller_service="migration",
        caller_function="validate_content"
    )
    assert deep_compare(source_data, target_data["records"][0]["a"])
```

**Tier 3: Functional Validation**
```python
# Compare query results between old and new systems
test_queries = [
    "find_similar_architectures",
    "get_evolution_path",
    "search_by_consciousness_level"
]

for query_name in test_queries:
    old_results = old_system.execute(query_name, params)
    new_results = await unified_system.execute(query_name, params)
    assert results_equivalent(old_results, new_results)
```

### Staged Rollback

Preserve source systems until validation passes:

1. **Stage 1**: Extract all data from source systems (read-only)
2. **Stage 2**: Transform and load into Neo4j
3. **Stage 3**: Run validation tests
4. **Stage 4**: If validation fails → rollback, fix, retry
5. **Stage 5**: If validation passes → archive source systems

---

## Performance Targets

### Query Performance (10,000+ nodes)

- **Single architecture lookup**: < 10ms
- **Vector similarity (top 10)**: < 100ms
- **Evolution path traversal**: < 50ms
- **Full-text narrative search**: < 200ms
- **Hybrid graph+vector query**: < 150ms

### Indexing Strategy

Use HNSW (Hierarchical Navigable Small World) for vector indexes:
- Fast approximate nearest neighbor search
- Sub-linear query time complexity
- Configurable recall vs speed tradeoff

```cypher
CREATE VECTOR INDEX architecture_embeddings IF NOT EXISTS
FOR (a:Architecture)
ON a.embedding
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 512,
    `vector.similarity_function`: 'cosine',
    `vector.hnsw.m`: 16,              -- Number of connections per layer
    `vector.hnsw.ef_construction`: 200 -- Build-time search depth
  }
}
```

---

## Failure Handling

### Graph Channel Unavailable

When Daedalus or LangGraph River is down, queue requests instead of bypassing:

```python
async def execute_read_with_retry(self, query: str, parameters: dict, max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            return await self.graph_channel.execute_read(
                query=query,
                parameters=parameters,
                caller_service=self.caller_service,
                caller_function="execute_with_retry"
            )
        except GraphChannelUnavailableError:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
            else:
                # Queue for later processing, DO NOT bypass to direct driver
                await self.queue_for_retry(query, parameters)
                raise
```

### Legacy Adapter Pattern

For scripts that cannot be immediately refactored:

```python
# Adapter wraps legacy script, translates to Graph Channel calls
class LegacyHealthCheckAdapter:
    def __init__(self):
        self.graph_channel = get_graph_channel()

    async def check_neo4j_health(self):
        # Legacy script expected direct driver access
        # Adapter translates to Graph Channel
        response = await self.graph_channel.health_check()

        # Log adapter usage for deprecation tracking
        logger.info("Legacy adapter used: health_check")

        return response["status"] == "healthy"
```

---

## Governance and Compliance

### CI Enforcement

Pre-commit and CI checks block direct Neo4j imports:

```yaml
# .pre-commit-config.yaml
- repo: local
  hooks:
    - id: block-neo4j-imports
      name: Block direct neo4j imports
      entry: python scripts/check_neo4j_imports.py
      language: system
      types: [python]
      exclude: ^daedalus-gateway/
```

```python
# scripts/check_neo4j_imports.py
import ast
import sys

def check_file(filepath):
    with open(filepath) as f:
        tree = ast.parse(f.read())

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "neo4j":
                    print(f"❌ VIOLATION: Direct neo4j import in {filepath}")
                    return 1
        elif isinstance(node, ast.ImportFrom):
            if node.module == "neo4j":
                print(f"❌ VIOLATION: Direct neo4j import in {filepath}")
                return 1

    return 0

if __name__ == "__main__":
    sys.exit(check_file(sys.argv[1]))
```

### Adapter Registry

Track legacy adapters and deprecation timeline:

```yaml
# legacy_adapters.yaml
adapters:
  - name: health_check_adapter
    script: backend/scripts/health_check.py
    adapter: backend/src/adapters/health_check_adapter.py
    usage_count: 45  # Updated by telemetry
    deprecation_date: 2025-12-01

  - name: schema_export_adapter
    script: backend/scripts/export_schema.py
    adapter: backend/src/adapters/schema_export_adapter.py
    usage_count: 12
    deprecation_date: 2025-11-15
```

---

## Implementation Checklist

- [ ] Create Neo4j vector indexes for all embedding fields
- [ ] Create full-text indexes for narrative content
- [ ] Implement uniqueness constraints on all entity IDs
- [ ] Create performance indexes for common query patterns
- [ ] Implement DaedalusGraphChannel read/write/schema operations
- [ ] Refactor Neo4jSearcher to use Graph Channel exclusively
- [ ] Implement multi-strategy search with deduplication
- [ ] Create migration validation framework (3-tier)
- [ ] Set up CI checks to block direct neo4j imports
- [ ] Create legacy adapter registry and tracking
- [ ] Document graph channel API and usage patterns
- [ ] Implement failure handling with queueing (no bypass)
- [ ] Benchmark query performance against targets
- [ ] Create monitoring dashboards for graph operations
