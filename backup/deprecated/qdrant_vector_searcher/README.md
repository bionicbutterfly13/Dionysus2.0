# Deprecated: Qdrant Vector Searcher

**Archived**: 2025-10-01
**Reason**: Replaced with Neo4j unified search (graph + vector + full-text)

## Why Qdrant Was Removed

After analysis of the hybrid architecture (`HYBRID_STORAGE_ANALYSIS.md`), we determined that:

1. **Neo4j provides sufficient vector search** for current scale (<100k documents)
2. **Graph relationships are critical** for consciousness tracking (basins, thoughtseeds)
3. **AutoSchemaKG integration** automatically constructs knowledge graph from documents
4. **Single database architecture** is simpler and more maintainable

## What Was Removed

### Files Deleted
- `backend/src/services/vector_searcher.py` (234 lines)
  - `VectorSearcher` class
  - Qdrant client initialization
  - Vector similarity search methods
  - Collection management

### Code Changes
- `backend/src/services/query_engine.py`
  - Removed `VectorSearcher` import
  - Removed parallel search to Qdrant
  - Changed to Neo4j-only search
  - Updated health check

- `backend/src/config/settings.py`
  - Removed `QDRANT_URL` setting
  - Removed Qdrant from `get_database_url()`

## VectorSearcher Functionality (Archived)

The removed `VectorSearcher` class provided:

```python
class VectorSearcher:
    """Search Qdrant vector database for semantically similar content"""

    def __init__(self, client: Optional[QdrantClient] = None):
        self._client = client
        self.collection_name = "documents"
        self.vector_dimensions = 384

    async def search(self, query: str, limit: int = 10):
        """Vector similarity search"""
        query_vector = await self._generate_embedding(query)
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit
        )
        return convert_to_search_results(results)

    async def search_with_filter(self, query, filters, limit):
        """Search with metadata filtering"""

    async def hybrid_search(self, query, semantic_weight, keyword_weight, limit):
        """Combine semantic and keyword search"""
```

## Neo4j Replacement

Neo4j now provides all vector search functionality plus graph relationships:

```python
# Neo4j vector search (replacement)
neo4j_searcher = Neo4jSearcher()

# Combines: vector similarity + graph traversal + full-text search
results = await neo4j_searcher.search(query, limit=10)

# Under the hood:
# 1. Full-text search on documents
# 2. Vector similarity search on embeddings (512-dim, cosine)
# 3. Graph pattern matching on concepts
# 4. Relationship traversal for related nodes
```

## When to Reconsider Qdrant

Consider re-adding Qdrant if:
- Document count exceeds **50,000+ documents**
- Neo4j vector search latency consistently > **100ms**
- Need specialized vector operations (batching, filtering, etc.)

In that case:
1. Restore `vector_searcher.py` from this archive
2. Re-add `QDRANT_URL` to settings
3. Implement Qdrant as **performance cache layer** (Neo4j remains source of truth)
4. Use hybrid architecture: Qdrant for fast vector search, Neo4j for graph enrichment

## Implementation Strategy (If Needed)

```python
class HybridSearchManager:
    def __init__(self):
        self.neo4j = Neo4jSearcher()  # Source of truth
        self.qdrant = VectorSearcher()  # Performance cache
        self.use_qdrant = True  # Feature flag

    async def search(self, query, limit):
        if self.use_qdrant:
            # Fast vector search in Qdrant
            qdrant_results = await self.qdrant.search(query, limit)

            # Enrich with Neo4j graph data
            neo4j_ids = [r.metadata["neo4j_id"] for r in qdrant_results]
            graph_data = await self.neo4j.get_nodes_with_relationships(neo4j_ids)

            return merge(qdrant_results, graph_data)
        else:
            # Use Neo4j directly
            return await self.neo4j.search(query, limit)
```

## Performance Comparison

**Before (Hybrid - Neo4j + Qdrant)**:
- Vector search: 20-50ms (Qdrant)
- Graph enrichment: 30-80ms (Neo4j)
- Total: 50-130ms

**After (Neo4j Only)**:
- Unified search: 40-100ms (graph + vector + full-text in one query)
- Simpler architecture
- No data synchronization needed

**Conclusion**: Neo4j alone is faster for small-medium scale (<50k docs) and provides better graph integration.

---

**Files in this archive**:
- `vector_searcher.py` - Complete VectorSearcher implementation
- `README.md` - This file (removal explanation)

**Last Updated**: 2025-10-01
**For questions**: See `HYBRID_STORAGE_ANALYSIS.md` in project root
