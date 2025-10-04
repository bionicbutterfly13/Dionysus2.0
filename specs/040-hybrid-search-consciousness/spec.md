# Spec 040: Hybrid Search with Consciousness Reranking

**Feature**: Triple hybrid search (semantic + graph + fulltext) with consciousness-enhanced reranking
**Type**: Backend Enhancement + Frontend Search UI
**Priority**: P1 (High)
**Complexity**: High
**Timeline**: 3 weeks

---

## Problem Statement

### Current State
- **Single search method**: Neo4j vector search only
- No combination of search strategies (semantic, graph, fulltext)
- No reranking based on consciousness context (basins, thoughtseeds)
- No Reciprocal Rank Fusion (RRF) to combine results
- Search quality limited by single vector similarity

### User Pain Points
1. **Missed Results**: Vector search alone misses keyword matches
2. **No Graph Context**: Don't leverage relationship paths in Neo4j
3. **Static Ranking**: Results ranked by similarity only, no consciousness context
4. **Lost Intelligence**: Basins/thoughtseeds don't influence search

### What We're NOT Changing
✅ Keep existing Neo4j database (no PostgreSQL migration)
✅ Keep existing query API structure
✅ Keep existing frontend pages
✅ Keep existing data models

---

## Solution Overview

### Triple Hybrid Search Architecture
**Combine** three complementary search methods:
1. **Semantic Search**: Vector embeddings (existing Neo4j functionality)
2. **Graph Search**: Cypher path queries (leverage relationships)
3. **Full-Text Search**: Keyword matching (Neo4j text indexes)

**Then Apply**:
- **Reciprocal Rank Fusion (RRF)**: Combine results intelligently
- **Consciousness Reranking**: Boost by basin resonance + thoughtseed similarity

**Result**: Better recall (find more), better precision (rank better)

---

## Functional Requirements

### FR-001: Semantic Search (Enhanced)
**Description**: Vector similarity search with embeddings

**Acceptance Criteria**:
- [ ] Use existing Neo4j vector search
- [ ] Query embedding via sentence transformer
- [ ] Cosine similarity ranking
- [ ] Top-K results (configurable, default 20)
- [ ] Metadata filtering (document_id, tags, date range)

**Current Implementation**: ✅ Already exists in [neo4j_searcher.py](backend/src/services/neo4j_searcher.py:1)

**Enhancements**:
- [ ] Add embedding cache (avoid re-embedding same queries)
- [ ] Hybrid vector search (multi-vector per document chunk)

### FR-002: Graph Search (New)
**Description**: Cypher path queries to find related content

**Acceptance Criteria**:
- [ ] Find documents via relationship paths
- [ ] Query patterns:
  - Direct: `(query)-[:MENTIONS]->(concept)<-[:HAS_CONCEPT]-(doc)`
  - Basin: `(query)-[:ATTRACTED_TO]->(basin)<-[:IN_BASIN]-(doc)`
  - ThoughtSeed: `(query)-[:TRIGGERS]->(ts)-[:LINKS]->(doc)`
- [ ] Configurable path depth (1-3 hops)
- [ ] Score by path strength (relationship weights)
- [ ] Return top-K graph matches

**Implementation**:
```python
async def graph_search(query: str, max_hops: int = 2) -> List[SearchResult]:
    """
    Find documents via graph relationships.

    Query → Concepts/Basins/ThoughtSeeds → Documents
    """
    cypher = f"""
    MATCH path = (q:Query {{text: $query}})-[*1..{max_hops}]-(doc:Document)
    WHERE NOT (q)-[:DIRECT_MATCH]-(doc)  // Exclude already found by semantic
    WITH doc, path, relationships(path) as rels
    WITH doc, reduce(score=1.0, r in rels | score * r.weight) as path_score
    RETURN doc, path_score
    ORDER BY path_score DESC
    LIMIT 20
    """

    results = await self.neo4j.run(cypher, query=query)
    return [SearchResult(
        document_id=r['doc'].id,
        score=r['path_score'],
        source='graph',
    ) for r in results]
```

### FR-003: Full-Text Search (New)
**Description**: Keyword-based text matching

**Acceptance Criteria**:
- [ ] Use Neo4j full-text indexes (already available)
- [ ] BM25 ranking algorithm
- [ ] Phrase matching ("exact phrase" in quotes)
- [ ] Boolean operators (AND, OR, NOT)
- [ ] Wildcard support (neural*)
- [ ] Return top-K text matches

**Implementation**:
```python
async def fulltext_search(query: str) -> List[SearchResult]:
    """
    Keyword matching via Neo4j fulltext index.

    Uses BM25 scoring for relevance.
    """
    cypher = """
    CALL db.index.fulltext.queryNodes('documentContent', $query)
    YIELD node, score
    RETURN node as doc, score
    LIMIT 20
    """

    results = await self.neo4j.run(cypher, query=query)
    return [SearchResult(
        document_id=r['doc'].id,
        score=r['score'],
        source='fulltext',
    ) for r in results]
```

### FR-004: Reciprocal Rank Fusion (RRF)
**Description**: Combine results from multiple search methods

**Acceptance Criteria**:
- [ ] Fuse semantic + graph + fulltext results
- [ ] RRF formula: `score = Σ 1/(k + rank_i)` where k=60 (standard)
- [ ] Handle duplicate documents across methods
- [ ] Preserve source information (which methods found it)
- [ ] Configurable fusion weights (if needed)

**Algorithm**:
```python
def reciprocal_rank_fusion(
    results_lists: List[List[SearchResult]],
    k: int = 60
) -> List[SearchResult]:
    """
    Combine multiple ranked lists using RRF.

    RRF is robust to different scoring scales.
    """
    fused_scores = {}

    for results in results_lists:
        for rank, result in enumerate(results, start=1):
            doc_id = result.document_id
            rrf_score = 1.0 / (k + rank)

            if doc_id in fused_scores:
                fused_scores[doc_id] += rrf_score
            else:
                fused_scores[doc_id] = rrf_score

    # Sort by fused score
    sorted_docs = sorted(
        fused_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        SearchResult(document_id=doc_id, score=score, source='rrf')
        for doc_id, score in sorted_docs
    ]
```

### FR-005: Consciousness Reranking (New)
**Description**: Boost results based on basin resonance and thoughtseed similarity

**Acceptance Criteria**:
- [ ] Match query to active basins (top 3)
- [ ] Boost documents in those basins (multiply by basin strength)
- [ ] Match query to thoughtseeds (similarity > 0.7)
- [ ] Boost thoughtseed-linked documents
- [ ] Final score: `rrf_score × basin_boost × thoughtseed_boost`
- [ ] Preserve RRF ranking if no consciousness match

**Algorithm**:
```python
async def consciousness_rerank(
    results: List[SearchResult],
    query: str,
) -> List[SearchResult]:
    """
    Rerank using consciousness context.

    Basin resonance + ThoughtSeed similarity boost.
    """
    # 1. Match query to basins
    basins = await match_query_to_basins(query, top_k=3)
    basin_map = {b.document_id: b.strength for b in basins}

    # 2. Match query to thoughtseeds
    thoughtseeds = await match_query_to_thoughtseeds(query, threshold=0.7)
    ts_docs = {ts.document_id for ts in thoughtseeds}

    # 3. Rerank
    for result in results:
        # Basin boost (1.0 to 2.0x)
        basin_boost = 1.0 + basin_map.get(result.document_id, 0.0)

        # ThoughtSeed boost (1.5x if linked)
        ts_boost = 1.5 if result.document_id in ts_docs else 1.0

        # Apply boosts
        result.score *= basin_boost * ts_boost

    # Resort by boosted scores
    return sorted(results, key=lambda x: x.score, reverse=True)
```

### FR-006: Search API Integration
**Description**: Unified search endpoint for frontend

**Acceptance Criteria**:
- [ ] Single endpoint: `POST /api/search/hybrid`
- [ ] Request params: query, filters, mode (semantic/hybrid)
- [ ] Response includes: results, metadata, timing
- [ ] Metadata shows which methods contributed (semantic/graph/fulltext)
- [ ] Timing breakdown (semantic: Xms, graph: Yms, fusion: Zms)

**API Contract**:
```typescript
// Request
POST /api/search/hybrid
{
  query: string
  filters?: {
    document_ids?: string[]
    tags?: string[]
    date_range?: { start: string, end: string }
  }
  mode?: 'semantic' | 'hybrid' | 'consciousness'  // Default: consciousness
  limit?: number  // Default: 20
}

// Response
{
  results: Array<{
    document_id: string
    title: string
    snippet: string  // Highlighted excerpt
    score: number
    sources: string[]  // ['semantic', 'graph', 'fulltext']
    basin_context?: { name: string, strength: number }
    thoughtseed_links?: number
  }>
  metadata: {
    total_found: number
    search_methods_used: string[]
    timing: {
      semantic_ms: number
      graph_ms: number
      fulltext_ms: number
      fusion_ms: number
      rerank_ms: number
      total_ms: number
    }
  }
}
```

---

## Non-Functional Requirements

### NFR-001: Performance
- Total search latency: <200ms p95 for hybrid search
- Semantic search: <50ms
- Graph search: <80ms
- Full-text search: <30ms
- RRF fusion: <10ms
- Consciousness rerank: <30ms

### NFR-002: Scalability
- Handle 100,000+ documents
- Support 100+ concurrent search requests
- Graph queries optimized with indexes
- Caching for repeated queries (Redis)

### NFR-003: Accuracy
- Precision@10: >0.8 (80% of top 10 results relevant)
- Recall@20: >0.6 (60% of relevant docs in top 20)
- Improvement over semantic-only: +15-20% precision

---

## Technical Design

### Backend Architecture
```
HybridSearchService
├── SemanticSearcher (Neo4j vector)
├── GraphSearcher (Cypher paths) [NEW]
├── FullTextSearcher (Neo4j fulltext) [NEW]
├── RRFFusion (combine results) [NEW]
└── ConsciousnessReranker (basin/ts boost) [NEW]

Flow:
1. Parallel execution:
   - semantic_search(query)
   - graph_search(query)
   - fulltext_search(query)

2. Combine with RRF:
   - fused_results = rrf([semantic, graph, fulltext])

3. Consciousness rerank:
   - final_results = consciousness_rerank(fused_results, query)

4. Return top-K
```

### Neo4j Indexes Required
```cypher
// Full-text index (if not exists)
CREATE FULLTEXT INDEX documentContent IF NOT EXISTS
FOR (d:Document)
ON EACH [d.content, d.title, d.summary]

// Vector index (already exists)
CREATE VECTOR INDEX documentEmbedding IF NOT EXISTS
FOR (d:Document)
ON d.embedding
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 384,
    `vector.similarity_function`: 'cosine'
  }
}

// Relationship indexes for graph search
CREATE INDEX relationship_weight IF NOT EXISTS
FOR ()-[r:ATTRACTED_TO]-()
ON (r.weight)
```

### Frontend Integration

**New Search Component**: Enhanced search bar with mode selector

```tsx
// In KnowledgeBase.tsx or new SearchPage.tsx
<SearchBar
  onSearch={(query, mode) => hybridSearch(query, mode)}
  modes={['semantic', 'hybrid', 'consciousness']}
  defaultMode="consciousness"
/>

<SearchResults
  results={searchResults}
  onResultClick={(doc) => navigate(`/document/${doc.id}`)}
  showMetadata={true}  // Shows sources, basins, timing
/>
```

---

## User Stories

### Story 1: Comprehensive Search
**As a** user
**I want to** find documents using keywords AND semantic meaning AND graph relationships
**So that** I don't miss relevant results

**Acceptance**: Hybrid search finds more relevant docs than semantic alone

### Story 2: Consciousness-Aware Search
**As a** user
**I want to** search results boosted by basin relevance
**So that** I see docs in active attractor basins first

**Acceptance**: Documents in query-matched basins ranked higher

### Story 3: Search Transparency
**As a** user
**I want to** see which search methods found each result
**So that** I understand why it's relevant

**Acceptance**: Results show badges (🔍 Semantic, 🕸️ Graph, 📝 Text)

### Story 4: Performance Insight
**As a** developer/power-user
**I want to** see search timing breakdown
**So that** I can understand performance

**Acceptance**: Metadata shows timing for each search stage

---

## Testing Strategy

### Unit Tests
- [ ] RRF algorithm (multiple ranked lists → fused scores)
- [ ] Consciousness reranking (basin/thoughtseed boost)
- [ ] Graph search query generation
- [ ] Full-text query parsing

### Integration Tests
- [ ] Hybrid search end-to-end
- [ ] Parallel search execution
- [ ] Result deduplication
- [ ] Metadata accuracy

### Performance Tests
- [ ] Latency under load (100 concurrent requests)
- [ ] Scalability (100K documents)
- [ ] Cache effectiveness (repeated queries)

---

## Success Metrics

### Accuracy
- **Target**: Precision@10 > 0.8 (vs 0.65 semantic-only)
- **Measure**: Manual relevance evaluation on 100 queries

### Performance
- **Target**: p95 latency < 200ms
- **Measure**: Production metrics (New Relic/Datadog)

### User Satisfaction
- **Target**: 4.3/5 rating for search quality
- **Measure**: In-app feedback survey

---

## Future Enhancements (Out of Scope)

### Phase 2
- [ ] FlashRank reranker integration (LLM-based reranking)
- [ ] Multi-vector search (different embeddings for title/content)
- [ ] Query expansion (synonyms, related terms)
- [ ] Personalized search (user history, preferences)

### Phase 3
- [ ] Learned ranking (ML model trained on click data)
- [ ] Real-time index updates (no lag for new docs)
- [ ] Federated search (multiple knowledge bases)
- [ ] Search analytics dashboard

---

## Dependencies

### Backend Changes
- [ ] New `HybridSearchService` class
- [ ] Neo4j full-text index creation
- [ ] Graph search Cypher queries
- [ ] RRF fusion algorithm
- [ ] Consciousness reranking logic

### External Libraries
- None! Use existing Neo4j, no FlashRank/Cohere initially

---

## Open Questions

1. **RRF Parameter k**: Use standard k=60 or tune for our data?
   - **Proposal**: Start with 60, A/B test 40/60/80

2. **Graph Search Depth**: 1, 2, or 3 hops maximum?
   - **Proposal**: 2 hops (balance recall/performance)

3. **Consciousness Boost**: How much to boost basin-matched docs?
   - **Proposal**: 1.0-2.0x based on basin strength

4. **Caching Strategy**: Cache final results or intermediate (semantic, graph, etc.)?
   - **Proposal**: Cache final results per (query, mode, filters)

---

## Appendix: SurfSense Comparison

### What We Adopt from SurfSense
✅ Reciprocal Rank Fusion (RRF) concept
✅ Hybrid search architecture (multiple methods)
✅ Reranking pattern (post-processing boost)

### What We Do Differently
🎯 Neo4j instead of PostgreSQL (graph queries!)
🎯 Consciousness reranking (basin + thoughtseed boost)
🎯 No pgvector (use Neo4j native vectors)
🎯 No FlashRank initially (simpler implementation)

### Why This is Superior
- **SurfSense**: Hybrid search with LLM reranker
- **Dionysus**: Hybrid search with consciousness context
- **Advantage**: Graph relationships + consciousness boost = smarter ranking
