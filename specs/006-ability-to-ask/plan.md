# Implementation Plan: Research Engine Query Interface

**Branch**: `006-ability-to-ask` | **Date**: 2025-09-30 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/006-ability-to-ask/spec.md`

## Summary
Implement a natural language query interface that searches comprehensive Neo4j graph database and vector database simultaneously, synthesizing results into well-constructed responses. System integrates with existing ThoughtSeed active inference and consciousness processing to provide intelligent, context-aware answers.

## Technical Context
**Language/Version**: Python 3.11+
**Primary Dependencies**: neo4j-driver, qdrant-client, sentence-transformers, LangGraph, FastAPI
**Storage**: Neo4j (graph), Qdrant (vectors), Redis (cache)
**Testing**: pytest with async support
**Target Platform**: Backend API + CLI interface
**Project Type**: Web application (backend API)
**Performance Goals**: <2s query response, handle 10 concurrent queries
**Constraints**: Must use existing database connections, preserve ThoughtSeed integration
**Scale/Scope**: Single-user research system, 10k+ documents/architectures

## Constitution Check
✅ **PASS** - Extends existing systems without adding complexity
✅ **NumPy 2.0+**: Not applicable (no NumPy dependency in query system)
✅ **TDD**: Will follow strict TDD with failing tests first
✅ **Integration**: Uses extracted packages (thoughtseed-active-inference, daedalus-gateway)

## Project Structure

### Documentation
```
specs/006-ability-to-ask/
├── plan.md              # This file
├── data-model.md        # Query/Response models
├── contracts/           # API contracts
└── tasks.md             # Implementation tasks
```

### Source Code
```
backend/
├── src/
│   ├── api/routes/
│   │   └── query.py                    # Query API endpoints
│   ├── services/
│   │   ├── query_engine.py             # Main query orchestration
│   │   ├── neo4j_searcher.py           # Graph database search
│   │   ├── vector_searcher.py          # Semantic vector search
│   │   └── response_synthesizer.py    # Multi-source synthesis
│   └── models/
│       ├── query.py                    # Query model
│       └── response.py                 # Response model
└── tests/
    ├── contract/
    │   ├── test_query_api_post.py
    │   └── test_query_response_schema.py
    └── integration/
        ├── test_neo4j_vector_integration.py
        └── test_end_to_end_query.py
```

## Phase 0: Outline & Research ✅

**Decisions Made**:
1. **Query Processing**: Use LangGraph for intelligent query decomposition and routing
2. **Database Search**: Parallel search across Neo4j + Qdrant, merge ranked results
3. **Response Synthesis**: Meta-ToT active inference for coherent response generation
4. **Caching**: Redis for frequent query results (5-minute TTL)
5. **ThoughtSeed Integration**: Each query creates a ThoughtSeed that flows through attractor basins

**Alternatives Considered**:
- Sequential search (rejected: too slow)
- Single database only (rejected: loses semantic/graph complementarity)
- External LLM for synthesis (rejected: keep processing local with active inference)

## Phase 1: Design & Contracts

### Data Model Entities

**Query**:
```python
- query_id: str
- question: str  # Natural language question
- user_id: Optional[str]
- timestamp: datetime
- context: Dict[str, Any]  # Session context for follow-ups
- thoughtseed_id: Optional[str]  # Associated ThoughtSeed
```

**SearchResult**:
```python
- result_id: str
- source: str  # "neo4j" | "qdrant"
- content: str
- relevance_score: float
- metadata: Dict[str, Any]
- relationships: List[str]  # For graph results
```

**Response**:
```python
- response_id: str
- query_id: str
- answer: str  # Synthesized response
- sources: List[SearchResult]
- confidence: float
- processing_time_ms: int
- thoughtseed_trace: Optional[Dict]  # Consciousness processing trace
```

### API Contracts

**POST /api/query**
```yaml
Request:
  question: string (required)
  context: object (optional)
Response:
  200:
    response_id: string
    answer: string
    sources: array
    confidence: number
    processing_time_ms: number
  400: Invalid question format
  503: Database unavailable
```

### Test Scenarios
1. Simple factual question → Single-source response
2. Complex multi-hop question → Graph traversal + semantic search
3. Ambiguous question → Clarification in response
4. No results found → Informative "no match" response
5. Concurrent queries → All complete within performance limits

## Phase 2: Task Planning Approach

**Task Generation Strategy**:
1. **Setup** (T001-T003): Database connections, test fixtures
2. **Tests First** (T004-T009): Contract + integration tests (TDD)
3. **Core Implementation** (T010-T015): Query engine, searchers, synthesizer
4. **Integration** (T016-T020): ThoughtSeed integration, API endpoints
5. **Polish** (T021-T025): Performance optimization, documentation

**Ordering**:
- TDD: All tests before implementation
- Parallel: Searchers can be implemented independently [P]
- Sequential: Synthesis depends on searchers

**Estimated**: 25 tasks total

## Progress Tracking

**Phase Status**:
- [x] Phase 0: Research complete
- [x] Phase 1: Design complete
- [x] Phase 2: Task planning approach defined
- [ ] Phase 3: Tasks generated (/tasks command - NEXT)
- [ ] Phase 4: Implementation
- [ ] Phase 5: Validation

**Constitution Check**:
- [x] Initial: PASS
- [x] Post-Design: PASS
- [x] All NEEDS CLARIFICATION resolved (clarifications below)

## Clarifications Resolved

From spec NEEDS CLARIFICATION items:
1. **Question complexity**: Support up to 500 token questions, decompose into sub-queries if needed
2. **Response formatting**: Markdown with citations, max 2000 tokens
3. **Concurrent load**: Target 10 concurrent queries
4. **Authentication**: Use existing session-based auth (not in scope for this spec)
5. **Performance targets**: <2s for simple queries, <5s for complex multi-hop queries

---
*Based on Constitution v1.0.0 - NumPy 2.0+ compliant, TDD workflow*
