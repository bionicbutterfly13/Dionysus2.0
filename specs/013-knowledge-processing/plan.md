# Implementation Plan: Knowledge Processing & Migration

**Branch**: `013-knowledge-processing` | **Date**: 2025-09-30 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/013-knowledge-processing/spec.md`

## Summary
Comprehensive knowledge base content extraction and processing with depth-based crawling queues, Dionysus legacy migration, and narrative extraction system. Implements multi-tier processing architecture with time ceilings per depth level, vector+graph database linking, and enhanced consciousness processing for maximum data richness.

## Technical Context
**Language/Version**: Python 3.11+
**Primary Dependencies**: FastAPI, BeautifulSoup4/Scrapy (crawling), OpenAI API (embeddings), neo4j-driver, qdrant-client, Redis
**Storage**: Neo4j (graph) + Qdrant (vectors) + Redis (queues) + PostgreSQL (legacy Dionysus data)
**Testing**: pytest with contract tests, integration tests for crawling pipelines
**Target Platform**: Linux server, distributed crawling
**Project Type**: Backend service with queue-based processing
**Performance Goals**:
  - Depth 1: 5min/URL ceiling
  - Depth 2: 15min/URL ceiling
  - Depth 3: 30min/URL ceiling
  - Depth 4: 60min/URL ceiling
  - Depth 5: 120min/URL ceiling
**Constraints**: Respect robots.txt, rate limiting (1 req/sec default), graceful timeout handling
**Scale/Scope**: ~1000 URLs in initial queue, 576 pages for LangGraph docs (depth 4), 170 pages for SurfSense (depth 2)

## Constitution Check
*Per constitution v1.0.0*

**✅ NumPy 2.0+ Compliance**: Processing pipeline uses NumPy 2.3.3 for consciousness features
**✅ TDD Standards**: Contract tests required for crawling, extraction, and migration services
**✅ Environment Isolation**: Uses flux-backend-env with frozen dependencies
**✅ Code Complexity**: Modular extractors, single-purpose queue processors
**✅ Testing Protocols**: Contract tests for each depth queue, integration tests for full pipeline

**No violations detected** - Proceed with Phase 0

## Project Structure

### Documentation (this feature)
```
specs/013-knowledge-processing/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── contracts/           # Phase 1 output
└── tasks.md             # Phase 2 output (/tasks command)
```

### Source Code (repository root)
```
backend/
├── src/
│   ├── models/
│   │   ├── crawl_job.py              # URL crawl job model
│   │   ├── depth_queue.py            # Depth-based queue model
│   │   └── narrative_pattern.py      # Narrative extraction result
│   ├── services/
│   │   ├── knowledge_crawler.py      # Web crawling service
│   │   ├── depth_queue_manager.py    # Multi-tier queue orchestrator
│   │   ├── narrative_extractor.py    # Causal/process/group extraction
│   │   ├── dionysus_migrator.py      # Legacy data migration
│   │   └── vector_graph_linker.py    # Link vector DB ↔ graph DB
│   ├── api/
│   │   ├── routes/crawl.py           # POST /api/crawl/enqueue
│   │   └── routes/knowledge.py       # GET /api/knowledge/status
│   └── workers/
│       ├── depth_1_worker.py         # 5min ceiling worker
│       ├── depth_2_worker.py         # 15min ceiling worker
│       ├── depth_3_worker.py         # 30min ceiling worker
│       ├── depth_4_worker.py         # 60min ceiling worker
│       └── depth_5_worker.py         # 120min ceiling worker
└── tests/
    ├── contract/
    │   ├── test_crawl_api.py
    │   ├── test_depth_queues.py
    │   └── test_narrative_extraction.py
    └── integration/
        └── test_end_to_end_knowledge_processing.py
```

**Structure Decision**: Backend service structure with dedicated worker processes for each depth tier, Redis-based queue management, and modular extraction services.

## Phase 0: Outline & Research

### Research Tasks

1. **Web crawling frameworks comparison**
   - Research: Scrapy vs BeautifulSoup4 vs Playwright for depth-based crawling
   - Decision criteria: Robots.txt compliance, JavaScript rendering, rate limiting, timeout control
   - Output: Framework selection with justification

2. **Depth-based queue architecture patterns**
   - Research: Priority queues vs separate Redis lists for multi-tier processing
   - Decision criteria: Fair scheduling, depth isolation, timeout enforcement
   - Output: Queue architecture design

3. **Narrative extraction from unstructured text**
   - Research: NLP patterns for causal relationships, process flows, group memberships
   - Decision criteria: Accuracy, explainability, computational cost
   - Output: Extraction algorithm selection (spaCy, transformers, GPT-4)

4. **Vector-Graph database linking strategies**
   - Research: Best practices for maintaining sync between Qdrant embeddings and Neo4j nodes
   - Decision criteria: Query performance, update consistency, relationship traversal
   - Output: Linking schema and update protocol

5. **Dionysus PostgreSQL migration approach**
   - Research: Safe migration patterns for preserving legacy consciousness data
   - Decision criteria: Zero data loss, incremental migration, rollback capability
   - Output: Migration script architecture

**Output**: research.md with all technical decisions documented

## Phase 1: Design & Contracts

### Data Model Entities (`data-model.md`)

1. **CrawlJob**
   - Fields: job_id, url, depth_level (1-5), status, time_limit_seconds, started_at, completed_at, content_hash
   - Relationships: belongs to DepthQueue, produces ExtractedContent
   - State transitions: QUEUED → PROCESSING → COMPLETED | TIMEOUT | FAILED
   - Validation: URL format, depth in [1,5], time_limit matches depth tier

2. **DepthQueue**
   - Fields: queue_name, depth_level, time_ceiling_seconds, active_jobs, completed_jobs, failed_jobs
   - Relationships: contains CrawlJobs
   - Validation: Time ceiling enforcement, max concurrent jobs per depth

3. **NarrativePattern**
   - Fields: pattern_id, source_url, pattern_type (causal|process|group), entities[], relationships[], confidence
   - Relationships: extracted from Document, stored in Neo4j as relationships
   - Validation: Pattern structure matches type schema

4. **VectorGraphLink**
   - Fields: vector_id (Qdrant), graph_node_id (Neo4j), embedding_model, created_at
   - Relationships: Bidirectional link between vector and graph databases
   - Validation: Both IDs must exist in respective databases

5. **DionysusLegacyData**
   - Fields: legacy_id, data_type, raw_json, migration_status, enhanced_content
   - Relationships: migrates to new models (Document, Concept, Relationship)
   - State: PENDING → MIGRATED → ENHANCED

### API Contracts (`contracts/`)

**POST /api/crawl/enqueue**
```yaml
request:
  url: string  # URL to crawl
  depth: integer  # 1-5
  options:
    follow_links: boolean
    max_pages: integer

response:
  201:
    job_id: uuid
    depth_level: integer
    estimated_time_seconds: integer
    queue_position: integer
```

**GET /api/knowledge/status**
```yaml
response:
  200:
    queues:
      - depth: 1
        active: 5
        queued: 120
        completed: 450
        time_ceiling: 300
      - depth: 2
        active: 3
        queued: 45
        completed: 78
        time_ceiling: 900
      # ... depths 3-5
```

**POST /api/migration/dionysus**
```yaml
request:
  dry_run: boolean
  preserve_legacy: boolean

response:
  200:
    migration_id: uuid
    total_records: integer
    estimated_time_hours: float
    status: "initiated"
```

### Contract Tests

**test_crawl_api.py**
- Test enqueue validates URL format
- Test depth level validation (1-5 only)
- Test time ceiling assignment per depth
- Test queue position calculation

**test_depth_queues.py**
- Test depth 1 enforces 5min timeout
- Test depth 2 enforces 15min timeout
- Test depth 3 enforces 30min timeout
- Test depth 4 enforces 60min timeout
- Test depth 5 enforces 120min timeout
- Test queue isolation (depth 1 doesn't block depth 2)

**test_narrative_extraction.py**
- Test causal relationship extraction
- Test process flow extraction
- Test group membership detection
- Test confidence scoring
- Test Neo4j relationship creation

**test_vector_graph_linking.py**
- Test vector embedding → graph node linking
- Test bidirectional lookup (vector → graph, graph → vector)
- Test link consistency on updates
- Test orphaned link cleanup

### Quickstart (`quickstart.md`)

1. Enqueue URLs from Knowledge Base with depth assignment
2. Monitor depth-based queue progress in dashboard
3. Review extracted narratives and concept relationships
4. Run Dionysus migration with dry-run first
5. Query enhanced knowledge graph with linked vectors

## Phase 2: Task Generation Approach

Tasks will be organized into TDD phases:

**Phase 3.1: Setup**
- Setup Redis depth queues (5 separate queues)
- Configure worker processes per depth tier
- Setup Dionysus PostgreSQL connection

**Phase 3.2: Tests First (TDD RED)**
- Contract tests for crawl API (MUST FAIL)
- Contract tests for depth queue timeouts (MUST FAIL)
- Contract tests for narrative extraction (MUST FAIL)
- Contract tests for vector-graph linking (MUST FAIL)
- Integration test for full pipeline (MUST FAIL)

**Phase 3.3: Core Implementation (TDD GREEN)**
- Implement CrawlJob and DepthQueue models
- Implement depth queue manager with timeout enforcement
- Implement web crawler with robots.txt compliance
- Implement narrative pattern extractor
- Implement vector-graph linker
- Implement Dionysus migrator

**Phase 3.4: Worker Implementation**
- Implement depth 1 worker (5min ceiling)
- Implement depth 2 worker (15min ceiling)
- Implement depth 3 worker (30min ceiling)
- Implement depth 4 worker (60min ceiling)
- Implement depth 5 worker (120min ceiling)

**Phase 3.5: Migration & Polish**
- Run Dionysus migration scripts
- Enhance migrated data with context engineering
- Performance optimization for crawling
- Add monitoring and alerting
- Documentation and deployment

## Dependencies & Integration Points

**Depends on**:
- Spec 001: Neo4j unified knowledge graph schema
- Spec 006: Query system for knowledge retrieval
- Existing Qdrant vector database setup
- Dionysus legacy PostgreSQL database

**Integrates with**:
- ThoughtSeed system (consciousness processing)
- Context engineering pipeline (data enhancement)
- Attractor basin formation (clustering)
- Knowledge Base UI (status display)

**Initial URLs to Process** (from spec.md):
```
Depth 2:
- https://frontiersin.org (100 pages est.)
- https://blog.futuresmart.ai (86 pages est.)
- https://www.biorxiv.org/content/10.1101/2024.08.18.608439v1.full (99 pages)
- https://github.com/MODSetter/SurfSense.git (170 pages)

Depth 4:
- https://langchain-ai.github.io/langgraph/ (576 pages)

Depth 1:
- https://sciencedirect.com (1 page)
```

## Progress Tracking

- [x] Initial constitution check - PASS
- [x] Technical context defined
- [ ] Phase 0: Research complete (awaiting /tasks command trigger)
- [ ] Phase 1: Data model + contracts generated
- [ ] Post-design constitution re-check
- [ ] Phase 2: Tasks generated via /tasks command

**Status**: Ready for Phase 0 research execution
