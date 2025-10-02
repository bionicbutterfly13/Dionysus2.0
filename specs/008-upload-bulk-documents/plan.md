# Implementation Plan: Bulk Document Processing Pipeline with Debug Visualization

**Branch**: `008-upload-bulk-documents` | **Date**: 2025-09-30 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/008-upload-bulk-documents/spec.md`

## Summary
Implement comprehensive bulk document upload system with real-time pipeline visualization showing document flow through consciousness processing stages, memory type classification, database storage, and attractor basin formation. Primary focus: end-to-end observability via debug panel in Flux interface with hierarchical explorable data trees.

## Technical Context
**Language/Version**: Python 3.11+ (backend), TypeScript/React (frontend)
**Primary Dependencies**: FastAPI, React 18, WebSocket (real-time), D3.js (visualization), Redis (state), Neo4j (graph), Qdrant (vectors)
**Storage**: Multi-database (Neo4j graph + Qdrant vectors + Redis cache)
**Testing**: pytest (backend), Vitest (frontend), contract tests for pipeline stages
**Target Platform**: Web application (browser + server)
**Project Type**: Web (backend + frontend)
**Performance Goals**: Handle 100+ concurrent document uploads, <5s pipeline visualization update, real-time status streaming
**Constraints**: <500MB memory per batch, graceful degradation on overload, error recovery with retry
**Scale/Scope**: 1000+ documents per batch, 5-level hierarchical trees, 10+ pipeline stages to visualize

## Constitution Check
*Per constitution v1.0.0*

**✅ NumPy 2.0+ Compliance**: Backend uses NumPy 2.3.3 frozen environment for consciousness processing
**✅ TDD Standards**: All pipeline stages require contract tests before implementation
**✅ Environment Isolation**: Uses existing flux-backend-env with frozen dependencies
**✅ Code Complexity**: Modular pipeline stages, single responsibility per processor
**✅ Testing Protocols**: Contract + integration tests for upload → storage → visualization flow

**No violations detected** - Proceed with Phase 0

## Project Structure

### Documentation (this feature)
```
specs/008-upload-bulk-documents/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── contracts/           # Phase 1 output (API schemas)
└── tasks.md             # Phase 2 output (/tasks command)
```

### Source Code (repository root)
```
backend/
├── src/
│   ├── models/
│   │   ├── document_batch.py         # Batch upload model
│   │   └── pipeline_stage.py         # Pipeline stage tracking
│   ├── services/
│   │   ├── bulk_processor.py         # Batch processing orchestrator
│   │   ├── pipeline_tracker.py       # Real-time pipeline state
│   │   └── memory_classifier.py      # Memory type routing
│   └── api/
│       ├── routes/bulk_upload.py     # POST /api/bulk/upload
│       └── routes/pipeline_status.py # WebSocket /ws/pipeline
└── tests/
    ├── contract/
    │   ├── test_bulk_upload_api.py
    │   └── test_pipeline_status_ws.py
    └── integration/
        └── test_end_to_end_bulk.py

frontend/
├── src/
│   ├── components/
│   │   ├── BulkUploader.tsx         # Drag-drop upload UI
│   │   ├── PipelineVisualizer.tsx   # Real-time pipeline view
│   │   └── DataTreeExplorer.tsx     # Hierarchical data tree
│   ├── services/
│   │   ├── bulkUploadService.ts     # Upload API client
│   │   └── pipelineWebSocket.ts     # WebSocket client
│   └── pages/
│       └── DebugPanel.tsx            # Main debug interface
└── tests/
    └── components/
        └── BulkUploader.test.tsx
```

**Structure Decision**: Web application structure selected - backend API services for document processing with frontend visualization components for real-time monitoring.

## Phase 0: Outline & Research

### Research Tasks

1. **WebSocket real-time streaming patterns**
   - Research: Best practices for streaming pipeline status updates
   - Decision criteria: Latency <100ms, handle 1000+ concurrent connections
   - Output: WebSocket vs SSE comparison, backpressure handling

2. **D3.js hierarchical tree visualization**
   - Research: D3.js force-directed graphs vs tree layouts for pipeline visualization
   - Decision criteria: Interactive drill-down, 5+ nesting levels, performance with 1000+ nodes
   - Output: Visualization library selection and interaction patterns

3. **Bulk document processing architectures**
   - Research: Queue-based vs streaming processing for document batches
   - Decision criteria: Error recovery, partial failure handling, progress tracking
   - Output: Processing architecture (Celery vs asyncio vs Redis queue)

4. **Memory type classification algorithms**
   - Research: How to classify documents into episodic/semantic/procedural memory
   - Decision criteria: Accuracy, explainability, real-time decision tracking
   - Output: Classification approach and confidence scoring

5. **Attractor basin visualization patterns**
   - Research: Graph clustering visualization for attractor basins
   - Decision criteria: Real-time updates, semantic grouping display
   - Output: Visualization approach for basin formation

**Output**: research.md with all technical decisions documented

## Phase 1: Design & Contracts

### Data Model Entities (`data-model.md`)

1. **DocumentBatch**
   - Fields: batch_id, uploaded_files[], status, progress_percentage, created_at
   - Relationships: contains Documents, tracked by PipelineStages
   - State transitions: UPLOADED → PROCESSING → COMPLETED | FAILED

2. **PipelineStage**
   - Fields: stage_id, stage_name, document_id, status, started_at, completed_at, metadata
   - Relationships: belongs to Document, produces StageOutput
   - Validation: Stage ordering must follow pipeline sequence

3. **MemoryClassification**
   - Fields: document_id, memory_type (episodic|semantic|procedural), confidence, reasoning
   - Relationships: belongs to Document, influences DatabaseRoute
   - State: Classification decision point in pipeline

4. **HierarchicalDataNode**
   - Fields: node_id, parent_id, node_type, data_payload, children[]
   - Relationships: Tree structure for debug panel exploration
   - Validation: No circular references, max depth 10

### API Contracts (`contracts/`)

**POST /api/bulk/upload**
```yaml
request:
  multipart/form-data:
    files: File[]  # Up to 1000 files
    metadata:
      batch_name: string
      processing_options: object

response:
  201:
    batch_id: uuid
    status: "UPLOADED"
    file_count: integer
    estimated_time_ms: integer
```

**WebSocket /ws/pipeline/{batch_id}**
```yaml
messages:
  - type: "stage_update"
    document_id: uuid
    stage_name: string
    status: "started" | "completed" | "failed"
    progress: float
    metadata: object

  - type: "memory_classification"
    document_id: uuid
    memory_type: string
    confidence: float
    reasoning: string

  - type: "basin_formation"
    basin_id: uuid
    document_ids: uuid[]
    center_concept: string
```

**GET /api/bulk/{batch_id}/tree**
```yaml
response:
  200:
    root_node:
      node_id: uuid
      type: "batch"
      children:
        - node_id: uuid
          type: "document"
          data: {...}
          children: [...]  # Recursive structure
```

### Contract Tests

**test_bulk_upload_api.py**
- Test bulk upload accepts 1-1000 files
- Test validation rejects oversized batches
- Test returns valid batch_id and status
- Test handles file type validation

**test_pipeline_status_ws.py**
- Test WebSocket connection establishment
- Test stage_update message format
- Test real-time progress streaming
- Test connection recovery on failure

**test_hierarchical_tree.py**
- Test tree structure validation
- Test max depth enforcement
- Test parent-child relationship integrity
- Test node data payload schemas

### Quickstart (`quickstart.md`)

1. Upload bulk documents via Flux UI drag-drop
2. Monitor pipeline in debug panel real-time view
3. Explore hierarchical data tree showing transformations
4. Inspect memory classifications and database routes
5. View attractor basin formations and clustering

## Phase 2: Task Generation Approach

Tasks will be organized into TDD phases:

**Phase 3.1: Setup**
- Setup WebSocket infrastructure
- Configure bulk upload endpoints
- Setup debug panel UI components

**Phase 3.2: Tests First (TDD RED)**
- Contract tests for bulk upload API (MUST FAIL)
- Contract tests for WebSocket streaming (MUST FAIL)
- Contract tests for tree structure API (MUST FAIL)
- Integration tests for end-to-end flow (MUST FAIL)

**Phase 3.3: Core Implementation (TDD GREEN)**
- Implement DocumentBatch model
- Implement bulk upload processor
- Implement pipeline stage tracker
- Implement WebSocket streaming service
- Implement memory classifier
- Implement hierarchical tree builder

**Phase 3.4: Frontend Integration**
- Implement BulkUploader component
- Implement PipelineVisualizer with D3.js
- Implement DataTreeExplorer
- Connect WebSocket for real-time updates

**Phase 3.5: Polish**
- Performance optimization (batch processing)
- Error recovery and retry logic
- UI/UX refinement for debug panel
- Documentation and deployment

## Dependencies & Integration Points

**Depends on**:
- Spec 006: Query system (for database access patterns)
- Existing document processing pipeline
- Neo4j schema from Spec 001
- Consciousness processing from Spec 024

**Integrates with**:
- ThoughtSeed system (extracted package)
- Daedalus gateway (document ingestion)
- Attractor basin dynamics
- Memory formation subsystems

## Progress Tracking

- [x] Initial constitution check - PASS
- [x] Technical context defined
- [ ] Phase 0: Research complete (awaiting /tasks command trigger)
- [ ] Phase 1: Data model + contracts generated
- [ ] Post-design constitution re-check
- [ ] Phase 2: Tasks generated via /tasks command

**Status**: Ready for Phase 0 research execution
