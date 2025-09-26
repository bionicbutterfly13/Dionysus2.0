# Tasks: Self-Teaching Consciousness Emulator (Flux)

**Input**: Design documents from `/specs/002-i-m-building/`
**Prerequisites**: research.md, data-model.md, contracts/, quickstart.md
**Tech Stack**: Python 3.11, ASI-Arch framework, Neo4j (canonical storage), Qdrant/SQLite (embeddings), Redis (curiosity signals), Ollama/LLaMA (local inference)
**Architecture**: Preserve ASI-Arch core pipeline with ThoughtSeed consciousness guidance enhancements

## Phase 3.0: HIGH PRIORITY - Daedalus Legacy Integration 🏛️
**CRITICAL PRIORITY**: Multi-modal processing pipeline and 10K+ node knowledge graph migration

- [x] **T000.1 [P]** Audit Daedalus legacy system - catalog perceptual processing, agent hierarchy, and memory systems in `backend/src/legacy/daedalus_audit.md`
- [ ] **T000.2** Create protected Daedalus integration layer in `backend/src/legacy/daedalus_bridge/` with crash-proof isolation (binary packaging if needed)
- [ ] **T000.3 [P]** Migrate hybrid memory system (MEM1 + ThoughtSeed) from `dionysus-source/hybrid_memory_wrapper.py` to `backend/src/services/hybrid_memory_service.py`
- [ ] **T000.4 [P]** Migrate enhanced perceptual gateway for video/audio/image processing from `dionysus-source/active_inference/enhanced_perceptual_gateway.py`
- [ ] **T000.5** Import existing knowledge graph (10,808 nodes: memories + papers) from Neo4j diagnostics with "legacy" status flags
- [ ] **T000.6** Migrate paper database and research memories (episodic/semantic/procedural .jsonl files) as DocumentArtifacts with thoughtseed_version="1.0"
- [ ] **T000.6a** Implement selective re-processing system: legacy data can be upgraded to ThoughtSeed 2.0 features on-demand
- [ ] **T000.6b** Create data migration dashboard showing legacy vs enhanced content statistics
- [ ] **T000.7 [P]** Integrate Daedalus constitutional controller and agent delegation patterns with Flux service layer
- [ ] **T000.8** Bridge Daedalus LangGraph patterns (state management + conditional edges) with Flux ThoughtSeed pipeline
- [ ] **T000.9** Create multi-modal processing endpoints: POST /api/v1/media/{video|audio|image} with episodic memory creation
- [ ] **T000.10** Test complete pipeline: video upload → perceptual processing → consciousness traces → episodic memories → knowledge graph
- [ ] **T000.11 [P]** Migrate LangExtract + Ultimate Preprocessing Pipeline from Dionysus - includes GEPA, LangGraph, Nemori, CPA-Meta-ToT (75% reasoning boost), River Metaphor flow
- [ ] **T000.12** Document import quality assessment system with before/after comparison metrics
- [ ] **T000.13 [URGENT]** Fix all timestamp systems - ensure episodic memories use correct date (September 25, 2025) instead of hardcoded old dates

## Phase 3.1: Setup
- [x] **T001** Initialize Flux backend project skeleton in `backend/src/` (FastAPI app, poetry/requirements sync, env loaders) per plan.md
- [x] **T002** Install and pin backend dependencies (FastAPI, Neo4j driver, Redis, Qdrant client, LangGraph, context-engineering library, Ollama bindings) in `backend/pyproject.toml` and lock file
- [x] **T003 [P]** Configure backend linting/formatting (`ruff`, `black`, `isort`) and CI hooks in `backend/pyproject.toml` and `.github/workflows/lint.yml`
- [x] **T004** Scaffold Flux frontend workspace in `frontend/` (React/Vite) and install SurfSense-inspired UI component dependencies with attribution markers
- [x] **T005 [P]** Add shared configuration files (`configs/flux.yaml`, `.env.example`) capturing Neo4j, Qdrant, Redis, Ollama endpoints and constitutional toggles

## Phase 3.2: Tests First (TDD)
**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation.**
- [x] **T006 [P]** Contract test for `POST /api/v1/documents` in `backend/tests/contract/test_documents_post.py`
- [x] **T007 [P]** Contract test for curiosity mission endpoints (`POST/GET/PATCH /api/v1/curiosity/missions`) in `backend/tests/contract/test_curiosity_missions.py`
- [x] **T008 [P]** Contract test for visualization WebSocket `/ws/v1/visualizations` in `backend/tests/contract/test_visualization_ws.py`
- [x] **T009 [P]** Integration test for Flux document ingestion flow (upload → ThoughtSeed trace → evaluation frame) in `backend/tests/integration/test_document_ingestion_flow.py`
- [x] **T010 [P]** Integration test for curiosity mission lifecycle (gap trigger → web crawl → trust scoring → replay scheduling) in `backend/tests/integration/test_curiosity_lifecycle.py`
- [x] **T011 [P]** Integration test for visualization stream (graph + card stack updates + mosaic state) in `backend/tests/integration/test_visualization_stream.py`
- [x] **T012 [P]** Integration test for nightly dreaming replay (dreaming flag + evaluation frame) in `backend/tests/integration/test_dream_replay.py`

## Phase 3.3: Core Implementation (ONLY after tests are failing)
### Models (Data-Layer)
- [x] **T013 [P]** Implement `UserProfile` model in `backend/src/models/user_profile.py`
- [x] **T014 [P]** Implement `AutobiographicalJourney` model in `backend/src/models/autobiographical_journey.py`
- [x] **T015 [P]** Implement `EventNode` model with Mosaic observation schema in `backend/src/models/event_node.py`
- [x] **T016 [P]** Implement `DocumentArtifact` model (mock data flags, hash) in `backend/src/models/document_artifact.py`
- [x] **T017 [P]** Implement `ConceptNode` model in `backend/src/models/concept_node.py`
- [x] **T018 [P]** Implement `ThoughtSeedTrace` model (consciousness_state incl. `dreaming`) in `backend/src/models/thoughtseed_trace.py`
- [x] **T019 [P]** Implement `CuriosityMission` model in `backend/src/models/curiosity_mission.py`
- [x] **T020 [P]** Implement `EvaluationFrame` model enforcing four constitutional fields in `backend/src/models/evaluation_frame.py`
- [x] **T021 [P]** Implement `VisualizationState` model for dashboard snapshots in `backend/src/models/visualization_state.py`

### Services & Pipelines
- [ ] **T022** Build Neo4j repository layer for all models in `backend/src/repositories/graph_repository.py`
- [ ] **T023** Create embedding sync service (Qdrant/SQLite) in `backend/src/services/embedding_sync_service.py`
- [ ] **T024** Implement ThoughtSeed pipeline service (context-engineering attractor activation + evaluation frames) in `backend/src/services/thoughtseed_pipeline.py`
- [ ] **T025** Implement curiosity mission orchestrator (PurpleLexica + Dionysus crawlers) in `backend/src/services/curiosity_service.py`
- [ ] **T026** Implement nightly replay scheduler (dreaming state, NEMORI decay) in `backend/src/services/replay_scheduler.py`
- [ ] **T027** Implement evaluation feedback logger (What’s good/broken/etc.) in `backend/src/services/evaluation_service.py`
- [ ] **T028** Implement visualization broadcaster (graph updates, curiosity signals, mosaic state) in `backend/src/services/visualization_stream.py`

### API & Interface
- [ ] **T029** Implement `POST /api/v1/documents` endpoint in `backend/src/api/routes/documents.py`
- [ ] **T030** Implement curiosity mission endpoints (`POST/GET/PATCH`) in `backend/src/api/routes/curiosity.py`
- [ ] **T031** Implement visualization WebSocket `/ws/v1/visualizations` in `backend/src/api/routes/visualization.py`
- [x] **T032** Add backend dependency wiring (routers, services, repositories) in `backend/src/app_factory.py`

### Frontend
- [ ] **T033** Create Flux upload dashboard (card stack + SurfSense-inspired UI) in `frontend/src/pages/UploadDashboard.tsx`
- [ ] **T034** Implement curiosity monitor UI (replay controls, trust scoring) in `frontend/src/pages/CuriosityMonitor.tsx`
- [ ] **T035** Implement visualization canvas (graph + mosaic gauges) in `frontend/src/pages/VisualizationCanvas.tsx`
- [ ] **T036** Wire frontend API/WebSocket clients in `frontend/src/services/apiClient.ts`

## Phase 3.4: Integration
- [ ] **T037** Configure Neo4j driver + AutoSchema KG sync in `backend/src/integrations/neo4j_setup.py`
- [ ] **T038** Configure Qdrant client and embedding pipeline in `backend/src/integrations/qdrant_setup.py`
- [ ] **T039** Configure Redis streams + durability mirroring in `backend/src/integrations/redis_setup.py`
- [ ] **T040** Integrate Ollama local inference wrapper (summaries, embeddings) in `backend/src/integrations/ollama_adapter.py`
- [ ] **T041** Migrate critical Dionysus ThoughtSeed modules into `backend/src/legacy/dionysus_adapter/` ensuring redundancy audit
- [ ] **T042** Hook SurfSense component styles into Flux frontend with attribution in `frontend/src/styles/fluxTheme.css`

## Phase 3.5: Polish & Validation
- [ ] **T043 [P]** Add unit tests for services (evaluation, curiosity, visualization) in `backend/tests/unit/test_services.py`
- [ ] **T044 [P]** Add unit tests for Mosaic observation serialization in `backend/tests/unit/test_mosaic_serialization.py`
- [ ] **T045 [P]** Document mock-data disclosure behavior in `docs/mock-data-policy.md`
- [ ] **T046** Update `README.md` with Flux vs SurfSense attribution and local-first instructions
- [ ] **T047 [P]** Performance smoke test for ingestion throughput (50 docs/hr target) in `backend/tests/performance/test_ingestion_performance.py`
- [ ] **T048** Manual validation following `quickstart.md`: rerun ingestion, curiosity, visualization, dreaming scenarios and log evaluation frames

## Dependencies
- **HIGHEST PRIORITY**: Daedalus integration (T000.1-T000.10) - enables multi-modal processing and 10K+ knowledge graph
- Setup (T001-T005) before any tests or implementation
- Contract & integration tests (T006-T012) must be failing before starting T013+ (TDD)
- Model tasks (T013-T021) block repository/service layer tasks (T022-T028)
- Services (T022-T028) block API endpoints (T029-T032)
- API endpoints (T029-T032) block frontend tasks (T033-T036) and integration wiring (T037-T041)
- Integration wiring (T037-T041) block polish tasks (T043-T048)
- **Daedalus integration can run in parallel with existing tasks** (marked [P]) to accelerate development

## Parallel Execution Example
```
# PHASE 3.0: HIGH PRIORITY - Launch Daedalus integration tasks in parallel:
specify.tasks.run(T000.1)  # Audit Daedalus system
specify.tasks.run(T000.3)  # Migrate hybrid memory system
specify.tasks.run(T000.4)  # Migrate perceptual gateway
specify.tasks.run(T000.7)  # Integrate constitutional controller

# After setup complete and before implementation, run contract/integration test tasks in parallel:
specify.tasks.run(T006)
specify.tasks.run(T007)
specify.tasks.run(T008)
specify.tasks.run(T009)
specify.tasks.run(T010)
specify.tasks.run(T011)
specify.tasks.run(T012)

# After tests fail, launch model tasks concurrently:
specify.tasks.run(T013)
specify.tasks.run(T014)
specify.tasks.run(T015)
specify.tasks.run(T016)
specify.tasks.run(T017)
specify.tasks.run(T018)
specify.tasks.run(T019)
specify.tasks.run(T020)
specify.tasks.run(T021)
```

## Notes
- `[P]` denotes tasks safe for parallel execution (different files, no direct dependencies)
- Always ensure tests written in Phase 3.2 fail before implementing related functionality
- Reference design artifacts (data-model, contracts, quickstart) during each task to maintain constitutional compliance
- Commit after each task completion; document evaluation frames for every workflow validation
- Use `/tasks run <ID>` (or equivalent agent command) to execute tasks incrementally
