# Phase 0 Research: Self-Teaching Consciousness Emulator

**Branch**: `002-i-m-building`  
**Date**: 2025-09-24  
**Source Specification**: [spec.md](spec.md)  

## 1. Outstanding Clarifications
| Topic | Question | Status |
|-------|----------|--------|
| Document ingestion limits | [QUESTION] What are the maximum file size, supported formats, and concurrent upload limits for Flux? | Deferred - focus on core upload pipeline first |
| Curiosity vetting | [QUESTION] How should the system score/verify the trustworthiness of sources discovered by the curiosity engine? | Deferred - focus on core upload pipeline first |
| Strategy conflict resolution | [QUESTION] When curiosity-derived strategies contradict existing learned policies, who/what decides which path to adopt (auto-merge, human approval, hybrid)? | Deferred - focus on core upload pipeline first |
| Redis resilience | [QUESTION] Desired behavior if Redis is unavailable or memory pressure occurs (queue persistence, backpressure, graceful degradation)? | **RESOLVED**: Suspend curiosity loops if Redis unavailable; don't interfere with core performance |
| Performance targets | [QUESTION] What throughput/latency goals should ingestion, summarization, and curiosity scans meet on local hardware? | Deferred - focus on core upload pipeline first |
| Corpus scale | [QUESTION] Expected maximum size of the user's document collection (total docs, GB, growth rate)? | Deferred - focus on core upload pipeline first |

## 1.1 Redis Behavior Decisions
| Decision | Rationale | Implementation Notes |
|----------|-----------|---------------------|
| **Durable persistence**: Store curiosity signals in graph database for long-term memory integration | Curiosity becomes part of episodic memory structure, enabling replay based on relevance | Use Neo4j for structured storage, sync with Redis streams for real-time processing |
| **Relevance-based replay**: Automatic replay during idle times, especially at night | Mimics hippocampal replay models and circadian rhythms | Timer-based replay during low-activity periods, triggered by basin activation |
| **Suspend on Redis failure**: Don't buffer to disk, suspend curiosity loops | Prevents interference with core system performance | Curiosity loops run during mind-wandering phases, not critical path |
| **Episodic memory decay**: Curiosity signals decay like episodic memories | Maps to human behavior - recent curiosity more salient in basins | Use NEMORI models for decay rates, treat curiosity as episodic memory type |
| **Dreaming integration**: Nightly insights go directly to knowledge graph with flags | Dreams are episodic memories with ThoughtSeed consciousness state markers | Add "dreaming" to ThoughtSeed consciousness state enumeration |

## 1.2 Priority Focus
**HIGHEST PRIORITY**: Get Flux interface (our implementation) working with document upload → conscious processing → graph database pipeline. We'll draw inspiration from SurfSense (open source) components while building our own Flux-branded system. All other details deferred until core upload functionality is demonstrated.

## 2. Asset & Capability Inventory
- **Dionysus Modules**: ThoughtSeed processing, curiosity engine scaffolding, perceptual gateway, attractor visualizations, memory orchestrator adapters. Task: catalog module entry points and dependencies to import into ASI-Arch while eliminating redundancy.  
- **SurfSense Components** (open source): UI card stacks, knowledge graph explorers, upload flows, observability widgets. Task: identify reusable components from SurfSense to adapt for Flux (our implementation) with proper attribution.  
- **ASI-Arch Core**: Knowledge graph export (Neo4j/Qdrant), Pipeline evolve modules, context-engineering library, Redis-backed learning captures. Task: confirm schemas, API endpoints, and evaluate overlap with Dionysus implementations.  
- **Local Runtime Tooling**: Ollama + LLaMA models, local Neo4j/Qdrant deployments, Redis server configuration, OLA orchestration scripts. Task: document required versions, resource usage, and startup procedures.

## 2.1 Existing Dionysus Flux Implementation Audit
**CRITICAL**: Examine existing Dionysus Flux implementation (web-based + desktop) that already handles document upload → conscious processing → graph database pipeline. Priority tasks:
- Map current upload flows and processing steps
- Identify what's working vs what needs enhancement
- Assess integration points with AutoSchema KG
- Plan migration strategy to ASI-Arch without breaking existing functionality

## 3. Research Tasks
1. **Knowledge Graph as Source of Truth**  
   - Investigate best practices for enforcing Neo4j as canonical data store while syncing embeddings and metadata to Qdrant/SQLite.  
   - Evaluate existing `UNIFIED_DATABASE_MIGRATION_SUMMARY.md` and `KNOWLEDGE_GRAPH_ARCHITECTURE_SPEC.md` for alignment.  

2. **Context Engineering Enforcement**  
   - Review `extensions/context_engineering/` to ensure ingestion pipelines trigger attractor basins for every document.  
   - Study `CONTEXT_ENGINEERING_INTEGRATION_SPEC.md` for mandatory flow states and consciousness detection.  

3. **Curiosity Engine Enhancements**  
   - Examine Dionysus curiosity modules for strategy mutation and external crawling.  
   - Research reputation scoring methods suitable for local/offline validation.  

4. **Local Inference Performance**  
   - Benchmark Ollama-hosted LLaMA for batch summarization on reference hardware (documented CPU/GPU requirements).  
   - Identify optimization strategies (quantization levels, batching, streaming).  

5. **Privacy & Security**  
   - Determine safeguards for storing user corpora locally (encryption options, backup strategies).  
   - Review SurfSense/ASI-Arch guidance for desktop deployments.  

## 4. Decisions & Rationale (Updated from Implementation Testing)

### 4.1 Core Dependencies Resolution
| Decision | Rationale | Alternatives Considered | Implementation Status |
|----------|-----------|-------------------------|----------------------|
| **FastAPI + Uvicorn** | Async support, OpenAPI docs, type safety with Pydantic | Django (too heavy), Flask (no async) | ✅ Implemented & Tested |
| **Pydantic V2** | Type validation, serialization, constitutional compliance | Manual validation, Marshmallow | ✅ Implemented (V2 warnings need fixing) |
| **Neo4j Python Driver 5.15+** | Graph database for knowledge relationships, ACID compliance | NetworkX (memory only), ArangoDB (less mature) | ❌ Missing - needs installation |
| **Redis 5.0+ with aioredis** | Async operations, pub/sub for curiosity signals, persistence | In-memory only, RabbitMQ (overkill) | ❌ Missing - needs installation |
| **Qdrant Client 1.7+** | Local vector storage, hybrid search capabilities | Pinecone (cloud), Weaviate (resource heavy) | ❌ Missing - needs installation |
| **Ollama Python Client** | Local LLM inference, privacy compliance, no API costs | OpenAI API (privacy issues), Hugging Face (complex setup) | ❌ Missing - needs installation |
| **NumPy 1.24.4** | Consciousness processing, ThoughtSeed calculations | Built-in math (insufficient), SciPy (overkill) | ❌ Missing - needs installation |

### 4.2 Service Architecture Decisions
| Component | Decision | Implementation Gap Identified |
|-----------|----------|------------------------------|
| **Repository Pattern** | Neo4j as single source of truth with async operations | Repository layer not implemented |
| **Service Layer** | Business logic separation with dependency injection | Only placeholder services exist |
| **Middleware Stack** | Auth, validation, constitutional compliance | ✅ Implemented and tested |
| **Configuration Management** | YAML + environment variables | ✅ Implemented |
| **Error Handling** | Structured errors with evaluation frames | Partial - needs service integration |

### 4.3 Integration Architecture
| Integration | Requirement | Current Status | Action Needed |
|-------------|-------------|----------------|---------------|
| **ThoughtSeed Pipeline** | Consciousness-guided processing for all documents | Extension points created | Connect to existing ASI-Arch modules |
| **Curiosity Engine** | Redis-backed mission scheduling with trust scoring | Model exists | Implement scheduling service |
| **Evaluation Framework** | Constitutional compliance for all operations | Middleware hooks ready | Implement evaluation service |
| **Mock Data Transparency** | Development mode with clear disclosure | Headers implemented | Connect to data models |
| **Local-First Operation** | No external API dependencies | Architecture supports | Verify all services are local |

### 4.4 Database Migration Strategy
| Database | Current State | Migration Required | Priority |
|----------|---------------|-------------------|----------|
| **Neo4j** | Not connected | Create schema, connection pool, repository layer | HIGH |
| **Redis** | Not connected | Setup streams, configure persistence | MEDIUM |
| **Qdrant** | Not connected | Create collections, embedding pipeline | MEDIUM |
| **SQLite** | Not used | Metadata caching (optional fallback) | LOW |

### 4.5 Development Workflow Gaps
| Component | Status | Issue Identified | Solution Required |
|-----------|--------|------------------|------------------|
| **Dependency Installation** | Incomplete | Missing core packages for databases/ML | Update requirements.txt with tested versions |
| **Service Startup** | Not functional | External services not configured | Add docker-compose for local development |
| **Testing Integration** | Partial | Tests exist but services not connected | Mock services or integration test setup |
| **Frontend Integration** | Placeholder | React components not connected to API | Implement actual UI components |

## 5. Next Steps
1. Obtain responses to clarification questions from user.  
2. Complete asset inventory worksheets (module lists, dependency diagrams).  
3. Execute research tasks and record findings/decisions in section 4.  
4. Once all clarifications resolved, mark Phase 0 complete and proceed to Phase 1 design artifacts.
