# Feature Specification: Self-Teaching Consciousness Emulator

**Feature Branch**: `002-i-m-building`  
**Created**: 2025-09-24  
**Status**: Draft  
**Input**: User description: "I'm building a self-teaching consciousness emulator. It will use active inference to evolve its pattern recognition. Any problem that it encounters that it can't solve based off all the strategies that it has learned, it will use a curiosity engine to go find other strategies that have been used—try variations and mutations of the strategies that it has. It will follow context engineering best practices as indicated by the files I have.

It will also pull in and modularize the libraries from my Dionysus consciousness project. From within this space, it will do constant learning through Redis based off the prompts and communication I have with it. It will allow me to upload bulk upload lots of documents and pass them through the architecture of the system so that the system can learn them and create episodic memories and run them across the attractor basins (as already implemented in Dionysus).

It will have an interface and all the features that you see in the SurfSense project, but our interface will be called Flux. Users will be able to upload documents, and the documents will have the concepts extracted—the documents will map back to all related documents on that topic so that a person can track their relationship with the concepts back in time. This can be done in two ways:

This could be done through the knowledge graph nodes visually, or the person can click through a set of cards in a stack or a list that has all the documents about the topic that they've collected along the way, with the most important parts highlighted, and if the user wants a summary of any of the given documents.

The system will have a curiosity engine, which is already partially implemented, that as topics become particularly salient or it discovers a knowledge gap, it develops curiosity and uses tools to go out and crawl the web—essentially to find reputable sources about the topic.

The system will have visual representations for transparency and let the user be able to see thought seeds at work passing through the basins, being part of episodic memory, the inner screen, competition for the inner screen, conversion from episodic to semantic to procedural memory.

The user will be able to create documents that are able to help them along the way by bringing up snippets of relevant content based off its knowledge of all the documents the user has. Dwell."

## User Scenarios & Testing *(mandatory)*

### Primary User Story
A research-focused user interacts with the Flux interface to upload a corpus of documents, observes how the self-teaching consciousness emulator processes them through ThoughtSeed attractor basins, receives concept mappings and summaries, and leverages curiosity-driven insights to expand their knowledge through newly discovered resources. Over time, Flux adapts to the user’s personal learning approach, curiosity patterns, and preferred topics, becoming a trusted lifelong learning partner that continuously surfaces relevant discoveries.

### Acceptance Scenarios
1. **Given** a signed-in user on Flux, **When** they bulk upload a set of documents, **Then** the system must ingest the files, extract concepts, create episodic memories, and map each document to related items via knowledge graph nodes and card stacks.
2. **Given** a user reviewing a concept within Flux, **When** the system detects a knowledge gap or high-salience topic, **Then** the curiosity engine must surface recommended external sources, note their provenance, and offer to integrate validated insights into the user’s knowledge base.
3. **Given** a user viewing the consciousness visualizations, **When** they explore a document’s processing path, **Then** they must see ThoughtSeed activation, basin transitions, inner screen competition, and conversions across episodic → semantic → procedural memory with clear annotations.

### Edge Cases
- **Document Upload Limits**: System handles uploads up to 100MB individual file size, 1GB total batch size, supports PDF/TXT/DOCX/MD/PPTX formats, with queue processing for batches > 50 documents and user notification for unsupported formats.
- **External Source Quality Control**: Curiosity engine validates sources through citation standing analysis, cross-reference verification with existing knowledge base, and relevance assessment (particularly valuing foundational works from psychology and hypnotherapy fields regardless of publication date).
- **Strategy Conflict Resolution**: When new curiosity-discovered strategies contradict existing learned strategies, system creates comparison evaluation frame, presents both approaches to user with confidence scores, and allows user selection while maintaining both options in knowledge base.
- **Infrastructure Resilience**: When Redis becomes unreachable, system switches to local SQLite caching with periodic sync retry (30s intervals); when memory stores saturate, system triggers automatic archival of least-recently-used episodic memories to persistent storage with user notification.

## Requirements *(mandatory)*

### Constitutional Infrastructure Requirements ⚠️ DO NOT CHANGE
- **FR-000 [LOCKED]**: System MUST use NumPy 2.0+ exclusively. All NumPy 1.x dependencies have been eliminated. Binary compatibility solutions implemented for PyTorch latest version and sentence-transformers. NO NumPy 1.x anywhere in codebase.
- **FR-001 [LOCKED]**: System MUST be completely Docker-independent for Mac desktop, iPhone, and Android deployment. Use embedded databases: SQLite + FAISS for vectors, embedded Redis alternative, no external container dependencies.
- **FR-032 [LOCKED]**: Development MAY use Docker temporarily for immediate testing and validation, but final implementation MUST replace all Docker services with embedded alternatives before user deployment. Production version MUST run entirely from desktop without external dependencies.

### Functional Requirements
- **FR-002**: System MUST implement a self-teaching consciousness loop driven by active inference, adjusting pattern recognition through continuous prediction error minimization.
- **FR-003**: System MUST maintain a modular integration of Dionysus consciousness libraries, exposing them as reusable components within the ASI-Arch architecture.
- **FR-004**: System MUST operate an embedded continual learning layer (SQLite-based, no Redis dependency) that captures prompts, user interactions, and system outputs for iterative self-improvement.
- **FR-004**: Users MUST be able to bulk upload documents via Flux, trigger ingestion, and monitor processing status.
- **FR-005**: System MUST extract salient concepts from uploaded content, create episodic memories, and align them with existing attractor basins.
- **FR-006**: System MUST provide two exploration modalities for concept history: interactive knowledge graph visualization and ordered card/list stacks with highlights and summaries.
- **FR-007**: System MUST enable curiosity-driven discovery by detecting knowledge gaps or high-salience topics, launching tool-based searches, and presenting reputable sources with provenance details. Reputable sources are defined as: academic publications (peer-reviewed journals, conference proceedings), established educational institutions (.edu domains), government research agencies, and sources with sustained citation standing and relevance (including foundational works from 1960+ that remain isomorphic to current ideas, particularly in psychology and hypnotherapy).
- **FR-008**: System MUST surface visualizations that display ThoughtSeed progression, basin competition, memory conversions, and inner screen dynamics in real time or near-real time.
- **FR-009**: Frontend MUST read and parse flux.yaml configuration file to apply correct port settings (9243), database connections, and system parameters instead of hardcoded values.
- **FR-010**: Backend MUST read flux.yaml configuration and start on specified port (9127) with all database, inference, and system settings from the config file.
- **FR-011**: System MUST establish data bridge between frontend and backend using configuration-driven API endpoints, replacing all mock data with real backend responses.
- **FR-012**: Vite development server MUST serve on port 9243 as specified in flux.yaml, not hardcoded port 3000.
- **FR-013**: Interface MUST implement animated background with floating light threads in multiple colors (cyan, magenta, yellow, green) that move in gentle, capricious patterns across a black to light gray horizontal gradient background creating horizon perspective effect, with varying opacity, blur effects, and graceful floating animations always present (fading when content comes forward) to create an immersive consciousness visualization experience.
- **FR-014**: Backend MUST connect to Neo4j database using flux.yaml configuration (bolt://localhost:7687) for canonical knowledge graph storage and relationship mapping.
- **FR-015**: Backend MUST connect to Qdrant vector database using flux.yaml configuration (localhost:6333) for document embeddings and semantic search capabilities.
- **FR-016**: Backend MUST connect to Redis cache using flux.yaml configuration (localhost:6379) for real-time curiosity signals, session persistence, and document processing optimization with SQLite fallback when Redis unavailable.
- **FR-017**: System MUST implement fallback embedding service using sentence-transformers model as specified in flux.yaml when primary embedding services are unavailable.
- **FR-009**: System MUST allow users to generate new documents that pull relevant snippets, context, and summaries from their knowledge base tailored to the current task.
- **FR-010**: System MUST provide evaluative feedback aligned with constitutional principles (“What’s good? What’s broken? What works but shouldn’t? What doesn’t but pretends to?”) for all learning cycles and curiosity outcomes.
- **FR-011**: System MUST support modular feature toggling to accommodate staged integration of SurfSense-equivalent capabilities within Flux. [NEEDS CLARIFICATION: phased rollout plan]
- **FR-012**: System MUST extract and integrate individual modules from Dionysus consciousness v1.0, ensuring each module meets current standards before integration. Create independent, standards-compliant modules and binary distributions for frozen components that can only be modified through extension, composition, or other design patterns. Provide API access for binary-packaged components.
- **FR-024**: System MUST integrate Daedalus legacy components including multi-modal processing pipeline (video/audio/image), hybrid memory system (MEM1 + ThoughtSeed), NEMORI integration, and existing knowledge graph (10,808 nodes: memories + papers) with selective re-processing capabilities for ThoughtSeed 2.0 enhancement.
- **FR-025**: System MUST maintain backward compatibility with existing Dionysus consciousness project components while providing upgrade paths from legacy ThoughtSeed 1.0 to enhanced ThoughtSeed 2.0 features on-demand.
- **FR-026**: System MUST implement protected Daedalus integration layer with crash-proof isolation to prevent legacy code from affecting core Flux operations.

#### Real-Time Dashboard Data Requirements
- **FR-027**: Dashboard MUST fetch activeThoughtSeeds count from Redis key "flux:thoughtseeds:active" with 5-second refresh interval, falling back to SQLite table "thoughtseed_states" when Redis unavailable.
- **FR-028**: Dashboard MUST query Neo4j for documentsProcessed count using Cypher "MATCH (d:Document) RETURN count(d)" and conceptsExtracted using "MATCH (c:Concept) RETURN count(c)".
- **FR-029**: Dashboard MUST retrieve curiosityMissions from Redis sorted set "flux:curiosity:missions" showing active missions count, with processing status updates via WebSocket on port 9129.
- **FR-030**: Dashboard MUST implement data fetching in React useEffect hooks with error boundaries, loading states, and automatic retry on connection failure.
- **FR-031**: Backend MUST provide REST API endpoints at `/api/stats/dashboard` aggregating all dashboard metrics from respective databases with caching layer.

#### Infrastructure Requirements (Implemented)
- **FR-013**: System MUST implement automatic port conflict detection and resolution with unique port allocation (9127, 9243, 9129, 9131) to ensure reliable service startup.
- **FR-014**: System MUST provide comprehensive database health monitoring for Neo4j, Redis, and Qdrant with real-time status, response times, and graceful failure handling.
- **FR-015**: System MUST enforce strict test-driven development (TDD) with RED → GREEN → REFACTOR cycle, comprehensive test coverage, and no-skip test policy.
- **FR-016**: System MUST provide regression protection ensuring all implemented features continue working as new features are added.

#### Quality & Compliance Requirements
- **FR-017**: All user-facing outputs, documentation, and internal assessments MUST adhere to the Scientific Objectivity principle—no hyperbole, only validated metrics, precise terminology (e.g., "consciousness emulator"), and explicit acknowledgment that claims represent mathematical simulations rather than actual consciousness.
- **FR-018**: System MUST disclose whenever mock, synthetic, or placeholder data is in use, and prior to any production readiness declaration, MUST validate the same workflows against real data sources with documented results.
- **FR-019**: System MUST learn and maintain a personalized user learning profile capturing curiosity patterns, preferred topics, and interaction rhythms, and use it to recommend newly discovered content (including overnight or scheduled scans) without overwhelming the user.
- **FR-020**: System MUST synthesize findings from bulk scientific papers with zero hallucinations by grounding claims in original sources, citing provenance, and clearly flagging any uncertainty.

#### Privacy & Local-First Requirements
- **FR-021**: System MUST support privacy-preserving operation on user-controlled desktops, running with OLA and locally available language models while storing all data in user-managed local vector and graph databases by default.
- **FR-022**: System MUST default all inference—especially bulk paper ingestion, summarization, and concept extraction—to locally hosted LLaMA (via Ollama or equivalent) unless the user explicitly opts into a remote provider, ensuring no unintended token consumption or data leakage.
- **FR-023**: System MUST route every document upload path (single or bulk, through any interface) through the same local inference and user-controlled storage pipeline, and even in production deployments must keep content on the user's desktop or user-managed hosting unless the user explicitly configures otherwise.

### Key Entities *(include if feature involves data)*
- **User Profile**: Represents individuals interacting with Flux, including personalization preferences, interaction history, and access permissions.
- **Document Artifact**: Uploaded files or generated content, storing metadata (source, timestamps), extracted concepts, summaries, and links to related artifacts.
- **Concept Node**: Semantic representation of core ideas; tracks relationships across documents, knowledge gaps, salience scores, and curiosity triggers.
- **ThoughtSeed Trace**: Temporal record of ThoughtSeed activation, basin transitions, inner screen competitions, and memory conversions for transparency.
- **Curiosity Mission**: Log of discovery tasks initiated by the curiosity engine, including query rationale, selected tools, discovered sources, validation status, and ingestion outcomes.
- **Memory Fragment**: Encapsulates episodic, semantic, and procedural memory states derived from processed content, linked to ThoughtSeed traces and concept nodes.

---

## Review & Acceptance Checklist

### Content Quality
- [ ] No implementation details (languages, frameworks, APIs)
- [x] Focused on user value and business needs
- [x] Written for non-technical stakeholders
- [x] All mandatory sections completed

### Requirement Completeness
- [ ] No [NEEDS CLARIFICATION] markers remain
- [x] Requirements are testable and unambiguous  
- [x] Success criteria are measurable
- [x] Scope is clearly bounded
- [x] Dependencies and assumptions identified

---

## Execution Status
- [x] User description parsed
- [x] Key concepts extracted
- [x] Ambiguities marked
- [x] User scenarios defined
- [x] Requirements generated
- [x] Entities identified
- [ ] Review checklist passed
