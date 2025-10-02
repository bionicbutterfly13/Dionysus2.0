# Agentic Knowledge Graph - Complete Roadmap

**Status**: READY FOR IMPLEMENTATION
**Created**: 2025-10-01
**Last Updated**: 2025-10-01

## Executive Summary

This roadmap defines the complete implementation plan for the **Agentic Knowledge Graph** system, where AI agents autonomously build, enrich, and explore a self-improving knowledge graph during bulk document processing.

### What We're Building

An intelligent document processing system that:
1. **Learns from every paper** - Agents get smarter as more documents flow through
2. **Discovers patterns automatically** - ThoughtSeeds connect related concepts across documents
3. **Explores knowledge gaps** - Curiosity agents investigate missing information in background
4. **Uses local LLM (ZERO COST)** - All inference runs on Ollama (DeepSeek-R1, Qwen2.5)
5. **Provides visual feedback** - Real-time interface shows agent decisions and graph updates

## Four Core Specifications

### Spec 027: Basin Frequency Strengthening
**Purpose**: Attractor basins grow stronger as concepts reappear across multiple papers

**Key Features**:
- Basin strength increases +0.2 per concept reappearance (capped at 2.0)
- LLM receives basin context during relationship extraction
- Stronger basins = higher priority concepts for LLM
- Co-occurrence tracking: Learn which concepts appear together
- Basin decay: Unused basins weaken by 0.1 per week

**Agent Improvement**:
```
Paper 1-10:   Basin "neural networks" = 1.0 strength
Paper 11-50:  Basin "neural networks" = 1.8 strength → LLM prioritizes this concept
Paper 51+:    LLM produces better relationships because it knows concept importance
```

**Status**: DRAFT - Ready for implementation
**Estimated Time**: 8-10 hours
**Dependencies**: Neo4j, Redis, AttractorBasinManager

---

### Spec 028: ThoughtSeed Generation During Bulk Upload
**Purpose**: Generate ThoughtSeeds for every concept, enabling cross-document linking and pattern emergence

**Key Features**:
- ThoughtSeed created for each concept (confidence >0.7)
- ThoughtSeeds propagate through consciousness system
- Cross-document linking (>0.8 similarity)
- Emergent pattern detection (5+ ThoughtSeeds in basin)
- 24-hour Redis TTL, permanent Neo4j archive

**Agent Roles**:
1. **ThoughtSeed Generation Agent**: Create ThoughtSeeds from concepts
2. **Basin Integration Agent**: Integrate ThoughtSeeds into basins
3. **Cross-Document Linking Agent**: Link related ThoughtSeeds from different papers
4. **Pattern Emergence Agent**: Detect when multiple ThoughtSeeds form coherent patterns
5. **ThoughtSeed Validation Agent**: Quality-check before propagation

**Example Flow**:
```
Paper 1: "neural architecture search" → ThoughtSeed TS_001 → Basin_NAS
Paper 5: "neural arch optimization" → ThoughtSeed TS_045 → Links to TS_001 (0.87 similarity)
Paper 10: "one-shot NAS" → ThoughtSeed TS_098 → Basin_NAS now has 3 ThoughtSeeds
Paper 15: "DARTS method" → ThoughtSeed TS_150 → Basin_NAS now has 4 ThoughtSeeds
Paper 20: "supernet training" → ThoughtSeed TS_200 → EMERGENT PATTERN DETECTED! (5+ ThoughtSeeds)
  → Creates EmergentPattern node: "neural_architecture_optimization"
  → Links to all 5 contributing documents
```

**Status**: DRAFT - Ready for implementation
**Estimated Time**: 13-18 hours
**Dependencies**: Spec 027, Redis, Neo4j, Active Inference

---

### Spec 029: Curiosity-Driven Background Agents (Local LLM)
**Purpose**: Spawn background agents to investigate knowledge gaps using **local Ollama (ZERO COST)**

**Key Features**:
- Curiosity triggers detected (prediction error >0.7)
- Background agents spawn asynchronously (max 5 concurrent)
- Question generation using **DeepSeek-R1** (local, free)
- Knowledge graph search for related concepts
- Document recommendations using **Qwen2.5** (local, free)
- **ZERO API COST** - All inference is local

**Agent Roles**:
1. **Curiosity Detection Agent**: Monitor for prediction errors and knowledge gaps
2. **Question Generation Agent** (Local LLM): Generate 3-5 research questions
3. **Knowledge Graph Search Agent**: 2-hop Cypher queries for related info
4. **Document Recommendation Agent** (Local LLM): Propose papers to ingest
5. **Exploration Task Queue Manager**: Prioritize curiosity triggers
6. **Background Agent Orchestrator**: Spawn/manage agents asynchronously

**Example Flow**:
```
Document processing: Extract concept "quantum neural networks"
  ↓
Active Inference: Prediction error = 0.85 (HIGH SURPRISE!)
  ↓
Curiosity Detection Agent: Create curiosity trigger (priority: 0.85)
  ↓
Background Agent Orchestrator: Spawn Agent #1 (runs in background, doesn't block)
  ↓
Question Generation Agent (DeepSeek-R1, LOCAL):
  - "How do quantum neural networks differ from classical ANNs?"
  - "What are current limitations of quantum computing for ML?"
  - "Which research groups are leading quantum ML?"
  ↓
Knowledge Graph Search Agent: Find 12 related concepts, 3 documents
  ↓
Document Recommendation Agent (Qwen2.5, LOCAL):
  - "Quantum ML Survey 2024" by Scott Aaronson
  - "Quantum Entanglement for Deep Learning" by Vedran Dunjko
  ↓
Store in Neo4j: Questions, Recommendations, Resolution status
  ↓
User reviews recommendations and decides to upload suggested papers
```

**Status**: DRAFT - Ready for implementation
**Estimated Time**: 12-17 hours
**Dependencies**: Spec 027, Spec 028, Ollama, Redis, Neo4j

---

### Spec 030: Visual Testing Interface
**Purpose**: Real-time visualization of agent orchestration, knowledge graph updates, and ThoughtSeed propagation

**Key Features**:
- Agent orchestration timeline (handoff pattern visualization)
- Live knowledge graph updates (3D force-directed layout)
- ThoughtSeed propagation tracking (green nodes, cross-doc links)
- Curiosity agent monitoring panel (background agents)
- Quality metrics dashboard
- Interactive controls (upload, pause/resume, step-through)

**Components**:
1. **AgentOrchestrationTimeline**: Sequential agent handoffs
2. **KnowledgeGraphViz**: 3D graph with live updates
3. **ThoughtSeedPanel**: Active ThoughtSeeds, cross-doc links
4. **CuriosityAgentPanel**: Background agent status
5. **QualityMetricsDashboard**: Overall system health

**WebSocket Events**:
- `agent_handoff`: Daedalus → Extractor → Analyst → ...
- `concept_extracted`: New concept appears in graph
- `relationship_created`: LLM-generated relationship
- `thoughtseed_created`: ThoughtSeed propagation begins
- `curiosity_triggered`: Background agent spawned
- `background_agent_status`: Agent progress updates

**User Experience**:
```
1. User uploads "neural_architecture_search.pdf"
2. Timeline shows: [Daedalus] → [Extractor] → [Analyst] → ...
3. Graph shows concepts appearing: "NAS", "AutoML", "meta-learning"
4. Relationships drawn: "DARTS" --EXTENDS→ "gradient-based optimization"
5. ThoughtSeeds created: TS_001, TS_002, TS_003
6. Basin strengthened: "neural_arch" (1.0 → 1.2)
7. Curiosity triggered: "quantum neural networks" (error: 0.85)
8. Background agent #1 spawns: Generating questions...
9. Questions appear: "How do quantum NNs work?"
10. Recommendations appear: "Quantum ML Survey 2024"
11. User clicks recommendation → Downloads paper → Uploads to system
```

**Status**: DRAFT - Ready for implementation
**Estimated Time**: 13-18 hours
**Dependencies**: Spec 027, Spec 028, Spec 029, React, Three.js, FastAPI WebSockets

---

## Integration Architecture

### Agent Orchestration Pattern: Handoff + Asynchronous Coordination

```
FOREGROUND (Sequential Handoff):
Upload → [Daedalus] → [Extractor] → [Analyst] → [Synthesizer] → [Storage] → Done
         Gateway     Concepts      Quality     Relations     Neo4j

BACKGROUND (Asynchronous Coordination):
         → [Curiosity Agent 1] (exploring "quantum neural networks")
         → [Curiosity Agent 2] (exploring "neuromorphic computing")
         → [Curiosity Agent 3] (exploring "meta-learning theory")
            ↓ (shared context: Neo4j Knowledge Graph)
            ↓ (no direct communication)
            ↓ (read/write asynchronously)
         → Results stored in Neo4j when ready
```

### Data Flow

```
Document Upload (PDF):
  ↓
[Daedalus Gateway]: Receive perceptual information
  ↓
[Concept Extractor]: Extract concepts from text
  ↓ [Emits: concept_extracted event]
  ↓
[Basin Manager]: Get or create basins, strengthen if reappearance
  ↓ [Emits: basin_strengthened event]
  ↓
[Relationship Extractor]: LLM-based with basin context (DeepSeek-R1/Qwen2.5)
  ↓ [Emits: relationship_created event]
  ↓
[ThoughtSeed Generator]: Create ThoughtSeeds for concepts
  ↓ [Emits: thoughtseed_created event]
  ↓
[Cross-Doc Linker]: Link similar ThoughtSeeds from different docs
  ↓ [Emits: thoughtseed_linked event]
  ↓
[Pattern Detector]: Detect emergent patterns (5+ ThoughtSeeds)
  ↓ [Emits: pattern_emerged event]
  ↓
[Curiosity Detector]: Check for prediction errors
  ↓ (if error >0.7) [Emits: curiosity_triggered event]
  ↓
[Background Orchestrator]: Spawn curiosity agent (async, doesn't block)
  ↓
[Question Generator] (Local LLM): Generate research questions
  ↓ [Emits: questions_generated event]
  ↓
[KG Search Agent]: Search Neo4j for related info
  ↓ [Emits: search_complete event]
  ↓
[Recommendation Agent] (Local LLM): Suggest documents to ingest
  ↓ [Emits: recommendations_ready event]
  ↓
[Neo4j Storage]: Persist everything
  ↓ [Emits: storage_complete event]
  ↓
DONE - All events broadcast via WebSocket to visual interface
```

## Technology Stack

### Backend
- **Python 3.11+**: Core language
- **FastAPI**: REST API + WebSocket server
- **LangGraph**: Agent orchestration workflow
- **Ollama**: Local LLM inference (DeepSeek-R1, Qwen2.5)
- **Neo4j 5.x**: Knowledge graph database
- **Redis 7.x**: ThoughtSeed state + task queues
- **Asyncio**: Background agent coordination

### Frontend
- **React 18**: UI framework
- **TypeScript**: Type safety
- **Three.js**: 3D graph visualization
- **react-force-graph**: Force-directed layout
- **WebSocket API**: Real-time updates
- **TailwindCSS**: Styling

### Local LLM Models (Ollama)
- **DeepSeek-R1**: Reasoning tasks (question generation, relationship extraction)
- **Qwen2.5:14b**: Structured output (document recommendations)
- **Nomic-Embed**: Text embeddings (concept similarity)

**Total Cost**: **$0** - All inference is local

## Implementation Timeline

### Phase 1: Basin Strengthening (Week 1)
**Duration**: 8-10 hours
**Tasks**:
1. Extend `AttractorBasin` dataclass with frequency tracking
2. Implement `_get_or_create_basin()` with strengthening logic
3. Update LLM prompt to include basin context
4. Add co-occurrence tracking
5. Implement basin decay
6. Write tests (unit + integration)

**Success Criteria**:
- Basins strengthen +0.2 per reappearance
- LLM receives basin strength in prompt
- Agent quality improves >15% after 50 papers

---

### Phase 2: ThoughtSeed Processing (Week 2)
**Duration**: 13-18 hours
**Tasks**:
1. Implement `ThoughtSeedGenerationAgent`
2. Implement `CrossDocumentLinkingAgent`
3. Implement `PatternEmergenceAgent`
4. Add `_generate_and_propagate_thoughtseeds` node to LangGraph
5. Redis storage with 24-hour TTL
6. Neo4j schema extensions
7. Write tests (unit + integration)

**Success Criteria**:
- ThoughtSeeds generated for all concepts
- Cross-document links created (>0.8 similarity)
- Emergent patterns detected (5+ ThoughtSeeds)
- Performance: 100 papers → 500-1000 ThoughtSeeds in <5 min

---

### Phase 3: Curiosity Agents (Week 3)
**Duration**: 12-17 hours
**Tasks**:
1. Implement `CuriosityDetectionAgent`
2. Implement `QuestionGenerationAgent` (Ollama DeepSeek-R1)
3. Implement `DocumentRecommendationAgent` (Ollama Qwen2.5)
4. Implement `BackgroundAgentOrchestrator`
5. Redis task queue with priority
6. Neo4j schema for curiosity questions/recommendations
7. Write tests (unit + integration)

**Success Criteria**:
- Curiosity triggers detected (prediction error >0.7)
- Background agents spawn asynchronously (max 5)
- **ZERO COST**: All LLM calls use local Ollama
- Questions + recommendations generated
- Results stored in Neo4j

---

### Phase 4: Visual Interface (Week 4)
**Duration**: 13-18 hours
**Tasks**:
1. Implement WebSocket backend (`ConnectionManager`)
2. Add event emission to all graph nodes
3. Implement `AgentOrchestrationTimeline` (React)
4. Implement `KnowledgeGraphViz` (Three.js)
5. Implement `CuriosityAgentPanel` (React)
6. Implement `QualityMetricsDashboard` (React)
7. Interactive controls (upload, pause/resume)
8. User acceptance testing

**Success Criteria**:
- Real-time agent handoffs visible
- Knowledge graph updates live
- ThoughtSeed propagation animated
- Curiosity agents monitored
- WebSocket latency <200ms

---

### Phase 5: Integration Testing (Week 5)
**Duration**: 8-12 hours
**Tasks**:
1. End-to-end testing with real papers
2. Performance optimization
3. Load testing (100+ papers)
4. User acceptance testing
5. Documentation updates
6. Bug fixes

**Success Criteria**:
- All 4 specs working together seamlessly
- Agent quality improvement validated (>15% after 50 papers)
- Visual interface shows all events correctly
- System stable under load

---

## Total Project Timeline

**Duration**: 5 weeks
**Total Hours**: 54-75 hours
**Team Size**: 1-2 developers
**Cost**: $0 (local LLM only)

## Success Metrics

### Agent Learning Quality
- [ ] Relationship extraction quality improves >15% after 50 papers
- [ ] Basin strength correlates with concept importance (validated manually)
- [ ] Co-occurrence patterns match domain expert expectations

### ThoughtSeed System
- [ ] >90% of ThoughtSeeds match at least one basin
- [ ] Cross-document links have >0.8 similarity
- [ ] Emergent patterns validated by human review (>80% precision)

### Curiosity Exploration
- [ ] >85% of curiosity triggers resolved within 10 minutes
- [ ] Questions generated are specific and actionable
- [ ] Document recommendations are relevant (>70% user acceptance)

### Visual Interface
- [ ] WebSocket latency <200ms for all events
- [ ] Graph supports 100+ nodes without lag
- [ ] User can pause/resume/step-through processing
- [ ] All agent decisions visible and inspectable

### System Performance
- [ ] 100 papers processed in <30 minutes
- [ ] 500-1000 ThoughtSeeds generated from 100 papers
- [ ] Max 5 concurrent background agents maintained
- [ ] Neo4j storage completes in <5 seconds per paper

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Redis memory growth | HIGH | TTL-based cleanup, compression |
| Neo4j performance degradation | HIGH | Batch writes, indexing, query optimization |
| Ollama inference latency | MEDIUM | Use Qwen2.5:7b for faster inference if needed |
| WebSocket connection instability | MEDIUM | Reconnection logic, buffering |
| Basin strength inflation | LOW | Cap at 2.0, decay mechanism |
| False emergent patterns | MEDIUM | Require 5+ ThoughtSeeds, human validation |

## Next Steps

### Immediate Actions (Before Implementation)
1. ✅ Create comprehensive specs (DONE)
2. ⏳ Review specs with stakeholders
3. ⏳ Set up development environment (Neo4j, Redis, Ollama)
4. ⏳ Create GitHub project board with tasks
5. ⏳ Assign developers to each phase

### Phase 1 Kickoff Checklist
- [ ] Neo4j running locally (Docker Compose)
- [ ] Redis running locally (Docker Compose)
- [ ] Ollama installed with DeepSeek-R1 and Qwen2.5:14b
- [ ] Test document processing pipeline functional
- [ ] Development branch created: `feature/agentic-knowledge-graph`
- [ ] Test suite skeleton created

## References

### Specifications
- [Spec 027: Basin Frequency Strengthening](./027-basin-frequency-strengthening/spec.md)
- [Spec 028: ThoughtSeed Bulk Processing](./028-thoughtseed-bulk-processing/spec.md)
- [Spec 029: Curiosity-Driven Background Agents](./029-curiosity-driven-background-agents/spec.md)
- [Spec 030: Visual Testing Interface](./030-visual-testing-interface/spec.md)

### Existing Documentation
- [Agentic Knowledge Graph Complete](../AGENTIC_KNOWLEDGE_GRAPH_COMPLETE.md)
- [Spec 021: Daedalus Gateway](./021-remove-all-that/spec.md)
- [Spec 005: ThoughtSeed System](./005-a-thought-seed/spec.md)
- [Spec 006: Query System](./006-ability-to-ask/spec.md)

### External Resources
- [LangGraph Documentation](https://python.langchain.com/docs/langgraph)
- [Ollama API](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [Neo4j Cypher Manual](https://neo4j.com/docs/cypher-manual/current/)
- [Redis Documentation](https://redis.io/docs/)
- [React Three Fiber](https://docs.pmnd.rs/react-three-fiber)

---

**Status**: All specs complete, ready for stakeholder review and implementation kickoff.

**Contact**: Development team ready to begin Phase 1 upon approval.
