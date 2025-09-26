# Implementation Plan: Remove ASI-Arch and Integrate ASI-GO-2

**Branch**: `004-remove-asi-arch` | **Date**: 2025-09-26 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/004-remove-asi-arch/spec.md`

## Execution Flow (/plan command scope)
```
1. Load feature spec from Input path → COMPLETE
   → Specification loaded with clarifications resolved
2. Fill Technical Context (scan for NEEDS CLARIFICATION) → COMPLETE
   → Detected Project Type: web (backend+frontend system)
   → Set Structure Decision: Option 2 (Web application)
3. Fill Constitution Check section → COMPLETE
   → No constitution file with actual content found (template only)
4. Evaluate Constitution Check section → COMPLETE
   → No violations detected, proceeding
   → Update Progress Tracking: Initial Constitution Check → PASS
5. Execute Phase 0 → research.md → COMPLETE
6. Execute Phase 1 → contracts, data-model.md, quickstart.md, CLAUDE.md → COMPLETE
7. Re-evaluate Constitution Check → COMPLETE
   → No new violations detected
   → Update Progress Tracking: Post-Design Constitution Check → PASS
8. Plan Phase 2 → Task generation approach described → COMPLETE
9. STOP - Ready for /tasks command
```

## Summary
Remove all ASI-Arch neural architecture discovery components and replace with ASI-GO-2 as primary research intelligence engine. Full integration includes Context Engineering framework with attractor basins, neural fields, 5-layer ThoughtSeed hierarchy (sensory→perceptual→conceptual→abstract→metacognitive), Daedalus delegation patterns, active inference loops, world modeling, structure learning, and narrative pattern recognition systems.

## Technical Context
**Language/Version**: Python 3.11
**Primary Dependencies**: FastAPI, ASI-GO-2 components, Daedalus framework, ThoughtSeed system, Context Engineering framework
**Storage**: Unified hybrid database (SQLite + JSON graph + Vector index), Neo4j (knowledge graph), Redis (caching)
**Testing**: pytest, integration testing for Context Engineering components
**Target Platform**: Linux server (Docker containerized)
**Project Type**: web - backend system with API endpoints
**Performance Goals**: Research query processing <2s response time, document ingestion <5s per document
**Constraints**: Must preserve existing document processing capabilities, maintain research intelligence quality during transition
**Scale/Scope**: System processes research documents and synthesizes intelligent responses using accumulated patterns

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**Status**: PASS - No constitution violations detected (constitution file contains only template placeholders)

## Project Structure

### Documentation (this feature)
```
specs/004-remove-asi-arch/
├── plan.md              # This file (/plan command output)
├── research.md          # Phase 0 output (/plan command)
├── data-model.md        # Phase 1 output (/plan command)
├── quickstart.md        # Phase 1 output (/plan command)
├── contracts/           # Phase 1 output (/plan command)
└── tasks.md             # Phase 2 output (/tasks command - NOT created by /plan)
```

### Source Code (repository root)
```
# Option 2: Web application (backend system)
backend/
├── src/
│   ├── models/           # ASI-GO-2 entities: CognitionBase, ResearchPattern, ThoughtseedWorkspace
│   ├── services/         # ASI-GO-2 core: Researcher, Engineer, Analyst, Context Engineering
│   └── api/              # REST endpoints for research queries, document processing
└── tests/
    ├── contract/         # API contract tests
    ├── integration/      # End-to-end ASI-GO-2 integration tests
    └── unit/             # Component-level tests

# ASI-Arch removal targets:
pipeline/                 # TO BE REMOVED
extensions/context_engineering/asi_arch_*  # ASI-Arch bridges TO BE REMOVED
```

**Structure Decision**: Option 2 (Web application) - Backend-focused system with API for research intelligence

## Phase 0: Outline & Research

**Research Topics Identified**:
1. ASI-GO-2 architecture deep dive and integration patterns
2. Context Engineering framework implementation (attractor basins, neural fields)
3. ThoughtSeed 5-layer hierarchy integration with ASI-GO-2
4. Daedalus delegation pattern integration with ASI-GO-2 components
5. Active inference implementation for research pattern recognition
6. Narrative/motif recognition system integration
7. ASI-Arch component identification and removal strategy

**Output**: research.md with comprehensive analysis of integration approach

## Phase 1: Design & Contracts

**Design Components**:
1. **Data Model**: ASI-GO-2 entities (CognitionBase, ResearchPattern, ThoughtseedWorkspace, etc.)
2. **API Contracts**: Research query endpoints, document processing endpoints, pattern management
3. **Integration Architecture**: Context Engineering + ThoughtSeed + Daedalus coordination
4. **Removal Strategy**: Systematic ASI-Arch component elimination
5. **Agent Context**: Update CLAUDE.md with ASI-GO-2 system knowledge

**Output**: data-model.md, /contracts/*, failing tests, quickstart.md, CLAUDE.md

## Phase 2: Task Planning Approach
*This section describes what the /tasks command will do - DO NOT execute during /plan*

**Task Generation Strategy**:
- ASI-Arch removal tasks (pipeline cleanup, dependency removal, data cleanup)
- ASI-GO-2 core integration tasks (Cognition Base, Researcher, Engineer, Analyst)
- Context Engineering integration tasks (attractor basins, neural fields)
- ThoughtSeed 5-layer hierarchy implementation tasks
- Daedalus delegation integration tasks
- API endpoint implementation for research queries
- Testing tasks for each integrated component

**Ordering Strategy**:
1. ASI-Arch removal (cleanup first)
2. ASI-GO-2 core installation and basic integration
3. Context Engineering framework integration
4. ThoughtSeed hierarchy implementation
5. Daedalus delegation integration
6. API layer implementation
7. Integration testing and validation

**Estimated Output**: 40-50 numbered, ordered tasks covering complete migration

## Phase 3+: Future Implementation
*These phases are beyond the scope of the /plan command*

**Phase 3**: Task execution (/tasks command creates tasks.md)
**Phase 4**: Implementation (systematic ASI-Arch removal + ASI-GO-2 integration)
**Phase 5**: Validation (research query testing, document processing validation, Context Engineering verification)

## Complexity Tracking
*No constitutional violations requiring justification*

## Progress Tracking
*This checklist is updated during execution flow*

**Phase Status**:
- [x] Phase 0: Research complete (/plan command)
- [x] Phase 1: Design complete (/plan command)
- [x] Phase 2: Task planning complete (/plan command - describe approach only)
- [ ] Phase 3: Tasks generated (/tasks command)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:
- [x] Initial Constitution Check: PASS
- [x] Post-Design Constitution Check: PASS
- [x] All NEEDS CLARIFICATION resolved
- [x] Complexity deviations documented (none required)

---
*Based on Constitution template - See `/.specify/memory/constitution.md`*