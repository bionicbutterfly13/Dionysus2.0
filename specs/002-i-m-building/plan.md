
# Implementation Plan: Self-Teaching Consciousness Emulator

**Branch**: `002-i-m-building` | **Date**: 2025-09-24 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/002-i-m-building/spec.md`

## Execution Flow (/plan command scope)
```
1. Load feature spec from Input path
   → If not found: ERROR "No feature spec at {path}"
2. Fill Technical Context (scan for NEEDS CLARIFICATION)
   → Detect Project Type from context (web=frontend+backend, mobile=app+api)
   → Set Structure Decision based on project type
3. Fill the Constitution Check section based on the content of the constitution document.
4. Evaluate Constitution Check section below
   → If violations exist: Document in Complexity Tracking
   → If no justification possible: ERROR "Simplify approach first"
   → Update Progress Tracking: Initial Constitution Check
5. Execute Phase 0 → research.md
   → If NEEDS CLARIFICATION remain: ERROR "Resolve unknowns"
6. Execute Phase 1 → contracts, data-model.md, quickstart.md, agent-specific template file (e.g., `CLAUDE.md` for Claude Code, `.github/copilot-instructions.md` for GitHub Copilot, `GEMINI.md` for Gemini CLI, `QWEN.md` for Qwen Code or `AGENTS.md` for opencode).
7. Re-evaluate Constitution Check section
   → If new violations: Refactor design, return to Phase 1
   → Update Progress Tracking: Post-Design Constitution Check
8. Plan Phase 2 → Describe task generation approach (DO NOT create tasks.md)
9. STOP - Ready for /tasks command
```

**IMPORTANT**: The /plan command STOPS at step 7. Phases 2-4 are executed by other commands:
- Phase 2: /tasks command creates tasks.md
- Phase 3-4: Implementation execution (manual or via tools)

## Summary
Build Flux (our lifelong learning partner) that processes documents through ASI-Arch ThoughtSeed attractor basins, creates autobiographical memories, and surfaces curiosity-driven insights. Priority: document upload → conscious processing → graph database pipeline. Integrate Dionysus modules, adapt SurfSense components, maintain local-first operation with Ollama/LLaMA.

## Technical Context
**Language/Version**: Python 3.11 (existing ASI-Arch virtualenv)  
**Primary Dependencies**: FastAPI, Neo4j/Qdrant clients, Redis, LangGraph, Context-Engineering library, Ollama/LLaMA runtime, SurfSense UI components (to be adapted)  
**Storage**: Neo4j Desktop/Embedded knowledge graph (progressive download, no Docker), Qdrant local mode (progressive download), embedded Redis alternative for transient learning queues  
**Testing**: pytest + pytest-asyncio + integration suites (existing in tests/)  
**Target Platform**: Mac desktop, iPhone, Android (Docker-free, progressive database downloads)  
**Project Type**: web (Flux interface + backend services)  
**Performance Goals**: Deferred - focus on core upload pipeline first  
**Constraints**: Local-first execution, no token leakage, zero hallucination tolerance, context-engineering compliance, knowledge-graph SSoT  
**Scale/Scope**: Desktop usage with growing personal corpora; deferred scaling decisions

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Technical Requirements Compliance
- Scientific Objectivity: ✅ All claims tied to documented mechanisms; planning artifacts avoid hype and ensure measurable validation.
- Specification-First: ✅ Working directly from approved `spec.md`.
- Test-First Development: ✅ Phase 1 includes contract/integration tests before implementation.
- No Shortcuts Policy: ✅ Plan integrates full Dionysus modules rather than stubs; requires real-data validation before readiness.
- Consciousness-Guided Development: ✅ Active inference pipeline, ThoughtSeed channels, curiosity engine all central.
- Modular Component Architecture: ✅ Flux services separated (ingestion, learning_loop, context_engineering, interfaces).
- Preserve Expert ASI-Arch Core: ✅ Extends knowledge graph/ThoughtSeed pipelines; no core replacements.
- Cross-Project Redundancy Safeguard: ✅ Mandatory reuse of Dionysus/SurfSense components with redundancy audits.
- Mock Data Transparency: ✅ Quickstart mandates disclosure and real-data validation before production claims.
- Local-First Operation: ✅ Ollama/LLaMA default, user-controlled storage in requirements.
- Context Engineering Best Practices: ✅ Every flow passes through attractor basins and neural fields per context-engineering specs.

### Constitutional Infrastructure Compliance ⚠️ LOCKED DECISION
- NumPy 2.0+ Mandatory: ✅ All requirements specify NumPy 2.0+ exclusively; binary compatibility solutions implemented.
- No NumPy 1.x: ✅ All legacy NumPy 1.x dependencies eliminated from codebase.
- PyTorch Latest: ✅ Updated to latest PyTorch version with NumPy 2.0 compatibility.
- Sentence-Transformers Fixed: ✅ Compatibility issues resolved for NumPy 2.0.

### Agent Constitution Compliance
- System Integration: ✅ ThoughtSeed integration, ASI-Arch pipeline compatibility, database standards included.
- Agent Behavior: ✅ Status reporting, conflict resolution, testing standards will be implemented in tasks phase.
- Compliance Monitoring: ✅ Pre-operation checks and violation reporting to be added to implementation tasks.

Result: **PASS** (Updated Constitution Check - Infrastructure + Agent Behavior)

### Post-Design Constitution Check (Phase 1)
- Generated artifacts (`data-model.md`, `contracts/*.yaml`, `quickstart.md`, `.github/copilot-instructions.md`) uphold all constitutional constraints.
- Contracts enforce evaluative feedback, mock-data disclosure, knowledge graph SSoT, and local inference requirements.
- Quickstart documents real-data validation prior to readiness and ensures SurfSense credit while emphasizing Flux implementation.

Result: **PASS** (Post-Design Constitution Check)

## Project Structure

### Documentation (this feature)
```
specs/[###-feature]/
├── plan.md              # This file (/plan command output)
├── research.md          # Phase 0 output (/plan command)
├── data-model.md        # Phase 1 output (/plan command)
├── quickstart.md        # Phase 1 output (/plan command)
├── contracts/           # Phase 1 output (/plan command)
└── tasks.md             # Phase 2 output (/tasks command - NOT created by /plan)
```

### Source Code (repository root)
```
# Option 1: Single project (DEFAULT)
src/
├── models/
├── services/
├── cli/
└── lib/

tests/
├── contract/
├── integration/
└── unit/

# Option 2: Web application (when "frontend" + "backend" detected)
backend/
├── src/
│   ├── models/
│   ├── services/
│   └── api/
└── tests/

frontend/
├── src/
│   ├── components/
│   ├── pages/
│   └── services/
└── tests/

# Option 3: Mobile + API (when "iOS/Android" detected)
api/
└── [same as backend above]

ios/ or android/
└── [platform-specific structure]
```

**Structure Decision**: Option 2 (web application) — Flux requires backend services + frontend interface.

## Phase 0: Outline & Research
1. **Extract unknowns from Technical Context** above:
   - For each NEEDS CLARIFICATION → research task
   - For each dependency → best practices task
   - For each integration → patterns task

2. **Generate and dispatch research agents**:
   ```
   For each unknown in Technical Context:
     Task: "Research {unknown} for {feature context}"
   For each technology choice:
     Task: "Find best practices for {tech} in {domain}"
   ```

3. **Consolidate findings** in `research.md` using format:
   - Decision: [what was chosen]
   - Rationale: [why chosen]
   - Alternatives considered: [what else evaluated]

**Output**: research.md with all NEEDS CLARIFICATION resolved

## Phase 1: Design & Contracts
*Prerequisites: research.md complete*

**Artifacts Generated (2025-09-24):**
- `data-model.md`
- `contracts/document-ingestion.yaml`
- `contracts/curiosity-missions.yaml`
- `contracts/visualization-stream.yml`
- `quickstart.md`
- `.github/copilot-instructions.md` (updated via helper script)

1. **Extract entities from feature spec** → `data-model.md`:
   - Entity name, fields, relationships
   - Validation rules from requirements
   - State transitions if applicable

2. **Generate API contracts** from functional requirements:
   - For each user action → endpoint
   - Use standard REST/GraphQL patterns
   - Output OpenAPI/GraphQL schema to `/contracts/`

3. **Generate contract tests** from contracts:
   - One test file per endpoint
   - Assert request/response schemas
   - Tests must fail (no implementation yet)

4. **Extract test scenarios** from user stories:
   - Each story → integration test scenario
   - Quickstart test = story validation steps

5. **Update agent file incrementally** (O(1) operation):
   - Run `.specify/scripts/bash/update-agent-context.sh copilot`
     **IMPORTANT**: Execute it exactly as specified above. Do not add or remove any arguments.
   - If exists: Add only NEW tech from current plan
   - Preserve manual additions between markers
   - Update recent changes (keep last 3)
   - Keep under 150 lines for token efficiency
   - Output to repository root

**Output**: data-model.md, /contracts/*, failing tests, quickstart.md, agent-specific file

## Phase 2: Task Planning Approach
*This section describes what the /tasks command will do - DO NOT execute during /plan*

**Task Generation Strategy**:
- Load `.specify/templates/tasks-template.md` as base
- Generate tasks from Phase 1 design docs (contracts, data model, quickstart)
- Each contract → contract test task [P]
- Each entity → model creation task [P] 
- Each user story → integration test task
- Implementation tasks to make tests pass

**Ordering Strategy**:
- TDD order: Tests before implementation 
- Dependency order: Models before services before UI
- Mark [P] for parallel execution (independent files)

**Estimated Output**: 25-30 numbered, ordered tasks in tasks.md

**IMPORTANT**: This phase is executed by the /tasks command, NOT by /plan

## Phase 3+: Future Implementation
*These phases are beyond the scope of the /plan command*

**Phase 3**: Task execution (/tasks command creates tasks.md)  
**Phase 4**: Implementation (execute tasks.md following constitutional principles)  
**Phase 5**: Validation (run tests, execute quickstart.md, performance validation)

## Complexity Tracking
*Fill ONLY if Constitution Check has violations that must be justified*

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |


## Progress Tracking
*This checklist is updated during execution flow*

**Phase Status**:
- [x] Phase 0: Research complete (/plan command)
- [x] Phase 1: Design complete (/plan command)
- [ ] Phase 2: Task planning complete (/plan command - describe approach only)
- [ ] Phase 3: Tasks generated (/tasks command)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:
- [x] Initial Constitution Check: PASS
- [x] Post-Design Constitution Check: PASS
- [ ] All NEEDS CLARIFICATION resolved
- [ ] Complexity deviations documented

---
*Based on Constitution v2.1.1 - See `/memory/constitution.md`*
