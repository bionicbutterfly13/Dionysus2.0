<!-- OPENSPEC:START -->
# OpenSpec Instructions

These instructions are for AI assistants working in this project.

Always open `@/openspec/AGENTS.md` when the request:
- Mentions planning or proposals (words like proposal, spec, change, plan)
- Introduces new capabilities, breaking changes, architecture shifts, or big performance/security work
- Sounds ambiguous and you need the authoritative spec before coding

Use `@/openspec/AGENTS.md` to learn:
- How to create and apply change proposals
- Spec format and conventions
- Project structure and guidelines

Keep this managed block so 'openspec update' can refresh the instructions.

<!-- OPENSPEC:END -->

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Dionysus 2.0 is a **spec-driven consciousness-enhanced document processing system** implementing:
- Multi-agent coordination via CLAUSE Phase 2 (SubgraphArchitect, CuratedPathNavigator, ProvenanceJournalist)
- LangGraph-based document processing with consciousness integration
- Neo4j-only unified storage (graph + vector + full-text)
- OpenSpec + Archon workflow integration for formal change management

## Architecture

### High-Level Structure
```
Dionysus-2.0/
├── backend/
│   ├── src/
│   │   ├── api/routes/          # FastAPI endpoints
│   │   ├── services/
│   │   │   ├── clause/          # Multi-agent coordination
│   │   │   ├── asi_go_2/        # ASI-GO-2 integration
│   │   │   └── daedalus.py      # Document processing gateway
│   │   └── models/              # Pydantic models
│   └── tests/
│       ├── contract/            # API contract tests
│       ├── integration/         # Component integration tests
│       └── unit/                # Unit tests
├── openspec/                    # OpenSpec change proposals and specs
├── specs/                       # Legacy formal specifications (001-058+)
└── dionysus-source/             # Legacy submodule (read-only)
```

### Core Systems

**Document Processing Pipeline**:
1. `daedalus.py` - Gateway receiving perceptual information
2. `document_processing_graph.py` - 6-node LangGraph workflow
3. ASI-GO-2 components - Cognition Base, Researcher, Analyst
4. Neo4j persistence - Unified storage layer

**Multi-Agent Coordination (CLAUSE Phase 2)**:
- SubgraphArchitect: Concept extraction and subgraph construction
- CuratedPathNavigator: Knowledge graph path finding
- ProvenanceJournalist: Evidence curation with source tracking
- Located in: `backend/src/services/clause/`

**Storage Architecture (Neo4j-Only)**:
- ✅ Graph relationships (attractor basins, thoughtseeds)
- ✅ Vector search (512-dim embeddings, cosine similarity)
- ✅ Full-text search (built-in indexing)
- ✅ Hybrid queries (single Cypher query)
- Note: Qdrant removed, archived in `backup/deprecated/qdrant_vector_searcher/`

## Development Approach

**Spec-Driven Development**: All implementation follows formal specifications using OpenSpec + Archon integration:

### OpenSpec + Archon Workflow

This project uses **OpenSpec** for specifications and **Archon MCP** for task management in an integrated workflow:

**OpenSpec (Specifications & Change Management)**:
- **Purpose**: Formal change proposals, specs, and design documentation
- **Location**: `openspec/` directory
- **Commands**:
  - `/openspec:proposal` - Create new change proposal with spec deltas and tasks
  - `/openspec:apply` - Implement approved change
  - `/openspec:archive` - Archive completed change
- **CLI**: `openspec list`, `openspec validate <id> --strict`, `openspec show <id>`

**Archon MCP (Task Management)**:
- **Purpose**: Project and task tracking, knowledge base integration
- **Primary System**: All task management flows through Archon
- **Tools**:
  - `find_projects()` / `manage_project()` - Project management
  - `find_tasks()` / `manage_task()` - Task lifecycle (todo → doing → review → done)
  - `rag_search_knowledge_base()` - Research before implementation

**Integration Workflow**:
```
1. OpenSpec Proposal → Create formal change specification
   `/openspec:proposal` generates:
   - proposal.md (what/why/how)
   - tasks.md (implementation tasks)
   - design.md (architecture decisions)
   - specs/ (requirement deltas)

2. Bridge to Archon → Create Archon project + tasks
   - Create project: `manage_project("create", title="...")`
   - Import tasks from OpenSpec tasks.md into Archon
   - Assign task_order for priority/sequence

3. Implementation → Work through Archon tasks
   - Get task: `find_tasks(filter_by="status", filter_value="todo")`
   - Start: `manage_task("update", task_id="...", status="doing")`
   - Research: `rag_search_knowledge_base()`, `rag_search_code_examples()`
   - Implement: Follow OpenSpec proposal and design
   - Review: `manage_task("update", task_id="...", status="review")`
   - Complete: `manage_task("update", task_id="...", status="done")`

4. Archive → Complete OpenSpec change
   - `/openspec:archive` moves to archive/ and updates main specs
   - Archive Archon project when all tasks done
```

**Best Practices**:
- ✅ **Always check existing** OpenSpec specs and Archon tasks before implementing
- ✅ **Research first**: Use Archon RAG to find relevant patterns/docs
- ✅ **One task in progress**: Mark exactly one Archon task as "doing" at a time
- ✅ **Update continuously**: Mark tasks complete immediately after finishing
- ✅ **Validate strictly**: Run `openspec validate <id> --strict` before sharing proposals

**Legacy Specs** (Pre-OpenSpec):
- `specs/`: 58+ numbered specifications (001-058+)
  - 6 completed implementations running in production (021, 034, 035, 040, 054, 055)
  - Still referenced by backend code, tests, and documentation
  - **Status**: ACTIVE (not deprecated), maintained for historical reference
- `spec-management/Consciousness-Specs/`:
  - `CLEAN_CONSCIOUSNESS_SPEC.md`: Core implementation specification
  - `UNIFIED_DATABASE_MIGRATION_SPEC.md`: Database architecture migration plan
  - `CONSCIOUSNESS_IMPLEMENTATION_SUMMARY.md`: Implementation status and results

### Current Specification State (2025-11-15)

**OpenSpec Migration Complete**:
- ✅ Migrated 3 core capabilities to OpenSpec format:
  - `openspec/specs/document-processing/` (from specs 021, 054, 055)
  - `openspec/specs/clause-multi-agent/` (from specs 034, 035)
  - `openspec/specs/knowledge-graph/` (from specs 001, 040)
- ✅ Each capability has `spec.md` (requirements) and `design.md` (implementation patterns)
- ✅ OpenSpec CLI v0.14.0 installed and validated

**Active Change Proposals**:
1. ~~**integrate-openspec-archon-sync**~~ → **✅ IMPLEMENTED** (2025-11-16)
   - Creates `/openspec:import-to-archon` command for automatic task import
   - Syncs Archon task completion back to OpenSpec `tasks.md`
   - Validates Archon completion on `/openspec:archive`
   - **Files**: `.claude/commands/openspec/{import-to-archon,sync-status}.md`, `scripts/sync_openspec_archon_status.py`

2. **ingest-specs-to-neo4j**: Pipeline to ingest OpenSpec specs into knowledge graph
   - Processes specs through Daedalus → LangGraph → Neo4j
   - Enables semantic search across specifications
   - Discovers cross-spec relationships via ThoughtSeeds
   - **Status**: Proposal complete, ready for implementation

**Dual-System Coexistence**:
- **Legacy `specs/`**: Historical numbered specifications (001-058+), read-only reference
- **OpenSpec `openspec/specs/`**: Current capability specifications, actively maintained
- Both systems valid: Legacy for completed features, OpenSpec for new changes

**OpenSpec ↔ Archon Integration** (✅ ACTIVE):
```
Current Workflow (Manual Sync):
  1. OpenSpec proposal → /openspec:import-to-archon <id> → Archon project + tasks
  2. Work on tasks → Archon MCP (todo → doing → review → done)
  3. Sync status → /openspec:sync-status <id> → Auto-update tasks.md checkboxes
  4. Archive → /openspec:archive <id> → Validate Archon completion → Archive both

Automated Sync (Background Daemon):
  1. Start daemon: python scripts/openspec_sync_daemon.py start
  2. Work on tasks → Archon MCP (todo → doing → review → done)
  3. Tasks.md auto-updates every 30s (no manual sync needed!)
  4. Archive → /openspec:archive <id> → Validate Archon completion → Archive both

Quick Sync (All Changes):
  ./scripts/sync_all_changes.sh
```

**Sync Options**:
- **Manual**: Run `/openspec:sync-status <id>` when needed
- **Daemon**: Background polling every 30s (see `docs/AUTOMATED_SYNC_GUIDE.md`)
- **Scheduled**: Cron/launchd for periodic sync
- **Git Hooks**: Auto-sync before commits

### Practical Examples

**Example 1: Starting a new feature**
```bash
# 1. Create OpenSpec change proposal
/openspec:proposal add-user-analytics

# 2. Import to Archon (creates project + tasks automatically)
/openspec:import-to-archon add-user-analytics

# Output:
# ✅ Import complete!
# Archon Project: 550e8400-e29b-41d4-a716-446655440000
# Tasks created: 12
# Reference stored: .archon-project-id

# 3. Get next task
mcp__archon__find_tasks(filter_by="status", filter_value="todo")

# 4. Start working
mcp__archon__manage_task("update", task_id="task-123", status="doing")

# 5. Implement feature...

# 6. Mark complete
mcp__archon__manage_task("update", task_id="task-123", status="done")

# 7. Sync progress to OpenSpec (updates checkboxes)
/openspec:sync-status add-user-analytics

# Output:
# ✅ Sync complete!
# Completion: 1/12 tasks (8%)
# Updates applied: 1
```

**Example 2: Checking progress**
```bash
# View all tasks for a project
mcp__archon__find_tasks(project_id="550e8400-...")

# Check completion percentage
/openspec:sync-status add-user-analytics

# Output shows:
# Completion: 8/12 tasks (67%)
```

**Example 3: Archiving completed change**
```bash
# After completing all tasks, sync one final time
/openspec:sync-status add-user-analytics

# Archive (with Archon validation)
/openspec:archive add-user-analytics

# If tasks incomplete:
# ⚠️ Archon project has 4 incomplete tasks (3 todo, 1 doing)
# Archive anyway? (This will leave incomplete tasks in Archon)

# If all complete:
# ✅ All Archon tasks complete (12 tasks)
# Archive Archon project too? (Recommended) → yes
# ✅ OpenSpec change archived to changes/archive/2025-11-16-add-user-analytics/
# ✅ Archon project archived
```

**Example 4: Researching before implementation**
```bash
# Search knowledge base for related patterns
mcp__archon__rag_search_knowledge_base(query="analytics tracking")

# Find code examples
mcp__archon__rag_search_code_examples(query="event logging")

# List available documentation sources
mcp__archon__rag_get_available_sources()
```

**OpenSpec → Neo4j Integration** (via ingest-specs-to-neo4j):
```
openspec/specs/*.md → POST /api/documents → Daedalus → LangGraph → Neo4j
  ↓
Document:Specification nodes with:
  - Semantic search capabilities
  - Cross-spec ThoughtSeeds
  - Requirement/Scenario extraction
  - AttractorBasin clustering
```

## Development Commands

### Running the System

```bash
# Start backend API server
cd backend
uvicorn src.app_factory:app --host 127.0.0.1 --port 9127 --reload

# Or use the main entry point
cd backend/src
python main.py

# Check server health
curl http://localhost:9127/health
```

### Testing (TDD Workflow)

```bash
# Run all tests
cd backend
pytest

# Run specific test categories
pytest -m contract      # API contract tests
pytest -m integration   # Integration tests
pytest -m unit         # Unit tests

# Run specific test file
pytest tests/contract/test_documents_persist_post.py

# Run with coverage
pytest --cov=src --cov-report=html
```

**TDD Cycle (see TDD_RULES.md)**:
1. RED: Write failing test first
2. GREEN: Minimal code to pass test
3. REFACTOR: Improve while keeping tests green

### Database Setup

**Neo4j (Required)**:
```bash
# Native installation
brew install neo4j && brew services start neo4j

# Or use Neo4j Aura cloud: https://neo4j.com/cloud/aura/
# Set environment variables:
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=your_password
```

**Redis (Required for consciousness processing)**:
```bash
# Native installation
brew install redis && brew services start redis

# Or use Redis Cloud: https://redis.com/try-free/
export REDIS_HOST=localhost
export REDIS_PORT=6379
```

### Linting and Code Quality

```bash
# Run linter
cd backend
ruff check src

# Auto-fix issues
ruff check src --fix

# Format code
ruff format src

# Type checking
mypy src
```

## CLAUSE Multi-Agent System

Access via: `POST /api/demo/process-document`

```typescript
// Upload document
const formData = new FormData();
formData.append('file', fileBlob, 'document.txt');

const response = await fetch('http://localhost:8001/api/demo/process-document', {
  method: 'POST',
  body: formData
});

// Response includes:
// - concepts_extracted: ["climate_change", ...]
// - agent_handoffs: [{agent: "SubgraphArchitect", latency_ms: 0.02}]
// - total_time_ms: 32.26
```

## Common Issues

### NumPy Version Conflicts
Backend requires NumPy < 2.0 (see `backend/pyproject.toml`). If you encounter NumPy 2.x issues:
```bash
pip install "numpy>=1.24.4,<2.0.0"
```

### Module Import Errors
Backend uses `backend/src/` as the root. Ensure imports use:
```python
from services.daedalus import Daedalus  # NOT from src.services
```

### Database Connection Issues
Check services are running:
```bash
# Neo4j
neo4j status

# Redis
redis-cli ping  # Should return PONG
```

## Important Project Conventions

1. **Spec-First Development**: All changes follow formal specifications in `specs/`
2. **Docker-Independent**: Native installation preferred over Docker
3. **Test-First**: Follow TDD cycle (see TDD_RULES.md)
4. **Contract Tests**: API changes require contract tests in `tests/contract/`
5. **Modular Components**: Some components extracted (e.g., daedalus-gateway)

## File Reference Patterns

When referencing code, use: `file_path:line_number`

Example: "The gateway implementation is in `backend/src/services/daedalus.py:118`"

## Consciousness Systems (New - 2025-11-13)

### Skills Database
Process your accumulated skills through the consciousness system:
```bash
# Check status
python backend/check_consciousness_systems.py

# Initialize skills database
python backend/initialize_skills.py
```

**What it does**:
- Scans `/Volumes/Asylum/skills-library/`
- Processes each skill through Daedalus → DocumentProcessingGraph
- Creates 5-level concepts, attractor basins, thoughtseeds
- Stores in Neo4j knowledge graph
- Builds searchable skill index at `~/.claude/skills/index.json`

### Session Startup Check
**Automatic Status Report**: Every session should start by checking consciousness systems:
```bash
python backend/check_consciousness_systems.py
```

Shows:
- ✅ Skills database status (initialized/not initialized)
- ✅ Neo4j health (node counts, consciousness nodes)

---

## Recent Major Changes

- **Consciousness Systems Integration** (2025-11-13): Added Skills Manager for consciousness-enhanced knowledge management
- **OpenSpec + Archon Integration**: Replaced BMAD framework with OpenSpec for change management, integrated with Archon MCP for task tracking
- **CLAUSE Phase 2**: Multi-agent coordination system operational
- **Neo4j Unified Storage**: Removed Qdrant, single database architecture
- **LangGraph Integration**: 6-node consciousness-enhanced document processing
