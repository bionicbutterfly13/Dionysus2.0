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
1. `daedalus.py` (118 lines) - Gateway receiving perceptual information
2. `document_processing_graph.py` - 6-node LangGraph workflow
3. ASI-GO-2 components - Cognition Base, Researcher, Analyst
4. Neo4j persistence - Unified storage layer

**Multi-Agent Coordination (CLAUSE Phase 2)**:
- SubgraphArchitect: Concept extraction and subgraph construction
- CuratedPathNavigator: Knowledge graph path finding
- ProvenanceJournalist: Evidence curation with source tracking
- Located in: `backend/src/services/clause/`

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

# TDD: Watch mode (requires pytest-watch)
ptw -- -m unit
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
# Set environment variables if needed:
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

# Format code (uses black via ruff)
ruff format src

# Type checking
mypy src
```

## Spec-Driven Development

### Using OpenSpec for Changes

When implementing new features or breaking changes:

1. **Check if spec exists**: Look in `specs/` directory
2. **Read existing spec**: Each spec has `spec.md` and `contract/` tests
3. **Follow TDD**: Tests should reflect spec requirements
4. **For new proposals**: See AGENTS.md for OpenSpec workflow

Example spec structure:
```
specs/054-document-persistence/
├── spec.md              # Specification document
├── contract/            # API contract tests
├── plan.md             # Implementation plan
└── tasks.md            # Task breakdown
```

## Key Concepts

### Storage Architecture (Neo4j-Only)

**Decision**: Removed Qdrant, using Neo4j for everything
- ✅ Graph relationships (attractor basins, thoughtseeds)
- ✅ Vector search (512-dim embeddings, cosine similarity)
- ✅ Full-text search (built-in indexing)
- ✅ Hybrid queries (single Cypher query)

**When removed**: Qdrant archived in `backup/deprecated/qdrant_vector_searcher/`

### CLAUSE Multi-Agent System

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

### External Dependencies

These are **external packages**, not internal implementations:
```bash
pip install thoughtseeds daedalus asi-go-2
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

## Recent Major Changes

- **OpenSpec + Archon Integration**: Replaced BMAD with OpenSpec for change management, integrated with Archon MCP for task tracking
- **CLAUSE Phase 2**: Multi-agent coordination system operational
- **Neo4j Unified Storage**: Removed Qdrant, single database architecture
- **LangGraph Integration**: 6-node consciousness-enhanced document processing
