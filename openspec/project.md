# Project Context

## Purpose
Dionysus 2.0 is a **spec-driven consciousness-enhanced document processing system** implementing:
- Multi-agent coordination via CLAUSE Phase 2 (SubgraphArchitect, CuratedPathNavigator, ProvenanceJournalist)
- LangGraph-based document processing with consciousness integration
- Neo4j-only unified storage (graph + vector + full-text)
- OpenSpec + Archon workflow integration for formal change management

## Tech Stack

### Backend
- **Language**: Python 3.11+
- **Framework**: FastAPI, Pydantic 2.x
- **Async**: asyncio, uvicorn
- **Document Processing**: LangGraph, Daedalus Gateway, ASI-GO-2
- **Consciousness Framework**: ThoughtSeeds, Active Inference

### Frontend
- **Language**: TypeScript
- **Framework**: React 18, Vite
- **Visualization**: Three.js (3D knowledge graph)
- **State Management**: TanStack Query

### Databases
- **Neo4j**: Primary unified storage (graph + vector + full-text search)
- **Redis**: Optional caching for consciousness processing
- **PostgreSQL**: Optional structured data (via Supabase)

### Infrastructure
- **Native Installation Preferred**: Docker-independent, uses brew/apt
- **Cloud Alternatives**: Neo4j Aura, Redis Cloud, Supabase

## Project Conventions

### Code Style
- **Python**: Ruff for linting and formatting, MyPy for type checking
- **TypeScript**: ESLint, Prettier
- **Imports**: Backend uses `backend/src/` as root (e.g., `from services.daedalus import Daedalus`)
- **NumPy**: < 2.0 required for PyTorch compatibility

### Architecture Patterns
- **Spec-Driven Development**: All changes follow formal specifications in `specs/`
- **Test-Driven Development**: RED-GREEN-REFACTOR cycle required
- **Constitutional Compliance**: All Neo4j operations via DaedalusGraphChannel (no direct neo4j imports)
- **Modular Components**: External packages (thoughtseeds, daedalus, asi-go-2) not internal implementations

### Testing Strategy
- **Contract Tests**: API changes require contract tests in `tests/contract/`
- **Integration Tests**: Component integration in `tests/integration/`
- **Unit Tests**: Granular testing in `tests/unit/`
- **Test Markers**: Use pytest markers (contract, integration, unit)
- **Coverage**: Run with `pytest --cov=src --cov-report=html`
- **Watch Mode**: Use `ptw -- -m unit` for TDD

### Git Workflow
- **Main Branch**: `main`
- **Feature Branches**: Recommended for major changes
- **Commit Messages**: Descriptive with context
- **Co-Authoring**: Include Claude Code co-authorship for AI-assisted commits

## Domain Context

### Consciousness-Enhanced Processing
- **5-Level Concept Hierarchy**: Documents processed through consciousness system
- **Attractor Basins**: Knowledge domain clustering
- **ThoughtSeeds**: Cross-document relationship discovery
- **Meta-Cognitive Tracking**: Self-reflective pattern development

### LangGraph Workflow (6 nodes)
1. Extract & Process (SurfSense patterns)
2. Generate Research Plan (ASI-GO-2 + R-Zero)
3. Consciousness Processing (Basins + ThoughtSeeds)
4. Analyze Results (Quality + Insights + Meta-cognitive)
5. Refine Processing (Iterative improvement)
6. Finalize Output (Package results)

### CLAUSE Multi-Agent System
- **SubgraphArchitect**: Concept extraction and subgraph construction
- **CuratedPathNavigator**: Knowledge graph path finding
- **ProvenanceJournalist**: Evidence curation with source tracking

## Important Constraints

### Constitutional Compliance (Spec 040)
- All Neo4j operations **MUST** use DaedalusGraphChannel
- **NO** direct neo4j imports allowed in application code
- Graph Channel is single point of database access

### NumPy Version Constraint
- NumPy **MUST** be < 2.0 for PyTorch compatibility
- Use frozen environment if NumPy 2.0 needed: `source activate-numpy2-frozen.sh`

### Storage Architecture
- **Neo4j Only**: Removed Qdrant (archived in `backup/deprecated/qdrant_vector_searcher/`)
- **Single Database**: No data duplication across multiple systems
- **Unified Schema**: Document, Concept, AttractorBasin, ThoughtSeed, Chunk nodes

### Legacy Code
- **BMAD Framework**: Completely removed (TDD verified)
- **dionysus-source**: Legacy submodule, read-only reference

## External Dependencies

### Required Packages (pip)
- `thoughtseeds`: ThoughtSeed generation and management
- `daedalus`: Daedalus Gateway for document processing
- `asi-go-2`: ASI-GO-2 learning architecture
- `daedalus_gateway`: Graph Channel for Neo4j access

### Database Services
- **Neo4j**: Required for document processing (native or Aura)
- **Redis**: Optional for consciousness caching (native or Cloud)
- **PostgreSQL**: Optional for structured data (native or Supabase)

### External APIs
- **OpenAI**: Embeddings and language models (via consciousness framework)
- **Anthropic**: Claude models for multi-agent coordination

### Development Tools
- **OpenSpec**: Spec-driven change management (`npm install -g openspec`)
- **Archon MCP**: Task tracking and knowledge base integration
