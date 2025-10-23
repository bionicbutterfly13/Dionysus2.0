# Project Context

## Purpose
Dionysus-2.0 is a consciousness-enhanced document processing system that implements consciousness-guided neural architecture discovery through active inference, consciousness framework integration, and unified database architecture. The system processes documents through a LangGraph workflow integrating ThoughtSeeds, attractor basins, and meta-cognitive awareness.

## Tech Stack

### Frontend (Desktop App - Tauri)
- **Desktop Framework**: Tauri 2.0 (Rust backend + WebView frontend)
- **UI Framework**: React 18 + TypeScript
- **Build Tool**: Vite 5.0
- **State Management**: Zustand 4.4
- **Data Fetching**: TanStack React Query 5.90
- **Routing**: React Router DOM 6.20 (HashRouter for desktop)
- **Styling**: Tailwind CSS 3.3
- **3D Visualization**: Three.js 0.158 + React Three Fiber 8.18
- **Native Features**: File system, menus, windows, dialogs via Tauri plugins

### Backend (API)
- **Framework**: Python 3.11+ with FastAPI
- **ML/AI**: PyTorch 2.6.0 + CUDA 12.4
- **Graph Processing**: LangGraph for consciousness workflows
- **Validation**: Pydantic 2.x

### Database Architecture (Neo4j-Only)
- **Primary**: Neo4j (graph + vector + full-text unified storage)
- **Integration**: AutoSchemaKG for automatic concept extraction
- **Optional Cache**: Redis (for performance optimization)
- **Deprecated**: Qdrant (archived, Neo4j now handles all vector operations)

### External Packages (NOT Internal Implementations)
- **ThoughtSeeds**: Consciousness-enhanced processing
- **Daedalus**: Universal coordinator gateway
- **ASI-GO-2**: 4-component learning architecture

## Project Conventions

### Code Style
- **Python**: PEP 8, type hints required, async/await patterns
- **TypeScript**: Strict mode, functional components with hooks
- **Naming**:
  - Frontend: camelCase for functions/variables, PascalCase for components
  - Backend: snake_case for Python, descriptive names
- **Formatting**: Prettier (frontend), Black (backend)

### Architecture Patterns
- **Spec-Driven Development**: All changes follow OpenSpec workflow
- **LangGraph Workflows**: 6-node consciousness processing pipeline
- **Clean Synthesis**: Single-responsibility components, clear source attribution
- **No Docker Dependency**: All services support native installation (brew, apt) or managed cloud

### File Organization
- **Source Attribution**: `SOURCES_AND_ATTRIBUTIONS.md` for external integrations
- **Protected Archives**: `backup/clean_implementations/` for reference implementations
- **Specs**: `spec-management/Consciousness-Specs/` for formal specifications
- **Deprecated Code**: `backup/deprecated/` for archived components

### Testing Strategy
- **Frontend**: Jest + React Testing Library + Playwright E2E
- **Backend**: pytest with integration and contract tests
- **Coverage**: Comprehensive test suites for consciousness processing
- **Validation**: OpenSpec validation for all changes

### Git Workflow
- **Spec-First**: Changes require OpenSpec proposals before implementation
- **Feature Branches**: All work in feature branches, never on main
- **Commits**: Descriptive messages following conventional commits
- **Preservation**: Original functionality completely preserved during enhancements

## Domain Context

### Consciousness Processing Core
The system implements a 6-node LangGraph workflow:
1. Extract & Process (SurfSense patterns)
2. Generate Research Plan (ASI-GO-2 + R-Zero curiosity)
3. Consciousness Processing (Basins + ThoughtSeeds)
4. Analyze Results (Quality + Insights + Meta-cognitive)
5. Refine Processing (Iterative improvement)
6. Finalize Output (Package results)

### Key Concepts
- **Attractor Basins**: Stable cognitive states in consciousness processing
- **ThoughtSeeds**: Consciousness enhancement units with meta-awareness
- **Active Inference**: Prediction error minimization for pattern evolution
- **Meta-Cognitive Awareness**: Self-reflective consciousness detection
- **AutoSchemaKG**: Automatic knowledge graph construction from concepts

### Integration Sources
- SurfSense (MIT) - Document extraction patterns
- ASI-GO-2 (MIT) - 4-component learning architecture
- R-Zero (Apache 2.0) - Curiosity-driven co-evolution
- David Kimai's Context Engineering (MIT) - 12x token efficiency
- OpenNotebook (MIT) - LangGraph workflow patterns
- Perplexica (MIT) - Text splitting validation

## Important Constraints

### Technical
- **NumPy 2.0 Frozen Environment**: Use frozen binary environment to avoid compatibility issues
- **No Docker Required**: All services must support native or cloud deployment
- **Desktop-First**: Flux is a native desktop app (Tauri framework, NOT web app)
- **Lightweight Distribution**: Target <15MB bundle size (Tauri) vs 85-100MB (Electron)
- **Token Efficiency**: Consciousness processing optimized for minimal token usage

### Business
- **Preservation Principle**: Original functionality must remain intact during enhancements
- **External Dependencies**: ThoughtSeeds, Daedalus, ASI-GO-2 are external packages, never implemented internally
- **Clean Source Attribution**: All external integrations documented with licenses

### Regulatory
- **Open Source Compliance**: MIT/Apache 2.0 license compatibility required
- **Attribution Required**: All external sources documented in SOURCES_AND_ATTRIBUTIONS.md

## External Dependencies

### Services (Docker-Independent)
- **Neo4j**: Graph database + vector search + full-text (native: brew/apt or Neo4j Aura)
- **Redis**: Optional caching layer (native: brew/apt or Redis Cloud)
- **PostgreSQL**: Optional structured data (native: brew/apt or Supabase)

### Python Packages
- **Core ML**: PyTorch 2.6.0, NumPy 2.3.3 (frozen)
- **Graph**: LangGraph, AutoSchemaKG, neo4j-driver
- **External**: thoughtseeds, daedalus, asi-go-2 (pip install)

### Frontend Packages
- **React Ecosystem**: @tanstack/react-query, zustand, react-router-dom
- **3D Graphics**: three, @react-three/fiber, @react-three/drei
- **Utils**: axios, react-dropzone, lucide-react

## Current Development Focus

### Zettlr Feature Parity Strategy
**Decision**: Build native desktop app (Tauri) with full Zettlr feature parity (154 features)
- **Framework**: Tauri 2.0 (90% smaller than Electron, 4x faster startup)
- **Editor**: CodeMirror 6 for markdown editing
- **Features**: Complete implementation of Zettlr's core capabilities
- **Enhancement**: Add Flux consciousness features (ThoughtSeeds, Basins, Active Inference)
- **Timeline**: 3-month MVP with 75 critical/high priority features

### Active Priorities (3-Month Roadmap)
**Month 1: Foundation + Core Editing**
1. Tauri + React desktop setup (Week 1)
2. CodeMirror 6 markdown editor (Week 3)
3. File system + workspace management (Week 4)

**Month 2: Essential Features**
4. Search & navigation (Week 5)
5. Tabs + split view (Week 7)
6. Graph view + WikiLinks (Week 8)

**Month 3: Advanced + Distribution**
7. Export system (Pandoc integration) (Week 9)
8. Citations (Zotero integration) (Week 10)
9. Themes + settings + distribution (Weeks 11-12)

### Completed Major Milestones
- ✅ LangGraph synthesis with clean source attribution (2025-10-01)
- ✅ Neo4j-only unified database architecture (2025-10-01)
- ✅ Qdrant removal and consolidation (2025-10-01)
- ✅ Document processing system with consciousness enhancement
- ✅ NumPy 2.0 frozen environment for stability
