# Dionysus Backend

Backend API for the Dionysus 2.0 consciousness-enhanced document processing system.

## Prerequisites

### Required Services

1. **Ollama** (Local LLM - REQUIRED)
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.com/install.sh | sh

   # Pull required model
   ollama pull qwen2.5:7b

   # Verify installation
   ollama list
   ollama serve  # Runs on http://localhost:11434
   ```

2. **Neo4j** (Knowledge Graph Database - REQUIRED)
   ```bash
   # Install via Homebrew
   brew install neo4j
   brew services start neo4j

   # Or use Neo4j Aura Cloud: https://neo4j.com/cloud/aura/

   # Default connection: bolt://localhost:7687
   # Access browser: http://localhost:7474
   ```

3. **Redis** (Caching & State Management - REQUIRED)
   ```bash
   # Install via Homebrew
   brew install redis
   brew services start redis

   # Or use Redis Cloud: https://redis.com/try-free/

   # Default connection: localhost:6379
   # Test: redis-cli ping  # Should return PONG
   ```

### Environment Configuration

Copy `.env.example` to `.env` and configure:

```bash
# LLM Configuration (uses local Ollama by default)
LLM_PROVIDER=ollama
OLLAMA_ENDPOINT=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b

# Database Connections
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password_here

REDIS_HOST=localhost
REDIS_PORT=6379
```

## Quick Start

### Running the Server

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

### Testing

```bash
# Run all tests
cd backend
pytest

# Run specific test categories
pytest -m contract      # API contract tests
pytest -m integration   # Integration tests
pytest -m unit         # Unit tests

# Run with coverage
pytest --cov=src --cov-report=html
```

## Scripts

### OpenSpec Specification Ingestion

**Location**: `backend/scripts/ingest_openspec_specs.py`

**Purpose**: Ingest OpenSpec specification documents (spec.md, design.md) into the Neo4j knowledge graph through the existing Daedalus → LangGraph → DocumentRepository consciousness pipeline.

**What it does**:
- Scans `openspec/specs/` directory for spec.md and design.md files
- Extracts metadata (capability name, spec type, content hash)
- Uploads files to the document processing API
- Processes specs through consciousness-enhanced pipeline
- Creates searchable Document:Specification nodes in Neo4j
- Extracts requirements and scenarios as Concept nodes
- Discovers cross-spec relationships via ThoughtSeeds

**How to run it**:

```bash
# Ingest all OpenSpec specifications
python backend/scripts/ingest_openspec_specs.py --all

# Ingest specific capability
python backend/scripts/ingest_openspec_specs.py --capability document-processing

# Preview without ingesting (dry run)
python backend/scripts/ingest_openspec_specs.py --all --dry-run

# Watch mode (auto-ingest on file changes)
python backend/scripts/ingest_openspec_specs.py --all --watch

# Use custom API URL
python backend/scripts/ingest_openspec_specs.py --all --api-url http://localhost:8000
```

**Available flags**:

| Flag | Description |
|------|-------------|
| `--all` | Ingest all capabilities in openspec/specs/ |
| `--capability <name>` | Ingest specific capability (e.g., document-processing) |
| `--dry-run` | Preview files without actually ingesting |
| `--watch` | Watch for file changes and auto-ingest |
| `--api-url <url>` | API base URL (default: http://localhost:9127) |

**Example usage**:

```bash
# First time setup - ingest all existing specs
python backend/scripts/ingest_openspec_specs.py --all

# Output:
# Found 6 spec files to ingest
# Ingesting: openspec/specs/document-processing/spec.md... ✓ Success
# Ingesting: openspec/specs/document-processing/design.md... ✓ Success
# Ingesting: openspec/specs/clause-multi-agent/spec.md... ✓ Success
# Ingesting: openspec/specs/clause-multi-agent/design.md... ✓ Success
# Ingesting: openspec/specs/knowledge-graph/spec.md... ✓ Success
# Ingesting: openspec/specs/knowledge-graph/design.md... ✓ Success
#
# Summary:
#   ✓ Success: 6
#   ⊘ Duplicates: 0
#   ✗ Failed: 0

# Check what would be ingested before running
python backend/scripts/ingest_openspec_specs.py --all --dry-run

# Output:
# Found 6 spec files to ingest
# [DRY RUN] Would ingest: openspec/specs/document-processing/spec.md
# [DRY RUN] Would ingest: openspec/specs/document-processing/design.md
# [DRY RUN] Would ingest: openspec/specs/clause-multi-agent/spec.md
# [DRY RUN] Would ingest: openspec/specs/clause-multi-agent/design.md
# [DRY RUN] Would ingest: openspec/specs/knowledge-graph/spec.md
# [DRY RUN] Would ingest: openspec/specs/knowledge-graph/design.md

# Ingest only one capability
python backend/scripts/ingest_openspec_specs.py --capability document-processing

# Output:
# Found 2 spec files to ingest
# Ingesting: openspec/specs/document-processing/spec.md... ✓ Success
# Ingesting: openspec/specs/document-processing/design.md... ✓ Success
#
# Summary:
#   ✓ Success: 2
#   ⊘ Duplicates: 0
#   ✗ Failed: 0

# Watch mode - auto-ingest when specs change
python backend/scripts/ingest_openspec_specs.py --all --watch

# Output:
# Watching openspec/specs/ for changes...
# Press Ctrl+C to stop
# [2025-11-17 10:30:45] Change detected: openspec/specs/document-processing/spec.md
# Ingesting: openspec/specs/document-processing/spec.md... ✓ Success
```

**Expected output**:

- **Success (200)**: Spec processed and stored in Neo4j
- **Duplicate (409)**: Spec already ingested (same content hash)
- **Failed (4xx/5xx)**: Error during processing (check API logs)

**Integration with OpenSpec + Archon workflow**:

1. Create OpenSpec change proposal: `/openspec:proposal add-feature`
2. **Ingest specs**: `python backend/scripts/ingest_openspec_specs.py --all`
3. Import to Archon: `/openspec:import-to-archon add-feature`
4. Work through tasks in Archon MCP
5. Sync progress: `/openspec:sync-status add-feature`
6. Archive when complete: `/openspec:archive add-feature`

**What gets created in Neo4j**:

```cypher
// Document nodes with Specification label
(d:Document:Specification {
  id: "uuid",
  title: "Document Processing",
  content_hash: "abc123...",
  source_type: "openspec",
  capability: "document-processing",
  spec_type: "spec",
  version: "1.0",
  created_at: datetime()
})

// Extracted requirements
(r:Concept:Requirement {
  title: "Import OpenSpec specifications"
})

// Scenarios with given/when/then
(s:Concept:Scenario {
  title: "Ingest single capability",
  given: "OpenSpec specs exist",
  when: "User runs script with --capability",
  then: "Only that capability is processed"
})

// Relationships
(d)-[:HAS_REQUIREMENT]->(r)
(r)-[:HAS_SCENARIO]->(s)
(d)-[:SIMILAR_TO]->(other_spec)  // Discovered via ThoughtSeeds
```

**Searching ingested specs**:

```bash
# Semantic search via API
curl http://localhost:9127/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "authentication patterns", "filters": {"source_type": "openspec"}}'

# Returns specs that mention authentication, ranked by relevance
```

## Architecture

See `/Volumes/Asylum/dev/Dionysus-2.0/CLAUDE.md` for full architecture documentation.

## Environment Variables

```bash
# Neo4j
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=your_password

# Redis
export REDIS_HOST=localhost
export REDIS_PORT=6379

# API
export API_HOST=127.0.0.1
export API_PORT=9127
```

## Development

See TDD_RULES.md for test-driven development workflow.
