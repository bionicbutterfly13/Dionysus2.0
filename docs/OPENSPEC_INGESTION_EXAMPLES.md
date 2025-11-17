# OpenSpec Ingestion CLI - Usage Examples

This document provides practical examples for using the OpenSpec ingestion CLI to import specifications into the Neo4j knowledge graph.

## Table of Contents

- [Quick Start](#quick-start)
- [Basic Usage](#basic-usage)
- [Advanced Usage](#advanced-usage)
- [Understanding Output](#understanding-output)
- [Error Scenarios](#error-scenarios)
- [Troubleshooting](#troubleshooting)
- [Integration Workflow](#integration-workflow)

---

## Quick Start

```bash
# 1. Start backend API server
cd backend
uvicorn src.app_factory:app --host 127.0.0.1 --port 9127 --reload

# 2. Ingest all OpenSpec specifications
python scripts/ingest_openspec_specs.py --all
```

---

## Basic Usage

### Ingest All Specifications

Ingest all `spec.md` and `design.md` files from all capabilities in `openspec/specs/`.

```bash
python backend/scripts/ingest_openspec_specs.py --all
```

**Expected Output:**
```
Found 6 spec files to ingest
Ingesting: openspec/specs/document-processing/spec.md... ✓ Success
Ingesting: openspec/specs/document-processing/design.md... ✓ Success
Ingesting: openspec/specs/clause-multi-agent/spec.md... ✓ Success
Ingesting: openspec/specs/clause-multi-agent/design.md... ✓ Success
Ingesting: openspec/specs/knowledge-graph/spec.md... ✓ Success
Ingesting: openspec/specs/knowledge-graph/design.md... ✓ Success

Summary:
  ✓ Success: 6
  ⊘ Duplicates: 0
  ✗ Failed: 0
```

**What Happens:**
1. Script scans `openspec/specs/` for all `.md` files
2. Extracts metadata:
   - `capability` = parent directory name (e.g., "document-processing")
   - `spec_type` = file name without extension (e.g., "spec" or "design")
   - `content_hash` = SHA-256 hash of file content (for deduplication)
3. Sends each file to `POST /api/documents` with metadata
4. Daedalus processes through DocumentProcessingGraph (6 nodes)
5. Neo4j stores as `Document:Specification` nodes with consciousness analysis

---

### Ingest Specific Capability

Ingest only the specifications for a single capability.

```bash
python backend/scripts/ingest_openspec_specs.py --capability document-processing
```

**Expected Output:**
```
Found 2 spec files to ingest
Ingesting: openspec/specs/document-processing/spec.md... ✓ Success
Ingesting: openspec/specs/document-processing/design.md... ✓ Success

Summary:
  ✓ Success: 2
  ⊘ Duplicates: 0
  ✗ Failed: 0
```

**Use Cases:**
- Selective ingestion after updating a specific capability
- Testing ingestion on a single capability
- Re-processing a capability after schema changes

---

## Advanced Usage

### Dry Run Mode

Preview which files would be ingested without actually processing them.

```bash
python backend/scripts/ingest_openspec_specs.py --all --dry-run
```

**Expected Output:**
```
Found 6 spec files to ingest
[DRY RUN] Would ingest: openspec/specs/document-processing/spec.md
[DRY RUN] Would ingest: openspec/specs/document-processing/design.md
[DRY RUN] Would ingest: openspec/specs/clause-multi-agent/spec.md
[DRY RUN] Would ingest: openspec/specs/clause-multi-agent/design.md
[DRY RUN] Would ingest: openspec/specs/knowledge-graph/spec.md
[DRY RUN] Would ingest: openspec/specs/knowledge-graph/design.md
```

**Use Cases:**
- Verify file discovery before ingestion
- Check capability naming and file structure
- Confirm which files will be processed

---

### Custom API URL

Specify a custom API endpoint (useful for testing or different environments).

```bash
# Production server
python backend/scripts/ingest_openspec_specs.py --all --api-url http://localhost:8001

# Remote staging server
python backend/scripts/ingest_openspec_specs.py --all --api-url https://staging.example.com

# With custom port
python backend/scripts/ingest_openspec_specs.py --all --api-url http://localhost:9127
```

**Expected Output:**
```
Found 6 spec files to ingest
Ingesting: openspec/specs/document-processing/spec.md... ✓ Success
...
```

**Use Cases:**
- Testing against different API endpoints
- Deploying to staging/production environments
- Using custom port configurations

---

### Combining Options

Mix flags for advanced scenarios.

```bash
# Dry run specific capability with custom API
python backend/scripts/ingest_openspec_specs.py \
  --capability knowledge-graph \
  --dry-run \
  --api-url http://localhost:8001
```

**Expected Output:**
```
Found 2 spec files to ingest
[DRY RUN] Would ingest: openspec/specs/knowledge-graph/spec.md
[DRY RUN] Would ingest: openspec/specs/knowledge-graph/design.md
```

---

## Understanding Output

### Status Indicators

The CLI provides three status indicators for each file:

| Indicator | Meaning | HTTP Status | Description |
|-----------|---------|-------------|-------------|
| `✓ Success` | File ingested successfully | 200 | Document processed and stored in Neo4j |
| `⊘ Duplicate` | File already ingested | 409 | Content hash matches existing document |
| `✗ Failed` | Ingestion failed | 4xx/5xx | Error occurred during processing |

---

### Summary Statistics

After processing all files, the CLI displays:

```
Summary:
  ✓ Success: 4        # Successfully ingested documents
  ⊘ Duplicates: 2     # Skipped due to duplicate content hash
  ✗ Failed: 0         # Failed ingestions (errors)
```

**Interpreting Results:**

**All Success:**
```
Summary:
  ✓ Success: 6
  ⊘ Duplicates: 0
  ✗ Failed: 0
```
Perfect! All specifications ingested successfully on first run.

**Some Duplicates (Normal):**
```
Summary:
  ✓ Success: 2
  ⊘ Duplicates: 4
  ✗ Failed: 0
```
4 specs were already ingested. 2 new specs added. This is expected when re-running ingestion.

**Failures (Investigation Required):**
```
Summary:
  ✓ Success: 4
  ⊘ Duplicates: 0
  ✗ Failed: 2
```
2 specs failed to ingest. Check error messages and troubleshooting section below.

---

### Response Codes

| HTTP Code | Status | Meaning | Action |
|-----------|--------|---------|--------|
| 200 | Success | Document ingested and processed | None required |
| 409 | Duplicate | Content hash matches existing document | Expected on re-ingestion |
| 400 | Bad Request | Invalid file format or metadata | Check file content and format |
| 404 | Not Found | API endpoint not available | Verify server is running |
| 500 | Server Error | Backend processing error | Check server logs |
| 503 | Service Unavailable | Neo4j or Redis connection issue | Verify database services |

---

## Error Scenarios

### Scenario 1: API Server Not Running

**Command:**
```bash
python backend/scripts/ingest_openspec_specs.py --all
```

**Output:**
```
Found 6 spec files to ingest
Ingesting: openspec/specs/document-processing/spec.md... ✗ Failed (Connection Error)
Ingesting: openspec/specs/document-processing/design.md... ✗ Failed (Connection Error)
...

Summary:
  ✓ Success: 0
  ⊘ Duplicates: 0
  ✗ Failed: 6

Error: Connection refused - Is the API server running?
```

**Solution:**
```bash
# Start backend server
cd backend
uvicorn src.app_factory:app --host 127.0.0.1 --port 9127 --reload
```

---

### Scenario 2: Neo4j Not Running

**Command:**
```bash
python backend/scripts/ingest_openspec_specs.py --all
```

**Output:**
```
Found 6 spec files to ingest
Ingesting: openspec/specs/document-processing/spec.md... ✗ Failed (503)
...

Summary:
  ✓ Success: 0
  ⊘ Duplicates: 0
  ✗ Failed: 6

Error: Service unavailable - Check Neo4j connection
```

**Solution:**
```bash
# Check Neo4j status
neo4j status

# Start Neo4j if not running
brew services start neo4j

# Or start Neo4j desktop application
```

---

### Scenario 3: Invalid Capability Name

**Command:**
```bash
python backend/scripts/ingest_openspec_specs.py --capability invalid-capability
```

**Output:**
```
Found 0 spec files to ingest

Summary:
  ✓ Success: 0
  ⊘ Duplicates: 0
  ✗ Failed: 0

Warning: No spec files found for capability 'invalid-capability'
```

**Solution:**
```bash
# List available capabilities
ls openspec/specs/

# Expected output:
# clause-multi-agent/
# document-processing/
# knowledge-graph/

# Use correct capability name
python backend/scripts/ingest_openspec_specs.py --capability document-processing
```

---

### Scenario 4: Missing Command-Line Flag

**Command:**
```bash
python backend/scripts/ingest_openspec_specs.py
```

**Output:**
```
Error: Specify --all or --capability <name>

Usage:
  python backend/scripts/ingest_openspec_specs.py --all
  python backend/scripts/ingest_openspec_specs.py --capability <name>
  python backend/scripts/ingest_openspec_specs.py --help
```

**Solution:**
Add required flag:
```bash
# Ingest all
python backend/scripts/ingest_openspec_specs.py --all

# Or specify capability
python backend/scripts/ingest_openspec_specs.py --capability document-processing
```

---

### Scenario 5: Duplicate Content (Expected Behavior)

**Command:**
```bash
# First run: Success
python backend/scripts/ingest_openspec_specs.py --all

# Second run: Duplicates detected
python backend/scripts/ingest_openspec_specs.py --all
```

**Output (First Run):**
```
Found 6 spec files to ingest
Ingesting: openspec/specs/document-processing/spec.md... ✓ Success
...

Summary:
  ✓ Success: 6
  ⊘ Duplicates: 0
  ✗ Failed: 0
```

**Output (Second Run):**
```
Found 6 spec files to ingest
Ingesting: openspec/specs/document-processing/spec.md... ⊘ Duplicate (already ingested)
...

Summary:
  ✓ Success: 0
  ⊘ Duplicates: 6
  ✗ Failed: 0
```

**Explanation:**
This is expected behavior. The CLI uses SHA-256 content hashing to detect duplicates and prevent re-ingesting unchanged files. The API returns HTTP 409 when content hash matches an existing document.

**When to Re-Ingest:**
- After modifying a spec file (content hash will change)
- After deleting documents from Neo4j
- After schema changes requiring re-processing

---

### Scenario 6: Large Spec File

**Command:**
```bash
python backend/scripts/ingest_openspec_specs.py --capability large-feature
```

**Output:**
```
Found 2 spec files to ingest
Ingesting: openspec/specs/large-feature/spec.md... ✓ Success (processing time: 12.4s)
Ingesting: openspec/specs/large-feature/design.md... ✓ Success (processing time: 8.2s)

Summary:
  ✓ Success: 2
  ⊘ Duplicates: 0
  ✗ Failed: 0
```

**Note:** Large files (1000+ lines) may take longer to process due to:
- Consciousness analysis (5-level concepts)
- ASI-GO-2 research integration
- ThoughtSeed generation
- Vector embedding computation

**Expected Processing Times:**
- Small spec (100 lines): 1-3 seconds
- Medium spec (500 lines): 3-8 seconds
- Large spec (1000+ lines): 8-15 seconds

---

## Troubleshooting

### Check API Server Health

```bash
curl http://localhost:9127/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "services": {
    "neo4j": "connected",
    "redis": "connected"
  }
}
```

---

### Verify Neo4j Connection

```bash
# Check Neo4j service
neo4j status

# Test connection
curl http://localhost:7474/

# Check environment variables
echo $NEO4J_URI
echo $NEO4J_USER
```

---

### View Ingested Documents

Query Neo4j to verify ingestion:

```cypher
// Count all specification documents
MATCH (s:Specification)
RETURN count(s) as total_specs

// List all capabilities
MATCH (s:Specification)
RETURN DISTINCT s.capability, count(*) as spec_count
ORDER BY s.capability

// View specific capability
MATCH (s:Specification {capability: "document-processing"})
RETURN s.title, s.spec_type, s.content_hash, s.created_at

// Check for duplicates
MATCH (s:Specification)
WITH s.content_hash as hash, count(*) as count
WHERE count > 1
RETURN hash, count
```

---

### Enable Debug Logging

Add verbose output to the script for debugging:

```bash
# Export debug environment variable
export DEBUG=1

# Run with Python debug mode
python -u backend/scripts/ingest_openspec_specs.py --all

# Or capture output to file
python backend/scripts/ingest_openspec_specs.py --all 2>&1 | tee ingestion.log
```

---

### Check Backend Logs

Monitor backend logs while ingesting:

```bash
# In one terminal: Start backend with logs
cd backend
uvicorn src.app_factory:app --host 127.0.0.1 --port 9127 --reload --log-level debug

# In another terminal: Run ingestion
python scripts/ingest_openspec_specs.py --all
```

---

### Common Issues and Solutions

| Issue | Symptom | Solution |
|-------|---------|----------|
| Connection refused | `✗ Failed (Connection Error)` | Start backend server |
| Neo4j not running | `✗ Failed (503)` | Start Neo4j service |
| Redis not running | `✗ Failed (500)` | Start Redis service |
| Wrong port | Connection timeout | Check `--api-url` matches server port |
| Invalid capability | 0 files found | Check `openspec/specs/` directory names |
| File not found | FileNotFoundError | Run from project root directory |
| Permission denied | PermissionError | Check file permissions on `openspec/specs/` |

---

## Integration Workflow

### Full OpenSpec → Archon → Neo4j Workflow

```bash
# 1. Create OpenSpec change proposal
/openspec:proposal add-new-feature

# 2. Import to Archon for task management
/openspec:import-to-archon add-new-feature

# 3. Work on tasks in Archon
# (Task completion updates via daemon or manual sync)

# 4. Once complete, ingest specs to Neo4j
python backend/scripts/ingest_openspec_specs.py --capability add-new-feature

# 5. Verify ingestion
curl http://localhost:9127/api/query \
  -d '{"query": "new feature specifications"}'

# 6. Archive change
/openspec:archive add-new-feature
```

---

### Continuous Integration Workflow

```bash
# In CI/CD pipeline (e.g., GitHub Actions)

# 1. Start services
docker-compose up -d neo4j redis

# 2. Start backend
cd backend && uvicorn src.app_factory:app --host 0.0.0.0 --port 9127 &

# 3. Wait for services to be ready
./scripts/wait-for-it.sh localhost:9127 -t 60

# 4. Ingest all specs
python backend/scripts/ingest_openspec_specs.py --all

# 5. Verify ingestion
python -c "
import requests
response = requests.get('http://localhost:9127/api/documents?source_type=openspec')
assert response.status_code == 200
specs = response.json()
assert len(specs) >= 6
print(f'✓ Verified {len(specs)} specs ingested')
"
```

---

## Reference

### CLI Help

```bash
python backend/scripts/ingest_openspec_specs.py --help
```

**Output:**
```
usage: ingest_openspec_specs.py [-h] [--capability CAPABILITY] [--all]
                                [--dry-run] [--api-url API_URL]

Ingest OpenSpec specifications into Neo4j knowledge graph

optional arguments:
  -h, --help            show this help message and exit
  --capability CAPABILITY
                        Ingest specific capability (e.g., document-processing)
  --all                 Ingest all capabilities
  --dry-run             Preview files without ingesting
  --api-url API_URL     API base URL (default: http://localhost:9127)
```

---

### Available Capabilities

Current capabilities in `openspec/specs/`:

```
document-processing/
├── spec.md          # Document processing requirements
└── design.md        # LangGraph implementation patterns

clause-multi-agent/
├── spec.md          # Multi-agent coordination requirements
└── design.md        # SubgraphArchitect, CuratedPathNavigator patterns

knowledge-graph/
├── spec.md          # Neo4j knowledge graph requirements
└── design.md        # Graph schema and query patterns
```

---

### Metadata Schema

Each ingested document includes:

```json
{
  "source_type": "openspec",
  "capability": "document-processing",
  "spec_type": "spec",
  "content_hash": "abc123...",
  "version": "1.0",
  "filename": "document-processing-spec.md",
  "extracted_text": "...",
  "summary": "...",
  "created_at": "2025-11-17T10:30:00Z"
}
```

---

## See Also

- [OpenSpec Workflow Guide](../openspec/AGENTS.md)
- [OpenSpec + Archon Integration Examples](../docs/OPENSPEC_ARCHON_EXAMPLES.md)
- [Automated Sync Guide](../docs/AUTOMATED_SYNC_GUIDE.md)
- [Backend README](../backend/README.md)
- [CLAUDE.md](../CLAUDE.md) - Project instructions
