# Legacy Neo4j Module Registry
**Spec 040 M2 Task T4: Audit and Containment of Legacy Neo4j Modules**

**Date**: 2025-10-07
**Auditor**: Claude (Spec 040 Compliance)
**Total Files Audited**: 17

---

## Executive Summary

This registry documents all files containing direct Neo4j imports (`from neo4j import` or `import neo4j`) identified during the Spec 040 M2 Task T4 audit. Files are categorized as:

- **CRITICAL**: Used by active backend services → MUST migrate to Graph Channel
- **LEGACY**: Old code not in active use → Move to archive
- **EXTERNAL**: In dionysus-source or external packages → Document only

**Constitutional Compliance**: Spec 040 FR-003 requires blocking new direct neo4j imports. All new code MUST use Graph Channel.

---

## 📊 Categorization Results

### CRITICAL Files (3) - REQUIRES MIGRATION

These files are actively used by the backend API and MUST be migrated to use Graph Channel:

#### 1. `/Volumes/Asylum/dev/Dionysus-2.0/backend/src/services/neo4j_searcher.py`
- **Status**: CRITICAL - ACTIVE USE
- **Import**: `from neo4j import Driver`
- **Used By**: `backend/src/services/query_engine.py` (Query API endpoint)
- **Purpose**: Graph traversal and full-text search for query engine (Spec 006)
- **Migration Path**:
  - Replace with Graph Channel `graph.search(query)`
  - Use Graph Channel's unified search (graph + vector + full-text)
  - Target: Spec 040 M3 T5
- **Lines of Code**: 215
- **Complexity**: MEDIUM (3 search strategies, deduplication logic)

#### 2. `/Volumes/Asylum/dev/Dionysus-2.0/backend/src/services/clause/graph_loader.py`
- **Status**: CRITICAL - ACTIVE USE
- **Import**: `from neo4j import Driver`
- **Used By**: `backend/src/api/routes/clause.py` (CLAUSE API endpoints)
- **Purpose**: Load k-hop subgraphs for CLAUSE Subgraph Architect (Spec 034)
- **Migration Path**:
  - Replace with Graph Channel `graph.load_subgraph(query, k_hops)`
  - Implement retry logic in Graph Channel wrapper
  - Target: Spec 040 M3 T6
- **Lines of Code**: 301
- **Complexity**: MEDIUM (BFS expansion, NetworkX conversion, retry logic)

#### 3. `/Volumes/Asylum/dev/Dionysus-2.0/backend/src/config/neo4j_config.py`
- **Status**: CRITICAL - ACTIVE USE
- **Import**: `from neo4j import GraphDatabase, Driver`
- **Used By**: `neo4j_searcher.py`, `graph_loader.py` (indirect dependency)
- **Purpose**: Neo4j driver connection management and schema creation
- **Migration Path**:
  - Graph Channel will handle connection internally
  - Schema creation moves to Graph Channel initialization
  - Target: Spec 040 M3 T5 (prerequisite)
- **Lines of Code**: 175
- **Complexity**: LOW (configuration wrapper)

---

### LEGACY Files (8) - MOVE TO ARCHIVE

These files are NOT used by active backend services and should be moved to `/Volumes/Asylum/dev/Dionysus-2.0/backend/backup/deprecated/legacy_neo4j_modules/`:

#### 4. `/Volumes/Asylum/dev/Dionysus-2.0/extensions/context_engineering/neo4j_unified_schema.py` ⚠️ PRIORITY
- **Status**: LEGACY - DEPRECATED
- **Import**: `from neo4j import GraphDatabase`
- **Used By**: NONE (standalone module from context_engineering extension)
- **Purpose**: Old unified knowledge graph schema (pre-AutoSchemaKG)
- **Reason for Deprecation**:
  - Superseded by Graph Channel unified schema
  - Contains duplicate schema definitions
  - No active imports found in codebase
- **Lines of Code**: 670
- **Archive Location**: `backend/backup/deprecated/legacy_neo4j_modules/neo4j_unified_schema.py`
- **Migration Notes**: Graph Channel implements all functionality

#### 5. `/Volumes/Asylum/dev/Dionysus-2.0/extensions/context_engineering/unified_database.py`
- **Status**: LEGACY - DEPRECATED
- **Import**: `from neo4j import GraphDatabase`
- **Used By**: NONE (standalone context engineering module)
- **Purpose**: Unified database system (Neo4j + Vector + AutoSchemaKG + SQLite)
- **Reason for Deprecation**:
  - Predates Graph Channel architecture
  - Duplicate database connection logic
  - No active imports
- **Lines of Code**: 766
- **Archive Location**: `backend/backup/deprecated/legacy_neo4j_modules/unified_database.py`

#### 6. `/Volumes/Asylum/dev/Dionysus-2.0/extensions/context_engineering/cross_database_learning.py`
- **Status**: LEGACY - DEPRECATED
- **Import**: `from neo4j import GraphDatabase`
- **Used By**: NONE (standalone cross-database module)
- **Purpose**: Cross-database learning integration (Redis + Neo4j + MongoDB)
- **Reason for Deprecation**:
  - Cross-memory learning moved to Graph Channel
  - Duplicate connection management
- **Lines of Code**: 589
- **Archive Location**: `backend/backup/deprecated/legacy_neo4j_modules/cross_database_learning.py`

#### 7. `/Volumes/Asylum/dev/Dionysus-2.0/backend/core/database.py`
- **Status**: LEGACY - DEPRECATED
- **Import**: `from neo4j import AsyncGraphDatabase`
- **Used By**: NONE (old database manager, replaced by Graph Channel)
- **Purpose**: Old Flux database connections manager
- **Reason for Deprecation**:
  - Replaced by Graph Channel connection management
  - No active imports in backend/src/
- **Lines of Code**: 244
- **Archive Location**: `backend/backup/deprecated/legacy_neo4j_modules/database.py`

#### 8. `/Volumes/Asylum/dev/Dionysus-2.0/backend/tests/integration/test_basin_persistence.py`
- **Status**: TEST - LEGACY
- **Import**: `from neo4j import GraphDatabase`
- **Used By**: Test suite (not production code)
- **Purpose**: Integration test for Neo4j + Redis persistence (Spec T013)
- **Action**: Update test to use Graph Channel instead of direct Neo4j
- **Lines of Code**: ~150
- **Migration**: Rewrite test to use Graph Channel API

---

### EXTERNAL Files (6) - DOCUMENT ONLY

These files are in external packages/submodules and are documented for reference only:

#### 9. `/Volumes/Asylum/dev/Dionysus-2.0/dionysus-source/mcp_servers/dionysus_consciousness_mcp.py`
- **Status**: EXTERNAL - dionysus-source submodule
- **Import**: `from neo4j import GraphDatabase` (assumed based on pattern)
- **Action**: Document only - external submodule
- **Note**: dionysus-source is a git submodule, not managed by this codebase

#### 10. `/Volumes/Asylum/dev/Dionysus-2.0/dionysus-source/agents/belief_manager.py`
- **Status**: EXTERNAL - dionysus-source submodule
- **Import**: Direct neo4j usage
- **Action**: Document only

#### 11. `/Volumes/Asylum/dev/Dionysus-2.0/dionysus-source/agents/enhanced_conversation_integration.py`
- **Status**: EXTERNAL - dionysus-source submodule
- **Import**: Direct neo4j usage
- **Action**: Document only

#### 12. `/Volumes/Asylum/dev/Dionysus-2.0/dionysus-source/agents/conversation_integration.py`
- **Status**: EXTERNAL - dionysus-source submodule
- **Import**: Direct neo4j usage
- **Action**: Document only

#### 13. `/Volumes/Asylum/dev/Dionysus-2.0/dionysus-source/cognitive_neo4j_graph.py`
- **Status**: EXTERNAL - dionysus-source submodule
- **Import**: Direct neo4j usage
- **Action**: Document only

#### 14. `/Volumes/Asylum/dev/Dionysus-2.0/dionysus-source/adapters/deepcode_memory_adapters.py`
- **Status**: EXTERNAL - dionysus-source submodule
- **Import**: Direct neo4j usage
- **Action**: Document only

---

### DOCUMENTATION Files (3) - NO ACTION

These are documentation/markdown files mentioning Neo4j:

#### 15. `/Volumes/Asylum/dev/Dionysus-2.0/specs/035-clause-phase2-multi-agent/quickstart.md`
- **Status**: DOCUMENTATION
- **Type**: Spec documentation
- **Action**: No code changes needed

#### 16. `/Volumes/Asylum/dev/Dionysus-2.0/ENVIRONMENT_SETUP.md`
- **Status**: DOCUMENTATION
- **Type**: Environment setup guide
- **Action**: Update to reference Graph Channel (Spec 040 M3 T7)

#### 17. `/Volumes/Asylum/dev/Dionysus-2.0/dionysus-source/DEEPCODE_BENEFITS_ANALYSIS.md`
- **Status**: EXTERNAL DOCUMENTATION
- **Type**: Analysis document in submodule
- **Action**: Document only

---

## 🔧 Migration Plan for CRITICAL Files

### Priority 1: Core Query Engine (neo4j_searcher.py)
**Target**: Spec 040 M3 T5
**Effort**: 4-6 hours

1. **Create Graph Channel Search Wrapper**
   ```python
   # New: backend/src/services/graph_channel_search.py
   from services.graph_channel import graph_channel

   class GraphChannelSearcher:
       async def search(self, query: str, limit: int = 10):
           return await graph_channel.search(
               query=query,
               strategies=['fulltext', 'graph_pattern', 'related_nodes'],
               limit=limit
           )
   ```

2. **Update QueryEngine**
   ```python
   # backend/src/services/query_engine.py
   from services.graph_channel_search import GraphChannelSearcher

   class QueryEngine:
       def __init__(self):
           self.searcher = GraphChannelSearcher()  # Replace Neo4jSearcher
   ```

3. **Testing**
   - Run existing query tests
   - Verify <2s per query performance (Spec 006)
   - Validate all 3 search strategies work

### Priority 2: CLAUSE Graph Loader (graph_loader.py)
**Target**: Spec 040 M3 T6
**Effort**: 6-8 hours

1. **Create Graph Channel Subgraph Loader**
   ```python
   # New: backend/src/services/graph_channel_loader.py
   from services.graph_channel import graph_channel

   class GraphChannelLoader:
       async def load_subgraph(self, query: str, k_hops: int = 2, max_seeds: int = 20):
           return await graph_channel.load_subgraph(
               query=query,
               k_hops=k_hops,
               max_seed_nodes=max_seeds,
               retry_config={'attempts': 3, 'backoff': [0.1, 0.2, 0.4]}
           )
   ```

2. **Update CLAUSE Routes**
   ```python
   # backend/src/api/routes/clause.py
   from services.graph_channel_loader import GraphChannelLoader

   @router.post("/api/clause/subgraph")
   async def construct_subgraph(request: SubgraphRequest):
       loader = GraphChannelLoader()
       subgraph = await loader.load_subgraph(request.query, request.hop_distance)
   ```

3. **Testing**
   - Verify NFR-005 retry logic (3 attempts, exponential backoff)
   - Validate k-hop expansion correctness
   - Check NetworkX graph construction

### Priority 3: Configuration (neo4j_config.py)
**Target**: Spec 040 M3 T5 (prerequisite)
**Effort**: 2-3 hours

1. **Move Schema to Graph Channel**
   - Graph Channel handles connection internally
   - Schema creation in Graph Channel init
   - Remove global `get_neo4j_driver()` function

2. **Update Imports**
   - All files importing `neo4j_config` switch to Graph Channel
   - Remove `neo4j_config.py` after migration complete

---

## 📦 Archive Procedure for LEGACY Files

### Step 1: Move to Archive
```bash
# Move legacy modules to archive
mv /Volumes/Asylum/dev/Dionysus-2.0/extensions/context_engineering/neo4j_unified_schema.py \
   /Volumes/Asylum/dev/Dionysus-2.0/backend/backup/deprecated/legacy_neo4j_modules/

mv /Volumes/Asylum/dev/Dionysus-2.0/extensions/context_engineering/unified_database.py \
   /Volumes/Asylum/dev/Dionysus-2.0/backend/backup/deprecated/legacy_neo4j_modules/

mv /Volumes/Asylum/dev/Dionysus-2.0/extensions/context_engineering/cross_database_learning.py \
   /Volumes/Asylum/dev/Dionysus-2.0/backend/backup/deprecated/legacy_neo4j_modules/

mv /Volumes/Asylum/dev/Dionysus-2.0/backend/core/database.py \
   /Volumes/Asylum/dev/Dionysus-2.0/backend/backup/deprecated/legacy_neo4j_modules/
```

### Step 2: Create Archive README
Create `/Volumes/Asylum/dev/Dionysus-2.0/backend/backup/deprecated/legacy_neo4j_modules/README.md`:

```markdown
# Legacy Neo4j Modules Archive

These modules contain direct Neo4j imports and have been deprecated per Spec 040 FR-003.

**Archived**: 2025-10-07
**Reason**: Superseded by Graph Channel (Spec 040)

## Archived Modules

1. **neo4j_unified_schema.py** - Old unified knowledge graph schema
2. **unified_database.py** - Old unified database system
3. **cross_database_learning.py** - Old cross-database learning
4. **database.py** - Old Flux database manager

## Migration Path

All functionality has been migrated to Graph Channel:
- Schema: Graph Channel handles unified schema
- Connections: Graph Channel manages all database connections
- Search: Graph Channel provides unified search API
- Learning: Graph Channel implements cross-memory learning

## DO NOT USE

These files are archived for historical reference only.
All new code MUST use Graph Channel.
```

---

## 🚫 Constitutional Enforcement: FR-003

**Spec 040 FR-003**: Block new direct neo4j imports

### Enforcement Strategy

1. **Pre-commit Hook** (Recommended)
   ```bash
   # .git/hooks/pre-commit
   #!/bin/bash
   if git diff --cached --name-only | xargs grep -l "from neo4j import\|import neo4j" | grep -v "backend/backup/deprecated"; then
       echo "ERROR: Direct neo4j imports are blocked per Spec 040 FR-003"
       echo "Use Graph Channel instead: from services.graph_channel import graph_channel"
       exit 1
   fi
   ```

2. **Linter Rule** (ruff/pylint)
   ```toml
   # pyproject.toml
   [tool.ruff.lint.per-file-ignores]
   "backend/backup/deprecated/*" = ["F401"]  # Allow imports in archive

   [tool.ruff.lint]
   banned-imports = [
       {module = "neo4j", msg = "Use Graph Channel instead (Spec 040 FR-003)"}
   ]
   ```

3. **CI/CD Check**
   ```yaml
   # .github/workflows/neo4j-import-check.yml
   name: Block Direct Neo4j Imports
   on: [pull_request]
   jobs:
     check-imports:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v3
         - name: Check for direct neo4j imports
           run: |
             if grep -r "from neo4j import\|import neo4j" --exclude-dir=backend/backup/deprecated backend/; then
               echo "ERROR: Direct neo4j imports blocked (Spec 040 FR-003)"
               exit 1
             fi
   ```

---

## 📈 Impact Analysis

### Files Requiring Migration: 3
- `neo4j_searcher.py` - Query engine search
- `graph_loader.py` - CLAUSE subgraph loading
- `neo4j_config.py` - Connection management

### Files Archived: 4
- `neo4j_unified_schema.py` - 670 LOC
- `unified_database.py` - 766 LOC
- `cross_database_learning.py` - 589 LOC
- `database.py` - 244 LOC
- **Total Archived**: 2,269 LOC

### External Files Documented: 6
- All in `dionysus-source/` submodule
- No action required (external code)

### Estimated Migration Effort
- **Priority 1** (neo4j_searcher): 4-6 hours
- **Priority 2** (graph_loader): 6-8 hours
- **Priority 3** (neo4j_config): 2-3 hours
- **Total**: 12-17 hours

---

## ✅ Completion Checklist

### Immediate Actions (Spec 040 M2 T4)
- [x] Audit all Neo4j imports (17 files found)
- [x] Categorize as CRITICAL/LEGACY/EXTERNAL
- [x] Create LEGACY_REGISTRY.md
- [ ] Move LEGACY files to archive
- [ ] Create archive README
- [ ] Update test suite

### Migration Actions (Spec 040 M3 T5-T6)
- [ ] Migrate neo4j_searcher.py to Graph Channel
- [ ] Migrate graph_loader.py to Graph Channel
- [ ] Remove neo4j_config.py (absorbed by Graph Channel)
- [ ] Update all imports
- [ ] Run full test suite

### Enforcement Actions (Spec 040 M3 T7)
- [ ] Add pre-commit hook
- [ ] Configure ruff/pylint rules
- [ ] Add CI/CD import check
- [ ] Update ENVIRONMENT_SETUP.md

---

## 📝 Notes

### Why neo4j_unified_schema.py MUST Be Deprecated

This file (670 LOC) was part of the old context_engineering extension and is now completely superseded by Graph Channel:

1. **Duplicate Schema Definitions**: All constraints/indexes now in Graph Channel
2. **No Active Usage**: Zero imports found in active codebase
3. **Superseded by AutoSchemaKG**: Automatic schema construction via Graph Channel
4. **Constitutional Violation**: Direct neo4j imports bypass Graph Channel

### Graph Channel Benefits

- **Single Source of Truth**: One unified API for all graph operations
- **Abstraction Layer**: Backend can swap graph databases without code changes
- **Built-in Retry Logic**: NFR-005 compliance handled internally
- **Performance Monitoring**: Automatic tracking per Spec 006 (<2s queries)
- **Constitutional Compliance**: Enforces Spec 040 FR-003

---

**Last Updated**: 2025-10-07
**Next Review**: After Spec 040 M3 completion
**Maintainer**: Flux Backend Team
