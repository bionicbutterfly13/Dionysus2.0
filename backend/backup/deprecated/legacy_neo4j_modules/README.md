# Legacy Neo4j Modules Archive

**Archived**: 2025-10-07
**Spec**: 040 M2 Task T4 - Legacy Module Containment
**Reason**: Superseded by Graph Channel (Spec 040)

---

## ⚠️ DEPRECATED - DO NOT USE

These modules contain direct Neo4j imports and have been deprecated per **Spec 040 FR-003: Block new direct neo4j imports**.

All functionality has been migrated to **Graph Channel**.

---

## 📦 Archived Modules

### 1. neo4j_unified_schema.py (670 LOC)
- **Original Location**: `extensions/context_engineering/`
- **Purpose**: Old unified knowledge graph schema for ASI-Arch Context Flow
- **Superseded By**: Graph Channel unified schema
- **Reason for Deprecation**:
  - Duplicate schema definitions (Graph Channel owns schema)
  - No active imports in codebase
  - Predates AutoSchemaKG integration
  - Direct neo4j imports bypass Graph Channel

### 2. unified_database.py (766 LOC)
- **Original Location**: `extensions/context_engineering/`
- **Purpose**: Unified database system (Neo4j + Vector + AutoSchemaKG + SQLite)
- **Superseded By**: Graph Channel database management
- **Reason for Deprecation**:
  - Duplicate database connection logic
  - Cross-database learning moved to Graph Channel
  - No active usage in backend services

### 3. cross_database_learning.py (589 LOC)
- **Original Location**: `extensions/context_engineering/`
- **Purpose**: Cross-database learning integration (Redis + Neo4j + MongoDB + PostgreSQL)
- **Superseded By**: Graph Channel cross-memory learning
- **Reason for Deprecation**:
  - Cross-memory patterns now in Graph Channel
  - Episodic/semantic/procedural analysis consolidated
  - Direct database connections replaced by Graph Channel abstraction

### 4. database.py (244 LOC)
- **Original Location**: `backend/core/`
- **Purpose**: Old Flux database connections manager (Neo4j + Qdrant + Redis + SQLite)
- **Superseded By**: Graph Channel connection management
- **Reason for Deprecation**:
  - Replaced by Graph Channel unified connections
  - No active imports in backend/src/
  - Async connection logic moved to Graph Channel

---

## 🔄 Migration Path

### What Replaced These Modules

**Graph Channel** (`backend/src/services/graph_channel.py`) now provides:

1. **Unified Schema** (replaces neo4j_unified_schema.py)
   ```python
   from services.graph_channel import graph_channel
   # Schema managed internally by Graph Channel
   ```

2. **Connection Management** (replaces database.py)
   ```python
   from services.graph_channel import graph_channel
   # Connections handled internally
   ```

3. **Cross-Memory Learning** (replaces cross_database_learning.py)
   ```python
   from services.graph_channel import graph_channel
   result = await graph_channel.cross_memory_search(
       episodic=True, semantic=True, procedural=True
   )
   ```

4. **Unified Database Operations** (replaces unified_database.py)
   ```python
   from services.graph_channel import graph_channel
   await graph_channel.store_architecture(arch_data)
   similar = await graph_channel.find_similar(query)
   ```

### Migration Examples

**OLD (deprecated):**
```python
from neo4j import GraphDatabase
from extensions.context_engineering.neo4j_unified_schema import Neo4jUnifiedSchema

schema = Neo4jUnifiedSchema(uri, user, password)
schema.connect()
schema.create_architecture_node(arch_data)
```

**NEW (Graph Channel):**
```python
from services.graph_channel import graph_channel

await graph_channel.store_architecture(arch_data)
# Schema, connections, all handled internally
```

---

## 🚫 Why These Were Archived

### Constitutional Violation: Spec 040 FR-003
Direct neo4j imports bypass the Graph Channel abstraction layer, violating:
- **Single Source of Truth**: Multiple database connection paths
- **Abstraction**: Backend code coupled to Neo4j implementation
- **Maintainability**: Changes require updates in multiple files

### Technical Debt
- **Duplicate Code**: 2,269 LOC performing same functions as Graph Channel
- **No Active Usage**: Zero imports found in active backend services
- **Testing Burden**: Separate test suites for duplicate functionality

### Architecture Evolution
- **AutoSchemaKG Integration**: Automatic schema construction via Graph Channel
- **Unified API**: Single API for graph, vector, and full-text operations
- **Performance Monitoring**: Built-in query performance tracking

---

## 📊 Files That DID NOT Get Archived

### CRITICAL Files (Still Active - Migration Pending)
These files still use direct Neo4j imports but are actively used:

1. **backend/src/services/neo4j_searcher.py** - Query engine search
   - Migration: Spec 040 M3 T5
2. **backend/src/services/clause/graph_loader.py** - CLAUSE subgraph loading
   - Migration: Spec 040 M3 T6
3. **backend/src/config/neo4j_config.py** - Connection config
   - Migration: Spec 040 M3 T5 (prerequisite)

See `LEGACY_REGISTRY.md` for migration plan.

---

## 🔒 Enforcement: Prevent New Direct Imports

**Pre-commit Hook** (blocks commits with direct neo4j imports):
```bash
#!/bin/bash
if git diff --cached --name-only | xargs grep -l "from neo4j import\|import neo4j" | grep -v "backend/backup/deprecated"; then
    echo "ERROR: Direct neo4j imports are blocked per Spec 040 FR-003"
    echo "Use Graph Channel instead: from services.graph_channel import graph_channel"
    exit 1
fi
```

**Linter Configuration** (ruff):
```toml
[tool.ruff.lint]
banned-imports = [
    {module = "neo4j", msg = "Use Graph Channel instead (Spec 040 FR-003)"}
]
```

---

## 📝 Archive Metadata

- **Total LOC Archived**: 2,269
- **Files Archived**: 4
- **Archive Date**: 2025-10-07
- **Archived By**: Spec 040 M2 Task T4 Audit
- **Restoration Policy**: These files should NOT be restored. Use Graph Channel instead.

---

## ❓ Questions?

**Q: Can I use these files for reference?**
A: Yes, for historical reference only. DO NOT copy code from here.

**Q: What if I need Neo4j-specific functionality?**
A: Use Graph Channel. If functionality missing, extend Graph Channel, don't bypass it.

**Q: Why not delete these files?**
A: Archived for:
- Historical reference
- Understanding evolution of architecture
- Potential data migration scripts (one-time use only)

**Q: I found a bug in archived code, should I fix it?**
A: No. These files are deprecated. If the bug affects Graph Channel, fix it there.

---

**Last Updated**: 2025-10-07
**Maintainer**: Flux Backend Team
**Related**: LEGACY_REGISTRY.md, Spec 040 Migration Plan
