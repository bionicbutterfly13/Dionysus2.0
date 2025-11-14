# Dionysus Consciousness Systems - Setup & Usage

**Created**: 2025-11-13
**Status**: Implementation Complete - Ready for Initialization

## Overview

This document describes the consciousness-enhanced knowledge management systems integrated into Dionysus 2.0:

1. **Skills Database**: Process skills through consciousness system
2. **BMAD Migration**: Migrate legacy decision data to consciousness processing
3. **Startup Status Check**: Automated system health reporting

All systems process data through the Dionysus consciousness pipeline:
```
Data → Daedalus → DocumentProcessingGraph → AutoSchemaKG → Neo4j
         ↓              ↓                        ↓
    Gateway    LangGraph Workflow    Knowledge Graph Storage
                    ↓
         5-Level Concepts
         Attractor Basins
         ThoughtSeeds
```

---

## System 1: Skills Database

### Purpose
Process your accumulated skills through the consciousness system to create:
- 5-level concept hierarchy from each skill
- Attractor basins for skill domains
- ThoughtSeeds linking related skills
- Searchable knowledge graph of capabilities

### Location
- Skills library: `/Volumes/Asylum/skills-library/`
- Service: `backend/src/services/skills_manager.py`
- Initialization: `backend/initialize_skills.py`
- Index: `~/.claude/skills/index.json`

### Structure
```
/Volumes/Asylum/skills-library/
├── official/           # Official skills
├── community/          # Community-contributed skills
├── personal/
│   └── superpowers/   # Personal skills collection
└── project-specific/  # Project-specific skills
```

### Usage

**Check Status**:
```bash
python backend/check_consciousness_systems.py
```

**Initialize Skills Database**:
```bash
python backend/initialize_skills.py
```

This will:
1. Scan all skill categories
2. Process each skill through Daedalus consciousness pipeline
3. Extract concepts, create basins, generate thoughtseeds
4. Store in Neo4j knowledge graph
5. Create Claude Code skill index

**Example Output**:
```
🚀 Dionysus Skills Database Initialization
============================================================
Skills discovered: 45
Skills processed: 45
Categories scanned: 4

Stats:
  • Successful: 42
  • Failed: 3
  • Total concepts extracted: 324
  • Average quality score: 0.82
```

---

## System 2: BMAD Migration

### Purpose
Migrate legacy BMAD decision tracking data from simple Neo4j nodes to consciousness-processed knowledge with full relational context.

### What Gets Migrated
- **Project Node** (ID: 5): "Info Product Architect" project metadata
- **Decision Node** (ID: 6): Decision tracking system architectural decision

### Why Migrate
The original BMAD nodes are simple property storage. Migration creates:
- Rich conceptual understanding of the decision
- Attractor basins for decision-making patterns
- ThoughtSeeds connecting to related architectural decisions
- Full-text search capabilities
- Relationship mapping to other knowledge

### Location
- Migration script: `backend/migrate_bmad_to_consciousness.py`

### Usage

**Dry Run (Preview)**:
```bash
python backend/migrate_bmad_to_consciousness.py --dry-run
```

**Execute Migration**:
```bash
python backend/migrate_bmad_to_consciousness.py
```

**Keep Original Nodes** (for comparison):
```bash
python backend/migrate_bmad_to_consciousness.py --keep-original
```

**Example Output**:
```
🚀 BMAD to Consciousness Migration
============================================================
✅ Extracted data from Project and Decision nodes
✅ Created migration document (3456 bytes)
🧠 Processing through consciousness system...
  ✓ Processing complete
    - Concepts extracted: 23
    - Basins created: 5
    - ThoughtSeeds: 12
    - Quality score: 0.87
🗑️  Removing original BMAD nodes...
  ✓ Removed 2 nodes

✅ Migration Complete!
```

---

## System 3: Startup Status Check

### Purpose
Automatically check consciousness systems health at session startup so you immediately know what needs attention.

### Location
- Status check: `backend/check_consciousness_systems.py`

### Integration with CLAUDE.md
This check will be added to the startup protocol in `CLAUDE.md` so every session begins with a status report.

### Usage

**Manual Check**:
```bash
python backend/check_consciousness_systems.py
```

**Example Output**:
```
============================================================
🧠 Dionysus Consciousness Systems Status
============================================================

📚 Skills Database:
  ✅ Initialized with 45 skills
     Categories: official, community, personal/superpowers, project-specific
     Success rate: 42/45
     Average quality: 0.82

🔄 BMAD Migration:
  ✅ BMAD data successfully migrated to consciousness system

🗄️  Neo4j Knowledge Graph:
  ✅ Healthy (152 total nodes)
     Consciousness nodes:
       • Document: 47
       • Concept: 68
       • AttractorBasin: 15
       • ThoughtSeed: 22

============================================================
```

### Automatic Startup Integration

Add to your session startup routine:
```bash
# In .bashrc, .zshrc, or equivalent
alias dionysus-status="python /Volumes/Asylum/dev/Dionysus-2.0/backend/check_consciousness_systems.py"
```

Or integrate directly into CLAUDE.md Phase 3 startup protocol.

---

## Implementation Details

### Skills Manager (`skills_manager.py`)
- **Class**: `SkillsManager`
- **Methods**:
  - `initialize_skills_database()` - Full initialization
  - `get_skill_status()` - Current status
  - `_scan_category()` - Scan skill directory
  - `_process_skill_through_consciousness()` - Process via Daedalus
  - `_create_claude_index()` - Create skill index

### BMAD Migrator (`migrate_bmad_to_consciousness.py`)
- **Class**: `BMADMigrator`
- **Methods**:
  - `extract_bmad_data()` - Get data from Neo4j
  - `create_migration_document()` - Format for processing
  - `process_through_consciousness()` - Process via Daedalus
  - `remove_original_nodes()` - Clean up old nodes
  - `migrate()` - Execute full migration

### Status Checker (`check_consciousness_systems.py`)
- **Functions**:
  - `check_skills_database()` - Skills status
  - `check_bmad_migration()` - Migration status
  - `check_neo4j_health()` - Neo4j status
  - `print_status_report()` - Comprehensive report

---

## Neo4j Schema Impact

### Before Initialization
```
Nodes: 9 (mostly Sessions, Agents, Preferences)
Consciousness Nodes: 0
```

### After Skills Initialization (Estimated)
```
Nodes: ~200+
- Document: ~45 (one per skill)
- Concept: ~300+ (extracted from skills)
- AttractorBasin: ~20+ (skill domains)
- ThoughtSeed: ~50+ (cross-skill relationships)
```

### After BMAD Migration
```
Additional Nodes:
- Document: +1 (migrated decision document)
- Concept: +23 (decision-related concepts)
- AttractorBasin: +5 (decision-making domains)
- ThoughtSeed: +12 (decision relationships)

Removed Nodes:
- Project: -1
- Decision: -1
```

---

## Next Steps

1. **Initialize Skills** (if not done):
   ```bash
   python backend/initialize_skills.py
   ```

2. **Migrate BMAD Data** (if not done):
   ```bash
   python backend/migrate_bmad_to_consciousness.py
   ```

3. **Check Status**:
   ```bash
   python backend/check_consciousness_systems.py
   ```

4. **Add to Startup Protocol**:
   - Edit `CLAUDE.md`
   - Add consciousness systems check to Phase 3
   - Ensures every session starts with status awareness

---

## Benefits

### Knowledge Unification
- All knowledge (skills, decisions, documents) in one graph
- Consciousness processing ensures consistent structure
- ThoughtSeeds create automatic cross-references

### Discoverability
- Full-text search across all processed content
- Concept-based navigation
- Attractor basin clustering
- Related skill/decision recommendations

### Session Continuity
- Status check shows what's available
- No need to remember initialization state
- Clear action items if setup incomplete

### Consciousness Enhancement
- Every piece of knowledge gets 5-level concept extraction
- Attractor basins reveal knowledge domains
- ThoughtSeeds enable serendipitous connections
- Meta-cognitive tracking shows knowledge evolution

---

## Troubleshooting

### Skills Not Initializing
**Symptom**: `initialize_skills.py` fails
**Check**:
1. Skills library exists: `ls /Volumes/Asylum/skills-library/`
2. Neo4j running: `neo4j status`
3. Daedalus available: `python -c "from backend.src.services.daedalus import Daedalus"`

### BMAD Migration Fails
**Symptom**: `migrate_bmad_to_consciousness.py` errors
**Check**:
1. BMAD nodes exist: `cypher-shell -u neo4j -p dionysus "MATCH (n) WHERE id(n) IN [5,6] RETURN n"`
2. Neo4j connection: `neo4j status`
3. Daedalus available (same as above)

### Status Check Shows Errors
**Symptom**: Red ❌ in status report
**Action**:
1. Read error message
2. Check Neo4j: `neo4j status`
3. Check Python environment: `which python3`
4. Check imports: `python -c "import sys; sys.path.insert(0, 'backend/src'); from services.skills_manager import SkillsManager"`

---

**Document Updated**: 2025-11-13
**Systems Status**: Implementation Complete, Ready for Initialization
