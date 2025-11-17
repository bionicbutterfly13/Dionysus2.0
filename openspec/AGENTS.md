# OpenSpec Instructions

Instructions for AI coding assistants using OpenSpec for spec-driven development.

## TL;DR Quick Checklist

- Search existing work: `openspec spec list --long`, `openspec list` (use `rg` only for full-text search)
- Decide scope: new capability vs modify existing capability
- Pick a unique `change-id`: kebab-case, verb-led (`add-`, `update-`, `remove-`, `refactor-`)
- Scaffold: `proposal.md`, `tasks.md`, `design.md` (only if needed), and delta specs per affected capability
- Write deltas: use `## ADDED|MODIFIED|REMOVED|RENAMED Requirements`; include at least one `#### Scenario:` per requirement
- Validate: `openspec validate [change-id] --strict` and fix issues
- Request approval: Do not start implementation until proposal is approved

## Three-Stage Workflow

### Stage 1: Creating Changes
Create proposal when you need to:
- Add features or functionality
- Make breaking changes (API, schema)
- Change architecture or patterns  
- Optimize performance (changes behavior)
- Update security patterns

Triggers (examples):
- "Help me create a change proposal"
- "Help me plan a change"
- "Help me create a proposal"
- "I want to create a spec proposal"
- "I want to create a spec"

Loose matching guidance:
- Contains one of: `proposal`, `change`, `spec`
- With one of: `create`, `plan`, `make`, `start`, `help`

Skip proposal for:
- Bug fixes (restore intended behavior)
- Typos, formatting, comments
- Dependency updates (non-breaking)
- Configuration changes
- Tests for existing behavior

**Workflow**
1. Review `openspec/project.md`, `openspec list`, and `openspec list --specs` to understand current context.
2. Choose a unique verb-led `change-id` and scaffold `proposal.md`, `tasks.md`, optional `design.md`, and spec deltas under `openspec/changes/<id>/`.
3. Draft spec deltas using `## ADDED|MODIFIED|REMOVED Requirements` with at least one `#### Scenario:` per requirement.
4. Run `openspec validate <id> --strict` and resolve any issues before sharing the proposal.

### Stage 2: Implementing Changes
Track these steps as TODOs and complete them one by one.
1. **Read proposal.md** - Understand what's being built
2. **Read design.md** (if exists) - Review technical decisions
3. **Read tasks.md** - Get implementation checklist
4. **Implement tasks sequentially** - Complete in order
5. **Confirm completion** - Ensure every item in `tasks.md` is finished before updating statuses
6. **Update checklist** - After all work is done, set every task to `- [x]` so the list reflects reality
7. **Approval gate** - Do not start implementation until the proposal is reviewed and approved

### Stage 3: Archiving Changes
After deployment, create separate PR to:
- Move `changes/[name]/` → `changes/archive/YYYY-MM-DD-[name]/`
- Update `specs/` if capabilities changed
- Use `openspec archive <change-id> --skip-specs --yes` for tooling-only changes (always pass the change ID explicitly)
- Run `openspec validate --strict` to confirm the archived change passes checks

## Archon Integration (Task Management)

OpenSpec integrates with Archon MCP for automated task tracking and bidirectional sync between specifications and project management.

### Integrated Workflow

```
1. Create OpenSpec change
   ↓
2. /openspec:import-to-archon <change-id>
   ├─ Creates Archon project
   ├─ Imports all tasks from tasks.md
   └─ Stores .archon-project-id for linking
   ↓
3. Work on tasks in Archon
   ├─ Get tasks: find_tasks(status="todo")
   ├─ Start: manage_task(status="doing")
   ├─ Complete: manage_task(status="done")
   └─ Track progress in real-time
   ↓
4. /openspec:sync-status <change-id>
   ├─ Queries Archon for task status
   ├─ Updates tasks.md checkboxes
   ├─ Commits changes automatically
   └─ Shows completion percentage
   ↓
5. /openspec:archive <change-id>
   ├─ Validates all Archon tasks done
   ├─ Archives Archon project (optional)
   ├─ Archives OpenSpec change
   └─ Updates main specs
```

### Commands

**Import to Archon**:
```bash
/openspec:import-to-archon <change-id>
```
- Creates Archon project with title from proposal.md
- Creates one Archon task per tasks.md checkbox
- Stores UUID in `.archon-project-id`
- Tasks inherit priority from document order (task_order: 100-1)

**Sync Status**:
```bash
/openspec:sync-status <change-id>
```
- Fetches Archon task status (todo/doing/review/done)
- Matches tasks by title (85% similarity threshold)
- Updates checkboxes: `[ ]` → `[x]` for completed tasks
- Auto-commits: `chore: sync task status from Archon [N/M complete]`
- Conflict resolution: Manual `[x]` checkboxes take precedence

**Archive with Validation**:
```bash
/openspec:archive <change-id>
```
- Reads `.archon-project-id` if exists
- Validates all Archon tasks are "done"
- If incomplete: warns + prompts for confirmation
- If complete: offers to archive Archon project
- Proceeds with OpenSpec archive

### Configuration

Create `.openspec.config.json` to customize sync behavior:

```json
{
  "archon_sync": {
    "sync_interval_seconds": 30,
    "auto_sync_on_archive": true,
    "conflict_resolution": "archon_wins",
    "similarity_threshold": 0.85,
    "auto_commit": true
  }
}
```

### Checkbox Symbols

- `[ ]` - Not started (todo)
- `[-]` - In progress (doing)
- `[~]` - Under review (review)
- `[x]` - Completed (done)

### Best Practices

1. **Import early**: Run import command right after creating change proposal
2. **Sync regularly**: Run sync before major commits or before archive
3. **Complete in Archon**: Mark tasks done in Archon (not manually in tasks.md)
4. **Validate before archive**: Ensure all tasks complete to avoid orphaned work

### Troubleshooting

**No .archon-project-id file**:
- Run `/openspec:import-to-archon <change-id>` first
- File created automatically during import

**Tasks not matching**:
- Task titles must be 85%+ similar between Archon and tasks.md
- Avoid major rewording after import
- Check match results in sync output

**Incomplete tasks blocking archive**:
```
⚠️ Archon project has 5 incomplete tasks (3 todo, 2 doing)
```
- Option 1: Complete remaining tasks, then re-run archive
- Option 2: Force archive (not recommended - leaves orphaned tasks)
- Option 3: Sync status first to verify tasks.md is current

## OpenSpec Metadata Schema (Neo4j Knowledge Graph)

OpenSpec specifications can be ingested into the Neo4j knowledge graph for semantic search, cross-spec relationship discovery, and consciousness-enhanced processing. This section documents the metadata schema used when specs are processed through the document pipeline.

### Node Labels

**Document:Specification**:
- Represents a capability's `spec.md` file (requirements and scenarios)
- Source: `openspec/specs/<capability>/spec.md`

**Document:DesignDocument**:
- Represents a capability's `design.md` file (implementation patterns)
- Source: `openspec/specs/<capability>/design.md`

**Concept:Requirement**:
- Individual requirement extracted from spec.md
- Created from `### Requirement:` headers

**Concept:Scenario**:
- Test scenario within a requirement
- Created from `#### Scenario:` headers
- Contains given/when/then structure

### Node Properties

**Specification/DesignDocument nodes**:
```cypher
(:Document:Specification {
  id: "uuid",                              # Unique document identifier
  title: "Document Processing",            # Derived from capability name
  content_hash: "abc123...",               # SHA-256 hash for deduplication
  source_type: "openspec",                 # Always "openspec" for ingested specs
  capability: "document-processing",       # Parent directory name (kebab-case)
  spec_type: "spec",                       # "spec" or "design"
  version: "1.0",                          # Semantic version (incremented on changes)
  extracted_text: "...",                   # Full markdown content
  summary: "...",                          # AI-generated summary
  created_at: datetime(),                  # Ingestion timestamp
  updated_at: datetime()                   # Last modification timestamp
})
```

**Requirement nodes**:
```cypher
(:Concept:Requirement {
  id: "uuid",
  title: "Import OpenSpec specifications",  # From ### Requirement: header
  description: "..."                         # Full requirement text
})
```

**Scenario nodes**:
```cypher
(:Concept:Scenario {
  id: "uuid",
  title: "Specification ingested successfully",  # From #### Scenario: header
  given: "...",                                  # Preconditions
  when: "...",                                   # Action
  then: "..."                                    # Expected outcome
})
```

### Relationships

**HAS_REQUIREMENT**:
```cypher
(:Specification)-[:HAS_REQUIREMENT]->(:Requirement)
```
Links a specification document to its extracted requirements.

**HAS_SCENARIO**:
```cypher
(:Requirement)-[:HAS_SCENARIO]->(:Scenario)
```
Links a requirement to its test scenarios (at least one per requirement).

**DEFINES_CAPABILITY**:
```cypher
(:Specification)-[:DEFINES_CAPABILITY]->(capability_string)
```
Connects specification to its capability domain (e.g., "document-processing", "clause-multi-agent").

**SIMILAR_TO**:
```cypher
(:Specification)-[:SIMILAR_TO {score: 0.87}]->(:Specification)
```
Cross-spec similarity discovered via ThoughtSeeds (semantic embeddings).

**HAS_THOUGHTSEED**:
```cypher
(:Specification)-[:HAS_THOUGHTSEED]->(:ThoughtSeed)
```
Links specification to consciousness-generated concepts that emerge during processing.

**BELONGS_TO_BASIN**:
```cypher
(:Specification)-[:BELONGS_TO_BASIN]->(:AttractorBasin)
```
Groups related specifications into conceptual clusters.

**NEXT_VERSION**:
```cypher
(:Specification {version: "1.0"})-[:NEXT_VERSION]->(:Specification {version: "1.1"})
```
Links version history when specs are updated (preserves old versions).

### ThoughtSeed Connections

**How ThoughtSeeds Connect Specifications**:

1. **During Ingestion**:
   - Each spec.md/design.md is processed through DocumentProcessingGraph
   - Consciousness layer generates 5-level concepts (ThoughtSeeds)
   - Embeddings created for semantic similarity

2. **Cross-Spec Discovery**:
   - ThoughtSeeds with similar embeddings link related concepts
   - Example: "Authentication patterns" ThoughtSeed connects:
     - `document-processing` spec (user credentials)
     - `clause-multi-agent` spec (agent authentication)
     - `knowledge-graph` spec (access control)

3. **Query Pattern**:
```cypher
// Find all specs related via shared ThoughtSeeds
MATCH (s1:Specification {capability: "document-processing"})
MATCH (s1)-[:HAS_THOUGHTSEED]->(ts:ThoughtSeed)
MATCH (ts)<-[:HAS_THOUGHTSEED]-(s2:Specification)
WHERE s1 <> s2
RETURN s1.capability, s2.capability, ts.content, ts.level
```

4. **AttractorBasin Clustering**:
```cypher
// Find specifications in same conceptual cluster
MATCH (s:Specification)-[:BELONGS_TO_BASIN]->(b:AttractorBasin)
RETURN b.name, collect(s.capability) as related_capabilities
```

### Schema Constraints and Indexes

**Constraints**:
```cypher
CREATE CONSTRAINT spec_id IF NOT EXISTS
FOR (s:Specification) REQUIRE s.id IS UNIQUE;

CREATE CONSTRAINT design_id IF NOT EXISTS
FOR (d:DesignDocument) REQUIRE d.id IS UNIQUE;
```

**Indexes**:
```cypher
CREATE INDEX spec_capability IF NOT EXISTS
FOR (s:Specification) ON (s.capability);

CREATE INDEX spec_content_hash IF NOT EXISTS
FOR (s:Specification) ON (s.content_hash);

CREATE INDEX spec_source_type IF NOT EXISTS
FOR (s:Specification) ON (s.source_type);
```

### Deduplication Strategy

**Content Hash Checking**:
1. Before ingestion, SHA-256 hash calculated from file content
2. Query Neo4j for existing node with same `content_hash`
3. If exists: Return 409 Conflict, skip processing
4. If new: Proceed with normal ingestion

**Version Updates**:
1. Detect change: `content_hash` differs from existing spec
2. Increment version: `1.0` → `1.1`
3. Create new node (don't overwrite old version)
4. Link versions: `(v1.0)-[:NEXT_VERSION]->(v1.1)`

### Search Integration

**Semantic Search Example**:
```python
# Query: "Find specs about authentication"
POST /api/query
{
  "query": "authentication patterns",
  "filters": {
    "source_type": "openspec"
  }
}

# Returns:
[
  {
    "document_id": "uuid-1",
    "title": "Document Processing",
    "capability": "document-processing",
    "relevance_score": 0.87,
    "matched_concepts": ["Authentication", "User credentials"]
  },
  {
    "document_id": "uuid-2",
    "title": "CLAUSE Multi-Agent System",
    "capability": "clause-multi-agent",
    "relevance_score": 0.72,
    "matched_concepts": ["Agent authentication", "Trust verification"]
  }
]
```

**Graph Traversal Example**:
```cypher
// Find requirements for a capability
MATCH (s:Specification {capability: "document-processing"})
MATCH (s)-[:HAS_REQUIREMENT]->(r:Requirement)
MATCH (r)-[:HAS_SCENARIO]->(sc:Scenario)
RETURN s.title, r.title, collect(sc.title) as scenarios
```

### Ingestion Workflow

**Command**:
```bash
# Ingest all capabilities
python backend/scripts/ingest_openspec_specs.py --all

# Ingest specific capability
python backend/scripts/ingest_openspec_specs.py --capability document-processing

# Preview without ingesting
python backend/scripts/ingest_openspec_specs.py --all --dry-run
```

**Flow**:
1. Script scans `openspec/specs/` for spec.md/design.md files
2. Extracts metadata (capability, spec_type, content_hash)
3. POSTs to `/api/documents` with multipart/form-data
4. Daedalus gateway receives and passes to DocumentProcessingGraph
5. 6-node LangGraph workflow processes (extract, research, consciousness, analyze, refine, finalize)
6. DocumentRepository persists to Neo4j with metadata
7. Requirements/scenarios parsed and linked via HAS_REQUIREMENT/HAS_SCENARIO
8. ThoughtSeeds discover cross-spec relationships via semantic similarity

### Performance Characteristics

**Ingestion**:
- Target: < 10 seconds per spec file
- Bottleneck: Consciousness processing (5-10s per document)
- Optimization: Parallel processing via asyncio

**Search**:
- Target: < 500ms for semantic search across all specs
- Optimization: Neo4j vector index on embeddings
- Storage: ~100KB per spec.md (text + embeddings + concepts)

### Related Documentation

- Design document: `openspec/changes/ingest-specs-to-neo4j/design.md`
- Implementation script: `backend/scripts/ingest_openspec_specs.py` (when created)
- API endpoint: `backend/src/api/routes/documents.py` (POST /api/documents)
- Processing pipeline: `backend/src/services/document_processing_graph.py`

## Before Any Task

**Context Checklist:**
- [ ] Read relevant specs in `specs/[capability]/spec.md`
- [ ] Check pending changes in `changes/` for conflicts
- [ ] Read `openspec/project.md` for conventions
- [ ] Run `openspec list` to see active changes
- [ ] Run `openspec list --specs` to see existing capabilities

**Before Creating Specs:**
- Always check if capability already exists
- Prefer modifying existing specs over creating duplicates
- Use `openspec show [spec]` to review current state
- If request is ambiguous, ask 1–2 clarifying questions before scaffolding

### Search Guidance
- Enumerate specs: `openspec spec list --long` (or `--json` for scripts)
- Enumerate changes: `openspec list` (or `openspec change list --json` - deprecated but available)
- Show details:
  - Spec: `openspec show <spec-id> --type spec` (use `--json` for filters)
  - Change: `openspec show <change-id> --json --deltas-only`
- Full-text search (use ripgrep): `rg -n "Requirement:|Scenario:" openspec/specs`

## Quick Start

### CLI Commands

```bash
# Essential commands
openspec list                  # List active changes
openspec list --specs          # List specifications
openspec show [item]           # Display change or spec
openspec validate [item]       # Validate changes or specs
openspec archive <change-id> [--yes|-y]   # Archive after deployment (add --yes for non-interactive runs)

# Project management
openspec init [path]           # Initialize OpenSpec
openspec update [path]         # Update instruction files

# Interactive mode
openspec show                  # Prompts for selection
openspec validate              # Bulk validation mode

# Debugging
openspec show [change] --json --deltas-only
openspec validate [change] --strict
```

### Command Flags

- `--json` - Machine-readable output
- `--type change|spec` - Disambiguate items
- `--strict` - Comprehensive validation
- `--no-interactive` - Disable prompts
- `--skip-specs` - Archive without spec updates
- `--yes`/`-y` - Skip confirmation prompts (non-interactive archive)

## Directory Structure

```
openspec/
├── project.md              # Project conventions
├── specs/                  # Current truth - what IS built
│   └── [capability]/       # Single focused capability
│       ├── spec.md         # Requirements and scenarios
│       └── design.md       # Technical patterns
├── changes/                # Proposals - what SHOULD change
│   ├── [change-name]/
│   │   ├── proposal.md     # Why, what, impact
│   │   ├── tasks.md        # Implementation checklist
│   │   ├── design.md       # Technical decisions (optional; see criteria)
│   │   └── specs/          # Delta changes
│   │       └── [capability]/
│   │           └── spec.md # ADDED/MODIFIED/REMOVED
│   └── archive/            # Completed changes
```

## Creating Change Proposals

### Decision Tree

```
New request?
├─ Bug fix restoring spec behavior? → Fix directly
├─ Typo/format/comment? → Fix directly  
├─ New feature/capability? → Create proposal
├─ Breaking change? → Create proposal
├─ Architecture change? → Create proposal
└─ Unclear? → Create proposal (safer)
```

### Proposal Structure

1. **Create directory:** `changes/[change-id]/` (kebab-case, verb-led, unique)

2. **Write proposal.md:**
```markdown
# Change: [Brief description of change]

## Why
[1-2 sentences on problem/opportunity]

## What Changes
- [Bullet list of changes]
- [Mark breaking changes with **BREAKING**]

## Impact
- Affected specs: [list capabilities]
- Affected code: [key files/systems]
```

3. **Create spec deltas:** `specs/[capability]/spec.md`
```markdown
## ADDED Requirements
### Requirement: New Feature
The system SHALL provide...

#### Scenario: Success case
- **WHEN** user performs action
- **THEN** expected result

## MODIFIED Requirements
### Requirement: Existing Feature
[Complete modified requirement]

## REMOVED Requirements
### Requirement: Old Feature
**Reason**: [Why removing]
**Migration**: [How to handle]
```
If multiple capabilities are affected, create multiple delta files under `changes/[change-id]/specs/<capability>/spec.md`—one per capability.

4. **Create tasks.md:**
```markdown
## 1. Implementation
- [ ] 1.1 Create database schema
- [ ] 1.2 Implement API endpoint
- [ ] 1.3 Add frontend component
- [ ] 1.4 Write tests
```

5. **Create design.md when needed:**
Create `design.md` if any of the following apply; otherwise omit it:
- Cross-cutting change (multiple services/modules) or a new architectural pattern
- New external dependency or significant data model changes
- Security, performance, or migration complexity
- Ambiguity that benefits from technical decisions before coding

Minimal `design.md` skeleton:
```markdown
## Context
[Background, constraints, stakeholders]

## Goals / Non-Goals
- Goals: [...]
- Non-Goals: [...]

## Decisions
- Decision: [What and why]
- Alternatives considered: [Options + rationale]

## Risks / Trade-offs
- [Risk] → Mitigation

## Migration Plan
[Steps, rollback]

## Open Questions
- [...]
```

## Spec File Format

### Critical: Scenario Formatting

**CORRECT** (use #### headers):
```markdown
#### Scenario: User login success
- **WHEN** valid credentials provided
- **THEN** return JWT token
```

**WRONG** (don't use bullets or bold):
```markdown
- **Scenario: User login**  ❌
**Scenario**: User login     ❌
### Scenario: User login      ❌
```

Every requirement MUST have at least one scenario.

### Requirement Wording
- Use SHALL/MUST for normative requirements (avoid should/may unless intentionally non-normative)

### Delta Operations

- `## ADDED Requirements` - New capabilities
- `## MODIFIED Requirements` - Changed behavior
- `## REMOVED Requirements` - Deprecated features
- `## RENAMED Requirements` - Name changes

Headers matched with `trim(header)` - whitespace ignored.

#### When to use ADDED vs MODIFIED
- ADDED: Introduces a new capability or sub-capability that can stand alone as a requirement. Prefer ADDED when the change is orthogonal (e.g., adding "Slash Command Configuration") rather than altering the semantics of an existing requirement.
- MODIFIED: Changes the behavior, scope, or acceptance criteria of an existing requirement. Always paste the full, updated requirement content (header + all scenarios). The archiver will replace the entire requirement with what you provide here; partial deltas will drop previous details.
- RENAMED: Use when only the name changes. If you also change behavior, use RENAMED (name) plus MODIFIED (content) referencing the new name.

Common pitfall: Using MODIFIED to add a new concern without including the previous text. This causes loss of detail at archive time. If you aren’t explicitly changing the existing requirement, add a new requirement under ADDED instead.

Authoring a MODIFIED requirement correctly:
1) Locate the existing requirement in `openspec/specs/<capability>/spec.md`.
2) Copy the entire requirement block (from `### Requirement: ...` through its scenarios).
3) Paste it under `## MODIFIED Requirements` and edit to reflect the new behavior.
4) Ensure the header text matches exactly (whitespace-insensitive) and keep at least one `#### Scenario:`.

Example for RENAMED:
```markdown
## RENAMED Requirements
- FROM: `### Requirement: Login`
- TO: `### Requirement: User Authentication`
```

## Troubleshooting

### Common Errors

**"Change must have at least one delta"**
- Check `changes/[name]/specs/` exists with .md files
- Verify files have operation prefixes (## ADDED Requirements)

**"Requirement must have at least one scenario"**
- Check scenarios use `#### Scenario:` format (4 hashtags)
- Don't use bullet points or bold for scenario headers

**Silent scenario parsing failures**
- Exact format required: `#### Scenario: Name`
- Debug with: `openspec show [change] --json --deltas-only`

### Validation Tips

```bash
# Always use strict mode for comprehensive checks
openspec validate [change] --strict

# Debug delta parsing
openspec show [change] --json | jq '.deltas'

# Check specific requirement
openspec show [spec] --json -r 1
```

## Happy Path Script

```bash
# 1) Explore current state
openspec spec list --long
openspec list
# Optional full-text search:
# rg -n "Requirement:|Scenario:" openspec/specs
# rg -n "^#|Requirement:" openspec/changes

# 2) Choose change id and scaffold
CHANGE=add-two-factor-auth
mkdir -p openspec/changes/$CHANGE/{specs/auth}
printf "## Why\n...\n\n## What Changes\n- ...\n\n## Impact\n- ...\n" > openspec/changes/$CHANGE/proposal.md
printf "## 1. Implementation\n- [ ] 1.1 ...\n" > openspec/changes/$CHANGE/tasks.md

# 3) Add deltas (example)
cat > openspec/changes/$CHANGE/specs/auth/spec.md << 'EOF'
## ADDED Requirements
### Requirement: Two-Factor Authentication
Users MUST provide a second factor during login.

#### Scenario: OTP required
- **WHEN** valid credentials are provided
- **THEN** an OTP challenge is required
EOF

# 4) Validate
openspec validate $CHANGE --strict
```

## Multi-Capability Example

```
openspec/changes/add-2fa-notify/
├── proposal.md
├── tasks.md
└── specs/
    ├── auth/
    │   └── spec.md   # ADDED: Two-Factor Authentication
    └── notifications/
        └── spec.md   # ADDED: OTP email notification
```

auth/spec.md
```markdown
## ADDED Requirements
### Requirement: Two-Factor Authentication
...
```

notifications/spec.md
```markdown
## ADDED Requirements
### Requirement: OTP Email Notification
...
```

## Best Practices

### Simplicity First
- Default to <100 lines of new code
- Single-file implementations until proven insufficient
- Avoid frameworks without clear justification
- Choose boring, proven patterns

### Complexity Triggers
Only add complexity with:
- Performance data showing current solution too slow
- Concrete scale requirements (>1000 users, >100MB data)
- Multiple proven use cases requiring abstraction

### Clear References
- Use `file.ts:42` format for code locations
- Reference specs as `specs/auth/spec.md`
- Link related changes and PRs

### Capability Naming
- Use verb-noun: `user-auth`, `payment-capture`
- Single purpose per capability
- 10-minute understandability rule
- Split if description needs "AND"

### Change ID Naming
- Use kebab-case, short and descriptive: `add-two-factor-auth`
- Prefer verb-led prefixes: `add-`, `update-`, `remove-`, `refactor-`
- Ensure uniqueness; if taken, append `-2`, `-3`, etc.

## Tool Selection Guide

| Task | Tool | Why |
|------|------|-----|
| Find files by pattern | Glob | Fast pattern matching |
| Search code content | Grep | Optimized regex search |
| Read specific files | Read | Direct file access |
| Explore unknown scope | Task | Multi-step investigation |

## Error Recovery

### Change Conflicts
1. Run `openspec list` to see active changes
2. Check for overlapping specs
3. Coordinate with change owners
4. Consider combining proposals

### Validation Failures
1. Run with `--strict` flag
2. Check JSON output for details
3. Verify spec file format
4. Ensure scenarios properly formatted

### Missing Context
1. Read project.md first
2. Check related specs
3. Review recent archives
4. Ask for clarification

## Quick Reference

### Stage Indicators
- `changes/` - Proposed, not yet built
- `specs/` - Built and deployed
- `archive/` - Completed changes

### File Purposes
- `proposal.md` - Why and what
- `tasks.md` - Implementation steps
- `design.md` - Technical decisions
- `spec.md` - Requirements and behavior

### CLI Essentials
```bash
openspec list              # What's in progress?
openspec show [item]       # View details
openspec validate --strict # Is it correct?
openspec archive <change-id> [--yes|-y]  # Mark complete (add --yes for automation)
```

Remember: Specs are truth. Changes are proposals. Keep them in sync.
