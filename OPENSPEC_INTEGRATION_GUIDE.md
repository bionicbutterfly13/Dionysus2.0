# OpenSpec Integration Guide for Flux Desktop App

**Date**: 2025-10-12
**Purpose**: How OpenSpec streamlines our 3-month desktop app development workflow

---

## Overview

OpenSpec provides **spec-driven development** that perfectly complements our **week-by-week roadmap**. Instead of ad-hoc implementation, we:

1. **Plan features as OpenSpec changes** (proposals)
2. **Track implementation through tasks.md** (checkboxes)
3. **Validate before deployment** (openspec validate)
4. **Archive completed work** (openspec archive)

---

## How OpenSpec Maps to Our 3-Month Roadmap

### Current Workflow (Without OpenSpec)
```
Week Plan → Implement → Hope it works → Debug → Ship
```

### OpenSpec Workflow (Structured)
```
Week Plan → Create Proposal → Validate → Implement → Track → Archive
```

---

## Week-by-Week OpenSpec Integration

### **Week 1: Foundation + Core Editing** (Current)

#### Traditional Approach
- ❌ Scattered TODOs in comments
- ❌ No formal spec of what "file tree" means
- ❌ Unclear what "success" looks like
- ❌ Hard to track progress across team

#### OpenSpec Approach
```bash
# Create proposal for Week 1 features
openspec list --specs   # Check existing capabilities
cd /Volumes/Asylum/dev/Dionysus-2.0

# Scaffold proposal
mkdir -p openspec/changes/week-1-foundation/specs/desktop-core
```

**Proposal Structure**:
```markdown
openspec/changes/week-1-foundation/
├── proposal.md           # Why: Need desktop foundation
├── tasks.md             # Week 1 checklist
├── design.md            # Tauri architecture decisions
└── specs/
    └── desktop-core/
        └── spec.md      # ADDED Requirements for:
                         # - File system operations
                         # - Workspace management
                         # - CodeMirror editor
                         # - Auto-save
```

**Benefits**:
- ✅ Clear requirements with scenarios
- ✅ Validated before implementation
- ✅ Trackable progress (tasks.md)
- ✅ Historical record (archive)

---

### **Week 2: File System & Workspace**

#### OpenSpec Change: `add-file-workspace-management`

**File**: `openspec/changes/add-file-workspace-management/specs/file-management/spec.md`

```markdown
## ADDED Requirements

### Requirement: Workspace Selection
The system SHALL allow users to select a local folder as a workspace.

#### Scenario: Open workspace
- **WHEN** user clicks "Open Workspace" in menu
- **THEN** system displays native folder picker
- **AND** loads all .md files from selected folder
- **AND** displays file tree in sidebar

### Requirement: File Tree Display
The system SHALL display workspace files in a hierarchical tree.

#### Scenario: Expand folder
- **WHEN** user clicks folder with arrow icon
- **THEN** folder expands to show children
- **AND** arrow icon rotates to indicate expanded state

### Requirement: File Operations
The system SHALL support create, rename, delete operations.

#### Scenario: Create new file
- **WHEN** user right-clicks folder → New File
- **THEN** system displays file name input dialog
- **AND** creates .md file with entered name
- **AND** opens file in editor
```

**Tasks.md**:
```markdown
## 1. File System Operations
- [ ] 1.1 Implement read_file command (Rust)
- [ ] 1.2 Implement write_file command (Rust)
- [ ] 1.3 Implement read_dir command (Rust)
- [ ] 1.4 Add file watcher (Rust)

## 2. Workspace Management
- [ ] 2.1 Create WorkspaceStore (Zustand)
- [ ] 2.2 Implement workspace selector UI
- [ ] 2.3 Persist workspace path on disk
- [ ] 2.4 Auto-load last workspace on start

## 3. File Tree Component
- [ ] 3.1 Create <FileTree> component
- [ ] 3.2 Implement expand/collapse
- [ ] 3.3 Add context menu
- [ ] 3.4 Handle file click to open
```

**Validation**:
```bash
openspec validate add-file-workspace-management --strict
# ✅ All requirements have scenarios
# ✅ All scenarios follow format
# ✅ No orphan requirements
```

---

### **Week 3: CodeMirror Editor**

#### OpenSpec Change: `add-markdown-editor`

**Key Requirement**:
```markdown
### Requirement: Markdown Editing
The system SHALL provide a WYSIWYM markdown editor with syntax highlighting.

#### Scenario: Edit markdown document
- **WHEN** user opens .md file
- **THEN** editor displays content with syntax highlighting
- **AND** bold text (**text**) renders in bold font
- **AND** headers (# Header) render larger
- **AND** links render as clickable

#### Scenario: Formatting shortcuts
- **WHEN** user selects text and presses Ctrl+B
- **THEN** text is wrapped with **markers**
- **AND** renders as bold
```

**Design.md Decision**:
```markdown
## Decision: CodeMirror 6 vs Monaco vs Lexical

**Chosen**: CodeMirror 6

**Rationale**:
- Lightweight (vs Monaco's 1MB+)
- Markdown-first (vs Monaco's code focus)
- React integration via @codemirror/basic-setup
- Extension system for WikiLinks, consciousness markers

**Alternatives Considered**:
- Monaco: Too heavy, VS Code-focused
- Lexical: Facebook's editor, less markdown maturity

**Trade-offs**:
- +: Performance, size, markdown focus
- -: Less feature-rich than Monaco
```

---

### **Week 5: Search & Navigation**

#### OpenSpec Change: `add-full-text-search`

**Requirement with Consciousness Integration**:
```markdown
### Requirement: Consciousness-Weighted Search
The system SHALL rank search results by relevance AND consciousness level.

#### Scenario: Search with consciousness boost
- **WHEN** user searches for "quantum"
- **THEN** results include all documents mentioning "quantum"
- **AND** documents with higher consciousness scores rank higher
- **AND** results display consciousness level badge

#### Scenario: Quick switcher
- **WHEN** user presses Ctrl+P
- **THEN** fuzzy search overlay appears
- **AND** typing filters file names instantly
- **AND** arrow keys navigate results
- **AND** Enter opens selected file
```

---

### **Week 8: Graph View & WikiLinks**

#### OpenSpec Change: `add-knowledge-graph`

**Complex Requirement**:
```markdown
### Requirement: WikiLink Syntax
The system SHALL detect [[Link]] syntax and create navigable connections.

#### Scenario: Create WikiLink
- **WHEN** user types [[Document Name]]
- **THEN** text renders as clickable link
- **AND** autocomplete suggests matching files
- **AND** clicking link opens target document

### Requirement: Graph Visualization
The system SHALL display document connections as interactive graph.

#### Scenario: View knowledge graph
- **WHEN** user opens Graph View panel
- **THEN** system displays D3 force-directed graph
- **AND** nodes represent documents (size = word count)
- **AND** edges represent WikiLinks
- **AND** node color indicates consciousness level
- **AND** clicking node opens document
- **AND** dragging rearranges layout

#### Scenario: Consciousness basin visualization
- **WHEN** graph view is open with consciousness data
- **THEN** nodes cluster by attractor basin
- **AND** cluster colors match basin identity
- **AND** hovering shows basin metadata
```

---

## OpenSpec Benefits for Our Project

### 1. **Clear Success Criteria**

**Without OpenSpec**:
> "Implement graph view" - What does "done" mean?

**With OpenSpec**:
```markdown
#### Scenario: Graph renders 500+ nodes
- **WHEN** workspace has 500+ documents
- **THEN** graph renders in <2 seconds
- **AND** interactions remain smooth (>30 FPS)
- **AND** memory usage <300MB
```

### 2. **Validation Before Coding**

```bash
# Before Week 8 implementation
openspec validate add-knowledge-graph --strict

# Catches issues:
# ❌ Requirement "Graph Filtering" missing scenario
# ❌ Requirement "Node Click" has invalid scenario format
# ✅ Fix before writing code
```

### 3. **Progress Tracking**

**tasks.md as Single Source of Truth**:
```markdown
## 1. WikiLink Parser
- [x] 1.1 Regex for [[Link]] detection
- [x] 1.2 CodeMirror extension for highlighting
- [ ] 1.3 Autocomplete integration
- [ ] 1.4 Link resolver (file matching)

## 2. Graph Visualization
- [ ] 2.1 Install D3.js
- [ ] 2.2 Create <GraphView> component
- [ ] 2.3 Force-directed layout algorithm
- [ ] 2.4 Node rendering with consciousness colors
```

**Real-time status**:
```bash
openspec show add-knowledge-graph
# Progress: 2/8 tasks complete (25%)
```

### 4. **Architecture Documentation**

**design.md Captures Decisions**:
```markdown
## Decision: D3.js vs Cytoscape.js vs Vis.js

**Chosen**: D3.js

**Context**: Need performant graph for 500+ nodes

**Rationale**:
- D3: Full control, WebGL support, React integration
- Cytoscape: Biological networks focus, less React-native
- Vis.js: Simpler but less performant at scale

**Trade-offs**:
- +: Performance, flexibility, community
- -: Steeper learning curve

**Migration Plan**:
- Abstract graph logic in GraphService interface
- If D3 inadequate, swap implementation

**Open Questions**:
- WebGL vs Canvas rendering? (Benchmark at 1000+ nodes)
```

### 5. **Prevent Scope Creep**

**Proposal Forces Scope Definition**:
```markdown
## What Changes

### In Scope (Week 8)
- WikiLink [[syntax]] detection and rendering
- Basic graph view (nodes + edges)
- Click to navigate
- Consciousness-based coloring

### Out of Scope (Future)
- Advanced graph layouts (hierarchical, radial)
- Graph editing (drag to create links)
- Time-based graph evolution
- 3D graph visualization
```

### 6. **Historical Record**

After deployment:
```bash
openspec archive add-knowledge-graph --yes

# Creates:
openspec/changes/archive/2025-11-30-add-knowledge-graph/
├── proposal.md      # What we built
├── tasks.md         # How we built it
├── design.md        # Why we built it this way
└── specs/           # Requirements that are now live
```

**Benefits**:
- Future developers understand decisions
- Onboarding: "Read archived changes"
- Knowledge preservation

---

## Integration with Existing Workflow

### Current Process (Week 1)
```
1. Check roadmap: FLUX_3MONTH_ROADMAP.md
2. Create TODOs in Claude
3. Implement features
4. Test manually
5. Ship
```

### Enhanced Process (OpenSpec)
```
1. Check roadmap: FLUX_3MONTH_ROADMAP.md
2. Create OpenSpec proposal:
   openspec changes/week-1-foundation/
3. Write specs with scenarios
4. Validate: openspec validate week-1-foundation --strict
5. Get approval (self-review or team)
6. Implement using tasks.md checklist
7. Mark tasks complete as you go
8. After deployment: openspec archive week-1-foundation
```

---

## Practical Example: Week 1 Right Now

### What We've Done (Without OpenSpec)
- ✅ Tauri project initialized
- ✅ Plugins configured
- ✅ Dependencies installed
- ⏳ Building first time

### What We Should Do (With OpenSpec)

```bash
# 1. Create Week 1 proposal
cd /Volumes/Asylum/dev/Dionysus-2.0
mkdir -p openspec/changes/week-1-foundation/specs/desktop-core

# 2. Write proposal.md
cat > openspec/changes/week-1-foundation/proposal.md << 'EOF'
## Why
Flux needs a desktop foundation to evolve from web app to native application.
Tauri provides lightweight (3-10MB), fast (<0.5s startup), secure desktop framework.

## What Changes
- Desktop project initialization (Tauri 2.0)
- File system operations (read/write/watch)
- Workspace management (open folder, persist path)
- CodeMirror 6 markdown editor
- Auto-save functionality
- Native menus

**BREAKING**: This changes deployment from web (browser) to desktop (app installer)

## Impact
- Affected specs: desktop-core (NEW capability)
- Affected code: flux-desktop/ (new directory)
- Migration: Users download .dmg/.exe instead of visiting URL
EOF

# 3. Write spec deltas
cat > openspec/changes/week-1-foundation/specs/desktop-core/spec.md << 'EOF'
## ADDED Requirements

### Requirement: Desktop Application Initialization
The system SHALL run as native desktop application using Tauri 2.0.

#### Scenario: First launch
- **WHEN** user double-clicks Flux.app
- **THEN** application window opens in <0.5 seconds
- **AND** bundle size is <15MB
- **AND** memory usage <150MB

### Requirement: File System Access
The system SHALL read and write local markdown files.

#### Scenario: Read markdown file
- **WHEN** user selects file from file tree
- **THEN** system reads file content from disk
- **AND** displays in editor within 100ms
EOF

# 4. Write tasks.md
cat > openspec/changes/week-1-foundation/tasks.md << 'EOF'
## 1. Project Setup
- [x] 1.1 Install Rust toolchain
- [x] 1.2 Install Tauri CLI
- [x] 1.3 Initialize Tauri project
- [x] 1.4 Install npm dependencies
- [x] 1.5 Configure Tauri plugins

## 2. File System Operations
- [ ] 2.1 Implement read_file command
- [ ] 2.2 Implement write_file command
- [ ] 2.3 Add file watcher
- [ ] 2.4 Test file operations

## 3. Editor Integration
- [ ] 3.1 Install CodeMirror 6
- [ ] 3.2 Create MarkdownEditor component
- [ ] 3.3 Add syntax highlighting
- [ ] 3.4 Test editor rendering

## 4. Validation
- [ ] 4.1 App launches in <0.5s
- [ ] 4.2 Hot reload works
- [ ] 4.3 Can edit and save files
EOF

# 5. Validate
openspec validate week-1-foundation --strict

# 6. Implement (what we're doing now)
# ... continue with implementation ...

# 7. After Week 1 complete
openspec archive week-1-foundation --yes
```

---

## OpenSpec Commands Cheat Sheet

### Discovery
```bash
openspec list                    # Active changes
openspec list --specs             # Existing capabilities
openspec show <change-id>        # Change details
openspec diff <change-id>        # What specs will change
```

### Creation
```bash
mkdir -p openspec/changes/<change-id>/specs/<capability>
# Write: proposal.md, tasks.md, design.md, specs/**/*.md
```

### Validation
```bash
openspec validate <change-id> --strict    # Full validation
openspec validate --strict                 # All changes
```

### Implementation
```bash
# Edit tasks.md, mark items complete as you go
# - [ ] Task → - [x] Task
```

### Archiving
```bash
openspec archive <change-id> --yes        # After deployment
```

---

## When to Create OpenSpec Proposals

### ✅ Create Proposal
- New desktop feature (editor, graph, export)
- Architecture change (Tauri config, plugin addition)
- Breaking changes (file format, API change)
- Performance optimizations (caching strategy)
- Security updates (permission changes)

### ❌ Skip Proposal
- Bug fixes (restore intended behavior)
- Typos, formatting, comments
- Dependency updates (patch versions)
- Configuration tweaks (color changes)
- Tests for existing behavior

---

## Recommended OpenSpec Structure for Our 3-Month Plan

```
openspec/changes/
├── week-1-foundation/           # Week 1
├── week-2-file-management/      # Week 2
├── week-3-markdown-editor/      # Week 3
├── week-4-state-persistence/    # Week 4
├── week-5-search-navigation/    # Week 5
├── week-6-writing-tools/        # Week 6
├── week-7-tabs-split-view/      # Week 7
├── week-8-knowledge-graph/      # Week 8
├── week-9-export-system/        # Week 9
├── week-10-citations/           # Week 10
├── week-11-themes-settings/     # Week 11
└── week-12-distribution/        # Week 12
```

Each with:
- `proposal.md` - Why this week's features
- `tasks.md` - Checklist (from roadmap)
- `design.md` - Key architecture decisions
- `specs/` - Requirements with scenarios

---

## Benefits Summary

| Aspect | Without OpenSpec | With OpenSpec |
|--------|------------------|---------------|
| **Planning** | Informal TODOs | Formal proposals |
| **Requirements** | Implicit | Explicit scenarios |
| **Validation** | Manual review | Automated validation |
| **Progress** | Ad-hoc tracking | tasks.md checkboxes |
| **History** | Lost over time | Archived & searchable |
| **Onboarding** | Ask teammates | Read archived changes |
| **Scope Control** | Creep risk | Defined boundaries |
| **Testing** | What to test? | Clear scenarios |

---

## Next Steps

### Immediate (After Tauri Build Completes)
1. Create `week-1-foundation` proposal
2. Document what we've already done in tasks.md
3. Continue implementation with OpenSpec tracking

### This Week
- Use OpenSpec for all Week 1 features
- Validate before deployment
- Archive on Friday (end of Week 1)

### Going Forward
- One OpenSpec change per week (aligned with roadmap)
- Review/approve proposals on Monday
- Implement Tuesday-Thursday
- Archive on Friday

---

## Conclusion

OpenSpec transforms our 3-month roadmap from a **plan** into a **living specification**. Instead of hoping we built the right thing, we:

1. **Define success upfront** (scenarios)
2. **Validate requirements** (openspec validate)
3. **Track progress systematically** (tasks.md)
4. **Preserve knowledge** (archive)

This doesn't add overhead - it replaces informal processes with structured ones, making our desktop app development **faster, clearer, and more maintainable**.

**Start now**: Create `week-1-foundation` proposal while Tauri builds! 🚀
