# OpenSpec-Archon Integration

## Overview

Automated bidirectional synchronization between OpenSpec change management (file-based specifications) and Archon MCP task tracking (API-based project management), eliminating manual task import and enabling real-time status updates between systems.

## ADDED Requirements

### Requirement: Import OpenSpec Change to Archon
The system SHALL import an OpenSpec change proposal into Archon as a project with tasks, creating bidirectional traceability between specification and implementation tracking.

#### Scenario: Import change with multiple tasks
**Given** an OpenSpec change exists at `openspec/changes/add-search-feature/` with `proposal.md` and `tasks.md` containing 5 checklist items
**When** developer runs `/openspec:import-to-archon add-search-feature`
**Then** the system:
- Creates Archon project with title from proposal.md H1
- Creates 5 Archon tasks (status: "todo") matching tasks.md items
- Writes Archon project UUID to `.archon-project-id` file
- Reports "✅ Created Archon project XYZ with 5 tasks"

#### Scenario: Import fails when change not found
**Given** no OpenSpec change exists at `openspec/changes/nonexistent/`
**When** developer runs `/openspec:import-to-archon nonexistent`
**Then** the system returns error "Change 'nonexistent' not found in openspec/changes/"

#### Scenario: Import fails when Archon unavailable
**Given** Archon MCP server is offline
**When** developer runs `/openspec:import-to-archon add-search-feature`
**Then** the system returns error "Cannot connect to Archon MCP server" with troubleshooting steps

---

### Requirement: Sync Archon Task Status to OpenSpec
The system SHALL automatically update OpenSpec `tasks.md` checkboxes when corresponding Archon tasks are completed, maintaining synchronization between specification and implementation progress.

#### Scenario: Task completion syncs to tasks.md
**Given** Archon project has 3 tasks, all status "todo"
**And** tasks.md shows `- [ ] Task 1`, `- [ ] Task 2`, `- [ ] Task 3`
**When** developer marks "Task 2" as "done" in Archon
**Then** within 30 seconds:
- tasks.md updates to `- [ ] Task 1`, `- [x] Task 2`, `- [ ] Task 3`
- Git commit created: "sync: Update tasks.md from Archon (change: add-search-feature)"

#### Scenario: Multiple tasks sync in batch
**Given** 3 Archon tasks completed in quick succession
**When** sync process runs next cycle
**Then** tasks.md updates all 3 checkboxes in a single git commit

#### Scenario: No sync when manually checked
**Given** tasks.md already shows `- [x] Task 1` (manually updated)
**When** sync process detects Archon task 1 is "done"
**Then** no update occurs (idempotent sync)

---

### Requirement: Validate Archon Completion on Archive
The system SHALL prevent archiving OpenSpec changes with incomplete Archon tasks, ensuring specifications match implementation completion status.

#### Scenario: Archive succeeds when all tasks done
**Given** OpenSpec change has `.archon-project-id` file
**And** all Archon tasks have status "done"
**When** developer runs `/openspec:archive add-search-feature`
**Then** the system:
- Validates all Archon tasks complete
- Archives Archon project
- Archives OpenSpec change to `changes/archive/`

#### Scenario: Archive fails with incomplete tasks
**Given** OpenSpec change has `.archon-project-id` file
**And** 2 out of 5 Archon tasks have status "todo"
**When** developer runs `/openspec:archive add-search-feature`
**Then** the system returns error listing 2 incomplete tasks and blocks archive

#### Scenario: Archive skips validation without Archon integration
**Given** OpenSpec change has no `.archon-project-id` file
**When** developer runs `/openspec:archive add-search-feature`
**Then** archive proceeds without Archon validation (backward compatible)

---

### Requirement: Bidirectional Traceability
The system SHALL maintain references between OpenSpec changes and Archon projects, enabling navigation from specification to tasks and vice versa.

#### Scenario: OpenSpec stores Archon project reference
**Given** import creates Archon project with UUID `550e8400-...`
**When** import completes
**Then** file `.archon-project-id` contains `550e8400-...`

#### Scenario: Archon stores OpenSpec change reference
**Given** importing change `add-search-feature`
**When** Archon project is created
**Then** project metadata includes `{"openspec_change_id": "add-search-feature"}`

#### Scenario: Archon tasks store task index
**Given** importing 5 tasks from tasks.md
**When** Archon tasks are created
**Then** each task metadata includes `{"openspec_task_index": 0..4}`

---

### Requirement: Parse Tasks from Markdown Checklist
The system SHALL extract task items from OpenSpec `tasks.md` markdown format for import into Archon, handling various checklist formats and edge cases.

#### Scenario: Parse standard checklist items
**Given** tasks.md contains:
```markdown
- [ ] Implement search API
- [ ] Add frontend component
- [x] Write tests (pre-checked)
```
**When** task parser runs
**Then** 3 tasks extracted:
- Task 0: "Implement search API" (unchecked)
- Task 1: "Add frontend component" (unchecked)
- Task 2: "Write tests" (checked, status: "done")

#### Scenario: Ignore subtasks
**Given** tasks.md contains:
```markdown
- [ ] Main task
  - [ ] Subtask 1
  - [ ] Subtask 2
```
**When** task parser runs
**Then** only 1 task extracted: "Main task" (subtasks ignored)

#### Scenario: Handle phase headers
**Given** tasks.md contains:
```markdown
## Phase 1: Setup
- [ ] Task A
## Phase 2: Implementation
- [ ] Task B
```
**When** task parser runs
**Then** 2 tasks extracted with section metadata:
- Task 0: "Task A" (section: "Phase 1: Setup")
- Task 1: "Task B" (section: "Phase 2: Implementation")

---

## Non-Functional Requirements

### Performance
- Import operation: < 5 seconds for 20 tasks
- Status sync polling: 30 second interval (configurable)
- Archive validation: < 2 seconds
- tasks.md update: < 1 second file write + git commit

### Reliability
- Idempotent sync: Multiple syncs with same status produce same result
- Atomic file updates: tasks.md never left in invalid state
- Error recovery: Archon API failures don't corrupt OpenSpec files

### Usability
- Clear error messages with troubleshooting steps
- Progress feedback during import (task count)
- Warning on parse failures (unparsed tasks reported)

### Compatibility
- Backward compatible: OpenSpec changes without `.archon-project-id` work normally
- Forward compatible: Additional metadata fields ignored by parser
- Git-friendly: Automatic commits use conventional commit format

---

## Acceptance Criteria

- [ ] `/openspec:import-to-archon <id>` command creates Archon project + tasks
- [ ] `.archon-project-id` file stores project UUID reference
- [ ] Sync updates tasks.md checkboxes when Archon tasks complete (30s latency)
- [ ] `/openspec:archive` validates all Archon tasks done before archiving
- [ ] Integration tested end-to-end with sample change (import → work → archive)
- [ ] Error handling tested (Archon offline, parse errors, incomplete tasks)
- [ ] Documentation updated in `openspec/AGENTS.md` with new workflow
- [ ] Backward compatibility verified (existing changes without integration work)
