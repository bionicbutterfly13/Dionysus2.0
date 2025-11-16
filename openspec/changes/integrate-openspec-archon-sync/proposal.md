# Integrate OpenSpec + Archon Bidirectional Sync

## What

Create automated bidirectional synchronization between OpenSpec change management and Archon MCP task tracking, eliminating manual task import and enabling real-time status updates.

## Why

**Current Pain Points**:
1. **Manual Task Import**: Creating an OpenSpec proposal generates `tasks.md`, but developers must manually create each Archon task via `manage_task()` calls
2. **No Status Sync**: Completing Archon tasks doesn't update OpenSpec `tasks.md` checkboxes
3. **Dual Tracking Overhead**: Developers track progress in both systems independently
4. **Lost Context**: No link from Archon tasks back to their originating OpenSpec change

**Benefits of Integration**:
- **One Command Import**: `/openspec:import-to-archon` creates Archon project + all tasks automatically
- **Real-Time Sync**: Archon task status updates flow back to OpenSpec `tasks.md`
- **Bidirectional Traceability**: Archon tasks reference OpenSpec change-id, OpenSpec shows Archon project ID
- **Simplified Workflow**: Developers work in Archon, specifications stay current automatically

## How

### Architecture

```
OpenSpec (File-based)                    Archon MCP (API-based)
┌──────────────────────┐                ┌─────────────────────────┐
│ changes/             │                │ Projects                │
│   <change-id>/       │   ┌────────┐   │ - id: project_id       │
│     proposal.md      │──>│ Import │──>│ - title: change title  │
│     tasks.md         │   │        │   │ - description: from     │
│     design.md        │   │ Slash  │   │   proposal.md          │
│     specs/           │   │ Command│   │                        │
│       <capability>/  │   └────────┘   │ Tasks                  │
│         spec.md      │                │ - [task_1]             │
└──────────────────────┘                │ - [task_2]             │
         ▲                              │ - [task_3]             │
         │                              └─────────────────────────┘
         │                                        │
         │   ┌──────────────────────────┐       │
         └───│ Status Sync (Webhook or  │<──────┘
             │ Polling)                 │
             └──────────────────────────┘
```

### Components

**1. OpenSpec Slash Command**: `/openspec:import-to-archon <change-id>`
   - Location: `.claude/commands/openspec/import-to-archon.md`
   - Reads: `openspec/changes/<change-id>/proposal.md` and `tasks.md`
   - Creates: Archon project via `manage_project("create", ...)`
   - Creates: Archon tasks via `manage_task("create", ...)` for each `tasks.md` item
   - Stores: Archon project_id in `openspec/changes/<change-id>/.archon-project-id`

**2. Archon Task Status Sync**
   - Trigger: On Archon task status update (todo → doing → review → done)
   - Action: Update corresponding checkbox in `tasks.md`
   - Mechanism: Polling via `find_tasks(project_id="...")` or webhook (if Archon supports)

**3. Archive Integration**
   - Enhanced: `/openspec:archive` checks Archon project status
   - Validation: All Archon tasks must be "done" before archiving
   - Cleanup: Archive Archon project when archiving OpenSpec change

### Data Flow

**Import Flow**:
1. Developer runs `/openspec:import-to-archon sync-integration`
2. Command reads `proposal.md` (title, description) and `tasks.md` (checklist)
3. Calls `manage_project("create", title="Integrate OpenSpec + Archon...", ...)`
4. For each `- [ ] task` in `tasks.md`:
   - Extract task title and description
   - Call `manage_task("create", project_id="...", title="...", task_order=...)`
5. Write Archon project_id to `.archon-project-id` file
6. Report: "✅ Created Archon project XYZ with N tasks"

**Status Sync Flow** (Polling):
1. On each Archon task update, background process polls `find_tasks(project_id="...")`
2. Compare Archon task statuses to `tasks.md` checkboxes
3. If Archon task is "done" but `tasks.md` shows `- [ ]`, update to `- [x]`
4. Commit change: `git commit -m "sync: Update tasks.md from Archon status"`

**Archive Flow**:
1. Developer runs `/openspec:archive sync-integration`
2. Read `.archon-project-id` to get Archon project
3. Validate all tasks are "done" via `find_tasks(project_id="...", filter_by="status")`
4. If incomplete: Error "Cannot archive: N Archon tasks still pending"
5. If complete: Archive Archon project, archive OpenSpec change

## Impact

### User Experience
- **Before**: 10 minutes to manually create 15 Archon tasks from `tasks.md`
- **After**: 10 seconds to run `/openspec:import-to-archon <id>`

### Developer Workflow
```bash
# Old workflow
openspec proposal             # Create proposal
# Manually create Archon project
# Manually create 15 tasks one by one
# Work in Archon
# Manually check off tasks.md
# Archive when done

# New workflow
openspec proposal             # Create proposal
/openspec:import-to-archon    # Auto-create project + tasks
# Work in Archon (tasks.md auto-updates!)
/openspec:archive             # Auto-validates Archon completion
```

### Traceability
- Archon task descriptions include: "OpenSpec change: sync-integration"
- OpenSpec `.archon-project-id` file links back to Archon project
- Git history shows automatic `tasks.md` updates

## Risks & Mitigations

**Risk 1: Sync Conflicts**
- Scenario: Developer manually edits `tasks.md` while Archon sync is active
- Mitigation: Sync process checks git diff before updating, prompts on conflict

**Risk 2: Archon API Downtime**
- Scenario: Archon MCP server unavailable during import
- Mitigation: Graceful error handling, allow manual fallback

**Risk 3: Incomplete Task Mapping**
- Scenario: `tasks.md` has subtasks or complex formatting
- Mitigation: Parse only top-level `- [ ]` items, document limitations

## Success Criteria

- [ ] `/openspec:import-to-archon <id>` command creates Archon project + tasks
- [ ] `.archon-project-id` file stores project reference
- [ ] Status sync updates `tasks.md` checkboxes when Archon tasks complete
- [ ] `/openspec:archive` validates all Archon tasks are done
- [ ] Integration tested with sample change (3+ tasks)
- [ ] Documentation updated in `openspec/AGENTS.md`
