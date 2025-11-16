# Design: OpenSpec + Archon Integration

## Architecture Overview

### System Components

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        OpenSpec (File System)                           │
│ ┌─────────────────────────────────────────────────────────────────────┐ │
│ │ openspec/changes/<change-id>/                                       │ │
│ │   ├── proposal.md          (What/Why/How)                          │ │
│ │   ├── tasks.md             (- [ ] Checklist) ◄──── Sync Updates    │ │
│ │   ├── .archon-project-id   (Reference)                             │ │
│ │   └── specs/               (Requirement deltas)                    │ │
│ └─────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ Import
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Integration Layer (This Change)                      │
│ ┌──────────────────┐   ┌──────────────────┐   ┌────────────────────┐  │
│ │ Import Command   │   │ Status Sync      │   │ Archive Validator  │  │
│ │ /import-archon   │   │ (Polling/Webhook)│   │ /archive enhanced  │  │
│ └──────────────────┘   └──────────────────┘   └────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ API Calls
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Archon MCP Server                               │
│ ┌─────────────────────────────────────────────────────────────────────┐ │
│ │ Projects                                                            │ │
│ │   ├── id: uuid                                                      │ │
│ │   ├── title: "Integrate OpenSpec + Archon..."                      │ │
│ │   ├── description: from proposal.md                                │ │
│ │   └── metadata: {openspec_change_id: "integrate-openspec-..."}    │ │
│ │                                                                     │ │
│ │ Tasks                                                               │ │
│ │   ├── [{id, title, status: "todo|doing|review|done", ...}]        │ │
│ │   └── metadata: {openspec_task_index: 0}                           │ │
│ └─────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

## Component Designs

### 1. Import Command (`/openspec:import-to-archon`)

**Location**: `.claude/commands/openspec/import-to-archon.md`

**Pseudocode**:
```python
def import_to_archon(change_id: str):
    # 1. Validate change exists
    change_path = f"openspec/changes/{change_id}"
    if not exists(change_path):
        error(f"Change {change_id} not found")

    # 2. Read proposal metadata
    proposal = read_markdown(f"{change_path}/proposal.md")
    title = extract_title(proposal)  # First H1
    description = extract_section(proposal, "What")

    # 3. Parse tasks from tasks.md
    tasks_md = read_markdown(f"{change_path}/tasks.md")
    tasks = parse_checklist(tasks_md)  # Extract all - [ ] items

    # 4. Create Archon project
    project = mcp__archon__manage_project(
        action="create",
        title=title,
        description=description,
        # Store OpenSpec reference
        data={"openspec_change_id": change_id}
    )
    project_id = project["id"]

    # 5. Create Archon tasks
    for idx, task in enumerate(tasks):
        mcp__archon__manage_task(
            action="create",
            project_id=project_id,
            title=task.title,
            description=task.description or "",
            status="todo",
            task_order=100 - idx,  # Reverse order for priority
            # Store task index for sync
            data={"openspec_task_index": idx}
        )

    # 6. Store Archon project reference
    write_file(
        f"{change_path}/.archon-project-id",
        project_id
    )

    # 7. Report success
    print(f"✅ Created Archon project {project_id}")
    print(f"   - {len(tasks)} tasks imported")
    print(f"   - Reference stored in .archon-project-id")
```

**Task Parsing Logic**:
```python
def parse_checklist(markdown: str) -> List[Task]:
    """Extract tasks from markdown checklist."""
    tasks = []
    lines = markdown.split('\n')

    for line in lines:
        # Match: - [ ] Task title
        if match := re.match(r'^-\s+\[( |x)\]\s+(.+)$', line):
            checked = match.group(1) == 'x'
            title = match.group(2).strip()
            tasks.append(Task(
                title=title,
                completed=checked,
                index=len(tasks)
            ))

    return tasks
```

### 2. Status Sync Service

**Trigger Options**:

**Option A: Polling** (Recommended for MVP)
- Every 30 seconds, check Archon tasks for changes
- Compare to last known state, update `tasks.md` on diff
- Pros: Simple, no server changes needed
- Cons: Latency up to 30 seconds

**Option B: Webhook** (Future Enhancement)
- Archon MCP sends webhook on task status change
- Immediate update to `tasks.md`
- Pros: Real-time sync
- Cons: Requires Archon server modification

**Polling Implementation**:
```python
async def sync_status_loop(change_id: str):
    """Continuously sync Archon task status to tasks.md."""
    change_path = f"openspec/changes/{change_id}"
    project_id = read_file(f"{change_path}/.archon-project-id")

    while True:
        # 1. Get current Archon task statuses
        archon_tasks = mcp__archon__find_tasks(
            filter_by="project",
            filter_value=project_id
        )

        # 2. Read current tasks.md
        tasks_md_content = read_file(f"{change_path}/tasks.md")

        # 3. Check for status changes
        updates_needed = []
        for task in archon_tasks:
            idx = task["data"]["openspec_task_index"]
            if task["status"] == "done":
                # Check if tasks.md has [ ] at this index
                if not is_checked_at_index(tasks_md_content, idx):
                    updates_needed.append(idx)

        # 4. Update tasks.md if needed
        if updates_needed:
            new_content = update_checkboxes(
                tasks_md_content,
                updates_needed
            )
            write_file(f"{change_path}/tasks.md", new_content)

            # 5. Git commit
            run_command([
                "git", "add", f"{change_path}/tasks.md",
                "&&", "git", "commit", "-m",
                f"sync: Update tasks.md from Archon (change: {change_id})"
            ])

        # 6. Sleep before next check
        await asyncio.sleep(30)
```

**Checkbox Update Logic**:
```python
def update_checkboxes(markdown: str, indices: List[int]) -> str:
    """Update checkboxes at given indices to [x]."""
    lines = markdown.split('\n')
    task_idx = 0

    for i, line in enumerate(lines):
        if re.match(r'^-\s+\[( |x)\]', line):
            if task_idx in indices:
                # Replace [ ] with [x]
                lines[i] = re.sub(r'\[ \]', '[x]', line)
            task_idx += 1

    return '\n'.join(lines)
```

### 3. Archive Validator (Enhanced `/openspec:archive`)

**Integration Point**: Modify existing archive command

**Pseudocode**:
```python
def archive_change(change_id: str):
    change_path = f"openspec/changes/{change_id}"

    # NEW: Check for Archon integration
    archon_id_file = f"{change_path}/.archon-project-id"
    if exists(archon_id_file):
        project_id = read_file(archon_id_file)

        # Validate all tasks are done
        tasks = mcp__archon__find_tasks(
            filter_by="project",
            filter_value=project_id
        )

        incomplete = [t for t in tasks if t["status"] != "done"]
        if incomplete:
            error(f"""
            Cannot archive: {len(incomplete)} Archon tasks still pending

            Incomplete tasks:
            {format_task_list(incomplete)}

            Complete all tasks in Archon before archiving.
            """)

        # Archive Archon project
        mcp__archon__manage_project(
            action="update",
            project_id=project_id,
            archived=True
        )

    # Continue with normal archive process
    archive_openspec_change(change_id)
```

## Data Structures

### .archon-project-id File Format
```
550e8400-e29b-41d4-a716-446655440000
```
Simple UUID, one line, no metadata.

### Archon Project Metadata
```json
{
  "openspec_change_id": "integrate-openspec-archon-sync",
  "imported_at": "2025-11-15T02:30:00Z",
  "imported_by": "Claude Code"
}
```

### Archon Task Metadata
```json
{
  "openspec_task_index": 0,
  "openspec_section": "Phase 1: Import Command"
}
```

## Error Handling

### Archon API Unavailable
```
❌ ERROR: Cannot connect to Archon MCP server

Attempted operation: Import OpenSpec change to Archon
Change ID: integrate-openspec-archon-sync

Please ensure Archon MCP server is running:
  1. Check MCP connection: claude mcp list
  2. Restart Archon if needed
  3. Retry: /openspec:import-to-archon integrate-openspec-archon-sync

Fallback: You can manually create Archon project and tasks.
```

### Incomplete Tasks on Archive
```
❌ ERROR: Cannot archive change with incomplete Archon tasks

Change: integrate-openspec-archon-sync
Archon Project: 550e8400-e29b-41d4-a716-446655440000

Incomplete tasks (3):
  [ ] Implement status sync poller
  [ ] Add git commit for automatic updates
  [ ] Test sync with completing tasks

Complete all tasks in Archon before archiving, or remove .archon-project-id
to skip validation.
```

### Parse Errors
```
⚠️  WARNING: Could not parse all tasks from tasks.md

Change: integrate-openspec-archon-sync
Parsed: 18 tasks
Unparsed lines: 2 (subtasks or invalid format)

Only top-level checklist items (- [ ] ...) are imported to Archon.
Subtasks and complex formatting are not supported.

Continue import? (y/n)
```

## Performance Considerations

### Import Performance
- **Target**: < 5 seconds for 20 tasks
- **Bottleneck**: Sequential Archon API calls
- **Optimization**: Batch task creation if Archon supports bulk API

### Sync Performance
- **Polling Interval**: 30 seconds (configurable)
- **File I/O**: Read tasks.md once per cycle
- **Git Commits**: Amortize with batch updates (update every N changes)

### Archive Performance
- **Target**: < 2 seconds validation
- **API Calls**: 1 for task list, 1 for project archive
- **Optimization**: Cache task statuses during sync

## Testing Strategy

### Unit Tests
- Test task parser with various markdown formats
- Test checkbox updater with edge cases (mixed checked/unchecked)
- Test .archon-project-id file I/O

### Integration Tests
1. Create sample OpenSpec change with 5 tasks
2. Run `/openspec:import-to-archon`
3. Verify Archon project + tasks created
4. Complete 2 tasks in Archon
5. Verify tasks.md updated with [x]
6. Run `/openspec:archive`
7. Verify validation passes

### Manual QA
- Test with real OpenSpec change (this change!)
- Test error handling (Archon offline, parse errors)
- Test sync conflict handling (manual edits during sync)

## Migration Path

### Existing Changes (Pre-Integration)
- No automatic migration
- Developers can optionally run `/openspec:import-to-archon <id>` on existing changes
- `.archon-project-id` file is optional marker for integration

### New Changes (Post-Integration)
- Workflow in `openspec/AGENTS.md` updated to recommend import
- Developers choose: import to Archon or work without integration
- Both workflows supported

## Future Enhancements

1. **Webhook-Based Sync**: Real-time updates instead of polling
2. **Subtask Support**: Parse and import nested checklist items
3. **Bidirectional Task Creation**: Create tasks in Archon that add to tasks.md
4. **Progress Dashboard**: Web UI showing OpenSpec + Archon sync status
5. **Conflict Resolution UI**: Interactive tool for resolving sync conflicts
