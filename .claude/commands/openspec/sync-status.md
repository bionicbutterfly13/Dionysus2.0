# Sync Archon Task Status to OpenSpec

Sync task completion status from Archon MCP back to OpenSpec tasks.md checkboxes, enabling bidirectional task tracking.

## Usage

```bash
/openspec:sync-status <change-id>
```

## What This Does

1. Reads `.archon-project-id` from change directory
2. Queries Archon MCP for all task statuses
3. Parses current `tasks.md` checkboxes
4. Updates checkboxes based on Archon task status
5. Commits changes to git with completion summary

## Implementation

You are executing the `/openspec:sync-status` command to sync Archon task status back to OpenSpec tasks.md.

**Steps**:

1. **Validate change exists**
   - Check that `openspec/changes/<change-id>/` directory exists
   - Check that `.archon-project-id` file exists
   - If not found, error: "No Archon project linked. Run /openspec:import-to-archon <change-id> first."

2. **Read Archon project ID**
   - Read `openspec/changes/<change-id>/.archon-project-id`
   - Extract project UUID (single line)

3. **Fetch Archon tasks**
   - Call `mcp__archon__find_tasks(project_id="<uuid>", per_page=100)`
   - Store all tasks with their status (todo/doing/review/done)
   - Report: "Found N tasks in Archon"

4. **Parse tasks.md**
   - Read `openspec/changes/<change-id>/tasks.md`
   - Extract all checklist items: `- [ ] Task title`, `- [x] Task title`, etc.
   - Parse phase headers (## Phase N:) for context
   - Store: task title, current checkbox state (' ', 'x', '~', '-'), line number

5. **Match tasks**
   - For each tasks.md entry, find matching Archon task by title
   - Use fuzzy matching (85%+ similarity) to handle minor edits
   - Example: "Create sync mechanism" matches "Design sync mechanism" at 87% similarity
   - Report: "Matched M/N tasks"

6. **Compute updates**
   - Map Archon status to checkbox symbols:
     - `done` → `[x]`
     - `review` → `[~]` (optional)
     - `doing` → `[-]` (optional)
     - `todo` → `[ ]`
   - Identify differences between Archon status and current checkbox
   - **Conflict resolution**: Never downgrade `[x]` to `[ ]` (manual completion wins)
   - Report changes: "- [ ] → [x] Task title"

7. **Apply updates**
   - For each update, modify the corresponding line in tasks.md
   - Preserve task title and formatting, only change checkbox
   - Write updated content back to file
   - Report: "Updated N checkboxes"

8. **Calculate completion**
   - Count total tasks and completed tasks (`[x]`)
   - Calculate percentage: `(completed / total) * 100`
   - Report: "Completion: X/Y tasks (Z%)"

9. **Git commit**
   - If updates were made:
     - Stage: `git add openspec/changes/<change-id>/tasks.md`
     - Commit: `git commit -m "chore: sync task status from Archon [X/Y complete]"`
     - Report: "✓ Changes committed"
   - If no updates: "✓ Already in sync!"

10. **Report success**
    ```
    ✅ Sync complete!

    Archon Project: <project_id>
    Tasks matched: M/N
    Updates applied: K
    Completion: X/Y tasks (Z%)

    Next steps:
    - Review changes: git diff openspec/changes/<change-id>/tasks.md
    - Continue working: manage_task("update", task_id="...", status="doing")
    - When all done: /openspec:archive <change-id>
    ```

## Checkbox Symbols

- `[ ]` - Not started (todo)
- `[-]` - In progress (doing)
- `[~]` - Under review (review)
- `[x]` - Completed (done)

## Fuzzy Matching Logic

Tasks are matched by title similarity using SequenceMatcher:
- Match threshold: 85% similarity
- Case-insensitive comparison
- Handles minor edits like typo fixes or rephrasing

Example matches:
- "Create status sync mechanism" ↔ "Design status sync mechanism" (90%)
- "Add error handling" ↔ "Add error handling logic" (92%)

## Conflict Resolution

**Archon wins**: If Archon task is "done" and checkbox is `[ ]`, update to `[x]`

**Manual override**: If checkbox is `[x]` and Archon task is "todo", keep `[x]`
- Reason: User may have manually marked as complete
- Log warning: "Warning: Task '<title>' marked done locally but todo in Archon"

## Error Handling

- **Change not found**: "Error: Change '<change-id>' not found in openspec/changes/"
- **No Archon link**: "Error: No .archon-project-id found. Run /openspec:import-to-archon first."
- **Archon unavailable**: "Error: Cannot connect to Archon MCP server. Check MCP status."
- **Parse errors**: "Warning: Could not parse tasks.md. Ensure valid markdown format."
- **No matches**: "Warning: No tasks matched between Archon and tasks.md. Check for title changes."

## Example

```bash
# Sync status for the integration project
/openspec:sync-status integrate-openspec-archon-sync

# Output:
# Syncing Archon status for: integrate-openspec-archon-sync
# ------------------------------------------------------------
# 1. Reading .archon-project-id...
#    Project ID: e500489f-84f1-47ca-81ac-32d2fb9dda33
#
# 2. Fetching Archon tasks...
#    Found 26 tasks in Archon
#
# 3. Parsing tasks.md...
#    Found 26 tasks in tasks.md
#
# 4. Matching tasks...
#    Matched 26/26 tasks
#
# 5. Computing updates...
#    2 checkboxes need updating
#
#    Updates:
#    - [ ] → [x] Design status sync mechanism
#    - [ ] → [x] Implement Archon task status poller
#
# 6. Applying updates to tasks.md...
#    ✓ tasks.md updated
#
# 7. Git commit...
#    ✓ Changes committed
#
# ============================================================
# ✅ Sync complete!
#
# Completion: 9/26 tasks (35%)
# Updates applied: 2
#
# Next steps:
#   - Review changes: git diff openspec/changes/integrate-openspec-archon-sync/tasks.md
#   - Continue working: manage_task("update", task_id="...", status="doing")
```

## Notes

- Sync is one-way: Archon → OpenSpec (for now)
- Can be run multiple times safely (idempotent)
- Use before `/openspec:archive` to ensure tasks.md reflects actual completion
- Future: Add automatic periodic polling or webhook triggers
