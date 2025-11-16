# Import OpenSpec Change to Archon

Import an OpenSpec change proposal into Archon MCP as a project with tasks, enabling automated task tracking and bidirectional sync.

## Usage

```bash
/openspec:import-to-archon <change-id>
```

## What This Does

1. Reads `openspec/changes/<change-id>/proposal.md` and `tasks.md`
2. Creates Archon project with title from proposal
3. Creates Archon tasks for each item in tasks.md
4. Stores Archon project UUID in `.archon-project-id` file
5. Reports success with project ID and task count

## Implementation

You are executing the `/openspec:import-to-archon` command to import an OpenSpec change proposal into Archon MCP.

**Steps**:

1. **Validate change exists**
   - Check that `openspec/changes/<change-id>/` directory exists
   - Check that `proposal.md` and `tasks.md` files exist
   - If not found, error: "Change '<change-id>' not found in openspec/changes/"

2. **Read proposal metadata**
   - Read `openspec/changes/<change-id>/proposal.md`
   - Extract title (first H1 heading, e.g., "# Integrate OpenSpec + Archon...")
   - Extract description from "## What" section

3. **Parse tasks from tasks.md**
   - Read `openspec/changes/<change-id>/tasks.md`
   - Extract all checklist items matching pattern: `- [ ] Task title`
   - Parse phase headers (## Phase N:) to add context
   - Ignore subtasks (indented items)
   - Store task index for sync mapping

4. **Create Archon project**
   - Call `mcp__archon__manage_project("create", title=<title>, description=<description>)`
   - Store project_id from response

5. **Create Archon tasks**
   - For each task from tasks.md (in order):
     - Call `mcp__archon__manage_task("create", project_id=<id>, title=<task>, status="todo", task_order=100-index)`
     - Higher task_order = higher priority (reverse order for top-down execution)
   - Report progress: "Creating task X/Y..."

6. **Store project reference**
   - Write project_id to `openspec/changes/<change-id>/.archon-project-id`
   - Format: Single line containing UUID

7. **Report success**
   ```
   ✅ Import complete!

   Archon Project: <project_id>
   Tasks created: <count>
   Reference stored: .archon-project-id

   Next steps:
   - View tasks: find_tasks(filter_by="project", filter_value="<project_id>")
   - Start working: manage_task("update", task_id="...", status="doing")
   - When done: /openspec:archive <change-id>
   ```

## Error Handling

- **Change not found**: "Error: Change '<change-id>' not found in openspec/changes/"
- **Archon unavailable**: "Error: Cannot connect to Archon MCP server. Check MCP status."
- **Parse errors**: "Warning: Could not parse N tasks from tasks.md. Only top-level checklist items (- [ ] ...) are supported."

## Example

```bash
# Import the OpenSpec + Archon sync proposal itself
/openspec:import-to-archon integrate-openspec-archon-sync

# Output:
# ✅ Import complete!
#
# Archon Project: 550e8400-e29b-41d4-a716-446655440000
# Tasks created: 25
# Reference stored: .archon-project-id
```

## Notes

- Only top-level checklist items (- [ ] ...) are imported
- Subtasks (indented items) are ignored
- Phase headers (## Phase N:) are captured as task metadata
- Task order: First task = highest priority (task_order=100)
- Pre-checked items (- [x] ...) are created with status="done"
