# OpenSpec + Archon Integration: Troubleshooting Guide

Common problems and solutions for the OpenSpec + Archon integrated workflow.

## Quick Diagnostics

```bash
# Check if Archon MCP is available
mcp__archon__health_check()

# List Archon projects
mcp__archon__find_projects()

# Check OpenSpec changes
openspec list

# Validate specific change
openspec validate <change-id> --strict
```

---

## Import Issues

### Error: "No .archon-project-id found"

**Symptom**:
```
/openspec:sync-status my-feature

Error: No .archon-project-id found for 'my-feature'.
Run /openspec:import-to-archon my-feature first.
```

**Cause**: Change was never imported to Archon

**Solution**:
```bash
/openspec:import-to-archon my-feature
```

---

### Error: "Change not found in openspec/changes/"

**Symptom**:
```
/openspec:import-to-archon my-feature

Error: Change 'my-feature' not found in openspec/changes/
```

**Cause**: Change directory doesn't exist or wrong change ID

**Solution**:
```bash
# List existing changes
openspec list

# Verify directory exists
ls openspec/changes/

# Use correct change ID (exact match required)
/openspec:import-to-archon correct-change-id
```

---

### Error: "Cannot connect to Archon MCP"

**Symptom**:
```
Error: Cannot connect to Archon MCP server. Check MCP status.
```

**Cause**: Archon MCP server is not running or not configured

**Solution**:
```bash
# Check MCP server status
mcp__archon__health_check()

# If not available, check Claude Code MCP configuration
# Restart Claude Code if needed
```

---

### Tasks Not Importing Correctly

**Symptom**: Some tasks from tasks.md not appearing in Archon

**Cause**: Only top-level checklist items are imported (subtasks ignored)

**Solution**:
- Ensure tasks use format: `- [ ] Task title` (top-level)
- Subtasks (indented items) are not imported
- Flatten task hierarchy for import

**Example**:
```markdown
## Phase 1
- [ ] Main task 1          ← Imported ✓
- [ ] Main task 2          ← Imported ✓
  - [ ] Subtask 2.1        ← Ignored ✗
  - [ ] Subtask 2.2        ← Ignored ✗
```

---

## Sync Issues

### Tasks Not Matching

**Symptom**:
```
/openspec:sync-status my-feature

✅ Sync complete!
Tasks matched: 8/12 tasks
```

**Cause**: Task titles changed after import (similarity < 85%)

**Solution**:
- Keep task titles stable between Archon and tasks.md
- Minor edits OK (85% similarity threshold)
- Major rewording breaks matching

**Example**:
```markdown
# Original (tasks.md)
- [ ] Create user authentication

# Archon task title (after import)
"Create user authentication"

# ✓ Match (100% similar)
- [ ] Create user authentication system

# ✗ No match (75% similar, below 85% threshold)
- [ ] Implement OAuth 2.0 login
```

---

### Checkboxes Not Updating

**Symptom**: Archon tasks are "done" but tasks.md checkboxes still `[ ]`

**Cause**: Sync command not run or task matching failed

**Solution**:
```bash
# Run sync manually
/openspec:sync-status my-feature

# Check output for match results
# Look for: "Tasks matched: M/N tasks"

# If M < N, check task titles for similarity
```

---

### Conflict: Manual vs Automatic Updates

**Symptom**: Manually marked checkbox `[x]` but Archon shows "todo"

**Behavior**: Manual checkbox takes precedence (won't downgrade)

**Resolution**: This is expected - conflict resolution favors manual completion

---

### Sync Shows No Updates

**Symptom**:
```
✅ Already in sync!
Updates applied: 0
```

**Cause**: Archon and tasks.md already match (no changes needed)

**Resolution**: No action needed - this is success state

---

## Archive Issues

### Error: "Archon project has N incomplete tasks"

**Symptom**:
```
/openspec:archive my-feature

⚠️ Archon project has 5 incomplete tasks (3 todo, 2 doing)
```

**Cause**: Not all Archon tasks marked "done"

**Solutions**:

**Option 1: Complete remaining tasks** (recommended)
```bash
# List incomplete tasks
mcp__archon__find_tasks(project_id="...", filter_by="status", filter_value="todo")

# Complete each task
# (implement feature, then mark done)
```

**Option 2: Sync and verify**
```bash
# Sync to ensure tasks.md is current
/openspec:sync-status my-feature

# Check if tasks were actually done but not synced
# Re-run archive after sync
```

**Option 3: Force archive** (not recommended)
```bash
# When prompted "Archive anyway?", answer: yes
# Warning: Leaves orphaned tasks in Archon
```

---

### Archive Succeeds But Specs Not Updated

**Symptom**: Archive completes but `openspec/specs/` unchanged

**Cause**: Used `--skip-specs` flag or no spec deltas in change

**Solution**:
- Remove `--skip-specs` flag
- Verify change has spec deltas in `changes/<id>/specs/`
- Run `openspec validate <id> --strict` to check deltas

---

### Can't Archive: "Change not found"

**Symptom**:
```
/openspec:archive my-feature

Error: Change 'my-feature' not found
```

**Cause**: Change already archived or wrong change ID

**Solution**:
```bash
# Check active changes
openspec list

# Check archive
ls openspec/changes/archive/

# If already archived, no action needed
```

---

## File Format Issues

### Invalid .archon-project-id

**Symptom**:
```
Error: Invalid project UUID in .archon-project-id
```

**Cause**: File contains invalid UUID or extra content

**Solution**:
```bash
# Check file content
cat openspec/changes/my-feature/.archon-project-id

# Should be single line UUID like:
# e500489f-84f1-47ca-81ac-32d2fb9dda33

# Fix if needed
echo "valid-uuid-here" > openspec/changes/my-feature/.archon-project-id
```

See [ARCHON_PROJECT_ID.md](./ARCHON_PROJECT_ID.md) for format specification.

---

### Tasks.md Parse Errors

**Symptom**: Tasks not importing or syncing correctly

**Cause**: Invalid markdown checkbox format

**Solution**:
```markdown
# ✓ CORRECT format
- [ ] Task title
- [x] Completed task
- [-] In progress task
- [~] Under review task

# ✗ WRONG format
* [ ] Task with asterisk
- [ ]Task without space
-[ ] Missing space after dash
- [X] Uppercase X (use lowercase)
```

---

## Performance Issues

### Slow Sync

**Symptom**: `/openspec:sync-status` takes >5 seconds

**Cause**: Large number of tasks or slow MCP connection

**Solution**:
- Normal for 50+ tasks
- Check Archon MCP server health
- Consider breaking large changes into smaller ones

---

### Slow Import

**Symptom**: `/openspec:import-to-archon` takes >10 seconds

**Cause**: Creating many Archon tasks

**Solution**:
- Normal for 20+ tasks (sequential creation)
- Wait for completion (progress shown)
- Don't interrupt process

---

## Configuration Issues

### Sync Behavior Not As Expected

**Symptom**: Checkboxes not updating as configured

**Cause**: .openspec.config.json not loaded or invalid

**Solution**:
```bash
# Check config file exists
cat .openspec.config.json

# Validate JSON
python -m json.tool .openspec.config.json

# Default values if file missing:
# - similarity_threshold: 0.85
# - conflict_resolution: "archon_wins"
# - auto_commit: true
```

See [.openspec.config.json](./.openspec.config.json) for schema.

---

## Data Recovery

### Orphaned Archon Project

**Symptom**: Archon project exists but no .archon-project-id

**Cause**: File deleted or change created before integration

**Solution**:
```bash
# Option 1: Manually create link
# Find project ID
mcp__archon__find_projects(query="my-feature")

# Write to file
echo "project-uuid-here" > openspec/changes/my-feature/.archon-project-id

# Option 2: Re-import (creates new project)
/openspec:import-to-archon my-feature
```

---

### Lost Progress After Sync

**Symptom**: Checkboxes reverted after sync

**Cause**: Archon status took precedence (by design)

**Solution**:
- Archon is source of truth
- Update task status in Archon, not tasks.md manually
- Use conflict resolution setting if needed

---

## Getting Help

### Verbose Output

```bash
# Enable detailed logging (if supported)
export DEBUG=true
/openspec:sync-status my-feature
```

### Check Logs

```bash
# OpenSpec validation
openspec validate <change-id> --strict

# Archon MCP status
mcp__archon__session_info()
mcp__archon__health_check()

# Show change details
openspec show <change-id>
```

### Report Issue

When reporting problems, include:
1. Change ID
2. Command run
3. Error message (full text)
4. Output of `openspec show <change-id>`
5. Archon project ID (from .archon-project-id)

---

## Related Documentation

- [Archon Integration Guide](./AGENTS.md#archon-integration-task-management)
- [.archon-project-id Format](./ARCHON_PROJECT_ID.md)
- [Workflow Test Results](../openspec/changes/integrate-openspec-archon-sync/WORKFLOW_TEST.md)
- [Archive Command Validation](../.claude/commands/openspec/archive.md#archon-validation-error-messages)
