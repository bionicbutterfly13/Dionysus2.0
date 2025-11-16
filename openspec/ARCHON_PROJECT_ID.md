# .archon-project-id File Format

## Purpose

The `.archon-project-id` file establishes a bidirectional link between an OpenSpec change and its corresponding Archon MCP project, enabling automated task tracking and status synchronization.

## Location

```
openspec/changes/<change-id>/.archon-project-id
```

**Example**:
```
openspec/changes/integrate-openspec-archon-sync/.archon-project-id
```

## Format

**Single-line plain text file containing a UUID v4 string**:

```
e500489f-84f1-47ca-81ac-32d2fb9dda33
```

### Specifications

- **Encoding**: UTF-8
- **Content**: Single UUID v4 (36 characters: 8-4-4-4-12 hex digits with hyphens)
- **Whitespace**: Trailing newline optional, leading/trailing whitespace stripped on read
- **Line breaks**: Single line only (no multi-line content)

### Valid Examples

```
550e8400-e29b-41d4-a716-446655440000
```

```
a1b2c3d4-e5f6-7890-abcd-ef1234567890

```
(trailing newline is acceptable)

### Invalid Examples

❌ **Empty file**:
```
(empty)
```

❌ **Multiple lines**:
```
550e8400-e29b-41d4-a716-446655440000
a1b2c3d4-e5f6-7890-abcd-ef1234567890
```

❌ **Invalid UUID format**:
```
not-a-valid-uuid
```

❌ **JSON or other formats**:
```json
{"project_id": "550e8400-e29b-41d4-a716-446655440000"}
```

## Creation

The file is created automatically by `/openspec:import-to-archon`:

```bash
/openspec:import-to-archon integrate-openspec-archon-sync

# Creates:
# openspec/changes/integrate-openspec-archon-sync/.archon-project-id
# Content: e500489f-84f1-47ca-81ac-32d2fb9dda33
```

**Manual creation** (not recommended):
```bash
echo "550e8400-e29b-41d4-a716-446655440000" > openspec/changes/my-change/.archon-project-id
```

## Usage

### Read (Sync Status)

```bash
# /openspec:sync-status reads this file to query Archon
/openspec:sync-status integrate-openspec-archon-sync

# Internally:
# 1. Read .archon-project-id → "e500489f-..."
# 2. Query: mcp__archon__find_tasks(project_id="e500489f-...")
# 3. Update tasks.md based on Archon status
```

### Read (Archive Validation)

```bash
# /openspec:archive reads this file to validate completion
/openspec:archive integrate-openspec-archon-sync

# Internally:
# 1. Read .archon-project-id → "e500489f-..."
# 2. Check: mcp__archon__find_tasks(project_id="e500489f-...", filter_by="status")
# 3. Validate all tasks are "done"
# 4. Optionally archive Archon project
```

### Read (Manual Query)

```bash
# Read file content
PROJECT_ID=$(cat openspec/changes/my-change/.archon-project-id)

# Query Archon directly
mcp__archon__find_tasks(project_id="$PROJECT_ID")
```

## Lifecycle

### 1. Creation (Import Phase)

```
/openspec:import-to-archon my-feature
    ↓
Creates Archon project (UUID: 550e8400-...)
    ↓
Writes .archon-project-id file
    ↓
File content: "550e8400-e29b-41d4-a716-446655440000"
```

### 2. Active Use (Development Phase)

```
.archon-project-id exists → Link active
    ↓
/openspec:sync-status reads file → Syncs task status
    ↓
tasks.md updated with [x] checkboxes for completed tasks
```

### 3. Archive Phase

```
/openspec:archive my-feature
    ↓
Reads .archon-project-id → Validates Archon completion
    ↓
Archives OpenSpec change → Moves to archive/2025-11-16-my-feature/
    ↓
.archon-project-id moves with change → Preserved in archive
```

### 4. Post-Archive

```
archive/2025-11-16-my-feature/.archon-project-id
```

File preserved for historical reference, showing which Archon project tracked this change.

## Error Handling

### Missing File

**Symptom**:
```bash
/openspec:sync-status my-feature

# Error: No .archon-project-id found for 'my-feature'.
# Run /openspec:import-to-archon my-feature first.
```

**Resolution**:
```bash
/openspec:import-to-archon my-feature
```

### Invalid UUID

**Symptom**:
```bash
# File contains: "invalid-id"

/openspec:sync-status my-feature

# Error: Invalid project UUID in .archon-project-id
```

**Resolution**:
- Delete file and re-run import, or
- Manually fix UUID to valid format

### Orphaned Link

**Symptom**:
```bash
# File exists, but Archon project was deleted

/openspec:sync-status my-feature

# Error: Archon project e500489f-... not found
```

**Resolution**:
- Delete `.archon-project-id` file, or
- Re-import to create new Archon project

## Security & Privacy

- **No sensitive data**: File contains only a UUID (project identifier)
- **Safe to commit**: Should be committed to git with the change
- **Safe to share**: No credentials or secrets
- **Read-only after creation**: Typically not modified manually

## Git Tracking

### Recommended `.gitignore` (none needed)

The file **should be committed** to version control:

```gitignore
# Do NOT ignore .archon-project-id
# ❌ **/.archon-project-id
```

### Commit Pattern

```bash
git add openspec/changes/my-feature/.archon-project-id
git commit -m "chore: link OpenSpec change to Archon project"
```

Typically committed automatically during `/openspec:import-to-archon`.

## Troubleshooting

### File Exists But Sync Fails

**Check file format**:
```bash
cat openspec/changes/my-feature/.archon-project-id

# Should show single line UUID like:
# e500489f-84f1-47ca-81ac-32d2fb9dda33
```

**Validate UUID**:
```bash
# Valid format: 8-4-4-4-12 hex digits
# Example: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
```

### Multiple Lines in File

**Symptom**: Only first line is read, rest ignored

**Resolution**:
```bash
# Keep only first line
head -n 1 openspec/changes/my-feature/.archon-project-id > temp
mv temp openspec/changes/my-feature/.archon-project-id
```

### Empty File

**Resolution**:
```bash
# Delete and re-import
rm openspec/changes/my-feature/.archon-project-id
/openspec:import-to-archon my-feature
```

## Related Files

- `proposal.md`: Change description (why, what, impact)
- `tasks.md`: Implementation checklist (synced with Archon)
- `design.md`: Technical decisions (optional)
- `specs/`: Spec deltas (requirements changes)

## See Also

- [OpenSpec + Archon Integration Guide](../AGENTS.md#archon-integration-task-management)
- [Sync Status Command](./.claude/commands/openspec/sync-status.md)
- [Import Command](./.claude/commands/openspec/import-to-archon.md)
- [Archive Command](./.claude/commands/openspec/archive.md)
