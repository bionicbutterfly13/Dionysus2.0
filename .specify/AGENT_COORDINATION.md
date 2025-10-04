# Agent Coordination System

**Per Constitution Article 0: Multi-Agent Collaboration Standards**

This system prevents merge conflicts when multiple Claude Code agents work on the same codebase.

## Quick Start

### 1. Register Agent (MANDATORY at session start)

```bash
.specify/scripts/register_agent.sh
```

This will:
- ✅ Verify you're on a feature branch (not main)
- ✅ Register your agent session
- ✅ Check for other active agents
- ✅ Show locked files to avoid

### 2. Lock File Before Editing (MANDATORY)

```bash
.specify/scripts/lock_file.sh backend/src/services/query.py
```

This will:
- ✅ Check if another agent is editing this file
- ✅ Show recent git history
- ✅ Lock the file for your agent
- ⚠️ Warn if conflicts detected

### 3. Check Conflicts Before Committing (RECOMMENDED)

```bash
.specify/scripts/check_conflicts.sh
```

This will:
- ✅ Fetch latest changes from all branches
- ✅ Check if your files conflict with main
- ✅ Check if your files conflict with other agents
- ✅ Provide recommendations

### 4. Unlock File After Committing

```bash
.specify/scripts/unlock_file.sh backend/src/services/query.py
```

This will:
- ✅ Remove file lock
- ✅ Allow other agents to edit

## Architecture

### Agent Activity Tracking

**File**: `.specify/memory/agent-activity.json`

```json
{
  "agents": [
    {
      "agent_id": "claude-1728063000",
      "branch": "035-clause-phase2-multi-agent",
      "active_files": [
        "backend/src/services/clause/path_navigator.py"
      ],
      "started_at": "2025-10-04T15:30:00Z",
      "last_heartbeat": "2025-10-04T15:45:00Z",
      "status": "active"
    }
  ],
  "last_updated": "2025-10-04T15:45:00Z",
  "schema_version": "1.0.0"
}
```

### Current Agent Session

**File**: `.specify/memory/current-agent.json`

Created by `register_agent.sh`, contains your current session info.

## Branch Strategy

### Feature Branch Naming

Format: `NNN-short-description`

Examples:
- `035-clause-phase2-multi-agent`
- `041-interface-health-checks`
- `042-playwright-ui-validation`

### Workflow

```bash
# 1. Create feature branch
git checkout -b 043-new-feature

# 2. Register agent
.specify/scripts/register_agent.sh

# 3. Lock file before editing
.specify/scripts/lock_file.sh path/to/file.py

# 4. Edit file
# ... make changes ...

# 5. Check for conflicts
.specify/scripts/check_conflicts.sh

# 6. Commit changes
git add path/to/file.py
git commit -m "feat: implement new feature

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"

# 7. Unlock file
.specify/scripts/unlock_file.sh path/to/file.py

# 8. Push to remote
git push -u origin 043-new-feature
```

## Conflict Prevention

### Pre-Edit Checks

Before modifying any file:

```bash
# Check recent git activity
git log --since="1 hour ago" --name-only --oneline -- path/to/file.py

# Check file modification time
ls -lh path/to/file.py

# Check who last modified
git blame -L 1,20 path/to/file.py --date=relative
```

### Shared Resource Notification

Before modifying shared files, notify user:

```
⚠️ SHARED RESOURCE MODIFICATION:
File: backend/requirements.txt
Change: Adding new dependency 'crawl4ai>=0.3.0'
Impact: All agents must pip install after this commit
Proceed? (Y/N)
```

Shared resources include:
- `package.json`
- `requirements.txt`
- Database schemas
- Core interfaces/contracts
- Configuration files

## Merge Conflict Resolution

If conflict detected:

```bash
# 1. Abort merge immediately
git merge --abort

# 2. Report to user
echo "⚠️ MERGE CONFLICT DETECTED in:"
git diff --name-only --diff-filter=U

# 3. Ask for guidance
# Options:
#   A) Manually resolve conflicts
#   B) Abort merge and continue on feature branch
#   C) Coordinate with other agent

# 4. NEVER auto-resolve without user approval
```

## Emergency Procedures

### Another Agent Modified Your File

```bash
# Check what changed
git fetch origin
git diff origin/their-branch -- path/to/file.py

# Options:
# 1. Coordinate with user to merge changes
# 2. Work on different file
# 3. Wait for other agent to finish
```

### Stale Agent Session

If agent died without unlocking files:

```bash
# Manually edit agent-activity.json
# Set agent status to "inactive" or remove from list
jq '.agents |= map(if .agent_id == "stale-id" then .status = "inactive" else . end)' \
  .specify/memory/agent-activity.json > tmp.json
mv tmp.json .specify/memory/agent-activity.json
```

## Best Practices

### ✅ DO

- Register agent at start of every session
- Lock files before editing
- Check for conflicts before committing
- Commit frequently (every logical change)
- Sync with main regularly (every 2 hours)
- Unlock files immediately after commit
- Communicate with user about shared resources

### ❌ DON'T

- Work directly on main/master
- Modify another agent's locked files
- Auto-resolve merge conflicts
- Force push to shared branches
- Modify files without locking
- Leave files locked when done

## Monitoring

### Check Active Agents

```bash
jq '.agents[] | select(.status == "active")' .specify/memory/agent-activity.json
```

### Check Locked Files

```bash
jq -r '.agents[] | select(.status == "active") | .active_files[]' .specify/memory/agent-activity.json
```

### Audit Trail

```bash
# See all agent commits
git log --pretty=format:"%h %s %aN" --grep="Claude Code"

# See what each agent modified
git log --pretty=format:"%h %aN %s" --name-only --grep="Claude Code"
```

## Troubleshooting

### "Agent not registered" error

```bash
# Solution: Register agent
.specify/scripts/register_agent.sh
```

### "File locked by another agent" warning

```bash
# Options:
# 1. Wait for other agent to finish
# 2. Check if agent is still active (heartbeat timestamp)
# 3. If stale (>1 hour), manually unlock
# 4. Coordinate with user
```

### Scripts not executable

```bash
chmod +x .specify/scripts/*.sh
```

## Constitution Reference

All rules in this document are enforced by **Constitution Article 0: Multi-Agent Collaboration Standards**.

Violations will cause:
- Pre-commit hook failures
- Git push rejections
- Agent coordination errors

For full details, see: `.specify/memory/constitution.md`
