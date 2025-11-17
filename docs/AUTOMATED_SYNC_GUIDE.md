# OpenSpec + Archon Automated Sync Guide

## Overview

Automated bidirectional sync between OpenSpec (file-based) and Archon (MCP API-based) eliminates manual `/openspec:sync-status` calls and keeps tasks.md always current.

## Architecture

```
┌────────────────────────────────────────────────────────┐
│ Background Sync Options                               │
├────────────────────────────────────────────────────────┤
│ Option 1: Daemon (Continuous Polling)                 │
│   - Runs in background                                │
│   - Polls every 30s (configurable)                    │
│   - Requires: Archon MCP HTTP access                  │
│   - Best for: Active development                      │
│                                                        │
│ Option 2: Cron/Scheduled (Periodic Sync)              │
│   - Runs on schedule (e.g., every minute)             │
│   - Requires: Archon MCP HTTP access                  │
│   - Best for: Lower overhead, less frequent updates   │
│                                                        │
│ Option 3: Git Hooks (On-Demand Sync)                  │
│   - Syncs before git operations                       │
│   - Requires: Claude Code session active              │
│   - Best for: Ensuring sync before commit/push        │
└────────────────────────────────────────────────────────┘
```

## Option 1: Background Daemon (Recommended for Active Development)

### Prerequisites

- ✅ Archon MCP server running and accessible via HTTP
- ✅ OpenSpec changes created with `/openspec:import-to-archon`
- ✅ `.openspec.config.json` configured

### Configuration

Edit `.openspec.config.json`:

```json
{
  "archon_sync": {
    "enabled": true,
    "sync_interval_seconds": 30,
    "auto_commit": true,
    "conflict_resolution": "archon_wins"
  }
}
```

### Usage

```bash
# Start daemon in background
python scripts/openspec_sync_daemon.py start

# Check status
python scripts/openspec_sync_daemon.py status

# Stop daemon
python scripts/openspec_sync_daemon.py stop

# Run in foreground (debug mode)
python scripts/openspec_sync_daemon.py run
```

### Daemon Features

- **Auto-discovery**: Finds all changes with `.archon-project-id`
- **Smart polling**: Only syncs when interval elapsed
- **Error handling**: Continues on failure, logs errors
- **Graceful shutdown**: SIGTERM/SIGINT support
- **Logging**: All activity logged to `.openspec-sync-daemon.log`

### Checking Logs

```bash
# View daemon log
tail -f .openspec-sync-daemon.log

# Recent activity
python scripts/openspec_sync_daemon.py status
```

## Option 2: Cron/Scheduled Sync (Lower Overhead)

For less frequent syncing, use cron or launchd:

### macOS (launchd)

Create `~/Library/LaunchAgents/com.openspec.sync.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.openspec.sync</string>

    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/python3</string>
        <string>/path/to/Dionysus-2.0/scripts/sync_openspec_archon_status.py</string>
        <string>--all</string>
    </array>

    <key>WorkingDirectory</key>
    <string>/path/to/Dionysus-2.0</string>

    <key>StartInterval</key>
    <integer>60</integer>  <!-- Run every 60 seconds -->

    <key>StandardOutPath</key>
    <string>/tmp/openspec-sync.log</string>

    <key>StandardErrorPath</key>
    <string>/tmp/openspec-sync-error.log</string>
</dict>
</plist>
```

Load it:

```bash
launchctl load ~/Library/LaunchAgents/com.openspec.sync.plist
launchctl start com.openspec.sync
```

### Linux (cron)

Add to crontab:

```bash
# Edit crontab
crontab -e

# Add entry (runs every minute)
* * * * * cd /path/to/Dionysus-2.0 && python3 scripts/sync_all_changes.sh >> /tmp/openspec-sync.log 2>&1
```

## Option 3: Git Hooks (Manual but Reliable)

Sync before git operations to ensure tasks.md is current:

### Pre-commit Hook

Create `.git/hooks/pre-commit`:

```bash
#!/bin/bash

# Sync all OpenSpec changes before commit
cd "$(git rev-parse --show-toplevel)"

echo "Syncing OpenSpec changes from Archon..."
python scripts/sync_all_changes.sh

# If sync made changes, add them to commit
git add openspec/changes/*/tasks.md 2>/dev/null || true

echo "✓ Sync complete"
```

Make executable:

```bash
chmod +x .git/hooks/pre-commit
```

## Helper Script: Sync All Changes

Create `scripts/sync_all_changes.sh` for bulk syncing:

```bash
#!/bin/bash
# Sync all changes with Archon integration

set -e

CHANGES_DIR="openspec/changes"

if [ ! -d "$CHANGES_DIR" ]; then
    echo "✗ No openspec/changes directory found"
    exit 1
fi

SYNCED=0
FAILED=0

for change_dir in "$CHANGES_DIR"/*; do
    if [ ! -d "$change_dir" ]; then
        continue
    fi

    change_id=$(basename "$change_dir")

    # Check for Archon integration
    if [ ! -f "$change_dir/.archon-project-id" ]; then
        continue
    fi

    echo "Syncing: $change_id"

    if python scripts/sync_openspec_archon_status.py "$change_id" --quiet; then
        ((SYNCED++))
    else
        echo "  ✗ Failed"
        ((FAILED++))
    fi
done

echo ""
echo "Results: $SYNCED synced, $FAILED failed"

exit $FAILED
```

Usage:

```bash
# Sync all changes manually
./scripts/sync_all_changes.sh

# Or via Python
python scripts/sync_openspec_archon_status.py --all
```

## Troubleshooting

### Daemon Won't Start

**Error**: `Cannot connect to Archon MCP`

**Solutions**:
1. Check Archon MCP is running:
   ```bash
   curl http://localhost:8051/health
   ```

2. Set correct URL in environment:
   ```bash
   export ARCHON_MCP_URL=http://localhost:8051
   ```

3. Check MCP server configuration in Claude Code

### Sync Not Happening

**Check**:
1. Daemon status: `python scripts/openspec_sync_daemon.py status`
2. Log file: `tail -f .openspec-sync-daemon.log`
3. Config enabled: `.openspec.config.json` → `archon_sync.enabled: true`
4. `.archon-project-id` exists in change directory

### Tasks Not Updating

**Check**:
1. Archon task status is actually changing (use `find_tasks()`)
2. Task titles match between Archon and tasks.md (85% similarity required)
3. Manual edits in tasks.md might override (see conflict resolution)

### Conflict Resolution

From `.openspec.config.json`:

- `archon_wins`: Archon status overwrites tasks.md (default)
- `manual_wins`: Manual `[x]` checkboxes preserved
- `prompt`: Ask before overwriting (not yet implemented)

**Exception**: Never downgrades `[x]` to `[ ]` (manual completion wins)

## Performance Considerations

### Daemon vs Cron

| Feature | Daemon | Cron/Scheduled |
|---------|--------|----------------|
| Latency | 30s (configurable) | 1-5 min typical |
| CPU Usage | Constant low | Periodic spikes |
| Memory | ~20 MB resident | ~5 MB per run |
| Best For | Active development | Background maintenance |

### Optimizations

1. **Increase sync interval**: For less active projects
   ```json
   "sync_interval_seconds": 120  // 2 minutes instead of 30s
   ```

2. **Skip quiet changes**: Daemon only syncs changes with updates

3. **Batch syncing**: Sync multiple changes in parallel (future enhancement)

## Future Enhancements

### Phase 1 (Current)
- ✅ Daemon implementation
- ✅ Config-driven sync interval
- ✅ Auto-discovery of changes
- ✅ Error handling and logging

### Phase 2 (Next)
- ⏳ Webhook support (instant sync on Archon task update)
- ⏳ `--all` flag for bulk sync script
- ⏳ Integration tests

### Phase 3 (Future)
- ⏳ Bidirectional sync (tasks.md → Archon)
- ⏳ Interactive conflict resolution
- ⏳ Parallel change syncing
- ⏳ Systemd service file for Linux

## Configuration Reference

### `.openspec.config.json`

```json
{
  "archon_sync": {
    "enabled": true,                  // Enable/disable sync
    "sync_interval_seconds": 30,      // Polling interval
    "auto_sync_on_archive": true,     // Sync before /openspec:archive
    "conflict_resolution": "archon_wins", // archon_wins | manual_wins | prompt
    "similarity_threshold": 0.85,     // Task matching threshold (0.5-1.0)
    "auto_commit": true,              // Auto git commit on updates
    "commit_message_template": "chore: sync task status from Archon [{completed}/{total} complete]"
  }
}
```

### Environment Variables

- `ARCHON_MCP_URL`: Archon MCP server URL (default: `http://localhost:8051`)

## Examples

### Example 1: Start Daemon for Active Project

```bash
# Terminal 1: Start Archon MCP
cd path/to/archon
./start-mcp-server.sh

# Terminal 2: Start sync daemon
cd path/to/Dionysus-2.0
python scripts/openspec_sync_daemon.py start

# Work on tasks in Archon...
# Tasks.md auto-updates every 30s!

# When done
python scripts/openspec_sync_daemon.py stop
```

### Example 2: One-Time Sync

```bash
# Sync single change
python scripts/sync_openspec_archon_status.py integrate-openspec-archon-sync

# Sync all changes
./scripts/sync_all_changes.sh
```

### Example 3: Verify Sync Working

```bash
# 1. Mark a task as "done" in Archon
mcp__archon__manage_task("update", task_id="...", status="done")

# 2. Wait 30s (or run sync manually)
python scripts/sync_openspec_archon_status.py <change-id>

# 3. Check tasks.md updated
git diff openspec/changes/<change-id>/tasks.md
# Should show: - [ ] → - [x]
```

---

## Quick Start Checklist

- [ ] Archon MCP server running
- [ ] `.openspec.config.json` configured
- [ ] Change created with `/openspec:import-to-archon`
- [ ] Daemon started: `python scripts/openspec_sync_daemon.py start`
- [ ] Verify: `python scripts/openspec_sync_daemon.py status`
- [ ] Test: Mark task "done" in Archon, wait 30s, check tasks.md

**Questions?** Check logs: `tail -f .openspec-sync-daemon.log`
