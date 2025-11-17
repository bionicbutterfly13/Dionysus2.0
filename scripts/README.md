# OpenSpec + Archon Sync Scripts

This directory contains scripts for synchronizing OpenSpec changes with Archon task management.

## Scripts Overview

### `sync_openspec_archon_status.py`
**Purpose**: Sync a single OpenSpec change's task status from Archon
**Usage**:
```bash
# Sync specific change
python scripts/sync_openspec_archon_status.py integrate-openspec-archon-sync

# Dry-run (no changes written)
python scripts/sync_openspec_archon_status.py <change-id> --dry-run

# Custom Archon URL
python scripts/sync_openspec_archon_status.py <change-id> --archon-url http://localhost:8051
```

**What it does**:
1. Reads `.archon-project-id` from change directory
2. Fetches all tasks from Archon MCP
3. Parses `tasks.md` checkboxes
4. Updates checkboxes based on Archon task status
5. Commits changes to git (if `auto_commit: true` in config)

### `openspec_sync_daemon.py`
**Purpose**: Background daemon for automatic periodic syncing
**Usage**:
```bash
# Start daemon in background
python scripts/openspec_sync_daemon.py start

# Check status
python scripts/openspec_sync_daemon.py status

# Stop daemon
python scripts/openspec_sync_daemon.py stop

# Run in foreground (debug)
python scripts/openspec_sync_daemon.py run
```

**Features**:
- Auto-discovers all changes with `.archon-project-id`
- Polls Archon every N seconds (configurable in `.openspec.config.json`)
- Graceful shutdown (SIGTERM/SIGINT)
- Logs to `.openspec-sync-daemon.log`
- PID file: `.openspec-sync-daemon.pid`

### `sync_all_changes.sh`
**Purpose**: Bulk sync all changes with Archon integration
**Usage**:
```bash
./scripts/sync_all_changes.sh
```

**What it does**:
- Discovers all changes in `openspec/changes/` with `.archon-project-id`
- Syncs each change sequentially
- Reports success/failure counts

## Configuration

All scripts read from `.openspec.config.json`:

```json
{
  "archon_sync": {
    "enabled": true,
    "sync_interval_seconds": 30,
    "auto_commit": true,
    "conflict_resolution": "archon_wins",
    "similarity_threshold": 0.85
  }
}
```

## Environment Variables

- `ARCHON_MCP_URL`: Archon MCP server URL (default: `http://localhost:8051`)

## Common Workflows

### Manual Sync Before Archive
```bash
# Sync all changes
./scripts/sync_all_changes.sh

# Archive change (will validate Archon completion)
/openspec:archive <change-id>
```

### Active Development with Auto-Sync
```bash
# Start daemon
python scripts/openspec_sync_daemon.py start

# Work on tasks in Archon
# (tasks.md auto-updates every 30s)

# Check progress
python scripts/openspec_sync_daemon.py status

# Stop when done
python scripts/openspec_sync_daemon.py stop
```

### One-Time Sync for Specific Change
```bash
python scripts/sync_openspec_archon_status.py my-feature-change
```

## Troubleshooting

### "Cannot connect to Archon MCP"
1. Check Archon is running: `curl http://localhost:8051/health`
2. Set correct URL: `export ARCHON_MCP_URL=http://localhost:PORT`
3. Check MCP server configuration

### "No .archon-project-id found"
Run `/openspec:import-to-archon <change-id>` first

### Tasks Not Matching
- Task titles must be 85%+ similar (fuzzy matching)
- Check for typos or significant rewording
- View matching details in sync output

### Daemon Won't Start
- Check if already running: `python scripts/openspec_sync_daemon.py status`
- Check PID file: `cat .openspec-sync-daemon.pid`
- Remove stale PID: `rm .openspec-sync-daemon.pid`

## See Also

- [Automated Sync Guide](../docs/AUTOMATED_SYNC_GUIDE.md) - Complete setup guide
- [CLAUDE.md](../CLAUDE.md) - OpenSpec + Archon workflow documentation
- `.openspec.config.json` - Configuration reference
