#!/bin/bash
# Sync all OpenSpec changes with Archon integration

set -e

CHANGES_DIR="openspec/changes"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -d "$CHANGES_DIR" ]; then
    echo "✗ No openspec/changes directory found"
    exit 1
fi

SYNCED=0
FAILED=0
SKIPPED=0

echo "Discovering changes with Archon integration..."

for change_dir in "$CHANGES_DIR"/*; do
    if [ ! -d "$change_dir" ]; then
        continue
    fi

    change_id=$(basename "$change_dir")

    # Check for Archon integration
    if [ ! -f "$change_dir/.archon-project-id" ]; then
        ((SKIPPED++))
        continue
    fi

    echo ""
    echo "Syncing: $change_id"
    echo "─────────────────────────────────────────────"

    if python3 "$SCRIPT_DIR/sync_openspec_archon_status.py" "$change_id"; then
        ((SYNCED++))
    else
        echo "  ✗ Failed to sync $change_id"
        ((FAILED++))
    fi
done

echo ""
echo "════════════════════════════════════════════════"
echo "Results:"
echo "  ✓ Synced:  $SYNCED"
echo "  ✗ Failed:  $FAILED"
echo "  ⊘ Skipped: $SKIPPED (no Archon integration)"
echo "════════════════════════════════════════════════"

exit $FAILED
