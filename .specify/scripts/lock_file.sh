#!/bin/bash
# File Locking Script - MANDATORY before editing
# Per Constitution Article 0, Section 0.1

set -e

if [ -z "$1" ]; then
  echo "Usage: $0 <file-path>"
  echo "Example: $0 backend/src/services/query.py"
  exit 1
fi

FILE_PATH="$1"
TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)

# Check if current agent is registered
if [ ! -f .specify/memory/current-agent.json ]; then
  echo "❌ Agent not registered. Run: .specify/scripts/register_agent.sh"
  exit 1
fi

AGENT_ID=$(jq -r '.agent_id' .specify/memory/current-agent.json)

# Check if file is locked by another agent
LOCKED_BY=$(jq -r '.agents[] | select(.status == "active" and .agent_id != "'$AGENT_ID'") | select(.active_files[] == "'$FILE_PATH'") | .agent_id' .specify/memory/agent-activity.json)

if [ -n "$LOCKED_BY" ]; then
  echo "⚠️  WARNING: $FILE_PATH is locked by another agent"
  LOCKED_BRANCH=$(jq -r '.agents[] | select(.agent_id == "'$LOCKED_BY'") | .branch' .specify/memory/agent-activity.json)
  LOCKED_SINCE=$(jq -r '.agents[] | select(.agent_id == "'$LOCKED_BY'") | .last_heartbeat' .specify/memory/agent-activity.json)
  echo "   Locked by: $LOCKED_BY"
  echo "   Branch: $LOCKED_BRANCH"
  echo "   Since: $LOCKED_SINCE"
  echo ""
  echo "⚠️  Proceeding may cause merge conflicts!"
  echo "Options:"
  echo "  A) Wait for other agent to finish"
  echo "  B) Coordinate with user to resolve"
  echo "  C) Work on different file"
  echo ""
  read -p "Proceed anyway? (y/N) " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Aborted. Choose a different file."
    exit 1
  fi
fi

# Check git history
RECENT_COMMITS=$(git log --since="1 hour ago" --name-only --oneline -- "$FILE_PATH" | head -5)
if [ -n "$RECENT_COMMITS" ]; then
  echo "📝 Recent activity on $FILE_PATH:"
  echo "$RECENT_COMMITS"
  echo ""
fi

# Add file to active_files list
jq --arg file "$FILE_PATH" --arg aid "$AGENT_ID" --arg ts "$TIMESTAMP" '
  .agents |= map(
    if .agent_id == $aid then
      .active_files += [$file] | .active_files |= unique | .last_heartbeat = $ts
    else . end
  ) | .last_updated = $ts
' .specify/memory/agent-activity.json > .specify/memory/agent-activity.tmp.json
mv .specify/memory/agent-activity.tmp.json .specify/memory/agent-activity.json

# Update current agent file
jq --arg file "$FILE_PATH" --arg ts "$TIMESTAMP" '
  .active_files += [$file] | .active_files |= unique | .last_heartbeat = $ts
' .specify/memory/current-agent.json > .specify/memory/current-agent.tmp.json
mv .specify/memory/current-agent.tmp.json .specify/memory/current-agent.json

echo "✅ File locked: $FILE_PATH"
echo "   By: $AGENT_ID"
echo "   At: $TIMESTAMP"
