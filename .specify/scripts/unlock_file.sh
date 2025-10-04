#!/bin/bash
# File Unlocking Script - Run after committing changes
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

# Remove file from active_files list
jq --arg file "$FILE_PATH" --arg aid "$AGENT_ID" --arg ts "$TIMESTAMP" '
  .agents |= map(
    if .agent_id == $aid then
      .active_files -= [$file] | .last_heartbeat = $ts
    else . end
  ) | .last_updated = $ts
' .specify/memory/agent-activity.json > .specify/memory/agent-activity.tmp.json
mv .specify/memory/agent-activity.tmp.json .specify/memory/agent-activity.json

# Update current agent file
jq --arg file "$FILE_PATH" --arg ts "$TIMESTAMP" '
  .active_files -= [$file] | .last_heartbeat = $ts
' .specify/memory/current-agent.json > .specify/memory/current-agent.tmp.json
mv .specify/memory/current-agent.tmp.json .specify/memory/current-agent.json

echo "✅ File unlocked: $FILE_PATH"
