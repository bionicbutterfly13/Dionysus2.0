#!/bin/bash
# Agent Registration Script - MANDATORY before starting work
# Per Constitution Article 0, Section 0.1

set -e

AGENT_ID="claude-$(date +%s)"
BRANCH=$(git branch --show-current)
TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)

# Check if on main branch (violation)
if [ "$BRANCH" = "main" ] || [ "$BRANCH" = "master" ]; then
  echo "❌ CONSTITUTION VIOLATION: Cannot work on main/master branch"
  echo "Create a feature branch first:"
  echo "  git checkout -b 043-your-feature-name"
  exit 1
fi

# Create current agent file
cat > .specify/memory/current-agent.json <<EOF
{
  "agent_id": "$AGENT_ID",
  "branch": "$BRANCH",
  "active_files": [],
  "started_at": "$TIMESTAMP",
  "last_heartbeat": "$TIMESTAMP",
  "status": "active"
}
EOF

# Add to agent-activity.json
if [ ! -f .specify/memory/agent-activity.json ]; then
  cat > .specify/memory/agent-activity.json <<EOF
{
  "agents": [],
  "last_updated": "$TIMESTAMP",
  "schema_version": "1.0.0"
}
EOF
fi

# Use jq to add this agent to the list
jq --argjson agent "$(cat .specify/memory/current-agent.json)" '.agents += [$agent] | .last_updated = "'$TIMESTAMP'"' \
  .specify/memory/agent-activity.json > .specify/memory/agent-activity.tmp.json
mv .specify/memory/agent-activity.tmp.json .specify/memory/agent-activity.json

echo "✅ Agent registered:"
echo "   ID: $AGENT_ID"
echo "   Branch: $BRANCH"
echo "   Started: $TIMESTAMP"
echo ""
echo "📢 Agent Status Report:"
echo "- Branch: $BRANCH"
echo "- Current work: [Describe your work here]"
echo "- Expected duration: [Estimate time]"
echo ""

# Check for other active agents
OTHER_AGENTS=$(jq -r '.agents[] | select(.status == "active" and .agent_id != "'$AGENT_ID'") | .agent_id' .specify/memory/agent-activity.json)
if [ -n "$OTHER_AGENTS" ]; then
  echo "⚠️  Other active agents detected:"
  jq -r '.agents[] | select(.status == "active" and .agent_id != "'$AGENT_ID'") | "  - \(.agent_id) on branch \(.branch)"' .specify/memory/agent-activity.json
  echo ""
  echo "📝 Check their locked files to avoid conflicts:"
  jq -r '.agents[] | select(.status == "active" and .agent_id != "'$AGENT_ID'") | .active_files[]' .specify/memory/agent-activity.json
fi
