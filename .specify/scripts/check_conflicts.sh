#!/bin/bash
# Conflict Detection Script - Run before committing
# Per Constitution Article 0, Section 0.2

set -e

echo "🔍 Checking for potential merge conflicts..."
echo ""

# Fetch latest from all branches
echo "📡 Fetching latest changes from all branches..."
git fetch --all --quiet

# Get current branch
CURRENT_BRANCH=$(git branch --show-current)
echo "   Current branch: $CURRENT_BRANCH"
echo ""

# Check if on main (violation)
if [ "$CURRENT_BRANCH" = "main" ] || [ "$CURRENT_BRANCH" = "master" ]; then
  echo "❌ CONSTITUTION VIOLATION: Working on main/master branch"
  exit 1
fi

# Get files changed in current branch
CHANGED_FILES=$(git diff --name-only origin/main...HEAD 2>/dev/null || git diff --name-only HEAD)
if [ -z "$CHANGED_FILES" ]; then
  echo "✅ No changes detected"
  exit 0
fi

echo "📝 Files modified in your branch:"
echo "$CHANGED_FILES" | sed 's/^/   /'
echo ""

# Check if these files were modified in other branches
echo "🔍 Checking for conflicts with other branches..."
CONFLICT_FOUND=0

for file in $CHANGED_FILES; do
  # Check if file was modified in main recently
  MAIN_CHANGES=$(git log origin/main --since="24 hours ago" --name-only --oneline -- "$file" 2>/dev/null | head -5)
  if [ -n "$MAIN_CHANGES" ]; then
    echo "⚠️  $file was modified in main branch:"
    echo "$MAIN_CHANGES" | sed 's/^/   /'
    echo ""
    CONFLICT_FOUND=1
  fi

  # Check if file is locked by another agent
  if [ -f .specify/memory/agent-activity.json ]; then
    AGENT_ID=$(jq -r '.agent_id' .specify/memory/current-agent.json 2>/dev/null || echo "unknown")
    LOCKED_BY=$(jq -r '.agents[] | select(.status == "active" and .agent_id != "'$AGENT_ID'") | select(.active_files[] == "'$file'") | .agent_id' .specify/memory/agent-activity.json 2>/dev/null)
    if [ -n "$LOCKED_BY" ]; then
      echo "⚠️  $file is locked by another agent:"
      jq -r '.agents[] | select(.agent_id == "'$LOCKED_BY'") | "   Agent: \(.agent_id)\n   Branch: \(.branch)\n   Since: \(.last_heartbeat)"' .specify/memory/agent-activity.json
      echo ""
      CONFLICT_FOUND=1
    fi
  fi
done

if [ $CONFLICT_FOUND -eq 1 ]; then
  echo "⚠️  Potential conflicts detected!"
  echo ""
  echo "Recommendations:"
  echo "1. Sync with main: git fetch origin main && git rebase origin/main"
  echo "2. Coordinate with other agents before committing"
  echo "3. Ask user for guidance on conflict resolution"
  echo ""
  read -p "Proceed with commit anyway? (y/N) " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Commit aborted. Resolve conflicts first."
    exit 1
  fi
else
  echo "✅ No conflicts detected"
fi

echo ""
echo "📊 Summary:"
echo "   Branch: $CURRENT_BRANCH"
echo "   Modified files: $(echo "$CHANGED_FILES" | wc -l | tr -d ' ')"
echo "   Conflicts: $CONFLICT_FOUND"
echo ""
