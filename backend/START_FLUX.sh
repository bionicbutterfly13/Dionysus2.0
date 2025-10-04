#!/bin/bash
set -e  # Exit on error

echo "🔍 Checking required services..."

# Check Neo4j
echo -n "Checking Neo4j... "
if curl -sf http://localhost:7474 > /dev/null 2>&1; then
  echo "✅ Running"
else
  echo "❌ Not running"
  echo ""
  echo "Neo4j is required for document uploads and queries."
  echo ""
  echo "Native installation (recommended):"
  echo "  macOS:   brew install neo4j && brew services start neo4j"
  echo "  Ubuntu:  apt install neo4j && systemctl start neo4j"
  echo "  Windows: Download from https://neo4j.com/download/"
  echo ""
  echo "Cloud alternative:"
  echo "  Neo4j Aura: https://neo4j.com/cloud/aura/"
  echo "  Set NEO4J_URI environment variable to your instance"
  echo ""
  exit 1
fi

# Check Redis (optional, show warning only)
echo -n "Checking Redis... "
if nc -z localhost 6379 > /dev/null 2>&1; then
  echo "✅ Running"
else
  echo "⚠️  Not running (optional - caching disabled)"
  echo ""
  echo "  Native installation:"
  echo "    macOS:   brew install redis && brew services start redis"
  echo "    Ubuntu:  apt install redis-server && systemctl start redis"
  echo "    Windows: Download from https://github.com/microsoftarchive/redis/releases"
  echo ""
  echo "  Cloud alternative:"
  echo "    Redis Cloud: https://redis.com/try-free/"
  echo "    Set REDIS_URL environment variable"
  echo ""
fi

echo "✅ All critical services available"
echo "🚀 Starting Flux backend..."
echo ""

cd /Volumes/Asylum/dev/Dionysus-2.0/backend
export PYTHONPATH=/Volumes/Asylum/dev/Dionysus-2.0/backend/src:$PYTHONPATH
/Volumes/Asylum/dev/Dionysus-2.0/backend/flux-backend-env/bin/uvicorn src.app_factory:app --host 127.0.0.1 --port 9127 --reload
