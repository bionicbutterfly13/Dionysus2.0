#!/bin/bash
# Clean backend startup script

echo "🧹 Cleaning up old processes..."
lsof -ti:9127 | xargs kill -9 2>/dev/null
sleep 2

echo "🚀 Starting backend on port 9127..."
cd "$(dirname "$0")"
python3 -m uvicorn src.app_factory:app --host 127.0.0.1 --port 9127 --reload

# Backend will run in foreground - Ctrl+C to stop
