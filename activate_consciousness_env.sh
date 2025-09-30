#!/bin/bash
# 🚀 Consciousness Processing Environment Activation
# =============================================
# 
# This script activates the complete development environment for
# consciousness processing multi-agent coordination system.
#
# Usage: source activate_consciousness_env.sh
#
# Author: Consciousness Processing Integration
# Date: 2025-09-30
# Version: 2.0.0

echo "🚀 Activating Consciousness Processing Environment..."
echo "=================================================="

# Check if we're in the right directory
if [ ! -f "CLAUDE.md" ]; then
    echo "❌ Error: Not in Dionysus-2.0 directory"
    echo "Please run: cd /Volumes/Asylum/dev/Dionysus-2.0"
    exit 1
fi

# Set environment variables
export CONSCIOUSNESS_ROOT="/Volumes/Asylum/dev/Dionysus-2.0"
export CONSCIOUSNESS_ENV="consciousness-env"
export PYTHONPATH="${CONSCIOUSNESS_ROOT}:${PYTHONPATH}"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
echo "🐍 Python Version: $PYTHON_VERSION"

if [[ "$PYTHON_VERSION" < "3.11.0" ]]; then
    echo "⚠️  Warning: Python 3.11.0+ recommended (current: $PYTHON_VERSION)"
fi

# Activate virtual environment
echo "📦 Activating Python virtual environment: $CONSCIOUSNESS_ENV"
if [ -d "$CONSCIOUSNESS_ENV" ]; then
    source $CONSCIOUSNESS_ENV/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "⚠️  Virtual environment '$CONSCIOUSNESS_ENV' not found"
    echo "Creating new virtual environment..."
    python3 -m venv $CONSCIOUSNESS_ENV
    source $CONSCIOUSNESS_ENV/bin/activate
    echo "✅ New virtual environment created and activated"
fi

# Install/upgrade required packages
echo "📚 Installing/updating required packages..."
pip install --upgrade pip

# Install core dependencies with NumPy 2.0+
echo "🧮 Installing NumPy 2.0+ and dependencies..."
pip install "numpy>=2.0" --upgrade
pip install -r requirements-consciousness.txt 2>/dev/null || {
    echo "⚠️  requirements-consciousness.txt not found, installing core packages..."
    pip install neo4j redis sentence-transformers pandas fastapi uvicorn
}

# Verify NumPy version compliance
python3 -c "
import numpy
assert numpy.__version__.startswith('2.'), f'NumPy {numpy.__version__} violates constitution (requires 2.0+)'
print(f'✅ NumPy {numpy.__version__} - Constitution compliant')
"

# Verify critical services
echo "🔍 Verifying critical services..."

# Check Neo4j connection
if command -v curl &> /dev/null; then
    if curl -s http://localhost:7474 > /dev/null 2>&1; then
        echo "✅ Neo4j: Connected (http://localhost:7474)"
    else
        echo "⚠️  Neo4j: Not running (start with: docker run -d --name neo4j-consciousness -p 7474:7474 -p 7687:7687 neo4j:5.15)"
    fi
fi

# Check Redis connection
if command -v redis-cli &> /dev/null; then
    if redis-cli ping > /dev/null 2>&1; then
        echo "✅ Redis: Connected (localhost:6379)"
    else
        echo "⚠️  Redis: Not running (start with: docker run -d --name redis-consciousness -p 6379:6379 redis:7-alpine)"
    fi
fi

# Check frontend and backend
echo "🌐 Checking web services..."
if curl -s http://localhost:9243 > /dev/null 2>&1; then
    echo "✅ Frontend: Running (http://localhost:9243)"
else
    echo "⚠️  Frontend: Not running (start with: cd frontend && npm run dev)"
fi

if curl -s http://127.0.0.1:9127 > /dev/null 2>&1; then
    echo "✅ Backend: Running (http://127.0.0.1:9127)"
else
    echo "⚠️  Backend: Not running (start with: cd backend && python main.py)"
fi

# Set up environment completion
echo ""
echo "✅ Environment Activation Complete!"
echo "=================================="
echo ""
echo "🎯 Available Commands:"
echo "  python test_consciousness_integration.py       - Test consciousness system"
echo "  python constitutional_compliance_checker.py    - Check NumPy compliance"
echo "  cd frontend && npm run dev                      - Start frontend (port 9243)"
echo "  cd backend && python main.py                    - Start backend (port 9127)"
echo ""
echo "📋 Next Steps:"
echo "  1. Check frontend: http://localhost:9243"
echo "  2. Check backend API: http://127.0.0.1:9127"
echo "  3. Check API docs: http://127.0.0.1:9127/docs"
echo ""
echo "🧠 Ready for consciousness processing development!"
echo ""

# Store activation timestamp
echo "$(date): Consciousness environment activated" >> .env_activation.log