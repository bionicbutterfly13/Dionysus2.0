#!/bin/bash
# 🚀 ASI-Arch ThoughtSeed Environment Activation
# =============================================
# 
# This script activates the complete development environment for
# ASI-Arch ThoughtSeed multi-agent coordination system.
#
# Usage: source activate_asi_env.sh
#
# Author: ASI-Arch ThoughtSeed Integration
# Date: 2025-09-24
# Version: 1.0.0

echo "🚀 Activating ASI-Arch ThoughtSeed Environment..."
echo "=================================================="

# Check if we're in the right directory
if [ ! -f "SPEC_DRIVEN_DEVELOPMENT_PROTOCOL.md" ]; then
    echo "❌ Error: Not in ASI-Arch-Thoughtseeds directory"
    echo "Please run: cd /Volumes/Asylum/devb/ASI-Arch-Thoughtseeds"
    exit 1
fi

# Set environment variables
export ASI_ARCH_ROOT="/Volumes/Asylum/devb/ASI-Arch-Thoughtseeds"
export ASI_ARCH_ENV="asi-arch-env"
export PYTHONPATH="${ASI_ARCH_ROOT}:${PYTHONPATH}"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
echo "🐍 Python Version: $PYTHON_VERSION"

if [[ "$PYTHON_VERSION" < "3.11.0" ]]; then
    echo "⚠️  Warning: Python 3.11.0+ recommended (current: $PYTHON_VERSION)"
fi

# Activate conda environment if available
if command -v conda &> /dev/null; then
    echo "📦 Activating Conda environment: $ASI_ARCH_ENV"
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate $ASI_ARCH_ENV 2>/dev/null || {
        echo "⚠️  Conda environment '$ASI_ARCH_ENV' not found"
        echo "Creating new conda environment..."
        conda create -n $ASI_ARCH_ENV python=3.11 -y
        conda activate $ASI_ARCH_ENV
    }
elif command -v python3 -m venv &> /dev/null; then
    echo "📦 Activating Python virtual environment"
    if [ -d "venv" ]; then
        source venv/bin/activate
    else
        echo "⚠️  Virtual environment not found, creating new one..."
        python3 -m venv venv
        source venv/bin/activate
    fi
else
    echo "⚠️  No virtual environment system found"
fi

# Install/upgrade required packages
echo "📚 Installing/updating required packages..."
pip install --upgrade pip

# Install core dependencies
pip install -r requirements.txt 2>/dev/null || {
    echo "⚠️  requirements.txt not found, installing core packages..."
    pip install neo4j redis langgraph langchain-core sentence-transformers numpy pandas
}

# Verify critical services
echo "🔍 Verifying critical services..."

# Check Neo4j connection
if command -v curl &> /dev/null; then
    if curl -s http://localhost:7474 > /dev/null 2>&1; then
        echo "✅ Neo4j: Connected (http://localhost:7474)"
    else
        echo "⚠️  Neo4j: Not running (start with: docker run -d --name neo4j-thoughtseed -p 7474:7474 -p 7687:7687 neo4j:5.15)"
    fi
fi

# Check Redis connection
if command -v redis-cli &> /dev/null; then
    if redis-cli ping > /dev/null 2>&1; then
        echo "✅ Redis: Connected (localhost:6379)"
    else
        echo "⚠️  Redis: Not running (start with: docker run -d --name redis-thoughtseed -p 6379:6379 redis:7-alpine)"
    fi
fi

# Initialize agent coordination system
echo "🤖 Initializing Multi-Agent Coordination System..."
python3 enhanced_multi_agent_coordination.py setup 2>/dev/null || {
    echo "⚠️  Agent coordination system initialization failed"
}

# Set up environment completion
echo ""
echo "✅ Environment Activation Complete!"
echo "=================================="
echo ""
echo "🎯 Available Commands:"
echo "  python enhanced_multi_agent_coordination.py setup     - Verify environment"
echo "  python enhanced_multi_agent_coordination.py status    - Check coordination status"
echo "  python enhanced_multi_agent_coordination.py dashboard - View dashboard"
echo "  python enhanced_multi_agent_coordination.py available - List available specifications"
echo ""
echo "📋 Next Steps:"
echo "  1. Check coordination status: python enhanced_multi_agent_coordination.py status"
echo "  2. View available specifications: python enhanced_multi_agent_coordination.py available"
echo "  3. Check-out a specification: python enhanced_multi_agent_coordination.py checkout --spec BP-006 --agent your_name"
echo ""
echo "🌱🧠 Ready for ThoughtSeed development with multi-agent coordination!"
echo ""

# Store activation timestamp
echo "$(date): ASI-Arch environment activated" >> .env_activation.log