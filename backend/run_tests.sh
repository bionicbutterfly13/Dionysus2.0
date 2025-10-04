#!/bin/bash
# Test Runner Script for Flux Backend
# Runs all test suites to verify system integrity

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=====================================${NC}"
echo -e "${GREEN}Flux Backend Test Suite${NC}"
echo -e "${GREEN}=====================================${NC}"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activate virtual environment
if [ -d "flux-backend-env/bin" ]; then
    echo -e "${YELLOW}Activating virtual environment...${NC}"
    source flux-backend-env/bin/activate
else
    echo -e "${RED}Virtual environment not found at flux-backend-env/${NC}"
    exit 1
fi

# Set PYTHONPATH
export PYTHONPATH="${SCRIPT_DIR}:$PYTHONPATH"

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}pytest not found. Installing...${NC}"
    pip install pytest pytest-cov
fi

echo ""
echo -e "${GREEN}=====================================${NC}"
echo -e "${GREEN}1. Backend Startup Tests${NC}"
echo -e "${GREEN}=====================================${NC}"
pytest tests/test_startup.py -v --tb=short || true

echo ""
echo -e "${GREEN}=====================================${NC}"
echo -e "${GREEN}2. Consciousness Pipeline Tests${NC}"
echo -e "${GREEN}=====================================${NC}"
pytest tests/test_consciousness_pipeline.py -v --tb=short || true

echo ""
echo -e "${GREEN}=====================================${NC}"
echo -e "${GREEN}3. Document Upload Tests${NC}"
echo -e "${GREEN}=====================================${NC}"
pytest tests/test_document_upload.py -v --tb=short || true

echo ""
echo -e "${GREEN}=====================================${NC}"
echo -e "${GREEN}4. Database Connection Tests${NC}"
echo -e "${GREEN}=====================================${NC}"
pytest tests/test_database_connections.py -v --tb=short || true

echo ""
echo -e "${GREEN}=====================================${NC}"
echo -e "${GREEN}Test Summary${NC}"
echo -e "${GREEN}=====================================${NC}"
echo ""

# Run all tests with coverage
echo -e "${YELLOW}Running complete test suite with coverage...${NC}"
pytest tests/ -v --tb=short --cov=src --cov-report=term-missing --cov-report=html

echo ""
echo -e "${GREEN}=====================================${NC}"
echo -e "${GREEN}Tests Complete!${NC}"
echo -e "${GREEN}=====================================${NC}"
echo ""
echo -e "${YELLOW}Coverage report saved to: htmlcov/index.html${NC}"
echo ""
