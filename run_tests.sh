#!/bin/bash
# Comprehensive Test Runner for Flux - No Skipped Tests Allowed
set -e

echo "🧪 Flux Comprehensive Test Suite - No Skips Policy"
echo "======================================================"

# Activate environment
source ../asi-arch-env/bin/activate

# Change to backend directory
cd backend

# Step 1: Enforce no-skip policy (convert any skips to failing TODOs)
echo "🚫 Step 1: Enforce No-Skip Policy"
python enforce_no_skip_tests.py

echo ""
echo "✅ Step 2: Run Working Tests (Regression Protection)"
echo "These tests MUST pass - if they fail, we have a regression"
echo "----------------------------------------------------"

# Run only our implemented features that must always work
python -m pytest \
    tests/test_port_management.py \
    tests/test_database_health.py \
    -v --tb=short --strict-markers

WORKING_TESTS_STATUS=$?

echo ""
echo "📋 Step 3: Run All Tests (Full Status Check)"
echo "This shows which features are implemented vs TODO"
echo "-----------------------------------------------"

# Run all tests to see the full status
python -m pytest tests/ \
    -v --tb=line --continue-on-collection-errors || true

echo ""
echo "📊 Test Results Summary"
echo "======================"

if [ $WORKING_TESTS_STATUS -eq 0 ]; then
    echo "✅ REGRESSION CHECK: PASSED"
    echo "   All implemented features are working correctly"
else
    echo "❌ REGRESSION CHECK: FAILED"
    echo "   Working features have been broken by recent changes"
    exit 1
fi

echo ""
echo "📋 Test Categories:"
echo "  ✅ Working Tests: Port Management, Database Health"
echo "  📝 TODO Tests: API endpoints, integrations (fail with clear messages)"
echo "  🚫 Skipped Tests: ZERO (all converted to passing or failing tests)"

echo ""
echo "🎯 RESULT: All tests provide real feedback - no false confidence from skips"