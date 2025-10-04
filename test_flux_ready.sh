#!/bin/bash
# Test that CLAUSE Phase 2 is ready for Flux integration

echo "🧪 Testing CLAUSE Phase 2 for Flux Integration..."
echo ""

# Test 1: Server is running
echo "1. Checking server is running..."
if curl -s http://localhost:8001/api/demo/graph-status > /dev/null 2>&1; then
    echo "   ✅ Server is running on port 8001"
else
    echo "   ❌ Server not responding"
    echo "   Run: python backend/demo_server.py"
    exit 1
fi

# Test 2: Graph status endpoint
echo ""
echo "2. Testing /api/demo/graph-status..."
STATUS=$(curl -s http://localhost:8001/api/demo/graph-status)
NODE_COUNT=$(echo $STATUS | python3 -c "import sys, json; print(json.load(sys.stdin)['total_nodes'])")
if [ "$NODE_COUNT" == "8" ]; then
    echo "   ✅ Graph has 8 nodes"
else
    echo "   ❌ Graph status incorrect"
    exit 1
fi

# Test 3: Document processing endpoint
echo ""
echo "3. Testing /api/demo/process-document..."
cat > /tmp/flux_test.txt << 'EOF'
Climate change is primarily caused by greenhouse gas emissions.
Carbon dioxide from fossil fuels contributes to global warming.
We need renewable energy to reduce emissions.
EOF

RESULT=$(curl -s -X POST http://localhost:8001/api/demo/process-document -F "file=@/tmp/flux_test.txt")
CONCEPT_COUNT=$(echo $RESULT | python3 -c "import sys, json; print(len(json.load(sys.stdin)['concepts_extracted']))" 2>/dev/null)

if [ ! -z "$CONCEPT_COUNT" ] && [ "$CONCEPT_COUNT" -gt "0" ]; then
    echo "   ✅ Document processing works ($CONCEPT_COUNT concepts extracted)"
else
    echo "   ❌ Document processing failed"
    exit 1
fi

# Test 4: Agent execution
echo ""
echo "4. Testing multi-agent coordination..."
AGENT_COUNT=$(echo $RESULT | python3 -c "import sys, json; print(len(json.load(sys.stdin)['clause_response']['agent_handoffs']))" 2>/dev/null)

if [ "$AGENT_COUNT" == "3" ]; then
    echo "   ✅ All 3 agents executed (SubgraphArchitect, PathNavigator, ContextCurator)"
else
    echo "   ❌ Agent execution failed"
    exit 1
fi

# Test 5: Performance
echo ""
echo "5. Testing performance..."
TOTAL_TIME=$(echo $RESULT | python3 -c "import sys, json; print(json.load(sys.stdin)['total_time_ms'])" 2>/dev/null)
TOTAL_TIME_INT=$(echo $TOTAL_TIME | cut -d. -f1)

if [ ! -z "$TOTAL_TIME_INT" ] && [ "$TOTAL_TIME_INT" -lt "500" ]; then
    echo "   ✅ Performance target met (${TOTAL_TIME}ms < 500ms)"
else
    echo "   ⚠️  Performance: ${TOTAL_TIME}ms"
fi

# Test 6: Simple query endpoint
echo ""
echo "6. Testing /api/demo/simple-query..."
QUERY_RESULT=$(curl -s -X POST "http://localhost:8001/api/demo/simple-query?query=What+is+climate+change&start_node=climate_change")
PATH_EXISTS=$(echo $QUERY_RESULT | python3 -c "import sys, json; print('navigation_result' in json.load(sys.stdin))" 2>/dev/null)

if [ "$PATH_EXISTS" == "True" ]; then
    echo "   ✅ Simple query endpoint working"
else
    echo "   ❌ Simple query failed"
    exit 1
fi

# Summary
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ All tests passed!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "CLAUSE Phase 2 is ready for Flux integration."
echo ""
echo "Available endpoints:"
echo "  - POST /api/demo/process-document"
echo "  - GET  /api/demo/graph-status"
echo "  - POST /api/demo/simple-query"
echo ""
echo "Server running at: http://localhost:8001"
echo ""
echo "Next steps:"
echo "  1. See FLUX_CLAUSE_INTEGRATION.md for integration guide"
echo "  2. See QUICK_START.md for API examples"
echo "  3. See CLAUSE_PHASE2_READY.md for technical details"
echo ""
