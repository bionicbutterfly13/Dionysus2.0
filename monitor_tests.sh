#!/bin/bash
# Real-time test monitoring with visual reports
source asi-arch-env/bin/activate
cd backend

echo "🧪 Running tests with HTML report..."
python -m pytest tests/ \
    --html=reports/tests.html \
    --self-contained-html \
    --cov=src \
    --cov-report=html:reports/coverage \
    -v

echo ""
echo "📊 Test Report: file://$(pwd)/reports/tests.html"
echo "📈 Coverage Report: file://$(pwd)/reports/coverage/index.html"
