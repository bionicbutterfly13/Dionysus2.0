#!/bin/bash
# Setup battle-tested monitoring tools for Flux development

echo "🛠️ Setting up Development Monitoring Tools"
echo "=========================================="

# 1. FastAPI automatic docs and monitoring (built-in)
echo "✅ 1. FastAPI Swagger UI: http://localhost:9127/docs"
echo "✅ 2. FastAPI ReDoc: http://localhost:9127/redoc"

# 3. pytest-html for beautiful test reports
echo "📊 3. Installing pytest-html for visual test reports..."
source asi-arch-env/bin/activate
pip install pytest-html

# 4. pytest-cov for test coverage visualization
echo "📈 4. Installing pytest-cov for coverage reports..."
pip install pytest-cov

# 5. Install httpx for API testing
echo "🔍 5. Installing httpx for API testing..."
pip install httpx

# 6. Create quick monitoring commands
echo "⚡ 6. Creating quick monitoring commands..."

cat > monitor_tests.sh << 'EOF'
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
EOF

cat > monitor_api.sh << 'EOF'
#!/bin/bash
# API monitoring and testing
echo "🔍 API Health Checks"
echo "=================="

echo "Basic Health:"
curl -s http://localhost:9127/health | python -m json.tool

echo -e "\nDatabase Health:"
curl -s http://localhost:9127/health/databases | python -m json.tool

echo -e "\n📊 API Documentation:"
echo "Swagger UI: http://localhost:9127/docs"
echo "ReDoc: http://localhost:9127/redoc"
EOF

chmod +x monitor_tests.sh
chmod +x monitor_api.sh

# Create reports directory
mkdir -p backend/reports

echo ""
echo "🎯 Available Monitoring Tools:"
echo "========================="
echo "📊 Visual Test Reports: ./monitor_tests.sh"
echo "🔍 API Monitoring: ./monitor_api.sh"
echo "📚 Swagger UI: http://localhost:9127/docs (when server running)"
echo "📖 ReDoc: http://localhost:9127/redoc (when server running)"
echo ""
echo "✅ All tools are industry-standard, well-maintained projects!"