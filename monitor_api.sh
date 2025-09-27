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
