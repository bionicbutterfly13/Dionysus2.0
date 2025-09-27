#!/bin/bash

# Flux - Self-Teaching Consciousness Emulator Startup Script
# Constitutional compliance: mock data transparency, local-first operation

echo "🧠 Starting Flux - Self-Teaching Consciousness Emulator"
echo "=================================================="
echo ""

# Check if we're in the right directory
if [ ! -f "backend/main.py" ] || [ ! -f "frontend/package.json" ]; then
    echo "❌ Error: Please run this script from the Flux project root directory"
    exit 1
fi

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo "🔍 Checking prerequisites..."

# Check Python
if ! command_exists python3; then
    echo "❌ Python 3 is required but not installed"
    exit 1
fi

# Check Node.js
if ! command_exists node; then
    echo "❌ Node.js is required but not installed"
    exit 1
fi

# Check if backend virtual environment exists
if [ ! -d "backend/flux-backend-env" ]; then
    echo "❌ Backend virtual environment not found. Please run setup first."
    exit 1
fi

# Check if frontend dependencies are installed
if [ ! -d "frontend/node_modules" ]; then
    echo "❌ Frontend dependencies not installed. Please run 'npm install' in frontend directory."
    exit 1
fi

echo "✅ Prerequisites check passed"
echo ""

# Constitutional compliance notice
echo "📋 Constitutional Compliance Notice:"
echo "   • Mock data is enabled for development"
echo "   • Local inference via Ollama/LLaMA (if available)"
echo "   • Evaluative feedback framework active"
echo "   • ThoughtSeed channels enabled"
echo "   • Context engineering best practices active"
echo ""

# Start backend
echo "🚀 Starting Flux Backend..."
cd backend
source flux-backend-env/bin/activate

# Start backend in background
python main.py &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

# Check if backend started successfully
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "❌ Backend failed to start"
    exit 1
fi

echo "✅ Backend started (PID: $BACKEND_PID)"
echo "   Backend URL: http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
echo ""

# Start frontend
echo "🎨 Starting Flux Frontend..."
cd ../frontend

# Start frontend in background
npm run dev &
FRONTEND_PID=$!

# Wait a moment for frontend to start
sleep 5

# Check if frontend started successfully
if ! kill -0 $FRONTEND_PID 2>/dev/null; then
    echo "❌ Frontend failed to start"
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

echo "✅ Frontend started (PID: $FRONTEND_PID)"
echo "   Frontend URL: http://localhost:3000"
echo ""

echo "🎉 Flux is now running!"
echo "======================"
echo ""
echo "📱 Access Flux at: http://localhost:3000"
echo "🔧 Backend API: http://localhost:8000"
echo "📚 API Documentation: http://localhost:8000/docs"
echo ""
echo "⚠️  Development Mode: Using mock data"
echo "   Real data validation required before production readiness"
echo ""
echo "Press Ctrl+C to stop all services"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down Flux services..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "✅ Flux services stopped"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Wait for user to stop services
wait
