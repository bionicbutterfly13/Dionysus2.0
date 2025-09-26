"""
Flux Backend - Self-Teaching Consciousness Emulator
Main FastAPI application entry point
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
from contextlib import asynccontextmanager

from core import settings, db_manager
from services import ollama_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management with constitutional compliance"""
    logger.info("Starting Flux backend services...")
    
    # Initialize database connections
    db_connected = await db_manager.connect_all()
    if not db_connected:
        logger.error("Failed to connect to databases")
    
    # Initialize Ollama/LLaMA service
    ollama_connected = await ollama_service.connect()
    if not ollama_connected:
        logger.warning("Ollama service not available - using mock data")
    
    logger.info("Flux backend services initialized")
    yield
    
    logger.info("Shutting down Flux backend services...")
    await db_manager.disconnect_all()
    await ollama_service.disconnect()
    logger.info("Flux backend services shut down")

app = FastAPI(
    title="Flux - Self-Teaching Consciousness Emulator",
    description="Backend API for lifelong learning partner with active inference",
    version="0.1.0",
    lifespan=lifespan
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React/Vite dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "Flux Backend",
        "status": "operational",
        "version": "0.1.0",
        "mock_data": True  # Constitutional requirement: explicit mock data disclosure
    }

@app.get("/api/v1/health")
async def health_check():
    """Detailed health check for all services with constitutional compliance"""
    # Get database health status
    db_health = await db_manager.health_check()
    
    # Get Ollama health status
    ollama_health = await ollama_service.health_check()
    
    return {
        "status": "healthy",
        "services": {
            "api": "operational",
            "neo4j": db_health.get("neo4j", "not_connected"),
            "qdrant": db_health.get("qdrant", "not_connected"),
            "redis": db_health.get("redis", "not_connected"),
            "sqlite": db_health.get("sqlite", "not_connected"),
            "ollama": ollama_health.get("status", "not_connected")
        },
        "mock_data": settings.mock_data,
        "constitutional_compliance": {
            "evaluation_feedback_enabled": settings.evaluation_feedback_enabled,
            "thoughtseed_channels_enabled": settings.thoughtseed_channels_enabled,
            "context_engineering_enabled": settings.context_engineering_enabled
        }
    }

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler with constitutional compliance"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "mock_data": True,
            "evaluation_frame": {
                "whats_good": "Error was logged for debugging",
                "whats_broken": "Unhandled exception occurred",
                "what_works_but_shouldnt": "None identified",
                "what_doesnt_but_pretends_to": "None identified"
            }
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
