"""
Minimal FastAPI Backend - Phase 1 TDD Implementation
Implements only what's needed to pass the tests.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

logger = logging.getLogger(__name__)

def create_minimal_app() -> FastAPI:
    """Create minimal FastAPI application for Phase 1."""
    
    app = FastAPI(
        title="Flux Minimal Backend",
        description="Minimal backend for TDD Phase 1",
        version="0.1.0"
    )

    # Add CORS middleware for frontend connection
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health_check():
        """Basic health check endpoint."""
        return {"status": "healthy", "service": "flux-backend"}

    @app.get("/api/stats/dashboard")
    async def dashboard_stats():
        """Get dashboard statistics - minimal implementation with mock data."""
        try:
            # For Phase 1, return mock data since external services aren't connected yet
            return {
                "documentsProcessed": 0,
                "conceptsExtracted": 0,
                "curiosityMissions": 0,
                "activeThoughtSeeds": 0,
                "mockData": True
            }
        except Exception as e:
            logger.error(f"Dashboard stats error: {e}")
            return {
                "documentsProcessed": 0,
                "conceptsExtracted": 0,
                "curiosityMissions": 0,
                "activeThoughtSeeds": 0,
                "mockData": True
            }

    return app

# Application instance for testing
app = create_minimal_app()