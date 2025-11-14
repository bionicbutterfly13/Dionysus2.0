"""
Dionysus 2.0 - Application Factory
FastAPI application setup with dependency injection and middleware configuration.
"""

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import os
import sys
import yaml
from pathlib import Path

BACKEND_SRC = Path(__file__).resolve().parent
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

from .api.routes import (  # noqa: E402  — import after sys.path adjustment
    documents,
    document_persistence,  # Spec 054 - Neo4j-backed persistence
    document_citations,  # Spec 058 - Citation trust interaction
    curiosity,
    visualization,
    stats,
    consciousness,
    query,
    health,
)

# Optional: crawl (requires crawl4ai)
try:
    from .api.routes import crawl
    CRAWL_AVAILABLE = True
except ImportError:
    CRAWL_AVAILABLE = False
    # Will log warning after logger is initialized

# from .api.routes import clause  # Import separately to avoid circular dependency
# from .api.routes import demo_clause  # Demo CLAUSE pipeline
from .middleware.auth import LocalAuthMiddleware
from .middleware.validation import ValidationMiddleware

logger = logging.getLogger(__name__)

if not CRAWL_AVAILABLE:
    logger.warning("crawl4ai not available, /api/crawl endpoints disabled")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management for startup/shutdown tasks."""
    logger.info("Starting Dionysus backend services...")

    # Initialize database connections
    # TODO: Add Neo4j, Redis initialization here

    yield

    logger.info("Shutting down Dionysus backend services...")
    # TODO: Add cleanup tasks here

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    # Default CORS origins
    cors_origins = ["http://localhost:3000", "http://127.0.0.1:3000"]

    app = FastAPI(
        title="Dionysus 2.0 API",
        description="Consciousness-enhanced document processing with multi-agent coordination",
        version="0.1.0",
        lifespan=lifespan
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add custom middleware
    app.add_middleware(ValidationMiddleware)
    app.add_middleware(LocalAuthMiddleware)

    # Include API routes
    app.include_router(documents.router, prefix="/api/v1", tags=["documents"])
    app.include_router(document_persistence.router, tags=["document_persistence"])  # Spec 054 - Neo4j persistence
    app.include_router(document_citations.router, tags=["citations"])  # Spec 058 - Citation trust interaction
    app.include_router(curiosity.router, prefix="/api/v1", tags=["curiosity"])
    app.include_router(visualization.router, prefix="/ws/v1", tags=["visualization"])
    app.include_router(stats.router, tags=["stats"])
    app.include_router(consciousness.router, tags=["consciousness"])
    app.include_router(query.router, tags=["query"])  # Query endpoint per Spec 006
    if CRAWL_AVAILABLE:
        app.include_router(crawl.router, prefix="/api", tags=["crawl"])  # Web crawling from Archon
    app.include_router(health.router, prefix="/api", tags=["health"])  # System health checks
    # app.include_router(clause.router, tags=["clause"])  # DISABLED - import issues
    # app.include_router(demo_clause.router, tags=["demo"])  # DISABLED - import issues

    @app.get("/")
    async def root():
        """Root endpoint with API information."""
        return {
            "name": "Dionysus 2.0 API",
            "version": "0.1.0",
            "status": "healthy",
            "documentation": "/docs",
            "endpoints": {
                "health": "/health",
                "api": "/api/v1",
                "dashboard_stats": "/api/stats/dashboard"
            }
        }

    @app.get("/health")
    async def health_check():
        """Basic health check endpoint."""
        return {"status": "healthy", "service": "dionysus-backend"}

    @app.get("/health/databases")
    async def database_health_check():
        """Database connectivity health check endpoint."""
        from .services.database_health import get_database_health
        return get_database_health()


    return app

# Application instance
app = create_app()

# Create app instance for uvicorn to import
app = create_app()

# Main entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9127, reload=True)
