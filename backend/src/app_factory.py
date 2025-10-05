"""
Flux Self-Evolving Consciousness Emulator - Application Factory
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
    curiosity,
    visualization,
    stats,
    consciousness,
    query,
    crawl,
    health,
)
# from .api.routes import clause  # Import separately to avoid circular dependency
# from .api.routes import demo_clause  # Demo CLAUSE pipeline
from .middleware.auth import LocalAuthMiddleware
from .middleware.validation import ValidationMiddleware

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management for startup/shutdown tasks."""
    logger.info("Starting Flux backend services...")

    # Initialize database connections
    # TODO: Add Neo4j, Redis, Qdrant initialization here

    yield

    logger.info("Shutting down Flux backend services...")
    # TODO: Add cleanup tasks here

def load_flux_config():
    """Load flux.yaml configuration."""
    config_path = Path(__file__).parent.parent.parent / "configs" / "flux.yaml"
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load flux.yaml: {e}")
        return {}

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    # Load configuration
    config = load_flux_config()
    cors_origins = config.get('server', {}).get('cors_origins', ["http://localhost:3000"])

    app = FastAPI(
        title="Flux Self-Evolving Consciousness Emulator API",
        description="Backend services for document ingestion, consciousness processing, and curiosity-driven learning",
        version="0.1.0",
        lifespan=lifespan
    )

    # Add CORS middleware with flux.yaml origins
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
    app.include_router(curiosity.router, prefix="/api/v1", tags=["curiosity"])
    app.include_router(visualization.router, prefix="/ws/v1", tags=["visualization"])
    app.include_router(stats.router, tags=["stats"])
    app.include_router(consciousness.router, tags=["consciousness"])
    app.include_router(query.router, tags=["query"])  # Query endpoint per Spec 006
    app.include_router(crawl.router, prefix="/api", tags=["crawl"])  # Web crawling from Archon
    app.include_router(health.router, prefix="/api", tags=["health"])  # System health checks
    # app.include_router(clause.router, tags=["clause"])  # DISABLED - import issues
    # app.include_router(demo_clause.router, tags=["demo"])  # DISABLED - import issues

    @app.get("/")
    async def root():
        """Root endpoint with API information."""
        return {
            "name": "Flux Self-Evolving Consciousness Emulator API",
            "version": "0.1.0",
            "status": "healthy",
            "documentation": "/docs",
            "endpoints": {
                "health": "/health",
                "api": "/api/v1",
                "dashboard_stats": "/api/stats/dashboard",
                "config": "/configs/flux.yaml"
            }
        }

    @app.get("/health")
    async def health_check():
        """Basic health check endpoint."""
        return {"status": "healthy", "service": "flux-backend"}

    @app.get("/health/databases")
    async def database_health_check():
        """Database connectivity health check endpoint."""
        from .services.database_health import get_database_health
        return get_database_health()

    @app.get("/configs/flux.yaml")
    async def flux_config_file():
        """Expose the flux.yaml so the frontend can load runtime settings."""
        config_path = Path(__file__).parent.parent.parent / "configs" / "flux.yaml"
        if config_path.exists():
            return FileResponse(config_path)
        return JSONResponse({"error": "flux.yaml not found"}, status_code=404)


    return app

# Application instance
app = create_app()

# Create app instance for uvicorn to import
app = create_app()

# Main entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9127, reload=True)
