"""
Flux Self-Teaching Consciousness Emulator - Application Factory
FastAPI application setup with dependency injection and middleware configuration.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import os

from backend.src.api.routes import documents, curiosity, visualization
from backend.src.middleware.auth import LocalAuthMiddleware
from backend.src.middleware.validation import ValidationMiddleware

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

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="Flux Self-Teaching Consciousness Emulator API",
        description="Backend services for document ingestion, consciousness processing, and curiosity-driven learning",
        version="0.1.0",
        lifespan=lifespan
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Frontend URLs
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

    @app.get("/health")
    async def health_check():
        """Basic health check endpoint."""
        return {"status": "healthy", "service": "flux-backend"}

    @app.get("/health/databases")
    async def database_health_check():
        """Database connectivity health check endpoint."""
        from backend.src.services.database_health import get_database_health
        return get_database_health()

    return app

# Application instance
app = create_app()