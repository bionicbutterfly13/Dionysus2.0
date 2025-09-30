"""
Flux Self-Teaching Consciousness Emulator - Application Factory
FastAPI application setup with dependency injection and middleware configuration.
"""

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import os
import yaml
from pathlib import Path

from api.routes import documents, curiosity, visualization
from middleware.auth import LocalAuthMiddleware
from middleware.validation import ValidationMiddleware

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
        title="Flux Self-Teaching Consciousness Emulator API",
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

    @app.get("/")
    async def root():
        """Root endpoint with API information."""
        return {
            "name": "Flux Self-Teaching Consciousness Emulator API",
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
        from services.database_health import get_database_health
        return get_database_health()

    @app.get("/configs/flux.yaml")
    async def flux_config_file():
        """Expose the flux.yaml so the frontend can load runtime settings."""
        config_path = Path(__file__).parent.parent.parent / "configs" / "flux.yaml"
        if config_path.exists():
            return FileResponse(config_path)
        return JSONResponse({"error": "flux.yaml not found"}, status_code=404)

    @app.get("/api/stats/dashboard")
    async def dashboard_stats():
        """Get dashboard statistics from databases."""
        import redis
        import subprocess
        
        try:
            # Connect to Redis
            r = redis.Redis(host='localhost', port=6379, decode_responses=True)
            active_thoughtseeds = int(r.get("flux:thoughtseeds:active") or 0)
            curiosity_missions = r.zcard("flux:curiosity:missions")
            
            # Query Neo4j for documents and concepts
            neo4j_result = subprocess.run([
                'docker', 'exec', 'neo4j-flux', 'cypher-shell', 
                '-u', 'neo4j', '-p', 'neo4j_password',
                'MATCH (d:Document) WITH count(d) as docs MATCH (c:Concept) RETURN docs, count(c) as concepts'
            ], capture_output=True, text=True)
            
            documents, concepts = 0, 0
            if neo4j_result.returncode == 0:
                lines = neo4j_result.stdout.strip().split('\n')
                if len(lines) > 1:
                    data_line = lines[1]
                    # Remove commas and split
                    clean_line = data_line.replace(',', '')
                    parts = clean_line.split()
                    if len(parts) >= 2:
                        documents = int(parts[0])
                        concepts = int(parts[1])
            
            return {
                "documentsProcessed": documents,
                "conceptsExtracted": concepts,
                "curiosityMissions": curiosity_missions,
                "activeThoughtSeeds": active_thoughtseeds,
                "mockData": False
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

# Application instance
app = create_app()
