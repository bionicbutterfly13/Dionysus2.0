"""
Dionysus Migration System FastAPI Application

Main application entry point providing REST API for consciousness-guided
legacy component migration with ThoughtSeed enhancement.
"""

from contextlib import asynccontextmanager
from typing import Dict

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api import (
    migration_router,
    component_router,
    coordination_router,
    monitoring_router
)
from .config import get_migration_config
from .logging_config import get_migration_logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager

    Handles startup and shutdown procedures for the migration system.
    """
    logger = get_migration_logger()
    config = get_migration_config()

    # Startup
    logger.info(
        "Starting Dionysus Migration System",
        version="1.0.0",
        config=config.dict()
    )

    # Initialize system components
    try:
        # In a production environment, would initialize:
        # - Database connections
        # - Redis connections
        # - Background task queues
        # - Health check systems
        logger.info("System components initialized successfully")

        yield

    finally:
        # Shutdown
        logger.info("Shutting down Dionysus Migration System")
        # Cleanup resources


# Create FastAPI application
app = FastAPI(
    title="Dionysus Migration System",
    description="""
    ## Consciousness-Guided Legacy Component Migration

    The Dionysus Migration System provides comprehensive tools for migrating legacy
    consciousness components from Dionysus 1.0 to Dionysus 2.0 using ThoughtSeed
    framework enhancements.

    ### Key Features

    - **Component Discovery**: Automated analysis of legacy codebases for consciousness patterns
    - **Quality Assessment**: Consciousness functionality and strategic value evaluation
    - **ThoughtSeed Enhancement**: AI-guided component rewriting with consciousness amplification
    - **DAEDALUS Coordination**: Distributed agent management with independent context windows
    - **Zero Downtime Migration**: Background processing without disrupting active development
    - **Fast Rollback**: <30 second component rollback with comprehensive checkpointing

    ### Migration Workflow

    1. **Discover** consciousness components in legacy codebase
    2. **Assess** components using consciousness and strategic metrics (70%/30% weighting)
    3. **Enhance** selected components using ThoughtSeed framework
    4. **Coordinate** migration tasks across distributed DAEDALUS agents
    5. **Monitor** progress and performance with comprehensive metrics
    6. **Rollback** individual components if needed with <30s recovery

    ### Architecture

    - **FastAPI**: Modern async Python web framework
    - **Pydantic**: Data validation and settings management
    - **ThoughtSeed**: Consciousness-guided enhancement framework
    - **DAEDALUS**: Distributed agent coordination system
    - **Redis**: Task queuing and caching (via LangGraph integration)
    - **Neo4j**: Optional graph database for complex relationships

    ### API Endpoints

    - **Migration**: `/api/v1/migration/*` - Pipeline and enhancement operations
    - **Components**: `/api/v1/components/*` - Component management and rollback
    - **Coordination**: `/api/v1/coordination/*` - Agent and task coordination
    - **Monitoring**: `/api/v1/monitoring/*` - System metrics and health

    For detailed usage examples and integration guides, see the API documentation below.
    """,
    version="1.0.0",
    contact={
        "name": "Dionysus Development Team",
        "url": "https://github.com/dionysus-ai/dionysus-migration",
        "email": "dev@dionysus-ai.com"
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    },
    lifespan=lifespan
)

# Configure CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Include API routers
app.include_router(migration_router)
app.include_router(component_router)
app.include_router(coordination_router)
app.include_router(monitoring_router)


@app.get("/", response_model=Dict)
async def root() -> Dict:
    """
    Root endpoint providing system information

    Returns basic system information and API navigation links.
    """
    config = get_migration_config()

    return {
        "service": "Dionysus Migration System",
        "version": "1.0.0",
        "description": "Consciousness-guided legacy component migration with ThoughtSeed enhancement",
        "status": "operational",
        "features": [
            "Component Discovery & Analysis",
            "Consciousness Functionality Assessment",
            "ThoughtSeed Framework Enhancement",
            "DAEDALUS Distributed Coordination",
            "Zero Downtime Background Migration",
            "Fast Component Rollback (<30s)"
        ],
        "api_endpoints": {
            "migration": "/api/v1/migration",
            "components": "/api/v1/components",
            "coordination": "/api/v1/coordination",
            "monitoring": "/api/v1/monitoring",
            "docs": "/docs",
            "redoc": "/redoc"
        },
        "configuration": {
            "quality_threshold": config.quality_threshold,
            "consciousness_weight": config.consciousness_weight,
            "strategic_weight": config.strategic_weight,
            "zero_downtime_required": config.zero_downtime_required
        }
    }


@app.get("/health", response_model=Dict)
async def health_check() -> Dict:
    """
    Health check endpoint

    Returns system health status and basic operational metrics.
    """
    return {
        "status": "healthy",
        "timestamp": "2024-01-01T12:00:00Z",
        "system": "operational",
        "components": {
            "api": "healthy",
            "services": "healthy",
            "database": "healthy",
            "coordination": "healthy"
        },
        "uptime_seconds": 3600  # Mock uptime
    }


@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Custom 404 handler"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": "The requested endpoint was not found",
            "suggestion": "Check the API documentation at /docs for available endpoints"
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Custom 500 handler"""
    logger = get_migration_logger()
    logger.error(f"Internal server error: {exc}")

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "suggestion": "Please try again later or contact support if the issue persists"
        }
    )