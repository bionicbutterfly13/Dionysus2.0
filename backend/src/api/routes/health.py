"""
System Health Check API
Validates all services required for upload/processing
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


class ServiceStatus(BaseModel):
    name: str
    status: str  # "healthy", "degraded", "down"
    message: str
    required_for: List[str]  # ["upload", "crawl", "query"]


class HealthResponse(BaseModel):
    overall_status: str  # "healthy", "degraded", "down"
    services: Dict[str, ServiceStatus]
    can_upload: bool
    can_crawl: bool
    can_query: bool
    errors: List[str]


@router.get("/health", response_model=HealthResponse)
async def check_health():
    """
    Comprehensive health check for all services.
    Returns detailed status of each service and what operations are possible.
    """
    services = {}
    errors = []

    # Check Neo4j
    try:
        from ...services.database_health import database_health_service
        neo4j_health = database_health_service.check_neo4j_health()

        if neo4j_health['status'] == 'healthy':
            services["neo4j"] = ServiceStatus(
                name="Neo4j",
                status="healthy",
                message=f"Connected in {neo4j_health['response_time_ms']}ms",
                required_for=["upload", "crawl", "query"]
            )
        else:
            services["neo4j"] = ServiceStatus(
                name="Neo4j",
                status="down",
                message="Cannot connect to Neo4j at localhost:7687. Start it with: docker start neo4j-memory",
                required_for=["upload", "crawl", "query"]
            )
            errors.append("❌ Neo4j is down - uploads and queries will fail")
    except Exception as e:
        services["neo4j"] = ServiceStatus(
            name="Neo4j",
            status="down",
            message=f"Neo4j health check failed: {str(e)}",
            required_for=["upload", "crawl", "query"]
        )
        errors.append(f"❌ Neo4j error: {str(e)}")

    # Check Redis
    try:
        from ...services.database_health import database_health_service
        redis_health = database_health_service.check_redis_health()

        if redis_health['status'] == 'healthy':
            services["redis"] = ServiceStatus(
                name="Redis",
                status="healthy",
                message=f"Connected in {redis_health['response_time_ms']}ms",
                required_for=["upload"]
            )
        else:
            services["redis"] = ServiceStatus(
                name="Redis",
                status="down",
                message="Cannot connect to Redis at localhost:6379. Start it with: docker run -d --name redis-dionysus -p 6379:6379 redis:7-alpine",
                required_for=["upload"]
            )
            errors.append("⚠️ Redis is down - basin caching unavailable (non-critical)")
    except Exception as e:
        services["redis"] = ServiceStatus(
            name="Redis",
            status="degraded",
            message=f"Redis check failed: {str(e)} (non-critical)",
            required_for=["upload"]
        )
        # Redis is not critical, so don't add to errors

    # Check Daedalus gateway
    try:
        from ...services.daedalus import Daedalus
        daedalus = Daedalus()

        services["daedalus"] = ServiceStatus(
            name="Daedalus Gateway",
            status="healthy",
            message="Document processing pipeline ready",
            required_for=["upload", "crawl"]
        )
    except Exception as e:
        services["daedalus"] = ServiceStatus(
            name="Daedalus Gateway",
            status="down",
            message=f"Daedalus initialization failed: {str(e)}",
            required_for=["upload", "crawl"]
        )
        errors.append(f"Daedalus error: {str(e)}")

    # Determine overall status
    critical_services_down = any(
        s.status == "down" and "upload" in s.required_for
        for s in services.values()
    )

    if critical_services_down:
        overall_status = "down"
    elif any(s.status == "degraded" for s in services.values()):
        overall_status = "degraded"
    else:
        overall_status = "healthy"

    # Determine capabilities
    can_upload = services.get("neo4j", ServiceStatus(name="", status="down", message="", required_for=[])).status == "healthy"
    can_crawl = can_upload  # Same requirements as upload
    can_query = services.get("neo4j", ServiceStatus(name="", status="down", message="", required_for=[])).status == "healthy"

    return HealthResponse(
        overall_status=overall_status,
        services=services,
        can_upload=can_upload,
        can_crawl=can_crawl,
        can_query=can_query,
        errors=errors
    )


@router.get("/health/simple")
async def simple_health():
    """Simple health check for monitoring (returns 200 if backend is up)"""
    return {"status": "ok"}
