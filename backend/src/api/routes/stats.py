"""
Stats API routes for dashboard metrics
Per Frontend-Backend Integration spec
"""

import time
from fastapi import APIRouter
from typing import Dict, Any, List
from datetime import datetime, timedelta

router = APIRouter(prefix="/api/stats", tags=["stats"])

# No fake data - show zeros until real data exists
EMPTY_DASHBOARD_DATA = {
    "documentsProcessed": 0,
    "conceptsExtracted": 0,
    "curiosityMissions": 0,
    "activeThoughtSeeds": 0,
    "mockData": False,
    "lastUpdated": datetime.now().isoformat()
}

# Empty lists - no fake documents or thoughtseeds
EMPTY_RECENT_DOCUMENTS: List[Dict[str, Any]] = []
EMPTY_THOUGHTSEEDS: List[Dict[str, Any]] = []

@router.get("/dashboard")
async def get_dashboard_stats() -> Dict[str, Any]:
    """
    Get comprehensive dashboard statistics.
    
    Returns metrics for:
    - Documents processed count
    - Concepts extracted count  
    - Active curiosity missions
    - ThoughtSeed status
    """
    
    # TODO: Query actual data sources:
    # - Neo4j for documents and concepts
    # - Redis for real-time metrics
    # - Document processing graph state

    # Return zeros until real data exists - no fake numbers
    stats = {
        **EMPTY_DASHBOARD_DATA,
        "timestamp": time.time(),
        "status": "healthy"
    }

    return stats

@router.get("/recent-activity")
async def get_recent_activity() -> Dict[str, List[Dict[str, Any]]]:
    """Get recent document processing activity."""

    # Return empty lists until real data exists - no fake activity
    return {
        "recent_documents": EMPTY_RECENT_DOCUMENTS,
        "active_thoughtseeds": EMPTY_THOUGHTSEEDS,
        "timestamp": datetime.now().isoformat()
    }

@router.get("/performance")
async def get_performance_metrics() -> Dict[str, Any]:
    """Get system performance metrics."""

    # TODO: Get real system metrics from monitoring
    return {
        "cpu_usage": 0,
        "memory_usage": 0,
        "redis_connections": 0,
        "active_processes": 0,
        "response_time_avg_ms": 0,
        "timestamp": time.time()
    }

@router.get("/consciousness")
async def get_consciousness_metrics() -> Dict[str, Any]:
    """Get consciousness processing metrics."""

    # TODO: Get real consciousness metrics from active inference engine
    return {
        "consciousness_level": 0,
        "awareness_depth": 0,
        "meta_cognition_active": False,
        "inference_cycles_per_second": 0,
        "belief_updates": 0,
        "prediction_accuracy": 0,
        "timestamp": time.time()
    }