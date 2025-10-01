"""
Stats API routes for dashboard metrics
Per Frontend-Backend Integration spec
"""

import time
from fastapi import APIRouter
from typing import Dict, Any, List
from datetime import datetime, timedelta

router = APIRouter(prefix="/api/stats", tags=["stats"])

# Mock data for demonstration - would connect to real data sources
MOCK_DASHBOARD_DATA = {
    "documentsProcessed": 127,
    "conceptsExtracted": 432,
    "curiosityMissions": 15,
    "activeThoughtSeeds": 8,
    "mockData": False,
    "lastUpdated": datetime.now().isoformat()
}

MOCK_RECENT_DOCUMENTS = [
    {
        "filename": "consciousness_theory_advanced.pdf",
        "status": "processed",
        "timestamp": (datetime.now() - timedelta(minutes=5)).isoformat(),
        "concepts_extracted": 23
    },
    {
        "filename": "neural_architecture_search.md", 
        "status": "processing",
        "timestamp": (datetime.now() - timedelta(minutes=12)).isoformat(),
        "concepts_extracted": 0
    },
    {
        "filename": "active_inference_principles.pdf",
        "status": "curious",
        "timestamp": (datetime.now() - timedelta(minutes=18)).isoformat(),
        "concepts_extracted": 41
    }
]

MOCK_THOUGHTSEEDS = [
    {
        "name": "Active Inference Learning",
        "status": "growing",
        "confidence": 0.87,
        "age_minutes": 245
    },
    {
        "name": "Pattern Recognition Enhancement", 
        "status": "stable",
        "confidence": 0.93,
        "age_minutes": 1440
    },
    {
        "name": "Meta-Cognitive Monitoring",
        "status": "exploring", 
        "confidence": 0.72,
        "age_minutes": 67
    }
]

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
    
    # In production, this would query actual data sources:
    # - Document processing database
    # - Concept extraction service
    # - ThoughtSeed monitoring system
    # - Redis for real-time metrics
    
    stats = {
        **MOCK_DASHBOARD_DATA,
        "timestamp": time.time(),
        "status": "healthy"
    }
    
    # Simulate slight variations to show live updates
    import random
    stats["documentsProcessed"] += random.randint(0, 2)
    stats["conceptsExtracted"] += random.randint(0, 5)
    
    return stats

@router.get("/recent-activity")
async def get_recent_activity() -> Dict[str, List[Dict[str, Any]]]:
    """Get recent document processing activity."""
    
    return {
        "recent_documents": MOCK_RECENT_DOCUMENTS,
        "active_thoughtseeds": MOCK_THOUGHTSEEDS,
        "timestamp": datetime.now().isoformat()
    }

@router.get("/performance")
async def get_performance_metrics() -> Dict[str, Any]:
    """Get system performance metrics."""
    
    return {
        "cpu_usage": 0.23,
        "memory_usage": 0.67,
        "redis_connections": 4,
        "active_processes": 12,
        "response_time_avg_ms": 145,
        "timestamp": time.time()
    }

@router.get("/consciousness")
async def get_consciousness_metrics() -> Dict[str, Any]:
    """Get consciousness processing metrics."""
    
    return {
        "consciousness_level": 0.78,
        "awareness_depth": 0.85,
        "meta_cognition_active": True,
        "inference_cycles_per_second": 12.4,
        "belief_updates": 247,
        "prediction_accuracy": 0.91,
        "timestamp": time.time()
    }