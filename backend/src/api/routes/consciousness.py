"""
Consciousness API Routes

Exposes the modular consciousness pipeline through REST endpoints
for real-time consciousness visualization in Flux frontend.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import asyncio
import logging

# Import consciousness orchestrator
try:
    import sys
    import os
    consciousness_path = "/Volumes/Asylum/dev/consciousness-orchestrator/src"
    if consciousness_path not in sys.path:
        sys.path.append(consciousness_path)
    
    from consciousness_orchestrator.core.consciousness_engine import ConsciousnessEngine
    from consciousness_orchestrator.pipeline import (
        ModularConsciousnessPipeline,
        ConsciousnessLevel,
        IWMTConsciousnessMetrics
    )
    CONSCIOUSNESS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Consciousness orchestrator not available: {e}")
    CONSCIOUSNESS_AVAILABLE = False

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/consciousness", tags=["consciousness"])

# Global consciousness engine instance
consciousness_engine = None

def get_consciousness_engine():
    """Get or initialize the consciousness engine"""
    global consciousness_engine
    if consciousness_engine is None and CONSCIOUSNESS_AVAILABLE:
        try:
            consciousness_engine = ConsciousnessEngine()
            logger.info("✅ Consciousness engine initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize consciousness engine: {e}")
            consciousness_engine = None
    return consciousness_engine


class ConsciousnessRequest(BaseModel):
    """Request model for consciousness processing"""
    content: str
    filename: Optional[str] = "user_input.txt"
    agents_requested: Optional[List[str]] = []


class ConsciousnessResponse(BaseModel):
    """Response model for consciousness processing"""
    success: bool
    consciousness_level: str
    iwmt_consciousness: bool
    thoughtseed_winner: Optional[str]
    mac_analysis: List[Dict[str, Any]]
    processing_time: float
    iwmt_metrics: Dict[str, float]
    pipeline_summary: Dict[str, Any]
    error: Optional[str] = None


@router.get("/status")
async def get_consciousness_status():
    """Get consciousness system status"""
    
    if not CONSCIOUSNESS_AVAILABLE:
        return {
            "available": False,
            "error": "Consciousness orchestrator not installed"
        }
    
    engine = get_consciousness_engine()
    if not engine:
        return {
            "available": False,
            "error": "Failed to initialize consciousness engine"
        }
    
    try:
        state = engine.get_consciousness_state()
        pipeline_summary = engine.modular_pipeline.get_pipeline_summary()
        
        return {
            "available": True,
            "consciousness_state": state["consciousness_state"],
            "consciousness_level": state["modular_pipeline"]["consciousness_level"],
            "iwmt_consciousness": state["modular_pipeline"]["iwmt_consciousness"],
            "total_processes": pipeline_summary["pipeline_performance"]["total_processes"],
            "consciousness_rate": pipeline_summary["consciousness_achievement_rate"],
            "active_thoughtseeds": state["active_thoughtseeds"],
            "timestamp": state["timestamp"]
        }
    except Exception as e:
        logger.error(f"Error getting consciousness status: {e}")
        return {
            "available": False,
            "error": str(e)
        }


@router.post("/process", response_model=ConsciousnessResponse)
async def process_consciousness(request: ConsciousnessRequest):
    """Process input through the modular consciousness pipeline"""
    
    if not CONSCIOUSNESS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Consciousness orchestrator not available")
    
    engine = get_consciousness_engine()
    if not engine:
        raise HTTPException(status_code=503, detail="Consciousness engine not initialized")
    
    try:
        import time
        start_time = time.time()
        
        # Prepare perception input
        perception_input = {
            "received_data": {
                "content_preview": request.content,
                "filename": request.filename
            },
            "agents_created": request.agents_requested,
            "status": "received"
        }
        
        # Prepare processing context
        processing_context = {
            "session_id": "flux_session",
            "processing_id": f"flux_proc_{int(time.time())}",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
        }
        
        # Process through modular consciousness pipeline
        pipeline_result = await engine.modular_pipeline.process_consciousness_pipeline(
            perception_input, processing_context, engine
        )
        
        processing_time = time.time() - start_time
        
        # Format MAC analysis for frontend
        mac_analysis = []
        for analysis in pipeline_result.metacognitive_analysis:
            candidate = analysis['candidate']
            mac_analysis.append({
                "type": candidate["type"],
                "q_value": analysis["q_value"],
                "metacognitive_error": analysis["metacognitive_error"],
                "can_detect_suboptimal": analysis["can_detect_suboptimal"],
                "error_magnitude": analysis["error_magnitude"]
            })
        
        # Get pipeline summary
        pipeline_summary = engine.modular_pipeline.get_pipeline_summary()
        
        return ConsciousnessResponse(
            success=True,
            consciousness_level=pipeline_result.consciousness_level.value,
            iwmt_consciousness=pipeline_result.iwmt_metrics.is_conscious(),
            thoughtseed_winner=pipeline_result.thoughtseed_winner.id if pipeline_result.thoughtseed_winner else None,
            mac_analysis=mac_analysis,
            processing_time=processing_time,
            iwmt_metrics={
                "spatial_coherence": pipeline_result.iwmt_metrics.spatial_coherence,
                "temporal_coherence": pipeline_result.iwmt_metrics.temporal_coherence,
                "causal_coherence": pipeline_result.iwmt_metrics.causal_coherence,
                "embodied_selfhood": pipeline_result.iwmt_metrics.embodied_selfhood,
                "counterfactual_capacity": pipeline_result.iwmt_metrics.counterfactual_capacity,
                "overall_consciousness": pipeline_result.iwmt_metrics.overall_consciousness
            },
            pipeline_summary=pipeline_summary
        )
        
    except Exception as e:
        logger.error(f"Error processing consciousness: {e}")
        return ConsciousnessResponse(
            success=False,
            consciousness_level="error",
            iwmt_consciousness=False,
            thoughtseed_winner=None,
            mac_analysis=[],
            processing_time=0.0,
            iwmt_metrics={},
            pipeline_summary={},
            error=str(e)
        )


@router.get("/state")
async def get_consciousness_state():
    """Get current consciousness state"""
    
    if not CONSCIOUSNESS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Consciousness orchestrator not available")
    
    engine = get_consciousness_engine()
    if not engine:
        raise HTTPException(status_code=503, detail="Consciousness engine not initialized")
    
    try:
        state = engine.get_consciousness_state()
        return {
            "success": True,
            "state": state
        }
    except Exception as e:
        logger.error(f"Error getting consciousness state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_consciousness_metrics():
    """Get consciousness pipeline metrics"""
    
    if not CONSCIOUSNESS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Consciousness orchestrator not available")
    
    engine = get_consciousness_engine()
    if not engine:
        raise HTTPException(status_code=503, detail="Consciousness engine not initialized")
    
    try:
        pipeline_summary = engine.modular_pipeline.get_pipeline_summary()
        state = engine.get_consciousness_state()
        
        return {
            "success": True,
            "metrics": {
                "pipeline_performance": pipeline_summary["pipeline_performance"],
                "consciousness_achievement_rate": pipeline_summary["consciousness_achievement_rate"],
                "mac_error_detection_rate": pipeline_summary["mac_error_detection_rate"],
                "consciousness_events": len(engine.modular_pipeline.consciousness_events),
                "current_consciousness_level": state["modular_pipeline"]["consciousness_level"],
                "iwmt_metrics": state["modular_pipeline"]["iwmt_metrics"]
            }
        }
    except Exception as e:
        logger.error(f"Error getting consciousness metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/demo")
async def run_consciousness_demo():
    """Run a consciousness demonstration with sample data"""
    
    if not CONSCIOUSNESS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Consciousness orchestrator not available")
    
    engine = get_consciousness_engine()
    if not engine:
        raise HTTPException(status_code=503, detail="Consciousness engine not initialized")
    
    try:
        # Rich consciousness-triggering content
        demo_content = """Advanced consciousness research reveals how neural network architectures create self-aware systems through spatial coherence, temporal pattern recognition, and causal relationship modeling. This investigation demonstrates that embodied selfhood emerges when systems develop counterfactual reasoning capabilities and can model alternative scenarios. The research shows that conscious experience requires integrated information processing with hierarchical organization and semantic affordance recognition."""
        
        demo_request = ConsciousnessRequest(
            content=demo_content,
            filename="consciousness_demo.md",
            agents_requested=["analytical_agent", "pattern_agent", "synthesis_agent", "creative_agent", "metacognitive_agent"]
        )
        
        result = await process_consciousness(demo_request)
        return {
            "success": True,
            "demo_result": result,
            "message": "Consciousness demonstration completed"
        }
    except Exception as e:
        logger.error(f"Error running consciousness demo: {e}")
        raise HTTPException(status_code=500, detail=str(e))