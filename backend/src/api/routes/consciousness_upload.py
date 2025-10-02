"""
Consciousness-Aware Document Upload Endpoint
Integrates with Dionysus consciousness pipeline and attractor basin dynamics
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List, Dict, Any
import sys
from pathlib import Path

# Add consciousness pipeline to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "extensions" / "context_engineering"))

from consciousness_enhanced_pipeline import ConsciousnessEnhancedPipeline
from attractor_basin_dynamics import AttractorBasinManager
import asyncio
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/consciousness", tags=["consciousness"])

# Initialize consciousness pipeline (singleton)
_consciousness_pipeline = None
_attractor_manager = None

def get_consciousness_pipeline() -> ConsciousnessEnhancedPipeline:
    global _consciousness_pipeline
    if _consciousness_pipeline is None:
        _consciousness_pipeline = ConsciousnessEnhancedPipeline()
    return _consciousness_pipeline

def get_attractor_manager() -> AttractorBasinManager:
    global _attractor_manager
    if _attractor_manager is None:
        _attractor_manager = AttractorBasinManager()
    return _attractor_manager

@router.post("/upload/document")
async def upload_document_with_consciousness(
    file: UploadFile = File(...),
) -> Dict[str, Any]:
    """
    Upload document and process through consciousness pipeline.
    
    Flow:
    1. Document received via Daedalus gateway
    2. Attractor basin activated based on semantic content  
    3. Consciousness processing (active inference, knowledge gaps)
    4. Experience stored in Neo4j + Redis
    5. Return consciousness metadata
    """
    try:
        # Read document content
        content = await file.read()
        text_content = content.decode('utf-8', errors='ignore')
        
        logger.info(f"🧠 Processing {file.filename} through consciousness pipeline")
        
        # Get pipeline and manager
        pipeline = get_consciousness_pipeline()
        attractor_mgr = get_attractor_manager()
        
        # Process through consciousness pipeline (YOUR ACTUAL CODE)
        consciousness_result = await pipeline.process_incoming_data(
            data=text_content,
            context=f"uploaded_file:{file.filename}",
            data_type="document"
        )
        
        # Integrate with attractor basins (YOUR ACTUAL CODE)
        thoughtseed_id = f"ts_{file.filename}_{consciousness_result.get('timestamp', '')}"
        basin_integration = await attractor_mgr.integrate_thoughtseed(
            thoughtseed_id=thoughtseed_id,
            concept_description=text_content[:500],  # First 500 chars as concept
            semantic_embedding=consciousness_result.get('consciousness_embedding', [])
        )
        
        return {
            "status": "consciousness_processed",
            "filename": file.filename,
            "thoughtseed_id": thoughtseed_id,
            "consciousness_state": {
                "coherence_level": consciousness_result.get('coherence_score', 0.0),
                "attractor_basin": basin_integration.get('primary_basin', 'unknown'),
                "semantic_richness": consciousness_result.get('semantic_density', 0.0),
                "knowledge_gaps_detected": len(consciousness_result.get('knowledge_gaps', [])),
                "processing_depth": consciousness_result.get('processing_depth', 0.0)
            },
            "attractor_dynamics": {
                "basin_influence": basin_integration.get('influence_type', 'unknown'),
                "basin_strength": basin_integration.get('basin_strength', 0.0),
                "new_basin_created": basin_integration.get('new_basin_created', False)
            },
            "storage": {
                "neo4j_node_id": consciousness_result.get('neo4j_id', 'pending'),
                "redis_keys": consciousness_result.get('redis_keys', [])
            }
        }
        
    except Exception as e:
        logger.error(f"Consciousness processing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Consciousness processing failed: {str(e)}")

@router.get("/attractor-basins")
async def get_attractor_basins() -> Dict[str, Any]:
    """Get current attractor basin landscape"""
    try:
        attractor_mgr = get_attractor_manager()
        basins = await attractor_mgr.get_all_basins()
        
        return {
            "basins": [
                {
                    "basin_id": b.basin_id,
                    "center_concept": b.center_concept,
                    "strength": b.strength,
                    "thoughtseeds_count": len(b.thoughtseeds),
                    "formation_time": b.formation_timestamp
                }
                for b in basins
            ],
            "total_basins": len(basins)
        }
    except Exception as e:
        logger.error(f"Failed to get basins: {e}")
        raise HTTPException(status_code=500, detail=str(e))

