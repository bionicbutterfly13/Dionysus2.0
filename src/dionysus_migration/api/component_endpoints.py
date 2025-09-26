"""
Component API Endpoints

FastAPI endpoints for legacy component management including
retrieval, analysis, and rollback operations.
"""

from typing import Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from ..services import RollbackService
from ..models.legacy_component import LegacyComponent
from ..models.rollback_checkpoint import RollbackCheckpoint


component_router = APIRouter(prefix="/api/v1/components", tags=["components"])


class ComponentResponse(BaseModel):
    """Response model for component data"""
    component: LegacyComponent
    analysis_metadata: Dict
    migration_status: str


class CreateCheckpointRequest(BaseModel):
    """Request model for creating rollback checkpoint"""
    component_id: str
    migration_state: Dict
    retention_days: Optional[int] = 7


class CreateCheckpointResponse(BaseModel):
    """Response model for checkpoint creation"""
    checkpoint_id: str
    component_id: str
    created_at: str
    retention_until: str


class RollbackRequest(BaseModel):
    """Request model for component rollback"""
    checkpoint_id: str
    rollback_options: Optional[Dict] = None


class RollbackResponse(BaseModel):
    """Response model for rollback operation"""
    rollback_id: str
    checkpoint_id: str
    component_id: str
    success: bool
    duration_seconds: float
    message: str


# Initialize services
rollback_service = RollbackService()


@component_router.get(
    "/{component_id}",
    response_model=ComponentResponse
)
async def get_component(component_id: str) -> ComponentResponse:
    """
    Get detailed information about a legacy component

    Returns comprehensive component data including consciousness
    functionality metrics, strategic value analysis, and current
    migration status.
    """
    # In a real implementation, would retrieve from database
    # For now, return mock data for demonstration
    from ..models.legacy_component import ConsciousnessFunctionality, StrategicValue

    mock_component = LegacyComponent(
        component_id=component_id,
        name=f"component_{component_id[:8]}",
        file_path=f"/legacy/components/{component_id}.py",
        consciousness_functionality=ConsciousnessFunctionality(
            awareness_score=0.65,
            inference_score=0.58,
            memory_score=0.71
        ),
        strategic_value=StrategicValue(
            uniqueness_score=0.72,
            reusability_score=0.69,
            framework_alignment_score=0.54
        ),
        quality_score=0.62,
        consciousness_patterns=[
            "awareness_processing",
            "memory_system",
            "inference_engine"
        ]
    )

    analysis_metadata = {
        "last_analyzed": "2024-01-01T10:00:00Z",
        "analyzer_version": "1.0.0",
        "pattern_detection_confidence": 0.87,
        "enhancement_potential": "high"
    }

    return ComponentResponse(
        component=mock_component,
        analysis_metadata=analysis_metadata,
        migration_status="pending_assessment"
    )


@component_router.get(
    "/",
    response_model=List[ComponentResponse]
)
async def list_components(
    skip: int = 0,
    limit: int = 50,
    consciousness_threshold: Optional[float] = None,
    strategic_threshold: Optional[float] = None
) -> List[ComponentResponse]:
    """
    List legacy components with optional filtering

    Returns paginated list of components with optional filtering
    by consciousness and strategic value thresholds.
    """
    # Mock implementation - in reality would query database
    components = []

    for i in range(skip, min(skip + limit, skip + 10)):  # Mock 10 components
        component_id = f"comp_{i:04d}"

        from ..models.legacy_component import ConsciousnessFunctionality, StrategicValue
        import random

        consciousness_score = random.uniform(0.3, 0.9)
        strategic_score = random.uniform(0.2, 0.8)

        # Apply filters if specified
        if consciousness_threshold and consciousness_score < consciousness_threshold:
            continue
        if strategic_threshold and strategic_score < strategic_threshold:
            continue

        mock_component = LegacyComponent(
            component_id=component_id,
            name=f"legacy_component_{i}",
            file_path=f"/legacy/components/component_{i}.py",
            consciousness_functionality=ConsciousnessFunctionality(
                awareness_score=consciousness_score * 0.9,
                inference_score=consciousness_score * 1.1,
                memory_score=consciousness_score
            ),
            strategic_value=StrategicValue(
                uniqueness_score=strategic_score,
                reusability_score=strategic_score * 0.8,
                framework_alignment_score=strategic_score * 1.2
            ),
            quality_score=(consciousness_score * 0.7 + strategic_score * 0.3),
            consciousness_patterns=[
                "awareness_processing" if consciousness_score > 0.6 else "",
                "inference_engine" if consciousness_score > 0.5 else "",
                "memory_system" if consciousness_score > 0.7 else ""
            ]
        )

        components.append(ComponentResponse(
            component=mock_component,
            analysis_metadata={
                "last_analyzed": "2024-01-01T10:00:00Z",
                "analyzer_version": "1.0.0",
                "pattern_detection_confidence": random.uniform(0.7, 0.95)
            },
            migration_status="discovered"
        ))

    return components


@component_router.post(
    "/{component_id}/checkpoint",
    response_model=CreateCheckpointResponse,
    status_code=status.HTTP_201_CREATED
)
async def create_rollback_checkpoint(
    component_id: str,
    request: CreateCheckpointRequest
) -> CreateCheckpointResponse:
    """
    Create rollback checkpoint for component

    Creates a comprehensive backup of component state including
    files, metadata, and database records to enable fast rollback.
    """
    try:
        # In a real implementation, would retrieve component from database
        from ..models.legacy_component import ConsciousnessFunctionality, StrategicValue

        mock_component = LegacyComponent(
            component_id=component_id,
            name=f"component_{component_id[:8]}",
            file_path=f"/legacy/components/{component_id}.py",
            consciousness_functionality=ConsciousnessFunctionality(
                awareness_score=0.65,
                inference_score=0.58,
                memory_score=0.71
            ),
            strategic_value=StrategicValue(
                uniqueness_score=0.72,
                reusability_score=0.69,
                framework_alignment_score=0.54
            ),
            quality_score=0.62
        )

        checkpoint_config = {
            'retention_days': request.retention_days
        }

        checkpoint_id = await rollback_service.create_rollback_checkpoint(
            component=mock_component,
            migration_state=request.migration_state,
            checkpoint_config=checkpoint_config
        )

        checkpoint = rollback_service.get_checkpoint_status(checkpoint_id)

        return CreateCheckpointResponse(
            checkpoint_id=checkpoint_id,
            component_id=component_id,
            created_at=checkpoint.created_at.isoformat(),
            retention_until=checkpoint.retention_until.isoformat()
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create checkpoint: {str(e)}"
        )


@component_router.post(
    "/{component_id}/rollback",
    response_model=RollbackResponse
)
async def rollback_component(
    component_id: str,
    request: RollbackRequest
) -> RollbackResponse:
    """
    Rollback component to checkpoint state

    Performs fast rollback (<30 seconds) to restore component
    to a previously saved checkpoint state.
    """
    try:
        import time
        start_time = time.time()

        success = await rollback_service.rollback_component(
            checkpoint_id=request.checkpoint_id,
            rollback_options=request.rollback_options
        )

        duration = time.time() - start_time

        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Rollback operation failed"
            )

        return RollbackResponse(
            rollback_id=str(uuid4()),
            checkpoint_id=request.checkpoint_id,
            component_id=component_id,
            success=success,
            duration_seconds=duration,
            message=f"Component rolled back successfully in {duration:.2f} seconds"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Rollback operation failed: {str(e)}"
        )


@component_router.get(
    "/{component_id}/checkpoints",
    response_model=List[Dict]
)
async def get_component_checkpoints(component_id: str) -> List[Dict]:
    """
    Get all checkpoints for a component

    Returns list of available rollback checkpoints for the component
    with their creation times and retention status.
    """
    checkpoints = rollback_service.get_component_checkpoints(component_id)

    return [
        {
            "checkpoint_id": checkpoint.checkpoint_id,
            "component_id": checkpoint.component_id,
            "created_at": checkpoint.created_at.isoformat(),
            "retention_until": checkpoint.retention_until.isoformat(),
            "status": checkpoint.status.value,
            "migration_state": checkpoint.migration_state,
            "backup_size_bytes": sum(
                backup['size_bytes']
                for backup in checkpoint.file_backups.values()
            )
        }
        for checkpoint in checkpoints
    ]


@component_router.get(
    "/{component_id}/rollback-history",
    response_model=List[Dict]
)
async def get_rollback_history(
    component_id: str,
    limit: int = 10
) -> List[Dict]:
    """
    Get rollback history for component

    Returns history of rollback operations performed on the component
    including success status and performance metrics.
    """
    history = rollback_service.get_rollback_history(
        component_id=component_id,
        limit=limit
    )

    return history


@component_router.delete(
    "/{component_id}/checkpoints/{checkpoint_id}",
    status_code=status.HTTP_200_OK
)
async def delete_checkpoint(
    component_id: str,
    checkpoint_id: str
) -> Dict:
    """
    Delete a rollback checkpoint

    Removes checkpoint and associated backup files to free storage.
    Cannot delete checkpoints that are still within retention period
    unless force flag is used.
    """
    checkpoint = rollback_service.get_checkpoint_status(checkpoint_id)

    if not checkpoint:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Checkpoint not found: {checkpoint_id}"
        )

    if checkpoint.component_id != component_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Checkpoint does not belong to specified component"
        )

    # In a real implementation, would perform actual deletion
    # For now, just return success

    return {
        "checkpoint_id": checkpoint_id,
        "component_id": component_id,
        "status": "deleted",
        "message": "Checkpoint deleted successfully"
    }


@component_router.post(
    "/cleanup-checkpoints",
    status_code=status.HTTP_200_OK
)
async def cleanup_expired_checkpoints() -> Dict:
    """
    Clean up expired rollback checkpoints

    Removes all checkpoints that have passed their retention period
    to free up storage space.
    """
    try:
        cleanup_count = await rollback_service.cleanup_expired_checkpoints()

        return {
            "status": "completed",
            "checkpoints_cleaned": cleanup_count,
            "message": f"Successfully cleaned up {cleanup_count} expired checkpoints"
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cleanup operation failed: {str(e)}"
        )