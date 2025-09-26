"""
Migration API Endpoints

FastAPI endpoints for migration pipeline operations including
component discovery, quality assessment, and ThoughtSeed enhancement.
"""

from typing import Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, BackgroundTasks, status
from pydantic import BaseModel

from ..services import (
    ComponentDiscoveryService,
    QualityAssessmentService,
    MigrationPipelineService,
    ThoughtSeedEnhancementService
)
from ..models.legacy_component import LegacyComponent
from ..models.migration_task import MigrationTask, TaskStatus
from ..models.quality_assessment import QualityAssessment


migration_router = APIRouter(prefix="/api/v1/migration", tags=["migration"])


class StartMigrationRequest(BaseModel):
    """Request model for starting migration pipeline"""
    codebase_path: str
    coordinator_id: str
    options: Optional[Dict] = None


class StartMigrationResponse(BaseModel):
    """Response model for migration pipeline start"""
    pipeline_id: str
    status: str
    message: str


class ComponentDiscoveryRequest(BaseModel):
    """Request model for component discovery"""
    codebase_path: str


class ComponentDiscoveryResponse(BaseModel):
    """Response model for component discovery"""
    discovery_id: str
    components: List[LegacyComponent]
    total_files_analyzed: int
    consciousness_components_found: int


class QualityAssessmentRequest(BaseModel):
    """Request model for quality assessment"""
    component_id: str
    assessor_agent_id: str


class QualityAssessmentResponse(BaseModel):
    """Response model for quality assessment"""
    assessment: QualityAssessment
    migration_recommended: bool
    enhancement_opportunities: List[str]
    risk_factors: List[str]


class EnhancementRequest(BaseModel):
    """Request model for ThoughtSeed enhancement"""
    component_id: str
    enhancement_agent_id: str
    thoughtseed_config: Optional[Dict] = None


class EnhancementResponse(BaseModel):
    """Response model for ThoughtSeed enhancement"""
    enhancement_id: str
    status: str
    estimated_completion_time: Optional[str] = None


# Initialize services
discovery_service = ComponentDiscoveryService()
quality_service = QualityAssessmentService()
pipeline_service = MigrationPipelineService()
enhancement_service = ThoughtSeedEnhancementService()


@migration_router.post(
    "/pipeline",
    response_model=StartMigrationResponse,
    status_code=status.HTTP_202_ACCEPTED
)
async def start_migration_pipeline(
    request: StartMigrationRequest,
    background_tasks: BackgroundTasks
) -> StartMigrationResponse:
    """
    Start complete migration pipeline for a codebase

    This endpoint initiates the full migration workflow including:
    - Component discovery and analysis
    - Quality assessment and prioritization
    - Migration task creation and coordination

    The pipeline runs asynchronously in the background.
    """
    try:
        pipeline_id = await pipeline_service.start_migration_pipeline(
            codebase_path=request.codebase_path,
            coordinator_id=request.coordinator_id,
            options=request.options
        )

        return StartMigrationResponse(
            pipeline_id=pipeline_id,
            status="started",
            message=f"Migration pipeline started with ID: {pipeline_id}"
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start migration pipeline: {str(e)}"
        )


@migration_router.get(
    "/pipeline/{pipeline_id}",
    response_model=Dict
)
async def get_pipeline_status(pipeline_id: str) -> Dict:
    """
    Get migration pipeline status and progress

    Returns detailed information about pipeline execution including:
    - Current phase and status
    - Components discovered and assessed
    - Migration tasks created
    - Completion metrics
    """
    pipeline_task = pipeline_service.get_pipeline_status(pipeline_id)

    if not pipeline_task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pipeline not found: {pipeline_id}"
        )

    return {
        "pipeline_id": pipeline_id,
        "status": pipeline_task.task_status.value,
        "created_at": pipeline_task.created_at.isoformat(),
        "updated_at": pipeline_task.updated_at.isoformat(),
        "discovered_components": getattr(pipeline_task, 'discovered_components', 0),
        "migration_candidates": getattr(pipeline_task, 'migration_candidates', 0),
        "created_tasks": getattr(pipeline_task, 'created_tasks', 0),
        "errors": pipeline_task.errors
    }


@migration_router.delete(
    "/pipeline/{pipeline_id}",
    status_code=status.HTTP_200_OK
)
async def cancel_pipeline(pipeline_id: str) -> Dict:
    """
    Cancel an active migration pipeline

    Stops pipeline execution and cleans up associated resources.
    Only active pipelines can be cancelled.
    """
    success = await pipeline_service.cancel_pipeline(pipeline_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pipeline not found or cannot be cancelled: {pipeline_id}"
        )

    return {
        "pipeline_id": pipeline_id,
        "status": "cancelled",
        "message": "Pipeline cancelled successfully"
    }


@migration_router.post(
    "/discover",
    response_model=ComponentDiscoveryResponse,
    status_code=status.HTTP_200_OK
)
async def discover_components(
    request: ComponentDiscoveryRequest
) -> ComponentDiscoveryResponse:
    """
    Discover consciousness components in a legacy codebase

    Analyzes Python files to identify components with consciousness
    functionality patterns including awareness, inference, and memory.
    """
    try:
        components = discovery_service.discover_components(request.codebase_path)

        consciousness_components = len([
            comp for comp in components
            if len(comp.consciousness_patterns) > 0
        ])

        return ComponentDiscoveryResponse(
            discovery_id=str(uuid4()),
            components=components,
            total_files_analyzed=len(components),
            consciousness_components_found=consciousness_components
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Component discovery failed: {str(e)}"
        )


@migration_router.post(
    "/assess",
    response_model=QualityAssessmentResponse,
    status_code=status.HTTP_200_OK
)
async def assess_component_quality(
    request: QualityAssessmentRequest
) -> QualityAssessmentResponse:
    """
    Perform quality assessment on a discovered component

    Evaluates component using consciousness functionality metrics
    and strategic value analysis to determine migration priority.
    """
    try:
        # In a real implementation, would retrieve component from storage
        # For now, create a mock component for demonstration
        from ..models.legacy_component import ConsciousnessFunctionality, StrategicValue

        mock_component = LegacyComponent(
            component_id=request.component_id,
            name="mock_component",
            file_path="/mock/path",
            consciousness_functionality=ConsciousnessFunctionality(
                awareness_score=0.7,
                inference_score=0.6,
                memory_score=0.5
            ),
            strategic_value=StrategicValue(
                uniqueness_score=0.8,
                reusability_score=0.6,
                framework_alignment_score=0.7
            ),
            quality_score=0.65
        )

        assessment = quality_service.assess_component(
            component=mock_component,
            assessor_agent_id=request.assessor_agent_id
        )

        return QualityAssessmentResponse(
            assessment=assessment,
            migration_recommended=assessment.migration_recommended,
            enhancement_opportunities=assessment.enhancement_opportunities,
            risk_factors=assessment.risk_factors
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Quality assessment failed: {str(e)}"
        )


@migration_router.post(
    "/enhance",
    response_model=EnhancementResponse,
    status_code=status.HTTP_202_ACCEPTED
)
async def enhance_component(
    request: EnhancementRequest,
    background_tasks: BackgroundTasks
) -> EnhancementResponse:
    """
    Apply ThoughtSeed enhancement to a component

    Performs consciousness-guided rewrite using ThoughtSeed framework
    to enhance awareness, inference, and memory capabilities.

    Enhancement runs asynchronously in the background.
    """
    try:
        # In a real implementation, would retrieve component from storage
        from ..models.legacy_component import ConsciousnessFunctionality, StrategicValue

        mock_component = LegacyComponent(
            component_id=request.component_id,
            name="mock_component",
            file_path="/mock/path",
            consciousness_functionality=ConsciousnessFunctionality(
                awareness_score=0.5,
                inference_score=0.4,
                memory_score=0.3
            ),
            strategic_value=StrategicValue(
                uniqueness_score=0.6,
                reusability_score=0.5,
                framework_alignment_score=0.4
            ),
            quality_score=0.45
        )

        # Start enhancement asynchronously
        enhancement_result = await enhancement_service.enhance_component(
            component=mock_component,
            enhancement_agent_id=request.enhancement_agent_id,
            thoughtseed_config=request.thoughtseed_config
        )

        return EnhancementResponse(
            enhancement_id=enhancement_result.enhancement_id,
            status=enhancement_result.status.value,
            estimated_completion_time="2024-01-01T12:00:00Z"  # Mock estimate
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Component enhancement failed: {str(e)}"
        )


@migration_router.get(
    "/enhance/{enhancement_id}",
    response_model=Dict
)
async def get_enhancement_status(enhancement_id: str) -> Dict:
    """
    Get ThoughtSeed enhancement status and results

    Returns detailed information about enhancement progress including:
    - Current enhancement status
    - Consciousness improvements achieved
    - Validation metrics
    - Enhanced component details
    """
    enhancement_result = enhancement_service.get_enhancement_status(enhancement_id)

    if not enhancement_result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Enhancement not found: {enhancement_id}"
        )

    response_data = {
        "enhancement_id": enhancement_id,
        "status": enhancement_result.status.value,
        "component_id": enhancement_result.component_id,
        "agent_id": enhancement_result.agent_id,
        "created_at": enhancement_result.created_at.isoformat(),
        "errors": enhancement_result.errors
    }

    # Add completion details if finished
    if enhancement_result.completed_at:
        response_data["completed_at"] = enhancement_result.completed_at.isoformat()

    if enhancement_result.enhanced_component:
        response_data["enhanced_component"] = enhancement_result.enhanced_component.dict()

    if enhancement_result.consciousness_improvements:
        response_data["consciousness_improvements"] = enhancement_result.consciousness_improvements

    if enhancement_result.validation_metrics:
        response_data["validation_metrics"] = enhancement_result.validation_metrics

    return response_data


@migration_router.get(
    "/active",
    response_model=List[Dict]
)
async def get_active_migrations() -> List[Dict]:
    """
    Get all active migration operations

    Returns a list of currently running pipelines and enhancements
    with their status and progress information.
    """
    # Get active pipelines
    active_pipelines = pipeline_service.get_active_pipelines()
    pipeline_data = [
        {
            "type": "pipeline",
            "id": task.pipeline_id,
            "status": task.task_status.value,
            "created_at": task.created_at.isoformat(),
            "component_id": task.component_id
        }
        for task in active_pipelines
    ]

    # Get active enhancements
    active_enhancements = enhancement_service.get_active_enhancements()
    enhancement_data = [
        {
            "type": "enhancement",
            "id": result.enhancement_id,
            "status": result.status.value,
            "created_at": result.created_at.isoformat(),
            "component_id": result.component_id
        }
        for result in active_enhancements
    ]

    return pipeline_data + enhancement_data