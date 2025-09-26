"""
Coordination API Endpoints

FastAPI endpoints for DAEDALUS coordination and agent management
including distributed task coordination and agent health monitoring.
"""

from typing import Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from ..services import DaedalusCoordinationService, AgentManagementService
from ..models.background_agent import BackgroundAgent, AgentStatus
from ..models.daedalus_coordination import DaedalusCoordination, CoordinatorStatus


coordination_router = APIRouter(prefix="/api/v1/coordination", tags=["coordination"])


class InitializeCoordinationRequest(BaseModel):
    """Request model for initializing coordination"""
    coordination_config: Optional[Dict] = None


class InitializeCoordinationResponse(BaseModel):
    """Response model for coordination initialization"""
    coordination_id: str
    status: str
    message: str


class SpawnAgentRequest(BaseModel):
    """Request model for spawning background agent"""
    coordination_id: str
    agent_config: Optional[Dict] = None


class SpawnAgentResponse(BaseModel):
    """Response model for agent spawning"""
    agent_id: str
    context_window_id: str
    coordination_id: str
    status: str


class AssignTaskRequest(BaseModel):
    """Request model for task assignment"""
    coordination_id: str
    task_id: str
    component_id: str
    task_type: str
    preferred_agent_id: Optional[str] = None


class AssignTaskResponse(BaseModel):
    """Response model for task assignment"""
    task_id: str
    agent_id: str
    status: str
    message: str


# Initialize services
coordination_service = DaedalusCoordinationService()
agent_service = AgentManagementService()


@coordination_router.post(
    "/initialize",
    response_model=InitializeCoordinationResponse,
    status_code=status.HTTP_201_CREATED
)
async def initialize_coordination(
    request: InitializeCoordinationRequest
) -> InitializeCoordinationResponse:
    """
    Initialize a new DAEDALUS coordination instance

    Creates a new coordination instance for managing distributed
    migration agents with independent context windows.
    """
    try:
        coordination_id = await coordination_service.initialize_coordination(
            coordination_config=request.coordination_config
        )

        return InitializeCoordinationResponse(
            coordination_id=coordination_id,
            status="initialized",
            message=f"DAEDALUS coordination initialized with ID: {coordination_id}"
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize coordination: {str(e)}"
        )


@coordination_router.get(
    "/{coordination_id}",
    response_model=Dict
)
async def get_coordination_status(coordination_id: str) -> Dict:
    """
    Get DAEDALUS coordination status and metrics

    Returns detailed information about coordination instance including
    active agents, task queue status, and performance metrics.
    """
    coordination = coordination_service.get_coordination_status(coordination_id)

    if not coordination:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Coordination instance not found: {coordination_id}"
        )

    return {
        "coordination_id": coordination_id,
        "status": coordination.coordinator_status.value,
        "active_subagents": coordination.active_subagents,
        "task_queue_size": len(coordination.task_queue),
        "completed_tasks": len(coordination.completed_tasks),
        "failed_tasks": len(coordination.failed_tasks),
        "performance_metrics": coordination.performance_metrics,
        "learning_state": {
            "optimization_cycles": coordination.learning_state.get('optimization_cycles', 0),
            "performance_trends_count": len(coordination.learning_state.get('performance_trends', [])),
            "best_practices_count": len(coordination.learning_state.get('best_practices', []))
        },
        "last_optimization": coordination.last_optimization.isoformat()
    }


@coordination_router.post(
    "/{coordination_id}/agents",
    response_model=SpawnAgentResponse,
    status_code=status.HTTP_201_CREATED
)
async def spawn_background_agent(
    coordination_id: str,
    request: SpawnAgentRequest
) -> SpawnAgentResponse:
    """
    Spawn a new background migration agent

    Creates a new agent with isolated context window for executing
    migration tasks independently.
    """
    try:
        agent_id = await coordination_service.spawn_background_agent(
            coordination_id=coordination_id,
            agent_config=request.agent_config
        )

        agent = coordination_service.get_agent_status(agent_id)

        if not agent:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Agent created but not found in registry"
            )

        # Register agent with management service
        await agent_service.register_agent(agent)

        return SpawnAgentResponse(
            agent_id=agent_id,
            context_window_id=agent.context_window_id,
            coordination_id=coordination_id,
            status="spawned"
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to spawn agent: {str(e)}"
        )


@coordination_router.post(
    "/{coordination_id}/assign-task",
    response_model=AssignTaskResponse
)
async def assign_task_to_agent(
    coordination_id: str,
    request: AssignTaskRequest
) -> AssignTaskResponse:
    """
    Assign migration task to available agent

    Assigns a migration task to an idle agent or queues it if no
    agents are available.
    """
    try:
        from ..models.migration_task import MigrationTask, TaskStatus

        # Create task object
        task = MigrationTask(
            task_id=request.task_id,
            task_type=request.task_type,
            component_id=request.component_id,
            pipeline_id="",  # Would be set from request in real implementation
            agent_id="",     # Will be assigned
            task_status=TaskStatus.PENDING
        )

        success = await coordination_service.assign_task(
            coordination_id=coordination_id,
            task=task,
            preferred_agent_id=request.preferred_agent_id
        )

        if success:
            return AssignTaskResponse(
                task_id=request.task_id,
                agent_id=task.agent_id,
                status="assigned",
                message=f"Task assigned to agent {task.agent_id}"
            )
        else:
            return AssignTaskResponse(
                task_id=request.task_id,
                agent_id="",
                status="queued",
                message="Task queued - no available agents"
            )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to assign task: {str(e)}"
        )


@coordination_router.get(
    "/{coordination_id}/agents",
    response_model=List[Dict]
)
async def get_coordination_agents(coordination_id: str) -> List[Dict]:
    """
    Get all agents managed by coordination instance

    Returns list of agents with their status, performance metrics,
    and current task assignments.
    """
    agents = agent_service.get_agents_by_coordinator(coordination_id)

    if not agents:
        # Check if coordination exists
        coordination = coordination_service.get_coordination_status(coordination_id)
        if not coordination:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Coordination instance not found: {coordination_id}"
            )

    return agents


@coordination_router.get(
    "/agents/{agent_id}",
    response_model=Dict
)
async def get_agent_status(agent_id: str) -> Dict:
    """
    Get detailed agent status and health metrics

    Returns comprehensive agent information including health status,
    performance metrics, and context isolation details.
    """
    agent_status = agent_service.get_agent_status(agent_id)

    if not agent_status:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent not found: {agent_id}"
        )

    return agent_status


@coordination_router.delete(
    "/agents/{agent_id}",
    status_code=status.HTTP_200_OK
)
async def unregister_agent(agent_id: str) -> Dict:
    """
    Unregister and stop a background agent

    Safely removes agent from management and coordination,
    ensuring any current tasks are properly handled.
    """
    success = await agent_service.unregister_agent(
        agent_id=agent_id,
        reason="manual_unregistration"
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent not found or could not be unregistered: {agent_id}"
        )

    return {
        "agent_id": agent_id,
        "status": "unregistered",
        "message": "Agent unregistered successfully"
    }


@coordination_router.delete(
    "/{coordination_id}",
    status_code=status.HTTP_200_OK
)
async def shutdown_coordination(coordination_id: str) -> Dict:
    """
    Shutdown coordination instance and all agents

    Gracefully shuts down coordination instance and all associated
    agents, ensuring tasks are properly completed or reassigned.
    """
    success = await coordination_service.shutdown_coordination(coordination_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Coordination instance not found: {coordination_id}"
        )

    return {
        "coordination_id": coordination_id,
        "status": "shutdown",
        "message": "Coordination instance shutdown successfully"
    }


@coordination_router.get(
    "/agents/context-isolation/report",
    response_model=Dict
)
async def get_context_isolation_report() -> Dict:
    """
    Get context isolation report for all agents

    Returns comprehensive report on context window isolation
    including any detected violations or optimization recommendations.
    """
    report = agent_service.get_context_isolation_report()
    return report


@coordination_router.post(
    "/agents/optimize-distribution",
    response_model=Dict
)
async def optimize_agent_distribution() -> Dict:
    """
    Optimize agent distribution across context windows

    Analyzes current agent distribution and provides recommendations
    for improving resource utilization and performance.
    """
    try:
        optimization_results = await agent_service.optimize_agent_distribution()
        return optimization_results

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Optimization failed: {str(e)}"
        )


@coordination_router.get(
    "/health",
    response_model=Dict
)
async def get_coordination_health() -> Dict:
    """
    Get overall coordination system health

    Returns system-wide health metrics including active coordinations,
    agent counts, and performance indicators.
    """
    all_agents = agent_service.get_all_agents()

    # Calculate system-wide metrics
    total_agents = len(all_agents)
    active_agents = len([
        agent for agent in all_agents
        if agent['agent']['agent_status'] != AgentStatus.IDLE.value
    ])

    # Calculate average health metrics
    if all_agents:
        avg_memory = sum(
            agent['health']['memory_usage'] for agent in all_agents
        ) / len(all_agents)
        avg_cpu = sum(
            agent['health']['cpu_usage'] for agent in all_agents
        ) / len(all_agents)
        avg_response_time = sum(
            agent['health']['average_response_time'] for agent in all_agents
        ) / len(all_agents)
    else:
        avg_memory = avg_cpu = avg_response_time = 0.0

    # Count coordination instances
    active_coordinations = len(coordination_service.coordination_instances)

    return {
        "system_status": "healthy" if total_agents > 0 else "no_agents",
        "active_coordinations": active_coordinations,
        "total_agents": total_agents,
        "active_agents": active_agents,
        "agent_utilization": active_agents / total_agents if total_agents > 0 else 0.0,
        "average_metrics": {
            "memory_usage": avg_memory,
            "cpu_usage": avg_cpu,
            "response_time": avg_response_time
        },
        "context_isolation": {
            "total_contexts": len(agent_service.context_isolation_registry),
            "violations": len(agent_service._detect_isolation_violations())
        }
    }