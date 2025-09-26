"""
Monitoring API Endpoints

FastAPI endpoints for system monitoring, performance metrics,
and migration progress tracking.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from ..services import (
    MigrationPipelineService,
    DaedalusCoordinationService,
    AgentManagementService,
    RollbackService
)


monitoring_router = APIRouter(prefix="/api/v1/monitoring", tags=["monitoring"])


class SystemMetricsResponse(BaseModel):
    """Response model for system metrics"""
    timestamp: str
    migration_metrics: Dict
    coordination_metrics: Dict
    agent_metrics: Dict
    rollback_metrics: Dict


class PerformanceMetricsResponse(BaseModel):
    """Response model for performance metrics"""
    timestamp: str
    throughput_metrics: Dict
    latency_metrics: Dict
    error_metrics: Dict
    resource_utilization: Dict


class AlertsResponse(BaseModel):
    """Response model for system alerts"""
    alerts: List[Dict]
    alert_counts: Dict
    last_updated: str


# Initialize services
pipeline_service = MigrationPipelineService()
coordination_service = DaedalusCoordinationService()
agent_service = AgentManagementService()
rollback_service = RollbackService()


@monitoring_router.get(
    "/metrics",
    response_model=SystemMetricsResponse
)
async def get_system_metrics() -> SystemMetricsResponse:
    """
    Get comprehensive system metrics

    Returns detailed metrics across all system components including
    migration pipelines, coordination instances, agents, and rollback operations.
    """
    timestamp = datetime.utcnow().isoformat()

    # Migration metrics
    active_pipelines = pipeline_service.get_active_pipelines()
    migration_metrics = {
        "active_pipelines": len(active_pipelines),
        "total_migrations": len(pipeline_service.active_migrations),
        "pipeline_status_distribution": _get_pipeline_status_distribution(active_pipelines)
    }

    # Coordination metrics
    coordination_instances = coordination_service.coordination_instances
    coordination_metrics = {
        "active_coordinations": len(coordination_instances),
        "total_agents_coordinated": sum(
            len(coord.active_subagents) for coord in coordination_instances.values()
        ),
        "average_tasks_per_coordination": _calculate_average_tasks_per_coordination(coordination_instances)
    }

    # Agent metrics
    all_agents = agent_service.get_all_agents()
    agent_metrics = {
        "total_agents": len(all_agents),
        "agent_status_distribution": _get_agent_status_distribution(all_agents),
        "average_health_metrics": _calculate_average_health_metrics(all_agents),
        "context_isolation_status": agent_service.get_context_isolation_report()
    }

    # Rollback metrics
    rollback_metrics = {
        "total_checkpoints": len(rollback_service.checkpoints),
        "rollback_operations": len(rollback_service.rollback_history),
        "success_rate": _calculate_rollback_success_rate(rollback_service.rollback_history),
        "average_rollback_time": _calculate_average_rollback_time(rollback_service.rollback_history)
    }

    return SystemMetricsResponse(
        timestamp=timestamp,
        migration_metrics=migration_metrics,
        coordination_metrics=coordination_metrics,
        agent_metrics=agent_metrics,
        rollback_metrics=rollback_metrics
    )


@monitoring_router.get(
    "/performance",
    response_model=PerformanceMetricsResponse
)
async def get_performance_metrics() -> PerformanceMetricsResponse:
    """
    Get system performance metrics

    Returns performance indicators including throughput, latency,
    error rates, and resource utilization across the system.
    """
    timestamp = datetime.utcnow().isoformat()

    # Throughput metrics
    throughput_metrics = {
        "components_discovered_per_hour": _calculate_discovery_throughput(),
        "assessments_completed_per_hour": _calculate_assessment_throughput(),
        "enhancements_completed_per_hour": _calculate_enhancement_throughput(),
        "rollbacks_per_hour": _calculate_rollback_throughput()
    }

    # Latency metrics
    all_agents = agent_service.get_all_agents()
    latency_metrics = {
        "average_response_time": _calculate_average_response_time(all_agents),
        "p95_response_time": _calculate_p95_response_time(all_agents),
        "task_assignment_latency": _calculate_task_assignment_latency(),
        "rollback_latency": _calculate_average_rollback_time(rollback_service.rollback_history)
    }

    # Error metrics
    error_metrics = {
        "failed_migrations": _count_failed_migrations(),
        "agent_failures": _count_agent_failures(all_agents),
        "rollback_failures": _count_rollback_failures(rollback_service.rollback_history),
        "error_rate_last_hour": _calculate_error_rate()
    }

    # Resource utilization
    resource_utilization = {
        "average_memory_usage": _calculate_average_memory_usage(all_agents),
        "average_cpu_usage": _calculate_average_cpu_usage(all_agents),
        "agent_utilization": _calculate_agent_utilization(all_agents),
        "context_window_utilization": _calculate_context_utilization()
    }

    return PerformanceMetricsResponse(
        timestamp=timestamp,
        throughput_metrics=throughput_metrics,
        latency_metrics=latency_metrics,
        error_metrics=error_metrics,
        resource_utilization=resource_utilization
    )


@monitoring_router.get(
    "/alerts",
    response_model=AlertsResponse
)
async def get_system_alerts() -> AlertsResponse:
    """
    Get system alerts and warnings

    Returns current system alerts including performance warnings,
    error conditions, and resource constraints.
    """
    alerts = []
    alert_counts = {"critical": 0, "warning": 0, "info": 0}

    # Check for performance alerts
    all_agents = agent_service.get_all_agents()

    # High memory usage alert
    high_memory_agents = [
        agent for agent in all_agents
        if agent['health']['memory_usage'] > 0.8
    ]
    if high_memory_agents:
        alerts.append({
            "id": "high_memory_usage",
            "severity": "warning",
            "message": f"{len(high_memory_agents)} agents have high memory usage (>80%)",
            "affected_agents": [agent['agent']['agent_id'] for agent in high_memory_agents],
            "timestamp": datetime.utcnow().isoformat()
        })
        alert_counts["warning"] += 1

    # Failed health checks alert
    failed_health_agents = [
        agent for agent in all_agents
        if agent['health']['consecutive_failures'] > 2
    ]
    if failed_health_agents:
        alerts.append({
            "id": "failed_health_checks",
            "severity": "critical",
            "message": f"{len(failed_health_agents)} agents have failed health checks",
            "affected_agents": [agent['agent']['agent_id'] for agent in failed_health_agents],
            "timestamp": datetime.utcnow().isoformat()
        })
        alert_counts["critical"] += 1

    # Context isolation violations
    isolation_report = agent_service.get_context_isolation_report()
    if isolation_report["isolation_violations"]:
        alerts.append({
            "id": "context_isolation_violations",
            "severity": "warning",
            "message": f"{len(isolation_report['isolation_violations'])} context isolation violations detected",
            "violations": isolation_report["isolation_violations"],
            "timestamp": datetime.utcnow().isoformat()
        })
        alert_counts["warning"] += 1

    # Long-running rollbacks
    slow_rollbacks = [
        rollback for rollback in rollback_service.rollback_history[-10:]
        if rollback.get('duration_seconds', 0) > 30
    ]
    if slow_rollbacks:
        alerts.append({
            "id": "slow_rollbacks",
            "severity": "warning",
            "message": f"{len(slow_rollbacks)} rollback operations exceeded 30-second target",
            "slow_rollbacks": slow_rollbacks,
            "timestamp": datetime.utcnow().isoformat()
        })
        alert_counts["warning"] += 1

    # System capacity alerts
    coordination_instances = coordination_service.coordination_instances
    overloaded_coordinations = []
    for coord_id, coord in coordination_instances.items():
        if len(coord.task_queue) > 50:  # Arbitrary threshold
            overloaded_coordinations.append(coord_id)

    if overloaded_coordinations:
        alerts.append({
            "id": "coordination_overload",
            "severity": "warning",
            "message": f"{len(overloaded_coordinations)} coordination instances have large task queues",
            "overloaded_coordinations": overloaded_coordinations,
            "timestamp": datetime.utcnow().isoformat()
        })
        alert_counts["warning"] += 1

    return AlertsResponse(
        alerts=alerts,
        alert_counts=alert_counts,
        last_updated=datetime.utcnow().isoformat()
    )


@monitoring_router.get(
    "/migration/{pipeline_id}/progress",
    response_model=Dict
)
async def get_migration_progress(pipeline_id: str) -> Dict:
    """
    Get detailed migration progress for a specific pipeline

    Returns comprehensive progress information including phase completion,
    component processing status, and estimated time remaining.
    """
    pipeline_task = pipeline_service.get_pipeline_status(pipeline_id)

    if not pipeline_task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pipeline not found: {pipeline_id}"
        )

    # Calculate progress metrics
    progress_data = {
        "pipeline_id": pipeline_id,
        "current_status": pipeline_task.task_status.value,
        "created_at": pipeline_task.created_at.isoformat(),
        "updated_at": pipeline_task.updated_at.isoformat(),
        "phases": {
            "discovery": {
                "status": "completed" if hasattr(pipeline_task, 'discovered_components') else "pending",
                "components_found": getattr(pipeline_task, 'discovered_components', 0)
            },
            "assessment": {
                "status": "completed" if hasattr(pipeline_task, 'migration_candidates') else "pending",
                "candidates_identified": getattr(pipeline_task, 'migration_candidates', 0)
            },
            "task_creation": {
                "status": "completed" if hasattr(pipeline_task, 'created_tasks') else "pending",
                "tasks_created": getattr(pipeline_task, 'created_tasks', 0)
            }
        }
    }

    # Calculate estimated completion time
    if pipeline_task.task_status.value in ["in_progress", "pending"]:
        progress_data["estimated_completion"] = _estimate_completion_time(pipeline_task)

    return progress_data


@monitoring_router.get(
    "/health",
    response_model=Dict
)
async def get_system_health() -> Dict:
    """
    Get overall system health status

    Returns high-level health indicators for the entire migration system
    including component status and critical metrics.
    """
    all_agents = agent_service.get_all_agents()
    coordination_instances = coordination_service.coordination_instances

    # Calculate health indicators
    total_agents = len(all_agents)
    healthy_agents = len([
        agent for agent in all_agents
        if agent['health']['consecutive_failures'] == 0
    ])

    agent_health_percentage = (healthy_agents / total_agents * 100) if total_agents > 0 else 0

    # Overall system status
    if agent_health_percentage >= 90:
        system_status = "healthy"
    elif agent_health_percentage >= 70:
        system_status = "warning"
    else:
        system_status = "critical"

    return {
        "overall_status": system_status,
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "migration_service": "healthy",
            "coordination_service": "healthy" if coordination_instances else "inactive",
            "agent_management": "healthy" if total_agents > 0 else "inactive",
            "rollback_service": "healthy"
        },
        "metrics": {
            "total_agents": total_agents,
            "healthy_agents": healthy_agents,
            "agent_health_percentage": agent_health_percentage,
            "active_coordinations": len(coordination_instances),
            "active_migrations": len(pipeline_service.active_migrations)
        }
    }


# Helper functions for metric calculations

def _get_pipeline_status_distribution(pipelines):
    """Calculate pipeline status distribution"""
    status_counts = {}
    for pipeline in pipelines:
        status = pipeline.task_status.value
        status_counts[status] = status_counts.get(status, 0) + 1
    return status_counts


def _calculate_average_tasks_per_coordination(coordinations):
    """Calculate average tasks per coordination"""
    if not coordinations:
        return 0.0
    total_tasks = sum(
        len(coord.completed_tasks) + len(coord.failed_tasks) + len(coord.task_queue)
        for coord in coordinations.values()
    )
    return total_tasks / len(coordinations)


def _get_agent_status_distribution(agents):
    """Calculate agent status distribution"""
    status_counts = {}
    for agent in agents:
        status = agent['agent']['agent_status']
        status_counts[status] = status_counts.get(status, 0) + 1
    return status_counts


def _calculate_average_health_metrics(agents):
    """Calculate average health metrics across agents"""
    if not agents:
        return {"memory_usage": 0.0, "cpu_usage": 0.0, "response_time": 0.0}

    total_memory = sum(agent['health']['memory_usage'] for agent in agents)
    total_cpu = sum(agent['health']['cpu_usage'] for agent in agents)
    total_response = sum(agent['health']['average_response_time'] for agent in agents)

    return {
        "memory_usage": total_memory / len(agents),
        "cpu_usage": total_cpu / len(agents),
        "response_time": total_response / len(agents)
    }


def _calculate_rollback_success_rate(rollback_history):
    """Calculate rollback success rate"""
    if not rollback_history:
        return 1.0
    successful = len([r for r in rollback_history if r.get('success', False)])
    return successful / len(rollback_history)


def _calculate_average_rollback_time(rollback_history):
    """Calculate average rollback time"""
    if not rollback_history:
        return 0.0
    times = [r.get('duration_seconds', 0) for r in rollback_history]
    return sum(times) / len(times)


def _calculate_discovery_throughput():
    """Calculate component discovery throughput (mock)"""
    return 120.0  # Mock: 120 components per hour


def _calculate_assessment_throughput():
    """Calculate assessment completion throughput (mock)"""
    return 80.0  # Mock: 80 assessments per hour


def _calculate_enhancement_throughput():
    """Calculate enhancement completion throughput (mock)"""
    return 25.0  # Mock: 25 enhancements per hour


def _calculate_rollback_throughput():
    """Calculate rollback operations throughput (mock)"""
    return 10.0  # Mock: 10 rollbacks per hour


def _calculate_average_response_time(agents):
    """Calculate average response time across agents"""
    if not agents:
        return 0.0
    times = [agent['health']['average_response_time'] for agent in agents]
    return sum(times) / len(times)


def _calculate_p95_response_time(agents):
    """Calculate 95th percentile response time (mock)"""
    avg_time = _calculate_average_response_time(agents)
    return avg_time * 1.5  # Mock: assume p95 is 1.5x average


def _calculate_task_assignment_latency():
    """Calculate task assignment latency (mock)"""
    return 2.5  # Mock: 2.5 seconds average


def _count_failed_migrations():
    """Count failed migration operations (mock)"""
    return 3  # Mock count


def _count_agent_failures(agents):
    """Count agent failures"""
    return len([
        agent for agent in agents
        if agent['health']['consecutive_failures'] > 0
    ])


def _count_rollback_failures(rollback_history):
    """Count rollback failures"""
    return len([r for r in rollback_history if not r.get('success', True)])


def _calculate_error_rate():
    """Calculate error rate in last hour (mock)"""
    return 0.05  # Mock: 5% error rate


def _calculate_average_memory_usage(agents):
    """Calculate average memory usage"""
    if not agents:
        return 0.0
    return sum(agent['health']['memory_usage'] for agent in agents) / len(agents)


def _calculate_average_cpu_usage(agents):
    """Calculate average CPU usage"""
    if not agents:
        return 0.0
    return sum(agent['health']['cpu_usage'] for agent in agents) / len(agents)


def _calculate_agent_utilization(agents):
    """Calculate agent utilization percentage"""
    if not agents:
        return 0.0
    active_agents = len([
        agent for agent in agents
        if agent['agent']['agent_status'] != 'idle'
    ])
    return active_agents / len(agents)


def _calculate_context_utilization():
    """Calculate context window utilization (mock)"""
    return 0.65  # Mock: 65% utilization


def _estimate_completion_time(pipeline_task):
    """Estimate pipeline completion time (mock)"""
    base_time = datetime.utcnow() + timedelta(hours=2)
    return base_time.isoformat()