"""
DAEDALUS Coordination Service

Central orchestration service managing distributed migration subagents
with independent context windows and iterative self-improvement.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from uuid import uuid4

from ..config import get_migration_config
from ..logging_config import get_migration_logger
from ..models.background_agent import BackgroundAgent, AgentStatus
from ..models.daedalus_coordination import DaedalusCoordination, CoordinatorStatus
from ..models.migration_task import MigrationTask, TaskStatus


class DaedalusCoordinationService:
    """DAEDALUS coordination service for managing distributed migration agents"""

    def __init__(self):
        self.config = get_migration_config()
        self.logger = get_migration_logger()
        self.coordination_instances: Dict[str, DaedalusCoordination] = {}
        self.active_agents: Dict[str, BackgroundAgent] = {}
        self.task_assignments: Dict[str, str] = {}  # task_id -> agent_id mapping

    async def initialize_coordination(
        self,
        coordination_config: Optional[Dict] = None
    ) -> str:
        """
        Initialize a new DAEDALUS coordination instance

        Args:
            coordination_config: Optional coordination configuration

        Returns:
            Coordination instance ID
        """
        coordination_id = str(uuid4())
        config = coordination_config or {}

        coordination = DaedalusCoordination(
            coordination_id=coordination_id,
            coordinator_status=CoordinatorStatus.INITIALIZING,
            performance_metrics={
                'tasks_completed': 0,
                'tasks_failed': 0,
                'average_completion_time': 0.0,
                'agent_utilization': 0.0,
                'coordination_efficiency': 0.0
            },
            learning_state={
                'optimization_cycles': 0,
                'performance_trends': [],
                'best_practices': [],
                'error_patterns': []
            }
        )

        self.coordination_instances[coordination_id] = coordination

        self.logger.info(
            "DAEDALUS coordination initialized",
            coordination_id=coordination_id,
            config=config
        )

        # Start coordination loop
        asyncio.create_task(
            self._coordination_loop(coordination_id)
        )

        return coordination_id

    async def spawn_background_agent(
        self,
        coordination_id: str,
        agent_config: Optional[Dict] = None
    ) -> str:
        """
        Spawn a new background migration agent

        Args:
            coordination_id: DAEDALUS coordination instance ID
            agent_config: Optional agent configuration

        Returns:
            Background agent ID
        """
        if coordination_id not in self.coordination_instances:
            raise ValueError(f"Coordination instance not found: {coordination_id}")

        agent_id = str(uuid4())
        context_window_id = str(uuid4())
        config = agent_config or {}

        agent = BackgroundAgent(
            agent_id=agent_id,
            context_window_id=context_window_id,
            coordinator_id=coordination_id,
            agent_status=AgentStatus.IDLE,
            performance_stats={
                'tasks_completed': 0,
                'tasks_failed': 0,
                'average_task_time': 0.0,
                'success_rate': 0.0,
                'context_switches': 0
            }
        )

        self.active_agents[agent_id] = agent

        # Update coordination instance
        coordination = self.coordination_instances[coordination_id]
        coordination.active_subagents.append({
            'agent_id': agent_id,
            'context_window_id': context_window_id,
            'status': AgentStatus.IDLE.value,
            'spawned_at': datetime.utcnow().isoformat()
        })

        self.logger.info(
            "Background agent spawned",
            coordination_id=coordination_id,
            agent_id=agent_id,
            context_window_id=context_window_id
        )

        return agent_id

    async def assign_task(
        self,
        coordination_id: str,
        task: MigrationTask,
        preferred_agent_id: Optional[str] = None
    ) -> bool:
        """
        Assign a migration task to an available agent

        Args:
            coordination_id: DAEDALUS coordination instance ID
            task: Migration task to assign
            preferred_agent_id: Optional preferred agent ID

        Returns:
            True if task assigned successfully
        """
        if coordination_id not in self.coordination_instances:
            return False

        coordination = self.coordination_instances[coordination_id]

        # Find available agent
        available_agent = None
        if preferred_agent_id and preferred_agent_id in self.active_agents:
            agent = self.active_agents[preferred_agent_id]
            if agent.agent_status == AgentStatus.IDLE and agent.coordinator_id == coordination_id:
                available_agent = agent

        if not available_agent:
            # Find any idle agent for this coordination
            for agent in self.active_agents.values():
                if (agent.coordinator_id == coordination_id and
                    agent.agent_status == AgentStatus.IDLE):
                    available_agent = agent
                    break

        if not available_agent:
            # No agents available, queue the task
            coordination.task_queue.append(task.task_id)
            self.logger.debug(
                "Task queued - no available agents",
                coordination_id=coordination_id,
                task_id=task.task_id
            )
            return False

        # Assign task to agent
        available_agent.current_task_id = task.task_id
        available_agent.assigned_component_id = task.component_id
        available_agent.agent_status = AgentStatus.ANALYZING
        available_agent.last_activity = datetime.utcnow()

        self.task_assignments[task.task_id] = available_agent.agent_id

        # Update task status
        task.task_status = TaskStatus.IN_PROGRESS
        task.agent_id = available_agent.agent_id
        task.started_at = datetime.utcnow()

        self.logger.info(
            "Task assigned to agent",
            coordination_id=coordination_id,
            task_id=task.task_id,
            agent_id=available_agent.agent_id,
            component_id=task.component_id
        )

        # Start task execution
        asyncio.create_task(
            self._execute_agent_task(available_agent.agent_id, task)
        )

        return True

    async def _execute_agent_task(
        self,
        agent_id: str,
        task: MigrationTask
    ) -> None:
        """
        Execute a migration task on a background agent

        Args:
            agent_id: Background agent ID
            task: Migration task to execute
        """
        agent = self.active_agents[agent_id]
        start_time = datetime.utcnow()

        try:
            self.logger.info(
                "Agent starting task execution",
                agent_id=agent_id,
                task_id=task.task_id,
                task_type=task.task_type
            )

            # Phase 1: Analysis
            agent.agent_status = AgentStatus.ANALYZING
            await self._simulate_task_phase("analysis", task)

            # Phase 2: Rewriting
            agent.agent_status = AgentStatus.REWRITING
            await self._simulate_task_phase("rewriting", task)

            # Phase 3: Testing
            agent.agent_status = AgentStatus.TESTING
            await self._simulate_task_phase("testing", task)

            # Phase 4: Reporting
            agent.agent_status = AgentStatus.REPORTING
            await self._simulate_task_phase("reporting", task)

            # Task completed successfully
            completion_time = datetime.utcnow()
            execution_duration = (completion_time - start_time).total_seconds()

            # Update agent stats
            agent.task_history.append(task.task_id)
            agent.performance_stats['tasks_completed'] += 1
            agent.performance_stats['average_task_time'] = (
                (agent.performance_stats['average_task_time'] *
                 (agent.performance_stats['tasks_completed'] - 1) + execution_duration) /
                agent.performance_stats['tasks_completed']
            )
            agent.performance_stats['success_rate'] = (
                agent.performance_stats['tasks_completed'] /
                (agent.performance_stats['tasks_completed'] + agent.performance_stats['tasks_failed'])
            )

            # Update task status
            task.task_status = TaskStatus.COMPLETED
            task.completed_at = completion_time

            # Update coordination metrics
            coordination = self.coordination_instances[agent.coordinator_id]
            coordination.completed_tasks.append(task.task_id)
            coordination.performance_metrics['tasks_completed'] += 1

            self.logger.info(
                "Agent completed task successfully",
                agent_id=agent_id,
                task_id=task.task_id,
                execution_time=execution_duration
            )

        except Exception as e:
            # Task failed
            agent.performance_stats['tasks_failed'] += 1
            task.task_status = TaskStatus.FAILED
            task.add_error(str(e))

            # Update coordination
            coordination = self.coordination_instances[agent.coordinator_id]
            coordination.failed_tasks.append(task.task_id)
            coordination.performance_metrics['tasks_failed'] += 1

            self.logger.error(
                "Agent task execution failed",
                agent_id=agent_id,
                task_id=task.task_id,
                error=str(e)
            )

        finally:
            # Reset agent to idle
            agent.agent_status = AgentStatus.IDLE
            agent.current_task_id = None
            agent.assigned_component_id = None
            agent.last_activity = datetime.utcnow()

            # Remove task assignment
            if task.task_id in self.task_assignments:
                del self.task_assignments[task.task_id]

    async def _simulate_task_phase(self, phase: str, task: MigrationTask) -> None:
        """
        Simulate task phase execution (placeholder for actual implementation)

        Args:
            phase: Task phase name
            task: Migration task
        """
        # Simulate work with variable duration based on task complexity
        base_duration = {
            'analysis': 2.0,
            'rewriting': 5.0,
            'testing': 3.0,
            'reporting': 1.0
        }.get(phase, 1.0)

        # Add some randomness to simulate real work
        import random
        duration = base_duration * (0.5 + random.random())
        await asyncio.sleep(duration)

    async def _coordination_loop(self, coordination_id: str) -> None:
        """
        Main coordination loop for managing agents and optimizing performance

        Args:
            coordination_id: Coordination instance ID
        """
        coordination = self.coordination_instances[coordination_id]
        coordination.coordinator_status = CoordinatorStatus.COORDINATING

        self.logger.info(
            "DAEDALUS coordination loop started",
            coordination_id=coordination_id
        )

        while coordination.coordinator_status == CoordinatorStatus.COORDINATING:
            try:
                # Process queued tasks
                await self._process_task_queue(coordination_id)

                # Monitor agent performance
                await self._monitor_agent_performance(coordination_id)

                # Optimize coordination strategies
                await self._optimize_coordination(coordination_id)

                # Self-improvement cycle
                await self._perform_self_improvement(coordination_id)

                # Wait before next cycle
                await asyncio.sleep(self.config.coordination_cycle_interval)

            except Exception as e:
                self.logger.error(
                    "Coordination loop error",
                    coordination_id=coordination_id,
                    error=str(e)
                )
                await asyncio.sleep(5)  # Brief pause before retry

    async def _process_task_queue(self, coordination_id: str) -> None:
        """Process queued tasks by assigning to available agents"""
        coordination = self.coordination_instances[coordination_id]

        if not coordination.task_queue:
            return

        # Find idle agents
        idle_agents = [
            agent for agent in self.active_agents.values()
            if (agent.coordinator_id == coordination_id and
                agent.agent_status == AgentStatus.IDLE)
        ]

        if not idle_agents:
            return

        # Assign queued tasks to idle agents
        tasks_to_assign = min(len(coordination.task_queue), len(idle_agents))
        for i in range(tasks_to_assign):
            task_id = coordination.task_queue.pop(0)
            # In a real implementation, would retrieve task from storage
            # For now, just log the assignment
            self.logger.debug(
                "Assigning queued task",
                coordination_id=coordination_id,
                task_id=task_id,
                agent_id=idle_agents[i].agent_id
            )

    async def _monitor_agent_performance(self, coordination_id: str) -> None:
        """Monitor and analyze agent performance metrics"""
        coordination = self.coordination_instances[coordination_id]

        # Calculate overall agent utilization
        total_agents = len([
            agent for agent in self.active_agents.values()
            if agent.coordinator_id == coordination_id
        ])

        if total_agents == 0:
            return

        active_agents = len([
            agent for agent in self.active_agents.values()
            if (agent.coordinator_id == coordination_id and
                agent.agent_status != AgentStatus.IDLE)
        ])

        utilization = active_agents / total_agents if total_agents > 0 else 0.0
        coordination.performance_metrics['agent_utilization'] = utilization

    async def _optimize_coordination(self, coordination_id: str) -> None:
        """Optimize coordination strategies based on performance data"""
        coordination = self.coordination_instances[coordination_id]

        # Simple optimization: spawn more agents if utilization is high
        utilization = coordination.performance_metrics.get('agent_utilization', 0.0)

        if utilization > 0.8 and len(coordination.task_queue) > 0:
            # High utilization and queued tasks - spawn more agents
            await self.spawn_background_agent(coordination_id)

        elif utilization < 0.2:
            # Low utilization - could scale down (but keep minimum agents)
            pass

    async def _perform_self_improvement(self, coordination_id: str) -> None:
        """Perform self-improvement and learning updates"""
        coordination = self.coordination_instances[coordination_id]

        # Update learning state
        coordination.learning_state['optimization_cycles'] += 1
        coordination.last_optimization = datetime.utcnow()

        # Analyze performance trends
        current_metrics = coordination.performance_metrics.copy()
        coordination.learning_state['performance_trends'].append({
            'timestamp': datetime.utcnow().isoformat(),
            'metrics': current_metrics
        })

        # Keep only recent trends
        max_trends = 100
        if len(coordination.learning_state['performance_trends']) > max_trends:
            coordination.learning_state['performance_trends'] = \
                coordination.learning_state['performance_trends'][-max_trends:]

    def get_coordination_status(self, coordination_id: str) -> Optional[DaedalusCoordination]:
        """
        Get coordination status

        Args:
            coordination_id: Coordination instance ID

        Returns:
            Coordination status or None if not found
        """
        return self.coordination_instances.get(coordination_id)

    def get_agent_status(self, agent_id: str) -> Optional[BackgroundAgent]:
        """
        Get agent status

        Args:
            agent_id: Agent ID

        Returns:
            Agent status or None if not found
        """
        return self.active_agents.get(agent_id)

    async def shutdown_coordination(self, coordination_id: str) -> bool:
        """
        Shutdown coordination instance and all associated agents

        Args:
            coordination_id: Coordination instance ID

        Returns:
            True if shutdown successful
        """
        if coordination_id not in self.coordination_instances:
            return False

        coordination = self.coordination_instances[coordination_id]
        coordination.coordinator_status = CoordinatorStatus.SHUTTING_DOWN

        # Stop all associated agents
        agents_to_stop = [
            agent_id for agent_id, agent in self.active_agents.items()
            if agent.coordinator_id == coordination_id
        ]

        for agent_id in agents_to_stop:
            del self.active_agents[agent_id]

        # Remove coordination instance
        del self.coordination_instances[coordination_id]

        self.logger.info(
            "DAEDALUS coordination shutdown",
            coordination_id=coordination_id,
            agents_stopped=len(agents_to_stop)
        )

        return True