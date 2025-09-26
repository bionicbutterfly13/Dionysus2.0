"""
Background Agent Management Service

Service for managing lifecycle, health monitoring, and coordination
of background migration agents within their isolated context windows.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from uuid import uuid4

from ..config import get_migration_config
from ..logging_config import get_migration_logger
from ..models.background_agent import BackgroundAgent, AgentStatus
from ..models.migration_task import MigrationTask, TaskStatus


class AgentManagementService:
    """Service for managing background migration agents"""

    def __init__(self):
        self.config = get_migration_config()
        self.logger = get_migration_logger()
        self.managed_agents: Dict[str, BackgroundAgent] = {}
        self.agent_health_data: Dict[str, Dict] = {}
        self.context_isolation_registry: Dict[str, Set[str]] = {}  # context_id -> agent_ids

    async def register_agent(
        self,
        agent: BackgroundAgent,
        health_check_config: Optional[Dict] = None
    ) -> bool:
        """
        Register a background agent for management

        Args:
            agent: Background agent to register
            health_check_config: Optional health monitoring configuration

        Returns:
            True if registration successful
        """
        if agent.agent_id in self.managed_agents:
            self.logger.warning(
                "Agent already registered",
                agent_id=agent.agent_id
            )
            return False

        # Verify context isolation
        if not await self._verify_context_isolation(agent):
            self.logger.error(
                "Context isolation verification failed",
                agent_id=agent.agent_id,
                context_window_id=agent.context_window_id
            )
            return False

        # Register agent
        self.managed_agents[agent.agent_id] = agent

        # Initialize health monitoring
        health_config = health_check_config or {}
        self.agent_health_data[agent.agent_id] = {
            'last_health_check': datetime.utcnow(),
            'health_check_interval': health_config.get('interval_seconds', 30),
            'consecutive_failures': 0,
            'max_failures': health_config.get('max_failures', 3),
            'memory_usage': 0.0,
            'cpu_usage': 0.0,
            'context_switches': 0,
            'response_times': []
        }

        # Register context isolation
        context_id = agent.context_window_id
        if context_id not in self.context_isolation_registry:
            self.context_isolation_registry[context_id] = set()
        self.context_isolation_registry[context_id].add(agent.agent_id)

        # Start health monitoring
        asyncio.create_task(
            self._monitor_agent_health(agent.agent_id)
        )

        self.logger.info(
            "Agent registered successfully",
            agent_id=agent.agent_id,
            context_window_id=agent.context_window_id,
            coordinator_id=agent.coordinator_id
        )

        return True

    async def _verify_context_isolation(self, agent: BackgroundAgent) -> bool:
        """
        Verify that agent operates in isolated context window

        Args:
            agent: Agent to verify

        Returns:
            True if context isolation verified
        """
        # Check if context window ID is unique enough
        if not agent.context_window_id or len(agent.context_window_id) < 10:
            return False

        # Verify context isolation flag
        if not agent.context_isolation:
            return False

        # Check for context conflicts
        existing_contexts = set(
            a.context_window_id for a in self.managed_agents.values()
            if a.agent_id != agent.agent_id
        )

        if agent.context_window_id in existing_contexts:
            self.logger.warning(
                "Context window ID collision detected",
                agent_id=agent.agent_id,
                context_window_id=agent.context_window_id
            )
            return False

        return True

    async def _monitor_agent_health(self, agent_id: str) -> None:
        """
        Monitor agent health and performance

        Args:
            agent_id: Agent to monitor
        """
        while agent_id in self.managed_agents:
            try:
                agent = self.managed_agents[agent_id]
                health_data = self.agent_health_data[agent_id]

                # Perform health check
                health_status = await self._perform_health_check(agent)

                if health_status['healthy']:
                    health_data['consecutive_failures'] = 0

                    # Update performance metrics
                    health_data['memory_usage'] = health_status.get('memory_usage', 0.0)
                    health_data['cpu_usage'] = health_status.get('cpu_usage', 0.0)

                    # Track response times
                    response_time = health_status.get('response_time', 0.0)
                    health_data['response_times'].append(response_time)

                    # Keep only recent response times
                    if len(health_data['response_times']) > 100:
                        health_data['response_times'] = health_data['response_times'][-100:]

                else:
                    health_data['consecutive_failures'] += 1

                    self.logger.warning(
                        "Agent health check failed",
                        agent_id=agent_id,
                        consecutive_failures=health_data['consecutive_failures'],
                        error=health_status.get('error')
                    )

                    # Handle unhealthy agent
                    if health_data['consecutive_failures'] >= health_data['max_failures']:
                        await self._handle_unhealthy_agent(agent_id)

                health_data['last_health_check'] = datetime.utcnow()

                # Wait for next health check
                await asyncio.sleep(health_data['health_check_interval'])

            except Exception as e:
                self.logger.error(
                    "Health monitoring error",
                    agent_id=agent_id,
                    error=str(e)
                )
                await asyncio.sleep(30)  # Wait before retry

    async def _perform_health_check(self, agent: BackgroundAgent) -> Dict:
        """
        Perform health check on agent

        Args:
            agent: Agent to check

        Returns:
            Health status dictionary
        """
        start_time = datetime.utcnow()

        try:
            # Check agent responsiveness
            if agent.agent_status not in [status for status in AgentStatus]:
                return {
                    'healthy': False,
                    'error': 'Invalid agent status'
                }

            # Check last activity
            time_since_activity = (datetime.utcnow() - agent.last_activity).total_seconds()
            if time_since_activity > 300:  # 5 minutes
                return {
                    'healthy': False,
                    'error': f'No activity for {time_since_activity} seconds'
                }

            # Check context isolation
            if not agent.context_isolation:
                return {
                    'healthy': False,
                    'error': 'Context isolation compromised'
                }

            # Simulate resource usage checks
            import random
            memory_usage = random.uniform(0.1, 0.8)  # Simulated memory usage
            cpu_usage = random.uniform(0.05, 0.6)     # Simulated CPU usage

            response_time = (datetime.utcnow() - start_time).total_seconds()

            return {
                'healthy': True,
                'memory_usage': memory_usage,
                'cpu_usage': cpu_usage,
                'response_time': response_time,
                'last_activity': agent.last_activity.isoformat()
            }

        except Exception as e:
            return {
                'healthy': False,
                'error': str(e)
            }

    async def _handle_unhealthy_agent(self, agent_id: str) -> None:
        """
        Handle unhealthy agent by attempting recovery or replacement

        Args:
            agent_id: Unhealthy agent ID
        """
        agent = self.managed_agents.get(agent_id)
        if not agent:
            return

        self.logger.warning(
            "Handling unhealthy agent",
            agent_id=agent_id,
            status=agent.agent_status.value
        )

        # Attempt to reassign current task if any
        if agent.current_task_id:
            await self._reassign_agent_task(agent)

        # Remove agent from management
        await self.unregister_agent(agent_id, reason="health_failure")

        # Notify coordinator about agent failure
        self.logger.log_agent_failure(
            agent_id=agent_id,
            coordinator_id=agent.coordinator_id,
            failure_reason="consecutive_health_check_failures"
        )

    async def _reassign_agent_task(self, failed_agent: BackgroundAgent) -> None:
        """
        Reassign task from failed agent to another available agent

        Args:
            failed_agent: Agent that failed
        """
        if not failed_agent.current_task_id:
            return

        self.logger.info(
            "Reassigning task from failed agent",
            agent_id=failed_agent.agent_id,
            task_id=failed_agent.current_task_id
        )

        # In a real implementation, this would:
        # 1. Retrieve the task from storage
        # 2. Reset task status to PENDING
        # 3. Notify coordinator to reassign
        # For now, just log the reassignment need

        self.logger.info(
            "Task marked for reassignment",
            task_id=failed_agent.current_task_id,
            failed_agent_id=failed_agent.agent_id
        )

    async def unregister_agent(
        self,
        agent_id: str,
        reason: str = "manual"
    ) -> bool:
        """
        Unregister an agent from management

        Args:
            agent_id: Agent to unregister
            reason: Reason for unregistration

        Returns:
            True if unregistration successful
        """
        if agent_id not in self.managed_agents:
            return False

        agent = self.managed_agents[agent_id]

        # Remove from context isolation registry
        context_id = agent.context_window_id
        if context_id in self.context_isolation_registry:
            self.context_isolation_registry[context_id].discard(agent_id)
            if not self.context_isolation_registry[context_id]:
                del self.context_isolation_registry[context_id]

        # Clean up health data
        if agent_id in self.agent_health_data:
            del self.agent_health_data[agent_id]

        # Remove from managed agents
        del self.managed_agents[agent_id]

        self.logger.info(
            "Agent unregistered",
            agent_id=agent_id,
            reason=reason,
            context_window_id=agent.context_window_id
        )

        return True

    def get_agent_status(self, agent_id: str) -> Optional[Dict]:
        """
        Get comprehensive agent status including health metrics

        Args:
            agent_id: Agent ID

        Returns:
            Agent status dictionary or None if not found
        """
        if agent_id not in self.managed_agents:
            return None

        agent = self.managed_agents[agent_id]
        health_data = self.agent_health_data.get(agent_id, {})

        # Calculate average response time
        response_times = health_data.get('response_times', [])
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0.0

        return {
            'agent': agent.dict(),
            'health': {
                'last_check': health_data.get('last_health_check'),
                'consecutive_failures': health_data.get('consecutive_failures', 0),
                'memory_usage': health_data.get('memory_usage', 0.0),
                'cpu_usage': health_data.get('cpu_usage', 0.0),
                'average_response_time': avg_response_time,
                'context_switches': health_data.get('context_switches', 0)
            }
        }

    def get_all_agents(self) -> List[Dict]:
        """
        Get status of all managed agents

        Returns:
            List of agent status dictionaries
        """
        return [
            self.get_agent_status(agent_id)
            for agent_id in self.managed_agents.keys()
        ]

    def get_agents_by_coordinator(self, coordinator_id: str) -> List[Dict]:
        """
        Get agents managed by specific coordinator

        Args:
            coordinator_id: Coordinator ID

        Returns:
            List of agent status dictionaries for the coordinator
        """
        return [
            self.get_agent_status(agent_id)
            for agent_id, agent in self.managed_agents.items()
            if agent.coordinator_id == coordinator_id
        ]

    def get_context_isolation_report(self) -> Dict:
        """
        Get context isolation report

        Returns:
            Context isolation status report
        """
        return {
            'total_contexts': len(self.context_isolation_registry),
            'total_agents': len(self.managed_agents),
            'context_details': {
                context_id: {
                    'agent_count': len(agent_ids),
                    'agent_ids': list(agent_ids)
                }
                for context_id, agent_ids in self.context_isolation_registry.items()
            },
            'isolation_violations': self._detect_isolation_violations()
        }

    def _detect_isolation_violations(self) -> List[Dict]:
        """
        Detect potential context isolation violations

        Returns:
            List of detected violations
        """
        violations = []

        # Check for contexts with multiple agents (potential violation)
        for context_id, agent_ids in self.context_isolation_registry.items():
            if len(agent_ids) > 1:
                violations.append({
                    'type': 'multiple_agents_same_context',
                    'context_id': context_id,
                    'agent_ids': list(agent_ids),
                    'severity': 'high'
                })

        # Check for agents without proper isolation flags
        for agent_id, agent in self.managed_agents.items():
            if not agent.context_isolation:
                violations.append({
                    'type': 'isolation_flag_disabled',
                    'agent_id': agent_id,
                    'context_id': agent.context_window_id,
                    'severity': 'medium'
                })

        return violations

    async def optimize_agent_distribution(self) -> Dict:
        """
        Optimize agent distribution across context windows

        Returns:
            Optimization results
        """
        optimization_results = {
            'agents_optimized': 0,
            'contexts_consolidated': 0,
            'performance_improvement': 0.0,
            'recommendations': []
        }

        # Analyze current distribution
        context_utilization = {}
        for context_id, agent_ids in self.context_isolation_registry.items():
            active_agents = sum(
                1 for agent_id in agent_ids
                if self.managed_agents[agent_id].agent_status != AgentStatus.IDLE
            )
            context_utilization[context_id] = {
                'total_agents': len(agent_ids),
                'active_agents': active_agents,
                'utilization': active_agents / len(agent_ids) if agent_ids else 0.0
            }

        # Generate recommendations
        for context_id, utilization in context_utilization.items():
            if utilization['utilization'] < 0.3 and utilization['total_agents'] > 1:
                optimization_results['recommendations'].append({
                    'type': 'consolidate_context',
                    'context_id': context_id,
                    'current_agents': utilization['total_agents'],
                    'recommended_agents': max(1, utilization['total_agents'] // 2)
                })

            elif utilization['utilization'] > 0.8:
                optimization_results['recommendations'].append({
                    'type': 'scale_context',
                    'context_id': context_id,
                    'current_agents': utilization['total_agents'],
                    'recommended_agents': utilization['total_agents'] + 1
                })

        return optimization_results