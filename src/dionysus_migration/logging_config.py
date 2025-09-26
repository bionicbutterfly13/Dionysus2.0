"""
Logging configuration for Dionysus Migration System

Provides structured logging for distributed agent coordination,
consciousness metrics, and migration process monitoring.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import structlog
from rich.console import Console
from rich.logging import RichHandler

from .config import get_logging_config


def setup_logging(
    level: Optional[str] = None,
    file_path: Optional[str] = None,
    enable_structured: bool = True
) -> None:
    """
    Setup logging configuration for the migration system

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        file_path: Optional file path for log output
        enable_structured: Enable structured logging with structlog
    """
    log_config = get_logging_config()

    # Use provided level or default from config
    log_level = level or log_config.level
    log_file = file_path or log_config.file_path

    # Setup handlers
    handlers = []

    # Console handler with rich formatting
    console = Console(stderr=True)
    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=True,
        enable_link_path=True,
        markup=True
    )
    rich_handler.setLevel(getattr(logging, log_level.upper()))
    handlers.append(rich_handler)

    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_formatter = logging.Formatter(log_config.format)
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        handlers=handlers,
        force=True
    )

    # Setup structured logging if enabled
    if enable_structured:
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.StackInfoRenderer(),
                structlog.dev.set_exc_info,
                structlog.processors.TimeStamper(fmt="ISO"),
                structlog.dev.ConsoleRenderer()
            ],
            wrapper_class=structlog.make_filtering_bound_logger(
                getattr(logging, log_level.upper())
            ),
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger instance

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured structured logger
    """
    return structlog.get_logger(name)


def get_migration_logger() -> structlog.stdlib.BoundLogger:
    """Get logger for migration operations"""
    return get_logger("dionysus_migration.migration")


def get_coordination_logger() -> structlog.stdlib.BoundLogger:
    """Get logger for DAEDALUS coordination"""
    return get_logger("dionysus_migration.coordination")


def get_consciousness_logger() -> structlog.stdlib.BoundLogger:
    """Get logger for consciousness metrics"""
    return get_logger("dionysus_migration.consciousness")


def get_thoughtseed_logger() -> structlog.stdlib.BoundLogger:
    """Get logger for ThoughtSeed integration"""
    return get_logger("dionysus_migration.thoughtseed")


def get_api_logger() -> structlog.stdlib.BoundLogger:
    """Get logger for API operations"""
    return get_logger("dionysus_migration.api")


class ConsciousnessMetricsLogger:
    """
    Specialized logger for consciousness metrics and component analysis
    """

    def __init__(self):
        self.logger = get_consciousness_logger()
        self.config = get_logging_config()

    def log_component_analysis(
        self,
        component_id: str,
        awareness_score: float,
        inference_score: float,
        memory_score: float,
        quality_score: float
    ) -> None:
        """Log component consciousness analysis results"""
        if self.config.log_consciousness_metrics:
            self.logger.info(
                "Component consciousness analysis completed",
                component_id=component_id,
                awareness_score=awareness_score,
                inference_score=inference_score,
                memory_score=memory_score,
                quality_score=quality_score,
                event_type="consciousness_analysis"
            )

    def log_enhancement_result(
        self,
        component_id: str,
        enhancement_id: str,
        before_metrics: dict,
        after_metrics: dict,
        enhancement_type: str
    ) -> None:
        """Log ThoughtSeed enhancement results"""
        if self.config.log_consciousness_metrics:
            self.logger.info(
                "ThoughtSeed enhancement completed",
                component_id=component_id,
                enhancement_id=enhancement_id,
                before_metrics=before_metrics,
                after_metrics=after_metrics,
                enhancement_type=enhancement_type,
                event_type="thoughtseed_enhancement"
            )

    def log_consciousness_degradation(
        self,
        component_id: str,
        previous_score: float,
        current_score: float,
        threshold: float
    ) -> None:
        """Log consciousness degradation warnings"""
        self.logger.warning(
            "Consciousness degradation detected",
            component_id=component_id,
            previous_score=previous_score,
            current_score=current_score,
            threshold=threshold,
            event_type="consciousness_degradation",
            alert_level="warning"
        )


class CoordinationMetricsLogger:
    """
    Specialized logger for DAEDALUS coordination and agent management
    """

    def __init__(self):
        self.logger = get_coordination_logger()
        self.config = get_logging_config()

    def log_agent_spawn(
        self,
        agent_id: str,
        context_window_id: str,
        coordinator_id: str
    ) -> None:
        """Log background agent spawn events"""
        if self.config.log_agent_coordination:
            self.logger.info(
                "Background agent spawned",
                agent_id=agent_id,
                context_window_id=context_window_id,
                coordinator_id=coordinator_id,
                event_type="agent_spawn"
            )

    def log_task_assignment(
        self,
        agent_id: str,
        task_id: str,
        component_id: str,
        task_type: str
    ) -> None:
        """Log task assignment to agents"""
        if self.config.log_agent_coordination:
            self.logger.info(
                "Task assigned to agent",
                agent_id=agent_id,
                task_id=task_id,
                component_id=component_id,
                task_type=task_type,
                event_type="task_assignment"
            )

    def log_coordination_performance(
        self,
        coordinator_id: str,
        active_agents: int,
        queue_size: int,
        throughput: float,
        success_rate: float
    ) -> None:
        """Log coordination performance metrics"""
        if self.config.log_agent_coordination:
            self.logger.info(
                "Coordination performance update",
                coordinator_id=coordinator_id,
                active_agents=active_agents,
                queue_size=queue_size,
                throughput=throughput,
                success_rate=success_rate,
                event_type="coordination_performance"
            )

    def log_context_isolation_violation(
        self,
        agent_id: str,
        context_window_id: str,
        violation_type: str,
        details: str
    ) -> None:
        """Log context isolation violations"""
        self.logger.error(
            "Context isolation violation detected",
            agent_id=agent_id,
            context_window_id=context_window_id,
            violation_type=violation_type,
            details=details,
            event_type="context_violation",
            alert_level="error"
        )


class MigrationMetricsLogger:
    """
    Specialized logger for migration process tracking
    """

    def __init__(self):
        self.logger = get_migration_logger()

    def log_migration_start(
        self,
        pipeline_id: str,
        total_components: int,
        quality_threshold: float
    ) -> None:
        """Log migration pipeline start"""
        self.logger.info(
            "Migration pipeline started",
            pipeline_id=pipeline_id,
            total_components=total_components,
            quality_threshold=quality_threshold,
            event_type="migration_start"
        )

    def log_component_approval(
        self,
        component_id: str,
        approved: bool,
        approver: str,
        approval_notes: str
    ) -> None:
        """Log component approval decisions"""
        self.logger.info(
            "Component approval decision",
            component_id=component_id,
            approved=approved,
            approver=approver,
            approval_notes=approval_notes,
            event_type="component_approval"
        )

    def log_rollback_execution(
        self,
        component_id: str,
        rollback_id: str,
        duration_seconds: float,
        success: bool
    ) -> None:
        """Log component rollback operations"""
        log_level = "info" if success else "error"
        getattr(self.logger, log_level)(
            "Component rollback executed",
            component_id=component_id,
            rollback_id=rollback_id,
            duration_seconds=duration_seconds,
            success=success,
            event_type="component_rollback"
        )

    def log_zero_downtime_violation(
        self,
        component_id: str,
        downtime_seconds: float,
        threshold_seconds: float
    ) -> None:
        """Log zero downtime requirement violations"""
        self.logger.error(
            "Zero downtime requirement violated",
            component_id=component_id,
            downtime_seconds=downtime_seconds,
            threshold_seconds=threshold_seconds,
            event_type="downtime_violation",
            alert_level="critical"
        )


# Initialize logging on module import
setup_logging()

# Provide logger instances for common use
migration_logger = MigrationMetricsLogger()
consciousness_logger = ConsciousnessMetricsLogger()
coordination_logger = CoordinationMetricsLogger()