"""
Migration Pipeline Service

Core orchestration service managing the end-to-end migration workflow
from component discovery through enhancement to deployment.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional
from uuid import uuid4

from ..config import get_migration_config
from ..logging_config import get_migration_logger
from ..models.legacy_component import LegacyComponent
from ..models.migration_task import MigrationTask, TaskStatus
from ..models.quality_assessment import QualityAssessment
from .component_discovery import ComponentDiscoveryService
from .quality_assessment import QualityAssessmentService


class MigrationPipelineService:
    """Service orchestrating the complete migration pipeline"""

    def __init__(self):
        self.config = get_migration_config()
        self.logger = get_migration_logger()
        self.discovery_service = ComponentDiscoveryService()
        self.quality_service = QualityAssessmentService()
        self.active_migrations: Dict[str, MigrationTask] = {}

    async def start_migration_pipeline(
        self,
        codebase_path: str,
        coordinator_id: str,
        options: Optional[Dict] = None
    ) -> str:
        """
        Start complete migration pipeline for a codebase

        Args:
            codebase_path: Path to legacy codebase
            coordinator_id: DAEDALUS coordinator managing the pipeline
            options: Optional pipeline configuration

        Returns:
            Pipeline ID for tracking progress
        """
        pipeline_id = str(uuid4())
        options = options or {}

        self.logger.info(
            "Starting migration pipeline",
            pipeline_id=pipeline_id,
            codebase_path=codebase_path,
            coordinator_id=coordinator_id
        )

        # Create pipeline task
        pipeline_task = MigrationTask(
            task_id=pipeline_id,
            task_type="pipeline",
            component_id="pipeline",
            pipeline_id=pipeline_id,
            agent_id=coordinator_id,
            task_status=TaskStatus.PENDING
        )

        self.active_migrations[pipeline_id] = pipeline_task

        # Start pipeline execution asynchronously
        asyncio.create_task(
            self._execute_pipeline(pipeline_id, codebase_path, coordinator_id, options)
        )

        return pipeline_id

    async def _execute_pipeline(
        self,
        pipeline_id: str,
        codebase_path: str,
        coordinator_id: str,
        options: Dict
    ) -> None:
        """
        Execute the complete migration pipeline

        Args:
            pipeline_id: Pipeline identifier
            codebase_path: Path to legacy codebase
            coordinator_id: DAEDALUS coordinator ID
            options: Pipeline options
        """
        try:
            # Update status to in progress
            self._update_pipeline_status(pipeline_id, TaskStatus.IN_PROGRESS)

            # Phase 1: Component Discovery
            self.logger.info(
                "Starting component discovery phase",
                pipeline_id=pipeline_id,
                phase="discovery"
            )

            discovered_components = await self._discover_components(
                codebase_path, pipeline_id
            )

            self.logger.info(
                "Component discovery completed",
                pipeline_id=pipeline_id,
                component_count=len(discovered_components)
            )

            # Phase 2: Quality Assessment
            self.logger.info(
                "Starting quality assessment phase",
                pipeline_id=pipeline_id,
                phase="assessment"
            )

            assessed_components = await self._assess_components(
                discovered_components, coordinator_id, pipeline_id
            )

            # Phase 3: Prioritization and Filtering
            migration_candidates = self._prioritize_components(
                assessed_components, options
            )

            self.logger.info(
                "Component prioritization completed",
                pipeline_id=pipeline_id,
                candidate_count=len(migration_candidates)
            )

            # Phase 4: Create Migration Tasks
            migration_tasks = await self._create_migration_tasks(
                migration_candidates, pipeline_id, coordinator_id
            )

            # Update pipeline with completed discovery
            pipeline_task = self.active_migrations[pipeline_id]
            pipeline_task.discovered_components = len(discovered_components)
            pipeline_task.migration_candidates = len(migration_candidates)
            pipeline_task.created_tasks = len(migration_tasks)

            self._update_pipeline_status(pipeline_id, TaskStatus.COMPLETED)

            self.logger.info(
                "Migration pipeline completed successfully",
                pipeline_id=pipeline_id,
                discovered=len(discovered_components),
                candidates=len(migration_candidates),
                tasks_created=len(migration_tasks)
            )

        except Exception as e:
            self.logger.error(
                "Migration pipeline failed",
                pipeline_id=pipeline_id,
                error=str(e)
            )
            self._update_pipeline_status(pipeline_id, TaskStatus.FAILED)
            # Store error details
            if pipeline_id in self.active_migrations:
                self.active_migrations[pipeline_id].add_error(str(e))

    async def _discover_components(
        self,
        codebase_path: str,
        pipeline_id: str
    ) -> List[LegacyComponent]:
        """
        Discover components in the legacy codebase

        Args:
            codebase_path: Path to legacy codebase
            pipeline_id: Pipeline identifier

        Returns:
            List of discovered components
        """
        try:
            # Run discovery in executor to avoid blocking
            loop = asyncio.get_event_loop()
            components = await loop.run_in_executor(
                None,
                self.discovery_service.discover_components,
                codebase_path
            )

            self.logger.info(
                "Component discovery successful",
                pipeline_id=pipeline_id,
                component_count=len(components)
            )

            return components

        except Exception as e:
            self.logger.error(
                "Component discovery failed",
                pipeline_id=pipeline_id,
                error=str(e)
            )
            raise

    async def _assess_components(
        self,
        components: List[LegacyComponent],
        assessor_id: str,
        pipeline_id: str
    ) -> List[QualityAssessment]:
        """
        Assess quality of discovered components

        Args:
            components: Components to assess
            assessor_id: ID of assessing agent
            pipeline_id: Pipeline identifier

        Returns:
            List of quality assessments
        """
        assessments = []

        for component in components:
            try:
                # Run assessment in executor to avoid blocking
                loop = asyncio.get_event_loop()
                assessment = await loop.run_in_executor(
                    None,
                    self.quality_service.assess_component,
                    component,
                    assessor_id
                )

                assessments.append(assessment)

                self.logger.debug(
                    "Component assessed",
                    pipeline_id=pipeline_id,
                    component_id=component.component_id,
                    quality_score=assessment.composite_score
                )

            except Exception as e:
                self.logger.warning(
                    "Component assessment failed",
                    pipeline_id=pipeline_id,
                    component_id=component.component_id,
                    error=str(e)
                )

        return assessments

    def _prioritize_components(
        self,
        assessments: List[QualityAssessment],
        options: Dict
    ) -> List[QualityAssessment]:
        """
        Prioritize components for migration based on quality and options

        Args:
            assessments: Quality assessments
            options: Prioritization options

        Returns:
            Prioritized list of migration candidates
        """
        # Filter by quality threshold
        quality_threshold = options.get(
            'quality_threshold',
            self.config.quality_threshold
        )

        candidates = [
            assessment for assessment in assessments
            if assessment.migration_recommended and
            assessment.composite_score >= quality_threshold
        ]

        # Sort by composite score (highest first)
        candidates.sort(
            key=lambda x: x.composite_score,
            reverse=True
        )

        # Apply limit if specified
        limit = options.get('max_components')
        if limit and limit > 0:
            candidates = candidates[:limit]

        return candidates

    async def _create_migration_tasks(
        self,
        candidates: List[QualityAssessment],
        pipeline_id: str,
        coordinator_id: str
    ) -> List[MigrationTask]:
        """
        Create migration tasks for approved components

        Args:
            candidates: Migration candidates
            pipeline_id: Pipeline identifier
            coordinator_id: DAEDALUS coordinator ID

        Returns:
            List of created migration tasks
        """
        tasks = []

        for assessment in candidates:
            try:
                task = MigrationTask(
                    task_id=str(uuid4()),
                    task_type="component_migration",
                    component_id=assessment.component_id,
                    pipeline_id=pipeline_id,
                    agent_id=coordinator_id,
                    task_status=TaskStatus.PENDING,
                    quality_assessment=assessment
                )

                tasks.append(task)

                self.logger.debug(
                    "Migration task created",
                    pipeline_id=pipeline_id,
                    task_id=task.task_id,
                    component_id=assessment.component_id
                )

            except Exception as e:
                self.logger.error(
                    "Failed to create migration task",
                    pipeline_id=pipeline_id,
                    component_id=assessment.component_id,
                    error=str(e)
                )

        return tasks

    def _update_pipeline_status(
        self,
        pipeline_id: str,
        status: TaskStatus
    ) -> None:
        """Update pipeline status"""
        if pipeline_id in self.active_migrations:
            self.active_migrations[pipeline_id].task_status = status
            self.active_migrations[pipeline_id].updated_at = datetime.utcnow()

    def get_pipeline_status(self, pipeline_id: str) -> Optional[MigrationTask]:
        """
        Get current pipeline status

        Args:
            pipeline_id: Pipeline identifier

        Returns:
            Pipeline task or None if not found
        """
        return self.active_migrations.get(pipeline_id)

    def get_active_pipelines(self) -> List[MigrationTask]:
        """
        Get all active migration pipelines

        Returns:
            List of active pipeline tasks
        """
        return [
            task for task in self.active_migrations.values()
            if task.task_status in [TaskStatus.PENDING, TaskStatus.IN_PROGRESS]
        ]

    async def cancel_pipeline(self, pipeline_id: str) -> bool:
        """
        Cancel an active migration pipeline

        Args:
            pipeline_id: Pipeline identifier

        Returns:
            True if cancelled successfully
        """
        if pipeline_id not in self.active_migrations:
            return False

        pipeline_task = self.active_migrations[pipeline_id]
        if pipeline_task.task_status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
            return False

        self._update_pipeline_status(pipeline_id, TaskStatus.CANCELLED)

        self.logger.info(
            "Migration pipeline cancelled",
            pipeline_id=pipeline_id
        )

        return True