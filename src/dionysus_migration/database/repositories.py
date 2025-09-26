"""
Database Repositories

Data access layer providing CRUD operations and business logic
queries for migration system entities.
"""

import json
from datetime import datetime
from typing import Dict, List, Optional
from uuid import uuid4

from .connection import DatabaseConnection
from ..models.legacy_component import LegacyComponent, ConsciousnessFunctionality, StrategicValue
from ..models.migration_task import MigrationTask, TaskStatus
from ..models.quality_assessment import QualityAssessment
from ..models.rollback_checkpoint import RollbackCheckpoint, CheckpointStatus
from ..logging_config import get_migration_logger


class BaseRepository:
    """Base repository with common functionality"""

    def __init__(self, db_connection: DatabaseConnection):
        self.db = db_connection
        self.logger = get_migration_logger()


class ComponentRepository(BaseRepository):
    """Repository for legacy component data access"""

    async def create(self, component: LegacyComponent) -> str:
        """
        Create a new legacy component record

        Args:
            component: Legacy component to create

        Returns:
            Component ID
        """
        query = """
        INSERT INTO legacy_components (
            component_id, name, file_path,
            consciousness_awareness_score, consciousness_inference_score,
            consciousness_memory_score, consciousness_composite_score,
            strategic_uniqueness_score, strategic_reusability_score,
            strategic_framework_alignment_score, strategic_composite_score,
            quality_score, analysis_status, source_code_hash,
            file_size_bytes, consciousness_patterns, dependencies
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        parameters = (
            component.component_id,
            component.name,
            component.file_path,
            component.consciousness_functionality.awareness_score,
            component.consciousness_functionality.inference_score,
            component.consciousness_functionality.memory_score,
            component.consciousness_functionality.composite_score,
            component.strategic_value.uniqueness_score,
            component.strategic_value.reusability_score,
            component.strategic_value.framework_alignment_score,
            component.strategic_value.composite_score,
            component.quality_score,
            component.analysis_status.value,
            component.source_code_hash,
            component.file_size_bytes,
            json.dumps(component.consciousness_patterns),
            json.dumps(component.dependencies)
        )

        await self.db.execute(query, parameters)

        self.logger.debug(
            "Component created in database",
            component_id=component.component_id
        )

        return component.component_id

    async def get_by_id(self, component_id: str) -> Optional[LegacyComponent]:
        """
        Get component by ID

        Args:
            component_id: Component identifier

        Returns:
            Legacy component or None if not found
        """
        query = "SELECT * FROM legacy_components WHERE component_id = ?"
        row = await self.db.fetchone(query, (component_id,))

        if row:
            return self._row_to_component(row)

        return None

    async def get_by_quality_threshold(
        self,
        min_quality: float,
        limit: Optional[int] = None
    ) -> List[LegacyComponent]:
        """
        Get components above quality threshold

        Args:
            min_quality: Minimum quality score
            limit: Optional result limit

        Returns:
            List of qualifying components
        """
        query = """
        SELECT * FROM legacy_components
        WHERE quality_score >= ?
        ORDER BY quality_score DESC
        """

        if limit:
            query += f" LIMIT {limit}"

        rows = await self.db.fetchall(query, (min_quality,))
        return [self._row_to_component(row) for row in rows]

    async def get_by_consciousness_threshold(
        self,
        min_consciousness: float,
        limit: Optional[int] = None
    ) -> List[LegacyComponent]:
        """
        Get components above consciousness threshold

        Args:
            min_consciousness: Minimum consciousness score
            limit: Optional result limit

        Returns:
            List of qualifying components
        """
        query = """
        SELECT * FROM legacy_components
        WHERE consciousness_composite_score >= ?
        ORDER BY consciousness_composite_score DESC
        """

        if limit:
            query += f" LIMIT {limit}"

        rows = await self.db.fetchall(query, (min_consciousness,))
        return [self._row_to_component(row) for row in rows]

    async def update_quality_score(self, component_id: str, quality_score: float) -> bool:
        """
        Update component quality score

        Args:
            component_id: Component identifier
            quality_score: New quality score

        Returns:
            True if updated successfully
        """
        query = """
        UPDATE legacy_components
        SET quality_score = ?, updated_at = CURRENT_TIMESTAMP
        WHERE component_id = ?
        """

        cursor = await self.db.execute(query, (quality_score, component_id))
        return cursor.rowcount > 0

    async def delete(self, component_id: str) -> bool:
        """
        Delete component

        Args:
            component_id: Component identifier

        Returns:
            True if deleted successfully
        """
        query = "DELETE FROM legacy_components WHERE component_id = ?"
        cursor = await self.db.execute(query, (component_id,))
        return cursor.rowcount > 0

    async def get_statistics(self) -> Dict:
        """
        Get component statistics

        Returns:
            Statistics dictionary
        """
        stats = {}

        # Total count
        query = "SELECT COUNT(*) as total FROM legacy_components"
        row = await self.db.fetchone(query)
        stats["total_components"] = row["total"]

        # Average scores
        query = """
        SELECT
            AVG(consciousness_composite_score) as avg_consciousness,
            AVG(strategic_composite_score) as avg_strategic,
            AVG(quality_score) as avg_quality
        FROM legacy_components
        """
        row = await self.db.fetchone(query)
        stats.update(row)

        # Score distributions
        query = """
        SELECT
            COUNT(CASE WHEN quality_score >= 0.8 THEN 1 END) as high_quality,
            COUNT(CASE WHEN quality_score >= 0.6 AND quality_score < 0.8 THEN 1 END) as medium_quality,
            COUNT(CASE WHEN quality_score < 0.6 THEN 1 END) as low_quality
        FROM legacy_components
        """
        row = await self.db.fetchone(query)
        stats.update(row)

        return stats

    def _row_to_component(self, row: Dict) -> LegacyComponent:
        """Convert database row to LegacyComponent"""
        from ..models.legacy_component import AnalysisStatus

        consciousness_functionality = ConsciousnessFunctionality(
            awareness_score=row["consciousness_awareness_score"],
            inference_score=row["consciousness_inference_score"],
            memory_score=row["consciousness_memory_score"]
        )

        strategic_value = StrategicValue(
            uniqueness_score=row["strategic_uniqueness_score"],
            reusability_score=row["strategic_reusability_score"],
            framework_alignment_score=row["strategic_framework_alignment_score"]
        )

        return LegacyComponent(
            component_id=row["component_id"],
            name=row["name"],
            file_path=row["file_path"],
            consciousness_functionality=consciousness_functionality,
            strategic_value=strategic_value,
            quality_score=row["quality_score"],
            analysis_status=AnalysisStatus(row["analysis_status"]),
            source_code_hash=row["source_code_hash"],
            file_size_bytes=row["file_size_bytes"],
            consciousness_patterns=json.loads(row["consciousness_patterns"] or "[]"),
            dependencies=json.loads(row["dependencies"] or "[]")
        )


class MigrationTaskRepository(BaseRepository):
    """Repository for migration task data access"""

    async def create(self, task: MigrationTask) -> str:
        """
        Create a new migration task

        Args:
            task: Migration task to create

        Returns:
            Task ID
        """
        query = """
        INSERT INTO migration_tasks (
            task_id, task_type, component_id, pipeline_id,
            agent_id, task_status, started_at, completed_at, errors
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        parameters = (
            task.task_id,
            task.task_type,
            task.component_id,
            task.pipeline_id,
            task.agent_id,
            task.task_status.value,
            task.started_at,
            task.completed_at,
            json.dumps(task.errors)
        )

        await self.db.execute(query, parameters)
        return task.task_id

    async def get_by_id(self, task_id: str) -> Optional[MigrationTask]:
        """
        Get task by ID

        Args:
            task_id: Task identifier

        Returns:
            Migration task or None if not found
        """
        query = "SELECT * FROM migration_tasks WHERE task_id = ?"
        row = await self.db.fetchone(query, (task_id,))

        if row:
            return self._row_to_task(row)

        return None

    async def get_by_status(self, status: TaskStatus) -> List[MigrationTask]:
        """
        Get tasks by status

        Args:
            status: Task status to filter by

        Returns:
            List of matching tasks
        """
        query = "SELECT * FROM migration_tasks WHERE task_status = ?"
        rows = await self.db.fetchall(query, (status.value,))
        return [self._row_to_task(row) for row in rows]

    async def get_by_agent(self, agent_id: str) -> List[MigrationTask]:
        """
        Get tasks assigned to agent

        Args:
            agent_id: Agent identifier

        Returns:
            List of agent's tasks
        """
        query = "SELECT * FROM migration_tasks WHERE agent_id = ?"
        rows = await self.db.fetchall(query, (agent_id,))
        return [self._row_to_task(row) for row in rows]

    async def update_status(
        self,
        task_id: str,
        status: TaskStatus,
        agent_id: Optional[str] = None
    ) -> bool:
        """
        Update task status

        Args:
            task_id: Task identifier
            status: New task status
            agent_id: Optional agent assignment

        Returns:
            True if updated successfully
        """
        if agent_id:
            query = """
            UPDATE migration_tasks
            SET task_status = ?, agent_id = ?, updated_at = CURRENT_TIMESTAMP
            WHERE task_id = ?
            """
            parameters = (status.value, agent_id, task_id)
        else:
            query = """
            UPDATE migration_tasks
            SET task_status = ?, updated_at = CURRENT_TIMESTAMP
            WHERE task_id = ?
            """
            parameters = (status.value, task_id)

        cursor = await self.db.execute(query, parameters)
        return cursor.rowcount > 0

    async def mark_completed(self, task_id: str) -> bool:
        """
        Mark task as completed

        Args:
            task_id: Task identifier

        Returns:
            True if updated successfully
        """
        query = """
        UPDATE migration_tasks
        SET task_status = ?, completed_at = CURRENT_TIMESTAMP,
            updated_at = CURRENT_TIMESTAMP
        WHERE task_id = ?
        """

        cursor = await self.db.execute(query, (TaskStatus.COMPLETED.value, task_id))
        return cursor.rowcount > 0

    async def add_error(self, task_id: str, error: str) -> bool:
        """
        Add error to task

        Args:
            task_id: Task identifier
            error: Error message

        Returns:
            True if updated successfully
        """
        # Get current errors
        task = await self.get_by_id(task_id)
        if not task:
            return False

        errors = task.errors.copy()
        errors.append(error)

        query = """
        UPDATE migration_tasks
        SET errors = ?, updated_at = CURRENT_TIMESTAMP
        WHERE task_id = ?
        """

        cursor = await self.db.execute(query, (json.dumps(errors), task_id))
        return cursor.rowcount > 0

    def _row_to_task(self, row: Dict) -> MigrationTask:
        """Convert database row to MigrationTask"""
        return MigrationTask(
            task_id=row["task_id"],
            task_type=row["task_type"],
            component_id=row["component_id"],
            pipeline_id=row["pipeline_id"],
            agent_id=row["agent_id"],
            task_status=TaskStatus(row["task_status"]),
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else datetime.utcnow(),
            started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
            completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
            errors=json.loads(row["errors"] or "[]")
        )


class QualityAssessmentRepository(BaseRepository):
    """Repository for quality assessment data access"""

    async def create(self, assessment: QualityAssessment) -> str:
        """
        Create a new quality assessment

        Args:
            assessment: Quality assessment to create

        Returns:
            Assessment ID
        """
        assessment_id = str(uuid4())

        query = """
        INSERT INTO quality_assessments (
            assessment_id, component_id, consciousness_impact,
            strategic_value, composite_score, assessment_method,
            assessor_agent_id, migration_recommended,
            enhancement_opportunities, risk_factors
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        parameters = (
            assessment_id,
            assessment.component_id,
            json.dumps(assessment.consciousness_impact.dict()) if assessment.consciousness_impact else "{}",
            json.dumps(assessment.strategic_value.dict()) if assessment.strategic_value else "{}",
            assessment.composite_score,
            assessment.assessment_method,
            assessment.assessor_agent_id,
            assessment.migration_recommended,
            json.dumps(assessment.enhancement_opportunities),
            json.dumps(assessment.risk_factors)
        )

        await self.db.execute(query, parameters)
        return assessment_id

    async def get_by_component(self, component_id: str) -> List[QualityAssessment]:
        """
        Get assessments for component

        Args:
            component_id: Component identifier

        Returns:
            List of quality assessments
        """
        query = """
        SELECT * FROM quality_assessments
        WHERE component_id = ?
        ORDER BY created_at DESC
        """

        rows = await self.db.fetchall(query, (component_id,))
        return [self._row_to_assessment(row) for row in rows]

    async def get_latest_by_component(self, component_id: str) -> Optional[QualityAssessment]:
        """
        Get latest assessment for component

        Args:
            component_id: Component identifier

        Returns:
            Latest quality assessment or None
        """
        query = """
        SELECT * FROM quality_assessments
        WHERE component_id = ?
        ORDER BY created_at DESC
        LIMIT 1
        """

        row = await self.db.fetchone(query, (component_id,))
        if row:
            return self._row_to_assessment(row)

        return None

    def _row_to_assessment(self, row: Dict) -> QualityAssessment:
        """Convert database row to QualityAssessment"""
        # This is a simplified conversion
        # In a real implementation, would properly reconstruct the complex objects
        assessment = QualityAssessment(
            component_id=row["component_id"],
            composite_score=row["composite_score"],
            assessment_method=row["assessment_method"],
            assessor_agent_id=row["assessor_agent_id"],
            migration_recommended=bool(row["migration_recommended"]),
            enhancement_opportunities=json.loads(row["enhancement_opportunities"] or "[]"),
            risk_factors=json.loads(row["risk_factors"] or "[]")
        )

        return assessment


class CheckpointRepository(BaseRepository):
    """Repository for rollback checkpoint data access"""

    async def create(self, checkpoint: RollbackCheckpoint) -> str:
        """
        Create a new rollback checkpoint

        Args:
            checkpoint: Rollback checkpoint to create

        Returns:
            Checkpoint ID
        """
        query = """
        INSERT INTO rollback_checkpoints (
            checkpoint_id, component_id, migration_state,
            file_backups, metadata_backup, database_backup,
            status, retention_until
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """

        parameters = (
            checkpoint.checkpoint_id,
            checkpoint.component_id,
            json.dumps(checkpoint.migration_state),
            json.dumps(checkpoint.file_backups),
            checkpoint.metadata_backup,
            json.dumps(checkpoint.database_backup),
            checkpoint.status.value,
            checkpoint.retention_until
        )

        await self.db.execute(query, parameters)
        return checkpoint.checkpoint_id

    async def get_by_id(self, checkpoint_id: str) -> Optional[RollbackCheckpoint]:
        """
        Get checkpoint by ID

        Args:
            checkpoint_id: Checkpoint identifier

        Returns:
            Rollback checkpoint or None if not found
        """
        query = "SELECT * FROM rollback_checkpoints WHERE checkpoint_id = ?"
        row = await self.db.fetchone(query, (checkpoint_id,))

        if row:
            return self._row_to_checkpoint(row)

        return None

    async def get_by_component(self, component_id: str) -> List[RollbackCheckpoint]:
        """
        Get checkpoints for component

        Args:
            component_id: Component identifier

        Returns:
            List of rollback checkpoints
        """
        query = """
        SELECT * FROM rollback_checkpoints
        WHERE component_id = ?
        ORDER BY created_at DESC
        """

        rows = await self.db.fetchall(query, (component_id,))
        return [self._row_to_checkpoint(row) for row in rows]

    async def get_expired_checkpoints(self) -> List[RollbackCheckpoint]:
        """
        Get expired checkpoints for cleanup

        Returns:
            List of expired checkpoints
        """
        query = """
        SELECT * FROM rollback_checkpoints
        WHERE retention_until < CURRENT_TIMESTAMP
        AND status = ?
        """

        rows = await self.db.fetchall(query, (CheckpointStatus.ACTIVE.value,))
        return [self._row_to_checkpoint(row) for row in rows]

    async def delete(self, checkpoint_id: str) -> bool:
        """
        Delete checkpoint

        Args:
            checkpoint_id: Checkpoint identifier

        Returns:
            True if deleted successfully
        """
        query = "DELETE FROM rollback_checkpoints WHERE checkpoint_id = ?"
        cursor = await self.db.execute(query, (checkpoint_id,))
        return cursor.rowcount > 0

    def _row_to_checkpoint(self, row: Dict) -> RollbackCheckpoint:
        """Convert database row to RollbackCheckpoint"""
        return RollbackCheckpoint(
            checkpoint_id=row["checkpoint_id"],
            component_id=row["component_id"],
            migration_state=json.loads(row["migration_state"] or "{}"),
            file_backups=json.loads(row["file_backups"] or "{}"),
            metadata_backup=row["metadata_backup"],
            database_backup=json.loads(row["database_backup"] or "{}"),
            status=CheckpointStatus(row["status"]),
            created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else datetime.utcnow(),
            retention_until=datetime.fromisoformat(row["retention_until"])
        )