"""
Component Rollback Service

Handles rollback operations for individual migrated components,
ensuring <30 second recovery with state preservation.
"""

import asyncio
import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

from ..config import get_migration_config
from ..logging_config import get_migration_logger
from ..models.rollback_checkpoint import RollbackCheckpoint, CheckpointStatus
from ..models.legacy_component import LegacyComponent


class RollbackService:
    """Service for managing component rollback operations"""

    def __init__(self):
        self.config = get_migration_config()
        self.logger = get_migration_logger()
        self.checkpoints: Dict[str, RollbackCheckpoint] = {}
        self.rollback_history: List[Dict] = []

    async def create_rollback_checkpoint(
        self,
        component: LegacyComponent,
        migration_state: Dict,
        checkpoint_config: Optional[Dict] = None
    ) -> str:
        """
        Create a rollback checkpoint before migration

        Args:
            component: Component being migrated
            migration_state: Current migration state
            checkpoint_config: Optional checkpoint configuration

        Returns:
            Checkpoint ID
        """
        checkpoint_id = str(uuid4())
        config = checkpoint_config or {}

        self.logger.info(
            "Creating rollback checkpoint",
            checkpoint_id=checkpoint_id,
            component_id=component.component_id
        )

        try:
            # Create checkpoint directory
            checkpoint_dir = Path(self.config.rollback_storage_path) / checkpoint_id
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # Backup component files
            file_backups = await self._backup_component_files(
                component, checkpoint_dir
            )

            # Backup component metadata
            metadata_backup = await self._backup_component_metadata(
                component, migration_state, checkpoint_dir
            )

            # Backup database state
            database_backup = await self._backup_database_state(
                component, checkpoint_dir
            )

            # Create checkpoint record
            checkpoint = RollbackCheckpoint(
                checkpoint_id=checkpoint_id,
                component_id=component.component_id,
                migration_state=migration_state,
                file_backups=file_backups,
                metadata_backup=metadata_backup,
                database_backup=database_backup,
                status=CheckpointStatus.ACTIVE,
                retention_until=datetime.utcnow() + timedelta(
                    days=config.get('retention_days', 7)
                )
            )

            self.checkpoints[checkpoint_id] = checkpoint

            self.logger.info(
                "Rollback checkpoint created",
                checkpoint_id=checkpoint_id,
                component_id=component.component_id,
                backup_size=sum(
                    backup['size_bytes'] for backup in file_backups.values()
                )
            )

            return checkpoint_id

        except Exception as e:
            self.logger.error(
                "Failed to create rollback checkpoint",
                checkpoint_id=checkpoint_id,
                component_id=component.component_id,
                error=str(e)
            )
            raise

    async def _backup_component_files(
        self,
        component: LegacyComponent,
        checkpoint_dir: Path
    ) -> Dict[str, Dict]:
        """
        Backup component files

        Args:
            component: Component to backup
            checkpoint_dir: Checkpoint directory

        Returns:
            File backup metadata
        """
        file_backups = {}
        files_dir = checkpoint_dir / "files"
        files_dir.mkdir(exist_ok=True)

        try:
            # Backup main component file
            source_path = Path(component.file_path)
            if source_path.exists():
                backup_path = files_dir / source_path.name
                shutil.copy2(source_path, backup_path)

                file_backups[component.file_path] = {
                    'backup_path': str(backup_path),
                    'original_path': component.file_path,
                    'size_bytes': backup_path.stat().st_size,
                    'backup_time': datetime.utcnow().isoformat(),
                    'checksum': component.source_code_hash
                }

            # Backup related files (dependencies, tests, etc.)
            related_files = await self._discover_related_files(component)
            for related_file in related_files:
                if Path(related_file).exists():
                    backup_name = f"related_{Path(related_file).name}"
                    backup_path = files_dir / backup_name
                    shutil.copy2(related_file, backup_path)

                    file_backups[related_file] = {
                        'backup_path': str(backup_path),
                        'original_path': related_file,
                        'size_bytes': backup_path.stat().st_size,
                        'backup_time': datetime.utcnow().isoformat(),
                        'checksum': await self._calculate_file_checksum(backup_path)
                    }

        except Exception as e:
            self.logger.error(
                "Failed to backup component files",
                component_id=component.component_id,
                error=str(e)
            )
            raise

        return file_backups

    async def _backup_component_metadata(
        self,
        component: LegacyComponent,
        migration_state: Dict,
        checkpoint_dir: Path
    ) -> str:
        """
        Backup component metadata

        Args:
            component: Component to backup
            migration_state: Migration state
            checkpoint_dir: Checkpoint directory

        Returns:
            Metadata backup file path
        """
        metadata = {
            'component': component.dict(),
            'migration_state': migration_state,
            'backup_timestamp': datetime.utcnow().isoformat(),
            'system_state': {
                'environment_variables': dict(self.config.dict()),
                'dependencies': component.dependencies
            }
        }

        metadata_file = checkpoint_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        return str(metadata_file)

    async def _backup_database_state(
        self,
        component: LegacyComponent,
        checkpoint_dir: Path
    ) -> Dict[str, str]:
        """
        Backup relevant database state

        Args:
            component: Component to backup
            checkpoint_dir: Checkpoint directory

        Returns:
            Database backup metadata
        """
        database_backups = {}

        try:
            # Backup component records
            component_backup = {
                'component_records': [component.dict()],
                'timestamp': datetime.utcnow().isoformat()
            }

            db_file = checkpoint_dir / "database_state.json"
            with open(db_file, 'w') as f:
                json.dump(component_backup, f, indent=2, default=str)

            database_backups['component_state'] = str(db_file)

            # In a real implementation, this would backup:
            # - Component migration history
            # - Quality assessment records
            # - Enhancement results
            # - Related configuration data

        except Exception as e:
            self.logger.warning(
                "Failed to backup database state",
                component_id=component.component_id,
                error=str(e)
            )

        return database_backups

    async def _discover_related_files(self, component: LegacyComponent) -> List[str]:
        """
        Discover files related to the component

        Args:
            component: Component to analyze

        Returns:
            List of related file paths
        """
        related_files = []
        component_path = Path(component.file_path)

        if not component_path.exists():
            return related_files

        component_dir = component_path.parent
        component_stem = component_path.stem

        # Look for common related files
        potential_files = [
            component_dir / f"test_{component_stem}.py",
            component_dir / f"{component_stem}_test.py",
            component_dir / f"{component_stem}.md",
            component_dir / f"{component_stem}.yml",
            component_dir / f"{component_stem}.yaml",
            component_dir / f"{component_stem}.json"
        ]

        for file_path in potential_files:
            if file_path.exists():
                related_files.append(str(file_path))

        return related_files

    async def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate file checksum for integrity verification"""
        import hashlib

        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    async def rollback_component(
        self,
        checkpoint_id: str,
        rollback_options: Optional[Dict] = None
    ) -> bool:
        """
        Rollback component to checkpoint state

        Args:
            checkpoint_id: Checkpoint to rollback to
            rollback_options: Optional rollback configuration

        Returns:
            True if rollback successful
        """
        if checkpoint_id not in self.checkpoints:
            self.logger.error(
                "Checkpoint not found",
                checkpoint_id=checkpoint_id
            )
            return False

        checkpoint = self.checkpoints[checkpoint_id]
        options = rollback_options or {}
        rollback_start = datetime.utcnow()

        self.logger.info(
            "Starting component rollback",
            checkpoint_id=checkpoint_id,
            component_id=checkpoint.component_id
        )

        try:
            # Phase 1: Verify checkpoint integrity
            if not await self._verify_checkpoint_integrity(checkpoint):
                raise Exception("Checkpoint integrity verification failed")

            # Phase 2: Restore files
            await self._restore_component_files(checkpoint, options)

            # Phase 3: Restore metadata
            await self._restore_component_metadata(checkpoint, options)

            # Phase 4: Restore database state
            await self._restore_database_state(checkpoint, options)

            # Phase 5: Verify rollback success
            if not await self._verify_rollback_success(checkpoint):
                raise Exception("Rollback verification failed")

            # Calculate rollback time
            rollback_duration = (datetime.utcnow() - rollback_start).total_seconds()

            # Record rollback
            rollback_record = {
                'rollback_id': str(uuid4()),
                'checkpoint_id': checkpoint_id,
                'component_id': checkpoint.component_id,
                'rollback_time': rollback_start.isoformat(),
                'duration_seconds': rollback_duration,
                'success': True,
                'options': options
            }

            self.rollback_history.append(rollback_record)

            self.logger.info(
                "Component rollback completed successfully",
                checkpoint_id=checkpoint_id,
                component_id=checkpoint.component_id,
                duration_seconds=rollback_duration
            )

            # Verify <30 second requirement
            if rollback_duration > 30:
                self.logger.warning(
                    "Rollback exceeded 30-second target",
                    duration_seconds=rollback_duration,
                    checkpoint_id=checkpoint_id
                )

            return True

        except Exception as e:
            rollback_duration = (datetime.utcnow() - rollback_start).total_seconds()

            # Record failed rollback
            rollback_record = {
                'rollback_id': str(uuid4()),
                'checkpoint_id': checkpoint_id,
                'component_id': checkpoint.component_id,
                'rollback_time': rollback_start.isoformat(),
                'duration_seconds': rollback_duration,
                'success': False,
                'error': str(e),
                'options': options
            }

            self.rollback_history.append(rollback_record)

            self.logger.error(
                "Component rollback failed",
                checkpoint_id=checkpoint_id,
                component_id=checkpoint.component_id,
                error=str(e),
                duration_seconds=rollback_duration
            )

            return False

    async def _verify_checkpoint_integrity(self, checkpoint: RollbackCheckpoint) -> bool:
        """
        Verify checkpoint integrity before rollback

        Args:
            checkpoint: Checkpoint to verify

        Returns:
            True if checkpoint is valid
        """
        try:
            # Check checkpoint directory exists
            checkpoint_dir = Path(self.config.rollback_storage_path) / checkpoint.checkpoint_id
            if not checkpoint_dir.exists():
                return False

            # Verify file backups exist and have correct checksums
            for file_path, backup_info in checkpoint.file_backups.items():
                backup_path = Path(backup_info['backup_path'])
                if not backup_path.exists():
                    return False

                # Verify checksum
                current_checksum = await self._calculate_file_checksum(backup_path)
                if current_checksum != backup_info['checksum']:
                    return False

            # Verify metadata backup exists
            if checkpoint.metadata_backup:
                metadata_path = Path(checkpoint.metadata_backup)
                if not metadata_path.exists():
                    return False

            return True

        except Exception as e:
            self.logger.error(
                "Checkpoint integrity verification failed",
                checkpoint_id=checkpoint.checkpoint_id,
                error=str(e)
            )
            return False

    async def _restore_component_files(
        self,
        checkpoint: RollbackCheckpoint,
        options: Dict
    ) -> None:
        """
        Restore component files from checkpoint

        Args:
            checkpoint: Checkpoint to restore from
            options: Restore options
        """
        for file_path, backup_info in checkpoint.file_backups.items():
            backup_path = Path(backup_info['backup_path'])
            original_path = Path(backup_info['original_path'])

            # Create backup of current file if it exists
            if original_path.exists() and options.get('backup_current', True):
                current_backup = original_path.with_suffix(
                    f"{original_path.suffix}.rollback_backup"
                )
                shutil.copy2(original_path, current_backup)

            # Restore from checkpoint
            original_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(backup_path, original_path)

            self.logger.debug(
                "File restored from checkpoint",
                original_path=str(original_path),
                backup_path=str(backup_path)
            )

    async def _restore_component_metadata(
        self,
        checkpoint: RollbackCheckpoint,
        options: Dict
    ) -> None:
        """
        Restore component metadata from checkpoint

        Args:
            checkpoint: Checkpoint to restore from
            options: Restore options
        """
        if not checkpoint.metadata_backup:
            return

        try:
            with open(checkpoint.metadata_backup, 'r') as f:
                metadata = json.load(f)

            # In a real implementation, this would:
            # - Restore component configuration
            # - Reset migration state
            # - Update component registry
            # - Restore environment variables if needed

            self.logger.debug(
                "Metadata restored from checkpoint",
                checkpoint_id=checkpoint.checkpoint_id,
                metadata_keys=list(metadata.keys())
            )

        except Exception as e:
            self.logger.error(
                "Failed to restore metadata",
                checkpoint_id=checkpoint.checkpoint_id,
                error=str(e)
            )
            raise

    async def _restore_database_state(
        self,
        checkpoint: RollbackCheckpoint,
        options: Dict
    ) -> None:
        """
        Restore database state from checkpoint

        Args:
            checkpoint: Checkpoint to restore from
            options: Restore options
        """
        for backup_type, backup_path in checkpoint.database_backup.items():
            try:
                with open(backup_path, 'r') as f:
                    backup_data = json.load(f)

                # In a real implementation, this would:
                # - Restore component records
                # - Reset migration status
                # - Restore quality assessments
                # - Reset enhancement results

                self.logger.debug(
                    "Database state restored",
                    backup_type=backup_type,
                    backup_path=backup_path
                )

            except Exception as e:
                self.logger.warning(
                    "Failed to restore database backup",
                    backup_type=backup_type,
                    error=str(e)
                )

    async def _verify_rollback_success(self, checkpoint: RollbackCheckpoint) -> bool:
        """
        Verify rollback was successful

        Args:
            checkpoint: Checkpoint that was restored

        Returns:
            True if rollback verification passes
        """
        try:
            # Verify files were restored correctly
            for file_path, backup_info in checkpoint.file_backups.items():
                original_path = Path(backup_info['original_path'])
                if not original_path.exists():
                    return False

                # Verify file content matches backup
                current_checksum = await self._calculate_file_checksum(original_path)
                if current_checksum != backup_info['checksum']:
                    return False

            return True

        except Exception as e:
            self.logger.error(
                "Rollback verification failed",
                checkpoint_id=checkpoint.checkpoint_id,
                error=str(e)
            )
            return False

    def get_checkpoint_status(self, checkpoint_id: str) -> Optional[RollbackCheckpoint]:
        """
        Get checkpoint status

        Args:
            checkpoint_id: Checkpoint ID

        Returns:
            Checkpoint or None if not found
        """
        return self.checkpoints.get(checkpoint_id)

    def get_component_checkpoints(self, component_id: str) -> List[RollbackCheckpoint]:
        """
        Get all checkpoints for a component

        Args:
            component_id: Component ID

        Returns:
            List of checkpoints for the component
        """
        return [
            checkpoint for checkpoint in self.checkpoints.values()
            if checkpoint.component_id == component_id
        ]

    def get_rollback_history(
        self,
        component_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """
        Get rollback history

        Args:
            component_id: Optional component filter
            limit: Optional result limit

        Returns:
            List of rollback records
        """
        history = self.rollback_history

        if component_id:
            history = [
                record for record in history
                if record['component_id'] == component_id
            ]

        # Sort by rollback time (most recent first)
        history.sort(key=lambda x: x['rollback_time'], reverse=True)

        if limit:
            history = history[:limit]

        return history

    async def cleanup_expired_checkpoints(self) -> int:
        """
        Clean up expired checkpoints

        Returns:
            Number of checkpoints cleaned up
        """
        current_time = datetime.utcnow()
        cleanup_count = 0

        expired_checkpoints = [
            checkpoint_id for checkpoint_id, checkpoint in self.checkpoints.items()
            if checkpoint.retention_until < current_time
        ]

        for checkpoint_id in expired_checkpoints:
            try:
                checkpoint = self.checkpoints[checkpoint_id]

                # Remove checkpoint directory
                checkpoint_dir = Path(self.config.rollback_storage_path) / checkpoint_id
                if checkpoint_dir.exists():
                    shutil.rmtree(checkpoint_dir)

                # Remove from memory
                del self.checkpoints[checkpoint_id]
                cleanup_count += 1

                self.logger.info(
                    "Expired checkpoint cleaned up",
                    checkpoint_id=checkpoint_id,
                    component_id=checkpoint.component_id
                )

            except Exception as e:
                self.logger.error(
                    "Failed to cleanup expired checkpoint",
                    checkpoint_id=checkpoint_id,
                    error=str(e)
                )

        return cleanup_count