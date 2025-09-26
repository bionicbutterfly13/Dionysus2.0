"""
Simple configuration for testing without complex dependencies
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SimpleMigrationConfig:
    """Simple migration configuration for testing"""
    quality_threshold: float = 0.7
    consciousness_weight: float = 0.7
    strategic_weight: float = 0.3
    zero_downtime_required: bool = True
    max_concurrent_agents: int = 10
    rollback_storage_path: str = "/tmp/dionysus_checkpoints"
    coordination_cycle_interval: int = 30
    database_path: str = "./dionysus_migration.db"
    database_url: str = "sqlite:///./dionysus_migration.db"


# Global simple config instance
_simple_config = SimpleMigrationConfig()


def get_migration_config():
    """Get simple migration config for testing"""
    return _simple_config