"""
Configuration management for Dionysus Migration System

Handles database connections, framework integrations, and system settings
for the distributed consciousness component migration process.
"""

import os
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class DatabaseConfig(BaseSettings):
    """Database connection configuration"""

    # Neo4j Configuration (Unified Database)
    neo4j_uri: str = Field(
        default="bolt://localhost:7687",
        env="NEO4J_URI",
        description="Neo4j database connection URI"
    )
    neo4j_user: str = Field(
        default="neo4j",
        env="NEO4J_USER",
        description="Neo4j username"
    )
    neo4j_password: str = Field(
        default="password",
        env="NEO4J_PASSWORD",
        description="Neo4j password"
    )
    neo4j_database: str = Field(
        default="neo4j",
        env="NEO4J_DATABASE",
        description="Neo4j database name"
    )

    # Redis Configuration (ThoughtSeed Integration)
    redis_host: str = Field(
        default="localhost",
        env="REDIS_HOST",
        description="Redis server host"
    )
    redis_port: int = Field(
        default=6379,
        env="REDIS_PORT",
        description="Redis server port"
    )
    redis_password: Optional[str] = Field(
        default=None,
        env="REDIS_PASSWORD",
        description="Redis password (optional)"
    )
    redis_db: int = Field(
        default=0,
        env="REDIS_DB",
        description="Redis database number"
    )

    # Legacy Database Configuration (for migration source)
    legacy_mongo_uri: str = Field(
        default="mongodb://localhost:27017",
        env="LEGACY_MONGO_URI",
        description="Legacy MongoDB connection URI"
    )
    legacy_mongo_database: str = Field(
        default="dionysus_legacy",
        env="LEGACY_MONGO_DATABASE",
        description="Legacy MongoDB database name"
    )


class FrameworkConfig(BaseSettings):
    """Framework integration configuration"""

    # ThoughtSeed Configuration
    thoughtseed_enabled: bool = Field(
        default=True,
        env="THOUGHTSEED_ENABLED",
        description="Enable ThoughtSeed integration"
    )
    thoughtseed_config_path: str = Field(
        default="./thoughtseed.yaml",
        env="THOUGHTSEED_CONFIG_PATH",
        description="Path to ThoughtSeed configuration file"
    )

    # DAEDALUS Configuration
    daedalus_enabled: bool = Field(
        default=True,
        env="DAEDALUS_ENABLED",
        description="Enable DAEDALUS coordination"
    )
    daedalus_agent_pool_size: int = Field(
        default=5,
        env="DAEDALUS_AGENT_POOL_SIZE",
        description="Number of DAEDALUS background agents"
    )
    daedalus_context_isolation: bool = Field(
        default=True,
        env="DAEDALUS_CONTEXT_ISOLATION",
        description="Enable independent context windows"
    )

    # CHIMERA Configuration
    chimera_enabled: bool = Field(
        default=True,
        env="CHIMERA_ENABLED",
        description="Enable CHIMERA consciousness integration"
    )
    chimera_consciousness_threshold: float = Field(
        default=0.7,
        env="CHIMERA_CONSCIOUSNESS_THRESHOLD",
        description="Minimum consciousness score for component migration"
    )


class MigrationConfig(BaseSettings):
    """Migration process configuration"""

    # Quality Assessment
    quality_threshold: float = Field(
        default=0.7,
        env="MIGRATION_QUALITY_THRESHOLD",
        description="Minimum quality score for component migration"
    )
    consciousness_weight: float = Field(
        default=0.7,
        env="CONSCIOUSNESS_WEIGHT",
        description="Weight for consciousness functionality in quality score"
    )
    strategic_weight: float = Field(
        default=0.3,
        env="STRATEGIC_WEIGHT",
        description="Weight for strategic value in quality score"
    )

    # Performance Requirements
    zero_downtime_required: bool = Field(
        default=True,
        env="ZERO_DOWNTIME_REQUIRED",
        description="Require zero downtime during migration"
    )
    rollback_timeout_seconds: int = Field(
        default=30,
        env="ROLLBACK_TIMEOUT_SECONDS",
        description="Maximum time allowed for component rollback"
    )

    # Processing Limits
    max_concurrent_migrations: int = Field(
        default=3,
        env="MAX_CONCURRENT_MIGRATIONS",
        description="Maximum number of components migrating simultaneously"
    )
    component_analysis_timeout: int = Field(
        default=300,
        env="COMPONENT_ANALYSIS_TIMEOUT",
        description="Timeout for component analysis in seconds"
    )

    # Legacy Codebase
    legacy_codebase_path: str = Field(
        default="./legacy/dionysus",
        env="LEGACY_CODEBASE_PATH",
        description="Path to legacy Dionysus consciousness codebase"
    )

    @field_validator("legacy_codebase_path")
    @classmethod
    def validate_legacy_path(cls, v):
        path = Path(v)
        # Skip validation for non-existent paths in development
        return str(path.absolute())


class APIConfig(BaseSettings):
    """API server configuration"""

    host: str = Field(
        default="localhost",
        env="API_HOST",
        description="API server host"
    )
    port: int = Field(
        default=8080,
        env="API_PORT",
        description="API server port"
    )
    debug: bool = Field(
        default=False,
        env="API_DEBUG",
        description="Enable debug mode"
    )
    cors_origins: list[str] = Field(
        default=["*"],
        env="CORS_ORIGINS",
        description="Allowed CORS origins"
    )

    # Security
    secret_key: str = Field(
        default="consciousness-migration-secret-key",
        env="SECRET_KEY",
        description="Secret key for API security"
    )
    access_token_expire_minutes: int = Field(
        default=30,
        env="ACCESS_TOKEN_EXPIRE_MINUTES",
        description="Access token expiration time in minutes"
    )


class LoggingConfig(BaseSettings):
    """Logging configuration"""

    level: str = Field(
        default="INFO",
        env="LOG_LEVEL",
        description="Logging level"
    )
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT",
        description="Log message format"
    )
    file_path: Optional[str] = Field(
        default=None,
        env="LOG_FILE_PATH",
        description="Log file path (optional)"
    )

    # Consciousness-specific logging
    log_consciousness_metrics: bool = Field(
        default=True,
        env="LOG_CONSCIOUSNESS_METRICS",
        description="Enable consciousness metrics logging"
    )
    log_agent_coordination: bool = Field(
        default=True,
        env="LOG_AGENT_COORDINATION",
        description="Enable agent coordination logging"
    )


class Config(BaseSettings):
    """Main configuration class combining all settings"""

    database: DatabaseConfig = DatabaseConfig()
    framework: FrameworkConfig = FrameworkConfig()
    migration: MigrationConfig = MigrationConfig()
    api: APIConfig = APIConfig()
    logging: LoggingConfig = LoggingConfig()

    # Environment
    environment: str = Field(
        default="development",
        env="ENVIRONMENT",
        description="Application environment"
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False
    }


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance"""
    return config


def get_database_config() -> DatabaseConfig:
    """Get database configuration"""
    return config.database


def get_framework_config() -> FrameworkConfig:
    """Get framework configuration"""
    return config.framework


def get_migration_config() -> MigrationConfig:
    """Get migration configuration"""
    return config.migration


def get_api_config() -> APIConfig:
    """Get API configuration"""
    return config.api


def get_logging_config() -> LoggingConfig:
    """Get logging configuration"""
    return config.logging