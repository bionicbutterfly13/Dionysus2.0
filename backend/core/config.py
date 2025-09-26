"""
Flux Backend Configuration
Environment-based configuration with constitutional compliance
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
import os

class Settings(BaseSettings):
    """Application settings with constitutional compliance"""
    
    # Application
    app_name: str = "Flux - Self-Teaching Consciousness Emulator"
    app_version: str = "0.1.0"
    debug: bool = Field(default=False, description="Debug mode")
    mock_data: bool = Field(default=True, description="Use mock data - constitutional requirement")
    
    # Database Connections
    # Neo4j (Knowledge Graph - Single Source of Truth)
    neo4j_uri: str = Field(default="bolt://localhost:7687", description="Neo4j connection URI")
    neo4j_user: str = Field(default="neo4j", description="Neo4j username")
    neo4j_password: str = Field(default="password", description="Neo4j password")
    neo4j_database: str = Field(default="neo4j", description="Neo4j database name")
    
    # Qdrant (Vector Database)
    qdrant_host: str = Field(default="localhost", description="Qdrant host")
    qdrant_port: int = Field(default=6333, description="Qdrant port")
    qdrant_collection: str = Field(default="flux_embeddings", description="Qdrant collection name")
    
    # SQLite (Structured Data)
    sqlite_path: str = Field(default="flux.db", description="SQLite database path")
    
    # Redis (Real-time Streams)
    redis_host: str = Field(default="localhost", description="Redis host")
    redis_port: int = Field(default=6379, description="Redis port")
    redis_db: int = Field(default=0, description="Redis database number")
    redis_password: Optional[str] = Field(default=None, description="Redis password")
    
    # Ollama/LLaMA (Local Inference)
    ollama_host: str = Field(default="http://localhost:11434", description="Ollama API host")
    ollama_model: str = Field(default="llama2", description="Default Ollama model")
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173"],
        description="CORS allowed origins"
    )
    
    # Security
    secret_key: str = Field(default="flux-secret-key-change-in-production", description="Secret key")
    
    # Constitutional Compliance
    evaluation_feedback_enabled: bool = Field(default=True, description="Enable evaluative feedback framework")
    thoughtseed_channels_enabled: bool = Field(default=True, description="Enable ThoughtSeed data channels")
    context_engineering_enabled: bool = Field(default=True, description="Enable context engineering best practices")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

# Global settings instance
settings = Settings()

# Constitutional validation
def validate_constitutional_compliance():
    """Validate that settings comply with constitutional requirements"""
    violations = []
    
    # Mock data is allowed in development - constitutional requirement for transparency
    # No violation for mock_data being enabled
    
    if not settings.evaluation_feedback_enabled:
        violations.append("Evaluative feedback framework must be enabled")
    
    if not settings.thoughtseed_channels_enabled:
        violations.append("ThoughtSeed channels must be enabled")
    
    if not settings.context_engineering_enabled:
        violations.append("Context engineering must be enabled")
    
    if violations:
        raise ValueError(f"Constitutional violations: {', '.join(violations)}")
    
    return True

# Validate on import
validate_constitutional_compliance()
