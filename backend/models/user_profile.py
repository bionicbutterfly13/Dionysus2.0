"""
User Profile Models
Constitutional compliance: mock data transparency, evaluative feedback
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class UserPreferences(BaseModel):
    """User preferences for consciousness emulator"""

    # Consciousness settings
    consciousness_threshold: float = Field(default=0.7, description="Minimum consciousness level for alerts", ge=0.0, le=1.0)
    enable_dreaming: bool = Field(default=True, description="Enable nightly dreaming replay")
    enable_curiosity_missions: bool = Field(default=True, description="Enable automatic curiosity gap detection")

    # UI preferences
    theme: str = Field(default="dark", description="UI theme preference")
    visualization_mode: str = Field(default="graph", description="Default visualization mode")
    dashboard_layout: Dict[str, Any] = Field(default_factory=dict, description="Dashboard layout configuration")

    # Processing preferences
    local_processing: bool = Field(default=True, description="Prefer local LLM processing")
    max_document_size: int = Field(default=10485760, description="Max document size in bytes (10MB)")
    auto_extract_concepts: bool = Field(default=True, description="Automatically extract concepts")

class UserProfile(BaseModel):
    """User profile with constitutional compliance"""

    # Identity
    id: str = Field(..., description="Unique user identifier")
    username: str = Field(..., description="User display name")
    email: Optional[str] = Field(default=None, description="User email address")

    # Profile metadata
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Profile creation timestamp")
    updated_at: Optional[datetime] = Field(default=None, description="Last profile update")
    last_active: Optional[datetime] = Field(default=None, description="Last activity timestamp")

    # Constitutional compliance
    mock_data: bool = Field(default=True, description="Constitutional requirement: mock data flag")
    data_retention_consent: bool = Field(default=False, description="User consent for data retention")
    evaluation_participation: bool = Field(default=True, description="Participate in evaluative feedback")

    # User state
    current_session_id: Optional[str] = Field(default=None, description="Current session identifier")
    total_documents_processed: int = Field(default=0, description="Total documents processed")
    total_consciousness_events: int = Field(default=0, description="Total consciousness emergence events")

    # Preferences
    preferences: UserPreferences = Field(default_factory=UserPreferences, description="User preferences")

    # Journey tracking
    autobiographical_journey_id: Optional[str] = Field(default=None, description="Current autobiographical journey")
    active_curiosity_missions: List[str] = Field(default_factory=list, description="Active curiosity mission IDs")
    recent_evaluations: List[str] = Field(default_factory=list, description="Recent evaluation frame IDs")

    # Statistics
    consciousness_stats: Dict[str, float] = Field(default_factory=dict, description="Consciousness emergence statistics")
    concept_graph_size: int = Field(default=0, description="Number of concept nodes created")
    dream_replay_count: int = Field(default=0, description="Number of dream replays completed")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        schema_extra = {
            "example": {
                "id": "user_123",
                "username": "consciousness_explorer",
                "email": "explorer@example.com",
                "mock_data": True,
                "data_retention_consent": True,
                "evaluation_participation": True,
                "total_documents_processed": 42,
                "total_consciousness_events": 7,
                "preferences": {
                    "consciousness_threshold": 0.8,
                    "enable_dreaming": True,
                    "enable_curiosity_missions": True,
                    "theme": "dark",
                    "local_processing": True
                },
                "consciousness_stats": {
                    "avg_consciousness_level": 0.75,
                    "peak_consciousness_level": 0.95,
                    "consciousness_events_per_session": 2.3
                },
                "concept_graph_size": 156,
                "dream_replay_count": 12
            }
        }

class UserSession(BaseModel):
    """User session tracking for consciousness emulator"""

    # Session identity
    id: str = Field(..., description="Unique session identifier")
    user_id: str = Field(..., description="Associated user identifier")

    # Session timing
    started_at: datetime = Field(default_factory=datetime.utcnow, description="Session start time")
    ended_at: Optional[datetime] = Field(default=None, description="Session end time")
    last_activity: datetime = Field(default_factory=datetime.utcnow, description="Last activity timestamp")

    # Session activity
    documents_processed: int = Field(default=0, description="Documents processed in this session")
    consciousness_events: int = Field(default=0, description="Consciousness events in this session")
    curiosity_missions_created: int = Field(default=0, description="Curiosity missions created")
    evaluation_frames_generated: int = Field(default=0, description="Evaluation frames generated")

    # Session state
    current_document_batch: Optional[str] = Field(default=None, description="Currently processing batch")
    active_thoughtseed_traces: List[str] = Field(default_factory=list, description="Active ThoughtSeed traces")
    session_context: Dict[str, Any] = Field(default_factory=dict, description="Session context data")

    # Constitutional compliance
    mock_data: bool = Field(default=True, description="Constitutional requirement: mock data flag")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class UserAnalytics(BaseModel):
    """User analytics and insights"""

    user_id: str = Field(..., description="Associated user identifier")
    period_start: datetime = Field(..., description="Analytics period start")
    period_end: datetime = Field(..., description="Analytics period end")

    # Activity metrics
    total_sessions: int = Field(default=0, description="Total sessions in period")
    total_active_time: int = Field(default=0, description="Total active time in seconds")
    avg_session_duration: float = Field(default=0.0, description="Average session duration in minutes")

    # Processing metrics
    documents_processed: int = Field(default=0, description="Documents processed in period")
    concepts_discovered: int = Field(default=0, description="New concepts discovered")
    consciousness_events: int = Field(default=0, description="Consciousness emergence events")

    # Growth metrics
    knowledge_graph_growth: int = Field(default=0, description="Knowledge graph nodes added")
    autobiographical_events: int = Field(default=0, description="Autobiographical events recorded")
    curiosity_missions_completed: int = Field(default=0, description="Curiosity missions completed")

    # Quality metrics
    avg_consciousness_level: float = Field(default=0.0, description="Average consciousness level", ge=0.0, le=1.0)
    peak_consciousness_level: float = Field(default=0.0, description="Peak consciousness level", ge=0.0, le=1.0)
    evaluation_feedback_score: float = Field(default=0.0, description="Average evaluation feedback score")

    # Constitutional compliance
    mock_data: bool = Field(default=True, description="Constitutional requirement: mock data flag")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }