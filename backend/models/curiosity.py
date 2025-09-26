"""
Curiosity Mission Models
Constitutional compliance: evaluative feedback, mock data transparency
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class CuriosityMode(str, Enum):
    """Curiosity mission modes"""
    ACTIVE = "active"
    PASSIVE = "passive"
    DREAMING = "dreaming"
    REPLAY = "replay"

class MissionStatus(str, Enum):
    """Mission status"""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"

class SourceReference(BaseModel):
    """Source reference with trust metrics"""
    
    url: str = Field(..., description="Source URL")
    title: str = Field(..., description="Source title")
    domain: str = Field(..., description="Source domain")
    trust_score: float = Field(..., ge=0.0, le=1.0, description="Trust score 0-1")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score 0-1")
    
    # Constitutional compliance
    mock_data: bool = Field(default=True, description="Constitutional requirement: mock data flag")
    
    # Metadata
    published_date: Optional[datetime] = Field(default=None, description="Publication date")
    author: Optional[str] = Field(default=None, description="Author information")
    summary: Optional[str] = Field(default=None, description="Source summary")

class CuriosityMission(BaseModel):
    """Curiosity mission with constitutional compliance"""
    
    # Core fields
    id: str = Field(..., description="Unique mission identifier")
    user_id: str = Field(..., description="User who initiated the mission")
    title: str = Field(..., description="Mission title")
    description: str = Field(..., description="Mission description")
    
    # Mission configuration
    curiosity_mode: CuriosityMode = Field(default=CuriosityMode.ACTIVE, description="Curiosity mode")
    trigger_event_id: Optional[str] = Field(default=None, description="Triggering event ID")
    prompt: str = Field(..., description="Curiosity prompt/question")
    
    # Status and progress
    status: MissionStatus = Field(default=MissionStatus.ACTIVE, description="Mission status")
    sources_found: int = Field(default=0, description="Number of sources discovered")
    knowledge_gaps: List[str] = Field(default_factory=list, description="Identified knowledge gaps")
    
    # Results
    sources: List[SourceReference] = Field(default_factory=list, description="Discovered sources")
    insights: List[str] = Field(default_factory=list, description="Generated insights")
    related_concepts: List[str] = Field(default_factory=list, description="Related concepts")
    
    # Configuration
    max_results: int = Field(default=10, description="Maximum results to return")
    require_user_confirmation: bool = Field(default=False, description="Require user confirmation")
    replay_priority: int = Field(default=1, ge=1, le=10, description="Replay priority 1-10")
    
    # Constitutional compliance
    mock_data: bool = Field(default=True, description="Constitutional requirement: mock data flag")
    evaluation_frame_id: Optional[str] = Field(default=None, description="Associated evaluation frame")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)
    last_activity: Optional[datetime] = Field(default=None)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class CuriosityMissionRequest(BaseModel):
    """Curiosity mission creation request"""
    
    trigger_event_id: Optional[str] = Field(default=None, description="Triggering event ID")
    prompt: str = Field(..., description="Curiosity prompt/question")
    curiosity_mode: CuriosityMode = Field(default=CuriosityMode.ACTIVE, description="Curiosity mode")
    replay_priority: int = Field(default=1, ge=1, le=10, description="Replay priority")
    max_results: int = Field(default=10, description="Maximum results")
    require_user_confirmation: bool = Field(default=False, description="Require confirmation")
    
    # Constitutional compliance
    mock_data: bool = Field(default=True, description="Constitutional requirement: mock data flag")
    
    class Config:
        schema_extra = {
            "example": {
                "prompt": "What are the latest developments in active inference?",
                "curiosity_mode": "active",
                "replay_priority": 5,
                "max_results": 10,
                "require_user_confirmation": False,
                "mock_data": True
            }
        }

class CuriosityMissionResponse(BaseModel):
    """Curiosity mission response"""
    
    mission_id: str = Field(..., description="Mission identifier")
    status: MissionStatus = Field(..., description="Current status")
    sources_found: int = Field(..., description="Sources discovered")
    knowledge_gaps: List[str] = Field(..., description="Knowledge gaps identified")
    
    # Constitutional compliance
    mock_data: bool = Field(..., description="Constitutional requirement: mock data flag")
    evaluation_frame_id: str = Field(..., description="Associated evaluation frame")
    
    # Results preview
    sources_preview: List[SourceReference] = Field(default_factory=list, description="Preview of sources")
    insights_preview: List[str] = Field(default_factory=list, description="Preview of insights")
    
    class Config:
        schema_extra = {
            "example": {
                "mission_id": "mission_123",
                "status": "active",
                "sources_found": 5,
                "knowledge_gaps": ["Bayesian optimization", "Variational inference"],
                "mock_data": True,
                "evaluation_frame_id": "eval_456",
                "sources_preview": [
                    {
                        "url": "https://example.com/paper.pdf",
                        "title": "Active Inference in AI",
                        "domain": "example.com",
                        "trust_score": 0.8,
                        "relevance_score": 0.9,
                        "mock_data": True
                    }
                ],
                "insights_preview": ["Active inference shows promise", "Bayesian approaches are key"]
            }
        }
