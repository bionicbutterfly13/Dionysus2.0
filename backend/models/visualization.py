"""
Visualization Models
Real-time consciousness visualization with constitutional compliance
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class VisualizationType(str, Enum):
    """Types of visualization messages"""
    GRAPH_UPDATE = "graph_update"
    CARD_STACK_UPDATE = "card_stack_update"
    CURIOSITY_SIGNAL = "curiosity_signal"
    EVALUATION_FRAME = "evaluation_frame"
    MOSAIC_STATE = "mosaic_state"

class ConsciousnessState(str, Enum):
    """Consciousness states for ThoughtSeeds"""
    AWARE = "aware"
    FOCUSED = "focused"
    WANDERING = "wandering"
    DREAMING = "dreaming"
    LEARNING = "learning"
    PROCESSING = "processing"

class GraphUpdate(BaseModel):
    """Knowledge graph update"""
    
    node_id: str = Field(..., description="Node identifier")
    node_type: str = Field(..., description="Node type: concept, document, thoughtseed")
    node_label: str = Field(..., description="Node label")
    connections: List[str] = Field(default_factory=list, description="Connected node IDs")
    strength: float = Field(default=1.0, ge=0.0, le=1.0, description="Connection strength")
    
    # Constitutional compliance
    mock_data: bool = Field(default=True, description="Constitutional requirement: mock data flag")

class CardStackUpdate(BaseModel):
    """Card stack update for document exploration"""
    
    stack_id: str = Field(..., description="Stack identifier")
    concept: str = Field(..., description="Concept being explored")
    documents: List[Dict[str, Any]] = Field(default_factory=list, description="Documents in stack")
    highlights: List[str] = Field(default_factory=list, description="Key highlights")
    summary: Optional[str] = Field(default=None, description="Stack summary")
    
    # Constitutional compliance
    mock_data: bool = Field(default=True, description="Constitutional requirement: mock data flag")

class CuriositySignal(BaseModel):
    """Curiosity engine signal"""
    
    signal_id: str = Field(..., description="Signal identifier")
    intensity: float = Field(..., ge=0.0, le=1.0, description="Curiosity intensity")
    topic: str = Field(..., description="Topic of curiosity")
    knowledge_gap: str = Field(..., description="Identified knowledge gap")
    sources_found: int = Field(default=0, description="Sources discovered")
    
    # Constitutional compliance
    mock_data: bool = Field(default=True, description="Constitutional requirement: mock data flag")

class EvaluationFrameMessage(BaseModel):
    """Evaluation frame message"""
    
    evaluation_id: str = Field(..., description="Evaluation identifier")
    evaluation_type: str = Field(..., description="Type of evaluation")
    whats_good: str = Field(..., description="What's good")
    whats_broken: str = Field(..., description="What's broken")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    
    # Constitutional compliance
    mock_data: bool = Field(default=True, description="Constitutional requirement: mock data flag")

class MosaicState(BaseModel):
    """Mosaic Systems LLC observation model state"""
    
    senses: Dict[str, Any] = Field(default_factory=dict, description="Sensory inputs")
    actions: Dict[str, Any] = Field(default_factory=dict, description="Action outputs")
    emotions: Dict[str, Any] = Field(default_factory=dict, description="Emotional states")
    impulses: Dict[str, Any] = Field(default_factory=dict, description="Impulse states")
    cognitions: Dict[str, Any] = Field(default_factory=dict, description="Cognitive states")
    
    # Constitutional compliance
    mock_data: bool = Field(default=True, description="Constitutional requirement: mock data flag")

class VisualizationMessage(BaseModel):
    """Real-time visualization message"""
    
    # Core fields
    type: VisualizationType = Field(..., description="Message type")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Message timestamp")
    user_id: str = Field(..., description="Target user ID")
    
    # Message data
    data: Dict[str, Any] = Field(..., description="Message data")
    
    # Constitutional compliance
    mock_data: bool = Field(default=True, description="Constitutional requirement: mock data flag")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        schema_extra = {
            "example": {
                "type": "graph_update",
                "timestamp": "2024-01-15T10:30:00Z",
                "user_id": "user_123",
                "data": {
                    "node_id": "node_456",
                    "node_type": "concept",
                    "node_label": "Active Inference",
                    "connections": ["node_789", "node_101"],
                    "strength": 0.8,
                    "mock_data": True
                },
                "mock_data": True
            }
        }
