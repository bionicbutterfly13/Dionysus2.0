"""
Evaluation Framework Models
Constitutional compliance: mandatory evaluative feedback framework
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

class EvaluationType(str, Enum):
    """Types of evaluations"""
    DOCUMENT_PROCESSING = "document_processing"
    CURIOSITY_MISSION = "curiosity_mission"
    KNOWLEDGE_EXTRACTION = "knowledge_extraction"
    THOUGHTSEED_FLOW = "thoughtseed_flow"
    SYSTEM_PERFORMANCE = "system_performance"

class EvaluationFrame(BaseModel):
    """Constitutional evaluative feedback framework"""
    
    # Core fields
    id: str = Field(..., description="Unique evaluation identifier")
    evaluation_type: EvaluationType = Field(..., description="Type of evaluation")
    entity_id: str = Field(..., description="ID of entity being evaluated")
    
    # Constitutional requirement: Four mandatory questions
    whats_good: str = Field(..., description="What's good? - Constitutional requirement")
    whats_broken: str = Field(..., description="What's broken? - Constitutional requirement")
    what_works_but_shouldnt: str = Field(..., description="What works but shouldn't? - Constitutional requirement")
    what_doesnt_but_pretends_to: str = Field(..., description="What doesn't but pretends to? - Constitutional requirement")
    
    # Additional evaluation data
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Quantitative metrics")
    recommendations: List[str] = Field(default_factory=list, description="Improvement recommendations")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in evaluation")
    
    # Context
    context_data: Dict[str, Any] = Field(default_factory=dict, description="Contextual information")
    user_feedback: Optional[str] = Field(default=None, description="User-provided feedback")
    
    # Constitutional compliance
    mock_data: bool = Field(default=True, description="Constitutional requirement: mock data flag")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = Field(default=None)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        schema_extra = {
            "example": {
                "id": "eval_123",
                "evaluation_type": "document_processing",
                "entity_id": "doc_456",
                "whats_good": "Successfully extracted 15 concepts with high confidence",
                "whats_broken": "Failed to process mathematical equations in the document",
                "what_works_but_shouldnt": "Using mock data instead of real processing",
                "what_doesnt_but_pretends_to": "Claims to understand consciousness when it's just pattern matching",
                "metrics": {
                    "concepts_extracted": 15,
                    "processing_time_seconds": 45,
                    "confidence_score": 0.87
                },
                "recommendations": [
                    "Implement mathematical equation parser",
                    "Replace mock data with real processing",
                    "Add confidence thresholds for consciousness claims"
                ],
                "confidence_score": 0.85,
                "mock_data": True
            }
        }

class EvaluationRequest(BaseModel):
    """Request for evaluation"""
    
    evaluation_type: EvaluationType = Field(..., description="Type of evaluation")
    entity_id: str = Field(..., description="ID of entity to evaluate")
    context_data: Dict[str, Any] = Field(default_factory=dict, description="Contextual information")
    user_feedback: Optional[str] = Field(default=None, description="User-provided feedback")
    
    # Constitutional compliance
    mock_data: bool = Field(default=True, description="Constitutional requirement: mock data flag")

class EvaluationResponse(BaseModel):
    """Evaluation response"""
    
    evaluation_id: str = Field(..., description="Evaluation identifier")
    evaluation_frame: EvaluationFrame = Field(..., description="Complete evaluation frame")
    
    # Constitutional compliance
    mock_data: bool = Field(..., description="Constitutional requirement: mock data flag")
    
    class Config:
        schema_extra = {
            "example": {
                "evaluation_id": "eval_123",
                "evaluation_frame": {
                    "id": "eval_123",
                    "evaluation_type": "document_processing",
                    "entity_id": "doc_456",
                    "whats_good": "Successfully extracted concepts",
                    "whats_broken": "Failed to process equations",
                    "what_works_but_shouldnt": "Using mock data",
                    "what_doesnt_but_pretends_to": "Claims consciousness understanding",
                    "metrics": {"concepts_extracted": 15},
                    "recommendations": ["Fix equation parser"],
                    "confidence_score": 0.85,
                    "mock_data": True
                },
                "mock_data": True
            }
        }
