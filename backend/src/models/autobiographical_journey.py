"""
AutobiographicalJourney Model
Constitutional compliance: mock data transparency, evaluative feedback framework
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid

class AutobiographicalJourney(BaseModel):
    """Autobiographical journey model with constitutional compliance"""
    
    # Core fields
    journey_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Story arc identifier")
    user_id: str = Field(..., description="Owner user ID")
    title: str = Field(..., max_length=200, description="User-labeled journey (e.g., project)")
    summary: Optional[str] = Field(default=None, description="Generated synopsis")
    
    # Vector and embedding references
    timeline_vector_id: Optional[str] = Field(default=None, description="Qdrant vector reference")
    embedding_id: Optional[str] = Field(default=None, description="Embedding identifier")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Journey creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last journey update")

    # Constitutional compliance
    mock_data: bool = Field(default=True, description="Constitutional requirement: mock data flag")
    mock_data_enabled: bool = Field(default=True, description="Mock data mode for development")
    
    # Journey metadata
    description: Optional[str] = Field(default=None, description="Detailed journey description")
    tags: List[str] = Field(default_factory=list, description="Journey tags")
    status: str = Field(default="active", description="Journey status: active, completed, archived")
    
    # Journey analytics
    total_events: int = Field(default=0, description="Total events in journey")
    total_documents: int = Field(default=0, description="Total documents in journey")
    total_concepts: int = Field(default=0, description="Total concepts in journey")
    total_thoughtseeds: int = Field(default=0, description="Total ThoughtSeeds in journey")
    
    # Journey timeline
    start_date: Optional[datetime] = Field(default=None, description="Journey start date")
    end_date: Optional[datetime] = Field(default=None, description="Journey end date")
    duration_days: Optional[int] = Field(default=None, description="Journey duration in days")
    
    # Journey insights
    key_insights: List[str] = Field(default_factory=list, description="Key insights from journey")
    learning_objectives: List[str] = Field(default_factory=list, description="Learning objectives")
    achievements: List[str] = Field(default_factory=list, description="Journey achievements")

    # Thoughtseed Integration - Enhanced with Archon research
    thoughtseed_trajectory: List[str] = Field(default_factory=list, description="Sequence of thoughtseed trace IDs")
    attractor_dynamics_history: List[Dict[str, Any]] = Field(default_factory=list, description="Evolution of attractor patterns")
    memory_context_shifts: List[Dict[str, Any]] = Field(default_factory=list, description="Working/episodic/procedural memory context changes")
    consciousness_evolution: Dict[str, List[float]] = Field(default_factory=dict, description="Consciousness metrics over time")
    
    # Journey relationships
    related_journeys: List[str] = Field(default_factory=list, description="Related journey IDs")
    parent_journey_id: Optional[str] = Field(default=None, description="Parent journey ID for sub-journeys")
    child_journey_ids: List[str] = Field(default_factory=list, description="Child journey IDs")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return self.model_dump()

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        schema_extra = {
            "example": {
                "journey_id": "550e8400-e29b-41d4-a716-446655440001",
                "user_id": "550e8400-e29b-41d4-a716-446655440000",
                "title": "Consciousness Research Project",
                "summary": "Exploring consciousness through active inference and ThoughtSeed attractor basins",
                "timeline_vector_id": "vector_123",
                "embedding_id": "emb_456",
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:30:00Z",
                "mock_data": True,
                "description": "A comprehensive exploration of consciousness theories and implementations",
                "tags": ["consciousness", "active-inference", "research"],
                "status": "active",
                "total_events": 25,
                "total_documents": 15,
                "total_concepts": 45,
                "total_thoughtseeds": 12,
                "start_date": "2024-01-01T00:00:00Z",
                "end_date": None,
                "duration_days": 15,
                "key_insights": [
                    "Active inference provides a mathematical framework for consciousness",
                    "ThoughtSeed attractor basins enable consciousness simulation"
                ],
                "learning_objectives": [
                    "Understand active inference principles",
                    "Implement consciousness emulation"
                ],
                "achievements": [
                    "Successfully implemented ThoughtSeed pipeline",
                    "Created consciousness visualization system"
                ],
                "related_journeys": ["journey_789"],
                "parent_journey_id": None,
                "child_journey_ids": ["journey_101", "journey_102"]
            }
        }

    def add_thoughtseed_trace(self, trace_id: str, attractor_patterns: Dict[str, Any] = None) -> None:
        """
        Add thoughtseed trace to journey trajectory.
        Enhanced with attractor dynamics from Archon research.
        """
        self.thoughtseed_trajectory.append(trace_id)
        self.total_thoughtseeds = len(self.thoughtseed_trajectory)

        if attractor_patterns:
            self.attractor_dynamics_history.append({
                "timestamp": datetime.utcnow().isoformat(),
                "trace_id": trace_id,
                "patterns": attractor_patterns
            })

        self.updated_at = datetime.utcnow()

    def record_memory_context_shift(self, context_type: str, shift_details: Dict[str, Any]) -> None:
        """
        Record memory context shifts (working/episodic/procedural).
        Based on attractor dynamics research from Archon knowledge base.
        """
        shift_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "context_type": context_type,  # "working", "episodic", "procedural"
            "shift_details": shift_details
        }

        self.memory_context_shifts.append(shift_record)
        self.updated_at = datetime.utcnow()

    def add_consciousness_measurement(self, metric_name: str, value: float) -> None:
        """Add consciousness measurement to evolution tracking"""
        if metric_name not in self.consciousness_evolution:
            self.consciousness_evolution[metric_name] = []

        self.consciousness_evolution[metric_name].append(value)
        self.updated_at = datetime.utcnow()

    def get_dominant_attractor_patterns(self) -> Dict[str, Any]:
        """
        Analyze dominant attractor patterns from journey history.
        Based on neural attractor dynamics research.
        """
        if not self.attractor_dynamics_history:
            return {}

        # Simple analysis - can be enhanced with more sophisticated pattern recognition
        pattern_frequencies = {}

        for record in self.attractor_dynamics_history:
            patterns = record.get("patterns", {})
            for pattern_type, pattern_data in patterns.items():
                if pattern_type not in pattern_frequencies:
                    pattern_frequencies[pattern_type] = []
                pattern_frequencies[pattern_type].append(pattern_data)

        return pattern_frequencies

class AutobiographicalJourneyCreate(BaseModel):
    """Autobiographical journey creation request"""
    
    user_id: str = Field(..., description="Owner user ID")
    title: str = Field(..., max_length=200, description="Journey title")
    description: Optional[str] = Field(default=None, description="Journey description")
    tags: List[str] = Field(default_factory=list, description="Journey tags")
    learning_objectives: List[str] = Field(default_factory=list, description="Learning objectives")
    
    # Constitutional compliance
    mock_data: bool = Field(default=True, description="Constitutional requirement: mock data flag")
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "550e8400-e29b-41d4-a716-446655440000",
                "title": "New Consciousness Journey",
                "description": "Exploring new aspects of consciousness",
                "tags": ["consciousness", "exploration"],
                "learning_objectives": [
                    "Learn about consciousness theories",
                    "Implement consciousness models"
                ],
                "mock_data": True
            }
        }

class AutobiographicalJourneyUpdate(BaseModel):
    """Autobiographical journey update request"""
    
    title: Optional[str] = Field(default=None, max_length=200, description="Updated title")
    description: Optional[str] = Field(default=None, description="Updated description")
    summary: Optional[str] = Field(default=None, description="Updated summary")
    tags: Optional[List[str]] = Field(default=None, description="Updated tags")
    status: Optional[str] = Field(default=None, description="Updated status")
    key_insights: Optional[List[str]] = Field(default=None, description="Updated key insights")
    achievements: Optional[List[str]] = Field(default=None, description="Updated achievements")
    
    # Constitutional compliance
    mock_data: bool = Field(default=True, description="Constitutional requirement: mock data flag")
    
    class Config:
        schema_extra = {
            "example": {
                "title": "Updated Journey Title",
                "description": "Updated journey description",
                "summary": "Updated journey summary",
                "tags": ["updated", "tags"],
                "status": "completed",
                "key_insights": ["New insight 1", "New insight 2"],
                "achievements": ["New achievement 1"],
                "mock_data": True
            }
        }

class AutobiographicalJourneyResponse(BaseModel):
    """Autobiographical journey response"""
    
    journey: AutobiographicalJourney = Field(..., description="Journey data")
    
    # Constitutional compliance
    mock_data: bool = Field(..., description="Constitutional requirement: mock data flag")
    evaluation_frame_id: Optional[str] = Field(default=None, description="Associated evaluation frame")
    
    class Config:
        schema_extra = {
            "example": {
                "journey": {
                    "journey_id": "550e8400-e29b-41d4-a716-446655440001",
                    "user_id": "550e8400-e29b-41d4-a716-446655440000",
                    "title": "Consciousness Research Project",
                    "summary": "Exploring consciousness through active inference",
                    "created_at": "2024-01-15T10:30:00Z",
                    "updated_at": "2024-01-15T10:30:00Z",
                    "mock_data": True
                },
                "mock_data": True,
                "evaluation_frame_id": "eval_123"
            }
        }
