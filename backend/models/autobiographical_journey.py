"""
Autobiographical Journey Models
Constitutional compliance: mock data transparency, evaluative feedback
Integrates with Nemori-inspired episodic memory system
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class JourneyStatus(str, Enum):
    """Autobiographical journey status"""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"

class MemoryType(str, Enum):
    """Types of autobiographical memories"""
    EPISODIC = "episodic"            # Specific events and experiences
    SEMANTIC = "semantic"            # General knowledge and concepts
    PROCEDURAL = "procedural"        # Skills and procedures
    EMOTIONAL = "emotional"          # Emotional responses and associations
    CONTEXTUAL = "contextual"        # Environmental and situational context

class ConsciousnessLevel(str, Enum):
    """Consciousness levels for autobiographical events"""
    UNCONSCIOUS = "unconscious"      # Below awareness threshold
    PRECONSCIOUS = "preconscious"    # Available to consciousness
    CONSCIOUS = "conscious"          # Currently in awareness
    METACONSCIOUS = "metaconscious"  # Aware of being aware

class AutobiographicalEvent(BaseModel):
    """Individual autobiographical event with consciousness tracking"""

    # Event identity
    id: str = Field(..., description="Unique event identifier")
    journey_id: str = Field(..., description="Associated journey identifier")
    user_id: str = Field(..., description="Associated user identifier")

    # Event timing
    occurred_at: datetime = Field(..., description="When the event occurred")
    duration: Optional[int] = Field(default=None, description="Event duration in seconds")
    recorded_at: datetime = Field(default_factory=datetime.utcnow, description="When event was recorded")

    # Event content
    title: str = Field(..., description="Brief event title")
    description: str = Field(..., description="Detailed event description")
    narrative: str = Field(..., description="Natural language narrative of the event")

    # Memory classification
    memory_type: MemoryType = Field(..., description="Type of autobiographical memory")
    consciousness_level: ConsciousnessLevel = Field(..., description="Consciousness level during event")
    emotional_valence: float = Field(default=0.0, description="Emotional valence (-1 to 1)", ge=-1.0, le=1.0)
    significance_score: float = Field(default=0.5, description="Subjective significance (0-1)", ge=0.0, le=1.0)

    # Context and associations
    context: Dict[str, Any] = Field(default_factory=dict, description="Contextual information")
    associated_documents: List[str] = Field(default_factory=list, description="Related document IDs")
    associated_concepts: List[str] = Field(default_factory=list, description="Related concept IDs")
    thought_seed_traces: List[str] = Field(default_factory=list, description="Associated ThoughtSeed traces")

    # Retrieval metadata (BM25-inspired)
    keywords: List[str] = Field(default_factory=list, description="Keywords for retrieval")
    retrieval_score: Optional[float] = Field(default=None, description="Latest retrieval relevance score")
    access_count: int = Field(default=0, description="Number of times accessed")
    last_accessed: Optional[datetime] = Field(default=None, description="Last access timestamp")

    # Constitutional compliance
    mock_data: bool = Field(default=True, description="Constitutional requirement: mock data flag")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class AutobiographicalJourney(BaseModel):
    """Complete autobiographical journey for consciousness tracking"""

    # Journey identity
    id: str = Field(..., description="Unique journey identifier")
    user_id: str = Field(..., description="Associated user identifier")

    # Journey metadata
    title: str = Field(..., description="Journey title")
    description: str = Field(..., description="Journey description")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Journey creation time")
    updated_at: Optional[datetime] = Field(default=None, description="Last journey update")

    # Journey status
    status: JourneyStatus = Field(default=JourneyStatus.ACTIVE, description="Current journey status")
    start_date: datetime = Field(..., description="Journey start date")
    end_date: Optional[datetime] = Field(default=None, description="Journey end date (if completed)")

    # Journey statistics
    total_events: int = Field(default=0, description="Total autobiographical events")
    consciousness_events: int = Field(default=0, description="Events with conscious awareness")
    metacognitive_events: int = Field(default=0, description="Events with metacognitive awareness")

    # Journey themes and patterns
    dominant_themes: List[str] = Field(default_factory=list, description="Dominant thematic patterns")
    consciousness_evolution: Dict[str, float] = Field(default_factory=dict, description="Consciousness level evolution")
    emotional_journey: Dict[str, float] = Field(default_factory=dict, description="Emotional journey patterns")

    # Knowledge integration
    concepts_discovered: List[str] = Field(default_factory=list, description="Concepts discovered during journey")
    knowledge_graph_nodes: List[str] = Field(default_factory=list, description="Knowledge graph nodes created")
    connection_insights: List[Dict[str, Any]] = Field(default_factory=list, description="Cross-domain insights")

    # Nemori-inspired episode boundaries
    major_episodes: List[Dict[str, Any]] = Field(default_factory=list, description="Major narrative episodes")
    episode_transitions: List[Dict[str, Any]] = Field(default_factory=list, description="Episode boundary events")

    # Retrieval and replay
    replay_sessions: List[str] = Field(default_factory=list, description="Dream replay session IDs")
    curiosity_missions: List[str] = Field(default_factory=list, description="Curiosity missions spawned")

    # Constitutional compliance
    mock_data: bool = Field(default=True, description="Constitutional requirement: mock data flag")
    evaluation_frames: List[str] = Field(default_factory=list, description="Associated evaluation frames")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        schema_extra = {
            "example": {
                "id": "journey_456",
                "user_id": "user_123",
                "title": "Consciousness Exploration Journey",
                "description": "A journey through document analysis and consciousness emergence",
                "status": "active",
                "total_events": 23,
                "consciousness_events": 8,
                "metacognitive_events": 3,
                "dominant_themes": ["active inference", "emergence", "meta-cognition"],
                "concepts_discovered": ["predictive processing", "free energy", "attention"],
                "major_episodes": [
                    {
                        "title": "First Consciousness Event",
                        "start": "2024-01-15T10:00:00",
                        "end": "2024-01-15T10:30:00",
                        "significance": 0.9
                    }
                ],
                "mock_data": True
            }
        }

class EpisodeBoundary(BaseModel):
    """Episode boundary detection for narrative coherence"""

    # Boundary identity
    id: str = Field(..., description="Unique boundary identifier")
    journey_id: str = Field(..., description="Associated journey identifier")

    # Boundary timing
    timestamp: datetime = Field(..., description="Boundary occurrence time")

    # Boundary characteristics
    boundary_type: str = Field(..., description="Type of boundary (temporal, thematic, contextual)")
    boundary_strength: float = Field(..., description="Strength of boundary (0-1)", ge=0.0, le=1.0)

    # Context shift indicators
    context_shift_magnitude: float = Field(..., description="Magnitude of context change", ge=0.0)
    thematic_coherence_break: bool = Field(default=False, description="Whether thematic coherence breaks")
    consciousness_level_change: bool = Field(default=False, description="Whether consciousness level changes")

    # Episode information
    preceding_episode_id: Optional[str] = Field(default=None, description="Previous episode ID")
    following_episode_id: Optional[str] = Field(default=None, description="Next episode ID")

    # Narrative elements
    transition_narrative: str = Field(..., description="Narrative description of the transition")
    key_events: List[str] = Field(default_factory=list, description="Key events at boundary")

    # Constitutional compliance
    mock_data: bool = Field(default=True, description="Constitutional requirement: mock data flag")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class JourneyInsight(BaseModel):
    """Cross-journey insights and patterns"""

    # Insight identity
    id: str = Field(..., description="Unique insight identifier")
    user_id: str = Field(..., description="Associated user identifier")

    # Source information
    source_journeys: List[str] = Field(..., description="Journey IDs that contributed to insight")
    insight_type: str = Field(..., description="Type of insight (pattern, connection, emergence)")

    # Insight content
    title: str = Field(..., description="Insight title")
    description: str = Field(..., description="Detailed insight description")
    confidence_score: float = Field(..., description="Confidence in insight (0-1)", ge=0.0, le=1.0)

    # Discovery metadata
    discovered_at: datetime = Field(default_factory=datetime.utcnow, description="Insight discovery time")
    discovery_method: str = Field(..., description="How insight was discovered")

    # Impact and validation
    validation_events: List[str] = Field(default_factory=list, description="Events that validate insight")
    impact_score: float = Field(default=0.0, description="Impact on understanding", ge=0.0)

    # Constitutional compliance
    mock_data: bool = Field(default=True, description="Constitutional requirement: mock data flag")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }