"""
Event Node Models with Mosaic Observation Schema
Constitutional compliance: mock data transparency, evaluative feedback
Implements Mosaic state observation for consciousness tracking
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum
import numpy as np

class ObservationType(str, Enum):
    """Types of Mosaic observations"""
    SENSORY = "sensory"              # Raw sensory input data
    COGNITIVE = "cognitive"          # Cognitive processing events
    EMOTIONAL = "emotional"          # Emotional state observations
    METACOGNITIVE = "metacognitive"  # Self-awareness observations
    BEHAVIORAL = "behavioral"        # Action and behavior events
    ENVIRONMENTAL = "environmental"  # Context and environment

class MosaicState(str, Enum):
    """Mosaic consciousness states"""
    DORMANT = "dormant"              # No conscious activity
    EMERGING = "emerging"            # Consciousness beginning to form
    ACTIVE = "active"                # Full conscious awareness
    REFLECTIVE = "reflective"        # Metacognitive reflection
    INTEGRATIVE = "integrative"      # Cross-domain integration
    TRANSCENDENT = "transcendent"    # Beyond normal awareness

class AttentionFocus(BaseModel):
    """Attention focus tracking for Mosaic observations"""

    # Focus identity
    target_id: str = Field(..., description="ID of attention target")
    target_type: str = Field(..., description="Type of target (concept, document, event)")

    # Attention metrics
    intensity: float = Field(..., description="Attention intensity (0-1)", ge=0.0, le=1.0)
    duration: int = Field(..., description="Attention duration in milliseconds")
    stability: float = Field(..., description="Attention stability (0-1)", ge=0.0, le=1.0)

    # Context
    trigger_event: Optional[str] = Field(default=None, description="Event that triggered attention")
    competing_foci: List[str] = Field(default_factory=list, description="Other attention targets")

class MosaicObservation(BaseModel):
    """Individual Mosaic state observation"""

    # Observation identity
    id: str = Field(..., description="Unique observation identifier")
    event_node_id: str = Field(..., description="Associated event node")

    # Observation timing
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Observation timestamp")
    duration: Optional[int] = Field(default=None, description="Observation duration in milliseconds")

    # Observation classification
    observation_type: ObservationType = Field(..., description="Type of observation")
    mosaic_state: MosaicState = Field(..., description="Consciousness state during observation")

    # Consciousness metrics
    consciousness_level: float = Field(..., description="Consciousness level (0-1)", ge=0.0, le=1.0)
    awareness_breadth: float = Field(..., description="Breadth of awareness (0-1)", ge=0.0, le=1.0)
    integration_depth: float = Field(..., description="Integration depth (0-1)", ge=0.0, le=1.0)

    # Attention tracking
    attention_foci: List[AttentionFocus] = Field(default_factory=list, description="Attention focus points")
    attention_dispersion: float = Field(default=0.0, description="Attention dispersion metric", ge=0.0)

    # Sensory data (if applicable)
    sensory_modalities: Dict[str, float] = Field(default_factory=dict, description="Sensory input strengths")
    perceptual_coherence: float = Field(default=0.0, description="Perceptual coherence score", ge=0.0, le=1.0)

    # Cognitive data
    working_memory_load: float = Field(default=0.0, description="Working memory utilization", ge=0.0, le=1.0)
    cognitive_complexity: float = Field(default=0.0, description="Processing complexity", ge=0.0)
    conceptual_connections: int = Field(default=0, description="Number of active concept connections")

    # Emotional data
    emotional_valence: float = Field(default=0.0, description="Emotional valence (-1 to 1)", ge=-1.0, le=1.0)
    emotional_arousal: float = Field(default=0.0, description="Emotional arousal (0-1)", ge=0.0, le=1.0)
    emotional_complexity: float = Field(default=0.0, description="Emotional complexity", ge=0.0)

    # Metacognitive data
    self_awareness_level: float = Field(default=0.0, description="Self-awareness level (0-1)", ge=0.0, le=1.0)
    meta_reflection_depth: float = Field(default=0.0, description="Meta-reflection depth", ge=0.0)
    self_model_coherence: float = Field(default=0.0, description="Self-model coherence", ge=0.0, le=1.0)

    # Integration metrics
    cross_domain_connections: int = Field(default=0, description="Cross-domain connections formed")
    insight_emergence_score: float = Field(default=0.0, description="Insight emergence likelihood", ge=0.0, le=1.0)
    pattern_recognition_hits: int = Field(default=0, description="Pattern recognition events")

    # Context and associations
    context: Dict[str, Any] = Field(default_factory=dict, description="Contextual information")
    associated_concepts: List[str] = Field(default_factory=list, description="Activated concepts")
    causal_influences: List[str] = Field(default_factory=list, description="Causal influence sources")

    # Constitutional compliance
    mock_data: bool = Field(default=True, description="Constitutional requirement: mock data flag")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class EventNode(BaseModel):
    """Event node with comprehensive Mosaic observation tracking"""

    # Node identity
    id: str = Field(..., description="Unique event node identifier")
    user_id: str = Field(..., description="Associated user identifier")
    session_id: Optional[str] = Field(default=None, description="Associated session")

    # Event classification
    event_type: str = Field(..., description="Type of event (document_processing, consciousness_emergence, etc.)")
    event_category: str = Field(..., description="Event category (cognitive, behavioral, etc.)")

    # Event timing
    started_at: datetime = Field(default_factory=datetime.utcnow, description="Event start time")
    ended_at: Optional[datetime] = Field(default=None, description="Event end time")
    duration: Optional[int] = Field(default=None, description="Event duration in milliseconds")

    # Event content
    title: str = Field(..., description="Event title")
    description: str = Field(..., description="Event description")
    narrative: Optional[str] = Field(default=None, description="Human-readable narrative")

    # Mosaic observations
    observations: List[MosaicObservation] = Field(default_factory=list, description="Mosaic state observations")
    observation_count: int = Field(default=0, description="Total number of observations")

    # Aggregated consciousness metrics
    peak_consciousness_level: float = Field(default=0.0, description="Peak consciousness level", ge=0.0, le=1.0)
    avg_consciousness_level: float = Field(default=0.0, description="Average consciousness level", ge=0.0, le=1.0)
    consciousness_variability: float = Field(default=0.0, description="Consciousness level variability")

    # State transitions
    state_transitions: List[Dict[str, Any]] = Field(default_factory=list, description="Mosaic state transitions")
    dominant_state: Optional[MosaicState] = Field(default=None, description="Dominant consciousness state")
    state_duration_map: Dict[str, int] = Field(default_factory=dict, description="Time in each state")

    # Causal relationships
    causal_predecessors: List[str] = Field(default_factory=list, description="Causally preceding events")
    causal_successors: List[str] = Field(default_factory=list, description="Causally following events")
    causal_strength_scores: Dict[str, float] = Field(default_factory=dict, description="Causal influence strengths")

    # Knowledge integration
    concepts_activated: List[str] = Field(default_factory=list, description="Concepts activated during event")
    knowledge_nodes_created: List[str] = Field(default_factory=list, description="New knowledge nodes")
    cross_references: List[str] = Field(default_factory=list, description="Cross-references to other nodes")

    # Processing metadata
    thoughtseed_traces: List[str] = Field(default_factory=list, description="Associated ThoughtSeed traces")
    evaluation_frames: List[str] = Field(default_factory=list, description="Associated evaluation frames")
    curiosity_missions: List[str] = Field(default_factory=list, description="Spawned curiosity missions")

    # Quality and validation
    data_quality_score: float = Field(default=1.0, description="Data quality assessment", ge=0.0, le=1.0)
    validation_status: str = Field(default="pending", description="Validation status")
    anomaly_flags: List[str] = Field(default_factory=list, description="Detected anomalies")

    # Constitutional compliance
    mock_data: bool = Field(default=True, description="Constitutional requirement: mock data flag")
    privacy_level: str = Field(default="standard", description="Privacy level for data")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        schema_extra = {
            "example": {
                "id": "event_789",
                "user_id": "user_123",
                "event_type": "document_processing",
                "event_category": "cognitive",
                "title": "Document Analysis with Consciousness Emergence",
                "description": "Processing research paper on active inference with consciousness emergence",
                "observation_count": 15,
                "peak_consciousness_level": 0.87,
                "avg_consciousness_level": 0.62,
                "dominant_state": "active",
                "concepts_activated": ["active_inference", "free_energy", "consciousness"],
                "state_duration_map": {
                    "emerging": 2000,
                    "active": 8500,
                    "reflective": 3000
                },
                "mock_data": True
            }
        }

class ConsciousnessEvent(BaseModel):
    """Special event node for consciousness emergence tracking"""

    # Event identity (inherits from EventNode structure)
    id: str = Field(..., description="Unique consciousness event identifier")
    event_node_id: str = Field(..., description="Associated event node")

    # Consciousness emergence characteristics
    emergence_type: str = Field(..., description="Type of consciousness emergence")
    emergence_trigger: str = Field(..., description="What triggered the emergence")
    emergence_context: Dict[str, Any] = Field(default_factory=dict, description="Context during emergence")

    # Quantitative measurements
    pre_emergence_baseline: float = Field(..., description="Consciousness level before emergence")
    post_emergence_peak: float = Field(..., description="Peak consciousness level reached")
    emergence_slope: float = Field(..., description="Rate of consciousness increase")
    sustenance_duration: int = Field(..., description="How long peak was sustained (ms)")

    # Qualitative characteristics
    subjective_description: str = Field(..., description="Subjective experience description")
    phenomenological_features: List[str] = Field(default_factory=list, description="Phenomenological characteristics")
    insight_content: Optional[str] = Field(default=None, description="Content of insights gained")

    # Validation and significance
    significance_score: float = Field(..., description="Significance of the emergence", ge=0.0, le=1.0)
    reproducibility_likelihood: float = Field(default=0.0, description="Likelihood of reproduction", ge=0.0, le=1.0)
    validation_criteria_met: List[str] = Field(default_factory=list, description="Validation criteria satisfied")

    # Constitutional compliance
    mock_data: bool = Field(default=True, description="Constitutional requirement: mock data flag")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }