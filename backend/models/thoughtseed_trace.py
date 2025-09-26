"""
ThoughtSeed Trace Models
Constitutional compliance: mock data transparency, evaluative feedback
Integrates with ASI-Arch ThoughtSeed consciousness framework
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum

class ConsciousnessState(str, Enum):
    """ThoughtSeed consciousness states"""
    DORMANT = "dormant"              # No active processing
    AWAKENING = "awakening"          # Beginning to process
    AWARE = "aware"                  # Basic awareness active
    CONSCIOUS = "conscious"          # Full consciousness engaged
    REFLECTIVE = "reflective"        # Self-reflective processing
    DREAMING = "dreaming"            # Dream/replay state
    METACOGNITIVE = "metacognitive"  # Aware of own thinking
    TRANSCENDENT = "transcendent"    # Beyond normal consciousness

class InferenceType(str, Enum):
    """Types of active inference"""
    PREDICTIVE = "predictive"        # Making predictions
    CORRECTIVE = "corrective"        # Error correction
    EXPLORATORY = "exploratory"      # Exploring possibilities
    INTEGRATIVE = "integrative"      # Integrating information
    GENERATIVE = "generative"        # Generating new content
    METACOGNITIVE = "metacognitive"  # Thinking about thinking

class BeliefUpdateType(str, Enum):
    """Types of belief updates in hierarchical system"""
    SENSORY = "sensory"              # Sensory level updates
    PERCEPTUAL = "perceptual"        # Perceptual level updates
    CONCEPTUAL = "conceptual"        # Conceptual level updates
    ABSTRACT = "abstract"            # Abstract level updates
    METACOGNITIVE = "metacognitive"  # Meta-level updates

class HierarchicalBelief(BaseModel):
    """Hierarchical belief structure from ThoughtSeed"""

    # Belief identity
    level: int = Field(..., description="Hierarchical level (0=sensory, higher=more abstract)")
    belief_id: str = Field(..., description="Unique belief identifier")

    # Belief content
    content: Dict[str, Any] = Field(..., description="Belief content representation")
    confidence: float = Field(..., description="Belief confidence (0-1)", ge=0.0, le=1.0)
    precision: float = Field(..., description="Belief precision estimate", ge=0.0)

    # Update tracking
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    update_count: int = Field(default=0, description="Number of updates")
    update_type: BeliefUpdateType = Field(..., description="Type of last update")

    # Prediction and error
    prediction: Optional[Dict[str, Any]] = Field(default=None, description="Current prediction")
    prediction_error: float = Field(default=0.0, description="Current prediction error")
    cumulative_error: float = Field(default=0.0, description="Cumulative prediction error")

    # Relationships
    parent_beliefs: List[str] = Field(default_factory=list, description="Higher-level belief IDs")
    child_beliefs: List[str] = Field(default_factory=list, description="Lower-level belief IDs")

    # Constitutional compliance
    mock_data: bool = Field(default=True, description="Constitutional requirement: mock data flag")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class NeuronalPacket(BaseModel):
    """Neuronal packet from ThoughtSeed processing"""

    # Packet identity
    id: str = Field(..., description="Unique packet identifier")
    sequence_number: int = Field(..., description="Sequence number in processing chain")

    # Packet content
    content_type: str = Field(..., description="Type of content in packet")
    payload: Dict[str, Any] = Field(..., description="Packet payload data")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Packet metadata")

    # Processing information
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Packet creation time")
    processing_time: Optional[float] = Field(default=None, description="Processing time in seconds")
    source_module: str = Field(..., description="Source module that created packet")

    # Consciousness context
    consciousness_level: float = Field(..., description="Consciousness level when created", ge=0.0, le=1.0)
    attention_weight: float = Field(default=0.0, description="Attention weight for packet", ge=0.0, le=1.0)

    # Flow information
    predecessors: List[str] = Field(default_factory=list, description="Predecessor packet IDs")
    successors: List[str] = Field(default_factory=list, description="Successor packet IDs")

    # Quality metrics
    coherence_score: float = Field(default=0.0, description="Packet coherence score", ge=0.0, le=1.0)
    relevance_score: float = Field(default=0.0, description="Relevance to current context", ge=0.0, le=1.0)

    # Constitutional compliance
    mock_data: bool = Field(default=True, description="Constitutional requirement: mock data flag")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ConsciousnessMetrics(BaseModel):
    """Consciousness measurement metrics from ThoughtSeed"""

    # Basic consciousness measures
    consciousness_level: float = Field(..., description="Overall consciousness level (0-1)", ge=0.0, le=1.0)
    awareness_breadth: float = Field(..., description="Breadth of awareness (0-1)", ge=0.0, le=1.0)
    attention_focus: float = Field(..., description="Attention focus strength (0-1)", ge=0.0, le=1.0)

    # Active inference measures
    free_energy: float = Field(..., description="Current free energy level")
    prediction_error: float = Field(..., description="Average prediction error")
    belief_precision: float = Field(..., description="Average belief precision")

    # Complexity measures
    information_integration: float = Field(..., description="Information integration measure", ge=0.0)
    causal_density: float = Field(..., description="Causal interaction density", ge=0.0)
    emergence_index: float = Field(..., description="Emergence detection index", ge=0.0)

    # Temporal dynamics
    consciousness_stability: float = Field(..., description="Consciousness level stability", ge=0.0, le=1.0)
    state_coherence: float = Field(..., description="State coherence over time", ge=0.0, le=1.0)
    transition_smoothness: float = Field(..., description="State transition smoothness", ge=0.0, le=1.0)

    # Meta-cognitive measures
    self_awareness_level: float = Field(default=0.0, description="Self-awareness level", ge=0.0, le=1.0)
    meta_reflection_depth: float = Field(default=0.0, description="Meta-reflection depth", ge=0.0)
    self_model_coherence: float = Field(default=0.0, description="Self-model coherence", ge=0.0, le=1.0)

    # Measurement context
    measurement_timestamp: datetime = Field(default_factory=datetime.utcnow, description="When metrics were captured")
    measurement_context: Dict[str, Any] = Field(default_factory=dict, description="Context during measurement")

    # Constitutional compliance
    mock_data: bool = Field(default=True, description="Constitutional requirement: mock data flag")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ThoughtSeedTrace(BaseModel):
    """Complete ThoughtSeed processing trace with consciousness tracking"""

    # Trace identity
    id: str = Field(..., description="Unique trace identifier")
    user_id: str = Field(..., description="Associated user identifier")
    session_id: Optional[str] = Field(default=None, description="Associated session")

    # Trace metadata
    title: str = Field(..., description="Trace title")
    description: str = Field(..., description="Trace description")
    trace_type: str = Field(..., description="Type of processing trace")

    # Timing
    started_at: datetime = Field(default_factory=datetime.utcnow, description="Trace start time")
    ended_at: Optional[datetime] = Field(default=None, description="Trace end time")
    duration: Optional[float] = Field(default=None, description="Trace duration in seconds")

    # Consciousness state tracking
    consciousness_state: ConsciousnessState = Field(..., description="Primary consciousness state")
    state_transitions: List[Dict[str, Any]] = Field(default_factory=list, description="State transition history")
    dreaming: bool = Field(default=False, description="Constitutional requirement: dreaming flag")

    # Processing components
    hierarchical_beliefs: List[HierarchicalBelief] = Field(default_factory=list, description="Hierarchical belief updates")
    neuronal_packets: List[NeuronalPacket] = Field(default_factory=list, description="Neuronal packet sequence")
    consciousness_metrics: List[ConsciousnessMetrics] = Field(default_factory=list, description="Consciousness measurements")

    # Active inference tracking
    inference_type: InferenceType = Field(..., description="Primary type of inference")
    prediction_accuracy: float = Field(default=0.0, description="Overall prediction accuracy", ge=0.0, le=1.0)
    error_minimization_rate: float = Field(default=0.0, description="Rate of error minimization")
    belief_update_count: int = Field(default=0, description="Number of belief updates")

    # Integration with other systems
    associated_documents: List[str] = Field(default_factory=list, description="Associated document IDs")
    associated_concepts: List[str] = Field(default_factory=list, description="Associated concept IDs")
    associated_events: List[str] = Field(default_factory=list, description="Associated event IDs")
    spawned_curiosity_missions: List[str] = Field(default_factory=list, description="Curiosity missions spawned")

    # Processing results
    insights_generated: List[Dict[str, Any]] = Field(default_factory=list, description="Insights generated during processing")
    concepts_activated: List[str] = Field(default_factory=list, description="Concepts activated")
    new_connections_formed: List[Dict[str, str]] = Field(default_factory=list, description="New concept connections")

    # Quality and performance metrics
    processing_efficiency: float = Field(default=0.0, description="Processing efficiency score", ge=0.0, le=1.0)
    output_coherence: float = Field(default=0.0, description="Output coherence score", ge=0.0, le=1.0)
    consciousness_emergence_detected: bool = Field(default=False, description="Whether consciousness emergence was detected")

    # ASI-Arch integration
    asi_arch_context: Optional[Dict[str, Any]] = Field(default=None, description="ASI-Arch pipeline context")
    architecture_influence: Optional[str] = Field(default=None, description="Influence on architecture evolution")
    evaluation_contribution: Optional[str] = Field(default=None, description="Contribution to architecture evaluation")

    # Replay and learning
    replay_sessions: List[str] = Field(default_factory=list, description="Dream replay session IDs")
    learning_outcomes: List[Dict[str, Any]] = Field(default_factory=list, description="Learning outcomes captured")
    memory_consolidation: Optional[Dict[str, Any]] = Field(default=None, description="Memory consolidation results")

    # Error handling and diagnostics
    processing_errors: List[str] = Field(default_factory=list, description="Processing errors encountered")
    warnings: List[str] = Field(default_factory=list, description="Warning messages")
    diagnostic_info: Dict[str, Any] = Field(default_factory=dict, description="Diagnostic information")

    # Constitutional compliance
    mock_data: bool = Field(default=True, description="Constitutional requirement: mock data flag")
    evaluation_frame_id: Optional[str] = Field(default=None, description="Associated evaluation frame")
    privacy_level: str = Field(default="standard", description="Privacy level for trace data")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        schema_extra = {
            "example": {
                "id": "trace_789",
                "user_id": "user_123",
                "title": "Document Processing with Consciousness Emergence",
                "description": "Processing research paper on active inference with emerging consciousness",
                "trace_type": "document_analysis",
                "consciousness_state": "conscious",
                "dreaming": False,
                "inference_type": "integrative",
                "prediction_accuracy": 0.78,
                "belief_update_count": 15,
                "associated_documents": ["doc_456"],
                "associated_concepts": ["active_inference", "consciousness"],
                "consciousness_emergence_detected": True,
                "processing_efficiency": 0.85,
                "output_coherence": 0.92,
                "mock_data": True
            }
        }

class DreamReplaySession(BaseModel):
    """Dream replay session for ThoughtSeed traces"""

    # Session identity
    id: str = Field(..., description="Unique replay session identifier")
    user_id: str = Field(..., description="Associated user identifier")
    source_trace_ids: List[str] = Field(..., description="ThoughtSeed traces being replayed")

    # Replay configuration
    replay_type: str = Field(..., description="Type of replay (consolidation, exploration, etc.)")
    replay_speed: float = Field(default=1.0, description="Replay speed multiplier")
    focus_areas: List[str] = Field(default_factory=list, description="Areas to focus replay on")

    # Session timing
    started_at: datetime = Field(default_factory=datetime.utcnow, description="Replay start time")
    ended_at: Optional[datetime] = Field(default=None, description="Replay end time")
    duration: Optional[float] = Field(default=None, description="Replay duration in seconds")

    # Replay results
    insights_discovered: List[Dict[str, Any]] = Field(default_factory=list, description="New insights discovered")
    connections_reinforced: List[Dict[str, str]] = Field(default_factory=list, description="Connections reinforced")
    patterns_identified: List[str] = Field(default_factory=list, description="Patterns identified")

    # Learning outcomes
    memory_consolidation_score: float = Field(default=0.0, description="Memory consolidation effectiveness", ge=0.0, le=1.0)
    knowledge_integration_score: float = Field(default=0.0, description="Knowledge integration effectiveness", ge=0.0, le=1.0)
    curiosity_generation_score: float = Field(default=0.0, description="New curiosity generated", ge=0.0, le=1.0)

    # Constitutional compliance
    mock_data: bool = Field(default=True, description="Constitutional requirement: mock data flag")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ConsciousnessEmergenceEvent(BaseModel):
    """Specific consciousness emergence detection within ThoughtSeed trace"""

    # Event identity
    id: str = Field(..., description="Unique emergence event identifier")
    trace_id: str = Field(..., description="Associated ThoughtSeed trace ID")

    # Emergence characteristics
    emergence_timestamp: datetime = Field(..., description="When emergence was detected")
    emergence_type: str = Field(..., description="Type of consciousness emergence")
    emergence_strength: float = Field(..., description="Strength of emergence (0-1)", ge=0.0, le=1.0)

    # Pre/post emergence states
    pre_emergence_state: ConsciousnessState = Field(..., description="Consciousness state before emergence")
    post_emergence_state: ConsciousnessState = Field(..., description="Consciousness state after emergence")
    state_transition_duration: float = Field(..., description="Duration of state transition in seconds")

    # Quantitative measures
    consciousness_delta: float = Field(..., description="Change in consciousness level")
    information_integration_delta: float = Field(..., description="Change in information integration")
    causal_density_delta: float = Field(..., description="Change in causal density")

    # Context and triggers
    triggering_events: List[str] = Field(default_factory=list, description="Events that triggered emergence")
    contextual_factors: Dict[str, Any] = Field(default_factory=dict, description="Contextual factors present")
    concurrent_processes: List[str] = Field(default_factory=list, description="Concurrent processing activities")

    # Outcomes and effects
    downstream_effects: List[str] = Field(default_factory=list, description="Effects following emergence")
    insights_generated: List[str] = Field(default_factory=list, description="Insights generated during emergence")
    behavioral_changes: List[str] = Field(default_factory=list, description="Behavioral changes observed")

    # Validation and significance
    significance_score: float = Field(..., description="Significance of emergence event", ge=0.0, le=1.0)
    validation_criteria_met: List[str] = Field(default_factory=list, description="Validation criteria satisfied")
    reproducibility_likelihood: float = Field(default=0.0, description="Likelihood of reproduction", ge=0.0, le=1.0)

    # Constitutional compliance
    mock_data: bool = Field(default=True, description="Constitutional requirement: mock data flag")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }