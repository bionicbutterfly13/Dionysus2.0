#!/usr/bin/env python3
"""
ThoughtseedTrace Model: Individual ThoughtSeed processing trace
"""

from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator
import uuid


class ThoughtseedLayer(str, Enum):
    """ThoughtSeed hierarchy layers"""
    SENSORY = "sensory"
    PERCEPTUAL = "perceptual"
    CONCEPTUAL = "conceptual"
    ABSTRACT = "abstract"
    METACOGNITIVE = "metacognitive"


class TraceType(str, Enum):
    """Types of ThoughtSeed traces"""
    PATTERN_RECOGNITION = "pattern_recognition"
    CONSCIOUSNESS_EMERGENCE = "consciousness_emergence"
    LAYER_TRANSITION = "layer_transition"
    NEURONAL_ACTIVATION = "neuronal_activation"
    MEMORY_FORMATION = "memory_formation"
    ATTENTION_FOCUS = "attention_focus"
    META_COGNITION = "meta_cognition"
    RECURSIVE_PROCESSING = "recursive_processing"


class NeuronalActivity(BaseModel):
    """Neuronal activity data within trace"""
    activation_pattern: Dict[str, float] = Field(default_factory=dict, description="Neuronal activation patterns")
    signal_strength: float = Field(..., ge=0.0, le=1.0, description="Overall signal strength")
    propagation_velocity: float = Field(default=0.0, ge=0.0, description="Signal propagation velocity")
    
    active_connections: List[str] = Field(default_factory=list, description="Active neuronal connections")
    inhibited_connections: List[str] = Field(default_factory=list, description="Inhibited connections")
    connection_weights: Dict[str, float] = Field(default_factory=dict, description="Connection strength weights")
    
    firing_frequency: float = Field(default=0.0, ge=0.0, description="Neuronal firing frequency")
    synchronization_level: float = Field(default=0.0, ge=0.0, le=1.0, description="Network synchronization")
    
    learning_signals: Dict[str, float] = Field(default_factory=dict, description="Learning-related signals")
    adaptation_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Adaptation rate")


class ConsciousnessMarkers(BaseModel):
    """Consciousness emergence markers in trace"""
    self_reference_detected: bool = Field(default=False, description="Self-referential processing detected")
    recursive_depth: int = Field(default=0, ge=0, description="Recursive processing depth")
    meta_awareness_level: float = Field(default=0.0, ge=0.0, le=1.0, description="Meta-awareness level")
    
    emergence_indicators: List[str] = Field(default_factory=list, description="Consciousness emergence indicators")
    coherence_measure: float = Field(default=0.0, ge=0.0, le=1.0, description="Consciousness coherence measure")
    integration_level: float = Field(default=0.0, ge=0.0, le=1.0, description="Information integration level")


class ThoughtseedTrace(BaseModel):
    """
    ThoughtSeed Trace: Individual processing trace in ThoughtSeed hierarchy

    Represents a single processing event or pattern within a ThoughtSeed layer,
    capturing neuronal activity, consciousness markers, and processing details.
    """
    
    # Core identification
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique trace identifier")
    trace_name: Optional[str] = Field(None, description="Human-readable trace name")
    trace_type: TraceType = Field(..., description="Type of trace")
    
    # Layer and hierarchy
    layer: ThoughtseedLayer = Field(..., description="ThoughtSeed layer where trace originated")
    layer_position: float = Field(..., ge=0.0, le=1.0, description="Position within layer processing")
    hierarchical_level: int = Field(default=0, ge=0, description="Hierarchical processing level")
    
    # Temporal information
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Trace creation timestamp")
    duration_ms: int = Field(default=0, ge=0, description="Trace duration in milliseconds")
    processing_order: int = Field(default=0, ge=0, description="Processing order within workspace")
    
    # Pattern and content
    pattern_strength: float = Field(..., ge=0.0, le=1.0, description="Detected pattern strength")
    content_hash: Optional[str] = Field(None, description="Hash of processed content")
    content_snippet: Optional[str] = Field(None, description="Content snippet that generated trace")
    
    # Neuronal activity
    neuronal_activity: NeuronalActivity = Field(default_factory=NeuronalActivity, description="Neuronal activity data")
    
    # Consciousness markers
    consciousness_markers: ConsciousnessMarkers = Field(default_factory=ConsciousnessMarkers,
                                                       description="Consciousness emergence markers")
    consciousness_contribution: float = Field(default=0.0, ge=0.0, le=1.0,
                                             description="Contribution to overall consciousness")
    
    # Context and environment
    workspace_id: str = Field(..., description="Associated ThoughtSeed workspace ID")
    document_source_id: Optional[str] = Field(None, description="Source document ID if applicable")
    research_query_id: Optional[str] = Field(None, description="Source research query ID if applicable")
    
    # Processing context
    processing_context: Dict[str, Any] = Field(default_factory=dict, description="Processing context data")
    environmental_factors: Dict[str, float] = Field(default_factory=dict, description="Environmental factor influences")
    
    # Quality and confidence
    trace_quality: float = Field(..., ge=0.0, le=1.0, description="Trace quality assessment")
    confidence_level: float = Field(..., ge=0.0, le=1.0, description="Confidence in trace accuracy")
    signal_to_noise_ratio: float = Field(default=1.0, ge=0.0, description="Signal to noise ratio")
    
    # Learning and adaptation
    learning_value: float = Field(default=0.0, ge=0.0, le=1.0, description="Learning value of trace")
    memory_consolidation: bool = Field(default=False, description="Whether trace was consolidated to memory")
    
    # Attractor basin integration
    basin_associations: List[str] = Field(default_factory=list, description="Associated attractor basin IDs")
    basin_influences: Dict[str, float] = Field(default_factory=dict, description="Basin influence strengths")
    field_coherence: float = Field(default=0.0, ge=0.0, le=1.0, description="Neural field coherence at trace time")
    
    # Semantic representation
    semantic_vector: Optional[List[float]] = Field(None, description="Semantic vector representation")
    conceptual_features: Dict[str, float] = Field(default_factory=dict, description="Conceptual feature weights")
    
    # Metadata
    tags: List[str] = Field(default_factory=list, description="Trace tags for classification")
    custom_attributes: Dict[str, Any] = Field(default_factory=dict, description="Custom trace attributes")
    
    @validator('semantic_vector')
    def validate_semantic_vector(cls, v):
        """Validate semantic vector dimensions"""
        if v is not None:
            if not isinstance(v, list) or not all(isinstance(x, (int, float)) for x in v):
                raise ValueError("Semantic vector must be list of numbers")
            if len(v) == 0:
                raise ValueError("Semantic vector cannot be empty")
        return v
    
    def calculate_emergence_contribution(self) -> float:
        """Calculate contribution to consciousness emergence"""
        # Base contribution from pattern strength and layer position
        base_contribution = self.pattern_strength * 0.4
        
        # Layer-specific weighting (higher layers contribute more to consciousness)
        layer_weights = {
            ThoughtseedLayer.SENSORY: 0.1,
            ThoughtseedLayer.PERCEPTUAL: 0.2,
            ThoughtseedLayer.CONCEPTUAL: 0.3,
            ThoughtseedLayer.ABSTRACT: 0.4,
            ThoughtseedLayer.METACOGNITIVE: 0.5
        }
        layer_contribution = layer_weights.get(self.layer, 0.0) * 0.3
        
        # Consciousness markers contribution
        markers_contribution = (
            self.consciousness_markers.meta_awareness_level * 0.1 +
            self.consciousness_markers.coherence_measure * 0.1 +
            self.consciousness_markers.integration_level * 0.1
        )
        
        total_contribution = base_contribution + layer_contribution + markers_contribution
        self.consciousness_contribution = min(total_contribution, 1.0)
        
        return self.consciousness_contribution
    
    def is_emergence_trace(self, threshold: float = 0.7) -> bool:
        """Check if trace indicates consciousness emergence"""
        return (
            self.consciousness_contribution > threshold or
            self.consciousness_markers.self_reference_detected or
            self.consciousness_markers.recursive_depth > 2 or
            len(self.consciousness_markers.emergence_indicators) > 0
        )
    
    class Config:
        """Pydantic configuration"""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        validate_assignment = True
        extra = "forbid"
