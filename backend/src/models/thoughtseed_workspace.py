#!/usr/bin/env python3
"""
ThoughtseedWorkspace Model: 5-layer ThoughtSeed processing workspace
"""

from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator
import uuid


class ProcessingStatus(str, Enum):
    """Workspace processing status"""
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    ERROR = "error"
    PAUSED = "paused"


class LayerStatus(str, Enum):
    """Individual layer processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"
    SKIPPED = "skipped"


class ThoughtseedLayer(str, Enum):
    """ThoughtSeed hierarchy layers"""
    SENSORY = "sensory"
    PERCEPTUAL = "perceptual"
    CONCEPTUAL = "conceptual"
    ABSTRACT = "abstract"
    METACOGNITIVE = "metacognitive"


class LayerProcessingState(BaseModel):
    """Processing state for individual ThoughtSeed layer"""
    layer: ThoughtseedLayer = Field(..., description="Layer identifier")
    status: LayerStatus = Field(default=LayerStatus.PENDING, description="Layer processing status")
    processing_time_ms: int = Field(default=0, ge=0, description="Processing time in milliseconds")
    patterns_detected: List[str] = Field(default_factory=list, description="Pattern IDs detected in this layer")
    
    # Layer-specific metrics
    activation_strength: float = Field(default=0.0, ge=0.0, le=1.0, description="Layer activation strength")
    complexity_measure: float = Field(default=0.0, ge=0.0, description="Processing complexity measure")
    consciousness_contribution: float = Field(default=0.0, ge=0.0, le=1.0, description="Contribution to consciousness")
    
    # Processing details
    start_time: Optional[datetime] = Field(None, description="Layer processing start time")
    end_time: Optional[datetime] = Field(None, description="Layer processing end time")
    error_message: Optional[str] = Field(None, description="Error message if status is ERROR")
    
    # Layer outputs
    processed_content: Dict[str, Any] = Field(default_factory=dict, description="Processed content from layer")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Layer processing metadata")


class NeuronalPacket(BaseModel):
    """Neuronal packet for ThoughtSeed processing"""
    packet_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique packet identifier")
    layer: ThoughtseedLayer = Field(..., description="Layer where packet is active")
    signal_strength: float = Field(..., ge=0.0, le=1.0, description="Signal strength")
    processing_state: str = Field(..., description="Packet processing state")
    
    # Packet connections and propagation
    connections: List[str] = Field(default_factory=list, description="Connected packet IDs")
    propagation_path: List[str] = Field(default_factory=list, description="Propagation path through layers")
    
    # Content and processing
    content_hash: Optional[str] = Field(None, description="Content hash for packet identification")
    processing_data: Dict[str, Any] = Field(default_factory=dict, description="Processing-specific data")
    
    # Temporal tracking
    creation_time: datetime = Field(default_factory=datetime.utcnow, description="Packet creation time")
    last_update: datetime = Field(default_factory=datetime.utcnow, description="Last update time")


class ThoughtseedTrace(BaseModel):
    """ThoughtSeed processing trace"""
    trace_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique trace identifier")
    layer: ThoughtseedLayer = Field(..., description="Layer where trace was generated")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Trace timestamp")
    pattern_strength: float = Field(..., ge=0.0, le=1.0, description="Pattern strength in trace")
    neuronal_activity: Dict[str, Any] = Field(default_factory=dict, description="Neuronal activity data")
    
    # Trace relationships
    parent_traces: List[str] = Field(default_factory=list, description="Parent trace IDs")
    child_traces: List[str] = Field(default_factory=list, description="Child trace IDs")
    
    # Content and context
    content_snippet: Optional[str] = Field(None, description="Content snippet that generated trace")
    context_data: Dict[str, Any] = Field(default_factory=dict, description="Context information")


class AttractorBasinState(BaseModel):
    """State of attractor basin in workspace"""
    basin_id: str = Field(..., description="Attractor basin identifier")
    stability: float = Field(..., ge=0.0, le=1.0, description="Basin stability")
    activation_level: float = Field(..., ge=0.0, le=1.0, description="Current activation level")
    connected_basins: List[str] = Field(default_factory=list, description="Connected basin IDs")
    
    # Basin dynamics
    field_influence: Dict[str, float] = Field(default_factory=dict, description="Neural field influence")
    energy_level: float = Field(default=0.0, description="Energy level in basin")
    convergence_rate: float = Field(default=0.0, description="Rate of convergence to attractor")


class ThoughtseedWorkspace(BaseModel):
    """
    ThoughtSeed Workspace: 5-layer consciousness processing workspace
    
    Manages processing through the ThoughtSeed hierarchy:
    sensory → perceptual → conceptual → abstract → metacognitive
    
    Integrates with Context Engineering attractor basins and neural fields.
    """
    
    # Core identification
    workspace_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique workspace identifier")
    workspace_name: Optional[str] = Field(None, description="Human-readable workspace name")
    description: Optional[str] = Field(None, description="Workspace description")
    
    # Processing state
    processing_status: ProcessingStatus = Field(default=ProcessingStatus.PENDING, description="Overall processing status")
    current_layer: str = Field(default="sensory", description="Currently active layer")
    
    # 5-layer processing states
    layer_states: Dict[str, LayerProcessingState] = Field(
        default_factory=lambda: {
            "sensory": LayerProcessingState(layer=ThoughtseedLayer.SENSORY),
            "perceptual": LayerProcessingState(layer=ThoughtseedLayer.PERCEPTUAL),
            "conceptual": LayerProcessingState(layer=ThoughtseedLayer.CONCEPTUAL),
            "abstract": LayerProcessingState(layer=ThoughtseedLayer.ABSTRACT),
            "metacognitive": LayerProcessingState(layer=ThoughtseedLayer.METACOGNITIVE)
        },
        description="Processing state for each layer"
    )
    
    # ThoughtSeed traces and patterns
    traces_generated: List[ThoughtseedTrace] = Field(default_factory=list, description="Generated traces")
    patterns_discovered: List[str] = Field(default_factory=list, description="Discovered pattern IDs")
    
    # Consciousness measurement
    consciousness_level: float = Field(default=0.0, ge=0.0, le=1.0, description="Overall consciousness level")
    consciousness_components: Dict[str, float] = Field(
        default_factory=lambda: {
            "self_awareness": 0.0,
            "meta_cognition": 0.0,
            "recursive_depth": 0.0,
            "emergence_indicators": 0.0
        },
        description="Consciousness component scores"
    )
    
    # Context Engineering integration
    attractor_basin_states: Dict[str, AttractorBasinState] = Field(
        default_factory=dict, description="Attractor basin states"
    )
    neural_field_coherence: float = Field(default=0.0, ge=0.0, le=1.0, description="Neural field coherence")
    
    # Neuronal packet processing
    neuronal_packets: List[NeuronalPacket] = Field(default_factory=list, description="Active neuronal packets")
    packet_flow: Dict[str, List[str]] = Field(
        default_factory=dict, description="Packet flow between layers"
    )
    
    # Temporal tracking
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Workspace creation time")
    started_at: Optional[datetime] = Field(None, description="Processing start time")
    completed_at: Optional[datetime] = Field(None, description="Processing completion time")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last update time")
    
    # Processing configuration
    requested_layers: List[ThoughtseedLayer] = Field(
        default_factory=lambda: list(ThoughtseedLayer),
        description="Requested layers for processing"
    )
    layer_dependencies: Dict[str, List[str]] = Field(
        default_factory=lambda: {
            "perceptual": ["sensory"],
            "conceptual": ["perceptual"],
            "abstract": ["conceptual"],
            "metacognitive": ["abstract"]
        },
        description="Layer dependency mapping"
    )
    
    # Performance metrics
    total_processing_time_ms: int = Field(default=0, ge=0, description="Total processing time")
    layer_transition_times: Dict[str, int] = Field(
        default_factory=dict, description="Time spent in layer transitions"
    )
    throughput_metrics: Dict[str, float] = Field(
        default_factory=dict, description="Throughput metrics per layer"
    )
    
    # Source and context
    source_document_id: Optional[str] = Field(None, description="Source document ID if applicable")
    research_query_id: Optional[str] = Field(None, description="Source research query ID if applicable")
    context_data: Dict[str, Any] = Field(default_factory=dict, description="Additional context data")
    
    # Configuration
    config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "consciousness_threshold": 0.7,
            "basin_activation_threshold": 0.5,
            "trace_retention_limit": 1000,
            "packet_timeout_ms": 30000
        },
        description="Workspace configuration"
    )
    
    @validator('layer_states')
    def validate_layer_states(cls, v):
        """Validate all required layers are present"""
        required_layers = {"sensory", "perceptual", "conceptual", "abstract", "metacognitive"}
        if set(v.keys()) != required_layers:
            raise ValueError(f"Must have states for all layers: {required_layers}")
        return v
    
    @validator('consciousness_components')
    def validate_consciousness_components(cls, v):
        """Validate consciousness components"""
        for component, value in v.items():
            if not isinstance(value, (int, float)) or not (0.0 <= value <= 1.0):
                raise ValueError(f"Consciousness component {component} must be between 0.0 and 1.0")
        return v
    
    def start_processing(self) -> None:
        """Start workspace processing"""
        if self.processing_status == ProcessingStatus.PENDING:
            self.processing_status = ProcessingStatus.ACTIVE
            self.started_at = datetime.utcnow()
            self.current_layer = "sensory"
            
            # Initialize first layer
            self.layer_states["sensory"].status = LayerStatus.PROCESSING
            self.layer_states["sensory"].start_time = datetime.utcnow()
    
    def complete_layer(self, layer: str, processing_time_ms: int, patterns_detected: List[str]) -> None:
        """Complete processing for a specific layer"""
        if layer in self.layer_states:
            layer_state = self.layer_states[layer]
            layer_state.status = LayerStatus.COMPLETED
            layer_state.end_time = datetime.utcnow()
            layer_state.processing_time_ms = processing_time_ms
            layer_state.patterns_detected = patterns_detected
            
            # Update total processing time
            self.total_processing_time_ms += processing_time_ms
            
            # Advance to next layer if dependencies are met
            self._advance_to_next_layer()
    
    def _advance_to_next_layer(self) -> None:
        """Advance to next layer based on dependencies"""
        layer_order = ["sensory", "perceptual", "conceptual", "abstract", "metacognitive"]
        
        for layer in layer_order:
            if layer in self.requested_layers and self.layer_states[layer].status == LayerStatus.PENDING:
                # Check if dependencies are met
                dependencies = self.layer_dependencies.get(layer, [])
                dependencies_met = all(
                    self.layer_states.get(dep_layer, LayerProcessingState(layer=dep_layer)).status == LayerStatus.COMPLETED
                    for dep_layer in dependencies
                )
                
                if dependencies_met:
                    self.current_layer = layer
                    self.layer_states[layer].status = LayerStatus.PROCESSING
                    self.layer_states[layer].start_time = datetime.utcnow()
                    return
        
        # If no more layers to process, complete workspace
        all_requested_complete = all(
            self.layer_states[layer].status == LayerStatus.COMPLETED
            for layer in self.requested_layers
        )
        
        if all_requested_complete:
            self.processing_status = ProcessingStatus.COMPLETED
            self.completed_at = datetime.utcnow()
            self.current_layer = "completed"
    
    def add_trace(self, layer: ThoughtseedLayer, pattern_strength: float, 
                  neuronal_activity: Dict[str, Any]) -> ThoughtseedTrace:
        """Add a new ThoughtSeed trace"""
        trace = ThoughtseedTrace(
            layer=layer,
            pattern_strength=pattern_strength,
            neuronal_activity=neuronal_activity
        )
        
        self.traces_generated.append(trace)
        
        # Limit trace retention
        max_traces = self.config.get("trace_retention_limit", 1000)
        if len(self.traces_generated) > max_traces:
            self.traces_generated = self.traces_generated[-max_traces:]
        
        return trace
    
    def update_consciousness_level(self) -> float:
        """Update consciousness level based on layer states and traces"""
        # Calculate based on completed layers and their contributions
        total_contribution = 0.0
        completed_layers = 0
        
        for layer_name, layer_state in self.layer_states.items():
            if layer_state.status == LayerStatus.COMPLETED:
                total_contribution += layer_state.consciousness_contribution
                completed_layers += 1
        
        # Factor in trace quality and quantity
        trace_factor = min(len(self.traces_generated) / 100, 1.0) * 0.1
        
        # Factor in neural field coherence
        coherence_factor = self.neural_field_coherence * 0.2
        
        # Calculate final consciousness level
        base_consciousness = total_contribution / max(1, len(self.requested_layers))
        self.consciousness_level = min(base_consciousness + trace_factor + coherence_factor, 1.0)
        
        return self.consciousness_level
    
    def get_active_packets(self) -> List[NeuronalPacket]:
        """Get currently active neuronal packets"""
        return [packet for packet in self.neuronal_packets 
                if packet.processing_state in ["active", "propagating"]]
    
    def is_layer_ready(self, layer: str) -> bool:
        """Check if layer is ready for processing"""
        if layer not in self.layer_states:
            return False
        
        # Check dependencies
        dependencies = self.layer_dependencies.get(layer, [])
        return all(
            self.layer_states.get(dep_layer, LayerProcessingState(layer=dep_layer)).status == LayerStatus.COMPLETED
            for dep_layer in dependencies
        )
    
    class Config:
        """Pydantic configuration"""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        validate_assignment = True
        extra = "forbid"