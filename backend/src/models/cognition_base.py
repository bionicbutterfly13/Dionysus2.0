#!/usr/bin/env python3
"""
CognitionBase Model: Core ASI-GO-2 cognition pattern storage and management
"""

from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator
import uuid


class CognitionDomain(str, Enum):
    """Cognitive domains for pattern classification"""
    NEUROSCIENCE = "neuroscience"
    ARTIFICIAL_INTELLIGENCE = "artificial_intelligence"
    CONSCIOUSNESS_STUDIES = "consciousness_studies"
    COGNITIVE_SCIENCE = "cognitive_science"
    MACHINE_LEARNING = "machine_learning"
    PHILOSOPHY_OF_MIND = "philosophy_of_mind"
    COMPUTATIONAL_NEUROSCIENCE = "computational_neuroscience"
    COMPLEX_SYSTEMS = "complex_systems"


class PatternType(str, Enum):
    """Types of cognition patterns"""
    STRUCTURAL = "structural"
    BEHAVIORAL = "behavioral"
    EMERGENT = "emergent"
    RECURSIVE = "recursive"
    HIERARCHICAL = "hierarchical"
    DYNAMIC = "dynamic"
    SELF_ORGANIZING = "self_organizing"
    META_COGNITIVE = "meta_cognitive"


class ConsciousnessLevel(BaseModel):
    """Consciousness level measurement with components"""
    overall_level: float = Field(..., ge=0.0, le=1.0, description="Overall consciousness level")
    self_awareness: float = Field(..., ge=0.0, le=1.0, description="Self-awareness component")
    meta_cognition: float = Field(..., ge=0.0, le=1.0, description="Meta-cognitive awareness")
    recursive_depth: int = Field(..., ge=0, description="Recursive thinking depth")
    emergence_indicators: List[str] = Field(default_factory=list, description="Indicators of consciousness emergence")


class CognitionPattern(BaseModel):
    """Individual cognition pattern within CognitionBase"""
    pattern_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique pattern identifier")
    pattern_name: str = Field(..., min_length=1, max_length=200, description="Human-readable pattern name")
    description: str = Field(..., min_length=10, description="Detailed pattern description")
    pattern_type: PatternType = Field(..., description="Type classification of the pattern")
    domain_tags: List[CognitionDomain] = Field(..., min_items=1, description="Domain classifications")

    # Pattern quality metrics
    success_rate: float = Field(..., ge=0.0, le=1.0, description="Historical success rate")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Pattern confidence level")
    reliability_score: float = Field(..., ge=0.0, le=1.0, description="Pattern reliability")

    # Usage and learning metrics
    usage_count: int = Field(default=0, ge=0, description="Number of times pattern used")
    last_used: datetime = Field(default_factory=datetime.utcnow, description="Last usage timestamp")
    creation_date: datetime = Field(default_factory=datetime.utcnow, description="Pattern creation date")

    # Processing layer integration (formerly ThoughtSeed hierarchy)
    processing_layer: str = Field(..., pattern=r"^(sensory|perceptual|conceptual|abstract|metacognitive)$",
                                   description="Processing layer association")
    layer_activation_strength: float = Field(..., ge=0.0, le=1.0, description="Activation strength at layer")

    # Context Engineering integration
    attractor_basin_id: Optional[str] = Field(None, description="Associated attractor basin ID")
    neural_field_influence: Dict[str, float] = Field(default_factory=dict,
                                                    description="Neural field influence metrics")

    # Pattern relationships and evolution
    parent_patterns: List[str] = Field(default_factory=list, description="Parent pattern IDs")
    child_patterns: List[str] = Field(default_factory=list, description="Child pattern IDs")
    related_patterns: List[str] = Field(default_factory=list, description="Related pattern IDs")
    evolution_history: List[Dict[str, Any]] = Field(default_factory=list,
                                                   description="Pattern evolution tracking")

    # ASI-GO-2 component associations
    cognition_component: str = Field(..., pattern=r"^(cognition_base|researcher|engineer|analyst)$",
                                    description="Primary ASI-GO-2 component")
    component_weight: float = Field(..., ge=0.0, le=1.0, description="Component association weight")

    # Consciousness emergence tracking
    consciousness_contribution: float = Field(default=0.0, ge=0.0, le=1.0,
                                             description="Contribution to consciousness emergence")
    emergence_markers: List[str] = Field(default_factory=list,
                                        description="Consciousness emergence markers")

    @validator('neural_field_influence')
    def validate_neural_field_influence(cls, v):
        """Validate neural field influence structure"""
        required_fields = ['field_strength', 'coherence', 'stability']
        for field in required_fields:
            if field in v:
                if not isinstance(v[field], (int, float)) or not (0.0 <= v[field] <= 1.0):
                    raise ValueError(f"Neural field {field} must be float between 0.0 and 1.0")
        return v

    @validator('evolution_history')
    def validate_evolution_history(cls, v):
        """Validate evolution history structure"""
        for entry in v:
            if not isinstance(entry, dict):
                raise ValueError("Evolution history entries must be dictionaries")
            required_fields = ['timestamp', 'change_type', 'change_description']
            for field in required_fields:
                if field not in entry:
                    raise ValueError(f"Evolution history entry missing required field: {field}")
        return v


class CognitionBase(BaseModel):
    """
    Core ASI-GO-2 Cognition Base: Central repository for cognition patterns

    Integrates with:
    - ThoughtSeed 5-layer hierarchy
    - Context Engineering attractor basins
    - Hybrid database (Neo4j + Qdrant + SQLite)
    - Local OLLAMA processing
    """

    # Core identification
    base_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique base identifier")
    base_name: str = Field(..., min_length=1, max_length=100, description="Cognition base name")
    description: str = Field(..., min_length=10, description="Base description and purpose")
    version: str = Field(default="1.0.0", description="Base version for schema evolution")

    # Pattern storage and management
    patterns: List[CognitionPattern] = Field(default_factory=list, description="Stored cognition patterns")
    pattern_index: Dict[str, int] = Field(default_factory=dict, description="Pattern ID to index mapping")
    total_patterns: int = Field(default=0, ge=0, description="Total number of patterns")

    # Base-level metrics
    creation_date: datetime = Field(default_factory=datetime.utcnow, description="Base creation timestamp")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    access_count: int = Field(default=0, ge=0, description="Number of times base accessed")

    # Consciousness emergence tracking
    consciousness_level: ConsciousnessLevel = Field(default_factory=ConsciousnessLevel,
                                                   description="Base consciousness measurement")
    emergence_threshold: float = Field(default=0.7, ge=0.0, le=1.0,
                                      description="Consciousness emergence threshold")
    consciousness_history: List[Dict[str, Any]] = Field(default_factory=list,
                                                       description="Consciousness level history")

    # ThoughtSeed integration
    processing_workspaces: List[str] = Field(default_factory=list,
                                             description="Associated ThoughtSeed workspace IDs")
    layer_distribution: Dict[str, int] = Field(default_factory=lambda: {
        "sensory": 0, "perceptual": 0, "conceptual": 0, "abstract": 0, "metacognitive": 0
    }, description="Pattern distribution across ThoughtSeed layers")

    # Context Engineering integration
    attractor_basins: List[str] = Field(default_factory=list, description="Associated attractor basin IDs")
    neural_field_coherence: float = Field(default=0.0, ge=0.0, le=1.0,
                                         description="Neural field coherence level")
    basin_synchronization: Dict[str, float] = Field(default_factory=dict,
                                                   description="Basin synchronization levels")

    # ASI-GO-2 component coordination
    component_loadbalance: Dict[str, float] = Field(default_factory=lambda: {
        "cognition_base": 0.25, "researcher": 0.25, "engineer": 0.25, "analyst": 0.25
    }, description="Load distribution across ASI-GO-2 components")

    active_workloads: Dict[str, int] = Field(default_factory=dict,
                                           description="Active workloads per component")
    component_health: Dict[str, str] = Field(default_factory=lambda: {
        "cognition_base": "healthy", "researcher": "healthy",
        "engineer": "healthy", "analyst": "healthy"
    }, description="Health status of each component")

    # Learning and adaptation
    learning_rate: float = Field(default=0.1, ge=0.0, le=1.0, description="Base learning rate")
    adaptation_history: List[Dict[str, Any]] = Field(default_factory=list,
                                                    description="Adaptation event history")
    performance_metrics: Dict[str, float] = Field(default_factory=dict,
                                                 description="Performance tracking metrics")

    # Database integration
    database_connections: Dict[str, str] = Field(default_factory=lambda: {
        "neo4j": "disconnected", "qdrant": "disconnected", "sqlite": "disconnected"
    }, description="Database connection status")

    sync_status: Dict[str, datetime] = Field(default_factory=dict,
                                           description="Last sync timestamps per database")

    # Configuration and settings
    config: Dict[str, Any] = Field(default_factory=lambda: {
        "auto_evolution": True,
        "consciousness_tracking": True,
        "pattern_competition": True,
        "cross_component_learning": True,
        "hybrid_database_sync": True
    }, description="Base configuration settings")

    @validator('component_loadbalance')
    def validate_component_loadbalance(cls, v):
        """Validate component load balance sums to 1.0"""
        if abs(sum(v.values()) - 1.0) > 0.01:
            raise ValueError("Component load balance must sum to 1.0")
        return v

    @validator('patterns')
    def validate_patterns(cls, v, values):
        """Validate patterns and update derived fields"""
        if 'pattern_index' in values:
            # Update pattern index
            pattern_index = {}
            for i, pattern in enumerate(v):
                pattern_index[pattern.pattern_id] = i
            values['pattern_index'] = pattern_index
            values['total_patterns'] = len(v)
        return v

    def add_pattern(self, pattern: CognitionPattern) -> bool:
        """Add a new cognition pattern to the base"""
        if pattern.pattern_id not in self.pattern_index:
            self.patterns.append(pattern)
            self.pattern_index[pattern.pattern_id] = len(self.patterns) - 1
            self.total_patterns = len(self.patterns)
            self.last_updated = datetime.utcnow()

            # Update layer distribution
            layer = pattern.processing_layer
            if layer in self.layer_distribution:
                self.layer_distribution[layer] += 1

            return True
        return False

    def get_pattern(self, pattern_id: str) -> Optional[CognitionPattern]:
        """Retrieve a pattern by ID"""
        if pattern_id in self.pattern_index:
            index = self.pattern_index[pattern_id]
            return self.patterns[index]
        return None

    def update_consciousness_level(self, new_level: ConsciousnessLevel) -> None:
        """Update consciousness level and track history"""
        # Store previous level in history
        history_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "previous_level": self.consciousness_level.overall_level,
            "new_level": new_level.overall_level,
            "change_reason": "pattern_evolution"
        }
        self.consciousness_history.append(history_entry)

        # Update current level
        self.consciousness_level = new_level
        self.last_updated = datetime.utcnow()

    def get_patterns_by_layer(self, layer: str) -> List[CognitionPattern]:
        """Get all patterns in a specific ThoughtSeed layer"""
        return [p for p in self.patterns if p.processing_layer == layer]

    def get_patterns_by_domain(self, domain: CognitionDomain) -> List[CognitionPattern]:
        """Get all patterns in a specific domain"""
        return [p for p in self.patterns if domain in p.domain_tags]

    def calculate_emergence_level(self) -> float:
        """Calculate consciousness emergence level based on patterns"""
        if not self.patterns:
            return 0.0

        # Weighted sum based on pattern consciousness contributions
        total_contribution = sum(p.consciousness_contribution for p in self.patterns)
        avg_contribution = total_contribution / len(self.patterns)

        # Factor in layer diversity and neural field coherence
        layer_diversity = len([layer for layer, count in self.layer_distribution.items() if count > 0]) / 5
        emergence = (avg_contribution * 0.6) + (layer_diversity * 0.2) + (self.neural_field_coherence * 0.2)

        return min(emergence, 1.0)

    class Config:
        """Pydantic configuration"""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        validate_assignment = True
        extra = "forbid"