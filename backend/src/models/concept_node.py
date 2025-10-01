"""
ConceptNode Model - T017
Flux Self-Evolving Consciousness Emulator

Represents concepts in the consciousness system with semantic relationships,
activation patterns, and integration with thoughtseed attractor dynamics.
Constitutional compliance: mock data transparency, evaluation feedback integration.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Set
from pydantic import BaseModel, Field
from enum import Enum
import uuid


class ConceptType(str, Enum):
    """Concept type classification"""
    PRIMITIVE = "primitive"          # Basic concepts (color, shape, etc.)
    COMPOSITE = "composite"          # Complex composed concepts
    ABSTRACT = "abstract"            # Abstract concepts (justice, beauty)
    PROCEDURAL = "procedural"        # Process/action concepts
    RELATIONAL = "relational"        # Relationship concepts
    EMERGENT = "emergent"            # Dynamically emerging concepts
    META_COGNITIVE = "meta_cognitive" # Self-reflective concepts


class ActivationPattern(str, Enum):
    """Concept activation patterns from neural research"""
    PERSISTENT = "persistent"        # Sustained activation
    TRANSIENT = "transient"         # Brief activation bursts
    OSCILLATORY = "oscillatory"     # Rhythmic activation patterns
    CASCADING = "cascading"         # Spreading activation
    COMPETITIVE = "competitive"     # Winner-take-all dynamics
    COOPERATIVE = "cooperative"     # Mutual enhancement


class ConceptRelation(BaseModel):
    """Relationship between concepts"""
    relation_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Relationship identifier")
    target_concept_id: str = Field(..., description="Target concept ID")
    relation_type: str = Field(..., description="Type of relationship")
    strength: float = Field(..., ge=0.0, le=1.0, description="Relationship strength")

    # Temporal dynamics
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Relationship creation time")
    last_activated: Optional[datetime] = Field(None, description="Last activation time")
    activation_count: int = Field(default=0, description="Number of times activated")

    # Context information
    context_tags: List[str] = Field(default_factory=list, description="Context tags for relationship")
    evidence_sources: List[str] = Field(default_factory=list, description="Evidence source IDs")


class ConceptActivation(BaseModel):
    """Concept activation instance"""
    activation_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Activation identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Activation timestamp")

    # Activation metrics
    activation_strength: float = Field(..., ge=0.0, le=1.0, description="Activation strength")
    activation_pattern: ActivationPattern = Field(..., description="Type of activation pattern")
    duration_ms: Optional[int] = Field(None, description="Activation duration in milliseconds")

    # Context
    trigger_source: Optional[str] = Field(None, description="What triggered this activation")
    context_data: Dict[str, Any] = Field(default_factory=dict, description="Contextual information")

    # Integration with thoughtseed system
    trace_id: Optional[str] = Field(None, description="Associated thoughtseed trace")
    attractor_influence: Dict[str, float] = Field(default_factory=dict, description="Attractor pattern influences")


class ConceptNode(BaseModel):
    """
    Concept node model for the Flux consciousness system.

    Represents individual concepts with semantic relationships, activation patterns,
    and integration with thoughtseed attractor dynamics and consciousness development.
    """

    # Core Identity
    concept_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique concept identifier")
    user_id: str = Field(..., description="Associated user ID")
    journey_id: Optional[str] = Field(None, description="Associated autobiographical journey")

    # Concept Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Concept creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last concept update")
    last_activated: Optional[datetime] = Field(None, description="Last activation timestamp")

    # Concept Definition
    concept_name: str = Field(..., description="Human-readable concept name")
    concept_type: ConceptType = Field(..., description="Type of concept")
    description: Optional[str] = Field(None, description="Concept description")
    aliases: List[str] = Field(default_factory=list, description="Alternative names for concept")

    # Semantic Properties
    semantic_embedding: Optional[List[float]] = Field(None, description="Concept embedding vector")
    semantic_features: Dict[str, float] = Field(default_factory=dict, description="Semantic feature weights")
    definition_sources: List[str] = Field(default_factory=list, description="Definition source IDs")

    # Activation Dynamics
    activation_history: List[ConceptActivation] = Field(default_factory=list, description="Historical activations")
    current_activation: float = Field(default=0.0, ge=0.0, le=1.0, description="Current activation level")
    base_activation: float = Field(default=0.1, ge=0.0, le=1.0, description="Baseline activation level")
    activation_threshold: float = Field(default=0.3, ge=0.0, le=1.0, description="Activation threshold")

    # Concept Relationships
    relations: List[ConceptRelation] = Field(default_factory=list, description="Outgoing concept relationships")
    parent_concepts: List[str] = Field(default_factory=list, description="Parent concept IDs (is-a relationships)")
    child_concepts: List[str] = Field(default_factory=list, description="Child concept IDs")

    # Consciousness Integration
    consciousness_relevance: float = Field(default=0.0, ge=0.0, le=1.0, description="Relevance to consciousness development")
    meta_cognitive_aspects: List[str] = Field(default_factory=list, description="Meta-cognitive aspects of concept")
    emergence_indicators: List[str] = Field(default_factory=list, description="Indicators of emergent properties")

    # ThoughtSeed Integration
    thoughtseed_associations: List[str] = Field(default_factory=list, description="Associated thoughtseed trace IDs")
    attractor_patterns: Dict[str, float] = Field(default_factory=dict, description="Attractor pattern influences")
    memory_context_bindings: Dict[str, float] = Field(default_factory=dict, description="Memory context associations")

    # Learning and Development
    learning_strength: float = Field(default=0.0, ge=0.0, le=1.0, description="How well learned this concept is")
    acquisition_date: Optional[datetime] = Field(None, description="When concept was first acquired")
    refinement_history: List[Dict[str, Any]] = Field(default_factory=list, description="Concept refinement history")

    # Usage Statistics
    total_activations: int = Field(default=0, description="Total number of activations")
    co_activation_patterns: Dict[str, int] = Field(default_factory=dict, description="Co-activation with other concepts")
    context_usage: Dict[str, int] = Field(default_factory=dict, description="Usage in different contexts")

    # Constitutional Compliance
    mock_data_enabled: bool = Field(default=True, description="Mock data mode for development")
    evaluation_feedback_enabled: bool = Field(default=True, description="Evaluation feedback collection enabled")
    privacy_level: str = Field(default="private", description="Concept privacy level")

    # Tags and Classification
    tags: List[str] = Field(default_factory=list, description="User-defined tags")
    categories: List[str] = Field(default_factory=list, description="Auto-classified categories")
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Concept importance score")

    class Config:
        """Pydantic configuration"""
        use_enum_values = True
        validate_assignment = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    def activate(self, strength: float, pattern: ActivationPattern,
                trigger_source: str = None, trace_id: str = None,
                context_data: Dict[str, Any] = None) -> str:
        """Activate concept with specified strength and pattern"""
        activation = ConceptActivation(
            activation_strength=strength,
            activation_pattern=pattern,
            trigger_source=trigger_source,
            trace_id=trace_id,
            context_data=context_data or {}
        )

        self.activation_history.append(activation)
        self.current_activation = max(self.current_activation, strength)
        self.last_activated = datetime.utcnow()
        self.total_activations += 1
        self.updated_at = datetime.utcnow()

        # Update co-activation patterns if triggered by another concept
        if trigger_source and trigger_source.startswith("concept_"):
            source_id = trigger_source.replace("concept_", "")
            self.co_activation_patterns[source_id] = self.co_activation_patterns.get(source_id, 0) + 1

        return activation.activation_id

    def add_relation(self, target_concept_id: str, relation_type: str, strength: float,
                    context_tags: List[str] = None, evidence_sources: List[str] = None) -> str:
        """Add relationship to another concept"""
        relation = ConceptRelation(
            target_concept_id=target_concept_id,
            relation_type=relation_type,
            strength=strength,
            context_tags=context_tags or [],
            evidence_sources=evidence_sources or []
        )

        self.relations.append(relation)
        self.updated_at = datetime.utcnow()

        return relation.relation_id

    def strengthen_relation(self, target_concept_id: str, strength_increment: float = 0.1) -> bool:
        """Strengthen relationship with target concept"""
        for relation in self.relations:
            if relation.target_concept_id == target_concept_id:
                relation.strength = min(relation.strength + strength_increment, 1.0)
                relation.activation_count += 1
                relation.last_activated = datetime.utcnow()
                self.updated_at = datetime.utcnow()
                return True
        return False

    def decay_activation(self, decay_rate: float = 0.1) -> None:
        """Apply activation decay over time"""
        if self.current_activation > self.base_activation:
            self.current_activation = max(
                self.current_activation - decay_rate,
                self.base_activation
            )
            self.updated_at = datetime.utcnow()

    def associate_thoughtseed(self, trace_id: str, relevance_strength: float = 1.0) -> None:
        """Associate concept with thoughtseed trace"""
        if trace_id not in self.thoughtseed_associations:
            self.thoughtseed_associations.append(trace_id)

        # Update attractor patterns
        self.attractor_patterns[f"trace_{trace_id}"] = relevance_strength
        self.updated_at = datetime.utcnow()

    def update_consciousness_relevance(self, new_relevance: float,
                                     emergence_indicators: List[str] = None) -> None:
        """Update consciousness relevance score"""
        self.consciousness_relevance = new_relevance

        if emergence_indicators:
            for indicator in emergence_indicators:
                if indicator not in self.emergence_indicators:
                    self.emergence_indicators.append(indicator)

        self.updated_at = datetime.utcnow()

    def calculate_concept_centrality(self) -> float:
        """Calculate concept centrality based on relationships and activations"""
        # Simple centrality calculation - can be enhanced with more sophisticated graph analysis
        relationship_score = len(self.relations) / 10.0  # Normalize to 0-1 range
        activation_score = min(self.total_activations / 100.0, 1.0)  # Normalize to 0-1 range
        co_activation_score = min(len(self.co_activation_patterns) / 20.0, 1.0)

        centrality = (relationship_score + activation_score + co_activation_score) / 3.0
        return min(centrality, 1.0)

    def get_activation_summary(self, days_back: int = 30) -> Dict[str, Any]:
        """Get summary of recent activation patterns"""
        cutoff_date = datetime.utcnow().timestamp() - (days_back * 24 * 3600)
        recent_activations = [
            act for act in self.activation_history
            if act.timestamp.timestamp() > cutoff_date
        ]

        if not recent_activations:
            return {"total_activations": 0, "avg_strength": 0.0, "patterns": {}}

        pattern_counts = {}
        total_strength = 0.0

        for activation in recent_activations:
            pattern = activation.activation_pattern.value
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            total_strength += activation.activation_strength

        return {
            "total_activations": len(recent_activations),
            "avg_strength": total_strength / len(recent_activations),
            "patterns": pattern_counts,
            "dominant_pattern": max(pattern_counts, key=pattern_counts.get) if pattern_counts else None
        }

    def get_related_concepts(self, min_strength: float = 0.3) -> List[Dict[str, Any]]:
        """Get strongly related concepts"""
        strong_relations = [
            {
                "concept_id": rel.target_concept_id,
                "relation_type": rel.relation_type,
                "strength": rel.strength,
                "activation_count": rel.activation_count
            }
            for rel in self.relations
            if rel.strength >= min_strength
        ]

        # Sort by strength
        strong_relations.sort(key=lambda x: x["strength"], reverse=True)
        return strong_relations

    def refine_concept(self, refinement_type: str, details: Dict[str, Any]) -> None:
        """Record concept refinement"""
        refinement = {
            "timestamp": datetime.utcnow().isoformat(),
            "refinement_type": refinement_type,  # "definition", "relationship", "activation_pattern"
            "details": details,
            "previous_state": {
                "learning_strength": self.learning_strength,
                "importance_score": self.importance_score
            }
        }

        self.refinement_history.append(refinement)
        self.updated_at = datetime.utcnow()

        # Update learning strength based on refinement
        if refinement_type in ["definition", "relationship"]:
            self.learning_strength = min(self.learning_strength + 0.1, 1.0)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return self.dict()

    @classmethod
    def create_mock_concept(cls, user_id: str, concept_name: str, concept_type: ConceptType,
                           journey_id: str = None) -> "ConceptNode":
        """
        Create mock concept node for development/testing.
        Constitutional compliance: clearly marked as mock data.
        """
        concept = cls(
            user_id=user_id,
            journey_id=journey_id,
            concept_name=concept_name,
            concept_type=concept_type,
            description=f"Mock {concept_type.value} concept: {concept_name}",
            mock_data_enabled=True,
            base_activation=0.2,
            current_activation=0.3,
            activation_threshold=0.25,
            consciousness_relevance=0.4,
            learning_strength=0.5,
            importance_score=0.6,
            tags=["mock", "development", concept_type.value],
            categories=["development", "consciousness_research"],
            semantic_features={
                "abstractness": 0.6,
                "complexity": 0.5,
                "familiarity": 0.7
            }
        )

        # Add mock activation
        concept.activate(
            strength=0.6,
            pattern=ActivationPattern.PERSISTENT,
            trigger_source="mock_initialization",
            context_data={"mock_context": True, "initialization_phase": "development"}
        )

        # Add mock relationships if it's a composite concept
        if concept_type == ConceptType.COMPOSITE:
            concept.add_relation(
                target_concept_id="mock_related_concept_1",
                relation_type="composed_of",
                strength=0.7,
                context_tags=["composition", "mock"],
                evidence_sources=["mock_evidence_1"]
            )

        return concept


# Type aliases for convenience
ConceptNodeDict = Dict[str, Any]
ConceptNodeList = List[ConceptNode]