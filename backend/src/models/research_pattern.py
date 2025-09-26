#!/usr/bin/env python3
"""
ResearchPattern Model: ASI-GO-2 research intelligence pattern representation
"""

from typing import List, Dict, Optional, Any, Set
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator
import uuid


class ResearchDomain(str, Enum):
    """Research domains for pattern classification"""
    NEUROSCIENCE = "neuroscience"
    ARTIFICIAL_INTELLIGENCE = "artificial_intelligence"
    CONSCIOUSNESS_STUDIES = "consciousness_studies"
    COGNITIVE_SCIENCE = "cognitive_science"
    MACHINE_LEARNING = "machine_learning"
    DEEP_LEARNING = "deep_learning"
    NATURAL_LANGUAGE_PROCESSING = "natural_language_processing"
    COMPUTER_VISION = "computer_vision"
    ROBOTICS = "robotics"
    COMPLEX_SYSTEMS = "complex_systems"
    PHILOSOPHY_OF_MIND = "philosophy_of_mind"
    COMPUTATIONAL_NEUROSCIENCE = "computational_neuroscience"


class PatternComplexity(str, Enum):
    """Pattern complexity levels"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    HIGHLY_COMPLEX = "highly_complex"
    EMERGENT = "emergent"


class PatternStatus(str, Enum):
    """Pattern lifecycle status"""
    DISCOVERED = "discovered"
    VALIDATED = "validated"
    ACTIVE = "active"
    EVOLVING = "evolving"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class CompetitionMetrics(BaseModel):
    """Pattern competition performance metrics"""
    competition_score: float = Field(..., ge=0.0, le=1.0, description="Overall competition performance")
    selection_frequency: float = Field(..., ge=0.0, le=1.0, description="How often pattern is selected")
    win_rate: float = Field(..., ge=0.0, le=1.0, description="Success rate when selected")
    average_rank: float = Field(..., ge=0.0, description="Average ranking in competitions")
    competitor_patterns: List[str] = Field(default_factory=list, description="Frequently competing pattern IDs")
    last_competition: Optional[datetime] = Field(None, description="Last competition timestamp")
    total_competitions: int = Field(default=0, ge=0, description="Total competitions participated")


class PatternEvolution(BaseModel):
    """Pattern evolution tracking"""
    evolution_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Evolution event ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Evolution timestamp")
    evolution_type: str = Field(..., description="Type of evolution (refinement, merge, split, etc.)")
    description: str = Field(..., min_length=10, description="Evolution description")
    trigger_event: str = Field(..., description="What triggered this evolution")
    previous_version: Optional[Dict[str, Any]] = Field(None, description="Previous pattern state")
    confidence_change: float = Field(..., description="Change in confidence (-1.0 to 1.0)")
    performance_impact: Dict[str, float] = Field(default_factory=dict, description="Performance impact metrics")


class ResearchPattern(BaseModel):
    """
    Research Pattern: Core intelligence pattern for ASI-GO-2 research capabilities

    Represents patterns discovered and used by the ASI-GO-2 system for
    research intelligence, pattern competition, and knowledge synthesis.
    """

    # Core identification
    pattern_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique pattern identifier")
    pattern_name: str = Field(..., min_length=1, max_length=200, description="Human-readable pattern name")
    description: str = Field(..., min_length=10, description="Detailed pattern description")
    pattern_type: str = Field(..., min_length=1, description="Pattern type classification")

    # Research domain and classification
    domain_tags: List[ResearchDomain] = Field(..., min_items=1, description="Research domain classifications")
    complexity_level: PatternComplexity = Field(..., description="Pattern complexity classification")
    status: PatternStatus = Field(default=PatternStatus.DISCOVERED, description="Pattern lifecycle status")

    # Core pattern metrics
    success_rate: float = Field(..., ge=0.0, le=1.0, description="Historical success rate")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Pattern confidence level")
    reliability_score: float = Field(..., ge=0.0, le=1.0, description="Pattern reliability")
    accuracy_measure: float = Field(..., ge=0.0, le=1.0, description="Pattern accuracy when applied")

    # Usage and application tracking
    usage_count: int = Field(default=0, ge=0, description="Number of times pattern applied")
    last_used: datetime = Field(default_factory=datetime.utcnow, description="Last usage timestamp")
    first_discovered: datetime = Field(default_factory=datetime.utcnow, description="Discovery timestamp")
    last_validated: Optional[datetime] = Field(None, description="Last validation timestamp")

    # ThoughtSeed hierarchy integration
    thoughtseed_layer: str = Field(..., regex=r"^(sensory|perceptual|conceptual|abstract|metacognitive)$",
                                   description="Primary ThoughtSeed layer")
    layer_activation_pattern: Dict[str, float] = Field(default_factory=dict,
                                                      description="Activation across all layers")
    hierarchical_dependencies: List[str] = Field(default_factory=list,
                                                description="Dependent layer relationships")

    # Pattern competition system
    competition_metrics: CompetitionMetrics = Field(default_factory=CompetitionMetrics,
                                                   description="Competition performance metrics")
    competitive_advantages: List[str] = Field(default_factory=list,
                                             description="Competitive advantage descriptions")
    weakness_indicators: List[str] = Field(default_factory=list,
                                          description="Known pattern weaknesses")

    # ASI-GO-2 component integration
    primary_component: str = Field(..., regex=r"^(cognition_base|researcher|engineer|analyst)$",
                                  description="Primary ASI-GO-2 component")
    component_affinities: Dict[str, float] = Field(default_factory=lambda: {
        "cognition_base": 0.0, "researcher": 0.0, "engineer": 0.0, "analyst": 0.0
    }, description="Affinity scores for each component")

    # Knowledge representation
    knowledge_structure: Dict[str, Any] = Field(default_factory=dict,
                                               description="Structured knowledge representation")
    semantic_embedding: Optional[List[float]] = Field(None, description="Vector embedding representation")
    conceptual_links: List[str] = Field(default_factory=list, description="Links to related concepts")

    # Pattern relationships
    parent_patterns: List[str] = Field(default_factory=list, description="Parent pattern IDs")
    child_patterns: List[str] = Field(default_factory=list, description="Child pattern IDs")
    related_patterns: List[str] = Field(default_factory=list, description="Related pattern IDs")
    antagonistic_patterns: List[str] = Field(default_factory=list,
                                            description="Patterns that conflict with this one")

    # Evolution and adaptation
    evolution_history: List[PatternEvolution] = Field(default_factory=list,
                                                     description="Pattern evolution tracking")
    adaptation_rate: float = Field(default=0.1, ge=0.0, le=1.0, description="Rate of pattern adaptation")
    stability_score: float = Field(..., ge=0.0, le=1.0, description="Pattern stability over time")

    # Context Engineering integration
    attractor_basin_associations: List[str] = Field(default_factory=list,
                                                   description="Associated attractor basin IDs")
    neural_field_signature: Dict[str, float] = Field(default_factory=dict,
                                                    description="Neural field influence signature")
    consciousness_markers: List[str] = Field(default_factory=list,
                                            description="Consciousness emergence markers")

    # Performance and optimization
    execution_time_ms: Optional[int] = Field(None, ge=0, description="Average execution time in milliseconds")
    memory_footprint: Optional[int] = Field(None, ge=0, description="Memory usage in bytes")
    computational_complexity: Optional[str] = Field(None, description="Big O complexity notation")
    optimization_level: int = Field(default=0, ge=0, le=10, description="Optimization level (0-10)")

    # Quality assurance
    validation_tests: List[str] = Field(default_factory=list, description="Validation test identifiers")
    test_coverage: float = Field(default=0.0, ge=0.0, le=1.0, description="Test coverage percentage")
    quality_score: float = Field(..., ge=0.0, le=1.0, description="Overall quality assessment")

    # Metadata and provenance
    source_documents: List[str] = Field(default_factory=list, description="Source document IDs")
    author_system: str = Field(..., description="System or component that created this pattern")
    verification_status: str = Field(default="unverified", description="Verification status")
    tags: Set[str] = Field(default_factory=set, description="Additional tags for classification")

    # Configuration and parameters
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Pattern-specific parameters")
    hyperparameters: Dict[str, float] = Field(default_factory=dict,
                                             description="Tunable hyperparameters")
    configuration: Dict[str, Any] = Field(default_factory=dict,
                                         description="Pattern configuration settings")

    @validator('layer_activation_pattern')
    def validate_layer_activation_pattern(cls, v):
        """Validate layer activation pattern structure"""
        valid_layers = {"sensory", "perceptual", "conceptual", "abstract", "metacognitive"}
        for layer, activation in v.items():
            if layer not in valid_layers:
                raise ValueError(f"Invalid layer: {layer}")
            if not isinstance(activation, (int, float)) or not (0.0 <= activation <= 1.0):
                raise ValueError(f"Layer activation must be float between 0.0 and 1.0")
        return v

    @validator('component_affinities')
    def validate_component_affinities(cls, v):
        """Validate component affinity scores"""
        required_components = {"cognition_base", "researcher", "engineer", "analyst"}
        if set(v.keys()) != required_components:
            raise ValueError(f"Must have affinities for all components: {required_components}")

        for component, affinity in v.items():
            if not isinstance(affinity, (int, float)) or not (0.0 <= affinity <= 1.0):
                raise ValueError(f"Component affinity must be float between 0.0 and 1.0")
        return v

    @validator('semantic_embedding')
    def validate_semantic_embedding(cls, v):
        """Validate semantic embedding dimensions"""
        if v is not None:
            if not isinstance(v, list) or not all(isinstance(x, (int, float)) for x in v):
                raise ValueError("Semantic embedding must be list of numbers")
            if len(v) == 0:
                raise ValueError("Semantic embedding cannot be empty")
        return v

    @validator('tags')
    def validate_tags(cls, v):
        """Validate tags format"""
        if v is not None:
            # Convert to set and validate each tag
            tag_set = set()
            for tag in v:
                if not isinstance(tag, str) or len(tag) == 0:
                    raise ValueError("Tags must be non-empty strings")
                tag_set.add(tag.lower().strip())
            return tag_set
        return set()

    def add_evolution(self, evolution_type: str, description: str, trigger_event: str,
                     confidence_change: float = 0.0) -> PatternEvolution:
        """Add evolution event to pattern history"""
        evolution = PatternEvolution(
            evolution_type=evolution_type,
            description=description,
            trigger_event=trigger_event,
            confidence_change=confidence_change,
            previous_version={
                "confidence": self.confidence,
                "success_rate": self.success_rate,
                "usage_count": self.usage_count
            }
        )

        self.evolution_history.append(evolution)

        # Apply confidence change
        new_confidence = max(0.0, min(1.0, self.confidence + confidence_change))
        self.confidence = new_confidence

        return evolution

    def calculate_overall_performance(self) -> float:
        """Calculate overall pattern performance score"""
        # Weighted combination of metrics
        performance = (
            self.success_rate * 0.3 +
            self.confidence * 0.2 +
            self.reliability_score * 0.2 +
            self.accuracy_measure * 0.15 +
            self.quality_score * 0.1 +
            self.competition_metrics.competition_score * 0.05
        )
        return min(performance, 1.0)

    def get_activation_for_layer(self, layer: str) -> float:
        """Get activation level for specific ThoughtSeed layer"""
        return self.layer_activation_pattern.get(layer, 0.0)

    def update_competition_metrics(self, selected: bool, rank: int, total_competitors: int) -> None:
        """Update competition metrics based on competition result"""
        # Update total competitions
        self.competition_metrics.total_competitions += 1

        # Update selection frequency
        if selected:
            current_selections = self.competition_metrics.selection_frequency * (self.competition_metrics.total_competitions - 1)
            self.competition_metrics.selection_frequency = (current_selections + 1) / self.competition_metrics.total_competitions

            # Update win rate (if selected, consider it a win)
            current_wins = self.competition_metrics.win_rate * current_selections if current_selections > 0 else 0
            self.competition_metrics.win_rate = (current_wins + 1) / (current_selections + 1)
        else:
            current_selections = self.competition_metrics.selection_frequency * (self.competition_metrics.total_competitions - 1)
            self.competition_metrics.selection_frequency = current_selections / self.competition_metrics.total_competitions

        # Update average rank
        current_total_rank = self.competition_metrics.average_rank * (self.competition_metrics.total_competitions - 1)
        self.competition_metrics.average_rank = (current_total_rank + rank) / self.competition_metrics.total_competitions

        # Update competition score based on rank performance
        rank_score = 1.0 - (rank - 1) / max(1, total_competitors - 1)
        current_total_score = self.competition_metrics.competition_score * (self.competition_metrics.total_competitions - 1)
        self.competition_metrics.competition_score = (current_total_score + rank_score) / self.competition_metrics.total_competitions

        # Update last competition time
        self.competition_metrics.last_competition = datetime.utcnow()

    def is_compatible_with(self, other_pattern: 'ResearchPattern') -> bool:
        """Check compatibility with another research pattern"""
        # Check for domain overlap
        domain_overlap = bool(set(self.domain_tags) & set(other_pattern.domain_tags))

        # Check for antagonistic relationship
        is_antagonistic = (
            other_pattern.pattern_id in self.antagonistic_patterns or
            self.pattern_id in other_pattern.antagonistic_patterns
        )

        # Check ThoughtSeed layer compatibility
        layer_compatibility = (
            self.thoughtseed_layer == other_pattern.thoughtseed_layer or
            abs(self._get_layer_index(self.thoughtseed_layer) -
                self._get_layer_index(other_pattern.thoughtseed_layer)) <= 1
        )

        return domain_overlap and not is_antagonistic and layer_compatibility

    def _get_layer_index(self, layer: str) -> int:
        """Get numerical index for ThoughtSeed layer"""
        layer_indices = {
            "sensory": 0, "perceptual": 1, "conceptual": 2,
            "abstract": 3, "metacognitive": 4
        }
        return layer_indices.get(layer, 0)

    class Config:
        """Pydantic configuration"""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            set: lambda v: list(v)
        }
        validate_assignment = True
        extra = "forbid"