"""
Concept Node Models
Constitutional compliance: mock data transparency, evaluative feedback
Implements knowledge graph concept nodes with consciousness tracking
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Set
from datetime import datetime
from enum import Enum

class ConceptType(str, Enum):
    """Types of concept nodes"""
    PRIMITIVE = "primitive"          # Basic, indivisible concepts
    COMPOUND = "compound"            # Composed of multiple primitives
    ABSTRACT = "abstract"            # Abstract conceptual structures
    EMERGENT = "emergent"            # Emerged during consciousness events
    ARCHETYPAL = "archetypal"        # Jungian archetypal patterns
    PROCEDURAL = "procedural"        # Process or method concepts
    RELATIONAL = "relational"        # Relationship concepts

class ConceptStatus(str, Enum):
    """Concept development status"""
    NASCENT = "nascent"              # Just formed, unstable
    DEVELOPING = "developing"        # Growing and stabilizing
    STABLE = "stable"                # Well-established
    MATURE = "mature"                # Fully developed
    DECLINING = "declining"          # Losing relevance
    ARCHIVED = "archived"            # Historical but inactive

class ConceptConfidence(str, Enum):
    """Confidence levels for concept validity"""
    LOW = "low"                      # 0.0 - 0.3
    MEDIUM = "medium"                # 0.3 - 0.7
    HIGH = "high"                    # 0.7 - 0.9
    VERY_HIGH = "very_high"          # 0.9 - 1.0

class ConceptRelationship(BaseModel):
    """Relationship between concept nodes"""

    # Relationship identity
    id: str = Field(..., description="Unique relationship identifier")
    source_concept_id: str = Field(..., description="Source concept node ID")
    target_concept_id: str = Field(..., description="Target concept node ID")

    # Relationship characteristics
    relationship_type: str = Field(..., description="Type of relationship")
    relationship_strength: float = Field(..., description="Strength of relationship (0-1)", ge=0.0, le=1.0)
    directionality: str = Field(default="bidirectional", description="Relationship direction")

    # Relationship dynamics
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Relationship creation time")
    last_reinforced: Optional[datetime] = Field(default=None, description="Last reinforcement timestamp")
    reinforcement_count: int = Field(default=0, description="Number of reinforcements")

    # Evidence and validation
    supporting_events: List[str] = Field(default_factory=list, description="Events that support this relationship")
    confidence_score: float = Field(default=0.5, description="Confidence in relationship", ge=0.0, le=1.0)

    # Context
    context_conditions: List[str] = Field(default_factory=list, description="Context conditions where relationship applies")

    # Constitutional compliance
    mock_data: bool = Field(default=True, description="Constitutional requirement: mock data flag")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ConceptActivation(BaseModel):
    """Concept activation during consciousness events"""

    # Activation identity
    id: str = Field(..., description="Unique activation identifier")
    concept_id: str = Field(..., description="Activated concept ID")
    event_id: str = Field(..., description="Event that triggered activation")

    # Activation characteristics
    activation_timestamp: datetime = Field(default_factory=datetime.utcnow, description="When concept was activated")
    activation_strength: float = Field(..., description="Activation strength (0-1)", ge=0.0, le=1.0)
    activation_duration: Optional[int] = Field(default=None, description="Duration in milliseconds")

    # Context during activation
    consciousness_level: float = Field(..., description="Consciousness level during activation", ge=0.0, le=1.0)
    attention_focus: float = Field(default=0.0, description="Attention focused on concept", ge=0.0, le=1.0)
    emotional_valence: float = Field(default=0.0, description="Emotional response to concept", ge=-1.0, le=1.0)

    # Co-activations
    co_activated_concepts: List[str] = Field(default_factory=list, description="Other concepts activated simultaneously")
    competing_concepts: List[str] = Field(default_factory=list, description="Concepts competing for activation")

    # Outcomes
    led_to_insights: bool = Field(default=False, description="Whether activation led to insights")
    spawned_new_concepts: List[str] = Field(default_factory=list, description="New concepts spawned")
    triggered_curiosity: bool = Field(default=False, description="Whether triggered curiosity mission")

    # Constitutional compliance
    mock_data: bool = Field(default=True, description="Constitutional requirement: mock data flag")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ConceptNode(BaseModel):
    """Knowledge graph concept node with consciousness integration"""

    # Node identity
    id: str = Field(..., description="Unique concept identifier")
    user_id: str = Field(..., description="Associated user identifier")

    # Concept metadata
    name: str = Field(..., description="Concept name")
    aliases: List[str] = Field(default_factory=list, description="Alternative names")
    description: str = Field(..., description="Concept description")
    definition: Optional[str] = Field(default=None, description="Formal definition")

    # Concept classification
    concept_type: ConceptType = Field(..., description="Type of concept")
    domain: str = Field(..., description="Domain or field of concept")
    subdomains: List[str] = Field(default_factory=list, description="Subdomains")
    tags: List[str] = Field(default_factory=list, description="Concept tags")

    # Development tracking
    status: ConceptStatus = Field(default=ConceptStatus.NASCENT, description="Concept development status")
    confidence: ConceptConfidence = Field(default=ConceptConfidence.LOW, description="Confidence level")
    maturity_score: float = Field(default=0.0, description="Concept maturity (0-1)", ge=0.0, le=1.0)

    # Creation and evolution
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Concept creation time")
    updated_at: Optional[datetime] = Field(default=None, description="Last update time")
    origin_event_id: Optional[str] = Field(default=None, description="Event that created this concept")
    evolution_history: List[Dict[str, Any]] = Field(default_factory=list, description="Concept evolution history")

    # Relationships
    parent_concepts: List[str] = Field(default_factory=list, description="Parent concept IDs")
    child_concepts: List[str] = Field(default_factory=list, description="Child concept IDs")
    related_concepts: List[str] = Field(default_factory=list, description="Related concept IDs")
    relationship_details: List[ConceptRelationship] = Field(default_factory=list, description="Detailed relationships")

    # Knowledge integration
    source_documents: List[str] = Field(default_factory=list, description="Source document IDs")
    supporting_evidence: List[str] = Field(default_factory=list, description="Supporting evidence IDs")
    contradicting_evidence: List[str] = Field(default_factory=list, description="Contradicting evidence IDs")

    # Activation tracking
    total_activations: int = Field(default=0, description="Total times activated")
    recent_activations: List[ConceptActivation] = Field(default_factory=list, description="Recent activation events")
    last_activated: Optional[datetime] = Field(default=None, description="Last activation timestamp")
    activation_contexts: Dict[str, int] = Field(default_factory=dict, description="Activation context frequency")

    # Consciousness integration
    consciousness_emergence_events: List[str] = Field(default_factory=list, description="Consciousness events involving concept")
    insight_contributions: List[str] = Field(default_factory=list, description="Insights this concept contributed to")
    metacognitive_reflections: List[str] = Field(default_factory=list, description="Metacognitive reflections on concept")

    # Semantic properties
    semantic_embedding: Optional[List[float]] = Field(default=None, description="Semantic embedding vector")
    semantic_similarity_cache: Dict[str, float] = Field(default_factory=dict, description="Cached similarity scores")
    conceptual_distance_map: Dict[str, float] = Field(default_factory=dict, description="Conceptual distances to other concepts")

    # ThoughtSeed integration
    thoughtseed_traces: List[str] = Field(default_factory=list, description="ThoughtSeed traces involving concept")
    active_inference_role: Optional[str] = Field(default=None, description="Role in active inference")
    prediction_error_contributions: List[float] = Field(default_factory=list, description="Prediction error contributions")

    # Quality and validation
    validation_status: str = Field(default="pending", description="Validation status")
    quality_score: float = Field(default=0.5, description="Overall quality score", ge=0.0, le=1.0)
    coherence_score: float = Field(default=0.5, description="Internal coherence score", ge=0.0, le=1.0)
    utility_score: float = Field(default=0.5, description="Practical utility score", ge=0.0, le=1.0)

    # Usage statistics
    reference_count: int = Field(default=0, description="Number of references to this concept")
    curiosity_triggers: int = Field(default=0, description="Times concept triggered curiosity")
    insight_generations: int = Field(default=0, description="Times concept led to insights")
    cross_domain_connections: int = Field(default=0, description="Cross-domain connections made")

    # Constitutional compliance
    mock_data: bool = Field(default=True, description="Constitutional requirement: mock data flag")
    evaluation_frames: List[str] = Field(default_factory=list, description="Associated evaluation frames")
    privacy_level: str = Field(default="standard", description="Privacy level for concept data")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        schema_extra = {
            "example": {
                "id": "concept_456",
                "user_id": "user_123",
                "name": "Active Inference",
                "aliases": ["predictive processing", "free energy principle"],
                "description": "A framework for understanding perception and action as minimizing prediction error",
                "concept_type": "abstract",
                "domain": "cognitive science",
                "subdomains": ["neuroscience", "philosophy of mind"],
                "status": "stable",
                "confidence": "high",
                "maturity_score": 0.85,
                "total_activations": 23,
                "consciousness_emergence_events": ["event_789"],
                "insight_contributions": ["insight_001", "insight_002"],
                "reference_count": 15,
                "curiosity_triggers": 3,
                "mock_data": True
            }
        }

class ConceptCluster(BaseModel):
    """Cluster of related concepts for thematic organization"""

    # Cluster identity
    id: str = Field(..., description="Unique cluster identifier")
    user_id: str = Field(..., description="Associated user identifier")

    # Cluster metadata
    name: str = Field(..., description="Cluster name")
    description: str = Field(..., description="Cluster description")
    theme: str = Field(..., description="Central theme")

    # Cluster composition
    core_concepts: List[str] = Field(..., description="Core concept IDs")
    peripheral_concepts: List[str] = Field(default_factory=list, description="Peripheral concept IDs")
    concept_count: int = Field(default=0, description="Total concepts in cluster")

    # Cluster characteristics
    coherence_score: float = Field(default=0.0, description="Cluster coherence", ge=0.0, le=1.0)
    density: float = Field(default=0.0, description="Connection density within cluster", ge=0.0, le=1.0)
    centrality_scores: Dict[str, float] = Field(default_factory=dict, description="Concept centrality within cluster")

    # Dynamics
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Cluster creation time")
    last_updated: Optional[datetime] = Field(default=None, description="Last cluster update")
    stability_score: float = Field(default=0.0, description="Cluster stability over time", ge=0.0, le=1.0)

    # Activity
    recent_activations: int = Field(default=0, description="Recent cluster activations")
    insight_generation_rate: float = Field(default=0.0, description="Insights generated per activation")

    # Constitutional compliance
    mock_data: bool = Field(default=True, description="Constitutional requirement: mock data flag")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ConceptInsight(BaseModel):
    """Insights derived from concept relationships and patterns"""

    # Insight identity
    id: str = Field(..., description="Unique insight identifier")
    user_id: str = Field(..., description="Associated user identifier")

    # Insight content
    title: str = Field(..., description="Insight title")
    description: str = Field(..., description="Detailed insight description")
    insight_type: str = Field(..., description="Type of insight")

    # Source concepts
    primary_concepts: List[str] = Field(..., description="Primary concepts involved")
    supporting_concepts: List[str] = Field(default_factory=list, description="Supporting concepts")
    novel_connections: List[Dict[str, str]] = Field(default_factory=list, description="Novel concept connections discovered")

    # Discovery metadata
    discovered_at: datetime = Field(default_factory=datetime.utcnow, description="Insight discovery time")
    discovery_event_id: Optional[str] = Field(default=None, description="Event that led to insight")
    consciousness_level: float = Field(..., description="Consciousness level during discovery", ge=0.0, le=1.0)

    # Validation and impact
    confidence_score: float = Field(..., description="Confidence in insight", ge=0.0, le=1.0)
    validation_attempts: List[str] = Field(default_factory=list, description="Validation attempt IDs")
    impact_score: float = Field(default=0.0, description="Impact on understanding", ge=0.0)

    # Follow-up effects
    spawned_curiosity_missions: List[str] = Field(default_factory=list, description="Curiosity missions spawned")
    influenced_concepts: List[str] = Field(default_factory=list, description="Concepts influenced by insight")

    # Constitutional compliance
    mock_data: bool = Field(default=True, description="Constitutional requirement: mock data flag")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }