"""
CuriosityMission Model - T019
Flux Self-Teaching Consciousness Emulator

Represents curiosity-driven exploration missions with active inference guidance,
attractor dynamics, and consciousness-aware goal formation and pursuit.
Constitutional compliance: mock data transparency, evaluation feedback integration.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from enum import Enum
import uuid


class MissionStatus(str, Enum):
    """Curiosity mission status states"""
    FORMING = "forming"           # Mission being conceived
    ACTIVE = "active"            # Mission in progress
    EXPLORING = "exploring"      # Active exploration phase
    INTEGRATING = "integrating"  # Integrating findings
    COMPLETED = "completed"      # Mission accomplished
    SUSPENDED = "suspended"      # Temporarily paused
    ABANDONED = "abandoned"      # Mission discontinued


class CuriosityType(str, Enum):
    """Types of curiosity driving the mission"""
    EPISTEMIC = "epistemic"          # Knowledge-seeking curiosity
    DIVERSIVE = "diversive"          # Novelty-seeking curiosity
    EMPATHIC = "empathic"           # Understanding others' perspectives
    AESTHETIC = "aesthetic"          # Beauty and pattern appreciation
    SOCIAL = "social"               # Social understanding curiosity
    SELF_REFLECTIVE = "self_reflective"  # Self-understanding curiosity
    CREATIVE = "creative"           # Creative exploration curiosity


class ExplorationStrategy(str, Enum):
    """Strategies for curiosity exploration"""
    RANDOM_WALK = "random_walk"        # Random exploration
    GRADIENT_ASCENT = "gradient_ascent"  # Following interest gradients
    SYSTEMATIC = "systematic"          # Methodical exploration
    INTUITIVE = "intuitive"           # Intuition-guided exploration
    COLLABORATIVE = "collaborative"    # Socially-guided exploration
    REFLECTIVE = "reflective"         # Self-reflective exploration


class CuriosityTrigger(BaseModel):
    """What triggered the curiosity mission"""
    trigger_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Trigger identifier")
    trigger_type: str = Field(..., description="Type of trigger")
    trigger_source: str = Field(..., description="Source that triggered curiosity")
    trigger_strength: float = Field(..., ge=0.0, le=1.0, description="Strength of curiosity trigger")
    trigger_timestamp: datetime = Field(default_factory=datetime.utcnow, description="When curiosity was triggered")

    # Contextual information
    context_data: Dict[str, Any] = Field(default_factory=dict, description="Context when curiosity triggered")
    related_concepts: List[str] = Field(default_factory=list, description="Concepts involved in trigger")
    emotional_context: Dict[str, float] = Field(default_factory=dict, description="Emotional state during trigger")


class ExplorationStep(BaseModel):
    """Individual step in curiosity exploration"""
    step_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Step identifier")
    step_number: int = Field(..., description="Step sequence number")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Step timestamp")

    # Step details
    action_type: str = Field(..., description="Type of exploration action")
    action_description: str = Field(..., description="Description of exploration action")
    strategy_used: ExplorationStrategy = Field(..., description="Strategy employed")

    # Results and findings
    findings: Dict[str, Any] = Field(default_factory=dict, description="What was discovered")
    surprise_level: float = Field(default=0.0, ge=0.0, le=1.0, description="Level of surprise in findings")
    satisfaction_level: float = Field(default=0.0, ge=0.0, le=1.0, description="Satisfaction with step outcome")

    # Attractor dynamics influence
    attractor_state: Dict[str, float] = Field(default_factory=dict, description="Attractor state during step")
    memory_context: str = Field(default="working", description="Active memory context")

    # Forward-looking
    next_step_hypotheses: List[str] = Field(default_factory=list, description="Hypotheses for next steps")
    curiosity_momentum: float = Field(default=0.5, ge=0.0, le=1.0, description="Curiosity momentum after step")


class CuriosityMission(BaseModel):
    """
    Curiosity mission model for consciousness-driven exploration.

    Represents a curiosity-driven exploration journey with active inference guidance,
    attractor dynamics, and integration with the thoughtseed system.
    """

    # Core Identity
    mission_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique mission identifier")
    user_id: str = Field(..., description="Associated user ID")
    journey_id: Optional[str] = Field(None, description="Associated autobiographical journey")

    # Mission Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Mission creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last mission update")
    started_at: Optional[datetime] = Field(None, description="Mission start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Mission completion timestamp")

    # Mission Definition
    mission_title: str = Field(..., description="Human-readable mission title")
    mission_description: str = Field(default="", description="Detailed mission description")
    mission_status: MissionStatus = Field(default=MissionStatus.FORMING, description="Current mission status")

    # Curiosity Characteristics
    primary_curiosity_type: CuriosityType = Field(..., description="Primary type of curiosity")
    curiosity_triggers: List[CuriosityTrigger] = Field(default_factory=list, description="What triggered this curiosity")
    curiosity_intensity: float = Field(default=0.5, ge=0.0, le=1.0, description="Current curiosity intensity")
    curiosity_evolution: List[Dict[str, float]] = Field(default_factory=list, description="How curiosity has evolved")

    # Mission Goals and Questions
    research_questions: List[str] = Field(default_factory=list, description="Questions driving the mission")
    success_criteria: List[str] = Field(default_factory=list, description="How to measure mission success")
    expected_outcomes: List[str] = Field(default_factory=list, description="Expected outcomes")

    # Exploration Process
    exploration_strategy: ExplorationStrategy = Field(default=ExplorationStrategy.INTUITIVE, description="Primary exploration strategy")
    exploration_steps: List[ExplorationStep] = Field(default_factory=list, description="Exploration step history")
    current_focus: Optional[str] = Field(None, description="Current area of focus")

    # Active Inference Integration
    prior_beliefs: Dict[str, float] = Field(default_factory=dict, description="Prior beliefs about mission domain")
    current_hypotheses: List[str] = Field(default_factory=list, description="Current working hypotheses")
    evidence_collected: List[Dict[str, Any]] = Field(default_factory=list, description="Evidence collected")
    belief_updates: List[Dict[str, Any]] = Field(default_factory=list, description="How beliefs have been updated")

    # Attractor Dynamics
    attractor_landscape: Dict[str, float] = Field(default_factory=dict, description="Current attractor landscape")
    exploration_trajectory: List[Dict[str, Any]] = Field(default_factory=list, description="Trajectory through attractor space")
    convergence_points: List[Dict[str, Any]] = Field(default_factory=list, description="Points of convergence")

    # ThoughtSeed Integration
    thoughtseed_traces: List[str] = Field(default_factory=list, description="Associated thoughtseed trace IDs")
    consciousness_insights: List[str] = Field(default_factory=list, description="Consciousness-related insights")
    meta_cognitive_reflections: List[str] = Field(default_factory=list, description="Meta-cognitive reflections")

    # Knowledge and Learning
    concepts_explored: List[str] = Field(default_factory=list, description="Concept IDs explored")
    documents_consulted: List[str] = Field(default_factory=list, description="Document artifact IDs consulted")
    new_concepts_formed: List[str] = Field(default_factory=list, description="New concepts formed during mission")

    # Social and Collaborative Aspects
    collaborators: List[str] = Field(default_factory=list, description="User IDs of collaborators")
    external_resources: List[Dict[str, Any]] = Field(default_factory=list, description="External resources used")
    sharing_permissions: Dict[str, bool] = Field(default_factory=dict, description="Sharing and privacy settings")

    # Mission Outcomes
    discoveries: List[Dict[str, Any]] = Field(default_factory=list, description="Key discoveries made")
    insights_gained: List[str] = Field(default_factory=list, description="Insights gained")
    satisfaction_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Overall mission satisfaction")
    learning_impact: float = Field(default=0.0, ge=0.0, le=1.0, description="Impact on learning and development")

    # Constitutional Compliance
    mock_data_enabled: bool = Field(default=True, description="Mock data mode for development")
    evaluation_feedback_enabled: bool = Field(default=True, description="Evaluation feedback collection enabled")
    privacy_level: str = Field(default="private", description="Mission privacy level")

    # Mission Relationships
    parent_mission_id: Optional[str] = Field(None, description="Parent mission ID")
    child_mission_ids: List[str] = Field(default_factory=list, description="Child mission IDs spawned")
    related_mission_ids: List[str] = Field(default_factory=list, description="Related mission IDs")

    class Config:
        """Pydantic configuration"""
        use_enum_values = True
        validate_assignment = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    def add_curiosity_trigger(self, trigger_type: str, trigger_source: str,
                            trigger_strength: float, context_data: Dict[str, Any] = None) -> str:
        """Add curiosity trigger that initiated or influences mission"""
        trigger = CuriosityTrigger(
            trigger_type=trigger_type,
            trigger_source=trigger_source,
            trigger_strength=trigger_strength,
            context_data=context_data or {}
        )

        self.curiosity_triggers.append(trigger)
        self.updated_at = datetime.utcnow()

        return trigger.trigger_id

    def start_mission(self) -> None:
        """Start the curiosity mission"""
        self.mission_status = MissionStatus.ACTIVE
        self.started_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def add_exploration_step(self, action_type: str, action_description: str,
                           strategy: ExplorationStrategy, findings: Dict[str, Any] = None,
                           surprise_level: float = 0.0) -> str:
        """Add exploration step to mission"""
        step = ExplorationStep(
            step_number=len(self.exploration_steps) + 1,
            action_type=action_type,
            action_description=action_description,
            strategy_used=strategy,
            findings=findings or {},
            surprise_level=surprise_level,
            attractor_state=dict(self.attractor_landscape),  # Snapshot current state
            memory_context="working"  # Default to working memory
        )

        self.exploration_steps.append(step)

        # Update mission status if actively exploring
        if self.mission_status == MissionStatus.ACTIVE:
            self.mission_status = MissionStatus.EXPLORING

        self.updated_at = datetime.utcnow()
        return step.step_id

    def update_beliefs(self, evidence: Dict[str, Any], belief_changes: Dict[str, float]) -> None:
        """Update beliefs based on new evidence (active inference)"""
        # Record evidence
        evidence_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "evidence": evidence,
            "source": evidence.get("source", "unknown")
        }
        self.evidence_collected.append(evidence_record)

        # Record belief updates
        belief_update = {
            "timestamp": datetime.utcnow().isoformat(),
            "previous_beliefs": dict(self.prior_beliefs),
            "changes": belief_changes,
            "evidence_id": len(self.evidence_collected) - 1
        }
        self.belief_updates.append(belief_update)

        # Update current beliefs
        for belief, change in belief_changes.items():
            current_value = self.prior_beliefs.get(belief, 0.5)
            self.prior_beliefs[belief] = max(0.0, min(1.0, current_value + change))

        self.updated_at = datetime.utcnow()

    def update_attractor_landscape(self, new_attractors: Dict[str, float]) -> None:
        """Update attractor landscape based on exploration"""
        # Record trajectory point
        trajectory_point = {
            "timestamp": datetime.utcnow().isoformat(),
            "previous_landscape": dict(self.attractor_landscape),
            "new_landscape": new_attractors,
            "step_number": len(self.exploration_steps)
        }
        self.exploration_trajectory.append(trajectory_point)

        # Update current landscape
        self.attractor_landscape.update(new_attractors)
        self.updated_at = datetime.utcnow()

    def associate_thoughtseed(self, trace_id: str) -> None:
        """Associate mission with thoughtseed trace"""
        if trace_id not in self.thoughtseed_traces:
            self.thoughtseed_traces.append(trace_id)
            self.updated_at = datetime.utcnow()

    def add_discovery(self, discovery_type: str, description: str,
                     significance: float, evidence: Dict[str, Any] = None) -> None:
        """Record significant discovery made during mission"""
        discovery = {
            "discovery_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "discovery_type": discovery_type,
            "description": description,
            "significance": significance,
            "evidence": evidence or {},
            "step_number": len(self.exploration_steps)
        }

        self.discoveries.append(discovery)
        self.updated_at = datetime.utcnow()

    def spawn_child_mission(self, child_title: str, child_curiosity_type: CuriosityType,
                          child_description: str = "") -> str:
        """Spawn child mission from current exploration"""
        child_mission_id = str(uuid.uuid4())
        self.child_mission_ids.append(child_mission_id)
        self.updated_at = datetime.utcnow()

        return child_mission_id

    def complete_mission(self, satisfaction_score: float, learning_impact: float,
                        final_insights: List[str] = None) -> None:
        """Complete the curiosity mission"""
        self.mission_status = MissionStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.satisfaction_score = satisfaction_score
        self.learning_impact = learning_impact

        if final_insights:
            self.insights_gained.extend(final_insights)

        self.updated_at = datetime.utcnow()

    def get_mission_summary(self) -> Dict[str, Any]:
        """Get comprehensive mission summary"""
        duration_hours = None
        if self.started_at and self.completed_at:
            duration_hours = (self.completed_at - self.started_at).total_seconds() / 3600
        elif self.started_at:
            duration_hours = (datetime.utcnow() - self.started_at).total_seconds() / 3600

        return {
            "mission_id": self.mission_id,
            "title": self.mission_title,
            "status": self.mission_status.value,
            "curiosity_type": self.primary_curiosity_type.value,
            "curiosity_intensity": self.curiosity_intensity,
            "exploration_steps": len(self.exploration_steps),
            "discoveries": len(self.discoveries),
            "concepts_explored": len(self.concepts_explored),
            "thoughtseed_associations": len(self.thoughtseed_traces),
            "duration_hours": duration_hours,
            "satisfaction_score": self.satisfaction_score,
            "learning_impact": self.learning_impact,
            "child_missions": len(self.child_mission_ids)
        }

    def calculate_exploration_momentum(self) -> float:
        """Calculate current exploration momentum"""
        if not self.exploration_steps:
            return self.curiosity_intensity

        # Recent steps have more influence
        recent_steps = self.exploration_steps[-5:]  # Last 5 steps
        momentum_factors = []

        for step in recent_steps:
            # High surprise and satisfaction increase momentum
            step_momentum = (step.surprise_level + step.satisfaction_level) / 2.0
            momentum_factors.append(step_momentum)

        if momentum_factors:
            avg_momentum = sum(momentum_factors) / len(momentum_factors)
            # Combine with current curiosity intensity
            return (avg_momentum + self.curiosity_intensity) / 2.0
        else:
            return self.curiosity_intensity

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return self.dict()

    @classmethod
    def create_mock_mission(cls, user_id: str, mission_title: str,
                          curiosity_type: CuriosityType, journey_id: str = None) -> "CuriosityMission":
        """
        Create mock curiosity mission for development/testing.
        Constitutional compliance: clearly marked as mock data.
        """
        mission = cls(
            user_id=user_id,
            journey_id=journey_id,
            mission_title=mission_title,
            mission_description=f"Mock {curiosity_type.value} curiosity mission: {mission_title}",
            primary_curiosity_type=curiosity_type,
            mock_data_enabled=True,
            curiosity_intensity=0.7,
            exploration_strategy=ExplorationStrategy.INTUITIVE,
            research_questions=[
                "What patterns emerge in consciousness development?",
                "How do attractor dynamics influence learning?"
            ],
            success_criteria=[
                "Identify key consciousness patterns",
                "Understand attractor influence mechanisms"
            ],
            prior_beliefs={
                "consciousness_is_emergent": 0.8,
                "attractors_guide_learning": 0.6,
                "curiosity_drives_growth": 0.9
            },
            current_hypotheses=[
                "Consciousness emerges from complex attractor dynamics",
                "Curiosity creates optimal learning conditions"
            ]
        )

        # Add mock curiosity trigger
        mission.add_curiosity_trigger(
            trigger_type="concept_gap",
            trigger_source="consciousness_research",
            trigger_strength=0.8,
            context_data={"gap_type": "understanding", "domain": "consciousness"}
        )

        # Add mock exploration step
        mission.start_mission()
        mission.add_exploration_step(
            action_type="research_review",
            action_description="Reviewing consciousness research literature",
            strategy=ExplorationStrategy.SYSTEMATIC,
            findings={"key_themes": ["emergence", "complexity", "integration"]},
            surprise_level=0.6
        )

        return mission


# Type aliases for convenience
CuriosityMissionDict = Dict[str, Any]
CuriosityMissionList = List[CuriosityMission]