"""
EvaluationFrame Model - T020
Flux Self-Evolving Consciousness Emulator

Represents evaluation frameworks and feedback collection for consciousness development
with constitutional compliance and user consent management.
Constitutional compliance: mock data transparency, evaluation feedback integration.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from enum import Enum
import uuid


class EvaluationType(str, Enum):
    """Types of evaluations"""
    CONSCIOUSNESS_ASSESSMENT = "consciousness_assessment"
    LEARNING_PROGRESS = "learning_progress"
    THOUGHTSEED_EFFECTIVENESS = "thoughtseed_effectiveness"
    USER_SATISFACTION = "user_satisfaction"
    SYSTEM_PERFORMANCE = "system_performance"
    ETHICAL_COMPLIANCE = "ethical_compliance"


class EvaluationStatus(str, Enum):
    """Evaluation status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    REVIEWED = "reviewed"
    ARCHIVED = "archived"


class ConsentLevel(str, Enum):
    """User consent levels"""
    FULL_CONSENT = "full_consent"
    LIMITED_CONSENT = "limited_consent"
    NO_CONSENT = "no_consent"
    WITHDRAWN = "withdrawn"


class EvaluationMetric(BaseModel):
    """Individual evaluation metric"""
    metric_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Metric identifier")
    metric_name: str = Field(..., description="Metric name")
    metric_value: float = Field(..., description="Metric value")
    metric_weight: float = Field(default=1.0, description="Metric weight in overall score")
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence in metric")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Metric timestamp")


class EvaluationFrame(BaseModel):
    """
    Evaluation frame model for consciousness system assessment.

    Handles evaluation data collection with strict constitutional compliance,
    user consent management, and transparent feedback mechanisms.
    """

    # Core Identity
    frame_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique frame identifier")
    user_id: str = Field(..., description="Associated user ID")
    journey_id: Optional[str] = Field(None, description="Associated journey ID")

    # Frame Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Frame creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last frame update")
    evaluation_date: datetime = Field(default_factory=datetime.utcnow, description="Evaluation date")

    # Evaluation Definition
    evaluation_type: EvaluationType = Field(..., description="Type of evaluation")
    evaluation_title: str = Field(..., description="Evaluation title")
    evaluation_description: str = Field(default="", description="Detailed description")
    evaluation_status: EvaluationStatus = Field(default=EvaluationStatus.PENDING, description="Current status")

    # Constitutional Compliance - REQUIRED
    mock_data_enabled: bool = Field(default=True, description="Constitutional requirement: mock data flag")
    user_consent_level: ConsentLevel = Field(..., description="User's consent level for evaluation")
    consent_timestamp: datetime = Field(default_factory=datetime.utcnow, description="When consent was obtained")
    consent_details: Dict[str, Any] = Field(default_factory=dict, description="Detailed consent information")

    # Evaluation Data
    metrics: List[EvaluationMetric] = Field(default_factory=list, description="Evaluation metrics")
    overall_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Overall evaluation score")
    confidence_interval: Dict[str, float] = Field(default_factory=dict, description="Confidence intervals")

    # Feedback and Results
    findings: List[str] = Field(default_factory=list, description="Key findings from evaluation")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations based on evaluation")
    feedback_provided: bool = Field(default=False, description="Whether feedback was provided to user")

    # Privacy and Access Control
    privacy_level: str = Field(default="private", description="Evaluation privacy level")
    access_permissions: Dict[str, bool] = Field(default_factory=dict, description="Access control settings")
    data_retention_days: int = Field(default=365, description="Data retention period")

    # Associated Data References
    thoughtseed_references: List[str] = Field(default_factory=list, description="Related thoughtseed traces")
    event_references: List[str] = Field(default_factory=list, description="Related events")
    document_references: List[str] = Field(default_factory=list, description="Related documents")

    class Config:
        """Pydantic configuration"""
        use_enum_values = True
        validate_assignment = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    def add_metric(self, metric_name: str, metric_value: float,
                   metric_weight: float = 1.0, confidence: float = 1.0) -> str:
        """Add evaluation metric"""
        metric = EvaluationMetric(
            metric_name=metric_name,
            metric_value=metric_value,
            metric_weight=metric_weight,
            confidence_score=confidence
        )

        self.metrics.append(metric)
        self._calculate_overall_score()
        self.updated_at = datetime.utcnow()

        return metric.metric_id

    def _calculate_overall_score(self) -> float:
        """Calculate weighted overall score"""
        if not self.metrics:
            self.overall_score = 0.0
            return 0.0

        weighted_sum = sum(m.metric_value * m.metric_weight for m in self.metrics)
        total_weight = sum(m.metric_weight for m in self.metrics)

        self.overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        return self.overall_score

    def complete_evaluation(self, findings: List[str], recommendations: List[str]) -> None:
        """Complete evaluation with findings and recommendations"""
        self.evaluation_status = EvaluationStatus.COMPLETED
        self.findings = findings
        self.recommendations = recommendations
        self.updated_at = datetime.utcnow()

    @classmethod
    def create_mock_evaluation(cls, user_id: str, evaluation_type: EvaluationType) -> "EvaluationFrame":
        """Create mock evaluation frame for development"""
        return cls(
            user_id=user_id,
            evaluation_type=evaluation_type,
            evaluation_title=f"Mock {evaluation_type.value} evaluation",
            mock_data_enabled=True,
            user_consent_level=ConsentLevel.FULL_CONSENT,
            consent_details={"mock_consent": True}
        )


class VisualizationState(BaseModel):
    """
    VisualizationState Model - T021
    Represents UI/visualization state for consciousness system.
    """

    # Core Identity
    state_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique state identifier")
    user_id: str = Field(..., description="Associated user ID")
    session_id: Optional[str] = Field(None, description="UI session ID")

    # State Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow, description="State creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last state update")

    # Visualization Settings
    active_view: str = Field(default="dashboard", description="Currently active view")
    layout_config: Dict[str, Any] = Field(default_factory=dict, description="UI layout configuration")
    theme_settings: Dict[str, str] = Field(default_factory=dict, description="Theme and appearance settings")

    # Display Preferences
    consciousness_metrics_visible: bool = Field(default=True, description="Show consciousness metrics")
    thoughtseed_traces_visible: bool = Field(default=True, description="Show thoughtseed traces")
    concept_graph_visible: bool = Field(default=True, description="Show concept relationships")

    # Constitutional Compliance
    mock_data_enabled: bool = Field(default=True, description="Mock data mode")

    class Config:
        use_enum_values = True
        json_encoders = {datetime: lambda v: v.isoformat()}

    @classmethod
    def create_mock_state(cls, user_id: str) -> "VisualizationState":
        """Create mock visualization state"""
        return cls(
            user_id=user_id,
            mock_data_enabled=True,
            theme_settings={"theme": "dark", "accent_color": "blue"}
        )


# Type aliases
EvaluationFrameDict = Dict[str, Any]
VisualizationStateDict = Dict[str, Any]