"""
UserProfile Model - T013
Flux Self-Teaching Consciousness Emulator

Represents user profile information for personalized consciousness modeling.
Constitutional compliance: mock data transparency, evaluation feedback integration.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum


class LearningStyle(str, Enum):
    """User learning style preferences"""
    VISUAL = "visual"
    AUDITORY = "auditory"
    KINESTHETIC = "kinesthetic"
    READING_WRITING = "reading_writing"
    MULTIMODAL = "multimodal"


class ConsciousnessLevel(str, Enum):
    """User consciousness level tracking"""
    EMERGING = "emerging"
    DEVELOPING = "developing"
    ESTABLISHED = "established"
    ADVANCED = "advanced"
    EXPERT = "expert"


class UserProfile(BaseModel):
    """
    User profile model for Flux consciousness emulator.

    Tracks user information, preferences, and consciousness development
    with constitutional compliance for mock data transparency.
    """

    # Core Identity
    user_id: str = Field(..., description="Unique user identifier")
    username: str = Field(..., description="User display name")
    email: Optional[str] = Field(None, description="User email address")

    # Profile Information
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Profile creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last profile update")
    last_active: Optional[datetime] = Field(None, description="Last user activity timestamp")

    # Learning Preferences
    learning_style: LearningStyle = Field(default=LearningStyle.MULTIMODAL, description="Primary learning style")
    preferred_topics: List[str] = Field(default_factory=list, description="User's preferred learning topics")
    expertise_areas: List[str] = Field(default_factory=list, description="User's areas of expertise")

    # Consciousness Development
    consciousness_level: ConsciousnessLevel = Field(default=ConsciousnessLevel.EMERGING, description="Current consciousness development level")
    consciousness_metrics: Dict[str, float] = Field(default_factory=dict, description="Consciousness measurement metrics")
    thoughtseed_traces_count: int = Field(default=0, description="Number of thoughtseed traces generated")

    # Personalization Data
    interface_preferences: Dict[str, Any] = Field(default_factory=dict, description="UI/UX preferences")
    notification_settings: Dict[str, bool] = Field(default_factory=dict, description="Notification preferences")
    privacy_settings: Dict[str, bool] = Field(default_factory=dict, description="Privacy configuration")

    # Constitutional Compliance
    mock_data_enabled: bool = Field(default=True, description="Mock data mode for development")
    evaluation_feedback_enabled: bool = Field(default=True, description="Evaluation feedback collection enabled")
    data_retention_days: int = Field(default=365, description="Data retention period in days")

    # Biographical Context
    autobiographical_journey_id: Optional[str] = Field(None, description="Associated autobiographical journey")
    primary_goals: List[str] = Field(default_factory=list, description="User's learning/development goals")
    learning_history: List[Dict[str, Any]] = Field(default_factory=list, description="Historical learning activities")

    class Config:
        """Pydantic configuration"""
        use_enum_values = True
        validate_assignment = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    def update_activity_timestamp(self) -> None:
        """Update last activity timestamp"""
        self.last_active = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def update_consciousness_metrics(self, metrics: Dict[str, float]) -> None:
        """Update consciousness development metrics"""
        self.consciousness_metrics.update(metrics)
        self.updated_at = datetime.utcnow()

    def add_thoughtseed_trace(self) -> None:
        """Increment thoughtseed trace counter"""
        self.thoughtseed_traces_count += 1
        self.updated_at = datetime.utcnow()

    def set_learning_preferences(self, style: LearningStyle, topics: List[str]) -> None:
        """Update learning preferences"""
        self.learning_style = style
        self.preferred_topics = topics
        self.updated_at = datetime.utcnow()

    def advance_consciousness_level(self) -> bool:
        """
        Attempt to advance consciousness level based on metrics.
        Returns True if level was advanced.
        """
        current_levels = list(ConsciousnessLevel)
        current_index = current_levels.index(self.consciousness_level)

        # Simple advancement logic - can be enhanced with more sophisticated rules
        if current_index < len(current_levels) - 1:
            avg_metric = sum(self.consciousness_metrics.values()) / len(self.consciousness_metrics) if self.consciousness_metrics else 0

            # Advancement thresholds
            advancement_thresholds = {
                ConsciousnessLevel.EMERGING: 0.3,
                ConsciousnessLevel.DEVELOPING: 0.5,
                ConsciousnessLevel.ESTABLISHED: 0.7,
                ConsciousnessLevel.ADVANCED: 0.85
            }

            threshold = advancement_thresholds.get(self.consciousness_level, 0.9)

            if avg_metric >= threshold:
                self.consciousness_level = current_levels[current_index + 1]
                self.updated_at = datetime.utcnow()
                return True

        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return self.dict()

    @classmethod
    def create_mock_profile(cls, user_id: str, username: str) -> "UserProfile":
        """
        Create a mock user profile for development/testing.
        Constitutional compliance: clearly marked as mock data.
        """
        return cls(
            user_id=user_id,
            username=username,
            mock_data_enabled=True,
            learning_style=LearningStyle.MULTIMODAL,
            preferred_topics=["consciousness", "ai", "philosophy", "neuroscience"],
            consciousness_level=ConsciousnessLevel.DEVELOPING,
            consciousness_metrics={
                "attention": 0.4,
                "awareness": 0.3,
                "integration": 0.35,
                "meta_cognition": 0.25
            },
            interface_preferences={
                "theme": "dark",
                "visualization_style": "graph",
                "animation_enabled": True
            },
            notification_settings={
                "thoughtseed_updates": True,
                "consciousness_milestones": True,
                "daily_summaries": False
            },
            privacy_settings={
                "share_learning_data": False,
                "anonymous_analytics": True,
                "data_export_enabled": True
            }
        )


# Type aliases for convenience
UserProfileDict = Dict[str, Any]
UserProfileList = List[UserProfile]