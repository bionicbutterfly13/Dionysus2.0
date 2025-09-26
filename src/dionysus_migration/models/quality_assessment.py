"""
Quality Assessment model

Evaluation metrics and scoring system for determining component value,
code quality, and migration priority with focus on consciousness
functionality impact and strategic value.
"""

from datetime import datetime
from typing import Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class ConsciousnessImpactAnalysis(BaseModel):
    """Detailed consciousness impact analysis"""
    awareness_processing_patterns: List[str] = Field(
        default_factory=list,
        description="Detected awareness processing patterns"
    )
    inference_capabilities: List[str] = Field(
        default_factory=list,
        description="Identified inference capabilities"
    )
    memory_integration_methods: List[str] = Field(
        default_factory=list,
        description="Memory integration approaches used"
    )
    consciousness_flow_complexity: str = Field(
        description="Complexity of consciousness data flow (low/medium/high)"
    )
    meta_cognitive_features: List[str] = Field(
        default_factory=list,
        description="Meta-cognitive features present"
    )
    consciousness_state_management: bool = Field(
        default=False,
        description="Whether component manages consciousness state"
    )


class StrategicValueAnalysis(BaseModel):
    """Detailed strategic value analysis"""
    architectural_uniqueness: float = Field(
        ge=0.0, le=1.0,
        description="Uniqueness of architectural approach"
    )
    code_reusability_factors: List[str] = Field(
        default_factory=list,
        description="Factors contributing to reusability"
    )
    framework_integration_points: List[str] = Field(
        default_factory=list,
        description="Potential integration points with Dionysus 2.0"
    )
    legacy_dependency_burden: float = Field(
        ge=0.0, le=1.0,
        description="Burden of legacy dependencies (higher = more burden)"
    )
    migration_effort_estimate: str = Field(
        description="Estimated migration effort (low/medium/high)"
    )
    business_value_impact: float = Field(
        ge=0.0, le=1.0,
        description="Impact on business value"
    )


class QualityAssessment(BaseModel):
    """
    Quality assessment model for legacy components focusing on
    consciousness functionality impact and strategic value
    """

    assessment_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique assessment identifier"
    )
    component_id: str = Field(
        description="ID of the legacy component being assessed"
    )
    consciousness_impact: ConsciousnessImpactAnalysis = Field(
        description="Detailed consciousness functionality analysis"
    )
    strategic_value: StrategicValueAnalysis = Field(
        description="Detailed strategic positioning analysis"
    )
    composite_score: float = Field(
        ge=0.0, le=1.0,
        description="Final priority score for migration"
    )
    assessment_method: str = Field(
        description="Algorithm version used for assessment"
    )
    assessed_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of assessment completion"
    )
    assessor_agent_id: str = Field(
        description="DAEDALUS subagent identifier that performed assessment"
    )

    # Assessment metadata
    analysis_duration_seconds: Optional[float] = Field(
        default=None,
        description="Time taken to complete analysis"
    )
    confidence_level: float = Field(
        ge=0.0, le=1.0,
        default=0.8,
        description="Confidence level in assessment accuracy"
    )
    assessment_version: str = Field(
        default="1.0.0",
        description="Version of assessment algorithm"
    )

    # Detailed scoring breakdown
    consciousness_score_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Detailed breakdown of consciousness scoring"
    )
    strategic_score_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Detailed breakdown of strategic value scoring"
    )

    # Assessment flags and recommendations
    migration_recommended: bool = Field(
        description="Whether migration is recommended"
    )
    enhancement_opportunities: List[str] = Field(
        default_factory=list,
        description="Identified enhancement opportunities with ThoughtSeed"
    )
    risk_factors: List[str] = Field(
        default_factory=list,
        description="Identified risk factors for migration"
    )
    prerequisites: List[str] = Field(
        default_factory=list,
        description="Prerequisites for successful migration"
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    @classmethod
    def calculate_composite_score(
        cls,
        consciousness_analysis: ConsciousnessImpactAnalysis,
        strategic_analysis: StrategicValueAnalysis,
        consciousness_weight: float = 0.7,
        strategic_weight: float = 0.3
    ) -> float:
        """
        Calculate composite score from consciousness and strategic analyses

        Args:
            consciousness_analysis: Consciousness impact analysis
            strategic_analysis: Strategic value analysis
            consciousness_weight: Weight for consciousness score
            strategic_weight: Weight for strategic score

        Returns:
            Composite score between 0.0 and 1.0
        """
        # Calculate consciousness composite score
        consciousness_factors = [
            len(consciousness_analysis.awareness_processing_patterns) / 10.0,
            len(consciousness_analysis.inference_capabilities) / 10.0,
            len(consciousness_analysis.memory_integration_methods) / 5.0,
            _complexity_to_score(consciousness_analysis.consciousness_flow_complexity),
            len(consciousness_analysis.meta_cognitive_features) / 5.0,
            1.0 if consciousness_analysis.consciousness_state_management else 0.0
        ]
        consciousness_score = min(1.0, sum(consciousness_factors) / len(consciousness_factors))

        # Calculate strategic composite score
        strategic_factors = [
            strategic_analysis.architectural_uniqueness,
            len(strategic_analysis.code_reusability_factors) / 8.0,
            len(strategic_analysis.framework_integration_points) / 6.0,
            1.0 - strategic_analysis.legacy_dependency_burden,  # Invert burden
            _effort_to_score(strategic_analysis.migration_effort_estimate),
            strategic_analysis.business_value_impact
        ]
        strategic_score = min(1.0, sum(strategic_factors) / len(strategic_factors))

        # Weighted composite
        return consciousness_score * consciousness_weight + strategic_score * strategic_weight

    def add_enhancement_opportunity(self, opportunity: str) -> None:
        """Add identified enhancement opportunity"""
        if opportunity not in self.enhancement_opportunities:
            self.enhancement_opportunities.append(opportunity)

    def add_risk_factor(self, risk: str) -> None:
        """Add identified risk factor"""
        if risk not in self.risk_factors:
            self.risk_factors.append(risk)

    def add_prerequisite(self, prerequisite: str) -> None:
        """Add migration prerequisite"""
        if prerequisite not in self.prerequisites:
            self.prerequisites.append(prerequisite)

    def is_high_priority(self, threshold: float = 0.8) -> bool:
        """Check if component is high priority for migration"""
        return self.composite_score >= threshold and self.migration_recommended

    def get_migration_recommendation_summary(self) -> Dict[str, any]:
        """Get summary of migration recommendation"""
        return {
            "recommended": self.migration_recommended,
            "score": self.composite_score,
            "confidence": self.confidence_level,
            "opportunities": len(self.enhancement_opportunities),
            "risks": len(self.risk_factors),
            "prerequisites": len(self.prerequisites)
        }

    def to_dict(self) -> Dict:
        """Convert to dictionary representation"""
        return self.dict()

    def __str__(self) -> str:
        status = "RECOMMENDED" if self.migration_recommended else "NOT RECOMMENDED"
        return f"QualityAssessment(component={self.component_id[:8]}..., score={self.composite_score:.3f}, {status})"


def _complexity_to_score(complexity: str) -> float:
    """Convert complexity string to score"""
    complexity_map = {
        "low": 0.3,
        "medium": 0.6,
        "high": 1.0
    }
    return complexity_map.get(complexity.lower(), 0.0)


def _effort_to_score(effort: str) -> float:
    """Convert effort estimate to score (lower effort = higher score)"""
    effort_map = {
        "low": 1.0,
        "medium": 0.6,
        "high": 0.3
    }
    return effort_map.get(effort.lower(), 0.0)