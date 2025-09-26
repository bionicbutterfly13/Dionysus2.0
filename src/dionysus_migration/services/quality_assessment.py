"""
Quality Assessment Service

Evaluates components using consciousness functionality impact
and strategic value metrics for migration prioritization.
"""

from datetime import datetime
from typing import List, Optional

from ..config import get_migration_config
from ..logging_config import get_consciousness_logger
from ..models.legacy_component import LegacyComponent
from ..models.quality_assessment import (
    ConsciousnessImpactAnalysis,
    QualityAssessment,
    StrategicValueAnalysis
)


class QualityAssessmentService:
    """Service for assessing component quality and migration priority"""

    def __init__(self):
        self.config = get_migration_config()
        self.logger = get_consciousness_logger()

    def assess_component(
        self,
        component: LegacyComponent,
        assessor_agent_id: str
    ) -> QualityAssessment:
        """
        Perform comprehensive quality assessment of a component

        Args:
            component: Legacy component to assess
            assessor_agent_id: ID of the DAEDALUS agent performing assessment

        Returns:
            Complete quality assessment
        """
        self.logger.info(
            "Starting quality assessment",
            component_id=component.component_id,
            assessor_agent_id=assessor_agent_id
        )

        # Perform consciousness impact analysis
        consciousness_impact = self._analyze_consciousness_impact(component)

        # Perform strategic value analysis
        strategic_value = self._analyze_strategic_value(component)

        # Calculate composite score
        composite_score = QualityAssessment.calculate_composite_score(
            consciousness_impact,
            strategic_value,
            consciousness_weight=self.config.consciousness_weight,
            strategic_weight=self.config.strategic_weight
        )

        # Determine migration recommendation
        migration_recommended = composite_score >= self.config.quality_threshold

        # Create assessment
        assessment = QualityAssessment(
            component_id=component.component_id,
            consciousness_impact=consciousness_impact,
            strategic_value=strategic_value,
            composite_score=composite_score,
            assessment_method="consciousness_strategic_v1.0",
            assessor_agent_id=assessor_agent_id,
            migration_recommended=migration_recommended
        )

        # Add enhancement opportunities and risk factors
        self._identify_enhancement_opportunities(assessment, component)
        self._identify_risk_factors(assessment, component)

        self.logger.log_component_analysis(
            component_id=component.component_id,
            awareness_score=component.consciousness_functionality.awareness_score,
            inference_score=component.consciousness_functionality.inference_score,
            memory_score=component.consciousness_functionality.memory_score,
            quality_score=composite_score
        )

        return assessment

    def _analyze_consciousness_impact(
        self,
        component: LegacyComponent
    ) -> ConsciousnessImpactAnalysis:
        """Analyze consciousness functionality impact"""
        consciousness_func = component.consciousness_functionality

        return ConsciousnessImpactAnalysis(
            awareness_processing_patterns=self._extract_awareness_patterns(component),
            inference_capabilities=self._extract_inference_capabilities(component),
            memory_integration_methods=self._extract_memory_methods(component),
            consciousness_flow_complexity=self._assess_complexity(component),
            meta_cognitive_features=self._extract_metacognitive_features(component),
            consciousness_state_management=self._has_state_management(component)
        )

    def _analyze_strategic_value(
        self,
        component: LegacyComponent
    ) -> StrategicValueAnalysis:
        """Analyze strategic value for migration"""
        strategic = component.strategic_value

        return StrategicValueAnalysis(
            architectural_uniqueness=strategic.uniqueness_score,
            code_reusability_factors=self._extract_reusability_factors(component),
            framework_integration_points=self._find_integration_points(component),
            legacy_dependency_burden=self._assess_dependency_burden(component),
            migration_effort_estimate=self._estimate_migration_effort(component),
            business_value_impact=strategic.framework_alignment_score
        )

    def _extract_awareness_patterns(self, component: LegacyComponent) -> List[str]:
        """Extract awareness processing patterns"""
        patterns = []
        for pattern in component.consciousness_patterns:
            if 'awareness' in pattern or 'conscious' in pattern:
                patterns.append(pattern)
        return patterns

    def _extract_inference_capabilities(self, component: LegacyComponent) -> List[str]:
        """Extract inference capabilities"""
        capabilities = []
        for pattern in component.consciousness_patterns:
            if 'inference' in pattern or 'reason' in pattern:
                capabilities.append(pattern)
        return capabilities

    def _extract_memory_methods(self, component: LegacyComponent) -> List[str]:
        """Extract memory integration methods"""
        methods = []
        for pattern in component.consciousness_patterns:
            if 'memory' in pattern:
                methods.append(pattern)
        return methods

    def _assess_complexity(self, component: LegacyComponent) -> str:
        """Assess consciousness flow complexity"""
        pattern_count = len(component.consciousness_patterns)
        if pattern_count >= 5:
            return "high"
        elif pattern_count >= 3:
            return "medium"
        else:
            return "low"

    def _extract_metacognitive_features(self, component: LegacyComponent) -> List[str]:
        """Extract meta-cognitive features"""
        features = []
        for pattern in component.consciousness_patterns:
            if 'meta' in pattern or 'self' in pattern:
                features.append(pattern)
        return features

    def _has_state_management(self, component: LegacyComponent) -> bool:
        """Check if component manages consciousness state"""
        return 'state_management' in component.consciousness_patterns

    def _extract_reusability_factors(self, component: LegacyComponent) -> List[str]:
        """Extract factors contributing to reusability"""
        factors = []
        if component.strategic_value.reusability_score > 0.7:
            factors.extend(['high_modularity', 'clean_interfaces', 'good_documentation'])
        elif component.strategic_value.reusability_score > 0.4:
            factors.extend(['moderate_modularity', 'basic_interfaces'])
        return factors

    def _find_integration_points(self, component: LegacyComponent) -> List[str]:
        """Find potential integration points with Dionysus 2.0"""
        integration_points = []
        if component.strategic_value.framework_alignment_score > 0.6:
            integration_points.extend([
                'async_processing',
                'pydantic_models',
                'fastapi_endpoints',
                'structured_logging'
            ])
        return integration_points

    def _assess_dependency_burden(self, component: LegacyComponent) -> float:
        """Assess legacy dependency burden"""
        dependency_count = len(component.dependencies)
        # More dependencies = higher burden
        return min(1.0, dependency_count / 10.0)

    def _estimate_migration_effort(self, component: LegacyComponent) -> str:
        """Estimate migration effort level"""
        if component.file_size_bytes and component.file_size_bytes > 5000:
            return "high"
        elif component.file_size_bytes and component.file_size_bytes > 2000:
            return "medium"
        else:
            return "low"

    def _identify_enhancement_opportunities(
        self,
        assessment: QualityAssessment,
        component: LegacyComponent
    ) -> None:
        """Identify ThoughtSeed enhancement opportunities"""
        if component.consciousness_functionality.awareness_score > 0.6:
            assessment.add_enhancement_opportunity("awareness_amplification")

        if component.consciousness_functionality.inference_score > 0.6:
            assessment.add_enhancement_opportunity("active_inference_integration")

        if component.consciousness_functionality.memory_score > 0.6:
            assessment.add_enhancement_opportunity("memory_system_enhancement")

        if 'meta' in str(component.consciousness_patterns):
            assessment.add_enhancement_opportunity("meta_cognitive_expansion")

    def _identify_risk_factors(
        self,
        assessment: QualityAssessment,
        component: LegacyComponent
    ) -> None:
        """Identify migration risk factors"""
        if len(component.dependencies) > 5:
            assessment.add_risk_factor("high_dependency_complexity")

        if component.strategic_value.framework_alignment_score < 0.3:
            assessment.add_risk_factor("low_framework_compatibility")

        if not component.consciousness_patterns:
            assessment.add_risk_factor("unclear_consciousness_benefits")