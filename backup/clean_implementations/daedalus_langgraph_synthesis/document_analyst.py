"""
Document Analyst - Evaluates processing results and extracts insights
Adapted from ASI-GO-2 analyst.py with consciousness integration
"""
from typing import Dict, List, Any
import logging
from datetime import datetime
from .document_cognition_base import DocumentCognitionBase

logger = logging.getLogger("dionysus.document_analyst")


class DocumentAnalyst:
    """Analyzes document processing results and extracts actionable insights"""

    def __init__(self, cognition_base: DocumentCognitionBase):
        self.cognition_base = cognition_base
        self.analysis_history = []

    def analyze_processing_result(self,
                                 processing_result: Dict[str, Any],
                                 research_plan: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze document processing results and extract insights

        Args:
            processing_result: Output from ConsciousnessDocumentProcessor
            research_plan: Output from DocumentResearcher (if available)

        Returns:
            Analysis with quality scores, insights, and recommendations
        """
        logger.info("Analyzing document processing result")

        # Quality assessment
        quality_scores = self._assess_quality(processing_result)

        # Extract insights
        insights = self._extract_insights(processing_result, research_plan)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            processing_result, quality_scores, insights
        )

        # Meta-cognitive analysis (consciousness system tracking)
        meta_cognitive = self._meta_cognitive_analysis(
            processing_result, research_plan
        )

        analysis = {
            "timestamp": datetime.now().isoformat(),
            "quality_scores": quality_scores,
            "insights": insights,
            "recommendations": recommendations,
            "meta_cognitive": meta_cognitive,
            "iteration": len(self.analysis_history) + 1
        }

        self.analysis_history.append(analysis)

        # Add significant insights to cognition base
        self._store_significant_insights(insights)

        logger.info(f"Analysis complete - Quality score: {quality_scores.get('overall', 0):.2f}")
        return analysis

    def _assess_quality(self, result: Dict[str, Any]) -> Dict[str, float]:
        """Assess quality of document processing"""
        scores = {}

        # Concept extraction quality
        concepts = result.get("concepts", [])
        scores["concept_extraction"] = min(len(concepts) / 50.0, 1.0)  # Expect ~50 concepts

        # Chunking quality
        chunks = result.get("chunks", [])
        avg_chunk_size = sum(len(c) for c in chunks) / len(chunks) if chunks else 0
        ideal_chunk_size = 1000
        scores["chunking"] = 1.0 - abs(avg_chunk_size - ideal_chunk_size) / ideal_chunk_size

        # Consciousness integration quality
        basins_created = result.get("basins_created", 0)
        thoughtseeds = result.get("thoughtseeds_generated", 0)
        scores["consciousness_integration"] = min(
            (basins_created + thoughtseeds) / (len(concepts) * 2), 1.0
        ) if concepts else 0.0

        # Content hash (deduplication working)
        scores["deduplication"] = 1.0 if result.get("content_hash") else 0.0

        # Summary quality (simple heuristic)
        summary = result.get("summary", {})
        summary_text = summary.get("summary", "")
        scores["summary_quality"] = min(len(summary_text) / 500.0, 1.0)  # Expect ~500 chars

        # Overall score (weighted average)
        weights = {
            "concept_extraction": 0.25,
            "chunking": 0.15,
            "consciousness_integration": 0.35,
            "deduplication": 0.10,
            "summary_quality": 0.15
        }

        scores["overall"] = sum(scores[k] * weights[k] for k in weights.keys())

        return scores

    def _extract_insights(self,
                         result: Dict[str, Any],
                         research_plan: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract actionable insights from processing results"""
        insights = []

        concepts = result.get("concepts", [])
        basins = result.get("basins_created", 0)

        # Insight 1: Concept density
        if len(concepts) > 100:
            insights.append({
                "type": "concept_density",
                "description": f"High concept density ({len(concepts)} concepts) suggests complex document",
                "significance": 0.8,
                "action": "Consider multi-pass processing for deeper understanding",
                "topic": "document_complexity"
            })

        # Insight 2: Basin creation efficiency
        if basins > 0:
            basin_efficiency = basins / len(concepts) if concepts else 0
            if basin_efficiency > 0.8:
                insights.append({
                    "type": "consciousness_efficiency",
                    "description": f"High basin creation rate ({basin_efficiency:.2f}) indicates strong pattern recognition",
                    "significance": 0.9,
                    "action": "System is learning effectively, continue current approach",
                    "topic": "pattern_learning"
                })
            elif basin_efficiency < 0.3:
                insights.append({
                    "type": "consciousness_inefficiency",
                    "description": f"Low basin creation rate ({basin_efficiency:.2f}) suggests weak pattern detection",
                    "significance": 0.7,
                    "action": "Review concept extraction parameters or document quality",
                    "topic": "pattern_learning"
                })

        # Insight 3: Curiosity triggers
        if research_plan:
            curiosity_triggers = research_plan.get("curiosity_triggers", [])
            if len(curiosity_triggers) > 5:
                high_priority = sum(1 for t in curiosity_triggers if t.get("priority") == "high")
                insights.append({
                    "type": "curiosity_analysis",
                    "description": f"{len(curiosity_triggers)} curiosity triggers found ({high_priority} high priority)",
                    "significance": 0.85,
                    "action": "Initiate web crawling for high-priority concepts",
                    "topic": "knowledge_gaps"
                })

        # Insight 4: Document type analysis
        filename = result.get("filename", "")
        if "research" in filename.lower() or "paper" in filename.lower():
            insights.append({
                "type": "document_classification",
                "description": "Academic/research document detected",
                "significance": 0.7,
                "action": "Apply academic extraction strategies (citations, methodology, results)",
                "topic": "document_type"
            })

        return insights

    def _generate_recommendations(self,
                                 result: Dict[str, Any],
                                 quality_scores: Dict[str, float],
                                 insights: List[Dict[str, Any]]) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []

        overall_quality = quality_scores.get("overall", 0)

        # Quality-based recommendations
        if overall_quality < 0.5:
            recommendations.append("CRITICAL: Low overall quality - review extraction parameters")
        elif overall_quality < 0.7:
            recommendations.append("MODERATE: Consider adjusting chunking or concept extraction")

        # Component-specific recommendations
        if quality_scores.get("concept_extraction", 0) < 0.5:
            recommendations.append("Improve concept extraction: try alternative NLP strategies")

        if quality_scores.get("consciousness_integration", 0) < 0.5:
            recommendations.append("Low consciousness integration: check basin creation logic")

        if quality_scores.get("chunking", 0) < 0.6:
            recommendations.append("Adjust chunk size parameters for better semantic coherence")

        # Insight-based recommendations
        for insight in insights:
            action = insight.get("action")
            if action and insight.get("significance", 0) > 0.7:
                recommendations.append(f"HIGH PRIORITY: {action}")

        # Strategy recommendations from cognition base
        failing_strategies = [
            k for k, v in quality_scores.items()
            if v < 0.6 and k != "overall"
        ]

        for strategy in failing_strategies:
            alternatives = self.cognition_base.get_relevant_strategies(
                context=strategy,
                category=None
            )
            if alternatives:
                top_alt = alternatives[0]
                recommendations.append(
                    f"Try alternative strategy for {strategy}: {top_alt.get('name')} "
                    f"(success rate: {top_alt.get('success_rate', 0):.2f})"
                )

        return recommendations

    def _meta_cognitive_analysis(self,
                                 result: Dict[str, Any],
                                 research_plan: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Meta-cognitive analysis: system's awareness of its own learning
        Tracks HOW the system is learning, not just WHAT it's learning
        """
        meta = {
            "learning_effectiveness": 0.0,
            "curiosity_alignment": 0.0,
            "pattern_recognition_trend": "stable",
            "exploration_vs_exploitation": "balanced"
        }

        # Learning effectiveness: are we creating knowledge structures?
        basins = result.get("basins_created", 0)
        concepts = len(result.get("concepts", []))
        if concepts > 0:
            meta["learning_effectiveness"] = min(basins / concepts, 1.0)

        # Curiosity alignment: are we exploring what we should?
        if research_plan:
            curiosity_triggers = research_plan.get("curiosity_triggers", [])
            questions = research_plan.get("challenging_questions", [])
            if curiosity_triggers:
                meta["curiosity_alignment"] = min(
                    len(questions) / (len(curiosity_triggers) * 3), 1.0
                )  # Expect ~3 questions per trigger

        # Pattern recognition trend: compare to previous analyses
        if len(self.analysis_history) >= 2:
            prev_quality = self.analysis_history[-1].get("quality_scores", {}).get("overall", 0)
            current_quality = result.get("quality_scores", {}).get("overall", 0)

            if current_quality > prev_quality + 0.1:
                meta["pattern_recognition_trend"] = "improving"
            elif current_quality < prev_quality - 0.1:
                meta["pattern_recognition_trend"] = "declining"

        # Exploration vs exploitation balance
        if research_plan:
            high_priority_questions = sum(
                1 for q in research_plan.get("challenging_questions", [])
                if q.get("difficulty") == "high"
            )
            total_questions = len(research_plan.get("challenging_questions", []))

            if total_questions > 0:
                exploration_ratio = high_priority_questions / total_questions
                if exploration_ratio > 0.6:
                    meta["exploration_vs_exploitation"] = "exploring"
                elif exploration_ratio < 0.3:
                    meta["exploration_vs_exploitation"] = "exploiting"

        return meta

    def _store_significant_insights(self, insights: List[Dict[str, Any]]):
        """Store high-significance insights in cognition base"""
        for insight in insights:
            if insight.get("significance", 0) > 0.75:
                self.cognition_base.add_insight(insight)
                logger.info(f"Stored significant insight: {insight.get('description')}")

    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of all analyses in session"""
        if not self.analysis_history:
            return {"message": "No analyses performed yet"}

        avg_quality = sum(
            a.get("quality_scores", {}).get("overall", 0)
            for a in self.analysis_history
        ) / len(self.analysis_history)

        total_insights = sum(
            len(a.get("insights", []))
            for a in self.analysis_history
        )

        return {
            "total_analyses": len(self.analysis_history),
            "average_quality_score": avg_quality,
            "total_insights_extracted": total_insights,
            "quality_trend": self._calculate_quality_trend(),
            "analysis_history": self.analysis_history
        }

    def _calculate_quality_trend(self) -> str:
        """Calculate overall quality trend across analyses"""
        if len(self.analysis_history) < 2:
            return "insufficient_data"

        qualities = [
            a.get("quality_scores", {}).get("overall", 0)
            for a in self.analysis_history
        ]

        # Simple linear trend
        first_half_avg = sum(qualities[:len(qualities)//2]) / (len(qualities)//2)
        second_half_avg = sum(qualities[len(qualities)//2:]) / (len(qualities) - len(qualities)//2)

        if second_half_avg > first_half_avg + 0.1:
            return "improving"
        elif second_half_avg < first_half_avg - 0.1:
            return "declining"
        else:
            return "stable"
