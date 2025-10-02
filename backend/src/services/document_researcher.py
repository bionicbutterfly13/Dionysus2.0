"""
Document Researcher - Generates research questions and exploration plans
Adapted from ASI-GO-2 researcher.py with consciousness and curiosity integration
"""
from typing import Dict, List, Any, Optional
import logging
from .document_cognition_base import DocumentCognitionBase

logger = logging.getLogger("dionysus.document_researcher")


class DocumentResearcher:
    """Generates research questions and exploration plans from document concepts"""

    def __init__(self, cognition_base: DocumentCognitionBase):
        self.cognition_base = cognition_base
        self.research_history = []

    def generate_research_questions(self,
                                   concepts: List[str],
                                   prediction_errors: Optional[Dict[str, float]] = None,
                                   previous_attempt: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate research questions for document concepts

        Args:
            concepts: Extracted document concepts
            prediction_errors: Active inference prediction errors (high = curiosity trigger)
            previous_attempt: Previous research attempt if refining

        Returns:
            Research plan with questions and exploration strategies
        """
        logger.info(f"Generating research questions for {len(concepts)} concepts")

        # Get relevant strategies from cognition base
        strategies = self.cognition_base.get_relevant_strategies(
            context=" ".join(concepts),
            category="curiosity_strategies"
        )

        # Identify high-curiosity concepts (high prediction error)
        curiosity_concepts = self._identify_curiosity_triggers(
            concepts, prediction_errors
        )

        # Generate challenging questions (R-Zero pattern)
        challenging_questions = self._generate_challenging_questions(
            curiosity_concepts, strategies
        )

        # Generate exploration plan
        exploration_plan = self._create_exploration_plan(
            concepts, challenging_questions, previous_attempt
        )

        research_output = {
            "concepts_analyzed": len(concepts),
            "curiosity_triggers": curiosity_concepts,
            "challenging_questions": challenging_questions,
            "exploration_plan": exploration_plan,
            "strategies_used": [s["name"] for s in strategies],
            "iteration": len(self.research_history) + 1
        }

        self.research_history.append(research_output)
        logger.info(f"Generated {len(challenging_questions)} research questions")

        return research_output

    def _identify_curiosity_triggers(self,
                                    concepts: List[str],
                                    prediction_errors: Optional[Dict[str, float]]) -> List[Dict[str, Any]]:
        """Identify concepts with high prediction error (Active Inference pattern)"""
        if not prediction_errors:
            # Default: all concepts have moderate curiosity
            return [{"concept": c, "prediction_error": 0.5, "priority": "medium"} for c in concepts]

        curiosity_threshold = 0.6
        curiosity_triggers = []

        for concept, error in prediction_errors.items():
            if error > curiosity_threshold:
                curiosity_triggers.append({
                    "concept": concept,
                    "prediction_error": error,
                    "priority": "high" if error > 0.8 else "medium",
                    "exploration_urgency": min(error, 1.0)
                })

        # Sort by prediction error (highest first)
        curiosity_triggers.sort(key=lambda x: x["prediction_error"], reverse=True)

        logger.info(f"Identified {len(curiosity_triggers)} high-curiosity concepts")
        return curiosity_triggers

    def _generate_challenging_questions(self,
                                       curiosity_concepts: List[Dict[str, Any]],
                                       strategies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate challenging questions targeting high prediction error concepts
        R-Zero pattern: Questions should push understanding boundaries
        """
        questions = []

        for concept_info in curiosity_concepts[:10]:  # Top 10 curiosity triggers
            concept = concept_info["concept"]
            error = concept_info["prediction_error"]

            # Question difficulty scales with prediction error
            if error > 0.8:
                # Very challenging questions for high uncertainty
                questions.extend([
                    {
                        "question": f"What are the fundamental principles underlying {concept}?",
                        "difficulty": "high",
                        "concept": concept,
                        "exploration_type": "foundational"
                    },
                    {
                        "question": f"How does {concept} relate to other concepts in this domain?",
                        "difficulty": "high",
                        "concept": concept,
                        "exploration_type": "relational"
                    },
                    {
                        "question": f"What are edge cases or limitations of {concept}?",
                        "difficulty": "high",
                        "concept": concept,
                        "exploration_type": "boundary_exploration"
                    }
                ])
            elif error > 0.6:
                # Moderate questions for medium uncertainty
                questions.extend([
                    {
                        "question": f"What are practical applications of {concept}?",
                        "difficulty": "medium",
                        "concept": concept,
                        "exploration_type": "application"
                    },
                    {
                        "question": f"How is {concept} implemented or realized?",
                        "difficulty": "medium",
                        "concept": concept,
                        "exploration_type": "implementation"
                    }
                ])

        logger.info(f"Generated {len(questions)} challenging questions")
        return questions

    def _create_exploration_plan(self,
                                concepts: List[str],
                                questions: List[Dict[str, Any]],
                                previous_attempt: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Create concrete exploration plan for knowledge acquisition"""

        # Group questions by exploration type
        exploration_types = {}
        for q in questions:
            exp_type = q["exploration_type"]
            if exp_type not in exploration_types:
                exploration_types[exp_type] = []
            exploration_types[exp_type].append(q)

        # Create phased exploration plan
        plan = {
            "phase_1_foundational": {
                "goal": "Understand core principles",
                "questions": exploration_types.get("foundational", []),
                "search_strategy": "academic sources, textbooks, foundational papers",
                "expected_duration": "high"
            },
            "phase_2_relational": {
                "goal": "Map concept relationships",
                "questions": exploration_types.get("relational", []),
                "search_strategy": "survey papers, knowledge graphs, domain ontologies",
                "expected_duration": "medium"
            },
            "phase_3_application": {
                "goal": "Understand practical usage",
                "questions": exploration_types.get("application", []),
                "search_strategy": "case studies, implementation guides, tutorials",
                "expected_duration": "medium"
            },
            "phase_4_boundary": {
                "goal": "Explore limitations and edge cases",
                "questions": exploration_types.get("boundary_exploration", []),
                "search_strategy": "research papers, discussion forums, critical analyses",
                "expected_duration": "low"
            }
        }

        # Refine based on previous attempt if available
        if previous_attempt:
            failed_questions = previous_attempt.get("failed_questions", [])
            plan["refinement_focus"] = {
                "retry_questions": failed_questions,
                "adjusted_strategy": "Simplify queries, use alternative sources"
            }

        return plan

    def refine_research_plan(self,
                           original_plan: Dict[str, Any],
                           feedback: Dict[str, Any]) -> Dict[str, Any]:
        """
        Refine research plan based on feedback from exploration attempts
        ASI-GO-2 iterative refinement pattern
        """
        logger.info("Refining research plan based on feedback")

        successful_questions = feedback.get("successful_questions", [])
        failed_questions = feedback.get("failed_questions", [])
        new_insights = feedback.get("insights", [])

        # Extract successful patterns
        successful_strategies = []
        for q in successful_questions:
            if "exploration_type" in q:
                successful_strategies.append(q["exploration_type"])

        # Generate refined questions for failed areas
        refined_questions = []
        for failed_q in failed_questions:
            # Simplify or reframe failed questions
            refined_questions.append({
                "question": f"Simplified: {failed_q.get('question', '')}",
                "difficulty": "low",
                "concept": failed_q.get("concept", ""),
                "exploration_type": "clarification",
                "previous_failure": failed_q.get("failure_reason", "unknown")
            })

        # Update cognition base with insights
        if new_insights:
            for insight in new_insights:
                self.cognition_base.add_insight({
                    "name": f"Research Insight: {insight.get('topic', 'Unknown')}",
                    "description": insight.get("description", ""),
                    "significance": insight.get("significance", 0.5),
                    "category": "research_insights"
                })

        refined_plan = {
            "original_plan_id": original_plan.get("iteration", 0),
            "successful_strategies": successful_strategies,
            "refined_questions": refined_questions,
            "new_exploration_focus": self._determine_new_focus(new_insights),
            "iteration": len(self.research_history) + 1
        }

        self.research_history.append(refined_plan)
        return refined_plan

    def _determine_new_focus(self, insights: List[Dict[str, Any]]) -> str:
        """Determine new research focus based on acquired insights"""
        if not insights:
            return "continue_current_exploration"

        # Analyze insight themes
        themes = {}
        for insight in insights:
            topic = insight.get("topic", "general")
            themes[topic] = themes.get(topic, 0) + 1

        # Focus on most common theme
        if themes:
            primary_theme = max(themes, key=themes.get)
            return f"deep_dive_{primary_theme}"

        return "broaden_exploration"

    def get_research_summary(self) -> Dict[str, Any]:
        """Get summary of research session"""
        total_questions = sum(
            len(r.get("challenging_questions", [])) for r in self.research_history
        )

        return {
            "total_iterations": len(self.research_history),
            "total_questions_generated": total_questions,
            "research_history": self.research_history,
            "avg_questions_per_iteration": total_questions / len(self.research_history) if self.research_history else 0
        }
