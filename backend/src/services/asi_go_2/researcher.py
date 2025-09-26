"""
Researcher - Proposes solutions by reasoning about which pattern to apply.
"""
import logging
from typing import Dict, Any, Optional, List
from llm_interface import LLMInterface
from cognitive_core.memory_system import CognitionBase, Pattern
from cognitive_core.thoughtseed_competition import InnerWorkspace, ThoughtGenerator, ThoughtType

logger = logging.getLogger("ASI-GO.Researcher")

class Researcher:
    """Generates solution proposals by selecting the best pattern through simulated reasoning."""

    def __init__(self, llm: LLMInterface, cognition_base: CognitionBase):
        self.llm = llm
        self.cognition_base = cognition_base
        self.proposal_history = []

    def propose_solution(self, goal: str, previous_attempt: Optional[Dict[str, Any]] = None) -> tuple[Dict[str, Any], Optional[InnerWorkspace]]:
        """Propose a solution for the given goal."""
        logger.info(f"Proposing solution for goal: {goal}")

        # 1. Get relevant patterns from long-term memory (CognitionBase)
        relevant_patterns = self.cognition_base.get_relevant_patterns(goal)
        workspace = None
        winning_pattern = None

        if not relevant_patterns:
            logger.warning("No relevant patterns found. Proceeding with a general approach.")
        else:
            # 2. Reason about the best pattern using the InnerWorkspace
            winning_pattern, workspace = self._select_best_pattern(goal, relevant_patterns)

        # 3. Build the prompt using the winning pattern
        system_prompt = ("You are an expert problem solver and programmer. "
                         "Your task is to propose a complete, working solution for the given problem, "
                         "applying the provided strategy.")

        prompt = f"Goal: {goal}\n\n"
        if winning_pattern:
            prompt += f"Chosen Strategy: {winning_pattern.name}\n"
            prompt += f"Description: {winning_pattern.description}\n\n"
        else:
            prompt += "No specific strategy chosen. Please determine the best approach.\n\n"

        if previous_attempt:
            prompt += f"A previous attempt failed with: {previous_attempt.get('error', 'Unknown error')}\n"
            prompt += "Please provide an improved solution based on the chosen strategy.\n\n"

        prompt += ("Please provide:\n"
                   "1. A clear explanation of your approach based on the strategy.\n"
                   "2. Complete, working Python code.\n"
                   "3. Expected output or results.")

        try:
            response = self.llm.query(prompt, system_prompt)
            proposal = {
                "goal": goal,
                "solution": response,
                "pattern_used": winning_pattern.name if winning_pattern else None,
                "iteration": len(self.proposal_history) + 1
            }
            self.proposal_history.append(proposal)
            logger.info(f"Solution proposal generated using pattern: {proposal['pattern_used']}")
            return proposal, workspace
        except Exception as e:
            logger.error(f"Failed to generate proposal: {e}")
            raise

    def _select_best_pattern(self, goal: str, patterns: List[Pattern]) -> tuple[Optional[Pattern], Optional[InnerWorkspace]]:
        """Uses an inner workspace to determine the best pattern to apply."""
        workspace = InnerWorkspace(capacity=len(patterns) + 1)
        generator = ThoughtGenerator(workspace)

        # Create a thought for the goal
        goal_thought = generator.generate_thought(goal, ThoughtType.GOAL)
        workspace.add_thought(goal_thought)

        # Create competing thoughts for each pattern
        pattern_thoughts = {}
        for p in patterns:
            content = f"Apply pattern: {p.name} - {p.description}"
            thought = generator.generate_thought(content, ThoughtType.ACTION, parent_ids=[goal_thought.id])
            thought.confidence = p.confidence
            thought.energy = p.success_rate # Use success rate as initial energy
            workspace.add_thought(thought)
            pattern_thoughts[thought.id] = p.name

        # Run the simulation for a few steps to let a winner emerge
        for _ in range(5):
            workspace.update()

        dominant_thought = workspace.get_dominant_thought()
        if dominant_thought and dominant_thought.id in pattern_thoughts:
            winner_name = pattern_thoughts[dominant_thought.id]
            logger.info(f"Pattern competition won by: '{winner_name}'")
            return self.cognition_base.patterns.get(winner_name), workspace
        
        logger.warning("No dominant pattern emerged from competition. Falling back to the best-rated pattern.")
        return (patterns[0], workspace) if patterns else (None, workspace)

    def refine_proposal(self, proposal: Dict[str, Any], feedback: Dict[str, Any]) -> Dict[str, Any]:
        """Refine a proposal based on feedback."""
        logger.info("Refining proposal based on feedback")
        system_prompt = "You are an expert at improving and debugging code based on feedback."
        prompt = f"""Original Goal: {proposal['goal']}

        Previous Solution (using pattern '{proposal.get('pattern_used', 'N/A')}'):
        {proposal['solution']}

        Feedback from testing:
        - Success: {feedback.get('success', False)}
        - Error: {feedback.get('error', 'None')}

        Please provide an improved solution that addresses the feedback, sticking to the original strategy.
        """
        try:
            response = self.llm.query(prompt, system_prompt)
            refined_proposal = {
                "goal": proposal['goal'],
                "solution": response,
                "pattern_used": proposal.get('pattern_used'),
                "iteration": proposal.get('iteration', 0) + 1,
                "refined_from": proposal.get('iteration', 0)
            }
            self.proposal_history.append(refined_proposal)
            return refined_proposal
        except Exception as e:
            logger.error(f"Failed to refine proposal: {e}")
            raise
