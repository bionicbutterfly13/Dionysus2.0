"""
Analyst - Analyzes results, extracts insights, and updates the memory system.
"""
import logging
from typing import Dict, Any, Optional, List
from llm_interface import LLMInterface
from cognitive_core.memory_system import CognitionBase, SemanticMemory, Pattern
from cognitive_core.thoughtseed_competition import InnerWorkspace, ThoughtSeed
from cognitive_core.thoughtseed_network import ThoughtseedNetwork

logger = logging.getLogger("ASI-GO.Analyst")

class Analyst:
    """Analyzes execution results to drive learning and memory consolidation."""

    def __init__(self, llm: LLMInterface, procedural_memory: CognitionBase, semantic_memory: SemanticMemory):
        self.llm = llm
        self.procedural_memory = procedural_memory
        self.semantic_memory = semantic_memory
        self.analyses = []

    def analyze_results(self, proposal: Dict[str, Any], test_result: Dict[str, Any],
                       validation: Dict[str, Any], workspace: Optional[InnerWorkspace] = None) -> Dict[str, Any]:
        """Analyzes results and updates all relevant parts of the memory system."""
        logger.info("Analyzing solution results...")

        # 1. Update procedural memory (pattern evolution)
        self._update_procedural_memory(proposal, test_result, workspace)

        # 2. Update semantic memory and potentially evolve patterns if successful
        if test_result.get('success') and validation.get('meets_goal'):
            self._update_semantic_memory(proposal, test_result)
            if workspace:
                self._enter_concentration_state(workspace, proposal.get('pattern_used'))

        # 3. Perform LLM-based analysis for deeper insights
        system_prompt = ("You are an expert at analyzing code execution results. 
                         "Focus on why the chosen strategy succeeded or failed and what can be learned.")
        prompt = self._build_llm_analysis_prompt(proposal, test_result, validation)

        try:
            response = self.llm.query(prompt, system_prompt)
            analysis = {
                "iteration": proposal.get('iteration', 1),
                "success": test_result['success'],
                "llm_analysis": response,
            }
            self.analyses.append(analysis)
            logger.info("LLM-based analysis complete.")
            return analysis
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return {"error": str(e)}

    def _enter_concentration_state(self, workspace: InnerWorkspace, pattern_name: str):
        """Identifies the winning thought and its cluster for potential evolution."""
        logger.info(f"Entering concentration state for pattern: {pattern_name}")

        winning_thought = None
        for thought in workspace.thoughts.values():
            if pattern_name in thought.content:
                winning_thought = thought
                break

        if not winning_thought:
            logger.warning("Could not find the winning thought in the workspace.")
            return

        # Get the spatial cluster and other active thoughts
        cluster_ids = workspace.get_thought_cluster(winning_thought.id)
        active_ids = {t.id for t in workspace.thoughts.values() if t.activation > workspace.activation_threshold}
        concentrated_ids = cluster_ids + list(active_ids)
        concentrated_thoughts = [workspace.thoughts[tid] for tid in set(concentrated_ids) if tid in workspace.thoughts]

        logger.info(f"Found {len(concentrated_thoughts)} thoughts in concentration state.")
        self._synthesize_new_pattern(concentrated_thoughts, workspace)

    def _synthesize_new_pattern(self, thoughts: List[ThoughtSeed], workspace: InnerWorkspace):
        """Analyzes a cluster of thoughts to synthesize a new pattern."""
        logger.info("Synthesizing new pattern from thought cluster...")
        if len(thoughts) < 3:
            logger.info("Cluster too small to synthesize a new pattern.")
            return

        network = ThoughtseedNetwork(thoughts)
        motifs = network.find_motifs()

        if not motifs:
            logger.info("No significant motifs found in the thought network.")
            return

        for motif in motifs:
            motif_thoughts = [workspace.thoughts[tid] for tid in motif]
            parent_patterns = [t for t in motif_thoughts if t.type == ThoughtType.ACTION]

            if not parent_patterns:
                continue

            # Generate new pattern name
            new_name = "_Evolved-" + "-".join(p.content.split(" ")[2] for p in parent_patterns)
            
            # Generate new description with LLM
            prompt = f"The following thought patterns were successful together:\n"
            for p in parent_patterns:
                prompt += f"- {p.content}\n"
            prompt += "\nPlease provide a concise description for a new, combined pattern."
            new_description = self.llm.query(prompt, "You are an expert in abstracting and describing problem-solving patterns.")

            # Combine applicable_to keywords
            applicable_to = set()
            for p_name in [p.content.split(" ")[2] for p in parent_patterns]:
                original_pattern = self.procedural_memory.patterns.get(p_name)
                if original_pattern:
                    applicable_to.update(original_pattern.applicable_to)

            new_pattern = Pattern(
                name=new_name,
                description=new_description,
                applicable_to=list(applicable_to),
                parent_pattern=",".join(p.content.split(" ")[2] for p in parent_patterns)
            )

            logger.info(f"Synthesized new pattern: {new_pattern.name}")
            self.procedural_memory.add_pattern(new_pattern)

    def _build_llm_analysis_prompt(self, proposal, test_result, validation):
        return f"""Goal: {proposal['goal']}
        Pattern Applied: {proposal.get('pattern_used', 'N/A')}

        Execution Results:
        - Success: {test_result['success']}
        - Output: {str(test_result.get('output', ''))[:500]}
        - Error: {test_result.get('error', 'None')}

        Please provide:
        1. A concise analysis of why the pattern succeeded or failed.
        2. One key lesson learned.
        3. A recommendation for the next step (REFINE, NEW_PATTERN, or GOAL_ACHIEVED).
        """

    def _update_procedural_memory(self, proposal: Dict[str, Any], test_result: Dict[str, Any], workspace: Optional[InnerWorkspace] = None):
        """Updates the procedural memory (CognitionBase) with the result."""
        pattern_name = proposal.get('pattern_used')
        if not pattern_name:
            return

        success = test_result.get('success', False)
        
        if workspace:
            # Find the thought corresponding to the pattern
            thought_to_update = None
            for thought in workspace.thoughts.values():
                if pattern_name in thought.content:
                    thought_to_update = thought
                    break
            
            if thought_to_update:
                # Update thought energy and confidence based on result
                if success:
                    thought_to_update.energy = min(1.0, thought_to_update.energy + 0.1)
                    thought_to_update.confidence = min(1.0, thought_to_update.confidence + 0.1)
                else:
                    thought_to_update.energy = max(0.0, thought_to_update.energy - 0.1)
                    thought_to_update.confidence = max(0.0, thought_to_update.confidence - 0.1)
                
                # Record the result with the updated thought properties
                self.procedural_memory.record_result(pattern_name, {
                    "goal": proposal['goal'],
                    "success": success,
                    "updated_energy": thought_to_update.energy,
                    "updated_confidence": thought_to_update.confidence
                })
                return

        # Fallback to existing logic if workspace is not available
        self.procedural_memory.record_result(pattern_name, {
            "goal": proposal['goal'],
            "success": success
        })

    def _update_semantic_memory(self, proposal: Dict[str, Any], test_result: Dict[str, Any]):
        """Extracts a new fact from a successful solution and adds it to semantic memory."""
        fact_content = f"The pattern '{proposal.get('pattern_used')}' successfully solved the goal: '{proposal['goal']}'"
        self.semantic_memory.add_fact(
            content=fact_content,
            source=f"successful_solution_iteration_{proposal.get('iteration', 0)}",
            confidence=0.8 # Confidence is high but not 1.0, as it's one data point
        )
        logger.info(f"Added new fact to semantic memory: {fact_content}")

    def generate_summary_report(self) -> str:
        """Generates a summary report of the session."""
        if not self.analyses:
            return "No analyses performed yet."
        successful = sum(1 for a in self.analyses if a['success'])
        total = len(self.analyses)
        report = f"""ASI-GO Session Summary
========================
Total Attempts: {total}
Success Rate: {successful/total*100:.1f}%

"""
        report += self.procedural_memory.get_pattern_summary()
        return report

    def recommend_next_action(self) -> str:
        """Recommends the next action based on the analysis history."""
        if not self.analyses:
            return "Start with a new proposal."
        last_analysis = self.analyses[-1]
        if last_analysis.get('success') and last_analysis.get('validation', {}).get('meets_goal'):
            return "Goal achieved!"
        if len(self.analyses) >= 5:
            return "Maximum iterations reached."
        if "NEW_PATTERN" in last_analysis.get('llm_analysis', ''):
            return "Try a different pattern."
        return "Refine the current solution."

    def _build_llm_analysis_prompt(self, proposal, test_result, validation):
        return f"""Goal: {proposal['goal']}
        Pattern Applied: {proposal.get('pattern_used', 'N/A')}

        Execution Results:
        - Success: {test_result['success']}
        - Output: {str(test_result.get('output', ''))[:500]}
        - Error: {test_result.get('error', 'None')}

        Please provide:
        1. A concise analysis of why the pattern succeeded or failed.
        2. One key lesson learned.
        3. A recommendation for the next step (REFINE, NEW_PATTERN, or GOAL_ACHIEVED).
        """

    def _update_procedural_memory(self, proposal: Dict[str, Any], test_result: Dict[str, Any], workspace: Optional[InnerWorkspace] = None):
        """Updates the procedural memory (CognitionBase) with the result."""
        pattern_name = proposal.get('pattern_used')
        if not pattern_name:
            return

        success = test_result.get('success', False)
        
        if workspace:
            # Find the thought corresponding to the pattern
            thought_to_update = None
            for thought in workspace.thoughts.values():
                if pattern_name in thought.content:
                    thought_to_update = thought
                    break
            
            if thought_to_update:
                # Update thought energy and confidence based on result
                if success:
                    thought_to_update.energy = min(1.0, thought_to_update.energy + 0.1)
                    thought_to_update.confidence = min(1.0, thought_to_update.confidence + 0.1)
                else:
                    thought_to_update.energy = max(0.0, thought_to_update.energy - 0.1)
                    thought_to_update.confidence = max(0.0, thought_to_update.confidence - 0.1)
                
                # Record the result with the updated thought properties
                self.procedural_memory.record_result(pattern_name, {
                    "goal": proposal['goal'],
                    "success": success,
                    "updated_energy": thought_to_update.energy,
                    "updated_confidence": thought_to_update.confidence
                })
                return

        # Fallback to existing logic if workspace is not available
        self.procedural_memory.record_result(pattern_name, {
            "goal": proposal['goal'],
            "success": success
        })

    def generate_summary_report(self) -> str:
        """Generates a summary report of the session."""
        if not self.analyses:
            return "No analyses performed yet."
        successful = sum(1 for a in self.analyses if a['success'])
        total = len(self.analyses)
        report = f"""ASI-GO Session Summary
========================
Total Attempts: {total}
Success Rate: {successful/total*100:.1f}%

"
        report += self.procedural_memory.get_pattern_summary()
        return report

    def recommend_next_action(self) -> str:
        """Recommends the next action based on the analysis history."""
        if not self.analyses:
            return "Start with a new proposal."
        last_analysis = self.analyses[-1]
        if last_analysis.get('success') and last_analysis.get('validation', {}).get('meets_goal'):
            return "Goal achieved!"
        if len(self.analyses) >= 5:
            return "Maximum iterations reached."
        if "NEW_PATTERN" in last_analysis.get('llm_analysis', ''):
            return "Try a different pattern."
        return "Refine the current solution."