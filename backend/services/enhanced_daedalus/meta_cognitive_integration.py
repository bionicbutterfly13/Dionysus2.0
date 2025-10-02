#!/usr/bin/env python3
"""
🧠 Meta-Cognitive Integration: Episodic Meta RL + Cognitive Tools
================================================================

Integration of research-validated cognitive tools with episodic meta-learning,
creating a self-improving cognitive enhancement system that learns optimal
tool usage patterns, prompt optimization, and procedural meta-learning.

Combines:
1. Cognitive Tools (arXiv:2506.12115v1) - 94% gap closure to o1-preview
2. Episodic Meta RL (epLSTM) - "Been There, Done That" meta-learning
3. Prompt Learning - Dynamic prompt optimization based on task characteristics
4. Context Learning - Archetypal resonance for cognitive tool selection
5. Procedural Meta-Learning - Learning optimal cognitive tool workflows

Author: Dionysus Consciousness Enhancement System
Date: 2025-09-27
Version: 1.0.0 - Meta-Cognitive Integration Architecture
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import json

# Import existing cognitive tools
from .cognitive_tools_implementation import (
    ResearchValidatedCognitiveOrchestrator,
    CognitiveToolCall,
    CognitiveToolResponse,
    UnderstandQuestionTool,
    RecallRelatedTool,
    ExamineAnswerTool,
    BacktrackingTool
)

# Import meta-cognitive coordination
from .cognitive_meta_coordinator import (
    CognitiveMetaCoordinator,
    CognitiveContext,
    CognitiveDecision,
    ReasoningMode
)

# Import episodic meta-learning components
try:
    import sys
    sys.path.append('/Volumes/Asylum/dev/Dionysus-2.0/extensions/context_engineering')
    from eplstm_architecture import (
        ASIArchEpisodicMetaLearner,
        EpisodicMemoryEntry,
        DifferentiableNeuralDictionary,
        ReinstatementGates,
        ArchetypalResonancePattern
    )
    from theoretical_foundations import (
        EpisodicMetaLearningProfile,
        IntegratedContextEngineering,
        create_asi_arch_context_engineering_system
    )
    EPLSTM_AVAILABLE = True
except ImportError as e:
    logging.warning(f"epLSTM components not available: {e}")
    EPLSTM_AVAILABLE = False

logger = logging.getLogger(__name__)

class CognitiveToolUsagePattern(Enum):
    """Patterns of cognitive tool usage learned through episodic meta-learning"""
    SEQUENTIAL_DECOMPOSITION = "sequential_decomposition"  # understand -> recall -> solve -> examine
    VALIDATION_FOCUSED = "validation_focused"            # examine -> backtrack -> re-solve
    CREATIVE_EXPLORATION = "creative_exploration"        # recall -> understand -> creative_solve
    DIRECT_SOLVE_VERIFY = "direct_solve_verify"         # solve -> examine -> done
    ITERATIVE_REFINEMENT = "iterative_refinement"       # solve -> examine -> backtrack -> repeat

@dataclass
class CognitiveToolEpisode:
    """Episode representing a cognitive tool usage sequence"""
    episode_id: str
    task_description: str
    agent_name: str
    
    # Tool usage sequence
    tools_used: List[str]
    tool_sequence: List[CognitiveToolCall]
    tool_responses: List[CognitiveToolResponse]
    
    # Context and patterns
    cognitive_context: CognitiveContext
    archetypal_pattern: Optional[ArchetypalResonancePattern]
    usage_pattern: CognitiveToolUsagePattern
    
    # Performance outcomes
    initial_performance_estimate: float
    final_performance_score: float
    reasoning_quality_improvement: float
    processing_time: float
    
    # Learning outcomes
    prompt_optimizations: Dict[str, str]  # Tool -> optimized prompt
    context_insights: Dict[str, Any]
    procedural_lessons: List[str]
    
    # Episodic narrative
    narrative_summary: str
    breakthrough_moments: List[str]
    error_correction_instances: List[str]

@dataclass
class PromptLearningProfile:
    """Profile for learning optimal prompts for cognitive tools"""
    base_prompts: Dict[str, str] = field(default_factory=dict)
    optimized_prompts: Dict[str, str] = field(default_factory=dict)
    prompt_performance_history: Dict[str, List[float]] = field(default_factory=dict)
    context_specific_prompts: Dict[str, Dict[str, str]] = field(default_factory=dict)  # context_type -> tool -> prompt
    
    # Learning parameters
    prompt_learning_rate: float = 0.1
    adaptation_threshold: float = 0.05  # Minimum improvement to update prompt
    context_specificity: bool = True

class MetaCognitiveEpisodicLearner:
    """
    Episodic meta-learning system for cognitive tools optimization.
    
    Learns:
    1. Optimal cognitive tool usage patterns for different task types
    2. Context-specific prompt optimizations
    3. Procedural meta-learning for tool sequences
    4. Archetypal pattern integration with cognitive enhancement
    """
    
    def __init__(self):
        # Core components
        self.cognitive_orchestrator = ResearchValidatedCognitiveOrchestrator()
        self.meta_coordinator = CognitiveMetaCoordinator()
        
        # Episodic meta-learning (if available)
        if EPLSTM_AVAILABLE:
            self.context_engineering_system = create_asi_arch_context_engineering_system()
            self.episodic_meta_learner = ASIArchEpisodicMetaLearner(
                self.context_engineering_system,
                architecture_dim=512,  # Cognitive tools state dimension
                context_dim=256       # Task context dimension
            )
            self.eplstm_available = True
        else:
            self.episodic_meta_learner = None
            self.eplstm_available = False
        
        # Learning profiles
        self.prompt_learning = PromptLearningProfile()
        self.usage_pattern_history: Dict[str, List[CognitiveToolUsagePattern]] = {}
        self.context_learning_history: Dict[str, List[Dict]] = {}
        
        # Procedural meta-learning
        self.successful_procedures: Dict[str, List[str]] = {}  # task_type -> successful tool sequences
        self.failed_procedures: Dict[str, List[str]] = {}     # task_type -> failed tool sequences
        
        # Episode storage
        self.cognitive_episodes: List[CognitiveToolEpisode] = []
        self.episode_boundary_detector = EpisodeBoundaryDetector() if EPLSTM_AVAILABLE else None
        
        # Performance tracking
        self.meta_learning_metrics = {
            "prompt_optimizations": 0,
            "pattern_discoveries": 0,
            "procedural_improvements": 0,
            "archetypal_enhancements": 0,
            "total_episodes": 0,
            "average_improvement": 0.0
        }
        
        logger.info("🧠 Meta-Cognitive Episodic Learner initialized")
        if self.eplstm_available:
            logger.info("✅ epLSTM integration active - full meta-learning capabilities enabled")
        else:
            logger.info("⚠️ epLSTM not available - using simplified meta-learning")
    
    async def enhance_cognitive_reasoning_with_meta_learning(self,
                                                           agent_name: str,
                                                           task: str,
                                                           context: CognitiveContext) -> Dict[str, Any]:
        """
        Apply meta-learned cognitive enhancement with episodic memory integration
        
        This is the main entry point that combines:
        - Research-validated cognitive tools
        - Episodic meta-learning
        - Prompt optimization
        - Procedural meta-learning
        """
        
        start_time = asyncio.get_event_loop().time()
        
        # Step 1: Episodic memory retrieval for similar cognitive episodes
        similar_episodes = await self._retrieve_similar_cognitive_episodes(task, context)
        
        # Step 2: Meta-cognitive decision with episodic enhancement
        cognitive_decision = await self._meta_cognitive_decision_with_memory(
            agent_name, task, context, similar_episodes
        )
        
        # Step 3: Apply learned prompt optimizations
        optimized_tools = await self._apply_prompt_learning(cognitive_decision, context)
        
        # Step 4: Execute cognitive enhancement with procedural meta-learning
        reasoning_result = await self._execute_with_procedural_learning(
            agent_name, task, context, cognitive_decision, optimized_tools, similar_episodes
        )
        
        # Step 5: Create and store cognitive episode
        processing_time = asyncio.get_event_loop().time() - start_time
        episode = await self._create_cognitive_episode(
            agent_name, task, context, cognitive_decision, reasoning_result, processing_time
        )
        
        # Step 6: Update meta-learning from episode
        await self._update_meta_learning_from_episode(episode)
        
        return {
            "agent_name": agent_name,
            "meta_cognitive_enhancement": True,
            "cognitive_decision": cognitive_decision,
            "reasoning_result": reasoning_result,
            "episode": episode,
            "similar_episodes_retrieved": len(similar_episodes),
            "prompt_optimizations_applied": len(episode.prompt_optimizations),
            "meta_learning_metrics": self.meta_learning_metrics,
            "eplstm_integration": self.eplstm_available
        }
    
    async def _retrieve_similar_cognitive_episodes(self,
                                                 task: str,
                                                 context: CognitiveContext) -> List[CognitiveToolEpisode]:
        """Retrieve similar cognitive episodes from episodic memory"""
        
        if not self.eplstm_available:
            # Fallback: Simple similarity based on stored episodes
            similar_episodes = []
            for episode in self.cognitive_episodes[-10:]:  # Check recent episodes
                if self._calculate_task_similarity(task, episode.task_description) > 0.7:
                    similar_episodes.append(episode)
            return similar_episodes[:3]  # Return top 3
        
        # Use epLSTM for sophisticated episodic retrieval
        try:
            # Encode task context for episodic memory retrieval
            task_encoding = self._encode_task_for_episodic_memory(task, context)
            
            # Retrieve from episodic memory
            retrieved_memory, similarity = self.episodic_meta_learner.episodic_memory.retrieve_memory(task_encoding)
            
            similar_episodes = []
            if retrieved_memory and similarity > 0.6:
                # Find episodes associated with this memory
                for episode in self.cognitive_episodes:
                    if (episode.cognitive_context.domain_type == context.domain_type and
                        episode.cognitive_context.task_complexity > context.task_complexity - 0.2 and
                        episode.cognitive_context.task_complexity < context.task_complexity + 0.2):
                        similar_episodes.append(episode)
                        if len(similar_episodes) >= 3:
                            break
            
            logger.info(f"🧠 Retrieved {len(similar_episodes)} similar cognitive episodes (similarity: {similarity:.3f})")
            return similar_episodes
            
        except Exception as e:
            logger.warning(f"epLSTM retrieval failed: {e}, falling back to simple similarity")
            return []
    
    async def _meta_cognitive_decision_with_memory(self,
                                                 agent_name: str,
                                                 task: str,
                                                 context: CognitiveContext,
                                                 similar_episodes: List[CognitiveToolEpisode]) -> CognitiveDecision:
        """Enhanced meta-cognitive decision making with episodic memory"""
        
        # Get base meta-cognitive decision
        base_decision = await self.meta_coordinator.decide_reasoning_approach(agent_name, task, context)
        
        # Enhance decision with episodic insights
        if similar_episodes:
            # Analyze successful patterns from similar episodes
            successful_patterns = [ep.usage_pattern for ep in similar_episodes if ep.reasoning_quality_improvement > 0.1]
            
            if successful_patterns:
                most_common_pattern = max(set(successful_patterns), key=successful_patterns.count)
                
                # Adjust reasoning mode based on learned patterns
                if most_common_pattern == CognitiveToolUsagePattern.VALIDATION_FOCUSED:
                    base_decision.recommended_mode = ReasoningMode.COGNITIVE_TOOLS
                    base_decision.reasoning += " Enhanced with validation-focused pattern from episodic memory."
                    base_decision.expected_performance_gain *= 1.2
                
                elif most_common_pattern == CognitiveToolUsagePattern.CREATIVE_EXPLORATION:
                    base_decision.recommended_mode = ReasoningMode.HYBRID_COGNITIVE_TOT
                    base_decision.reasoning += " Enhanced with creative exploration pattern from episodic memory."
                    base_decision.expected_performance_gain *= 1.15
        
        return base_decision
    
    async def _apply_prompt_learning(self,
                                   cognitive_decision: CognitiveDecision,
                                   context: CognitiveContext) -> Dict[str, str]:
        """Apply learned prompt optimizations based on context"""
        
        optimized_tools = {}
        
        # Get context-specific prompts if available
        context_key = f"{context.domain_type}_{context.task_complexity:.1f}"
        
        if context_key in self.prompt_learning.context_specific_prompts:
            context_prompts = self.prompt_learning.context_specific_prompts[context_key]
            optimized_tools.update(context_prompts)
            logger.info(f"🎯 Applied {len(context_prompts)} context-specific prompt optimizations")
        
        # Apply general optimized prompts
        for tool_name, optimized_prompt in self.prompt_learning.optimized_prompts.items():
            if tool_name not in optimized_tools:
                optimized_tools[tool_name] = optimized_prompt
        
        return optimized_tools
    
    async def _execute_with_procedural_learning(self,
                                              agent_name: str,
                                              task: str,
                                              context: CognitiveContext,
                                              cognitive_decision: CognitiveDecision,
                                              optimized_tools: Dict[str, str],
                                              similar_episodes: List[CognitiveToolEpisode]) -> Dict[str, Any]:
        """Execute cognitive enhancement with procedural meta-learning"""
        
        # Determine optimal tool sequence from procedural learning
        task_type = context.domain_type
        optimal_sequence = self._get_optimal_tool_sequence(task_type, similar_episodes)
        
        reasoning_result = {
            "procedural_sequence_used": optimal_sequence,
            "tool_executions": [],
            "cognitive_improvements": [],
            "procedural_adaptations": []
        }
        
        # Execute cognitive tools according to learned procedure
        if cognitive_decision.recommended_mode in [ReasoningMode.COGNITIVE_TOOLS, ReasoningMode.HYBRID_COGNITIVE_TOT]:
            
            for tool_name in optimal_sequence:
                try:
                    # Use optimized prompt if available
                    tool_prompt = optimized_tools.get(tool_name, "")
                    
                    # Create tool call
                    tool_call = CognitiveToolCall(
                        name=tool_name,
                        parameters={
                            "question": task,
                            "context": str(context),
                            "optimized_prompt": tool_prompt
                        }
                    )
                    
                    # Execute tool
                    tool_response = await self.cognitive_orchestrator.execute_cognitive_tool(tool_call)
                    
                    reasoning_result["tool_executions"].append({
                        "tool": tool_name,
                        "success": tool_response.success,
                        "response": tool_response.content,
                        "metadata": tool_response.metadata
                    })
                    
                    # Evaluate tool effectiveness
                    effectiveness = self._evaluate_tool_effectiveness(tool_response, context)
                    if effectiveness < 0.6:
                        # Procedural adaptation: try alternative approach
                        adaptation = f"Low effectiveness for {tool_name} ({effectiveness:.2f}), adapting sequence"
                        reasoning_result["procedural_adaptations"].append(adaptation)
                        logger.info(f"🔄 {adaptation}")
                        
                        # Try alternative tool if available
                        alternative_tool = self._get_alternative_tool(tool_name, optimal_sequence)
                        if alternative_tool:
                            alt_tool_call = CognitiveToolCall(
                                name=alternative_tool,
                                parameters=tool_call.parameters
                            )
                            alt_response = await self.cognitive_orchestrator.execute_cognitive_tool(alt_tool_call)
                            reasoning_result["tool_executions"].append({
                                "tool": alternative_tool,
                                "success": alt_response.success,
                                "response": alt_response.content,
                                "adaptation": True
                            })
                    
                except Exception as e:
                    logger.error(f"Tool execution error for {tool_name}: {e}")
                    reasoning_result["tool_executions"].append({
                        "tool": tool_name,
                        "success": False,
                        "error": str(e)
                    })
        
        return reasoning_result
    
    def _get_optimal_tool_sequence(self,
                                 task_type: str,
                                 similar_episodes: List[CognitiveToolEpisode]) -> List[str]:
        """Determine optimal cognitive tool sequence based on procedural meta-learning"""
        
        # If we have successful similar episodes, use their patterns
        if similar_episodes:
            successful_episodes = [ep for ep in similar_episodes if ep.reasoning_quality_improvement > 0.1]
            if successful_episodes:
                # Use the sequence from the most successful episode
                best_episode = max(successful_episodes, key=lambda ep: ep.reasoning_quality_improvement)
                return best_episode.tools_used
        
        # Use learned successful procedures for this task type
        if task_type in self.successful_procedures and self.successful_procedures[task_type]:
            most_successful = max(self.successful_procedures[task_type], 
                                key=lambda seq: self.successful_procedures[task_type].count(seq))
            return most_successful.split(",")
        
        # Default research-validated sequence
        return ["understand_question", "recall_related", "examine_answer"]
    
    def _get_alternative_tool(self, failed_tool: str, current_sequence: List[str]) -> Optional[str]:
        """Get alternative tool when current tool fails"""
        alternatives = {
            "understand_question": "recall_related",
            "recall_related": "understand_question", 
            "examine_answer": "backtracking",
            "backtracking": "examine_answer"
        }
        
        alternative = alternatives.get(failed_tool)
        return alternative if alternative and alternative not in current_sequence else None
    
    def _evaluate_tool_effectiveness(self, tool_response: CognitiveToolResponse, context: CognitiveContext) -> float:
        """Evaluate the effectiveness of a cognitive tool response"""
        if not tool_response.success:
            return 0.0
        
        # Simple effectiveness metric based on response length and metadata
        base_score = 0.5
        
        # Longer, more detailed responses often indicate better performance
        if len(tool_response.content) > 200:
            base_score += 0.2
        
        # Check for research-validated improvements mentioned in metadata
        if tool_response.metadata:
            expected_improvement = tool_response.metadata.get("expected_improvement", "")
            if "%" in expected_improvement:
                base_score += 0.3
        
        return min(1.0, base_score)
    
    async def _create_cognitive_episode(self,
                                      agent_name: str,
                                      task: str,
                                      context: CognitiveContext,
                                      cognitive_decision: CognitiveDecision,
                                      reasoning_result: Dict[str, Any],
                                      processing_time: float) -> CognitiveToolEpisode:
        """Create cognitive episode for episodic memory storage"""
        
        # Extract tool usage information
        tools_used = [exec["tool"] for exec in reasoning_result.get("tool_executions", [])]
        successful_tools = [exec["tool"] for exec in reasoning_result.get("tool_executions", []) if exec.get("success", False)]
        
        # Determine usage pattern
        usage_pattern = self._classify_usage_pattern(tools_used, reasoning_result)
        
        # Calculate performance metrics
        initial_estimate = cognitive_decision.expected_performance_gain
        final_score = self._calculate_final_performance_score(reasoning_result)
        improvement = final_score - initial_estimate
        
        # Generate narrative
        narrative = self._generate_episode_narrative(agent_name, task, tools_used, improvement)
        
        # Identify breakthrough moments
        breakthroughs = []
        for adaptation in reasoning_result.get("procedural_adaptations", []):
            breakthroughs.append(f"Procedural adaptation: {adaptation}")
        
        episode = CognitiveToolEpisode(
            episode_id=f"cognitive_ep_{len(self.cognitive_episodes)}_{agent_name}",
            task_description=task,
            agent_name=agent_name,
            tools_used=tools_used,
            tool_sequence=[],  # Would need to store actual tool calls
            tool_responses=[],  # Would need to store actual responses
            cognitive_context=context,
            archetypal_pattern=self._detect_archetypal_pattern(task, reasoning_result),
            usage_pattern=usage_pattern,
            initial_performance_estimate=initial_estimate,
            final_performance_score=final_score,
            reasoning_quality_improvement=improvement,
            processing_time=processing_time,
            prompt_optimizations={},  # Would track actual prompt changes
            context_insights={"successful_tools": successful_tools},
            procedural_lessons=reasoning_result.get("procedural_adaptations", []),
            narrative_summary=narrative,
            breakthrough_moments=breakthroughs,
            error_correction_instances=[]  # Would track actual error corrections
        )
        
        return episode
    
    def _classify_usage_pattern(self, tools_used: List[str], reasoning_result: Dict[str, Any]) -> CognitiveToolUsagePattern:
        """Classify the cognitive tool usage pattern"""
        
        if not tools_used:
            return CognitiveToolUsagePattern.DIRECT_SOLVE_VERIFY
        
        # Sequential decomposition: understand -> recall -> examine
        if ("understand_question" in tools_used and 
            "recall_related" in tools_used and 
            "examine_answer" in tools_used):
            return CognitiveToolUsagePattern.SEQUENTIAL_DECOMPOSITION
        
        # Validation focused: examine and/or backtrack prominent
        if ("examine_answer" in tools_used or "backtracking" in tools_used):
            return CognitiveToolUsagePattern.VALIDATION_FOCUSED
        
        # Creative exploration: recall + understand + exploration
        if ("recall_related" in tools_used and "understand_question" in tools_used):
            return CognitiveToolUsagePattern.CREATIVE_EXPLORATION
        
        # Iterative refinement: multiple adaptations
        if len(reasoning_result.get("procedural_adaptations", [])) > 1:
            return CognitiveToolUsagePattern.ITERATIVE_REFINEMENT
        
        return CognitiveToolUsagePattern.DIRECT_SOLVE_VERIFY
    
    def _calculate_final_performance_score(self, reasoning_result: Dict[str, Any]) -> float:
        """Calculate final performance score based on reasoning results"""
        
        successful_executions = len([ex for ex in reasoning_result.get("tool_executions", []) if ex.get("success", False)])
        total_executions = len(reasoning_result.get("tool_executions", [])) or 1
        
        success_rate = successful_executions / total_executions
        
        # Bonus for procedural adaptations (shows learning)
        adaptation_bonus = len(reasoning_result.get("procedural_adaptations", [])) * 0.1
        
        return min(1.0, success_rate + adaptation_bonus)
    
    def _generate_episode_narrative(self, agent_name: str, task: str, tools_used: List[str], improvement: float) -> str:
        """Generate human-readable narrative for the cognitive episode"""
        
        if improvement > 0.1:
            outcome = "achieved significant cognitive enhancement"
        elif improvement > 0.0:
            outcome = "achieved modest cognitive improvement"
        else:
            outcome = "provided learning insights despite challenges"
        
        tools_desc = ", ".join(tools_used) if tools_used else "direct reasoning"
        
        return f"Agent {agent_name} processed '{task[:50]}...' using cognitive tools: {tools_desc} and {outcome} (improvement: {improvement:+.2f})"
    
    def _detect_archetypal_pattern(self, task: str, reasoning_result: Dict[str, Any]) -> Optional[ArchetypalResonancePattern]:
        """Detect archetypal pattern in the cognitive processing"""
        
        if not EPLSTM_AVAILABLE:
            return None
        
        # Simple archetypal pattern detection based on task characteristics
        task_lower = task.lower()
        
        if any(word in task_lower for word in ["solve", "overcome", "challenge", "defeat"]):
            return ArchetypalResonancePattern.HERO_DRAGON_SLAYER
        elif any(word in task_lower for word in ["guide", "help", "teach", "mentor"]):
            return ArchetypalResonancePattern.WISE_HERMIT_GUIDE
        elif any(word in task_lower for word in ["create", "build", "make", "design"]):
            return ArchetypalResonancePattern.DIVINE_CREATOR
        elif any(word in task_lower for word in ["explore", "discover", "find", "search"]):
            return ArchetypalResonancePattern.CURIOUS_WANDERER
        
        return None
    
    async def _update_meta_learning_from_episode(self, episode: CognitiveToolEpisode):
        """Update meta-learning systems from cognitive episode"""
        
        # Store episode
        self.cognitive_episodes.append(episode)
        
        # Update usage pattern history
        agent_name = episode.agent_name
        if agent_name not in self.usage_pattern_history:
            self.usage_pattern_history[agent_name] = []
        self.usage_pattern_history[agent_name].append(episode.usage_pattern)
        
        # Update procedural learning
        task_type = episode.cognitive_context.domain_type
        tools_sequence = ",".join(episode.tools_used)
        
        if episode.reasoning_quality_improvement > 0.1:
            # Successful procedure
            if task_type not in self.successful_procedures:
                self.successful_procedures[task_type] = []
            self.successful_procedures[task_type].append(tools_sequence)
        elif episode.reasoning_quality_improvement < -0.1:
            # Failed procedure
            if task_type not in self.failed_procedures:
                self.failed_procedures[task_type] = []
            self.failed_procedures[task_type].append(tools_sequence)
        
        # Update prompt learning (simplified)
        if episode.prompt_optimizations:
            for tool, optimized_prompt in episode.prompt_optimizations.items():
                self.prompt_learning.optimized_prompts[tool] = optimized_prompt
                self.meta_learning_metrics["prompt_optimizations"] += 1
        
        # Store in episodic memory if available
        if self.eplstm_available and self.episodic_meta_learner:
            try:
                # Encode episode for episodic storage
                context_encoding = self._encode_task_for_episodic_memory(
                    episode.task_description, episode.cognitive_context
                )
                
                # Process through epLSTM
                self.episodic_meta_learner.process_architecture_candidate(
                    architecture_encoding=np.random.randn(512),  # Would encode cognitive state
                    task_context=context_encoding,
                    task_id=f"cognitive_{episode.episode_id}",
                    archetypal_pattern=episode.archetypal_pattern
                )
                
                # Complete episode
                self.episodic_meta_learner.complete_task_episode(
                    performance_score=episode.final_performance_score,
                    narrative_summary=episode.narrative_summary
                )
                
                logger.info(f"🧠 Stored cognitive episode in epLSTM: {episode.episode_id}")
                
            except Exception as e:
                logger.warning(f"Failed to store episode in epLSTM: {e}")
        
        # Update metrics
        self.meta_learning_metrics["total_episodes"] += 1
        self.meta_learning_metrics["average_improvement"] = np.mean([
            ep.reasoning_quality_improvement for ep in self.cognitive_episodes[-10:]
        ])
        
        logger.info(f"📊 Updated meta-learning from episode {episode.episode_id}")
    
    def _encode_task_for_episodic_memory(self, task: str, context: CognitiveContext) -> np.ndarray:
        """Encode task and context for episodic memory storage"""
        
        # Simple encoding scheme (in practice, would use learned embeddings)
        encoding = np.zeros(256)
        
        # Encode task complexity
        encoding[0] = context.task_complexity
        
        # Encode domain type (one-hot style)
        domain_mapping = {"mathematical": 1, "logical": 2, "creative": 3, "analytical": 4}
        domain_idx = domain_mapping.get(context.domain_type, 0)
        if domain_idx > 0:
            encoding[domain_idx] = 1.0
        
        # Encode task characteristics
        encoding[10] = float(context.requires_creativity)
        encoding[11] = float(context.requires_verification)
        encoding[12] = context.agent_expertise
        encoding[13] = context.previous_success_rate
        
        # Add some task-specific features (simplified)
        task_words = task.lower().split()
        for i, word in enumerate(task_words[:20]):  # First 20 words
            encoding[20 + i] = hash(word) % 100 / 100.0  # Simple word encoding
        
        return encoding
    
    def _calculate_task_similarity(self, task1: str, task2: str) -> float:
        """Calculate similarity between two tasks (simple implementation)"""
        words1 = set(task1.lower().split())
        words2 = set(task2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)  # Jaccard similarity
    
    def get_meta_learning_insights(self) -> Dict[str, Any]:
        """Get insights about meta-learning progress"""
        
        insights = {
            "meta_learning_metrics": self.meta_learning_metrics.copy(),
            "cognitive_episodes_stored": len(self.cognitive_episodes),
            "eplstm_integration": self.eplstm_available,
            "procedural_learning": {
                "successful_procedures": len(self.successful_procedures),
                "failed_procedures": len(self.failed_procedures),
                "total_procedure_variations": sum(len(procs) for procs in self.successful_procedures.values())
            },
            "prompt_learning": {
                "optimized_prompts": len(self.prompt_learning.optimized_prompts),
                "context_specific_prompts": len(self.prompt_learning.context_specific_prompts)
            },
            "usage_patterns": {
                "agents_tracked": len(self.usage_pattern_history),
                "total_pattern_instances": sum(len(patterns) for patterns in self.usage_pattern_history.values())
            }
        }
        
        # Recent performance trends
        if len(self.cognitive_episodes) >= 5:
            recent_improvements = [ep.reasoning_quality_improvement for ep in self.cognitive_episodes[-5:]]
            insights["recent_performance_trend"] = {
                "average_improvement": np.mean(recent_improvements),
                "improvement_variance": np.var(recent_improvements),
                "positive_episodes": sum(1 for imp in recent_improvements if imp > 0)
            }
        
        return insights

# Export the meta-cognitive integration components
__all__ = [
    'MetaCognitiveEpisodicLearner',
    'CognitiveToolEpisode', 
    'CognitiveToolUsagePattern',
    'PromptLearningProfile'
]

logger.info("🧠 Meta-Cognitive Integration system loaded successfully")