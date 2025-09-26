"""
Context Isolated Agent - LangGraph Implementation
================================================

Implements context window isolation for agents using LangGraph patterns.
Each agent runs in its own context with complete information, preventing context rot.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from datetime import datetime
import uuid
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel

logger = logging.getLogger(__name__)

@dataclass
class AgentContext:
    """Complete context for an isolated agent"""
    agent_id: str
    thread_id: str
    task_description: str
    full_context: Dict[str, Any]
    tools: List[Callable]
    created_at: datetime = field(default_factory=datetime.now)

class AgentState(MessagesState):
    """Extended state for context-isolated agents"""
    agent_id: str
    context: Dict[str, Any]
    task_status: str = "pending"
    result: Optional[Any] = None

class ContextIsolatedAgent:
    """
    LangGraph agent that runs in complete isolation with its own context window.
    Implements proper delegation patterns and clean handoffs.
    """

    def __init__(self, agent_type: str, capabilities: List[str]):
        self.agent_type = agent_type
        self.capabilities = capabilities
        self.graph: Optional[StateGraph] = None
        self.compiled_graph = None

    def create_isolated_context(self, task: str, context: Dict[str, Any]) -> AgentContext:
        """Create completely isolated context for agent execution"""
        agent_id = f"{self.agent_type}_{uuid.uuid4().hex[:8]}"
        thread_id = f"thread_{agent_id}"

        # Complete context - agent gets everything it needs
        full_context = {
            "task": task,
            "agent_type": self.agent_type,
            "capabilities": self.capabilities,
            "knowledge_base": context.get("knowledge_base", {}),
            "previous_results": context.get("previous_results", []),
            "user_preferences": context.get("user_preferences", {}),
            "system_state": context.get("system_state", {}),
        }

        return AgentContext(
            agent_id=agent_id,
            thread_id=thread_id,
            task_description=task,
            full_context=full_context,
            tools=context.get("tools", [])
        )

    def build_graph(self) -> StateGraph:
        """Build LangGraph with proper state management and conditional edges"""

        # Create workflow graph
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("process_task", self._process_task)
        workflow.add_node("validate_result", self._validate_result)
        workflow.add_node("prepare_handoff", self._prepare_handoff)

        # Add conditional edges
        workflow.add_conditional_edges(
            "process_task",
            self._should_validate,
            {
                "validate": "validate_result",
                "complete": "prepare_handoff"
            }
        )

        workflow.add_edge("validate_result", "prepare_handoff")
        workflow.set_entry_point("process_task")

        return workflow

    async def _process_task(self, state: AgentState) -> Dict[str, Any]:
        """Process task with complete context isolation"""
        try:
            logger.info(f"Agent {state.agent_id} processing task with isolated context")

            # Agent has complete context - no external dependencies
            context = state.context
            task = context["task"]

            # Simulate task processing with consciousness features
            result = await self._execute_with_consciousness(task, context)

            return {
                "task_status": "processed",
                "result": result,
                "messages": state.messages + [
                    AIMessage(content=f"Task processed by {self.agent_type}: {result}")
                ]
            }

        except Exception as e:
            logger.error(f"Task processing failed: {e}")
            return {
                "task_status": "failed",
                "result": {"error": str(e)},
                "messages": state.messages + [
                    AIMessage(content=f"Task failed: {str(e)}")
                ]
            }

    async def _execute_with_consciousness(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task with consciousness features (ThoughtSeed, Active Inference)"""
        # This would integrate with ThoughtSeed and active inference
        # For now, return structured result
        return {
            "task": task,
            "agent_type": self.agent_type,
            "consciousness_trace": {
                "thoughtseed_activation": True,
                "attractor_basin": "task_processing",
                "active_inference_state": "prediction_error_minimization"
            },
            "result_data": f"Processed: {task}",
            "handoff_ready": True
        }

    async def _validate_result(self, state: AgentState) -> Dict[str, Any]:
        """Validate processing result"""
        result = state.result

        if result and result.get("handoff_ready"):
            return {
                "task_status": "validated",
                "messages": state.messages + [
                    AIMessage(content="Result validated successfully")
                ]
            }
        else:
            return {
                "task_status": "validation_failed",
                "messages": state.messages + [
                    AIMessage(content="Result validation failed")
                ]
            }

    async def _prepare_handoff(self, state: AgentState) -> Dict[str, Any]:
        """Prepare clean handoff to next agent (no context rot)"""

        # Clean handoff data - only essential information
        handoff_data = {
            "from_agent": state.agent_id,
            "task_completed": state.context["task"],
            "result_summary": state.result.get("result_data") if state.result else None,
            "next_actions": [],  # Determined by conditional edges
            "context_preserved": True
        }

        return {
            "task_status": "ready_for_handoff",
            "result": handoff_data,
            "messages": state.messages + [
                AIMessage(content=f"Ready for clean handoff: {handoff_data}")
            ]
        }

    def _should_validate(self, state: AgentState) -> str:
        """Conditional edge: determine if validation needed"""
        if state.task_status == "processed" and state.result:
            return "validate"
        return "complete"

    def compile_graph(self):
        """Compile the workflow graph"""
        if not self.graph:
            self.graph = self.build_graph()
        self.compiled_graph = self.graph.compile()

    async def execute(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agent with complete context isolation"""

        # Create isolated context
        agent_context = self.create_isolated_context(task, context)

        # Compile graph if needed
        if not self.compiled_graph:
            self.compile_graph()

        # Initialize state
        initial_state = AgentState(
            messages=[HumanMessage(content=task)],
            agent_id=agent_context.agent_id,
            context=agent_context.full_context
        )

        # Execute with thread isolation
        result = await self.compiled_graph.ainvoke(
            initial_state,
            config={"thread_id": agent_context.thread_id}
        )

        logger.info(f"Agent {agent_context.agent_id} completed execution")
        return result

# Factory functions for different agent types
def create_consciousness_agent() -> ContextIsolatedAgent:
    """Create agent for consciousness processing"""
    return ContextIsolatedAgent(
        agent_type="consciousness_processor",
        capabilities=["thoughtseed_activation", "active_inference", "episodic_memory"]
    )

def create_perceptual_agent() -> ContextIsolatedAgent:
    """Create agent for perceptual processing"""
    return ContextIsolatedAgent(
        agent_type="perceptual_processor",
        capabilities=["video_processing", "audio_processing", "image_processing"]
    )

def create_curiosity_agent() -> ContextIsolatedAgent:
    """Create agent for curiosity-driven discovery"""
    return ContextIsolatedAgent(
        agent_type="curiosity_processor",
        capabilities=["knowledge_gap_detection", "web_crawling", "source_validation"]
    )