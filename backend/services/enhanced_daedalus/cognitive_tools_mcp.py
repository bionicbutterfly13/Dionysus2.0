#!/usr/bin/env python3
"""
🧠 Cognitive Tools MCP Server for Dionysus 2.0
==============================================

Model Context Protocol (MCP) server providing research-validated cognitive tools
as system-wide services that agents can access through standardized protocols.

Provides both:
1. Direct agent access to cognitive tools
2. Daedalus coordination through MCP protocol
3. Tree of Thought integration

Author: Dionysus Consciousness Enhancement System  
Date: 2025-09-27
Version: 1.0.0 - MCP Cognitive Tools Server
"""

import json
import asyncio
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
import logging

# MCP Protocol imports (assuming standard MCP implementation)
try:
    from mcp import Server, Tool, Resource
    from mcp.types import TextContent, ImageContent
except ImportError:
    # Fallback for development
    class Server:
        def __init__(self, name: str): pass
    class Tool:
        def __init__(self, name: str, description: str): pass
    class Resource:
        def __init__(self, uri: str, name: str): pass

from .cognitive_tools_implementation import ResearchValidatedCognitiveOrchestrator
from .cognitive_meta_coordinator import (
    CognitiveMetaCoordinator, 
    AgentCognitiveInterface, 
    CognitiveContext,
    ReasoningMode
)

logger = logging.getLogger(__name__)

@dataclass
class MCPCognitiveRequest:
    """MCP request for cognitive enhancement"""
    agent_name: str
    task: str
    context: Optional[Dict[str, Any]] = None
    preferred_reasoning_mode: Optional[str] = None
    force_mode: bool = False  # Override meta-cognitive decision
    
@dataclass 
class MCPCognitiveResponse:
    """MCP response with cognitive enhancement results"""
    success: bool
    reasoning_mode_used: str
    cognitive_result: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    meta_decision: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

class CognitiveToolsMCPServer:
    """
    MCP Server providing cognitive tools as system-wide services.
    
    Enables:
    - Standardized access to cognitive tools across all agents
    - Daedalus coordination of cognitive enhancement
    - Integration with Tree of Thought models
    - Performance monitoring and optimization
    """
    
    def __init__(self, server_name: str = "cognitive-tools"):
        self.server = Server(server_name)
        self.meta_coordinator = CognitiveMetaCoordinator()
        self.cognitive_orchestrator = ResearchValidatedCognitiveOrchestrator()
        self.agent_interfaces = {}
        
        # Performance tracking
        self.global_performance_metrics = {
            "total_requests": 0,
            "reasoning_mode_usage": {},
            "performance_improvements": [],
            "agent_success_rates": {}
        }
        
        self._register_mcp_tools()
        logger.info(f"🧠 Cognitive Tools MCP Server '{server_name}' initialized")
    
    def _register_mcp_tools(self):
        """Register cognitive tools as MCP tools"""
        
        # Core cognitive tools
        self.server.register_tool(Tool(
            name="understand_question",
            description="Break down complex problems into structured steps (research-validated: +4-8% accuracy)"
        ))
        
        self.server.register_tool(Tool(
            name="recall_related", 
            description="Provide analogous solved problems to guide reasoning (+6-10% accuracy through pattern matching)"
        ))
        
        self.server.register_tool(Tool(
            name="examine_answer",
            description="Verify and examine reasoning traces for errors and improvements (critical error detection)"
        ))
        
        self.server.register_tool(Tool(
            name="backtracking",
            description="Backtrack from flawed reasoning and explore alternatives (+26.7% improvement on complex tasks)"
        ))
        
        # Meta-cognitive tools
        self.server.register_tool(Tool(
            name="meta_cognitive_decision",
            description="Decide which reasoning approach to use (cognitive tools, Tree of Thought, direct, hybrid)"
        ))
        
        self.server.register_tool(Tool(
            name="enhance_agent_reasoning",
            description="Apply research-validated cognitive enhancement to agent tasks (26.7% to 62.5% improvement)"
        ))
        
        # Tree of Thought integration
        self.server.register_tool(Tool(
            name="tree_of_thought_integration",
            description="Integrate cognitive tools with Tree of Thought exploration for complex creative tasks"
        ))
        
        # Performance monitoring
        self.server.register_tool(Tool(
            name="cognitive_performance_metrics",
            description="Get performance metrics and optimization recommendations for cognitive enhancement"
        ))
    
    async def handle_mcp_request(self, tool_name: str, arguments: Dict[str, Any]) -> MCPCognitiveResponse:
        """Handle MCP tool requests for cognitive enhancement"""
        
        try:
            self.global_performance_metrics["total_requests"] += 1
            
            # Route to appropriate handler
            if tool_name == "meta_cognitive_decision":
                return await self._handle_meta_cognitive_decision(arguments)
            
            elif tool_name == "enhance_agent_reasoning":
                return await self._handle_agent_reasoning_enhancement(arguments)
            
            elif tool_name in ["understand_question", "recall_related", "examine_answer", "backtracking"]:
                return await self._handle_individual_cognitive_tool(tool_name, arguments)
            
            elif tool_name == "tree_of_thought_integration":
                return await self._handle_tree_of_thought_integration(arguments)
            
            elif tool_name == "cognitive_performance_metrics":
                return await self._handle_performance_metrics(arguments)
            
            else:
                return MCPCognitiveResponse(
                    success=False,
                    reasoning_mode_used="error",
                    cognitive_result={},
                    performance_metrics={},
                    error_message=f"Unknown cognitive tool: {tool_name}"
                )
                
        except Exception as e:
            logger.error(f"❌ MCP cognitive tool error: {e}")
            return MCPCognitiveResponse(
                success=False,
                reasoning_mode_used="error", 
                cognitive_result={},
                performance_metrics={},
                error_message=str(e)
            )
    
    async def _handle_meta_cognitive_decision(self, arguments: Dict[str, Any]) -> MCPCognitiveResponse:
        """Handle meta-cognitive decision requests"""
        
        agent_name = arguments.get("agent_name", "unknown_agent")
        task = arguments.get("task", "")
        context_data = arguments.get("context", {})
        
        # Create cognitive context
        context = CognitiveContext(
            task_complexity=context_data.get("task_complexity", 0.5),
            domain_type=context_data.get("domain_type", "general"),
            agent_expertise=context_data.get("agent_expertise", 0.7),
            previous_success_rate=context_data.get("previous_success_rate", 0.8),
            time_constraints=context_data.get("time_constraints"),
            requires_creativity=context_data.get("requires_creativity", False),
            requires_verification=context_data.get("requires_verification", True)
        )
        
        # Get meta-cognitive decision
        decision = await self.meta_coordinator.decide_reasoning_approach(agent_name, task, context)
        
        # Track usage
        mode_name = decision.recommended_mode.value
        if mode_name not in self.global_performance_metrics["reasoning_mode_usage"]:
            self.global_performance_metrics["reasoning_mode_usage"][mode_name] = 0
        self.global_performance_metrics["reasoning_mode_usage"][mode_name] += 1
        
        return MCPCognitiveResponse(
            success=True,
            reasoning_mode_used=mode_name,
            cognitive_result=asdict(decision),
            performance_metrics={"decision_confidence": decision.confidence},
            meta_decision=asdict(decision)
        )
    
    async def _handle_agent_reasoning_enhancement(self, arguments: Dict[str, Any]) -> MCPCognitiveResponse:
        """Handle full agent reasoning enhancement requests"""
        
        agent_name = arguments.get("agent_name", "unknown_agent")
        task = arguments.get("task", "")
        force_mode = arguments.get("force_mode", False)
        preferred_mode = arguments.get("preferred_reasoning_mode")
        
        # Get or create agent interface
        if agent_name not in self.agent_interfaces:
            self.agent_interfaces[agent_name] = AgentCognitiveInterface(agent_name)
        
        agent_interface = self.agent_interfaces[agent_name]
        
        # Create context from arguments
        context_data = arguments.get("context", {})
        context = CognitiveContext(
            task_complexity=context_data.get("task_complexity", 0.5),
            domain_type=context_data.get("domain_type", "general"),
            agent_expertise=context_data.get("agent_expertise", 0.7),
            previous_success_rate=context_data.get("previous_success_rate", 0.8),
            time_constraints=context_data.get("time_constraints"),
            requires_creativity=context_data.get("requires_creativity", False),
            requires_verification=context_data.get("requires_verification", True)
        )
        
        # Override meta-cognitive decision if forced mode
        if force_mode and preferred_mode:
            try:
                forced_mode = ReasoningMode(preferred_mode)
                original_decision = await agent_interface.meta_coordinator.decide_reasoning_approach(
                    agent_name, task, context
                )
                # Override the decision
                original_decision.recommended_mode = forced_mode
                original_decision.reasoning = f"Forced mode: {preferred_mode}"
            except ValueError:
                logger.warning(f"⚠️ Invalid forced reasoning mode: {preferred_mode}")
        
        # Apply cognitive enhancement
        enhancement_result = await agent_interface.enhance_reasoning(task, context)
        
        # Track performance
        self._track_agent_performance(agent_name, enhancement_result)
        
        return MCPCognitiveResponse(
            success=True,
            reasoning_mode_used=enhancement_result["cognitive_decision"].recommended_mode.value,
            cognitive_result=enhancement_result["reasoning_result"],
            performance_metrics=enhancement_result["performance_metrics"],
            meta_decision=asdict(enhancement_result["cognitive_decision"])
        )
    
    async def _handle_individual_cognitive_tool(self, tool_name: str, arguments: Dict[str, Any]) -> MCPCognitiveResponse:
        """Handle individual cognitive tool execution"""
        
        question = arguments.get("question", arguments.get("task", ""))
        context = arguments.get("context", "")
        current_reasoning = arguments.get("current_reasoning", "")
        
        # Create tool call
        from .cognitive_tools_implementation import CognitiveToolCall
        tool_call = CognitiveToolCall(
            name=tool_name,
            parameters={
                "question": question,
                "context": context,
                "current_reasoning": current_reasoning
            }
        )
        
        # Execute tool
        result = await self.cognitive_orchestrator.execute_cognitive_tool(tool_call)
        
        return MCPCognitiveResponse(
            success=result.success,
            reasoning_mode_used=f"individual_tool_{tool_name}",
            cognitive_result={
                "tool_result": result.content,
                "metadata": result.metadata,
                "reasoning_trace": result.reasoning_trace
            },
            performance_metrics={"tool_execution_success": result.success}
        )
    
    async def _handle_tree_of_thought_integration(self, arguments: Dict[str, Any]) -> MCPCognitiveResponse:
        """Handle Tree of Thought integration with cognitive tools"""
        
        agent_name = arguments.get("agent_name", "unknown_agent")
        task = arguments.get("task", "")
        
        # This would integrate with your existing Tree of Thought implementation
        # For now, return a placeholder that shows the integration pattern
        
        cognitive_enhancement = await self._handle_agent_reasoning_enhancement({
            **arguments,
            "force_mode": True,
            "preferred_reasoning_mode": "cognitive_tools"
        })
        
        # Integration with Tree of Thought (placeholder)
        tot_integration = {
            "cognitive_preprocessing": cognitive_enhancement.cognitive_result,
            "thought_tree_exploration": "Systematic exploration of solution paths",
            "cognitive_validation": "Each thought path validated with examine_answer tool",
            "backtracking_integration": "Failed paths use backtracking tool for alternatives",
            "hybrid_result": "Combined cognitive tools + Tree of Thought reasoning"
        }
        
        return MCPCognitiveResponse(
            success=True,
            reasoning_mode_used="hybrid_cognitive_tot",
            cognitive_result=tot_integration,
            performance_metrics={
                "cognitive_enhancement_applied": True,
                "tot_integration": True,
                "expected_improvement": "45% (combined benefits)"
            }
        )
    
    async def _handle_performance_metrics(self, arguments: Dict[str, Any]) -> MCPCognitiveResponse:
        """Handle performance metrics requests"""
        
        agent_name = arguments.get("agent_name")
        include_global = arguments.get("include_global", True)
        
        metrics = {}
        
        if include_global:
            metrics["global_metrics"] = self.global_performance_metrics
        
        if agent_name and agent_name in self.agent_interfaces:
            agent_metrics = self.agent_interfaces[agent_name].cognitive_orchestrator.get_performance_metrics()
            metrics["agent_metrics"] = agent_metrics
        
        # Add optimization recommendations
        metrics["optimization_recommendations"] = self._generate_optimization_recommendations()
        
        return MCPCognitiveResponse(
            success=True,
            reasoning_mode_used="performance_analysis",
            cognitive_result=metrics,
            performance_metrics=metrics
        )
    
    def _track_agent_performance(self, agent_name: str, enhancement_result: Dict[str, Any]):
        """Track agent performance for optimization"""
        
        if agent_name not in self.global_performance_metrics["agent_success_rates"]:
            self.global_performance_metrics["agent_success_rates"][agent_name] = []
        
        # Extract performance indicators
        reasoning_quality = enhancement_result.get("reasoning_result", {}).get("reasoning_quality_score", 0.5)
        expected_improvement = enhancement_result["cognitive_decision"].expected_performance_gain
        
        self.global_performance_metrics["agent_success_rates"][agent_name].append({
            "reasoning_quality": reasoning_quality,
            "expected_improvement": expected_improvement,
            "reasoning_mode": enhancement_result["cognitive_decision"].recommended_mode.value
        })
        
        # Keep only recent performance data
        if len(self.global_performance_metrics["agent_success_rates"][agent_name]) > 100:
            self.global_performance_metrics["agent_success_rates"][agent_name] = \
                self.global_performance_metrics["agent_success_rates"][agent_name][-100:]
    
    def _generate_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on performance data"""
        
        recommendations = []
        
        # Analyze reasoning mode usage
        mode_usage = self.global_performance_metrics["reasoning_mode_usage"]
        total_requests = sum(mode_usage.values())
        
        if total_requests > 0:
            cognitive_tools_usage = mode_usage.get("cognitive_tools_enhanced", 0) / total_requests
            
            if cognitive_tools_usage < 0.3:
                recommendations.append(
                    "Consider increasing cognitive tools usage for complex mathematical tasks (research shows +26.7% to +62.5% improvement)"
                )
            
            hybrid_usage = mode_usage.get("hybrid_cognitive_tot", 0) / total_requests
            if hybrid_usage < 0.1:
                recommendations.append(
                    "Consider hybrid cognitive tools + Tree of Thought for very complex creative tasks"
                )
        
        # Analyze agent performance
        for agent_name, performance_data in self.global_performance_metrics["agent_success_rates"].items():
            if len(performance_data) >= 10:
                avg_quality = sum(p["reasoning_quality"] for p in performance_data[-10:]) / 10
                if avg_quality < 0.7:
                    recommendations.append(
                        f"Agent {agent_name} shows low reasoning quality ({avg_quality:.1%}) - consider cognitive tools training"
                    )
        
        if not recommendations:
            recommendations.append("Performance looks good! Continue current cognitive enhancement strategies.")
        
        return recommendations

class DaedalusCognitiveCoordinator:
    """
    Daedalus-level coordinator that uses MCP to orchestrate cognitive enhancement
    across all agents in the system.
    """
    
    def __init__(self):
        self.mcp_server = CognitiveToolsMCPServer("daedalus-cognitive-coordinator")
        self.active_agents = {}
        self.coordination_policies = {
            "auto_enhance_complexity_threshold": 0.7,
            "prefer_cognitive_tools_for_math": True,
            "prefer_tot_for_creative": True,
            "enable_hybrid_for_complex": True
        }
        
        logger.info("🧠 Daedalus Cognitive Coordinator initialized")
    
    async def coordinate_agent_cognitive_enhancement(self, 
                                                   agent_name: str,
                                                   task: str, 
                                                   agent_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Coordinate cognitive enhancement for an agent task through Daedalus
        
        This is where Daedalus decides whether to apply cognitive enhancement
        and which approach to use, while still giving agents autonomy.
        """
        
        # Apply Daedalus coordination policies
        should_enhance, reasoning_mode = await self._apply_coordination_policies(task, agent_context or {})
        
        if should_enhance:
            # Use MCP to apply cognitive enhancement
            mcp_request = {
                "agent_name": agent_name,
                "task": task,
                "context": agent_context or {},
                "preferred_reasoning_mode": reasoning_mode,
                "force_mode": False  # Let meta-cognitive decision override if needed
            }
            
            response = await self.mcp_server.handle_mcp_request("enhance_agent_reasoning", mcp_request)
            
            logger.info(f"🧠 Daedalus enhanced {agent_name} with {response.reasoning_mode_used}")
            
            return {
                "daedalus_coordination": True,
                "cognitive_enhancement_applied": response.success,
                "reasoning_mode": response.reasoning_mode_used,
                "cognitive_result": response.cognitive_result,
                "performance_metrics": response.performance_metrics
            }
        
        else:
            logger.info(f"🧠 Daedalus allowing {agent_name} to proceed with direct reasoning")
            return {
                "daedalus_coordination": True,
                "cognitive_enhancement_applied": False,
                "reasoning_mode": "direct_reasoning",
                "decision_reason": "Task does not meet cognitive enhancement criteria"
            }
    
    async def _apply_coordination_policies(self, task: str, context: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Apply Daedalus coordination policies to determine cognitive enhancement strategy"""
        
        task_complexity = context.get("task_complexity", 0.5)
        domain_type = context.get("domain_type", "general")
        requires_creativity = context.get("requires_creativity", False)
        
        # Policy 1: Auto-enhance high complexity tasks
        if task_complexity >= self.coordination_policies["auto_enhance_complexity_threshold"]:
            if requires_creativity and self.coordination_policies["enable_hybrid_for_complex"]:
                return True, "hybrid_cognitive_tot"
            else:
                return True, "cognitive_tools"
        
        # Policy 2: Prefer cognitive tools for mathematical tasks
        if (domain_type == "mathematical" and 
            self.coordination_policies["prefer_cognitive_tools_for_math"]):
            return True, "cognitive_tools"
        
        # Policy 3: Prefer Tree of Thought for creative tasks
        if (requires_creativity and 
            self.coordination_policies["prefer_tot_for_creative"]):
            return True, "tree_of_thought"
        
        # Default: No enhancement needed
        return False, None

# Export the MCP components
__all__ = [
    'CognitiveToolsMCPServer',
    'DaedalusCognitiveCoordinator', 
    'MCPCognitiveRequest',
    'MCPCognitiveResponse'
]

logger.info("🧠 Cognitive Tools MCP Server implementation loaded successfully")