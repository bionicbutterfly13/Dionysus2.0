# Daedalus Enhancement: Agent Coordination & Memory Integration

**Feature Branch**: `017-daedalus-enhancement`  
**Created**: 2025-09-27  
**Status**: Critical Enhancement  
**Purpose**: Enhance Daedalus with agent feedback coordination, expert agent creation, and optimal memory system integration

## Overview

Enhancement specification for Daedalus to implement Professor Synapse-style expert agent creation, agent feedback coordination, LangGraph state management, and optimal memory system selection (LangMem vs MEM0 vs Context Engineering).

## Current Daedalus Analysis

### ✅ What Daedalus Currently Has
- **Professor Synapse Template Integration** (line 141 in unified_daedalus.py)
- **LangGraph Integration** (107+ files found with LangGraph)
- **Constitutional Agent Framework** with validation
- **Redis Coordination** for agent communication
- **Multiple Delegation Strategies** (constitutional, memory_informed, parallel_spawning, conversational)
- **Container Use Integration** for isolated agent execution

### ❌ What's Missing
- **Agent Feedback Loop** - Daedalus doesn't get feedback from agents or coordinate when they need expertise
- **Dynamic Expert Agent Creation** - Not using Professor Synapse patterns for specialized expertise
- **LangGraph State Management** - Not leveraging LangGraph states, tools, and coordination patterns
- **Optimal Memory System** - No clear integration with best memory system (LangMem/MEM0/Context Engineering)

## Research Foundation

### Existing LangGraph Integration
Found comprehensive LangGraph integration:
- **LangGraphDaedalusSystem** with StateGraph orchestration
- **Agent spawning integration** via ExecutiveAssistant
- **Knowledge allocation** and delegation coordination
- **Three StateGraphs**: knowledge_allocator, delegation_coordinator, specialist_template

### Memory System Analysis
**LangMem**: Simple in-memory with semantic/episodic/procedural types
**MEM0**: Production-ready vector+graph storage with LLM integration  
**Context Engineering**: Advanced attractor basins and consciousness integration

## Core Requirements

### FR-001: Agent Feedback Coordination System
- System MUST implement feedback loop where agents can request expert assistance
- System MUST enable Daedalus to coordinate multi-agent problem solving
- System MUST detect when agents are blocked and need additional expertise
- System MUST route requests to appropriate expert agents
- System MUST maintain conversation state during agent handoffs

### FR-002: Professor Synapse Expert Agent Creation
- System MUST implement Professor Synapse expert agent patterns
- System MUST create specialized expert agents for different domains
- System MUST enable expert agents to delegate to sub-specialists
- System MUST maintain expert agent registry with capabilities
- System MUST support dynamic expert agent instantiation based on need

### FR-003: Enhanced LangGraph State Management
- System MUST leverage LangGraph StateGraph for agent coordination
- System MUST implement proper state transitions with checkpoints
- System MUST use LangGraph tools for agent communication
- System MUST maintain agent state persistence across interactions
- System MUST enable state inspection and debugging

### FR-004: Optimal Memory System Integration
- System MUST evaluate and select best memory system for Flux CE
- System MUST integrate chosen memory system with agent coordination
- System MUST support episodic, semantic, procedural, and working memory types
- System MUST enable memory sharing between agents
- System MUST maintain memory consistency during agent coordination

### FR-005: Agent Expertise Detection and Routing
- System MUST detect when agents need expertise outside their domain
- System MUST route expertise requests to appropriate specialists
- System MUST create new expert agents when existing ones are insufficient
- System MUST maintain expertise mapping and capability tracking
- System MUST enable agent-to-agent consultation and collaboration

## Technical Implementation

### Enhanced Daedalus Architecture

```python
# /Volumes/Asylum/dev/Dionysus-2.0/backend/services/enhanced_daedalus/

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from typing import Dict, List, Any, Optional, TypedDict
from enum import Enum

class AgentFeedbackType(Enum):
    EXPERTISE_REQUEST = "expertise_request"
    COLLABORATION_REQUEST = "collaboration_request"
    RESOURCE_REQUEST = "resource_request"
    ESCALATION_REQUEST = "escalation_request"
    COMPLETION_REPORT = "completion_report"

class AgentState(TypedDict):
    agent_id: str
    task_description: str
    current_status: str
    expertise_needed: Optional[List[str]]
    resources_needed: Optional[List[str]]
    collaboration_agents: Optional[List[str]]
    progress_report: Optional[Dict[str, Any]]
    memory_context: Optional[Dict[str, Any]]
    langgraph_checkpoint: Optional[str]

class EnhancedDaedalus:
    """Enhanced Daedalus with agent feedback coordination"""
    
    def __init__(self):
        self.base_daedalus = UnifiedDaedalus()
        self.expert_registry = ExpertAgentRegistry()
        self.memory_system = self._select_optimal_memory_system()
        self.coordination_graph = self._create_coordination_graph()
        self.feedback_processor = AgentFeedbackProcessor()
        
    def _create_coordination_graph(self) -> StateGraph:
        """Create LangGraph for agent coordination"""
        
        workflow = StateGraph(AgentState)
        
        # Add nodes for different coordination patterns
        workflow.add_node("analyze_request", self._analyze_agent_request)
        workflow.add_node("route_expertise", self._route_to_expert)
        workflow.add_node("create_expert", self._create_expert_agent)
        workflow.add_node("coordinate_collaboration", self._coordinate_agents)
        workflow.add_node("monitor_progress", self._monitor_agent_progress)
        workflow.add_node("escalate_issue", self._escalate_complex_issue)
        
        # Define edges and conditions
        workflow.add_edge(START, "analyze_request")
        workflow.add_conditional_edges(
            "analyze_request",
            self._determine_coordination_path,
            {
                "route_expertise": "route_expertise",
                "create_expert": "create_expert", 
                "coordinate_collaboration": "coordinate_collaboration",
                "monitor_progress": "monitor_progress"
            }
        )
        
        workflow.add_edge("route_expertise", "monitor_progress")
        workflow.add_edge("create_expert", "coordinate_collaboration")
        workflow.add_edge("coordinate_collaboration", "monitor_progress")
        workflow.add_conditional_edges(
            "monitor_progress",
            self._check_completion_status,
            {
                "continue": "monitor_progress",
                "escalate": "escalate_issue",
                "complete": END
            }
        )
        
        # Compile with memory checkpoints
        memory_saver = MemorySaver()
        return workflow.compile(checkpointer=memory_saver)
    
    async def receive_agent_feedback(self, 
                                   agent_id: str,
                                   feedback_type: AgentFeedbackType,
                                   feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process feedback from active agents"""
        
        # Create state for coordination graph
        state = AgentState(
            agent_id=agent_id,
            task_description=feedback_data.get("task_description", ""),
            current_status=feedback_data.get("status", "requesting_help"),
            expertise_needed=feedback_data.get("expertise_needed", []),
            resources_needed=feedback_data.get("resources_needed", []),
            collaboration_agents=feedback_data.get("collaboration_agents", []),
            progress_report=feedback_data.get("progress_report", {}),
            memory_context=await self._get_agent_memory_context(agent_id)
        )
        
        # Process through coordination graph
        config = {"configurable": {"thread_id": f"agent_coordination_{agent_id}"}}
        result = await self.coordination_graph.ainvoke(state, config)
        
        return result
    
    async def _analyze_agent_request(self, state: AgentState) -> AgentState:
        """Analyze agent request to determine coordination strategy"""
        
        # Analyze what kind of help the agent needs
        analysis = await self.feedback_processor.analyze_request(
            agent_id=state["agent_id"],
            expertise_needed=state["expertise_needed"],
            current_task=state["task_description"],
            progress=state["progress_report"]
        )
        
        # Update state with analysis
        state["coordination_strategy"] = analysis["strategy"]
        state["complexity_level"] = analysis["complexity"]
        state["recommended_experts"] = analysis["experts"]
        
        return state
    
    async def _route_to_expert(self, state: AgentState) -> AgentState:
        """Route request to existing expert agent"""
        
        expert_id = await self.expert_registry.find_best_expert(
            expertise_areas=state["expertise_needed"],
            context=state["memory_context"]
        )
        
        if expert_id:
            # Delegate to existing expert
            delegation_result = await self.base_daedalus.delegate_task({
                "task": f"Assist agent {state['agent_id']} with expertise request",
                "expertise_areas": state["expertise_needed"],
                "original_task": state["task_description"],
                "requesting_agent": state["agent_id"]
            }, target_agent=expert_id)
            
            state["assigned_expert"] = expert_id
            state["delegation_result"] = delegation_result
        else:
            # No suitable expert found, need to create one
            state["needs_new_expert"] = True
            
        return state
    
    async def _create_expert_agent(self, state: AgentState) -> AgentState:
        """Create new expert agent using Professor Synapse patterns"""
        
        # Use Professor Synapse template for expert creation
        expert_spec = await self._generate_expert_specification(
            expertise_areas=state["expertise_needed"],
            context=state["memory_context"],
            requesting_agent=state["agent_id"]
        )
        
        # Create expert agent
        expert_agent = await self.base_daedalus.delegate_task(
            expert_spec, 
            strategy="constitutional"
        )
        
        # Register new expert
        await self.expert_registry.register_expert(
            expert_id=expert_agent[0].agent_id,
            expertise_areas=state["expertise_needed"],
            capabilities=expert_spec["capabilities"],
            creation_context=state
        )
        
        state["created_expert"] = expert_agent[0].agent_id
        state["expert_specification"] = expert_spec
        
        return state

class ExpertAgentRegistry:
    """Registry of expert agents with capabilities tracking"""
    
    def __init__(self):
        self.experts: Dict[str, Dict[str, Any]] = {}
        self.expertise_map: Dict[str, List[str]] = {}
        self.performance_metrics: Dict[str, Dict[str, float]] = {}
        
    async def register_expert(self,
                            expert_id: str,
                            expertise_areas: List[str],
                            capabilities: List[str],
                            creation_context: Dict[str, Any]) -> None:
        """Register new expert agent"""
        
        self.experts[expert_id] = {
            "expertise_areas": expertise_areas,
            "capabilities": capabilities,
            "creation_context": creation_context,
            "created_at": datetime.now(),
            "usage_count": 0,
            "success_rate": 0.0
        }
        
        # Update expertise mapping
        for area in expertise_areas:
            if area not in self.expertise_map:
                self.expertise_map[area] = []
            self.expertise_map[area].append(expert_id)
    
    async def find_best_expert(self,
                             expertise_areas: List[str],
                             context: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Find best expert for given expertise areas"""
        
        candidates = []
        for area in expertise_areas:
            if area in self.expertise_map:
                candidates.extend(self.expertise_map[area])
        
        if not candidates:
            return None
            
        # Score candidates based on expertise match and performance
        best_expert = None
        best_score = 0.0
        
        for expert_id in set(candidates):
            expert_info = self.experts[expert_id]
            
            # Calculate expertise match score
            expertise_match = len(set(expertise_areas) & set(expert_info["expertise_areas"]))
            match_score = expertise_match / len(expertise_areas)
            
            # Factor in performance metrics
            performance_score = expert_info.get("success_rate", 0.5)
            
            # Combined score
            total_score = (match_score * 0.7) + (performance_score * 0.3)
            
            if total_score > best_score:
                best_score = total_score
                best_expert = expert_id
                
        return best_expert

class MemorySystemSelector:
    """Evaluates and selects optimal memory system"""
    
    @staticmethod
    async def evaluate_memory_systems() -> Dict[str, Any]:
        """Evaluate available memory systems"""
        
        evaluations = {
            "langmem": {
                "pros": [
                    "Simple implementation",
                    "Direct memory type support (semantic/episodic/procedural)", 
                    "Low overhead",
                    "Fast in-memory operations"
                ],
                "cons": [
                    "No persistence",
                    "Limited search capabilities",
                    "No vector embeddings",
                    "No relationship modeling"
                ],
                "score": 6.0,
                "use_case": "Simple applications with basic memory needs"
            },
            "mem0": {
                "pros": [
                    "Production-ready",
                    "Vector + graph storage", 
                    "LLM integration",
                    "Persistent storage",
                    "Advanced search capabilities"
                ],
                "cons": [
                    "External dependency",
                    "More complex setup",
                    "Higher resource usage",
                    "Learning curve"
                ],
                "score": 8.5,
                "use_case": "Production applications with complex memory requirements"
            },
            "context_engineering": {
                "pros": [
                    "Consciousness integration",
                    "Attractor basin support",
                    "Advanced cognitive features",
                    "Research-validated approaches",
                    "Custom memory architectures"
                ],
                "cons": [
                    "Complex implementation",
                    "Research-stage features", 
                    "Higher cognitive overhead",
                    "Specialized use cases"
                ],
                "score": 9.0,
                "use_case": "Advanced consciousness applications with research requirements"
            }
        }
        
        # Recommendation based on Flux CE requirements
        recommendation = "context_engineering"
        reasoning = """
        For Flux CE consciousness emulator with narrative extraction, archetypal patterns,
        and IFS integration, Context Engineering provides the best fit because:
        
        1. Consciousness Integration: Native support for consciousness processing
        2. Attractor Basins: Essential for ThoughtSeed coordination
        3. Research Foundation: Aligns with active inference and narrative research
        4. Advanced Features: Supports complex cognitive architectures
        5. Extensibility: Can integrate with other memory systems as needed
        """
        
        return {
            "evaluations": evaluations,
            "recommendation": recommendation,
            "reasoning": reasoning,
            "hybrid_approach": "Use Context Engineering as primary with MEM0 for persistence"
        }
```

### LangGraph State Integration

```python
# Enhanced LangGraph state management for agent coordination

class DaedalusCoordinationState(TypedDict):
    """Standardized state for Daedalus coordination workflows"""
    
    # Core identification
    coordination_id: str
    requesting_agent_id: str
    target_agent_id: Optional[str]
    
    # Request details
    request_type: AgentFeedbackType
    task_description: str
    expertise_needed: List[str]
    priority_level: int
    
    # Coordination state
    coordination_status: str  # "analyzing", "routing", "coordinating", "completed"
    assigned_experts: List[str]
    active_collaborations: List[str]
    
    # Memory and context
    memory_context: Dict[str, Any]
    conversation_history: List[Dict[str, Any]]
    shared_resources: List[str]
    
    # Progress tracking
    progress_metrics: Dict[str, float]
    completion_criteria: List[str]
    success_indicators: List[str]
    
    # LangGraph metadata
    checkpoint_id: Optional[str]
    state_version: int
    last_updated: str
```

## Agent Delegation Tasks

### Task 1: Daedalus Feedback Coordination Agent
**Specialization**: Agent coordination and feedback processing  
**Deliverables**:
- Implement agent feedback loop system
- Create expertise request routing
- Build multi-agent coordination patterns
- Enable dynamic agent collaboration

### Task 2: Expert Agent Creation Agent  
**Specialization**: Professor Synapse pattern implementation
**Deliverables**:
- Implement Professor Synapse expert agent patterns
- Create expert agent registry and capability tracking
- Build dynamic expert agent instantiation
- Enable expert agent performance monitoring

### Task 3: LangGraph Enhancement Agent
**Specialization**: LangGraph state management and coordination
**Deliverables**:
- Implement enhanced LangGraph state management
- Create standardized state transitions
- Build checkpoint and persistence system
- Enable state inspection and debugging

### Task 4: Memory System Integration Agent
**Specialization**: Memory system evaluation and integration
**Deliverables**:
- Evaluate memory systems (LangMem vs MEM0 vs Context Engineering)
- Implement optimal memory system integration
- Create memory sharing between agents
- Build hybrid memory architecture if needed

## Success Metrics

### Coordination Effectiveness
- Agent feedback response time: <10 seconds
- Expert routing accuracy: >90%
- Multi-agent collaboration success rate: >85%
- Issue escalation resolution time: <5 minutes

### Expert Agent Performance  
- Expert creation time: <30 seconds
- Expert capability match accuracy: >95%
- Expert reuse efficiency: >70%
- Expert performance tracking accuracy: >90%

### LangGraph Integration
- State transition reliability: >99%
- Checkpoint persistence success: >99%
- State inspection availability: 100%
- Memory consistency across states: >95%

## Implementation Priority

### Phase 1: Core Coordination (Week 1)
1. Implement agent feedback coordination system
2. Create expert agent registry and routing
3. Build basic LangGraph state management
4. Set up memory system evaluation

### Phase 2: Advanced Features (Week 2)  
1. Implement Professor Synapse expert creation
2. Create dynamic multi-agent coordination
3. Build enhanced LangGraph workflows
4. Integrate optimal memory system

### Phase 3: Optimization (Week 3)
1. Optimize coordination performance
2. Implement advanced state management
3. Create monitoring and debugging tools
4. Build hybrid memory architecture

## Updated Clarification Questions

**21. Agent Coordination Preferences**
- Should agents be able to directly communicate or only through Daedalus?
- How should conflicting expertise requests be prioritized?

**22. Expert Agent Lifecycle**  
- Should expert agents be persistent or created on-demand?
- How should expert agent performance be measured and optimized?

**23. Memory System Selection**
- Prefer Context Engineering (consciousness) or MEM0 (production-ready)?
- Should we implement hybrid memory approach with multiple systems?

**24. LangGraph Usage Depth**
- Should we use full LangGraph features (tools, checkpoints, etc.) or basic state management?
- How should LangGraph state be synchronized with agent memory?

---

**This enhancement will make Daedalus a true agent coordination orchestrator with feedback loops, expert creation, and optimal memory integration!**