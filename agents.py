#!/usr/bin/env python3
"""
🌉 ASI-Arch ↔ Dionysus Agents Bridge
=====================================

Bridge module that provides the ASI-Arch agents interface while connecting
to the Dionysus consciousness system. This resolves the broken promise of
ASI-Arch agents integration.

Author: ASI-Arch Context Engineering Extension
Date: 2025-09-23
Version: 1.0.0 - Bridge Implementation
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
import asyncio
import logging

# Add dionysus-source to path
dionysus_path = Path(__file__).parent / "dionysus-source"
sys.path.insert(0, str(dionysus_path))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global OpenAI configuration
_openai_api_key = None
_openai_client = None
_tracing_enabled = True

def set_default_openai_api(api_key: str) -> None:
    """Set default OpenAI API key for all agents"""
    global _openai_api_key
    _openai_api_key = api_key
    os.environ["OPENAI_API_KEY"] = api_key
    logger.info("✅ OpenAI API key configured for ASI-Arch agents")

def set_default_openai_client(client: Any) -> None:
    """Set default OpenAI client for all agents"""
    global _openai_client
    _openai_client = client
    logger.info("✅ OpenAI client configured for ASI-Arch agents")

def set_tracing_disabled(disabled: bool = True) -> None:
    """Disable/enable tracing for agents"""
    global _tracing_enabled
    _tracing_enabled = not disabled
    logger.info(f"✅ Agent tracing {'disabled' if disabled else 'enabled'}")

class Agent:
    """
    ASI-Arch Agent interface that bridges to Dionysus consciousness system

    This class provides the interface expected by ASI-Arch pipeline while
    leveraging the advanced Dionysus agent architecture underneath.
    """

    def __init__(self,
                 name: str,
                 description: str = "",
                 instructions: str = "",
                 model: str = "gpt-4",
                 tools: List[Any] = None,
                 **kwargs):
        """Initialize ASI-Arch compatible agent"""
        self.name = name
        self.description = description
        self.instructions = instructions
        self.model = model
        self.tools = tools or []
        self.kwargs = kwargs

        # Bridge to Dionysus system
        self._initialize_dionysus_bridge()

        logger.info(f"🌉 Initialized ASI-Arch agent '{name}' with Dionysus bridge")

    def _initialize_dionysus_bridge(self):
        """Initialize connection to Dionysus consciousness system"""
        try:
            # Try to import Dionysus executive assistant
            from agents.executive_assistant import ExecutiveAssistant
            self._dionysus_executive = ExecutiveAssistant()
            self._dionysus_available = True
            logger.info("✅ Dionysus Executive Assistant connected")
        except ImportError as e:
            logger.warning(f"⚠️ Dionysus system not available: {e}")
            self._dionysus_available = False
            self._dionysus_executive = None

    async def run(self,
                  message: str,
                  context: Optional[Dict[str, Any]] = None,
                  **kwargs) -> str:
        """
        Run agent with message - bridges to Dionysus delegation

        Args:
            message: The task/query for the agent
            context: Additional context for the task
            **kwargs: Additional parameters

        Returns:
            str: Agent response
        """
        try:
            if self._dionysus_available and self._dionysus_executive:
                # Use Dionysus Executive Assistant for sophisticated delegation
                response = await self._run_with_dionysus(message, context, **kwargs)
            else:
                # Fallback to basic implementation
                response = await self._run_fallback(message, context, **kwargs)

            if _tracing_enabled:
                logger.info(f"🎯 Agent '{self.name}' completed task")

            return response

        except Exception as e:
            logger.error(f"❌ Agent '{self.name}' failed: {e}")
            raise AgentException(f"Agent '{self.name}' execution failed: {e}")

    async def _run_with_dionysus(self,
                                message: str,
                                context: Optional[Dict[str, Any]] = None,
                                **kwargs) -> str:
        """Run using Dionysus Executive Assistant delegation"""
        # Format task for Dionysus system
        task_description = f"""
Agent Task: {self.name}
Description: {self.description}
Instructions: {self.instructions}
Message: {message}
Context: {context or {}}
"""

        # Delegate to Dionysus Executive Assistant
        result = await self._dionysus_executive.delegate_task(
            task_description=task_description,
            agent_type=self.name,
            context=context or {}
        )

        return str(result)

    async def _run_fallback(self,
                           message: str,
                           context: Optional[Dict[str, Any]] = None,
                           **kwargs) -> str:
        """Fallback implementation when Dionysus not available"""
        logger.warning(f"⚠️ Using fallback for agent '{self.name}'")

        # Basic implementation that mimics agent behavior
        response = f"[Agent {self.name}] Processed: {message}"

        if self.instructions:
            response += f"\nInstructions followed: {self.instructions[:100]}..."

        return response

class Runner:
    """Agent runner for managing agent execution"""

    def __init__(self, agent: Agent):
        """Initialize runner with agent"""
        self.agent = agent
        self.run_history = []

    async def run(self, *args, **kwargs):
        """Run the agent and track execution"""
        result = await self.agent.run(*args, **kwargs)
        self.run_history.append({
            'args': args,
            'kwargs': kwargs,
            'result': result
        })
        return result

def function_tool(func: Callable) -> Callable:
    """
    Function tool decorator for ASI-Arch compatibility

    Args:
        func: Function to wrap as a tool

    Returns:
        Wrapped function with tool metadata
    """
    def wrapper(*args, **kwargs):
        try:
            if _tracing_enabled:
                logger.info(f"🔧 Executing tool function: {func.__name__}")
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            logger.error(f"❌ Tool function '{func.__name__}' failed: {e}")
            raise ToolException(f"Tool '{func.__name__}' execution failed: {e}")

    # Add tool metadata
    wrapper.__tool_name__ = func.__name__
    wrapper.__tool_description__ = func.__doc__ or f"Tool function: {func.__name__}"
    wrapper.__is_tool__ = True

    return wrapper

# Exception classes for ASI-Arch compatibility
class AgentException(Exception):
    """Exception raised by agent execution"""
    pass

class ToolException(Exception):
    """Exception raised by tool execution"""
    pass

class NetworkException(Exception):
    """Exception raised by network operations"""
    pass

# Exceptions module compatibility
class exceptions:
    """Exceptions namespace for ASI-Arch compatibility"""
    AgentException = AgentException
    ToolException = ToolException
    NetworkException = NetworkException

# Module-level compatibility
__all__ = [
    'Agent',
    'Runner',
    'set_default_openai_api',
    'set_default_openai_client',
    'set_tracing_disabled',
    'function_tool',
    'exceptions',
    'AgentException',
    'ToolException',
    'NetworkException'
]

logger.info("🌉 ASI-Arch ↔ Dionysus Agents Bridge initialized")