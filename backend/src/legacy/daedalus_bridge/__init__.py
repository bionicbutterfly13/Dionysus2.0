"""
Daedalus Bridge - Protected Integration Layer
============================================

This module provides crash-proof isolation for Daedalus legacy components.
Uses context isolation patterns and binary packaging where needed.

Key Features:
- Context window isolation for each agent
- Crash-proof wrapper with fallback mechanisms
- Progressive database downloads (Neo4j, Qdrant)
- LangGraph delegation patterns with clean handoffs
- NEMORI integration with legacy compatibility
"""

from .protected_wrapper import ProtectedDaedalusWrapper
from .context_isolator import ContextIsolatedAgent
from .legacy_bridge import LegacyBridge

__all__ = [
    'ProtectedDaedalusWrapper',
    'ContextIsolatedAgent',
    'LegacyBridge'
]

__version__ = "1.0.0"