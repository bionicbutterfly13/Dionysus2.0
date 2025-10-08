#!/usr/bin/env python3
"""
Adapters module entry point.

Exposes the EnhancedMemoryAdapter that bridges the Dionysus perceptual
gateway with the enhanced episodic memory orchestrator.
"""

from .enhanced_memory_adapter import EnhancedMemoryAdapter, create_enhanced_memory_adapter

__all__ = [
    "EnhancedMemoryAdapter",
    "create_enhanced_memory_adapter",
]
