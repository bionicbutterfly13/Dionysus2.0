#!/usr/bin/env python3
"""
🔧 Interactive Debugger for Pattern Evolution System
==================================================

Provides breakpoint debugging, variable inspection, and step-through capabilities
for the complete pattern evolution system.

Features:
- Breakpoint setting at any function
- Variable state inspection at breakpoints
- Step-through execution with pause/resume
- Memory state snapshots
- Real-time data inspection

Usage: python interactive_debugger.py
"""

import pdb
import sys
import redis
import json
import asyncio
from typing import Dict, List, Any
sys.path.append('.')
from attractor_basin_dynamics import AttractorBasinManager
from knowledge_graph_evolution_storage import PermanentLearningGraph

class PatternEvolutionDebugger:
    """Interactive debugger with breakpoint support"""

    def __init__(self):
        self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.basin_manager = AttractorBasinManager()
        self.learning_graph = PermanentLearningGraph()
        self.breakpoints = set()
        self.watch_variables = {}

    def set_breakpoint(self, function_name: str, line_number: int = None):
        """Set a breakpoint at a specific function or line"""
        breakpoint_id = f"{function_name}:{line_number}" if line_number else function_name
        self.breakpoints.add(breakpoint_id)
        print(f"🔴 Breakpoint set: {breakpoint_id}")

    def watch_variable(self, var_name: str, var_value: Any):
        """Watch a variable for changes"""
        self.watch_variables[var_name] = var_value
        print(f"👁️  Watching variable: {var_name} = {var_value}")

    def inspect_redis_state(self, key_pattern: str = "*"):
        """Inspect current Redis state with breakpoint"""
        print(f"\n🔍 BREAKPOINT: Redis State Inspection")
        print(f"Key Pattern: {key_pattern}")

        keys = self.redis_client.keys(key_pattern)
        print(f"Found {len(keys)} keys")

        # Set debugger breakpoint here
        pdb.set_trace()  # BREAKPOINT: Inspect keys array

        for key in keys:
            data = self.redis_client.get(key)
            if data:
                try:
                    parsed_data = json.loads(data)
                    print(f"Key: {key}")
                    print(f"Data: {parsed_data}")
                except:
                    print(f"Key: {key} (non-JSON data)")

    async def debug_thoughtseed_integration(self, thoughtseed_id: str, concept: str):
        """Debug thoughtseed integration with breakpoints"""
        print(f"\n🔧 DEBUGGING: Thoughtseed Integration")
        print(f"ThoughtSeed ID: {thoughtseed_id}")
        print(f"Concept: {concept}")

        # BREAKPOINT 1: Pre-integration state
        pre_state = {
            'basin_count': len(self.basin_manager.basins),
            'thoughtseed_id': thoughtseed_id,
            'concept': concept
        }
        print(f"\n📊 Pre-integration state: {pre_state}")
        pdb.set_trace()  # BREAKPOINT: Examine pre-state

        # BREAKPOINT 2: Basin similarity calculation
        print(f"\n🧮 Calculating basin similarities...")
        similarities = {}
        for basin_id, basin in self.basin_manager.basins.items():
            similarity = self.basin_manager.calculate_concept_similarity(concept, basin.center_concept)
            similarities[basin_id] = similarity

        print(f"Similarities calculated: {similarities}")
        pdb.set_trace()  # BREAKPOINT: Examine similarities

        # BREAKPOINT 3: Integration execution
        print(f"\n⚡ Executing integration...")
        integration_event = await self.basin_manager.integrate_new_thoughtseed(
            thoughtseed_id, concept, {}
        )

        print(f"Integration event: {integration_event}")
        pdb.set_trace()  # BREAKPOINT: Examine integration result

        # BREAKPOINT 4: Post-integration state
        post_state = {
            'basin_count': len(self.basin_manager.basins),
            'event_id': integration_event.event_id,
            'influence_type': integration_event.influence_type.value
        }
        print(f"\n📈 Post-integration state: {post_state}")
        pdb.set_trace()  # BREAKPOINT: Examine post-state

        return integration_event

    def debug_knowledge_graph_operations(self):
        """Debug knowledge graph with variable watching"""
        print(f"\n📚 DEBUGGING: Knowledge Graph Operations")

        # Watch variables
        self.watch_variable('total_nodes', len(self.learning_graph.nodes))
        self.watch_variable('total_triples', len(self.learning_graph.triples))

        # Inspect current graph state
        current_nodes = list(self.learning_graph.nodes.keys())
        current_triples = [(t.subject, t.predicate, t.object) for t in self.learning_graph.triples]

        print(f"Current nodes: {current_nodes}")
        print(f"Current triples: {current_triples}")

        pdb.set_trace()  # BREAKPOINT: Examine knowledge graph state

    def start_interactive_session(self):
        """Start interactive debugging session"""
        print("🔧 INTERACTIVE PATTERN EVOLUTION DEBUGGER")
        print("=" * 50)
        print("Available commands:")
        print("  redis_state() - Inspect Redis state")
        print("  debug_integration(id, concept) - Debug thoughtseed integration")
        print("  debug_kg() - Debug knowledge graph")
        print("  set_breakpoint(func) - Set breakpoint")
        print("  quit() - Exit debugger")
        print("\nType 'help()' for Python debugger commands when at breakpoint")
        print("=" * 50)

        # Make debugger functions available in local scope
        locals_dict = {
            'debugger': self,
            'redis_state': lambda pattern='*': self.inspect_redis_state(pattern),
            'debug_integration': lambda id, concept: asyncio.run(self.debug_thoughtseed_integration(id, concept)),
            'debug_kg': lambda: self.debug_knowledge_graph_operations(),
            'set_breakpoint': self.set_breakpoint,
            'basin_manager': self.basin_manager,
            'learning_graph': self.learning_graph,
            'redis_client': self.redis_client
        }

        # Start interactive Python session with debugger context
        import code
        code.interact(local=locals_dict, banner="🔧 Pattern Evolution Debug Console Active")

def main():
    """Start the interactive debugger"""
    debugger = PatternEvolutionDebugger()
    debugger.start_interactive_session()

if __name__ == "__main__":
    main()