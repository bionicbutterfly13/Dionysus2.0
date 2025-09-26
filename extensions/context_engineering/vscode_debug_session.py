#!/usr/bin/env python3
"""
🔧 VS Code Debug Session with Breakpoints
=========================================

This provides true debugging with:
- VS Code breakpoint integration
- Variable inspection panels
- Step-through execution
- Watch expressions
- Call stack visualization

Run this to start a debug session that VS Code can attach to.
"""

import debugpy
import sys
import time
import redis
import json
import asyncio
from typing import Dict, List, Any

# Add paths for imports
sys.path.append('.')
sys.path.append('/Volumes/Asylum/devb/ASI-Arch/extensions/context_engineering')

from attractor_basin_dynamics import AttractorBasinManager, BasinInfluenceType
from knowledge_graph_evolution_storage import PermanentLearningGraph

class VSCodeDebugSession:
    """Debug session with VS Code integration"""

    def __init__(self):
        print("🔧 Initializing VS Code Debug Session...")

        # Start debug server for VS Code to attach to
        debugpy.listen(5678)
        print("📡 Debug server listening on port 5678")
        print("🔗 In VS Code: Run & Debug > 'Attach to Running Process'")
        print("⏳ Waiting for debugger to attach...")

        # Wait for VS Code debugger to attach
        debugpy.wait_for_client()
        print("✅ VS Code debugger attached!")

        # Initialize components
        self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.basin_manager = AttractorBasinManager()
        self.learning_graph = PermanentLearningGraph()

        print("🔧 Debug session ready - you can now set breakpoints in VS Code")

    def inspect_full_redis_data(self, pattern: str = "*") -> Dict[str, Any]:
        """Inspect Redis data with full content - BREAKPOINT TARGET"""
        print(f"🔍 Inspecting Redis data with pattern: {pattern}")

        # Get all keys
        keys = self.redis_client.keys(pattern)
        full_data = {}

        # BREAKPOINT: Set a breakpoint here in VS Code to inspect 'keys' variable
        debugpy.breakpoint()  # VS Code will stop here

        for key in keys:
            raw_data = self.redis_client.get(key)
            if raw_data:
                try:
                    parsed_data = json.loads(raw_data)
                    full_data[key] = parsed_data

                    # BREAKPOINT: Inspect each key's data
                    debugpy.breakpoint()  # Stop for each key

                except json.JSONDecodeError:
                    full_data[key] = {"error": "non-JSON data", "raw": raw_data}

        return full_data

    async def debug_thoughtseed_with_breakpoints(self, thoughtseed_id: str, concept: str):
        """Debug thoughtseed integration with VS Code breakpoints"""
        print(f"🌱 Starting thoughtseed integration debug: {thoughtseed_id}")

        # === BREAKPOINT 1: Initial state ===
        initial_basin_count = len(self.basin_manager.basins)
        initial_basins = dict(self.basin_manager.basins)

        # BREAKPOINT: Inspect initial state in VS Code panels
        debugpy.breakpoint()  # Examine: initial_basin_count, initial_basins

        # === BREAKPOINT 2: Similarity calculations ===
        print("🧮 Calculating basin similarities...")
        similarities = {}
        best_match = {"basin_id": None, "similarity": 0.0, "influence_strength": 0.0}

        for basin_id, basin in self.basin_manager.basins.items():
            similarity = self.basin_manager.calculate_concept_similarity(concept, basin.center_concept)
            influence_type, influence_strength = basin.calculate_influence_on(concept, similarity)

            similarities[basin_id] = {
                "center_concept": basin.center_concept,
                "similarity": similarity,
                "influence_type": influence_type.value,
                "influence_strength": influence_strength,
                "basin_strength": basin.strength
            }

            if influence_strength > best_match["influence_strength"]:
                best_match = {
                    "basin_id": basin_id,
                    "similarity": similarity,
                    "influence_strength": influence_strength
                }

        # BREAKPOINT: Inspect similarity calculations
        debugpy.breakpoint()  # Examine: similarities, best_match

        # === BREAKPOINT 3: Pre-integration state capture ===
        pre_integration_state = {
            'basin_count': len(self.basin_manager.basins),
            'basin_strengths': {bid: basin.strength for bid, basin in self.basin_manager.basins.items()},
            'total_thoughtseeds': sum(len(basin.thoughtseeds) for basin in self.basin_manager.basins.values())
        }

        # BREAKPOINT: Inspect pre-integration state
        debugpy.breakpoint()  # Examine: pre_integration_state

        # === BREAKPOINT 4: Integration execution ===
        print("⚡ Executing integration...")
        integration_event = await self.basin_manager.integrate_new_thoughtseed(
            thoughtseed_id, concept, {"debug_session": True}
        )

        # BREAKPOINT: Inspect integration result
        debugpy.breakpoint()  # Examine: integration_event

        # === BREAKPOINT 5: Post-integration analysis ===
        post_integration_state = {
            'basin_count': len(self.basin_manager.basins),
            'basin_strengths': {bid: basin.strength for bid, basin in self.basin_manager.basins.items()},
            'total_thoughtseeds': sum(len(basin.thoughtseeds) for basin in self.basin_manager.basins.values())
        }

        # Calculate changes
        changes = {
            'basin_count_delta': post_integration_state['basin_count'] - pre_integration_state['basin_count'],
            'thoughtseed_count_delta': post_integration_state['total_thoughtseeds'] - pre_integration_state['total_thoughtseeds'],
            'strength_changes': {}
        }

        for basin_id in set(list(pre_integration_state['basin_strengths'].keys()) +
                          list(post_integration_state['basin_strengths'].keys())):
            pre_strength = pre_integration_state['basin_strengths'].get(basin_id, 0.0)
            post_strength = post_integration_state['basin_strengths'].get(basin_id, 0.0)
            if pre_strength != post_strength:
                changes['strength_changes'][basin_id] = {
                    'before': pre_strength,
                    'after': post_strength,
                    'delta': post_strength - pre_strength
                }

        # BREAKPOINT: Inspect final changes
        debugpy.breakpoint()  # Examine: changes, post_integration_state

        return integration_event, changes

    def inspect_knowledge_graph_with_breakpoints(self):
        """Inspect knowledge graph with VS Code breakpoints"""
        print("📚 Inspecting knowledge graph...")

        # Get current graph state
        nodes_array = []
        triples_array = []

        # Process nodes
        for node_id, node in self.learning_graph.nodes.items():
            node_data = {
                'node_id': node.node_id,
                'pattern_type': node.pattern_type,
                'description': node.description,
                'influence_strength': node.influence_strength,
                'metadata': node.metadata
            }
            nodes_array.append(node_data)

        # BREAKPOINT: Inspect nodes array
        debugpy.breakpoint()  # Examine: nodes_array

        # Process triples
        for triple in self.learning_graph.triples:
            triple_data = {
                'subject': triple.subject,
                'predicate': triple.predicate,
                'object': triple.object,
                'confidence': triple.confidence,
                'source_event': triple.source_event
            }
            triples_array.append(triple_data)

        # BREAKPOINT: Inspect triples array
        debugpy.breakpoint()  # Examine: triples_array

        # Create summary
        summary = {
            'total_nodes': len(nodes_array),
            'total_triples': len(triples_array),
            'nodes_by_type': {},
            'triples_by_predicate': {}
        }

        for node in nodes_array:
            pattern_type = node['pattern_type']
            summary['nodes_by_type'][pattern_type] = summary['nodes_by_type'].get(pattern_type, 0) + 1

        for triple in triples_array:
            predicate = triple['predicate']
            summary['triples_by_predicate'][predicate] = summary['triples_by_predicate'].get(predicate, 0) + 1

        # BREAKPOINT: Inspect summary
        debugpy.breakpoint()  # Examine: summary

        return nodes_array, triples_array, summary

    async def run_debug_session(self):
        """Run the main debug session"""
        print("🚀 Starting comprehensive debug session...")

        print("\n=== 1. REDIS DATA INSPECTION ===")
        redis_data = self.inspect_full_redis_data("attractor_basin:*")

        print("\n=== 2. KNOWLEDGE GRAPH INSPECTION ===")
        kg_nodes, kg_triples, kg_summary = self.inspect_knowledge_graph_with_breakpoints()

        print("\n=== 3. THOUGHTSEED INTEGRATION DEBUG ===")
        integration_event, changes = await self.debug_thoughtseed_with_breakpoints(
            "vscode_debug_001",
            "VS Code debugging integration patterns"
        )

        print("\n✅ Debug session complete!")
        print(f"📊 Final state: {len(self.basin_manager.basins)} basins, {changes}")

def main():
    """Start VS Code debug session"""
    session = VSCodeDebugSession()
    asyncio.run(session.run_debug_session())

if __name__ == "__main__":
    main()