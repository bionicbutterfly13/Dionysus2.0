#!/usr/bin/env python3
"""
🔧 Terminal Debug Session with Data Inspection
==============================================

This provides debugging directly in your terminal with:
- Full data array inspection
- Step-through execution
- Variable state snapshots
- Interactive breakpoints

Just run: python terminal_debug_session.py
"""

import sys
import redis
import json
import asyncio
import pprint
from typing import Dict, List, Any

# Add paths for imports
sys.path.append('.')
sys.path.append('/Volumes/Asylum/devb/ASI-Arch/extensions/context_engineering')

from attractor_basin_dynamics import AttractorBasinManager
from knowledge_graph_evolution_storage import PermanentLearningGraph

class TerminalDebugger:
    """Terminal-based debugger with full data inspection"""

    def __init__(self):
        print("🔧 Terminal Debug Session Starting...")
        self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        self.basin_manager = AttractorBasinManager()
        self.learning_graph = PermanentLearningGraph()
        self.step_mode = True

    def wait_for_user(self, message="Press Enter to continue, 'q' to quit, 's' to skip steps"):
        """Interactive breakpoint"""
        if not self.step_mode:
            return True

        print(f"\\n🔴 BREAKPOINT: {message}")
        user_input = input(">>> ")

        if user_input.lower() == 'q':
            print("Debug session terminated by user")
            return False
        elif user_input.lower() == 's':
            self.step_mode = False
            print("Step mode disabled - running to completion")

        return True

    def pretty_print_data(self, data, title="Data"):
        """Pretty print data with colors and formatting"""
        print(f"\\n📊 === {title} ===")
        if isinstance(data, dict) and len(data) > 0:
            for key, value in data.items():
                print(f"  {key}:")
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        print(f"    {sub_key}: {sub_value}")
                elif isinstance(value, list):
                    print(f"    Array with {len(value)} items:")
                    for i, item in enumerate(value[:3]):  # Show first 3
                        print(f"      [{i}]: {item}")
                    if len(value) > 3:
                        print(f"      ... ({len(value) - 3} more items)")
                else:
                    print(f"    {value}")
        elif isinstance(data, list):
            print(f"  Array with {len(data)} items:")
            for i, item in enumerate(data):
                print(f"    [{i}]: {item}")
        else:
            pprint.pprint(data, width=80, depth=3)

    def inspect_redis_arrays(self):
        """Show complete Redis data arrays"""
        print("\\n🔍 === REDIS DATA ARRAYS INSPECTION ===")

        # Attractor Basins Array
        basin_keys = self.redis_client.keys('attractor_basin:*')
        basins_array = []

        print(f"\\n📊 Loading {len(basin_keys)} attractor basins...")
        if not self.wait_for_user("About to load basin array data"):
            return

        for key in basin_keys:
            basin_data = json.loads(self.redis_client.get(key))
            basins_array.append(basin_data)

        self.pretty_print_data(basins_array, "ATTRACTOR BASINS ARRAY")
        if not self.wait_for_user(f"Inspected {len(basins_array)} basins"):
            return

        # Knowledge Graph Nodes Array
        kg_keys = self.redis_client.keys('kg_node:*')
        kg_nodes_array = []

        print(f"\\n📚 Loading {len(kg_keys)} knowledge graph nodes...")
        for key in kg_keys:
            node_data = json.loads(self.redis_client.get(key))
            kg_nodes_array.append(node_data)

        self.pretty_print_data(kg_nodes_array, "KNOWLEDGE GRAPH NODES ARRAY")
        if not self.wait_for_user(f"Inspected {len(kg_nodes_array)} KG nodes"):
            return

        # Integration Events Array
        event_keys = self.redis_client.keys('integration_event:*')
        events_array = []

        print(f"\\n🔄 Loading {len(event_keys)} integration events...")
        for key in event_keys:
            event_data = json.loads(self.redis_client.get(key))
            events_array.append(event_data)

        self.pretty_print_data(events_array, "INTEGRATION EVENTS ARRAY")
        if not self.wait_for_user(f"Inspected {len(events_array)} events"):
            return

        return {
            'basins_array': basins_array,
            'kg_nodes_array': kg_nodes_array,
            'events_array': events_array
        }

    async def debug_thoughtseed_integration(self):
        """Debug thoughtseed integration with step-by-step data inspection"""
        print("\\n🌱 === THOUGHTSEED INTEGRATION DEBUG ===")

        thoughtseed_id = "terminal_debug_001"
        concept = "terminal-based debugging patterns"

        print(f"ThoughtSeed ID: {thoughtseed_id}")
        print(f"Concept: {concept}")

        if not self.wait_for_user("Starting thoughtseed integration debug"):
            return

        # === STEP 1: Pre-integration state ===
        print("\\n📊 STEP 1: Capturing pre-integration state...")
        pre_state = {
            'basin_count': len(self.basin_manager.basins),
            'basin_list': list(self.basin_manager.basins.keys()),
            'basin_strengths': {bid: basin.strength for bid, basin in self.basin_manager.basins.items()},
            'total_thoughtseeds': sum(len(basin.thoughtseeds) for basin in self.basin_manager.basins.values())
        }

        self.pretty_print_data(pre_state, "PRE-INTEGRATION STATE")
        if not self.wait_for_user("Pre-state captured"):
            return

        # === STEP 2: Basin similarity calculations ===
        print("\\n🧮 STEP 2: Calculating basin similarities...")
        similarities_data = {}

        for basin_id, basin in self.basin_manager.basins.items():
            similarity = self.basin_manager.calculate_concept_similarity(concept, basin.center_concept)
            influence_type, influence_strength = basin.calculate_influence_on(concept, similarity)

            similarities_data[basin_id] = {
                'center_concept': basin.center_concept,
                'similarity_score': similarity,
                'influence_type': influence_type.value,
                'influence_strength': influence_strength,
                'basin_strength': basin.strength,
                'basin_radius': basin.radius
            }

        self.pretty_print_data(similarities_data, "BASIN SIMILARITY CALCULATIONS")
        if not self.wait_for_user("Similarity calculations complete"):
            return

        # === STEP 3: Find best match ===
        print("\\n🎯 STEP 3: Determining best basin match...")
        best_match = max(similarities_data.items(),
                        key=lambda x: x[1]['influence_strength'])

        best_match_data = {
            'best_basin_id': best_match[0],
            'best_basin_data': best_match[1],
            'decision_logic': f"Highest influence strength: {best_match[1]['influence_strength']:.4f}"
        }

        self.pretty_print_data(best_match_data, "BEST MATCH DECISION")
        if not self.wait_for_user("Best match determined"):
            return

        # === STEP 4: Execute integration ===
        print("\\n⚡ STEP 4: Executing integration...")
        integration_event = await self.basin_manager.integrate_new_thoughtseed(
            thoughtseed_id, concept, {"debug_mode": "terminal"}
        )

        integration_data = {
            'event_id': integration_event.event_id,
            'thoughtseed_id': integration_event.thoughtseed_id,
            'influence_type': integration_event.influence_type.value,
            'influence_strength': integration_event.influence_strength,
            'target_basin_id': integration_event.target_basin_id
        }

        self.pretty_print_data(integration_data, "INTEGRATION EXECUTION RESULT")
        if not self.wait_for_user("Integration executed"):
            return

        # === STEP 5: Post-integration analysis ===
        print("\\n📈 STEP 5: Post-integration analysis...")
        post_state = {
            'basin_count': len(self.basin_manager.basins),
            'basin_list': list(self.basin_manager.basins.keys()),
            'basin_strengths': {bid: basin.strength for bid, basin in self.basin_manager.basins.items()},
            'total_thoughtseeds': sum(len(basin.thoughtseeds) for basin in self.basin_manager.basins.values())
        }

        # Calculate changes
        changes = {
            'basin_count_change': post_state['basin_count'] - pre_state['basin_count'],
            'thoughtseed_count_change': post_state['total_thoughtseeds'] - pre_state['total_thoughtseeds'],
            'new_basins': list(set(post_state['basin_list']) - set(pre_state['basin_list'])),
            'strength_changes': {}
        }

        for basin_id in set(list(pre_state['basin_strengths'].keys()) + list(post_state['basin_strengths'].keys())):
            pre_strength = pre_state['basin_strengths'].get(basin_id, 0.0)
            post_strength = post_state['basin_strengths'].get(basin_id, 0.0)
            if pre_strength != post_strength:
                changes['strength_changes'][basin_id] = {
                    'before': pre_strength,
                    'after': post_strength,
                    'delta': post_strength - pre_strength
                }

        self.pretty_print_data(post_state, "POST-INTEGRATION STATE")
        self.pretty_print_data(changes, "INTEGRATION CHANGES")

        if not self.wait_for_user("Post-integration analysis complete"):
            return

        return integration_event, changes

    async def run_complete_debug_session(self):
        """Run the complete debug session"""
        print("🚀 === COMPLETE TERMINAL DEBUG SESSION ===")
        print("This will walk through all system components with full data inspection")

        if not self.wait_for_user("Start complete debug session?"):
            return

        # 1. Redis data inspection
        print("\\n=== PHASE 1: REDIS DATA INSPECTION ===")
        redis_data = self.inspect_redis_arrays()

        # 2. Thoughtseed integration debug
        print("\\n=== PHASE 2: THOUGHTSEED INTEGRATION ===")
        integration_result = await self.debug_thoughtseed_integration()

        # 3. Final summary
        print("\\n=== PHASE 3: SESSION SUMMARY ===")
        summary = {
            'redis_arrays_inspected': len(redis_data) if redis_data else 0,
            'integration_completed': integration_result is not None,
            'total_basins_now': len(self.basin_manager.basins),
            'debug_session_complete': True
        }

        self.pretty_print_data(summary, "DEBUG SESSION SUMMARY")
        print("\\n✅ Terminal debug session complete!")

def main():
    """Start terminal debug session"""
    debugger = TerminalDebugger()
    asyncio.run(debugger.run_complete_debug_session())

if __name__ == "__main__":
    main()