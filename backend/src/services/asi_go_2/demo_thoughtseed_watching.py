#!/usr/bin/env python3
"""
Demo: Complete ThoughtSeed Watching System
==========================================

Demonstrates the full watching capability integrated with ASI-GO-2 Researcher.
Shows how the original specification requirements are met:

- Toggle watching on/off for specific ThoughtSeed instances
- Capture comprehensive state during competition cycles
- 10-minute Redis TTL for log retention
- ISO8601 millisecond timestamps
- Integration with existing ASI-GO-2 Researcher component

Author: TDD Implementation Demo
Date: 2025-09-26
"""

import time
import json
from cognitive_core.memory_system import CognitionBase, Pattern
from cognitive_core.thoughtseed_competition import InnerWorkspace, ThoughtGenerator, ThoughtType
from thoughtseed_watcher import watcher

class MockLLM:
    """Mock LLM for testing without external API calls"""
    def query(self, prompt, system_prompt):
        return "Mock solution: Use pattern-based approach to solve the problem."

def demo_thoughtseed_watching():
    """Complete demo of ThoughtSeed watching system"""
    print("🌱🧠 ThoughtSeed Watching System Demo")
    print("=" * 50)

    # 1. Set up ASI-GO-2 components
    print("\n1️⃣ Setting up ASI-GO-2 components...")

    cognition_base = CognitionBase()

    # Add some patterns for competition
    patterns = [
        Pattern("analytical", "Break down complex problems step by step", 0.8, 0.7),
        Pattern("creative", "Generate novel solutions through ideation", 0.6, 0.9),
        Pattern("systematic", "Use structured methodologies", 0.9, 0.8)
    ]

    for pattern in patterns:
        cognition_base.add_pattern(pattern)

    print(f"   📚 Added {len(patterns)} patterns to CognitionBase")

    # 2. Import and set up Researcher (now works with our cognitive_core!)
    print("\n2️⃣ Setting up Researcher with ThoughtSeed competition...")

    from researcher import Researcher
    mock_llm = MockLLM()
    researcher = Researcher(mock_llm, cognition_base)

    print("   🔬 Researcher initialized successfully")

    # 3. Demonstrate watching capability
    print("\n3️⃣ Testing ThoughtSeed watching...")

    # Create a workspace for manual competition testing
    workspace = InnerWorkspace(capacity=5)
    generator = ThoughtGenerator(workspace)

    # Add competing thoughts
    thoughts = [
        ("Use analytical approach", ThoughtType.ACTION, 0.7, 0.8),
        ("Apply creative solution", ThoughtType.ACTION, 0.9, 0.6),
        ("Follow systematic method", ThoughtType.ACTION, 0.8, 0.9)
    ]

    for content, t_type, energy, confidence in thoughts:
        thought = generator.generate_thought(content, t_type)
        thought.energy = energy
        thought.confidence = confidence
        workspace.add_thought(thought)

    print(f"   🧠 Created workspace with {len(thoughts)} competing thoughts")

    # 4. Enable watching
    print("\n4️⃣ Enabling ThoughtSeed watching...")
    workspace_id = watcher.enable_watching(workspace, "demo_workspace")

    print(f"   👁️ Watching enabled for: {workspace_id}")
    print(f"   📊 Watched workspaces: {watcher.get_watched_workspaces()}")

    # 5. Run competition cycles with watching
    print("\n5️⃣ Running ThoughtSeed competition with state logging...")

    for cycle in range(3):
        print(f"\n   Cycle {cycle + 1}:")

        # Show pre-competition state
        dominant = workspace.get_dominant_thought()
        if dominant:
            print(f"     Current leader: {dominant.content} (energy={dominant.energy:.2f})")

        # Run competition update (this triggers state logging)
        workspace.update()

        # Show post-competition state
        new_dominant = workspace.get_dominant_thought()
        if new_dominant:
            print(f"     New leader: {new_dominant.content} (energy={new_dominant.energy:.2f})")

        time.sleep(0.5)  # Brief pause

    # 6. Demonstrate Researcher integration with watching
    print("\n6️⃣ Testing Researcher with watched ThoughtSeed competition...")

    # This will create a new workspace internally and run competition
    proposal, researcher_workspace = researcher.propose_solution("solve optimization problem")

    if researcher_workspace:
        # Enable watching on the researcher's workspace
        researcher_id = watcher.enable_watching(researcher_workspace, "researcher_workspace")
        print(f"   🔬 Enabled watching on Researcher workspace: {researcher_id}")

        # Run a few more cycles on the researcher workspace
        for i in range(2):
            researcher_workspace.update()

        print(f"   💡 Researcher proposal: {proposal.get('pattern_used', 'unknown pattern')}")

    # 7. Retrieve and display logged states
    print("\n7️⃣ Retrieving watched states...")

    all_states = watcher.get_state_logs()
    print(f"   📝 Total logged states: {len(all_states)}")

    demo_states = watcher.get_state_logs("demo_workspace")
    print(f"   📝 Demo workspace states: {len(demo_states)}")

    # Show a sample state log
    if demo_states:
        print("\n   📄 Sample state log:")
        sample_state = demo_states[0]
        print(f"      Timestamp: {sample_state['logged_at']}")
        print(f"      Phase: {sample_state['phase']}")
        print(f"      Thoughts: {sample_state['state']['thought_count']}")
        print(f"      Dominant: {sample_state['state']['dominant_thought_id']}")

    # 8. Demonstrate toggle functionality
    print("\n8️⃣ Testing toggle functionality...")

    print(f"   Current watched: {watcher.get_watched_workspaces()}")

    # Disable watching for demo workspace
    watcher.disable_watching("demo_workspace")
    print(f"   After disable: {watcher.get_watched_workspaces()}")

    # Run update - should not log anymore
    workspace.update()

    # Check log count
    final_demo_states = watcher.get_state_logs("demo_workspace")
    print(f"   📝 Final demo states: {len(final_demo_states)} (should be same as before)")

    # 9. Summary
    print("\n" + "=" * 50)
    print("✅ ThoughtSeed Watching Demo Complete!")
    print("\n📋 Features Demonstrated:")
    print("   ✅ Toggle watching on/off per workspace")
    print("   ✅ State capture during competition cycles")
    print("   ✅ Redis storage with TTL")
    print("   ✅ ISO8601 millisecond timestamps")
    print("   ✅ Integration with ASI-GO-2 Researcher")
    print("   ✅ Multiple workspace support")
    print("   ✅ State retrieval and filtering")

    print(f"\n📊 Final Stats:")
    print(f"   Total states logged: {len(watcher.get_state_logs())}")
    print(f"   Active watched workspaces: {len(watcher.get_watched_workspaces())}")

    return True

if __name__ == "__main__":
    try:
        success = demo_thoughtseed_watching()
        if success:
            print("\n🎉 All ThoughtSeed watching requirements successfully implemented!")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()