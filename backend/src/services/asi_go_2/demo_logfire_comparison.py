#!/usr/bin/env python3
"""
Demo: LogFire vs Standard Logging for ThoughtSeed Debugging
==========================================================

Shows the difference between our current logging and LogFire enhanced observability.

Run this to see:
1. Standard approach with print statements
2. LogFire enhanced with hierarchical traces, structured data, live view

To install LogFire: pip install logfire pydantic
"""

import time
from cognitive_core.thoughtseed_competition import InnerWorkspace, ThoughtGenerator, ThoughtType
from cognitive_core.logfire_thoughtseed_tracer import create_traced_workspace

def demo_standard_vs_logfire():
    """Compare standard vs LogFire enhanced debugging"""
    print("🎯 ThoughtSeed Debugging: Standard vs LogFire")
    print("=" * 60)

    # 1. Standard workspace (current approach)
    print("\n1️⃣ STANDARD WORKSPACE (Current Approach)")
    print("-" * 40)

    standard_workspace = InnerWorkspace(capacity=5)
    generator = ThoughtGenerator(standard_workspace)

    # Add competing thoughts
    thoughts = [
        ("Analyze the data systematically", ThoughtType.ACTION, 0.7, 0.8),
        ("Use creative brainstorming", ThoughtType.ACTION, 0.9, 0.6),
        ("Apply proven methodologies", ThoughtType.ACTION, 0.6, 0.9)
    ]

    for content, t_type, energy, confidence in thoughts:
        thought = generator.generate_thought(content, t_type)
        thought.energy = energy
        thought.confidence = confidence
        standard_workspace.add_thought(thought)

    print("Standard output:")
    for i in range(3):
        print(f"  Cycle {i+1}:")
        standard_workspace.update()
        dominant = standard_workspace.get_dominant_thought()
        if dominant:
            print(f"    Winner: {dominant.content} (E:{dominant.energy:.2f})")
        time.sleep(0.5)

    # 2. LogFire enhanced workspace
    print(f"\n2️⃣ LOGFIRE ENHANCED WORKSPACE")
    print("-" * 40)

    traced_workspace = create_traced_workspace(capacity=5)
    traced_generator = ThoughtGenerator(traced_workspace)

    # Add the same competing thoughts
    for content, t_type, energy, confidence in thoughts:
        thought = traced_generator.generate_thought(content, t_type)
        thought.energy = energy
        thought.confidence = confidence
        traced_workspace.add_thought(thought)

    print("LogFire enhanced output (check LogFire dashboard for full traces):")
    for i in range(3):
        print(f"  Cycle {i+1}: Running with full observability...")
        traced_workspace.update()
        time.sleep(0.5)

    print("\n📊 What LogFire Provides vs Standard Logging:")
    print("=" * 60)

    comparison = [
        ("Visibility", "Print statements", "Hierarchical traces with spans"),
        ("Data Structure", "Unstructured text", "Structured, queryable data"),
        ("Relationships", "Not captured", "Parent-child thought relationships"),
        ("Performance", "Basic timing", "Detailed performance metrics"),
        ("Filtering", "None", "SQL queries on thought patterns"),
        ("Live View", "Terminal output", "Real-time dashboard with graphs"),
        ("History", "None", "Time-series analysis of consciousness"),
        ("Debugging", "Manual inspection", "Interactive exploration"),
        ("Export", "Copy-paste", "API access, data export"),
        ("Alerting", "None", "Consciousness level alerts")
    ]

    for aspect, standard, logfire in comparison:
        print(f"  {aspect:12} | {standard:20} | {logfire}")

    print(f"\n🚀 TO ENABLE LOGFIRE:")
    print("1. pip install logfire")
    print("2. Run this demo again")
    print("3. Open LogFire dashboard to see:")
    print("   - Real-time thought competition visualization")
    print("   - Hierarchical spans showing competition cycles")
    print("   - Structured data about energy, confidence, relationships")
    print("   - Consciousness emergence metrics over time")
    print("   - SQL queries to analyze thought patterns")

    return True

if __name__ == "__main__":
    success = demo_logfire_comparison()
    if success:
        print("\n✅ Demo complete! Install LogFire to see the enhanced debugging.")
        print("🌐 LogFire Dashboard: https://logfire.pydantic.dev/")