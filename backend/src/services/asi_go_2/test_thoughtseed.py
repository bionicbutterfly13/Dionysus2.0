"""
Test script demonstrating thoughtseed competition in an inner mental workspace
"""
from thoughtseed_competition import (
    InnerWorkspace, ThoughtGenerator, ThoughtType, create_example_workspace
)
import time
from typing import Dict, Any
import json

def print_workspace_state(state: Dict[str, Any]):
    """Pretty print workspace state"""
    print("\n=== Mental Workspace State ===")
    print(f"Time step: {state['time_step']}")
    print(f"Total thoughts: {state['thought_count']}")
    print(f"Active thoughts: {state['active_thoughts']}")
    print(f"Dominant thought: {state['dominant_thought']}")
    print(f"Average energy: {state['average_energy']:.2f}")
    print(f"Average confidence: {state['average_confidence']:.2f}")
    print("\nSpatial distribution:")
    for t_id, pos in state['spatial_distribution'].items():
        print(f"  {t_id}: {[f'{x:.2f}' for x in pos]}")
    print("=" * 30 + "\n")

def run_thought_competition(steps: int = 10):
    """Run thoughtseed competition simulation"""
    # Create workspace and generator
    workspace = create_example_workspace()
    generator = ThoughtGenerator(workspace)
    
    print("Starting thoughtseed competition simulation...")
    
    # Add some competing thoughts
    thoughts = [
        ("Pattern suggests exponential growth", ThoughtType.BELIEF),
        ("Data might be oscillating", ThoughtType.BELIEF),
        ("Need to check for seasonality", ThoughtType.GOAL),
        ("Previous analysis showed linear trend", ThoughtType.PERCEPTION)
    ]
    
    for content, type in thoughts:
        thought = generator.generate_thought(content, type)
        workspace.add_thought(thought)
        print(f"Added thought: {content} ({type.value})")
    
    # Run competition
    print("\nRunning competition...")
    for step in range(steps):
        print(f"\nStep {step + 1}/{steps}")
        
        # Update workspace
        workspace.update()
        
        # Get and print state
        state = workspace.get_workspace_state()
        print_workspace_state(state)
        
        # Get dominant thought
        dominant = workspace.get_dominant_thought()
        if dominant:
            print(f"Dominant thought: {dominant.content}")
            
            # Get thought cluster around dominant
            cluster = workspace.get_thought_cluster(dominant.id)
            if cluster:
                print("Related thoughts:")
                for t_id in cluster:
                    thought = workspace.thoughts[t_id]
                    print(f"  - {thought.content} (Energy: {thought.energy:.2f})")
        
        # Create thought chain
        if dominant:
            chain = workspace.create_thought_chain(dominant.id)
            print("\nThought chain:")
            for t_id, connections in chain.items():
                thought = workspace.thoughts[t_id]
                print(f"  {thought.content} →")
                for conn_id in connections:
                    conn = workspace.thoughts[conn_id]
                    print(f"    └─ {conn.content}")
        
        time.sleep(1)  # Pause to make output readable

if __name__ == "__main__":
    run_thought_competition(steps=5)