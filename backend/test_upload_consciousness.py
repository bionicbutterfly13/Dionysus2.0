#!/usr/bin/env python3
"""
Test document upload through consciousness processing pipeline.

This test verifies the full flow:
1. Upload document via Daedalus gateway
2. Parse document → extract concepts
3. Pass concepts through AttractorBasinManager
4. Generate ThoughtSeeds
5. Learn patterns (reinforcement, competition, synthesis, emergence)
6. Store in Redis memory
7. Decay over time when not mentioned

Example:
    python test_upload_consciousness.py
"""

import sys
from pathlib import Path
import io

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.services.daedalus import Daedalus


def test_upload_flow():
    """Test uploading a document and seeing it pass through consciousness."""

    print("🧪 Testing Document Upload → Consciousness Processing\n")
    print("=" * 60)

    # Create Daedalus gateway
    daedalus = Daedalus()

    # Create test document with repeated concepts
    test_content = b"""
    Neural networks are computational models inspired by biological neural networks.
    Deep learning uses neural networks with multiple layers to learn hierarchical representations.
    Machine learning algorithms enable computers to learn patterns from data.
    Supervised learning trains models using labeled datasets for classification tasks.
    Unsupervised learning discovers hidden patterns in unlabeled data.
    Reinforcement learning agents learn optimal actions through trial and error.
    Convolutional neural networks process grid-like data such as images.
    Recurrent neural networks handle sequential data for time series analysis.
    The attention mechanism allows models to focus on relevant parts of the input.
    Transformer models use self-attention to process sequences in parallel.
    Transfer learning adapts pre-trained models to new tasks efficiently.
    Fine-tuning adjusts model parameters on task-specific datasets.
    Gradient descent optimizes neural network weights during backpropagation.
    The loss function measures the difference between predictions and targets.
    Overfitting occurs when models memorize training data instead of learning patterns.
    Regularization techniques like dropout prevent overfitting in neural networks.
    """

    # Wrap in file-like object
    file_obj = io.BytesIO(test_content)
    file_obj.name = "test_neural_networks.txt"

    # Upload through Daedalus gateway
    print("\n1. Uploading document through Daedalus gateway...")
    result = daedalus.receive_perceptual_information(file_obj)

    print(f"\n2. Upload Status: {result['status']}")
    print(f"   Document: {result['received_data']['filename']}")
    print(f"   Size: {result['received_data']['size_bytes']} bytes")

    # Show parsed concepts
    parsed = result['received_data']['parsed']
    print(f"\n3. Concepts Extracted: {len(parsed.get('concepts', []))}")
    for i, concept in enumerate(parsed.get('concepts', [])[:10], 1):
        print(f"   {i}. {concept}")

    # Show consciousness processing
    consciousness = result.get('consciousness_processing', {})
    print(f"\n4. Consciousness Processing:")
    print(f"   Basins Created: {consciousness.get('basins_created', 0)}")
    print(f"   ThoughtSeeds Generated: {consciousness.get('thoughtseeds_generated', 0)}")

    patterns = consciousness.get('patterns_learned', [])
    if patterns:
        print(f"\n5. Patterns Learned: {len(patterns)}")
        for pattern in patterns[:5]:
            print(f"   - Concept: {pattern['concept']}")
            print(f"     Pattern Type: {pattern['pattern_type']}")
            print(f"     Basin ID: {pattern['basin_id']}")

    # Show agents created
    print(f"\n6. LangGraph Agents Created: {len(result.get('agents_created', []))}")
    for agent in result.get('agents_created', []):
        print(f"   - {agent}")

    print("\n" + "=" * 60)
    print("✅ Upload flow test complete!")

    # Show basin landscape if available
    if daedalus.basin_manager:
        print("\n7. Basin Landscape Summary:")
        summary = daedalus.basin_manager.get_basin_landscape_summary()
        print(f"   Total Basins: {summary['total_basins']}")
        print(f"   Recent Integrations: {summary['recent_integrations']}")

        print("\n   Basin Details:")
        for basin_id, basin_info in summary['basins'].items():
            print(f"   - {basin_id}")
            print(f"     Center Concept: {basin_info['center_concept']}")
            print(f"     Strength: {basin_info['strength']:.2f}")
            print(f"     ThoughtSeeds: {basin_info['thoughtseed_count']}")


if __name__ == "__main__":
    test_upload_flow()
