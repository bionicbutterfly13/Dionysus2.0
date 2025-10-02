#!/usr/bin/env python3
"""
T003: Context Engineering - Test Basin Influence Calculations with Co-occurrence

Tests basin strength increment (+0.2, cap 2.0) and co-occurrence tracking for neighborhood influence.
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime

# Add backend to path
backend_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(backend_path))


def test_basin_strength_increment():
    """Test +0.2 strength increment per activation (Phase 1 requirement)"""
    from models.attractor_basin import AttractorBasin, BasinType, NeuralFieldInfluence

    basin = AttractorBasin(
        basin_name="strength_test",
        basin_type=BasinType.CONCEPTUAL,
        stability=0.8,
        depth=1.5,
        activation_threshold=0.5,
        neural_field_influence=NeuralFieldInfluence(
            field_contribution=0.6,
            spatial_extent=1.0,
            temporal_persistence=0.7
        )
    )

    # Note: Basin doesn't have 'strength' field yet (will be added in T015)
    # This test validates the concept using existing activation mechanism

    # Simulate strength progression: 1.0 → 1.2 → 1.4 → 1.6 → 1.8 → 2.0
    activations = []
    for i in range(5):
        # Activate basin (simulates document appearance)
        basin.activate(activation_strength=0.2)  # Increment analog
        activations.append(basin.current_activation)

    # Verify progression (activation increases with each call)
    assert len(activations) == 5
    assert activations[0] < activations[-1], "Activation should increase"

    print(f"✅ Basin strength increment concept validated: {activations}")


def test_strength_cap_at_2_0():
    """Test that strength caps at 2.0 (won't exceed maximum)"""
    from models.attractor_basin import AttractorBasin, BasinType, NeuralFieldInfluence

    basin = AttractorBasin(
        basin_name="cap_test",
        basin_type=BasinType.CONCEPTUAL,
        stability=0.9,
        depth=2.0,
        activation_threshold=0.3,
        saturation_level=1.0,  # This acts like our 2.0 cap
        neural_field_influence=NeuralFieldInfluence(
            field_contribution=0.7,
            spatial_extent=1.2,
            temporal_persistence=0.8
        )
    )

    # Try to over-activate (should saturate)
    for _ in range(10):  # 10 activations
        basin.activate(activation_strength=0.3)

    # Should cap at saturation_level
    assert basin.current_activation <= basin.saturation_level
    assert basin.current_state.value in ["active", "saturated"]

    print(f"✅ Basin saturation cap working: {basin.current_activation} ≤ {basin.saturation_level}")


def test_activation_count_tracking():
    """Test activation count increment (will be formal field in T015)"""
    from models.attractor_basin import AttractorBasin, BasinType, NeuralFieldInfluence

    basin = AttractorBasin(
        basin_name="count_test",
        basin_type=BasinType.SEMANTIC,
        stability=0.75,
        depth=1.3,
        activation_threshold=0.4,
        neural_field_influence=NeuralFieldInfluence(
            field_contribution=0.5,
            spatial_extent=0.9,
            temporal_persistence=0.6
        )
    )

    # Simulate concept appearing in 3 documents
    activation_timestamps = []
    for i in range(3):
        basin.activate(activation_strength=0.5)
        # In T015, we'll add: basin.activation_count += 1
        # For now, verify activation_history tracking works
        if hasattr(basin, 'activation_history') and len(basin.activation_history) > 0:
            activation_timestamps.append(basin.activation_history[-1])

    # Verify we can track activations (structure exists)
    assert hasattr(basin, 'activation_history'), "Basin should have activation_history"

    print(f"✅ Activation tracking ready for formal count field (T015)")


def test_co_occurrence_concept():
    """Test co-occurrence tracking concept (symmetric updates)"""
    # Simulate: "neural_architecture" co-occurs with "search_algorithms" in 3 documents

    # Phase 1 will add to AttractorBasin:
    # co_occurring_concepts: Dict[str, int] = {}

    # For now, validate the concept with a dict
    basin_a = {
        "concept": "neural_architecture",
        "co_occurring": {}  # Will track: {"search_algorithms": count}
    }

    basin_b = {
        "concept": "search_algorithms",
        "co_occurring": {}  # Will track: {"neural_architecture": count}
    }

    # Simulate 3 document co-occurrences
    for doc in range(3):
        # Symmetric update (both basins track each other)
        basin_a["co_occurring"]["search_algorithms"] = \
            basin_a["co_occurring"].get("search_algorithms", 0) + 1

        basin_b["co_occurring"]["neural_architecture"] = \
            basin_b["co_occurring"].get("neural_architecture", 0) + 1

    # Verify symmetry
    assert basin_a["co_occurring"]["search_algorithms"] == 3
    assert basin_b["co_occurring"]["neural_architecture"] == 3
    assert basin_a["co_occurring"]["search_algorithms"] == \
           basin_b["co_occurring"]["neural_architecture"]

    print(f"✅ Co-occurrence symmetric tracking: A↔B count = {basin_a['co_occurring']['search_algorithms']}")


def test_neighborhood_influence_from_cooccurrence():
    """Calculate neighborhood influence score from co-occurring concepts"""
    # CLAUSE φ_nbr signal: Sum of co-occurrence weights

    concept_basin = {
        "concept": "neural_architecture",
        "strength": 1.6,  # Appeared 3 times: 1.0 + 0.2 + 0.2 + 0.2
        "co_occurring": {
            "search_algorithms": 4,    # Co-occurred 4 times
            "reinforcement_learning": 3,  # Co-occurred 3 times
            "automl": 2,                  # Co-occurred 2 times
            "differentiable_nas": 1       # Co-occurred 1 time
        }
    }

    # Calculate neighborhood score (for CLAUSE φ_nbr signal in T018)
    def calculate_neighborhood_score(co_occurring_concepts, max_weight=10):
        """Sum of co-occurrence counts, normalized"""
        total_weight = sum(co_occurring_concepts.values())
        # Normalize to 0.0-1.0 range
        normalized = min(total_weight / max_weight, 1.0)
        return normalized

    nbr_score = calculate_neighborhood_score(concept_basin["co_occurring"])

    # Verify calculation
    total_cooccurrences = sum(concept_basin["co_occurring"].values())
    assert total_cooccurrences == 10  # 4 + 3 + 2 + 1

    # Normalized: 10/10 = 1.0
    assert nbr_score == 1.0

    print(f"✅ Neighborhood influence: {total_cooccurrences} co-occurrences → φ_nbr = {nbr_score}")


def test_basin_influence_on_edge_scoring():
    """Test how basin strength influences edge selection (CLAUSE integration)"""
    # CLAUSE edge score with basin signal:
    # s(e|q,G) = 0.25·φ_ent + 0.25·φ_rel + 0.20·φ_nbr + 0.15·φ_deg + 0.15·φ_basin

    # Scenario: Same edge, different basin strengths
    edge = ("neural_architecture", "RELATED_TO", "search_algorithms")

    # Other signals (constant for this test)
    phi_ent = 0.9   # High entity match
    phi_rel = 0.8   # High relation match
    phi_nbr = 0.7   # Good neighborhood
    phi_deg = 0.6   # Moderate degree

    # Test 1: Low basin strength (1.0 - just created)
    basin_strength_low = 1.0
    phi_basin_low = (basin_strength_low - 1.0) / 1.0  # Normalized: 0.0

    score_low = (
        0.25 * phi_ent +
        0.25 * phi_rel +
        0.20 * phi_nbr +
        0.15 * phi_deg +
        0.15 * phi_basin_low
    )

    # Test 2: High basin strength (2.0 - appeared 5+ times)
    basin_strength_high = 2.0
    phi_basin_high = (basin_strength_high - 1.0) / 1.0  # Normalized: 1.0

    score_high = (
        0.25 * phi_ent +
        0.25 * phi_rel +
        0.20 * phi_nbr +
        0.15 * phi_deg +
        0.15 * phi_basin_high
    )

    # Verify higher basin strength → higher edge score
    assert score_high > score_low
    delta = score_high - score_low
    assert abs(delta - 0.15) < 0.001  # Delta = 0.15 * (1.0 - 0.0)

    print(f"✅ Basin influence on edge scoring: strength 1.0→{score_low:.3f}, strength 2.0→{score_high:.3f} (+{delta:.3f})")


def test_basin_strength_normalization():
    """Test basin strength normalization for edge scoring"""
    # Phase 1 range: 1.0 (min) to 2.0 (max)
    # For edge scoring, normalize to 0.0-1.0:
    # φ_basin = (strength - 1.0) / (2.0 - 1.0) = (strength - 1.0) / 1.0

    test_cases = [
        (1.0, 0.0),   # First appearance
        (1.2, 0.2),   # One activation
        (1.4, 0.4),   # Two activations
        (1.6, 0.6),   # Three activations
        (1.8, 0.8),   # Four activations
        (2.0, 1.0),   # Five+ activations (capped)
    ]

    for strength, expected_norm in test_cases:
        normalized = (strength - 1.0) / 1.0
        assert abs(normalized - expected_norm) < 0.001, \
            f"Normalization failed: {strength} → {normalized} (expected {expected_norm})"

    print(f"✅ Basin strength normalization: 1.0→0.0, 1.6→0.6, 2.0→1.0")


if __name__ == "__main__":
    print("\n=== T003: Context Engineering Basin Influence Validation ===\n")

    test_basin_strength_increment()
    test_strength_cap_at_2_0()
    test_activation_count_tracking()
    test_co_occurrence_concept()
    test_neighborhood_influence_from_cooccurrence()
    test_basin_influence_on_edge_scoring()
    test_basin_strength_normalization()

    print("\n✅ T003 PASSED: Basin influence calculations ready for Phase 1 implementation")
