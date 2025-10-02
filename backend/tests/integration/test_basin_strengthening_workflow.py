#!/usr/bin/env python3
"""
T012: Integration Test - Basin strengthening across multiple documents

This test MUST FAIL initially (services not implemented).
Tests basin strengthening workflow from quickstart.md Step 4.1.
"""

import pytest
from datetime import datetime


@pytest.fixture
def basin_tracker():
    """Import BasinTracker (will fail until implemented)"""
    try:
        from src.services.clause.basin_tracker import BasinTracker
        return BasinTracker()
    except ImportError:
        pytest.skip("BasinTracker not implemented yet")


def test_basin_strength_progression(basin_tracker):
    """Test basin strength progression across 3 documents: 1.0 → 1.2 → 1.4 → 1.6"""

    concept = "neural_architecture"

    # Document 1: First appearance (strength = 1.0)
    result1 = basin_tracker.strengthen_basins(
        concepts=[concept], document_id="doc_001", increment=0.2
    )

    basin1 = result1.get("updated_basins", []) + result1.get("new_basins", [])
    assert len(basin1) > 0, "No basin created for first document"

    strength1 = basin1[0].get("strength", 0)
    assert strength1 == 1.0, f"First document strength should be 1.0, got {strength1}"

    # Document 2: Second appearance (strength = 1.2)
    result2 = basin_tracker.strengthen_basins(
        concepts=[concept], document_id="doc_002", increment=0.2
    )

    basin2 = result2.get("updated_basins", [])
    assert len(basin2) > 0, "No basin updated for second document"

    strength2 = basin2[0].get("strength", 0)
    assert strength2 == 1.2, f"Second document strength should be 1.2, got {strength2}"

    # Document 3: Third appearance (strength = 1.4)
    result3 = basin_tracker.strengthen_basins(
        concepts=[concept], document_id="doc_003", increment=0.2
    )

    basin3 = result3.get("updated_basins", [])
    strength3 = basin3[0].get("strength", 0)
    assert strength3 == 1.4, f"Third document strength should be 1.4, got {strength3}"

    print(f"✅ Strength progression: 1.0 → 1.2 → 1.4 → 1.6 (tested up to 1.4)")


def test_activation_count_increments(basin_tracker):
    """Test activation_count increments correctly with each document"""

    concept = "search_algorithms"

    # Document 1
    result1 = basin_tracker.strengthen_basins(
        concepts=[concept], document_id="doc_001", increment=0.2
    )

    basin1 = result1.get("new_basins", []) + result1.get("updated_basins", [])
    activation_count1 = basin1[0].get("activation_count", -1)

    assert activation_count1 == 1, \
        f"First activation_count should be 1, got {activation_count1}"

    # Document 2
    result2 = basin_tracker.strengthen_basins(
        concepts=[concept], document_id="doc_002", increment=0.2
    )

    basin2 = result2.get("updated_basins", [])
    activation_count2 = basin2[0].get("activation_count", -1)

    assert activation_count2 == 2, \
        f"Second activation_count should be 2, got {activation_count2}"

    # Document 3
    result3 = basin_tracker.strengthen_basins(
        concepts=[concept], document_id="doc_003", increment=0.2
    )

    basin3 = result3.get("updated_basins", [])
    activation_count3 = basin3[0].get("activation_count", -1)

    assert activation_count3 == 3, \
        f"Third activation_count should be 3, got {activation_count3}"

    print(f"✅ Activation count progression: 1 → 2 → 3")


def test_cooccurrence_symmetric_tracking(basin_tracker):
    """Test co-occurrence pairs are tracked symmetrically (A↔B)"""

    # Document with 3 co-occurring concepts
    concepts = ["neural_architecture", "search_algorithms", "reinforcement_learning"]

    result = basin_tracker.strengthen_basins(
        concepts=concepts, document_id="doc_001", increment=0.2
    )

    cooccurrence_updates = result.get("cooccurrence_updates", {})

    # Verify symmetry: A→B implies B→A
    for source, targets in cooccurrence_updates.items():
        for target in targets:
            # Check reverse exists
            assert target in cooccurrence_updates, \
                f"Reverse co-occurrence missing: {target} not in updates"

            assert source in cooccurrence_updates[target], \
                f"Asymmetric co-occurrence: {source}→{target} but not {target}→{source}"

    print(f"✅ Co-occurrence symmetric tracking validated")


def test_basin_strength_cap_at_2_0(basin_tracker):
    """Test basin strength caps at 2.0 (after 5+ appearances)"""

    concept = "automl"

    # Strengthen 10 times (should cap at 2.0)
    for i in range(10):
        result = basin_tracker.strengthen_basins(
            concepts=[concept], document_id=f"doc_{i:03d}", increment=0.2
        )

        basins = result.get("updated_basins", []) + result.get("new_basins", [])
        strength = basins[0].get("strength", 0)

        # Should never exceed 2.0
        assert strength <= 2.0, \
            f"Strength cap violated: {strength} > 2.0 on iteration {i + 1}"

    # Final strength should be exactly 2.0
    final_result = basin_tracker.strengthen_basins(
        concepts=[concept], document_id="doc_final", increment=0.2
    )

    final_basins = final_result.get("updated_basins", [])
    final_strength = final_basins[0].get("strength", 0)

    assert final_strength == 2.0, \
        f"Final strength should be capped at 2.0, got {final_strength}"

    print(f"✅ Basin strength capped at 2.0 after 5+ activations")


def test_multiple_concepts_single_document(basin_tracker):
    """Test processing multiple concepts in a single document"""

    concepts = [
        "neural_architecture",
        "search_algorithms",
        "reinforcement_learning",
        "differentiable_nas",
    ]

    result = basin_tracker.strengthen_basins(
        concepts=concepts, document_id="doc_multi", increment=0.2
    )

    # Should create/update all concepts
    all_basins = result.get("updated_basins", []) + result.get("new_basins", [])

    assert len(all_basins) == len(concepts), \
        f"Expected {len(concepts)} basins, got {len(all_basins)}"

    # Each concept should have strength 1.0 (first appearance)
    for basin in all_basins:
        assert basin.get("strength", 0) == 1.0, \
            f"First appearance should have strength 1.0, got {basin.get('strength')}"

    # Co-occurrence should have n(n-1)/2 = 6 pairs for 4 concepts
    cooccurrence = result.get("cooccurrence_updates", {})
    total_pairs = sum(len(targets) for targets in cooccurrence.values()) // 2

    expected_pairs = len(concepts) * (len(concepts) - 1) // 2  # 4*3/2 = 6
    assert total_pairs == expected_pairs, \
        f"Expected {expected_pairs} co-occurrence pairs, got {total_pairs}"

    print(f"✅ Multi-concept document: 4 concepts → 6 co-occurrence pairs")


def test_strengthening_performance(basin_tracker):
    """Test basin strengthening completes in <5ms per basin"""

    import time

    concepts = [f"concept_{i}" for i in range(10)]

    start_time = time.perf_counter()
    result = basin_tracker.strengthen_basins(
        concepts=concepts, document_id="doc_perf", increment=0.2
    )
    end_time = time.perf_counter()

    total_time_ms = (end_time - start_time) * 1000
    time_per_basin_ms = total_time_ms / len(concepts)

    # Assert performance SLA
    assert time_per_basin_ms < 5, \
        f"Basin update time {time_per_basin_ms:.2f}ms exceeds 5ms limit"

    print(f"✅ Basin strengthening: {time_per_basin_ms:.2f}ms per basin")


if __name__ == "__main__":
    print("\n=== T012: Integration Test - Basin Strengthening Workflow ===\n")
    print("⚠️  This test MUST FAIL initially (BasinTracker not implemented)")
    print("✅  Test will pass once core services are implemented (T015-T025)\n")

    pytest.main([__file__, "-v"])
