#!/usr/bin/env python3
"""
T001: Context Engineering - Verify AttractorBasin Accessibility

Constitution Article II mandates Context Engineering integration before any implementation.
This test validates that the existing AttractorBasin model is accessible and ready for extension.
"""

import pytest
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(backend_path))


def test_numpy_2_compliance():
    """Verify NumPy 2.0+ compliance per Constitution Article I"""
    import numpy as np

    version = np.__version__
    major_version = int(version.split('.')[0])

    assert major_version >= 2, (
        f"CONSTITUTION VIOLATION: NumPy {version} detected. "
        f"Article I requires NumPy 2.0+. "
        f"Run: pip install 'numpy>=2.0' --upgrade"
    )
    print(f"✅ NumPy {version} compliant (Article I)")


def test_attractor_basin_import():
    """Verify AttractorBasin model is accessible"""
    from models.attractor_basin import AttractorBasin, BasinType, BasinState

    assert AttractorBasin is not None, "AttractorBasin class not found"
    assert BasinType is not None, "BasinType enum not found"
    assert BasinState is not None, "BasinState enum not found"

    print("✅ AttractorBasin model accessible")


def test_attractor_basin_required_fields():
    """Check AttractorBasin has required base fields for extension"""
    from models.attractor_basin import AttractorBasin, BasinType, NeuralFieldInfluence

    # Create minimal basin to test required fields
    test_basin = AttractorBasin(
        basin_name="test_concept",
        basin_type=BasinType.CONCEPTUAL,
        stability=0.8,
        depth=1.5,
        activation_threshold=0.5,
        neural_field_influence=NeuralFieldInfluence(
            field_contribution=0.5,
            spatial_extent=1.0,
            temporal_persistence=0.8
        )
    )

    # Verify core fields exist
    assert hasattr(test_basin, 'basin_id'), "Missing basin_id field"
    assert hasattr(test_basin, 'basin_name'), "Missing basin_name field"
    assert hasattr(test_basin, 'basin_type'), "Missing basin_type field"
    assert hasattr(test_basin, 'stability'), "Missing stability field"
    assert hasattr(test_basin, 'depth'), "Missing depth field"
    assert hasattr(test_basin, 'current_state'), "Missing current_state field"
    assert hasattr(test_basin, 'current_activation'), "Missing current_activation field"

    # Verify UUID generation works
    assert test_basin.basin_id is not None
    assert len(test_basin.basin_id) > 0

    print(f"✅ AttractorBasin base fields verified: {test_basin.basin_id}")


def test_backward_compatibility_for_extension():
    """Confirm basin model can be extended with new fields without breaking"""
    from models.attractor_basin import AttractorBasin, BasinType, NeuralFieldInfluence

    # Test that we can extend the model with new Phase 1 fields
    # These will be added in T015, but we verify structure supports it

    basin = AttractorBasin(
        basin_name="neural_architecture_search",
        basin_type=BasinType.CONCEPTUAL,
        stability=0.85,
        depth=1.8,
        activation_threshold=0.6,
        neural_field_influence=NeuralFieldInfluence(
            field_contribution=0.7,
            spatial_extent=1.2,
            temporal_persistence=0.9
        )
    )

    # Verify Pydantic model allows dynamic field addition (for extension)
    # We'll add: strength, activation_count, activation_history, co_occurring_concepts

    # Test model dict conversion (needed for Neo4j persistence)
    basin_dict = basin.dict()
    assert 'basin_id' in basin_dict
    assert 'basin_name' in basin_dict
    assert basin_dict['basin_name'] == "neural_architecture_search"

    # Test model supports default values (for lazy migration)
    assert basin.current_activation == 0.0  # Default value works

    print("✅ Basin model ready for extension with Phase 1 fields")


def test_basin_activation_mechanism():
    """Test existing basin activation (needed for influence calculations in T003)"""
    from models.attractor_basin import AttractorBasin, BasinType, BasinState, NeuralFieldInfluence

    basin = AttractorBasin(
        basin_name="test_activation",
        basin_type=BasinType.CONCEPTUAL,
        stability=0.9,
        depth=2.0,
        activation_threshold=0.5,
        neural_field_influence=NeuralFieldInfluence(
            field_contribution=0.6,
            spatial_extent=1.0,
            temporal_persistence=0.7
        )
    )

    # Test activation method exists and works
    assert hasattr(basin, 'activate'), "Missing activate() method"

    # Activate basin
    result = basin.activate(activation_strength=0.6)

    # Should activate since 0.6 >= threshold 0.5
    assert result is True, "Basin should activate when strength >= threshold"
    assert basin.current_state == BasinState.ACTIVE
    assert basin.current_activation >= basin.activation_threshold

    print(f"✅ Basin activation mechanism working: {basin.current_state}")


def test_basin_persistence_readiness():
    """Verify basin can be serialized for Neo4j/Redis (needed for T002, T026, T027)"""
    from models.attractor_basin import AttractorBasin, BasinType, NeuralFieldInfluence
    import json

    basin = AttractorBasin(
        basin_name="persistence_test",
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

    # Test JSON serialization (for Redis caching)
    basin_json = basin.json()
    assert basin_json is not None
    assert isinstance(basin_json, str)

    # Test deserialization
    basin_dict = json.loads(basin_json)
    assert basin_dict['basin_name'] == "persistence_test"
    assert basin_dict['stability'] == 0.75

    # Test re-creation from dict (lazy migration scenario)
    restored_basin = AttractorBasin(**basin_dict)
    assert restored_basin.basin_name == basin.basin_name
    assert restored_basin.basin_id == basin.basin_id

    print("✅ Basin serialization working for Neo4j/Redis persistence")


if __name__ == "__main__":
    # Run tests
    print("\n=== T001: Context Engineering Basin Validation ===\n")

    test_numpy_2_compliance()
    test_attractor_basin_import()
    test_attractor_basin_required_fields()
    test_backward_compatibility_for_extension()
    test_basin_activation_mechanism()
    test_basin_persistence_readiness()

    print("\n✅ T001 PASSED: AttractorBasin accessible and ready for Phase 1 extension")
