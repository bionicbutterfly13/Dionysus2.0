"""
Tests for EventNode Model (T015)
TDD compliance tests for Mosaic observation schema and consciousness tracking
"""

import pytest
from datetime import datetime, timedelta
from src.models.event_node import (
    EventNode, EventType, MosaicObservation, MosaicObservationType
)


class TestEventNode:
    """Test EventNode model functionality"""

    def test_event_node_creation(self):
        """Test basic event node creation"""
        event = EventNode(
            user_id="test_user",
            event_type=EventType.PERCEPTION,
            event_name="Test Perception Event"
        )

        assert event.user_id == "test_user"
        assert event.event_type == EventType.PERCEPTION
        assert event.event_name == "Test Perception Event"
        assert event.mock_data_enabled is True
        assert len(event.mosaic_observations) == 0

    def test_mosaic_observation_addition(self):
        """Test adding mosaic observations to events"""
        event = EventNode(
            user_id="test_user",
            event_type=EventType.COGNITION,
            event_name="Test Cognition"
        )

        observation_id = event.add_mosaic_observation(
            MosaicObservationType.ATTENTION_SHIFT,
            attention_level=0.8,
            awareness_depth=0.7,
            integration_strength=0.6,
            context_data={"focus_target": "consciousness_concept"},
            attractor_influence={"working_memory": 0.5}
        )

        assert len(event.mosaic_observations) == 1
        assert event.mosaic_observations[0].observation_type == MosaicObservationType.ATTENTION_SHIFT
        assert event.mosaic_observations[0].attention_level == 0.8
        assert event.primary_observation_type == MosaicObservationType.ATTENTION_SHIFT
        assert isinstance(observation_id, str)

    def test_primary_observation_update(self):
        """Test primary observation type updates based on integration strength"""
        event = EventNode(
            user_id="test_user",
            event_type=EventType.REFLECTION,
            event_name="Test Reflection"
        )

        # Add first observation with lower integration
        event.add_mosaic_observation(
            MosaicObservationType.MEMORY_RETRIEVAL,
            attention_level=0.5,
            awareness_depth=0.5,
            integration_strength=0.4
        )

        # Add second observation with higher integration
        event.add_mosaic_observation(
            MosaicObservationType.INSIGHT_MOMENT,
            attention_level=0.9,
            awareness_depth=0.8,
            integration_strength=0.9
        )

        assert event.primary_observation_type == MosaicObservationType.INSIGHT_MOMENT

    def test_consciousness_snapshot_update(self):
        """Test consciousness state snapshot updates"""
        event = EventNode(
            user_id="test_user",
            event_type=EventType.INTEGRATION,
            event_name="Test Integration"
        )

        metrics = {
            "attention": 0.8,
            "awareness": 0.7,
            "integration": 0.9,
            "meta_cognition": 0.6
        }

        event.update_consciousness_snapshot(metrics)

        assert event.consciousness_snapshot["attention"] == 0.8
        assert event.consciousness_snapshot["awareness"] == 0.7
        assert event.consciousness_impact >= 0.75  # Average of metrics

    def test_causal_relationship_addition(self):
        """Test adding causal relationships between events"""
        event = EventNode(
            user_id="test_user",
            event_type=EventType.ACTION,
            event_name="Test Action"
        )

        event.add_causal_relationship(
            "related_event_123",
            "causes",
            0.8,
            "This action causes the related event"
        )

        assert len(event.causal_relationships) == 1
        assert event.causal_relationships[0]["related_event_id"] == "related_event_123"
        assert event.causal_relationships[0]["relationship_type"] == "causes"
        assert event.causal_relationships[0]["strength"] == 0.8
        assert "related_event_123" in event.related_event_ids

    def test_temporal_relationship_addition(self):
        """Test adding temporal relationships between events"""
        event = EventNode(
            user_id="test_user",
            event_type=EventType.EMERGENCE,
            event_name="Test Emergence"
        )

        event.add_temporal_relationship(
            "temporal_event_456",
            "before",
            -5000  # 5 seconds before
        )

        assert len(event.temporal_relationships) == 1
        assert event.temporal_relationships[0]["related_event_id"] == "temporal_event_456"
        assert event.temporal_relationships[0]["relationship_type"] == "before"
        assert event.temporal_relationships[0]["time_delta_ms"] == -5000

    def test_significance_score_calculation(self):
        """Test event significance score calculation"""
        event = EventNode(
            user_id="test_user",
            event_type=EventType.COGNITION,
            event_name="Test Cognition"
        )

        # Set up consciousness snapshot
        event.update_consciousness_snapshot({
            "attention": 0.8,
            "awareness": 0.7,
            "integration": 0.9
        })

        # Add mosaic observations
        event.add_mosaic_observation(
            MosaicObservationType.PATTERN_RECOGNITION,
            attention_level=0.8,
            awareness_depth=0.7,
            integration_strength=0.9
        )

        # Add causal relationships
        event.add_causal_relationship("related_1", "enables", 0.7)
        event.add_causal_relationship("related_2", "causes", 0.8)

        # Set learning impact
        event.learning_impact = 0.6

        significance = event.calculate_significance_score()

        assert 0.0 <= significance <= 1.0
        assert event.significance_score == significance

    def test_dominant_mosaic_patterns_analysis(self):
        """Test dominant mosaic patterns analysis"""
        event = EventNode(
            user_id="test_user",
            event_type=EventType.PERCEPTION,
            event_name="Test Perception"
        )

        # Add multiple observations
        event.add_mosaic_observation(
            MosaicObservationType.ATTENTION_SHIFT,
            attention_level=0.8,
            awareness_depth=0.7,
            integration_strength=0.6
        )

        event.add_mosaic_observation(
            MosaicObservationType.ATTENTION_SHIFT,
            attention_level=0.9,
            awareness_depth=0.8,
            integration_strength=0.7
        )

        event.add_mosaic_observation(
            MosaicObservationType.CONCEPT_ACTIVATION,
            attention_level=0.7,
            awareness_depth=0.6,
            integration_strength=0.5
        )

        patterns = event.get_dominant_mosaic_patterns()

        assert patterns["dominant_observation_type"] == "attention_shift"
        assert patterns["observation_diversity"] == 2  # Two different types
        assert patterns["total_observations"] == 3
        assert 0.7 <= patterns["avg_attention"] <= 0.9

    def test_child_event_creation(self):
        """Test child event creation"""
        parent_event = EventNode(
            user_id="test_user",
            event_type=EventType.REFLECTION,
            event_name="Parent Event"
        )

        child_id = parent_event.create_child_event(
            EventType.INTEGRATION,
            "Child Integration Event",
            "Child event spawned from reflection"
        )

        assert isinstance(child_id, str)
        assert child_id in parent_event.child_event_ids

    def test_mock_event_creation(self):
        """Test mock event creation for development"""
        mock_event = EventNode.create_mock_event(
            "mock_user",
            EventType.COGNITION,
            "Mock Cognition Event",
            "mock_journey"
        )

        assert mock_event.user_id == "mock_user"
        assert mock_event.journey_id == "mock_journey"
        assert mock_event.event_type == EventType.COGNITION
        assert mock_event.mock_data_enabled is True
        assert len(mock_event.mosaic_observations) > 0
        assert len(mock_event.consciousness_snapshot) > 0

    def test_event_serialization(self):
        """Test event serialization to dict"""
        event = EventNode(
            user_id="test_user",
            event_type=EventType.ACTION,
            event_name="Test Action"
        )

        event_dict = event.to_dict()

        assert isinstance(event_dict, dict)
        assert event_dict["user_id"] == "test_user"
        assert event_dict["event_type"] == "action"

    def test_enum_validation(self):
        """Test enum validation for event and observation types"""
        with pytest.raises(ValueError):
            EventNode(
                user_id="test",
                event_type="invalid_type",
                event_name="test"
            )

        event = EventNode(
            user_id="test",
            event_type=EventType.PERCEPTION,
            event_name="test"
        )

        with pytest.raises(ValueError):
            event.add_mosaic_observation(
                "invalid_observation_type",
                0.5, 0.5, 0.5
            )


class TestMosaicObservation:
    """Test MosaicObservation functionality"""

    def test_mosaic_observation_creation(self):
        """Test mosaic observation creation"""
        observation = MosaicObservation(
            observation_type=MosaicObservationType.INSIGHT_MOMENT,
            attention_level=0.9,
            awareness_depth=0.8,
            integration_strength=0.9
        )

        assert observation.observation_type == MosaicObservationType.INSIGHT_MOMENT
        assert observation.attention_level == 0.9
        assert observation.awareness_depth == 0.8
        assert observation.integration_strength == 0.9
        assert isinstance(observation.observation_id, str)

    def test_observation_validation(self):
        """Test observation field validation"""
        with pytest.raises(ValueError):
            MosaicObservation(
                observation_type=MosaicObservationType.CONCEPT_ACTIVATION,
                attention_level=1.5,  # Invalid: > 1.0
                awareness_depth=0.5,
                integration_strength=0.5
            )

        with pytest.raises(ValueError):
            MosaicObservation(
                observation_type=MosaicObservationType.CONCEPT_ACTIVATION,
                attention_level=-0.1,  # Invalid: < 0.0
                awareness_depth=0.5,
                integration_strength=0.5
            )