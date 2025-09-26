"""
Comprehensive test suite for all Dionysus 2.0 models
TDD compliance validation for the complete model layer
"""

import pytest
from datetime import datetime, timedelta

# Import all models
from src.models.user_profile import UserProfile, LearningStyle, ConsciousnessLevel
from src.models.autobiographical_journey import AutobiographicalJourney
from src.models.thoughtseed_trace import ThoughtSeedTrace, ConsciousnessState, MemoryContextType
from src.models.event_node import EventNode, EventType, MosaicObservationType
from src.models.document_artifact import DocumentArtifact, DocumentType, ProcessingStatus
from src.models.concept_node import ConceptNode, ConceptType, ActivationPattern
from src.models.curiosity_mission import CuriosityMission, CuriosityType, MissionStatus, ExplorationStrategy
from src.models.evaluation_frame import EvaluationFrame, EvaluationType, ConsentLevel
from src.models.visualization_state import VisualizationState, ViewType, ThemeType


class TestModelIntegration:
    """Test integration between all models"""

    def test_complete_consciousness_flow(self):
        """Test complete consciousness development flow across all models"""

        # 1. Create user profile
        user_profile = UserProfile.create_mock_profile("integration_user", "test_user")
        user_id = user_profile.user_id

        # 2. Create autobiographical journey
        journey = AutobiographicalJourney(
            user_id=user_id,
            title="Complete Consciousness Journey",
            mock_data=True
        )

        # 3. Create thoughtseed trace
        trace = ThoughtSeedTrace(user_id=user_id, journey_id=journey.journey_id)
        trace.add_attractor_pattern("consciousness_attractor", 0.8, 0.9)

        # 4. Create consciousness event
        event = EventNode(
            user_id=user_id,
            journey_id=journey.journey_id,
            trace_id=trace.trace_id,
            event_type=EventType.EMERGENCE,
            event_name="Consciousness Emergence Event"
        )

        # 5. Add mosaic observation
        event.add_mosaic_observation(
            MosaicObservationType.INSIGHT_MOMENT,
            attention_level=0.9,
            awareness_depth=0.8,
            integration_strength=0.9
        )

        # 6. Create concept node
        concept = ConceptNode(
            user_id=user_id,
            journey_id=journey.journey_id,
            concept_name="consciousness",
            concept_type=ConceptType.ABSTRACT
        )

        # 7. Create document artifact
        document = DocumentArtifact(
            user_id=user_id,
            journey_id=journey.journey_id,
            title="Consciousness Research Paper",
            document_type=DocumentType.RESEARCH_PAPER
        )

        # 8. Create curiosity mission
        mission = CuriosityMission(
            user_id=user_id,
            journey_id=journey.journey_id,
            mission_title="Explore Consciousness Nature",
            primary_curiosity_type=CuriosityType.EPISTEMIC
        )

        # 9. Create evaluation frame
        evaluation = EvaluationFrame(
            user_id=user_id,
            journey_id=journey.journey_id,
            evaluation_type=EvaluationType.CONSCIOUSNESS_ASSESSMENT,
            evaluation_title="Consciousness Development Assessment",
            user_consent_level=ConsentLevel.FULL_CONSENT
        )

        # 10. Create visualization state
        viz_state = VisualizationState(
            user_id=user_id,
            journey_id=journey.journey_id,
            active_view=ViewType.CONSCIOUSNESS_MAP
        )

        # Verify all models are properly connected
        assert user_profile.user_id == user_id
        assert journey.user_id == user_id
        assert trace.user_id == user_id
        assert event.user_id == user_id
        assert concept.user_id == user_id
        assert document.user_id == user_id
        assert mission.user_id == user_id
        assert evaluation.user_id == user_id
        assert viz_state.user_id == user_id

        # Verify journey connections
        assert trace.journey_id == journey.journey_id
        assert event.journey_id == journey.journey_id
        assert concept.journey_id == journey.journey_id

        # Test cross-model interactions
        # Event already associated with thoughtseed via trace_id
        assert event.trace_id == trace.trace_id

        # Associate concept with thoughtseed
        concept.associate_thoughtseed(trace.trace_id, 0.9)

        # Associate document with mission
        mission.documents_consulted.append(document.artifact_id)

        # Update journey with thoughtseed
        journey.add_thoughtseed_trace(trace.trace_id, trace.get_current_attractor_state())

        # Verify associations
        assert event.trace_id == trace.trace_id
        assert trace.trace_id in concept.thoughtseed_associations
        assert document.artifact_id in mission.documents_consulted

    def test_constitutional_compliance_across_models(self):
        """Test constitutional compliance flags across all models"""
        user_id = "compliance_test_user"

        # Create instances of all models
        models = [
            UserProfile.create_mock_profile(user_id, "test"),
            AutobiographicalJourney(user_id=user_id, title="Test", mock_data=True),
            ThoughtSeedTrace.create_mock_trace(user_id),
            EventNode.create_mock_event(user_id, EventType.COGNITION, "Test"),
            DocumentArtifact.create_mock_document(user_id, "Test", DocumentType.TEXT),
            ConceptNode.create_mock_concept(user_id, "test_concept", ConceptType.ABSTRACT),
            CuriosityMission.create_mock_mission(user_id, "Test Mission", CuriosityType.EPISTEMIC),
            EvaluationFrame.create_mock_evaluation(user_id, EvaluationType.CONSCIOUSNESS_ASSESSMENT),
            VisualizationState.create_mock_state(user_id)
        ]

        # Verify all models have constitutional compliance
        for model in models:
            assert hasattr(model, 'mock_data_enabled')
            assert model.mock_data_enabled is True

    def test_model_serialization_compatibility(self):
        """Test that all models serialize properly"""
        user_id = "serialization_test_user"

        models = [
            UserProfile.create_mock_profile(user_id, "test"),
            AutobiographicalJourney(user_id=user_id, title="Test", mock_data=True),
            ThoughtSeedTrace.create_mock_trace(user_id),
            EventNode.create_mock_event(user_id, EventType.COGNITION, "Test"),
            DocumentArtifact.create_mock_document(user_id, "Test", DocumentType.TEXT),
            ConceptNode.create_mock_concept(user_id, "test_concept", ConceptType.ABSTRACT),
            CuriosityMission.create_mock_mission(user_id, "Test Mission", CuriosityType.EPISTEMIC),
            EvaluationFrame.create_mock_evaluation(user_id, EvaluationType.CONSCIOUSNESS_ASSESSMENT),
            VisualizationState.create_mock_state(user_id)
        ]

        for model in models:
            # Test to_dict method
            model_dict = model.to_dict()
            assert isinstance(model_dict, dict)
            assert 'user_id' in model_dict
            assert model_dict['user_id'] == user_id

    def test_enum_consistency_across_models(self):
        """Test that enum values are consistent across models"""

        # Test that all enum values are strings and lowercase/snake_case
        test_enums = [
            LearningStyle, ConsciousnessLevel,
            ConsciousnessState, MemoryContextType,
            EventType, MosaicObservationType,
            DocumentType, ProcessingStatus,
            ConceptType, ActivationPattern,
            CuriosityType, MissionStatus, ExplorationStrategy,
            EvaluationType, ConsentLevel,
            ViewType, ThemeType
        ]

        for enum_class in test_enums:
            for enum_value in enum_class:
                assert isinstance(enum_value.value, str)
                # Most enum values should be snake_case
                assert '_' in enum_value.value or enum_value.value.islower()

    def test_timestamp_handling_consistency(self):
        """Test consistent timestamp handling across models"""
        user_id = "timestamp_test_user"
        start_time = datetime.utcnow()

        models = [
            UserProfile.create_mock_profile(user_id, "test"),
            AutobiographicalJourney(user_id=user_id, title="Test", mock_data=True),
            ThoughtSeedTrace.create_mock_trace(user_id),
            EventNode.create_mock_event(user_id, EventType.COGNITION, "Test"),
            DocumentArtifact.create_mock_document(user_id, "Test", DocumentType.TEXT),
            ConceptNode.create_mock_concept(user_id, "test_concept", ConceptType.ABSTRACT),
            CuriosityMission.create_mock_mission(user_id, "Test Mission", CuriosityType.EPISTEMIC),
            EvaluationFrame.create_mock_evaluation(user_id, EvaluationType.CONSCIOUSNESS_ASSESSMENT),
            VisualizationState.create_mock_state(user_id)
        ]

        end_time = datetime.utcnow()

        for model in models:
            # All models should have created_at and updated_at
            assert hasattr(model, 'created_at')
            assert hasattr(model, 'updated_at')

            # Timestamps should be within test time range
            assert start_time <= model.created_at <= end_time
            assert start_time <= model.updated_at <= end_time

    def test_uuid_generation_uniqueness(self):
        """Test UUID generation uniqueness across models"""
        user_id = "uuid_test_user"

        # Create multiple instances of each model
        all_ids = []

        for _ in range(5):  # Create 5 of each
            models = [
                UserProfile.create_mock_profile(user_id, f"test_{_}"),
                AutobiographicalJourney(user_id=user_id, title=f"Test{_}", mock_data=True),
                ThoughtSeedTrace.create_mock_trace(user_id),
                EventNode.create_mock_event(user_id, EventType.COGNITION, f"Test{_}"),
                DocumentArtifact.create_mock_document(user_id, f"Test{_}", DocumentType.TEXT),
                ConceptNode.create_mock_concept(user_id, f"test_concept_{_}", ConceptType.ABSTRACT),
                CuriosityMission.create_mock_mission(user_id, f"Test Mission {_}", CuriosityType.EPISTEMIC),
                EvaluationFrame.create_mock_evaluation(user_id, EvaluationType.CONSCIOUSNESS_ASSESSMENT),
                VisualizationState.create_mock_state(user_id)
            ]

            for model in models:
                # Get the primary ID field (varies by model)
                if hasattr(model, 'user_id') and hasattr(model, 'journey_id'):
                    model_id = getattr(model, 'journey_id', None)
                elif hasattr(model, 'trace_id'):
                    model_id = model.trace_id
                elif hasattr(model, 'event_id'):
                    model_id = model.event_id
                elif hasattr(model, 'artifact_id'):
                    model_id = model.artifact_id
                elif hasattr(model, 'concept_id'):
                    model_id = model.concept_id
                elif hasattr(model, 'mission_id'):
                    model_id = model.mission_id
                elif hasattr(model, 'frame_id'):
                    model_id = model.frame_id
                elif hasattr(model, 'state_id'):
                    model_id = model.state_id
                else:
                    continue  # Skip models without clear ID field

                if model_id:
                    all_ids.append(model_id)

        # All IDs should be unique
        assert len(all_ids) == len(set(all_ids))

    def test_field_validation_across_models(self):
        """Test field validation consistency across models"""

        # Test required field validation
        with pytest.raises(ValueError):
            UserProfile(username="test")  # Missing user_id

        with pytest.raises(ValueError):
            AutobiographicalJourney(title="test")  # Missing user_id

        with pytest.raises(ValueError):
            ThoughtSeedTrace()  # Missing user_id

        with pytest.raises(ValueError):
            EventNode(user_id="test", event_name="test")  # Missing event_type

        with pytest.raises(ValueError):
            DocumentArtifact(user_id="test", title="test")  # Missing document_type

        with pytest.raises(ValueError):
            ConceptNode(user_id="test", concept_type=ConceptType.ABSTRACT)  # Missing concept_name

        with pytest.raises(ValueError):
            CuriosityMission(user_id="test", mission_title="test")  # Missing primary_curiosity_type

        with pytest.raises(ValueError):
            EvaluationFrame(user_id="test", evaluation_title="test")  # Missing evaluation_type and user_consent_level

        # VisualizationState only requires user_id, so should work
        viz_state = VisualizationState(user_id="test")
        assert viz_state.user_id == "test"


if __name__ == "__main__":
    pytest.main([__file__])