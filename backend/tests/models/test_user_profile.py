"""
Tests for UserProfile Model (T013)
TDD compliance tests for consciousness development tracking
"""

import pytest
from datetime import datetime, timedelta
from src.models.user_profile import UserProfile, LearningStyle, ConsciousnessLevel


class TestUserProfile:
    """Test UserProfile model functionality"""

    def test_user_profile_creation(self):
        """Test basic user profile creation"""
        profile = UserProfile(
            user_id="test_user_1",
            username="test_user",
            email="test@example.com"
        )

        assert profile.user_id == "test_user_1"
        assert profile.username == "test_user"
        assert profile.email == "test@example.com"
        assert profile.learning_style == LearningStyle.MULTIMODAL
        assert profile.consciousness_level == ConsciousnessLevel.EMERGING
        assert profile.mock_data_enabled is True

    def test_consciousness_metrics_update(self):
        """Test consciousness metrics updating"""
        profile = UserProfile(user_id="test_user", username="test")

        metrics = {
            "attention": 0.7,
            "awareness": 0.6,
            "integration": 0.5
        }

        profile.update_consciousness_metrics(metrics)

        assert profile.consciousness_metrics["attention"] == 0.7
        assert profile.consciousness_metrics["awareness"] == 0.6
        assert profile.consciousness_metrics["integration"] == 0.5

    def test_consciousness_level_advancement(self):
        """Test consciousness level advancement logic"""
        profile = UserProfile(user_id="test_user", username="test")
        profile.consciousness_level = ConsciousnessLevel.EMERGING

        # Set metrics above threshold for EMERGING (0.3)
        profile.update_consciousness_metrics({
            "attention": 0.4,
            "awareness": 0.4,
            "integration": 0.4
        })

        advanced = profile.advance_consciousness_level()
        assert advanced is True
        assert profile.consciousness_level == ConsciousnessLevel.DEVELOPING

    def test_consciousness_level_no_advancement(self):
        """Test consciousness level doesn't advance prematurely"""
        profile = UserProfile(user_id="test_user", username="test")
        profile.consciousness_level = ConsciousnessLevel.EMERGING

        # Set metrics below threshold
        profile.update_consciousness_metrics({
            "attention": 0.2,
            "awareness": 0.2,
            "integration": 0.2
        })

        advanced = profile.advance_consciousness_level()
        assert advanced is False
        assert profile.consciousness_level == ConsciousnessLevel.EMERGING

    def test_thoughtseed_trace_counter(self):
        """Test thoughtseed trace counter increment"""
        profile = UserProfile(user_id="test_user", username="test")
        initial_count = profile.thoughtseed_traces_count

        profile.add_thoughtseed_trace()
        assert profile.thoughtseed_traces_count == initial_count + 1

    def test_learning_preferences_update(self):
        """Test learning preferences update"""
        profile = UserProfile(user_id="test_user", username="test")

        topics = ["consciousness", "neuroscience", "ai"]
        profile.set_learning_preferences(LearningStyle.VISUAL, topics)

        assert profile.learning_style == LearningStyle.VISUAL
        assert profile.preferred_topics == topics

    def test_activity_timestamp_update(self):
        """Test activity timestamp updates"""
        profile = UserProfile(user_id="test_user", username="test")
        old_timestamp = profile.last_active

        # Small delay to ensure timestamp difference
        import time
        time.sleep(0.001)

        profile.update_activity_timestamp()

        if old_timestamp is not None:
            assert profile.last_active != old_timestamp
            assert profile.last_active > old_timestamp
        else:
            # last_active was None initially
            assert profile.last_active is not None

    def test_mock_profile_creation(self):
        """Test mock profile creation for development"""
        mock_profile = UserProfile.create_mock_profile("mock_user", "mock_username")

        assert mock_profile.user_id == "mock_user"
        assert mock_profile.username == "mock_username"
        assert mock_profile.mock_data_enabled is True
        assert len(mock_profile.preferred_topics) > 0
        assert len(mock_profile.consciousness_metrics) > 0

    def test_profile_serialization(self):
        """Test profile serialization to dict"""
        profile = UserProfile(user_id="test_user", username="test")
        profile_dict = profile.to_dict()

        assert isinstance(profile_dict, dict)
        assert profile_dict["user_id"] == "test_user"
        assert profile_dict["username"] == "test"

    def test_enum_validation(self):
        """Test enum validation for learning style and consciousness level"""
        with pytest.raises(ValueError):
            UserProfile(
                user_id="test",
                username="test",
                learning_style="invalid_style"
            )

        with pytest.raises(ValueError):
            UserProfile(
                user_id="test",
                username="test",
                consciousness_level="invalid_level"
            )