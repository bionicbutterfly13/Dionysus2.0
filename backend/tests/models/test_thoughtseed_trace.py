"""
Tests for ThoughtSeedTrace Model (T018)
TDD compliance tests for attractor dynamics and consciousness state tracking
"""

import pytest
from datetime import datetime, timedelta
from src.models.thoughtseed_trace import (
    ThoughtSeedTrace, ConsciousnessState, MemoryContextType,
    AttractorDynamicsPattern, MemoryContextData
)


class TestThoughtSeedTrace:
    """Test ThoughtSeedTrace model functionality"""

    def test_thoughtseed_trace_creation(self):
        """Test basic thoughtseed trace creation"""
        trace = ThoughtSeedTrace(user_id="test_user")

        assert trace.user_id == "test_user"
        assert trace.consciousness_state == ConsciousnessState.FOCUSED
        assert trace.primary_memory_context == MemoryContextType.WORKING
        assert trace.mock_data_enabled is True
        assert len(trace.attractor_patterns) == 0

    def test_attractor_pattern_addition(self):
        """Test adding attractor dynamics patterns"""
        trace = ThoughtSeedTrace(user_id="test_user")

        pattern_id = trace.add_attractor_pattern(
            "working_memory_pattern",
            strength=0.8,
            stability=0.7,
            memory_context=MemoryContextType.WORKING
        )

        assert len(trace.attractor_patterns) == 1
        assert trace.attractor_patterns[0].pattern_type == "working_memory_pattern"
        assert trace.attractor_patterns[0].strength == 0.8
        assert trace.attractor_patterns[0].stability == 0.7
        assert isinstance(pattern_id, str)

    def test_consciousness_state_update(self):
        """Test consciousness state updates with metrics"""
        trace = ThoughtSeedTrace(user_id="test_user")

        metrics = {
            "attention": 0.9,
            "awareness": 0.8,
            "integration": 0.7
        }

        trace.update_consciousness_state(ConsciousnessState.REFLECTIVE, metrics)

        assert trace.consciousness_state == ConsciousnessState.REFLECTIVE
        assert trace.consciousness_metrics["attention"] == 0.9
        assert trace.attention_focus == 0.9
        assert trace.awareness_level == 0.8

    def test_memory_context_addition(self):
        """Test adding memory contexts"""
        trace = ThoughtSeedTrace(user_id="test_user")

        context_data = {
            "recency_bias": 0.3,
            "contraction_bias": 0.2,
            "activation_threshold": 0.5
        }

        trace.add_memory_context(MemoryContextType.WORKING, context_data)

        assert len(trace.memory_contexts) == 1
        assert trace.memory_contexts[0].context_type == MemoryContextType.WORKING
        assert trace.memory_contexts[0].recency_bias == 0.3

    def test_memory_context_transition(self):
        """Test memory context transitions"""
        trace = ThoughtSeedTrace(user_id="test_user")

        trace.transition_memory_context(
            MemoryContextType.WORKING,
            MemoryContextType.EPISODIC,
            "Context switch due to pattern recognition"
        )

        assert len(trace.context_transitions) == 1
        assert trace.context_transitions[0]["from_context"] == "working"
        assert trace.context_transitions[0]["to_context"] == "episodic"
        assert trace.primary_memory_context == MemoryContextType.EPISODIC

    def test_temporal_dynamics_calculation(self):
        """Test temporal dynamics calculation for different memory contexts"""
        trace = ThoughtSeedTrace(user_id="test_user")

        # Test working memory dynamics
        working_dynamics = trace._calculate_temporal_dynamics("test_pattern", MemoryContextType.WORKING)
        assert "adaptation_rate" in working_dynamics
        assert working_dynamics["adaptation_rate"] == 0.8

        # Test episodic memory dynamics
        episodic_dynamics = trace._calculate_temporal_dynamics("test_pattern", MemoryContextType.EPISODIC)
        assert "switch_threshold" in episodic_dynamics
        assert episodic_dynamics["switch_threshold"] == 0.6

        # Test procedural memory dynamics
        procedural_dynamics = trace._calculate_temporal_dynamics("test_pattern", MemoryContextType.PROCEDURAL)
        assert "convergence_bias" in procedural_dynamics
        assert procedural_dynamics["convergence_bias"] == 0.3

    def test_integration_score_calculation(self):
        """Test information integration score calculation"""
        trace = ThoughtSeedTrace(user_id="test_user")

        # Set consciousness metrics
        trace.consciousness_metrics = {
            "attention": 0.8,
            "awareness": 0.7,
            "integration": 0.6
        }

        # Add attractor pattern for coherence boost
        trace.add_attractor_pattern("test_pattern", 0.8, 0.9)
        trace.pattern_stability = 0.8

        integration_score = trace.calculate_integration_score()

        assert integration_score > 0.0
        assert integration_score <= 1.0
        assert trace.integration_score == integration_score

    def test_emergence_detection(self):
        """Test consciousness emergence detection"""
        trace = ThoughtSeedTrace(user_id="test_user")

        # Set up conditions for emergence detection
        trace.consciousness_metrics = {"integration": 0.8, "awareness": 0.9}
        trace.pattern_stability = 0.9
        trace.temporal_thickness = 0.6

        # Add multiple memory contexts
        trace.add_memory_context(MemoryContextType.WORKING)
        trace.add_memory_context(MemoryContextType.EPISODIC)

        indicators = trace.detect_emergence()

        assert len(indicators) > 0
        assert "high_integration" in indicators
        assert "stable_attractors" in indicators
        assert "multi_context_activation" in indicators
        assert "temporal_depth" in indicators

    def test_attractor_state_retrieval(self):
        """Test current attractor state retrieval"""
        trace = ThoughtSeedTrace(user_id="test_user")

        # Add some patterns
        trace.add_attractor_pattern("pattern1", 0.7, 0.8)
        trace.add_attractor_pattern("pattern2", 0.6, 0.9)
        trace.pattern_stability = 0.85

        state = trace.get_current_attractor_state()

        assert state["pattern_count"] == 2
        assert 0.6 <= state["avg_strength"] <= 0.7
        assert 0.8 <= state["avg_stability"] <= 0.9
        assert state["convergence_score"] == 0.85

    def test_mock_trace_creation(self):
        """Test mock trace creation for development"""
        mock_trace = ThoughtSeedTrace.create_mock_trace("mock_user", "mock_journey")

        assert mock_trace.user_id == "mock_user"
        assert mock_trace.journey_id == "mock_journey"
        assert mock_trace.mock_data_enabled is True
        assert len(mock_trace.attractor_patterns) > 0
        assert len(mock_trace.memory_contexts) > 0
        assert len(mock_trace.consciousness_metrics) > 0

    def test_trace_serialization(self):
        """Test trace serialization to dict"""
        trace = ThoughtSeedTrace(user_id="test_user")
        trace_dict = trace.to_dict()

        assert isinstance(trace_dict, dict)
        assert trace_dict["user_id"] == "test_user"
        assert "attractor_patterns" in trace_dict

    def test_enum_validation(self):
        """Test enum validation for consciousness state and memory context"""
        with pytest.raises(ValueError):
            ThoughtSeedTrace(
                user_id="test",
                consciousness_state="invalid_state"
            )

        trace = ThoughtSeedTrace(user_id="test")
        with pytest.raises(ValueError):
            trace.add_memory_context("invalid_context")


class TestAttractorDynamicsPattern:
    """Test AttractorDynamicsPattern functionality"""

    def test_pattern_creation(self):
        """Test attractor pattern creation"""
        pattern = AttractorDynamicsPattern(
            pattern_type="test_pattern",
            strength=0.8,
            stability=0.9
        )

        assert pattern.pattern_type == "test_pattern"
        assert pattern.strength == 0.8
        assert pattern.stability == 0.9
        assert pattern.convergence_time == 0.0


class TestMemoryContextData:
    """Test MemoryContextData functionality"""

    def test_working_memory_context(self):
        """Test working memory context data"""
        context = MemoryContextData(
            context_type=MemoryContextType.WORKING,
            recency_bias=0.3,
            contraction_bias=0.2
        )

        assert context.context_type == MemoryContextType.WORKING
        assert context.recency_bias == 0.3
        assert context.contraction_bias == 0.2

    def test_episodic_memory_context(self):
        """Test episodic memory context data"""
        context = MemoryContextData(
            context_type=MemoryContextType.EPISODIC,
            context_switch_threshold=0.7,
            global_remapping_strength=0.8
        )

        assert context.context_type == MemoryContextType.EPISODIC
        assert context.context_switch_threshold == 0.7
        assert context.global_remapping_strength == 0.8

    def test_procedural_memory_context(self):
        """Test procedural memory context data"""
        context = MemoryContextData(
            context_type=MemoryContextType.PROCEDURAL,
            asymmetric_convergence=0.4,
            experience_weight=0.6
        )

        assert context.context_type == MemoryContextType.PROCEDURAL
        assert context.asymmetric_convergence == 0.4
        assert context.experience_weight == 0.6