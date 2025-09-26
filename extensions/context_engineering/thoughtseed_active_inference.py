#!/usr/bin/env python3
"""
🌱🧠 Thoughtseed Active Inference for ASI-Arch
==============================================

Incremental implementation of Thoughtseed framework with active inference
for conscious neural architecture discovery in ASI-Arch.

This implements the core Thoughtseed classes we discussed:
- Thoughtseed: Autonomous cognitive subagents
- NeuronalPacket: Discrete processing units
- ActiveInferenceThoughtseedBridge: Connects active inference to thoughtseeds
- WorldModelTheory: Self-modeling capabilities

Author: ASI-Arch Context Engineering Extension
Date: 2025-09-22
Version: 1.0.0 - Incremental Implementation
"""

import asyncio
import json
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Core Enums (defined early for use throughout the module)
# =============================================================================

class ThoughtseedType(Enum):
    """Types of thoughtseeds in the hierarchical cognitive architecture"""
    SENSORIMOTOR = "sensorimotor"      # Direct action-perception loops
    PERCEPTUAL = "perceptual"          # Pattern recognition and categorization
    CONCEPTUAL = "conceptual"          # Abstract concept formation
    ABSTRACT = "abstract"              # High-level reasoning
    METACOGNITIVE = "metacognitive"    # Self-awareness and monitoring

class EvolutionaryPriorType(Enum):
    """Four types of evolutionary priors from ThoughtSeed theory"""
    BASAL = "basal"                    # Basic survival and homeostatic patterns
    LINEAGE_SPECIFIC = "lineage_specific"  # Species/clade-specific adaptations
    DISPOSITIONAL = "dispositional"    # Individual temperament and predispositions
    LEARNED = "learned"                # Experience-dependent acquired patterns

# =============================================================================
# Core Data Classes
# =============================================================================

@dataclass
class NeuronalPacket:
    """Discrete unit of cognitive processing in the thoughtseed network"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: Dict[str, Any] = field(default_factory=dict)
    activation_level: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    source_thoughtseed: Optional[str] = None
    target_thoughtseeds: Set[str] = field(default_factory=set)
    processing_priority: int = 0
    prediction_error: float = 0.0
    surprise: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'content': self.content,
            'activation_level': self.activation_level,
            'timestamp': self.timestamp.isoformat(),
            'source_thoughtseed': self.source_thoughtseed,
            'target_thoughtseeds': list(self.target_thoughtseeds),
            'processing_priority': self.processing_priority,
            'prediction_error': self.prediction_error,
            'surprise': self.surprise
        }

# =============================================================================
# Forward References and Supporting Classes
# =============================================================================

@dataclass
class EvolutionaryPrior:
    """Represents an evolutionary prior with hierarchical organization"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: EvolutionaryPriorType = EvolutionaryPriorType.LEARNED
    strength: float = 0.5
    activation_threshold: float = 0.3
    context_specificity: Dict[str, float] = field(default_factory=dict)
    temporal_dynamics: Dict[str, float] = field(default_factory=dict)
    hierarchical_level: int = 0
    parent_priors: Set[str] = field(default_factory=set)
    child_priors: Set[str] = field(default_factory=set)

    def evaluate_context_relevance(self, context: Dict[str, Any]) -> float:
        """Evaluate how relevant this prior is in the given context"""
        relevance_score = 0.0
        total_weight = 0.0

        for context_key, context_value in context.items():
            if context_key in self.context_specificity:
                weight = self.context_specificity[context_key]
                if isinstance(context_value, (int, float)):
                    similarity = 1.0 - abs(context_value - weight)
                    relevance_score += similarity * weight
                else:
                    if str(context_value) == str(weight):
                        relevance_score += weight
                total_weight += weight

        return relevance_score / total_weight if total_weight > 0 else self.strength

    def compute_activation_probability(self, input_strength: float, context: Dict[str, Any]) -> float:
        """Compute probability of prior activation given input and context"""
        context_relevance = self.evaluate_context_relevance(context)
        weighted_input = input_strength * context_relevance * self.strength
        activation_prob = 1.0 / (1.0 + np.exp(-(weighted_input - self.activation_threshold) * 10))

        if self.hierarchical_level > 0:
            hierarchical_dampening = 0.9 ** self.hierarchical_level
            activation_prob *= hierarchical_dampening

        return min(1.0, max(0.0, activation_prob))

    def update_dynamics(self, activation_level: float, learning_rate: float = 0.01):
        """Update temporal dynamics based on activation"""
        if 'activation_history' not in self.temporal_dynamics:
            self.temporal_dynamics['activation_history'] = []

        self.temporal_dynamics['activation_history'].append(activation_level)
        history = self.temporal_dynamics['activation_history']

        if len(history) > 1:
            self.temporal_dynamics['mean_activation'] = np.mean(history[-20:])
            self.temporal_dynamics['activation_variance'] = np.var(history[-20:])

        if activation_level > self.activation_threshold:
            self.strength = min(1.0, self.strength + learning_rate * 0.1)
        else:
            self.strength = max(0.1, self.strength - learning_rate * 0.05)

        if len(history) > 100:
            self.temporal_dynamics['activation_history'] = history[-50:]

@dataclass
class NestedMarkovBlanket:
    """Nested Markov blanket for hierarchical organization"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    nesting_level: int = 0
    internal_states: Set[str] = field(default_factory=set)
    external_states: Set[str] = field(default_factory=set)
    sensory_states: Set[str] = field(default_factory=set)
    active_states: Set[str] = field(default_factory=set)
    parent_blanket_id: Optional[str] = None
    child_blanket_ids: Set[str] = field(default_factory=set)

    def compute_information_flow(self, direction: str = "inward") -> float:
        """Compute information flow across the Markov blanket"""
        if direction == "inward":
            flow = len(self.sensory_states) / (len(self.sensory_states) + len(self.internal_states) + 1)
        else:
            flow = len(self.active_states) / (len(self.active_states) + len(self.internal_states) + 1)
        return flow

    def update_blanket_states(self, new_internal: Set[str], new_external: Set[str]):
        """Update the states within the Markov blanket"""
        self.internal_states.update(new_internal)
        self.external_states.update(new_external)
        self.sensory_states.update(new_external)
        self.active_states.update(new_internal)

@dataclass
class PullbackAttractor:
    """Pullback attractor for neurodynamic evolution"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    basin_center: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5]))
    basin_width: float = 0.3
    pulling_strength: float = 0.7
    temporal_decay: float = 0.95

    def compute_attractor_force(self, current_state: np.ndarray) -> np.ndarray:
        """Compute attractor force on current state"""
        direction = self.basin_center - current_state
        distance = np.linalg.norm(direction)

        if distance > 0:
            normalized_direction = direction / distance
            force_magnitude = self.pulling_strength * np.exp(-distance / self.basin_width)
            return normalized_direction * force_magnitude
        else:
            return np.zeros_like(current_state)

    def is_within_basin(self, state: np.ndarray) -> bool:
        """Check if state is within attractor basin"""
        distance = np.linalg.norm(self.basin_center - state)
        return distance <= self.basin_width

# =============================================================================
# Embodied Cognition and Umwelt Components
# =============================================================================

@dataclass
class Umwelt:
    """Umwelt - Subjective perceptual world within which an organism acts and reacts"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    perceptual_channels: Dict[str, Any] = field(default_factory=dict)
    action_repertoire: Dict[str, Any] = field(default_factory=dict)
    environmental_affordances: List[str] = field(default_factory=list)
    sensorimotor_mapping: Dict[str, Dict[str, float]] = field(default_factory=dict)
    temporal_dynamics: Dict[str, float] = field(default_factory=dict)

    def perceive(self, environmental_state: Dict[str, Any]) -> Dict[str, Any]:
        """Perceive environmental state through embodied perceptual channels"""
        perception = {}

        for channel, config in self.perceptual_channels.items():
            if channel in environmental_state:
                raw_signal = environmental_state[channel]
                # Apply perceptual filtering based on embodiment
                sensitivity = config.get('sensitivity', 1.0)
                noise_level = config.get('noise', 0.1)

                # Add embodied perception characteristics
                filtered_signal = raw_signal * sensitivity
                if isinstance(filtered_signal, (int, float)):
                    filtered_signal += np.random.normal(0, noise_level)
                    filtered_signal = max(0.0, min(1.0, filtered_signal))

                perception[channel] = {
                    'signal': filtered_signal,
                    'confidence': config.get('confidence', 0.8),
                    'salience': self._compute_salience(channel, filtered_signal)
                }

        return perception

    def identify_affordances(self, environmental_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify action affordances in the current environment"""
        affordances = []

        for affordance_type in self.environmental_affordances:
            # Check if conditions for this affordance are met
            affordance_strength = self._evaluate_affordance(affordance_type, environmental_state)

            if affordance_strength > 0.3:  # Threshold for perceivable affordances
                affordances.append({
                    'type': affordance_type,
                    'strength': affordance_strength,
                    'action_potential': self._compute_action_potential(affordance_type),
                    'embodiment_compatibility': self._check_embodiment_compatibility(affordance_type)
                })

        return sorted(affordances, key=lambda x: x['strength'], reverse=True)

    def _compute_salience(self, channel: str, signal: Any) -> float:
        """Compute perceptual salience based on embodied characteristics"""
        # Simplified salience computation
        if isinstance(signal, (int, float)):
            return min(1.0, abs(signal - 0.5) * 2.0)  # Higher salience for extreme values
        return 0.5

    def _evaluate_affordance(self, affordance_type: str, environmental_state: Dict[str, Any]) -> float:
        """Evaluate the strength of an affordance in the current environment"""
        # Simplified affordance evaluation
        relevance_score = 0.0

        if affordance_type == 'exploration' and 'novelty' in environmental_state:
            relevance_score = environmental_state['novelty']
        elif affordance_type == 'attention_focus' and 'task_relevance' in environmental_state:
            relevance_score = environmental_state['task_relevance']
        elif affordance_type == 'learning' and 'prediction_error' in environmental_state:
            relevance_score = environmental_state['prediction_error']

        return min(1.0, relevance_score)

    def _compute_action_potential(self, affordance_type: str) -> float:
        """Compute action potential for a given affordance"""
        return self.action_repertoire.get(affordance_type, {}).get('potential', 0.5)

    def _check_embodiment_compatibility(self, affordance_type: str) -> float:
        """Check how compatible an affordance is with current embodiment"""
        return self.action_repertoire.get(affordance_type, {}).get('compatibility', 0.5)

@dataclass
class EmbodiedCognitionComponent:
    """Component for embodied cognition processing in thoughtseeds"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    umwelt: Umwelt = field(default_factory=Umwelt)
    sensorimotor_history: List[Dict[str, Any]] = field(default_factory=list)
    motor_schema: Dict[str, Any] = field(default_factory=dict)
    embodiment_parameters: Dict[str, float] = field(default_factory=dict)

    def initialize_embodiment(self, thoughtseed_type: ThoughtseedType):
        """Initialize embodiment based on thoughtseed type"""
        # Configure Umwelt based on thoughtseed type
        if thoughtseed_type == ThoughtseedType.SENSORIMOTOR:
            self.umwelt.perceptual_channels = {
                'haptic': {'sensitivity': 1.2, 'noise': 0.05, 'confidence': 0.9},
                'proprioceptive': {'sensitivity': 1.0, 'noise': 0.03, 'confidence': 0.95},
                'motor_feedback': {'sensitivity': 1.1, 'noise': 0.08, 'confidence': 0.85}
            }
            self.umwelt.environmental_affordances = ['movement', 'manipulation', 'exploration']

        elif thoughtseed_type == ThoughtseedType.PERCEPTUAL:
            self.umwelt.perceptual_channels = {
                'visual_patterns': {'sensitivity': 1.3, 'noise': 0.1, 'confidence': 0.8},
                'auditory_patterns': {'sensitivity': 0.9, 'noise': 0.12, 'confidence': 0.75},
                'pattern_recognition': {'sensitivity': 1.5, 'noise': 0.05, 'confidence': 0.9}
            }
            self.umwelt.environmental_affordances = ['pattern_detection', 'categorization', 'recognition']

        elif thoughtseed_type == ThoughtseedType.CONCEPTUAL:
            self.umwelt.perceptual_channels = {
                'semantic_space': {'sensitivity': 1.0, 'noise': 0.15, 'confidence': 0.7},
                'abstract_relations': {'sensitivity': 1.2, 'noise': 0.1, 'confidence': 0.8},
                'conceptual_similarity': {'sensitivity': 1.1, 'noise': 0.12, 'confidence': 0.75}
            }
            self.umwelt.environmental_affordances = ['abstraction', 'reasoning', 'concept_formation']

        elif thoughtseed_type == ThoughtseedType.METACOGNITIVE:
            self.umwelt.perceptual_channels = {
                'self_monitoring': {'sensitivity': 1.4, 'noise': 0.08, 'confidence': 0.85},
                'cognitive_state': {'sensitivity': 1.2, 'noise': 0.1, 'confidence': 0.8},
                'meta_awareness': {'sensitivity': 1.5, 'noise': 0.05, 'confidence': 0.9}
            }
            self.umwelt.environmental_affordances = ['monitoring', 'control', 'reflection']

        # Set embodiment parameters
        self.embodiment_parameters = {
            'sensorimotor_coupling': 0.8,
            'environmental_sensitivity': 0.7,
            'action_readiness': 0.6,
            'temporal_binding': 0.75
        }

    def process_environmental_interaction(self, environmental_state: Dict[str, Any],
                                        current_packet: NeuronalPacket) -> Dict[str, Any]:
        """Process interaction with environment through embodied perception-action cycle"""
        # Perceive environment through Umwelt
        perception = self.umwelt.perceive(environmental_state)

        # Identify affordances
        affordances = self.umwelt.identify_affordances(environmental_state)

        # Update sensorimotor history
        sensorimotor_event = {
            'timestamp': datetime.now(),
            'perception': perception,
            'affordances': affordances,
            'packet_content': current_packet.content,
            'environmental_state': environmental_state
        }
        self.sensorimotor_history.append(sensorimotor_event)

        # Keep history manageable
        if len(self.sensorimotor_history) > 50:
            self.sensorimotor_history = self.sensorimotor_history[-50:]

        # Compute embodied response
        embodied_response = self._compute_embodied_response(perception, affordances, current_packet)

        return {
            'perception': perception,
            'affordances': affordances,
            'embodied_response': embodied_response,
            'sensorimotor_coupling': self._compute_sensorimotor_coupling(),
            'environmental_engagement': self._compute_environmental_engagement(affordances)
        }

    def _compute_embodied_response(self, perception: Dict[str, Any],
                                 affordances: List[Dict[str, Any]],
                                 current_packet: NeuronalPacket) -> Dict[str, Any]:
        """Compute embodied response based on perception and affordances"""
        response = {
            'action_tendency': 0.0,
            'exploration_drive': 0.0,
            'attention_modulation': 0.0,
            'learning_signal': 0.0
        }

        # Compute action tendency based on strongest affordances
        if affordances:
            strongest_affordance = affordances[0]
            response['action_tendency'] = strongest_affordance['strength'] * strongest_affordance['action_potential']

        # Compute exploration drive based on environmental novelty
        novelty_signals = [p.get('salience', 0.0) for p in perception.values()]
        if novelty_signals:
            response['exploration_drive'] = sum(novelty_signals) / len(novelty_signals)

        # Compute attention modulation based on perceptual confidence
        confidence_signals = [p.get('confidence', 0.0) for p in perception.values()]
        if confidence_signals:
            response['attention_modulation'] = sum(confidence_signals) / len(confidence_signals)

        # Compute learning signal based on prediction error
        response['learning_signal'] = current_packet.prediction_error

        return response

    def _compute_sensorimotor_coupling(self) -> float:
        """Compute strength of sensorimotor coupling"""
        if len(self.sensorimotor_history) < 2:
            return 0.5

        # Analyze consistency in sensorimotor patterns
        recent_events = self.sensorimotor_history[-5:]
        coupling_strengths = []

        for event in recent_events:
            perception_strength = sum(p.get('salience', 0.0) for p in event['perception'].values())
            affordance_strength = sum(a['strength'] for a in event['affordances'])

            # Coupling is stronger when perception and affordances are aligned
            coupling = min(1.0, perception_strength * affordance_strength)
            coupling_strengths.append(coupling)

        return sum(coupling_strengths) / len(coupling_strengths) if coupling_strengths else 0.5

    def _compute_environmental_engagement(self, affordances: List[Dict[str, Any]]) -> float:
        """Compute level of engagement with environment"""
        if not affordances:
            return 0.0

        # Engagement based on number and strength of perceived affordances
        total_affordance_strength = sum(a['strength'] for a in affordances)
        affordance_diversity = len(set(a['type'] for a in affordances))

        engagement = min(1.0, (total_affordance_strength + affordance_diversity * 0.1) / 2.0)
        return engagement

# =============================================================================
# 4-Layer Architecture Components
# =============================================================================

@dataclass
class NeuronalPacketDispatcher:
    """NPD - Neuronal Packet Dispatcher for routing neural packets"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    routing_rules: Dict[str, Any] = field(default_factory=dict)
    packet_buffer: List[NeuronalPacket] = field(default_factory=list)
    dispatch_history: List[Dict[str, Any]] = field(default_factory=list)
    priority_queue: List[Tuple[int, str]] = field(default_factory=list)  # (priority, packet_id)

    def route_packet(self, packet: NeuronalPacket, knowledge_domains: Dict[str, 'KnowledgeDomain']) -> List[str]:
        """Route packet to appropriate knowledge domains"""
        target_domains = []

        # Route based on content type
        content_type = packet.content.get('type', 'unknown')

        if content_type in ['architecture', 'neural_network', 'design']:
            target_domains.extend([kd_id for kd_id, kd in knowledge_domains.items()
                                 if kd.domain_type == 'architecture'])
        elif content_type in ['performance', 'evaluation', 'metrics']:
            target_domains.extend([kd_id for kd_id, kd in knowledge_domains.items()
                                 if kd.domain_type == 'performance'])
        elif content_type in ['meta', 'learning', 'strategy']:
            target_domains.extend([kd_id for kd_id, kd in knowledge_domains.items()
                                 if kd.domain_type == 'meta_learning'])

        # Default routing - send to all domains if no specific match
        if not target_domains:
            target_domains = list(knowledge_domains.keys())

        # Record dispatch
        self.dispatch_history.append({
            'packet_id': packet.id,
            'targets': target_domains,
            'timestamp': datetime.now(),
            'content_type': content_type
        })

        return target_domains

@dataclass
class KnowledgeDomain:
    """KD - Knowledge Domain for organizing thoughtseeds by expertise"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    domain_type: str = "general"  # architecture, performance, meta_learning, etc.
    thoughtseed_ids: Set[str] = field(default_factory=set)
    knowledge_patterns: Dict[str, Any] = field(default_factory=dict)
    domain_expertise: float = 0.5
    interaction_matrix: Dict[str, float] = field(default_factory=dict)  # domain_id -> interaction_strength

    def add_thoughtseed(self, thoughtseed_id: str):
        """Add a thoughtseed to this knowledge domain"""
        self.thoughtseed_ids.add(thoughtseed_id)

    def remove_thoughtseed(self, thoughtseed_id: str):
        """Remove a thoughtseed from this knowledge domain"""
        self.thoughtseed_ids.discard(thoughtseed_id)

    def process_domain_packet(self, packet: NeuronalPacket, thoughtseeds: Dict[str, 'Thoughtseed']) -> List[NeuronalPacket]:
        """Process packet through all thoughtseeds in this domain"""
        responses = []

        for ts_id in self.thoughtseed_ids:
            if ts_id in thoughtseeds:
                thoughtseed = thoughtseeds[ts_id]
                ts_responses = thoughtseed.process_packet(packet)
                responses.extend(ts_responses)

        # Update domain knowledge patterns
        self._update_domain_knowledge(packet, responses)

        return responses

    def _update_domain_knowledge(self, packet: NeuronalPacket, responses: List[NeuronalPacket]):
        """Update domain knowledge based on packet processing"""
        # Extract patterns from packet and responses
        content_type = packet.content.get('type', 'unknown')

        if content_type not in self.knowledge_patterns:
            self.knowledge_patterns[content_type] = {
                'count': 0,
                'avg_activation': 0.0,
                'response_patterns': []
            }

        # Update statistics
        pattern = self.knowledge_patterns[content_type]
        pattern['count'] += 1
        pattern['avg_activation'] = (pattern['avg_activation'] * (pattern['count'] - 1) + packet.activation_level) / pattern['count']

        # Store response pattern (simplified)
        if responses:
            avg_response_activation = sum(r.activation_level for r in responses) / len(responses)
            pattern['response_patterns'].append(avg_response_activation)

            # Keep only recent patterns
            if len(pattern['response_patterns']) > 100:
                pattern['response_patterns'] = pattern['response_patterns'][-100:]

# =============================================================================
# Core Thoughtseed Classes
# =============================================================================

@dataclass
class EvolutionaryPrior:
    """Represents an evolutionary prior with hierarchical organization"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: EvolutionaryPriorType = EvolutionaryPriorType.BASAL
    strength: float = 1.0  # Prior strength/confidence
    patterns: Dict[str, Any] = field(default_factory=dict)  # Pattern templates
    activation_threshold: float = 0.5
    markov_blanket: Dict[str, Any] = field(default_factory=dict)  # Statistical boundary
    nesting_level: int = 0  # Hierarchical nesting depth
    parent_prior_id: Optional[str] = None
    child_prior_ids: Set[str] = field(default_factory=set)

    def calculate_prior_probability(self, observation: Dict[str, Any]) -> float:
        """Calculate prior probability for an observation"""
        if not self.patterns:
            return 0.5  # Neutral prior

        # Simple pattern matching for prior calculation
        match_score = 0.0
        pattern_count = len(self.patterns)

        for pattern_key, pattern_value in self.patterns.items():
            if pattern_key in observation:
                obs_value = observation[pattern_key]
                if isinstance(pattern_value, (int, float)) and isinstance(obs_value, (int, float)):
                    # Numerical similarity
                    similarity = 1.0 - abs(pattern_value - obs_value) / max(abs(pattern_value), abs(obs_value), 1.0)
                    match_score += similarity
                elif pattern_value == obs_value:
                    # Exact match
                    match_score += 1.0
                else:
                    # String/categorical similarity (simplified)
                    if str(pattern_value).lower() in str(obs_value).lower():
                        match_score += 0.5

        if pattern_count > 0:
            prior_prob = (match_score / pattern_count) * self.strength
            return min(1.0, max(0.0, prior_prob))

        return 0.5


@dataclass
class NestedMarkovBlanket:
    """Nested Markov blanket for hierarchical organization"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    nesting_level: int = 0
    internal_states: Set[str] = field(default_factory=set)
    external_states: Set[str] = field(default_factory=set)
    sensory_states: Set[str] = field(default_factory=set)
    active_states: Set[str] = field(default_factory=set)
    parent_blanket_id: Optional[str] = None
    child_blanket_ids: Set[str] = field(default_factory=set)

    def compute_free_energy(self, internal_beliefs: Dict[str, Any], observations: Dict[str, Any]) -> float:
        """Compute free energy for this Markov blanket"""
        # Simplified free energy calculation
        energy = 0.0

        for state_id in self.sensory_states:
            if state_id in observations and state_id in internal_beliefs:
                predicted = internal_beliefs[state_id]
                observed = observations[state_id]
                if isinstance(predicted, (int, float)) and isinstance(observed, (int, float)):
                    energy += abs(predicted - observed) ** 2

        return energy / max(len(self.sensory_states), 1)

@dataclass
class PullbackAttractor:
    """Pullback attractor for thoughtseed dynamics"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    basin_center: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5]))  # 2D for simplicity
    basin_width: float = 0.3
    pulling_strength: float = 1.0
    temporal_decay: float = 0.95  # How quickly it decays over time
    stability_threshold: float = 0.8

    def compute_attractor_force(self, current_state: np.ndarray) -> np.ndarray:
        """Compute the attractor force pulling toward the basin center"""
        if len(current_state) != len(self.basin_center):
            # Pad or truncate to match dimensions
            if len(current_state) < len(self.basin_center):
                current_state = np.pad(current_state, (0, len(self.basin_center) - len(current_state)))
            else:
                current_state = current_state[:len(self.basin_center)]

        # Calculate distance to basin center
        distance_vector = self.basin_center - current_state
        distance = np.linalg.norm(distance_vector)

        # Apply pulling force (stronger when further from center)
        if distance > 0:
            normalized_direction = distance_vector / distance
            force_magnitude = self.pulling_strength * min(1.0, distance / self.basin_width)
            return normalized_direction * force_magnitude
        else:
            return np.zeros_like(self.basin_center)

    def is_in_basin(self, state: np.ndarray) -> bool:
        """Check if state is within the attractor basin"""
        if len(state) != len(self.basin_center):
            return False
        distance = np.linalg.norm(self.basin_center - state)
        return distance <= self.basin_width

@dataclass
class Thoughtseed:
    """An autonomous cognitive subagent with attractor dynamics and evolutionary priors"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: ThoughtseedType = ThoughtseedType.CONCEPTUAL
    activation_state: float = 0.0
    attractor_strength: float = 1.0
    markov_blanket: NestedMarkovBlanket = field(default_factory=NestedMarkovBlanket)
    memory_buffer: List[NeuronalPacket] = field(default_factory=list)
    connections: Dict[str, float] = field(default_factory=dict)
    embodiment_state: Dict[str, Any] = field(default_factory=dict)

    # Evolutionary priors system
    evolutionary_priors: Dict[str, EvolutionaryPrior] = field(default_factory=dict)
    prior_hierarchy: Dict[int, List[str]] = field(default_factory=dict)  # level -> prior_ids

    # Active inference properties
    prediction_model: Optional[Dict[str, Any]] = None
    world_model: Optional[Dict[str, Any]] = None
    surprise_threshold: float = 0.5
    learning_rate: float = 0.1

    # Inner screen for consciousness projection
    inner_screen_content: Optional[Dict[str, Any]] = None
    consciousness_threshold: float = 0.7

    # Pullback attractor dynamics
    pullback_attractors: Dict[str, PullbackAttractor] = field(default_factory=dict)
    current_attractor_state: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5]))
    attractor_history: List[np.ndarray] = field(default_factory=list)
    temporal_dynamics: Dict[str, float] = field(default_factory=dict)
    
    def process_packet(self, packet: NeuronalPacket) -> List[NeuronalPacket]:
        """Process an incoming neuronal packet and generate responses with pullback attractor dynamics"""
        # Update activation based on packet content and prediction error
        prediction_error = self._calculate_prediction_error(packet)
        surprise = self._calculate_surprise(packet)

        # Update packet with calculated values
        packet.prediction_error = prediction_error
        packet.surprise = surprise

        # Update activation state with attractor dynamics
        external_input = packet.activation_level * self.attractor_strength
        self.update_attractor_dynamics(external_input)

        # Store in memory buffer
        self.memory_buffer.append(packet)

        # Keep only recent packets (last 10)
        if len(self.memory_buffer) > 10:
            self.memory_buffer = self.memory_buffer[-10:]

        # Update world model
        self.update_world_model(packet)

        # Update inner screen if this is the dominant content
        if self.activation_state > self.consciousness_threshold:
            self.update_inner_screen(packet.content)

        # Generate response packets based on active inference and attractor state
        response_packets = self._generate_response_packets(packet, prediction_error, surprise)

        return response_packets
    
    def add_evolutionary_prior(self, prior: EvolutionaryPrior):
        """Add an evolutionary prior to this thoughtseed"""
        self.evolutionary_priors[prior.id] = prior

        # Organize in hierarchy
        level = prior.nesting_level
        if level not in self.prior_hierarchy:
            self.prior_hierarchy[level] = []
        self.prior_hierarchy[level].append(prior.id)

    def calculate_prior_weighted_prediction(self, observation: Dict[str, Any]) -> float:
        """Calculate prediction using evolutionary priors"""
        if not self.evolutionary_priors:
            return 0.5

        weighted_prediction = 0.0
        total_weight = 0.0

        for prior in self.evolutionary_priors.values():
            prior_prob = prior.calculate_prior_probability(observation)
            weight = prior.strength
            weighted_prediction += prior_prob * weight
            total_weight += weight

        if total_weight > 0:
            return weighted_prediction / total_weight
        return 0.5

    def update_inner_screen(self, dominant_content: Dict[str, Any]):
        """Update inner screen with dominant thoughtseed content (Inner Screen model)"""
        # Project current dominant content to inner screen
        self.inner_screen_content = {
            'content': dominant_content,
            'timestamp': datetime.now(),
            'consciousness_level': self.activation_state,
            'thoughtseed_id': self.id,
            'attractor_state': self.attractor_strength,
            'evolutionary_prior_influence': self._get_prior_influence_summary(),
            'markov_blanket_level': self.markov_blanket.nesting_level,
            'phenomenological_quality': self._compute_phenomenological_quality(dominant_content)
        }

    def _get_prior_influence_summary(self) -> Dict[str, float]:
        """Get summary of evolutionary prior influences on current state"""
        influence_summary = {}
        for prior_type in EvolutionaryPriorType:
            type_priors = [p for p in self.evolutionary_priors.values() if p.type == prior_type]
            if type_priors:
                avg_strength = sum(p.strength for p in type_priors) / len(type_priors)
                influence_summary[prior_type.value] = avg_strength
            else:
                influence_summary[prior_type.value] = 0.0
        return influence_summary

    def _compute_phenomenological_quality(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Compute phenomenological qualities of conscious experience"""
        return {
            'clarity': self.activation_state,
            'emotional_valence': self._extract_emotional_valence(content),
            'attention_focus': min(1.0, self.attractor_strength),
            'embodiment_level': self._compute_embodiment_level(),
            'temporal_coherence': self._compute_temporal_coherence()
        }

    def _extract_emotional_valence(self, content: Dict[str, Any]) -> float:
        """Extract emotional valence from content (simplified)"""
        # Simple heuristic based on content type
        content_type = content.get('type', 'neutral')
        if 'positive' in str(content_type).lower() or 'success' in str(content_type).lower():
            return 0.7
        elif 'negative' in str(content_type).lower() or 'error' in str(content_type).lower():
            return -0.7
        return 0.0

    def _compute_embodiment_level(self) -> float:
        """Compute embodiment level based on sensorimotor connections"""
        embodiment_factors = self.embodiment_state.get('sensorimotor_connections', 0.5)
        return min(1.0, float(embodiment_factors))

    def _compute_temporal_coherence(self) -> float:
        """Compute temporal coherence based on memory buffer consistency"""
        if len(self.memory_buffer) < 2:
            return 0.5

        # Calculate consistency across recent activations
        recent_activations = [p.activation_level for p in self.memory_buffer[-5:]]
        variance = np.var(recent_activations) if recent_activations else 0.5
        coherence = 1.0 / (1.0 + variance)  # Higher coherence = lower variance
        return min(1.0, coherence)

    def initialize_pullback_attractors(self):
        """Initialize pullback attractors based on thoughtseed type and evolutionary priors"""
        # Create core attractor
        core_attractor = PullbackAttractor(
            basin_center=np.array([0.7, 0.8]),  # High activation, high stability
            basin_width=0.2,
            pulling_strength=1.0,
            temporal_decay=0.95
        )
        self.pullback_attractors['core'] = core_attractor

        # Create type-specific attractors
        if self.type == ThoughtseedType.CONCEPTUAL:
            conceptual_attractor = PullbackAttractor(
                basin_center=np.array([0.8, 0.6]),  # High conceptual processing
                basin_width=0.25,
                pulling_strength=0.8
            )
            self.pullback_attractors['conceptual'] = conceptual_attractor

        elif self.type == ThoughtseedType.METACOGNITIVE:
            meta_attractor = PullbackAttractor(
                basin_center=np.array([0.6, 0.9]),  # High meta-awareness
                basin_width=0.3,
                pulling_strength=0.9
            )
            self.pullback_attractors['metacognitive'] = meta_attractor

        # Create evolutionary prior-based attractors
        for prior_type, prior in self.evolutionary_priors.items():
            if prior.strength > 0.5:  # Only strong priors get attractors
                prior_attractor = PullbackAttractor(
                    basin_center=np.array([prior.strength, prior.activation_threshold]),
                    basin_width=0.2 * prior.strength,
                    pulling_strength=prior.strength * 0.8
                )
                self.pullback_attractors[f'prior_{prior.type.value}'] = prior_attractor

    def update_attractor_dynamics(self, external_input: float = 0.0):
        """Update pullback attractor dynamics based on current state"""
        if not self.pullback_attractors:
            self.initialize_pullback_attractors()

        # Current state vector [activation, prediction_accuracy]
        prediction_accuracy = 1.0 - (self.prediction_model.get('last_error', 0.5) if self.prediction_model else 0.5)
        current_state = np.array([self.activation_state, prediction_accuracy])

        # Calculate combined attractor force
        total_force = np.zeros(2)
        active_attractors = 0

        for attractor_id, attractor in self.pullback_attractors.items():
            # Check if this attractor should be active
            if self._should_attractor_be_active(attractor_id, attractor):
                force = attractor.compute_attractor_force(current_state)
                weight = self._compute_attractor_weight(attractor_id)
                total_force += force * weight
                active_attractors += 1

        # Apply attractor force with external input
        if active_attractors > 0:
            total_force /= active_attractors  # Average force

        # Update state with attractor dynamics
        state_change = total_force * self.learning_rate + external_input * 0.1
        new_state = current_state + state_change

        # Ensure bounds [0, 1]
        new_state = np.clip(new_state, 0.0, 1.0)

        # Update thoughtseed state
        self.activation_state = new_state[0]
        if self.prediction_model:
            self.prediction_model['accuracy'] = new_state[1]

        # Store in history
        self.current_attractor_state = new_state.copy()
        self.attractor_history.append(new_state.copy())

        # Keep history manageable
        if len(self.attractor_history) > 100:
            self.attractor_history = self.attractor_history[-100:]

    def _should_attractor_be_active(self, attractor_id: str, attractor: PullbackAttractor) -> bool:
        """Determine if an attractor should be active based on context"""
        # Core attractor is always active
        if attractor_id == 'core':
            return True

        # Type-specific attractors are active when thoughtseed is processing relevant content
        if attractor_id.startswith(self.type.value):
            return self.activation_state > 0.3

        # Prior-based attractors are active when their conditions are met
        if attractor_id.startswith('prior_'):
            prior_type = attractor_id.split('_', 1)[1]
            for prior in self.evolutionary_priors.values():
                if prior.type.value == prior_type and prior.strength > 0.5:
                    return True

        return False

    def _compute_attractor_weight(self, attractor_id: str) -> float:
        """Compute weight for an attractor based on current context"""
        if attractor_id == 'core':
            return 1.0

        if attractor_id.startswith(self.type.value):
            return 0.8

        if attractor_id.startswith('prior_'):
            prior_type = attractor_id.split('_', 1)[1]
            for prior in self.evolutionary_priors.values():
                if prior.type.value == prior_type:
                    return prior.strength

        return 0.5

    def get_attractor_landscape_info(self) -> Dict[str, Any]:
        """Get information about the current attractor landscape"""
        active_attractors = []
        for attractor_id, attractor in self.pullback_attractors.items():
            if self._should_attractor_be_active(attractor_id, attractor):
                distance_to_center = np.linalg.norm(self.current_attractor_state - attractor.basin_center)
                active_attractors.append({
                    'id': attractor_id,
                    'distance_to_center': distance_to_center,
                    'in_basin': attractor.is_in_basin(self.current_attractor_state),
                    'pulling_strength': attractor.pulling_strength,
                    'weight': self._compute_attractor_weight(attractor_id)
                })

        return {
            'current_state': self.current_attractor_state.tolist(),
            'active_attractors': active_attractors,
            'state_history_length': len(self.attractor_history),
            'dominant_attractor': self._find_dominant_attractor()
        }

    def _find_dominant_attractor(self) -> Optional[str]:
        """Find the currently dominant attractor"""
        min_distance = float('inf')
        dominant_attractor = None

        for attractor_id, attractor in self.pullback_attractors.items():
            if self._should_attractor_be_active(attractor_id, attractor):
                distance = np.linalg.norm(self.current_attractor_state - attractor.basin_center)
                weighted_distance = distance / self._compute_attractor_weight(attractor_id)

                if weighted_distance < min_distance:
                    min_distance = weighted_distance
                    dominant_attractor = attractor_id

        return dominant_attractor

    def _calculate_prediction_error(self, packet: NeuronalPacket) -> float:
        """Calculate prediction error with evolutionary priors for active inference"""
        if not self.prediction_model:
            # Initialize prediction model if not available
            self.prediction_model = {
                'expected_activation': 0.5,
                'pattern_expectations': {},
                'temporal_predictions': []
            }
            # For first packet, calculate initial prediction error against default expectation
            # This represents genuine uncertainty at system startup
            initial_prediction = 0.5  # Neutral expectation
            return abs(packet.activation_level - initial_prediction)

        # Get prior-weighted prediction
        prior_prediction = self.calculate_prior_weighted_prediction(packet.content)

        # Enhanced prediction error calculation with priors
        predicted = self.prediction_model.get('expected_activation', 0.5)
        # Combine model prediction with evolutionary priors
        combined_prediction = (predicted + prior_prediction) / 2.0
        actual = packet.activation_level

        # Calculate temporal prediction error if we have history
        temporal_error = 0.0
        if len(self.memory_buffer) > 1:
            recent_activations = [p.activation_level for p in self.memory_buffer[-3:]]
            predicted_next = np.mean(recent_activations) if recent_activations else 0.5
            temporal_error = abs(predicted_next - actual)

        # Combine immediate and temporal prediction errors
        total_error = (abs(combined_prediction - actual) + temporal_error) / 2.0

        # Update prediction model
        self.prediction_model['expected_activation'] = (
            self.prediction_model['expected_activation'] * 0.9 + actual * 0.1
        )

        return min(1.0, total_error)
    
    def _calculate_surprise(self, packet: NeuronalPacket) -> float:
        """Calculate surprise (unexpected information)"""
        if not self.world_model:
            return 0.0
        
        # Simple surprise calculation based on content novelty
        content_hash = hash(str(packet.content))
        known_patterns = self.world_model.get('known_patterns', set())
        
        if content_hash in known_patterns:
            return 0.0
        else:
            return 1.0
    
    def _generate_response_packets(self, packet: NeuronalPacket, prediction_error: float, surprise: float) -> List[NeuronalPacket]:
        """Generate response packets based on active inference principles"""
        responses = []
        
        # If high prediction error, generate corrective packets
        if prediction_error > self.surprise_threshold:
            corrective_packet = NeuronalPacket(
                content={
                    'type': 'corrective',
                    'original_packet_id': packet.id,
                    'prediction_error': prediction_error,
                    'correction_signal': -prediction_error * self.learning_rate
                },
                activation_level=prediction_error,
                source_thoughtseed=self.id,
                processing_priority=1
            )
            responses.append(corrective_packet)
        
        # If high surprise, generate exploratory packets
        if surprise > self.surprise_threshold:
            exploratory_packet = NeuronalPacket(
                content={
                    'type': 'exploratory',
                    'original_packet_id': packet.id,
                    'surprise': surprise,
                    'exploration_direction': 'investigate'
                },
                activation_level=surprise,
                source_thoughtseed=self.id,
                processing_priority=2
            )
            responses.append(exploratory_packet)
        
        return responses
    
    def update_world_model(self, packet: NeuronalPacket):
        """Update world model based on new information"""
        if not self.world_model:
            self.world_model = {'known_patterns': set(), 'patterns': []}
        
        # Add pattern to known patterns
        content_hash = hash(str(packet.content))
        self.world_model['known_patterns'].add(content_hash)
        
        # Update pattern database
        pattern = {
            'content': packet.content,
            'activation': packet.activation_level,
            'timestamp': packet.timestamp,
            'prediction_error': packet.prediction_error,
            'surprise': packet.surprise
        }
        self.world_model['patterns'].append(pattern)
        
        # Keep only recent patterns (last 100)
        if len(self.world_model['patterns']) > 100:
            self.world_model['patterns'] = self.world_model['patterns'][-100:]

# =============================================================================
# Active Inference Thoughtseed Bridge
# =============================================================================

class FourLayerArchitecture:
    """4-Layer Architecture: NPDs -> KDs -> Thoughtseed Network -> Meta-cognition"""

    def __init__(self):
        # Layer 1: Neuronal Packet Dispatchers
        self.npds: Dict[str, NeuronalPacketDispatcher] = {}

        # Layer 2: Knowledge Domains
        self.knowledge_domains: Dict[str, KnowledgeDomain] = {}

        # Layer 3: Thoughtseed Network (managed by bridge)
        self.thoughtseed_bridge: Optional['ActiveInferenceThoughtseedBridge'] = None

        # Layer 4: Meta-cognition layer
        self.meta_cognition_state: Dict[str, Any] = {
            'global_attention': 0.0,
            'system_monitoring': {},
            'learning_progress': {},
            'consciousness_level': 0.0
        }

    def setup_architecture(self, bridge: 'ActiveInferenceThoughtseedBridge'):
        """Initialize the 4-layer architecture"""
        self.thoughtseed_bridge = bridge

        # Create NPDs
        main_npd = NeuronalPacketDispatcher()
        self.npds['main'] = main_npd

        # Create Knowledge Domains
        self.knowledge_domains['architecture'] = KnowledgeDomain(domain_type='architecture')
        self.knowledge_domains['performance'] = KnowledgeDomain(domain_type='performance')
        self.knowledge_domains['meta_learning'] = KnowledgeDomain(domain_type='meta_learning')

    def process_through_layers(self, packet: NeuronalPacket) -> Dict[str, Any]:
        """Process packet through all 4 layers"""
        results = {
            'layer_1_routing': {},
            'layer_2_domain_processing': {},
            'layer_3_thoughtseed_responses': {},
            'layer_4_meta_cognition': {}
        }

        # Layer 1: NPD Routing
        main_npd = self.npds['main']
        target_domains = main_npd.route_packet(packet, self.knowledge_domains)
        results['layer_1_routing'] = {
            'npd_id': main_npd.id,
            'target_domains': target_domains
        }

        # Layer 2: Knowledge Domain Processing
        all_domain_responses = []
        for domain_id in target_domains:
            if domain_id in self.knowledge_domains:
                domain = self.knowledge_domains[domain_id]
                domain_responses = domain.process_domain_packet(packet, self.thoughtseed_bridge.thoughtseeds)
                all_domain_responses.extend(domain_responses)
                results['layer_2_domain_processing'][domain_id] = len(domain_responses)

        # Layer 3: Thoughtseed Network (already processed in domains)
        results['layer_3_thoughtseed_responses'] = {
            'total_responses': len(all_domain_responses),
            'avg_activation': sum(r.activation_level for r in all_domain_responses) / max(len(all_domain_responses), 1)
        }

        # Layer 4: Meta-cognition
        self._update_meta_cognition(packet, all_domain_responses)
        results['layer_4_meta_cognition'] = self.meta_cognition_state.copy()

        return results

    def _update_meta_cognition(self, packet: NeuronalPacket, responses: List[NeuronalPacket]):
        """Update meta-cognitive state based on processing results"""
        # Update global attention based on packet importance
        packet_importance = packet.activation_level * packet.processing_priority
        self.meta_cognition_state['global_attention'] = (
            self.meta_cognition_state['global_attention'] * 0.9 + packet_importance * 0.1
        )

        # Monitor system performance
        response_quality = sum(r.activation_level for r in responses) / max(len(responses), 1)
        self.meta_cognition_state['system_monitoring']['last_response_quality'] = response_quality

        # Track consciousness level
        if self.thoughtseed_bridge:
            active_thoughtseeds = [ts for ts in self.thoughtseed_bridge.thoughtseeds.values()
                                 if ts.activation_state > 0.5]
            consciousness_level = len(active_thoughtseeds) / max(len(self.thoughtseed_bridge.thoughtseeds), 1)
            self.meta_cognition_state['consciousness_level'] = consciousness_level

class ActiveInferenceThoughtseedBridge:
    """Bridges active inference with thoughtseed network for ASI-Arch"""

    def __init__(self):
        self.thoughtseeds: Dict[str, Thoughtseed] = {}
        self.packet_queue: List[NeuronalPacket] = []
        self.consciousness_threshold = 0.7
        self.evolutionary_priors_library: Dict[str, EvolutionaryPrior] = {}
        self.nested_blanket_hierarchy: Dict[int, List[str]] = {}  # level -> blanket_ids
        self.inner_screen: Optional[Dict[str, Any]] = None

        # 4-Layer Architecture
        self.four_layer_arch = FourLayerArchitecture()
        self.four_layer_arch.setup_architecture(self)

    def create_evolutionary_priors_system(self):
        """Create the four types of evolutionary priors for ASI-Arch"""

        # 1. Basal Priors - Basic survival and homeostatic patterns
        basal_prior = EvolutionaryPrior(
            type=EvolutionaryPriorType.BASAL,
            strength=1.0,
            patterns={
                'survival_relevant': True,
                'energy_efficiency': 0.8,
                'stability_preference': 0.9,
                'basic_pattern_recognition': True
            },
            activation_threshold=0.1,  # Always active
            nesting_level=0  # Deepest level
        )

        # 2. Lineage-Specific Priors - Neural network architecture preferences
        lineage_prior = EvolutionaryPrior(
            type=EvolutionaryPriorType.LINEAGE_SPECIFIC,
            strength=0.8,
            patterns={
                'attention_mechanisms': 0.9,
                'hierarchical_processing': 0.8,
                'transformer_like': 0.7,
                'gradient_flow': 0.9
            },
            activation_threshold=0.3,
            nesting_level=1,
            parent_prior_id=basal_prior.id
        )
        basal_prior.child_prior_ids.add(lineage_prior.id)

        # 3. Dispositional Priors - Individual architectural preferences
        dispositional_prior = EvolutionaryPrior(
            type=EvolutionaryPriorType.DISPOSITIONAL,
            strength=0.6,
            patterns={
                'exploration_tendency': 0.7,
                'novelty_seeking': 0.6,
                'computational_efficiency': 0.8,
                'interpretability': 0.5
            },
            activation_threshold=0.4,
            nesting_level=2,
            parent_prior_id=lineage_prior.id
        )
        lineage_prior.child_prior_ids.add(dispositional_prior.id)

        # 4. Learned Priors - Experience-dependent patterns
        learned_prior = EvolutionaryPrior(
            type=EvolutionaryPriorType.LEARNED,
            strength=0.4,  # Most flexible
            patterns={
                'successful_architectures': [],
                'failure_patterns': [],
                'context_specific_adaptations': {},
                'performance_correlations': {}
            },
            activation_threshold=0.5,
            nesting_level=3,  # Highest level
            parent_prior_id=dispositional_prior.id
        )
        dispositional_prior.child_prior_ids.add(learned_prior.id)

        # Store in library
        for prior in [basal_prior, lineage_prior, dispositional_prior, learned_prior]:
            self.evolutionary_priors_library[prior.id] = prior

        return {
            'basal': basal_prior,
            'lineage_specific': lineage_prior,
            'dispositional': dispositional_prior,
            'learned': learned_prior
        }

    def create_nested_markov_blankets(self, thoughtseed: Thoughtseed, nesting_depth: int = 4):
        """Create nested Markov blankets for hierarchical organization"""
        blankets = []

        for level in range(nesting_depth):
            blanket = NestedMarkovBlanket(
                nesting_level=level,
                internal_states={f"internal_{level}_{i}" for i in range(5)},
                external_states={f"external_{level}_{i}" for i in range(3)},
                sensory_states={f"sensory_{level}_{i}" for i in range(4)},
                active_states={f"active_{level}_{i}" for i in range(2)}
            )

            # Set up hierarchy
            if level > 0 and blankets:
                parent_blanket = blankets[level - 1]
                blanket.parent_blanket_id = parent_blanket.id
                parent_blanket.child_blanket_ids.add(blanket.id)

            blankets.append(blanket)

            # Store in hierarchy mapping
            if level not in self.nested_blanket_hierarchy:
                self.nested_blanket_hierarchy[level] = []
            self.nested_blanket_hierarchy[level].append(blanket.id)

        # Assign the top-level blanket to the thoughtseed
        if blankets:
            thoughtseed.markov_blanket = blankets[-1]  # Outermost blanket

        return blankets
        
    def create_thoughtseed(self, thoughtseed_type: ThoughtseedType,
                          initial_state: Optional[Dict[str, Any]] = None) -> Thoughtseed:
        """Create a new thoughtseed with evolutionary priors and nested Markov blankets"""
        thoughtseed = Thoughtseed(
            type=thoughtseed_type,
            embodiment_state=initial_state or {}
        )

        # Initialize evolutionary priors system if not already done
        if not self.evolutionary_priors_library:
            self.create_evolutionary_priors_system()

        # Add all evolutionary priors to the thoughtseed
        for prior in self.evolutionary_priors_library.values():
            thoughtseed.add_evolutionary_prior(prior)

        # Create nested Markov blankets
        self.create_nested_markov_blankets(thoughtseed)

        self.thoughtseeds[thoughtseed.id] = thoughtseed
        logger.info(f"Created thoughtseed {thoughtseed.id} of type {thoughtseed_type.value} with evolutionary priors and nested blankets")
        return thoughtseed

    def update_global_inner_screen(self):
        """Update the global inner screen with dominant thoughtseed content"""
        if not self.thoughtseeds:
            return

        # Find most activated thoughtseed
        dominant_thoughtseed = max(self.thoughtseeds.values(), key=lambda ts: ts.activation_state)

        if dominant_thoughtseed.activation_state > self.consciousness_threshold:
            # Project content to global inner screen
            self.inner_screen = {
                'dominant_thoughtseed_id': dominant_thoughtseed.id,
                'content': dominant_thoughtseed.inner_screen_content,
                'consciousness_level': dominant_thoughtseed.activation_state,
                'timestamp': datetime.now(),
                'type': dominant_thoughtseed.type.value
            }

            # Update the thoughtseed's inner screen
            if dominant_thoughtseed.inner_screen_content:
                dominant_thoughtseed.update_inner_screen(dominant_thoughtseed.inner_screen_content['content'])

    def compute_hierarchical_free_energy(self, thoughtseed: Thoughtseed, observations: Dict[str, Any]) -> Dict[str, float]:
        """Compute free energy across nested Markov blanket hierarchy"""
        free_energies = {}

        # Get all blankets in hierarchy (simplified - just the one blanket for now)
        blanket = thoughtseed.markov_blanket

        # Create internal beliefs from thoughtseed state
        internal_beliefs = {
            'activation_state': thoughtseed.activation_state,
            'prediction_model': thoughtseed.prediction_model or {},
            'world_model': thoughtseed.world_model or {}
        }

        # Convert to numerical values for free energy calculation
        numerical_beliefs = {}
        numerical_observations = {}

        for state_id in blanket.sensory_states:
            if state_id in internal_beliefs:
                numerical_beliefs[state_id] = float(hash(str(internal_beliefs)) % 100) / 100.0
            if state_id in observations:
                numerical_observations[state_id] = float(hash(str(observations)) % 100) / 100.0

        free_energy = blanket.compute_free_energy(numerical_beliefs, numerical_observations)
        free_energies[f"level_{blanket.nesting_level}"] = free_energy

        return free_energies
    
    def organize_thoughtseeds_into_domains(self):
        """Organize thoughtseeds into knowledge domains based on their types"""
        for ts_id, thoughtseed in self.thoughtseeds.items():
            # Assign to knowledge domains based on thoughtseed type
            if thoughtseed.type == ThoughtseedType.CONCEPTUAL:
                self.four_layer_arch.knowledge_domains['architecture'].add_thoughtseed(ts_id)
            elif thoughtseed.type == ThoughtseedType.PERCEPTUAL:
                self.four_layer_arch.knowledge_domains['performance'].add_thoughtseed(ts_id)
            elif thoughtseed.type == ThoughtseedType.METACOGNITIVE:
                self.four_layer_arch.knowledge_domains['meta_learning'].add_thoughtseed(ts_id)
            else:
                # Default to architecture domain
                self.four_layer_arch.knowledge_domains['architecture'].add_thoughtseed(ts_id)

    def process_asi_arch_context(self, context: str, architecture_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process ASI-Arch context through 4-layer architecture"""

        # Create context packet
        context_packet = NeuronalPacket(
            content={
                'type': 'asi_arch_context',
                'context': context,
                'architecture_data': architecture_data,
                'timestamp': datetime.now().isoformat()
            },
            activation_level=1.0,
            processing_priority=1
        )

        # Ensure thoughtseeds are organized into domains
        self.organize_thoughtseeds_into_domains()

        # Process through 4-layer architecture
        layer_results = self.four_layer_arch.process_through_layers(context_packet)

        # Update global inner screen
        self.update_global_inner_screen()

        # Calculate hierarchical free energy for each thoughtseed
        hierarchical_energies = {}
        for ts_id, thoughtseed in self.thoughtseeds.items():
            energies = self.compute_hierarchical_free_energy(thoughtseed, architecture_data)
            hierarchical_energies[ts_id] = energies

        # Analyze consciousness emergence from all layers
        consciousness_level = self._analyze_consciousness_emergence_4layer(layer_results)

        # Extract responses from layer processing for compatibility
        all_responses = []
        for domain_id, response_count in layer_results.get('layer_2_domain_processing', {}).items():
            # Create mock response objects from domain processing
            if response_count > 0:
                for i in range(min(response_count, 3)):  # Limit to 3 responses per domain
                    all_responses.append({
                        'content': {
                            'type': f'{domain_id}_analysis',
                            'domain': domain_id,
                            'activation_level': layer_results['layer_3_thoughtseed_responses']['avg_activation'],
                            'source': 'thoughtseed_network'
                        },
                        'activation_level': layer_results['layer_3_thoughtseed_responses']['avg_activation'],
                        'surprise': 0.5,  # Default surprise level
                        'prediction_error': 0.3  # Default prediction error
                    })

        return {
            'context_packet': context_packet.to_dict(),
            'layer_processing': layer_results,
            'hierarchical_free_energies': hierarchical_energies,
            'consciousness_level': consciousness_level,
            'thoughtseeds_activated': len([t for t in self.thoughtseeds.values() if t.activation_state > 0.5]),
            'inner_screen': self.inner_screen,
            'evolutionary_priors_active': len([p for p in self.evolutionary_priors_library.values() if p.strength > 0.5]),
            'responses': all_responses  # Add the expected responses key
        }

    def _analyze_consciousness_emergence_4layer(self, layer_results: Dict[str, Any]) -> float:
        """Analyze consciousness emergence across all 4 layers"""
        # Layer 1 (NPD routing efficiency)
        routing_efficiency = len(layer_results['layer_1_routing']['target_domains']) / max(len(self.four_layer_arch.knowledge_domains), 1)

        # Layer 2 (Knowledge domain activation)
        total_domain_responses = sum(layer_results['layer_2_domain_processing'].values())
        domain_activation = min(1.0, total_domain_responses / 10.0)  # Normalize

        # Layer 3 (Thoughtseed network response)
        thoughtseed_activation = layer_results['layer_3_thoughtseed_responses']['avg_activation']

        # Layer 4 (Meta-cognition level)
        meta_cognition_level = layer_results['layer_4_meta_cognition']['consciousness_level']

        # Combine all layers with different weights
        consciousness = (
            routing_efficiency * 0.1 +
            domain_activation * 0.2 +
            thoughtseed_activation * 0.4 +
            meta_cognition_level * 0.3
        )

        return min(1.0, consciousness)
    
    def _process_through_network(self, packet: NeuronalPacket) -> List[NeuronalPacket]:
        """Process packet through the thoughtseed network"""
        all_responses = []
        processed_thoughtseeds = set()
        
        # Start with conceptual thoughtseeds (most relevant for ASI-Arch)
        conceptual_thoughtseeds = [t for t in self.thoughtseeds.values() 
                                 if t.type == ThoughtseedType.CONCEPTUAL]
        
        for thoughtseed in conceptual_thoughtseeds:
            if thoughtseed.id not in processed_thoughtseeds:
                responses = thoughtseed.process_packet(packet)
                all_responses.extend(responses)
                processed_thoughtseeds.add(thoughtseed.id)
                
                # Update world model
                thoughtseed.update_world_model(packet)
        
        return all_responses
    
    def _analyze_consciousness_emergence(self, responses: List[NeuronalPacket]) -> float:
        """Analyze consciousness emergence from response patterns"""
        if not responses:
            return 0.0
        
        # Calculate consciousness based on response complexity and coordination
        total_activation = sum(r.activation_level for r in responses)
        avg_surprise = np.mean([r.surprise for r in responses])
        response_diversity = len(set(r.content.get('type', 'unknown') for r in responses))
        
        # Consciousness emerges from coordinated, surprising, diverse responses
        consciousness = min(1.0, (total_activation * avg_surprise * response_diversity) / 10.0)
        
        return consciousness

# =============================================================================
# ASI-Arch Integration
# =============================================================================

class ASIArchThoughtseedIntegration:
    """Integration layer for ASI-Arch and Thoughtseed framework"""
    
    def __init__(self):
        self.bridge = ActiveInferenceThoughtseedBridge()
        self._setup_asi_arch_thoughtseeds()
    
    def _setup_asi_arch_thoughtseeds(self):
        """Set up thoughtseeds specifically for ASI-Arch architecture discovery"""
        
        # Architecture Evolution Thoughtseed
        self.bridge.create_thoughtseed(
            ThoughtseedType.CONCEPTUAL,
            {
                'domain': 'architecture_evolution',
                'focus': 'neural_architecture_discovery',
                'capabilities': ['pattern_recognition', 'architecture_analysis']
            }
        )
        
        # Performance Analysis Thoughtseed
        self.bridge.create_thoughtseed(
            ThoughtseedType.PERCEPTUAL,
            {
                'domain': 'performance_analysis',
                'focus': 'benchmark_evaluation',
                'capabilities': ['performance_prediction', 'efficiency_analysis']
            }
        )
        
        # Meta-Learning Thoughtseed
        self.bridge.create_thoughtseed(
            ThoughtseedType.METACOGNITIVE,
            {
                'domain': 'meta_learning',
                'focus': 'learning_to_learn',
                'capabilities': ['strategy_optimization', 'pattern_generalization']
            }
        )
    
    async def enhance_evolution_context(self, context: str, architecture_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance ASI-Arch evolution context with thoughtseed processing"""
        
        # Process through thoughtseed network
        thoughtseed_result = self.bridge.process_asi_arch_context(context, architecture_data)
        
        # Generate enhanced context
        enhanced_context = self._generate_enhanced_context(context, thoughtseed_result)
        
        return {
            'original_context': context,
            'enhanced_context': enhanced_context,
            'thoughtseed_insights': thoughtseed_result,
            'consciousness_level': thoughtseed_result['consciousness_level']
        }
    
    def _generate_enhanced_context(self, original_context: str, thoughtseed_result: Dict[str, Any]) -> str:
        """Generate enhanced context based on thoughtseed processing"""
        
        consciousness_level = thoughtseed_result['consciousness_level']
        responses = thoughtseed_result['responses']
        
        # Base enhanced context
        enhanced = f"ORIGINAL CONTEXT: {original_context}\n\n"
        
        # Add consciousness awareness
        if consciousness_level > 0.5:
            enhanced += f"CONSCIOUSNESS DETECTED (Level: {consciousness_level:.2f}): "
            enhanced += "The architecture shows signs of emergent consciousness patterns. "
            enhanced += "Consider this in your evolution strategy.\n\n"
        
        # Add thoughtseed insights
        if responses:
            enhanced += "THOUGHTSEED INSIGHTS:\n"
            for i, response in enumerate(responses[:3]):  # Top 3 responses
                response_type = response['content'].get('type', 'unknown')
                enhanced += f"- {response_type.upper()}: {response['content']}\n"
            enhanced += "\n"
        
        # Add active inference guidance
        enhanced += "ACTIVE INFERENCE GUIDANCE:\n"
        enhanced += "- Focus on prediction error minimization\n"
        enhanced += "- Explore surprising architectural patterns\n"
        enhanced += "- Balance exploitation and exploration\n"
        
        return enhanced

# =============================================================================
# Test and Demo Functions
# =============================================================================

async def test_thoughtseed_asi_arch_integration():
    """Test the Thoughtseed integration with ASI-Arch"""
    
    print("🌱🧠 Testing Thoughtseed Active Inference for ASI-Arch")
    print("=" * 60)
    
    # Initialize integration
    integration = ASIArchThoughtseedIntegration()
    
    # Test with mock ASI-Arch data
    mock_context = "Design a new linear attention mechanism for transformer architectures"
    mock_architecture_data = {
        'name': 'LinearAttention_v1',
        'performance': 0.85,
        'complexity': 'O(n)',
        'innovations': ['gating', 'retention']
    }
    
    print(f"\n📊 Processing Context: {mock_context}")
    print(f"📊 Architecture Data: {mock_architecture_data}")
    
    # Process through thoughtseed integration
    result = await integration.enhance_evolution_context(mock_context, mock_architecture_data)
    
    print(f"\n✅ Enhanced Context Generated:")
    print(f"Consciousness Level: {result['consciousness_level']:.2f}")
    print(f"\nEnhanced Context:\n{result['enhanced_context']}")
    
    return result

if __name__ == "__main__":
    asyncio.run(test_thoughtseed_asi_arch_integration())
