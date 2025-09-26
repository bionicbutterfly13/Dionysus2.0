"""
Affordance-Context Integration Service
Based on Scholz et al. "Inference of Affordances and Active Motor Control in Simulated Agents"
Inspired by Julian Kiverstein's ecological-enactive approach
Constitutional compliance: mock data transparency, evaluative feedback
"""

import asyncio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass
from enum import Enum

from ..models.thoughtseed_trace import (
    ThoughtSeedTrace, ConsciousnessState, HierarchicalBelief, 
    NeuronalPacket, PredictionError, InferenceType
)

logger = logging.getLogger(__name__)

class AffordanceType(Enum):
    """Types of affordances based on ecological-enactive cognition"""
    READY_TO_HAND = "ready_to_hand"        # Immediate, skillful action
    PRESENT_AT_HAND = "present_at_hand"    # Deliberate, reflective action
    SOCIAL_COORDINATION = "social_coordination"  # Collaborative affordances
    CULTURAL_SYMBOLIC = "cultural_symbolic"  # Language-based affordances
    NESTED_TEMPORAL = "nested_temporal"    # Multi-timescale affordances

class ContextualAffordance:
    """Represents an affordance in its contextual setting"""
    
    def __init__(self, 
                 action_type: str,
                 context_requirements: List[str],
                 temporal_horizon: float,
                 social_coordination: bool = False,
                 cultural_scaffolding: Optional[str] = None):
        self.action_type = action_type
        self.context_requirements = context_requirements
        self.temporal_horizon = temporal_horizon
        self.social_coordination = social_coordination
        self.cultural_scaffolding = cultural_scaffolding
        self.salience = 0.0
        self.availability = 0.0
        self.timestamp = datetime.now()

class AffordanceContextEngine:
    """Engine for managing affordance-context relationships"""
    
    def __init__(self, hidden_size: int = 256):
        self.hidden_size = hidden_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(512, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        ).to(self.device)
        
        # Affordance detector
        self.affordance_detector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 128)  # Affordance features
        ).to(self.device)
        
        # Context-affordance coupling network
        self.coupling_network = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=hidden_size * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=3
        ).to(self.device)
        
        # Salience predictor
        self.salience_predictor = nn.Sequential(
            nn.Linear(hidden_size + 128, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        ).to(self.device)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for stable training"""
        for module in [self.context_encoder, self.affordance_detector, 
                      self.coupling_network, self.salience_predictor]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
    
    async def detect_affordances(self, 
                               context_data: Dict[str, Any],
                               consciousness_state: Dict[str, Any]) -> List[ContextualAffordance]:
        """
        Detect available affordances given current context and consciousness state
        """
        try:
            # Encode context
            context_features = self._encode_context(context_data)
            context_tensor = torch.tensor(context_features, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Encode consciousness state
            consciousness_features = self._encode_consciousness_state(consciousness_state)
            consciousness_tensor = torch.tensor(consciousness_features, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Combine context and consciousness
            combined_input = torch.cat([context_tensor, consciousness_tensor], dim=-1)
            
            # Process through context encoder
            context_embedding = self.context_encoder(combined_input)
            
            # Detect affordances
            affordance_features = self.affordance_detector(context_embedding)
            
            # Generate affordance candidates
            affordances = await self._generate_affordance_candidates(
                context_embedding, affordance_features, context_data
            )
            
            # Calculate salience for each affordance
            for affordance in affordances:
                salience = await self._calculate_affordance_salience(
                    context_embedding, affordance, consciousness_state
                )
                affordance.salience = salience
            
            # Sort by salience
            affordances.sort(key=lambda x: x.salience, reverse=True)
            
            return affordances
            
        except Exception as e:
            logger.error(f"Error detecting affordances: {e}")
            return []
    
    def _encode_context(self, context_data: Dict[str, Any]) -> np.ndarray:
        """Encode context data into neural network input"""
        features = []
        
        # Environmental context
        env = context_data.get('environment', {})
        features.extend([
            env.get('spatial_layout', 0.0),
            env.get('temporal_phase', 0.0),
            env.get('social_presence', 0.0),
            env.get('cultural_artifacts', 0.0)
        ])
        
        # Task context
        task = context_data.get('task', {})
        features.extend([
            task.get('goal_clarity', 0.0),
            task.get('complexity', 0.0),
            task.get('urgency', 0.0),
            task.get('social_nature', 0.0)
        ])
        
        # Historical context
        history = context_data.get('history', {})
        features.extend([
            history.get('recent_actions', 0.0),
            history.get('success_rate', 0.0),
            history.get('learning_progress', 0.0)
        ])
        
        # Pad or truncate to fixed size
        target_size = 256
        if len(features) < target_size:
            features.extend([0.0] * (target_size - len(features)))
        else:
            features = features[:target_size]
            
        return np.array(features, dtype=np.float32)
    
    def _encode_consciousness_state(self, consciousness_state: Dict[str, Any]) -> np.ndarray:
        """Encode consciousness state into neural network input"""
        features = []
        
        # Consciousness level
        features.append(consciousness_state.get('consciousness_level', 0.0))
        
        # Attention patterns
        attention = consciousness_state.get('attention', {})
        features.extend([
            attention.get('focus_intensity', 0.0),
            attention.get('distraction_level', 0.0),
            attention.get('sustained_attention', 0.0)
        ])
        
        # Memory states
        memory = consciousness_state.get('memory', {})
        features.extend([
            memory.get('working_memory_load', 0.0),
            memory.get('episodic_accessibility', 0.0),
            memory.get('semantic_activation', 0.0)
        ])
        
        # Emotional states
        emotion = consciousness_state.get('emotion', {})
        features.extend([
            emotion.get('valence', 0.0),
            emotion.get('arousal', 0.0),
            emotion.get('dominance', 0.0)
        ])
        
        # Pad or truncate to fixed size
        target_size = 256
        if len(features) < target_size:
            features.extend([0.0] * (target_size - len(features)))
        else:
            features = features[:target_size]
            
        return np.array(features, dtype=np.float32)
    
    async def _generate_affordance_candidates(self, 
                                            context_embedding: torch.Tensor,
                                            affordance_features: torch.Tensor,
                                            context_data: Dict[str, Any]) -> List[ContextualAffordance]:
        """Generate candidate affordances based on context"""
        affordances = []
        
        # Extract context information
        task_type = context_data.get('task', {}).get('type', 'general')
        environment = context_data.get('environment', {})
        social_context = context_data.get('social', {})
        
        # Generate affordances based on context type
        if task_type == 'problem_solving':
            affordances.extend([
                ContextualAffordance(
                    action_type="analyze_problem",
                    context_requirements=["clear_goal", "sufficient_information"],
                    temporal_horizon=1.0,
                    social_coordination=False
                ),
                ContextualAffordance(
                    action_type="seek_help",
                    context_requirements=["social_presence", "expertise_available"],
                    temporal_horizon=0.5,
                    social_coordination=True
                ),
                ContextualAffordance(
                    action_type="break_down_problem",
                    context_requirements=["complex_problem", "analytical_skills"],
                    temporal_horizon=2.0,
                    social_coordination=False
                )
            ])
        
        elif task_type == 'social_interaction':
            affordances.extend([
                ContextualAffordance(
                    action_type="initiate_conversation",
                    context_requirements=["social_presence", "appropriate_timing"],
                    temporal_horizon=0.1,
                    social_coordination=True
                ),
                ContextualAffordance(
                    action_type="active_listening",
                    context_requirements=["ongoing_conversation", "attention_capacity"],
                    temporal_horizon=0.5,
                    social_coordination=True
                ),
                ContextualAffordance(
                    action_type="cultural_expression",
                    context_requirements=["cultural_knowledge", "appropriate_context"],
                    temporal_horizon=1.0,
                    social_coordination=True,
                    cultural_scaffolding="language_use"
                )
            ])
        
        elif task_type == 'creative_work':
            affordances.extend([
                ContextualAffordance(
                    action_type="explore_possibilities",
                    context_requirements=["open_mindset", "time_available"],
                    temporal_horizon=3.0,
                    social_coordination=False
                ),
                ContextualAffordance(
                    action_type="collaborate_creatively",
                    context_requirements=["creative_partners", "shared_vision"],
                    temporal_horizon=2.0,
                    social_coordination=True
                ),
                ContextualAffordance(
                    action_type="cultural_innovation",
                    context_requirements=["cultural_knowledge", "creative_skills"],
                    temporal_horizon=5.0,
                    social_coordination=True,
                    cultural_scaffolding="artistic_practices"
                )
            ])
        
        # Add general affordances
        affordances.extend([
            ContextualAffordance(
                action_type="observe_environment",
                context_requirements=["sensory_capacity", "attention"],
                temporal_horizon=0.1,
                social_coordination=False
            ),
            ContextualAffordance(
                action_type="reflect_on_experience",
                context_requirements=["memory_access", "reflective_capacity"],
                temporal_horizon=1.0,
                social_coordination=False
            )
        ])
        
        return affordances
    
    async def _calculate_affordance_salience(self, 
                                           context_embedding: torch.Tensor,
                                           affordance: ContextualAffordance,
                                           consciousness_state: Dict[str, Any]) -> float:
        """Calculate salience of an affordance given context and consciousness"""
        try:
            # Encode affordance features
            affordance_features = self._encode_affordance(affordance)
            affordance_tensor = torch.tensor(affordance_features, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Combine context and affordance
            combined = torch.cat([context_embedding, affordance_tensor], dim=-1)
            
            # Predict salience
            with torch.no_grad():
                salience = self.salience_predictor(combined).item()
            
            # Adjust based on consciousness state
            consciousness_level = consciousness_state.get('consciousness_level', 0.5)
            attention_focus = consciousness_state.get('attention', {}).get('focus_intensity', 0.5)
            
            # Higher consciousness and attention increase salience
            salience *= (1.0 + consciousness_level * 0.5)
            salience *= (1.0 + attention_focus * 0.3)
            
            return min(1.0, salience)
            
        except Exception as e:
            logger.error(f"Error calculating affordance salience: {e}")
            return 0.0
    
    def _encode_affordance(self, affordance: ContextualAffordance) -> np.ndarray:
        """Encode affordance into neural network input"""
        features = []
        
        # Action type (one-hot encoded)
        action_types = ["analyze_problem", "seek_help", "initiate_conversation", 
                       "active_listening", "explore_possibilities", "observe_environment"]
        action_encoding = [1.0 if affordance.action_type == at else 0.0 for at in action_types]
        features.extend(action_encoding)
        
        # Context requirements
        features.append(len(affordance.context_requirements) / 10.0)  # Normalized count
        
        # Temporal horizon
        features.append(affordance.temporal_horizon / 10.0)  # Normalized
        
        # Social coordination
        features.append(1.0 if affordance.social_coordination else 0.0)
        
        # Cultural scaffolding
        features.append(1.0 if affordance.cultural_scaffolding else 0.0)
        
        # Pad to fixed size
        target_size = 128
        if len(features) < target_size:
            features.extend([0.0] * (target_size - len(features)))
        else:
            features = features[:target_size]
            
        return np.array(features, dtype=np.float32)

class NestedAffordanceCoordinator:
    """Coordinates nested affordances across multiple timescales"""
    
    def __init__(self, hidden_size: int = 256):
        self.hidden_size = hidden_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Multi-scale affordance processor
        self.timescale_processors = nn.ModuleDict({
            'immediate': nn.LSTM(hidden_size, hidden_size, batch_first=True),
            'short_term': nn.LSTM(hidden_size, hidden_size, batch_first=True),
            'medium_term': nn.LSTM(hidden_size, hidden_size, batch_first=True),
            'long_term': nn.LSTM(hidden_size, hidden_size, batch_first=True)
        }).to(self.device)
        
        # Affordance coordination network
        self.coordination_network = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=hidden_size * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2
        ).to(self.device)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_size, 1).to(self.device)
        
    async def coordinate_nested_affordances(self, 
                                          affordances: List[ContextualAffordance],
                                          context_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Coordinate nested affordances across multiple timescales
        """
        try:
            # Group affordances by temporal horizon
            timescale_groups = {
                'immediate': [a for a in affordances if a.temporal_horizon <= 0.5],
                'short_term': [a for a in affordances if 0.5 < a.temporal_horizon <= 2.0],
                'medium_term': [a for a in affordances if 2.0 < a.temporal_horizon <= 5.0],
                'long_term': [a for a in affordances if a.temporal_horizon > 5.0]
            }
            
            # Process each timescale
            timescale_embeddings = {}
            for timescale, group_affordances in timescale_groups.items():
                if group_affordances:
                    embedding = await self._process_timescale_affordances(
                        group_affordances, timescale
                    )
                    timescale_embeddings[timescale] = embedding
            
            # Coordinate across timescales
            coordination_result = await self._coordinate_timescales(
                timescale_embeddings, context_data
            )
            
            return coordination_result
            
        except Exception as e:
            logger.error(f"Error coordinating nested affordances: {e}")
            return {'error': str(e)}
    
    async def _process_timescale_affordances(self, 
                                           affordances: List[ContextualAffordance],
                                           timescale: str) -> torch.Tensor:
        """Process affordances for a specific timescale"""
        if not affordances:
            return torch.zeros(1, self.hidden_size).to(self.device)
        
        # Encode affordances
        affordance_encodings = []
        for affordance in affordances:
            encoding = self._encode_affordance_for_coordination(affordance)
            affordance_encodings.append(encoding)
        
        # Stack and process
        affordance_tensor = torch.stack(affordance_encodings).unsqueeze(0).to(self.device)
        
        # Process through timescale-specific LSTM
        with torch.no_grad():
            output, (hidden, cell) = self.timescale_processors[timescale](affordance_tensor)
            # Use final hidden state as timescale embedding
            timescale_embedding = hidden[-1]
        
        return timescale_embedding
    
    def _encode_affordance_for_coordination(self, affordance: ContextualAffordance) -> torch.Tensor:
        """Encode affordance for coordination processing"""
        features = []
        
        # Basic affordance features
        features.append(affordance.salience)
        features.append(affordance.temporal_horizon / 10.0)
        features.append(1.0 if affordance.social_coordination else 0.0)
        features.append(1.0 if affordance.cultural_scaffolding else 0.0)
        
        # Pad to hidden size
        while len(features) < self.hidden_size:
            features.append(0.0)
        
        return torch.tensor(features[:self.hidden_size], dtype=torch.float32)
    
    async def _coordinate_timescales(self, 
                                   timescale_embeddings: Dict[str, torch.Tensor],
                                   context_data: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate affordances across different timescales"""
        if not timescale_embeddings:
            return {'coordination': 'no_affordances', 'recommendations': []}
        
        # Stack timescale embeddings
        timescale_keys = list(timescale_embeddings.keys())
        stacked_embeddings = torch.stack([timescale_embeddings[k] for k in timescale_keys])
        
        # Use transformer decoder for coordination
        with torch.no_grad():
            # Self-attention across timescales
            coordinated = self.coordination_network(
                stacked_embeddings.unsqueeze(0),
                stacked_embeddings.unsqueeze(0)
            )
            
            # Generate coordination output
            coordination_scores = self.output_projection(coordinated).squeeze()
        
        # Generate recommendations
        recommendations = []
        for i, timescale in enumerate(timescale_keys):
            score = coordination_scores[i].item()
            if score > 0.5:  # Threshold for recommendation
                recommendations.append({
                    'timescale': timescale,
                    'priority': score,
                    'action': f'Focus on {timescale} affordances'
                })
        
        return {
            'coordination': 'successful',
            'timescales_active': timescale_keys,
            'recommendations': recommendations,
            'coordination_scores': coordination_scores.tolist()
        }

class AffordanceContextService:
    """Main service for affordance-context integration"""
    
    def __init__(self):
        self.affordance_engine = AffordanceContextEngine()
        self.nested_coordinator = NestedAffordanceCoordinator()
        self.active_affordances: Dict[str, List[ContextualAffordance]] = {}
        
    async def process_context_affordances(self, 
                                        trace_id: str,
                                        context_data: Dict[str, Any],
                                        consciousness_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process context and detect relevant affordances
        """
        try:
            # Detect affordances
            affordances = await self.affordance_engine.detect_affordances(
                context_data, consciousness_state
            )
            
            # Store active affordances
            self.active_affordances[trace_id] = affordances
            
            # Coordinate nested affordances
            coordination_result = await self.nested_coordinator.coordinate_nested_affordances(
                affordances, context_data
            )
            
            return {
                'trace_id': trace_id,
                'affordances_detected': len(affordances),
                'top_affordances': [
                    {
                        'action_type': a.action_type,
                        'salience': a.salience,
                        'temporal_horizon': a.temporal_horizon,
                        'social_coordination': a.social_coordination
                    } for a in affordances[:5]  # Top 5
                ],
                'coordination_result': coordination_result,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing context affordances: {e}")
            return {
                'trace_id': trace_id,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def get_affordance_analysis(self, trace_id: str) -> Dict[str, Any]:
        """Get affordance analysis for a specific trace"""
        if trace_id not in self.active_affordances:
            return {'error': 'No affordances found for trace'}
        
        affordances = self.active_affordances[trace_id]
        
        # Analyze affordance patterns
        analysis = {
            'total_affordances': len(affordances),
            'social_affordances': len([a for a in affordances if a.social_coordination]),
            'cultural_affordances': len([a for a in affordances if a.cultural_scaffolding]),
            'temporal_distribution': {
                'immediate': len([a for a in affordances if a.temporal_horizon <= 0.5]),
                'short_term': len([a for a in affordances if 0.5 < a.temporal_horizon <= 2.0]),
                'medium_term': len([a for a in affordances if 2.0 < a.temporal_horizon <= 5.0]),
                'long_term': len([a for a in affordances if a.temporal_horizon > 5.0])
            },
            'avg_salience': np.mean([a.salience for a in affordances]) if affordances else 0.0
        }
        
        return analysis

# Global service instance
affordance_context_service = AffordanceContextService()
