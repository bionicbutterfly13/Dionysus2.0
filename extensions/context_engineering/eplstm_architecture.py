#!/usr/bin/env python3
"""
🧠 Episodic LSTM Architecture for ASI-Arch Context Engineering
=============================================================

Implementation of episodic LSTM (epLSTM) architecture based on Ritter et al. (2018)
"Been There, Done That: Meta-Learning with Episodic Recall" integrated with our
archetypal resonance framework and ASI-Arch neural architecture discovery.

Key Features:
- Differentiable Neural Dictionary (DND) for episodic memory
- Reinstatement gates for memory integration
- Context-based retrieval for task reoccurrence
- Integration with archetypal pattern recognition
- Meta-learning for architecture evolution

Author: ASI-Arch Context Engineering Extension  
Date: 2025-09-22
Version: 1.0.0 - Initial epLSTM Architecture Design
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np
import json
import logging

# Import our theoretical foundations
from .theoretical_foundations import (
    EpisodicMetaLearningProfile,
    TaskReoccurrenceStrategy,
    EpisodicMemoryType,
    ReinstatementGateFunction,
    ArchetypalResonancePattern,
    IntegratedContextEngineering
)

# Nemori-inspired components (https://github.com/nemori-ai/nemori.git)
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    print("Warning: rank_bm25 not available. Install with: pip install rank-bm25")

# =============================================================================
# Nemori-Inspired Architecture Episode Components
# =============================================================================

@dataclass
class ArchitectureEpisode:
    """Nemori-inspired episode for architecture evolution (inspired by https://github.com/nemori-ai/nemori.git)"""
    
    # Core Episode Content (Human episodic memory granularity)
    episode_id: str                         # Unique identifier
    title: str                              # Human-readable episode title
    narrative_summary: str                  # Natural language description
    
    # Temporal Context (Nemori's time-aware approach)
    start_evaluation: int                   # Starting evaluation number
    end_evaluation: int                     # Ending evaluation number
    relative_time_markers: List[str]        # "after initial exploration", etc.
    episode_duration: float                 # Time span of episode
    
    # Architectural Content (Architecture-specific adaptation)
    key_architectural_patterns: List[str]   # Main architectural features
    performance_trajectory: Dict[str, float] # Performance metrics over episode
    breakthrough_moments: List[str]         # Key discoveries or improvements
    exploration_phase: str                  # "exploration", "exploitation", "refinement"
    
    # Semantic Details (Nemori's semantic memory preservation)
    architectural_parameters: Dict[str, Any] # Specific parameters and values
    evaluation_details: List[Dict]          # Detailed evaluation results
    technical_keywords: List[str]           # Architecture-specific terms
    
    # Archetypal Context (Our unique addition)
    dominant_archetype: Optional[ArchetypalResonancePattern]
    narrative_coherence_score: float
    archetypal_resonance_strength: float
    
    # Retrieval Metadata (Nemori's BM25 approach)
    bm25_tokens: List[str] = field(default_factory=list)  # Tokenized content for BM25
    semantic_keywords: List[str] = field(default_factory=list) # Key terms for retrieval
    
    def to_bm25_text(self) -> str:
        """Convert episode to text for BM25 indexing"""
        text_parts = [
            self.title,
            self.narrative_summary,
            " ".join(self.key_architectural_patterns),
            " ".join(self.breakthrough_moments),
            " ".join(self.technical_keywords),
            self.exploration_phase
        ]
        
        # Add archetype information if available
        if self.dominant_archetype:
            text_parts.append(self.dominant_archetype.value.replace('_', ' '))
        
        return " ".join(filter(None, text_parts))

class EpisodeBoundaryDetector:
    """Detect episode boundaries in architecture evolution sequences (Nemori-inspired)"""
    
    def __init__(self, min_episode_length: int = 5, max_episode_length: int = 50):
        self.min_episode_length = min_episode_length
        self.max_episode_length = max_episode_length
        self.logger = logging.getLogger(__name__)
    
    def detect_boundaries(self, evolution_sequence: List[Dict[str, Any]]) -> List[int]:
        """Detect natural episode boundaries in architecture evolution"""
        
        if len(evolution_sequence) < self.min_episode_length:
            return []
        
        boundaries = []
        
        # Analyze sequence for natural boundaries
        for i in range(self.min_episode_length, len(evolution_sequence) - self.min_episode_length):
            
            # Skip if would create too short episodes
            if boundaries and (i - boundaries[-1]) < self.min_episode_length:
                continue
            
            # Check for performance shifts
            if self._detect_performance_shift(evolution_sequence, i):
                boundaries.append(i)
                continue
            
            # Check for architectural pattern changes
            if self._detect_architectural_change(evolution_sequence, i):
                boundaries.append(i)
                continue
            
            # Check for phase transitions (exploration -> exploitation)
            if self._detect_phase_transition(evolution_sequence, i):
                boundaries.append(i)
                continue
        
        # Ensure no episode exceeds max length
        boundaries = self._enforce_max_length(boundaries, len(evolution_sequence))
        
        self.logger.info(f"Detected {len(boundaries)} episode boundaries in sequence of {len(evolution_sequence)}")
        return boundaries
    
    def _detect_performance_shift(self, sequence: List[Dict], position: int) -> bool:
        """Detect significant performance shifts"""
        window_size = 3
        
        if position < window_size or position >= len(sequence) - window_size:
            return False
        
        # Calculate performance before and after
        before_scores = [seq.get('performance', 0.0) for seq in sequence[position-window_size:position]]
        after_scores = [seq.get('performance', 0.0) for seq in sequence[position:position+window_size]]
        
        if not before_scores or not after_scores:
            return False
        
        before_avg = np.mean(before_scores)
        after_avg = np.mean(after_scores)
        
        # Detect significant improvement or degradation
        performance_shift = abs(after_avg - before_avg) / max(before_avg, 0.01)
        return performance_shift > 0.2  # 20% performance shift threshold
    
    def _detect_architectural_change(self, sequence: List[Dict], position: int) -> bool:
        """Detect major architectural pattern changes"""
        if position == 0 or position >= len(sequence):
            return False
        
        current = sequence[position]
        previous = sequence[position - 1]
        
        # Check for architecture type changes (simplified)
        current_arch = current.get('architecture_type', '')
        previous_arch = previous.get('architecture_type', '')
        
        return current_arch != previous_arch and current_arch and previous_arch
    
    def _detect_phase_transition(self, sequence: List[Dict], position: int) -> bool:
        """Detect transitions between exploration and exploitation phases"""
        window_size = 5
        
        if position < window_size or position >= len(sequence) - window_size:
            return False
        
        # Analyze diversity before and after
        before_diversity = self._calculate_diversity(sequence[position-window_size:position])
        after_diversity = self._calculate_diversity(sequence[position:position+window_size])
        
        # Detect phase transition (high->low diversity = exploration->exploitation)
        diversity_change = before_diversity - after_diversity
        return diversity_change > 0.3  # Significant diversity reduction
    
    def _calculate_diversity(self, segment: List[Dict]) -> float:
        """Calculate architectural diversity in a segment"""
        if not segment:
            return 0.0
        
        # Simple diversity measure based on unique architecture types
        arch_types = [s.get('architecture_type', 'unknown') for s in segment]
        unique_types = len(set(arch_types))
        
        return unique_types / len(segment)
    
    def _enforce_max_length(self, boundaries: List[int], total_length: int) -> List[int]:
        """Ensure no episode exceeds maximum length"""
        adjusted_boundaries = []
        last_boundary = 0
        
        for boundary in boundaries:
            if boundary - last_boundary > self.max_episode_length:
                # Insert intermediate boundary
                intermediate = last_boundary + self.max_episode_length
                adjusted_boundaries.append(intermediate)
            adjusted_boundaries.append(boundary)
            last_boundary = boundary
        
        # Check final segment
        if total_length - last_boundary > self.max_episode_length:
            final_boundary = last_boundary + self.max_episode_length
            if final_boundary < total_length - self.min_episode_length:
                adjusted_boundaries.append(final_boundary)
        
        return adjusted_boundaries

class ArchitectureEpisodeGenerator:
    """Generate human-readable architecture episodes (Nemori-inspired)"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def generate_episode(self, 
                        segment: List[Dict[str, Any]], 
                        start_idx: int, 
                        end_idx: int,
                        archetypal_pattern: Optional[ArchetypalResonancePattern] = None) -> ArchitectureEpisode:
        """Generate episodic summary of architecture evolution segment"""
        
        if not segment:
            raise ValueError("Cannot generate episode from empty segment")
        
        # Extract key information from segment
        episode_analysis = self._analyze_segment(segment)
        
        # Generate human-readable narrative
        narrative = self._generate_narrative(segment, episode_analysis, archetypal_pattern)
        
        # Create episode
        episode = ArchitectureEpisode(
            episode_id=f"episode_{start_idx}_{end_idx}",
            title=episode_analysis['title'],
            narrative_summary=narrative,
            start_evaluation=start_idx,
            end_evaluation=end_idx,
            relative_time_markers=episode_analysis['time_markers'],
            episode_duration=end_idx - start_idx,
            key_architectural_patterns=episode_analysis['patterns'],
            performance_trajectory=episode_analysis['performance'],
            breakthrough_moments=episode_analysis['breakthroughs'],
            exploration_phase=episode_analysis['phase'],
            architectural_parameters=episode_analysis['parameters'],
            evaluation_details=segment,
            technical_keywords=episode_analysis['keywords'],
            dominant_archetype=archetypal_pattern,
            narrative_coherence_score=episode_analysis['coherence'],
            archetypal_resonance_strength=episode_analysis.get('resonance', 0.5)
        )
        
        # Generate BM25 tokens
        episode.bm25_tokens = episode.to_bm25_text().lower().split()
        episode.semantic_keywords = episode_analysis['keywords']
        
        self.logger.info(f"Generated episode: {episode.title}")
        return episode
    
    def _analyze_segment(self, segment: List[Dict]) -> Dict[str, Any]:
        """Analyze architecture evolution segment"""
        
        # Extract performance trajectory
        performances = [s.get('performance', 0.0) for s in segment]
        performance_trajectory = {
            'start_performance': performances[0] if performances else 0.0,
            'end_performance': performances[-1] if performances else 0.0,
            'max_performance': max(performances) if performances else 0.0,
            'improvement': (performances[-1] - performances[0]) if len(performances) > 1 else 0.0
        }
        
        # Identify architectural patterns
        patterns = set()
        parameters = {}
        keywords = set()
        
        for eval_data in segment:
            arch_type = eval_data.get('architecture_type', '')
            if arch_type:
                patterns.add(arch_type)
                keywords.add(arch_type.lower())
            
            # Extract parameters
            if 'parameters' in eval_data:
                parameters.update(eval_data['parameters'])
            
            # Extract technical keywords
            for key in eval_data.keys():
                if key in ['layers', 'attention', 'embedding', 'dropout', 'activation']:
                    keywords.add(key)
        
        # Identify breakthrough moments
        breakthroughs = []
        if len(performances) > 1:
            for i in range(1, len(performances)):
                improvement = performances[i] - performances[i-1]
                if improvement > 0.1:  # Significant improvement
                    breakthroughs.append(f"Performance breakthrough at evaluation {i}")
        
        # Determine exploration phase
        diversity = len(patterns) / max(len(segment), 1)
        if diversity > 0.7:
            phase = "exploration"
        elif diversity > 0.3:
            phase = "mixed_exploration_exploitation"
        else:
            phase = "exploitation"
        
        # Generate title
        if performance_trajectory['improvement'] > 0.1:
            title = f"Architecture Improvement Episode ({phase})"
        elif len(patterns) > 1:
            title = f"Architecture Exploration Episode"
        else:
            title = f"Architecture Refinement Episode"
        
        # Time markers
        time_markers = [f"evaluations {segment[0].get('eval_id', 0)}-{segment[-1].get('eval_id', len(segment))}"]
        if phase == "exploration":
            time_markers.append("during exploration phase")
        elif phase == "exploitation":
            time_markers.append("during optimization phase")
        
        return {
            'title': title,
            'patterns': list(patterns),
            'performance': performance_trajectory,
            'breakthroughs': breakthroughs,
            'phase': phase,
            'parameters': parameters,
            'keywords': list(keywords),
            'time_markers': time_markers,
            'coherence': min(1.0, 0.5 + len(breakthroughs) * 0.2)  # Higher coherence with more breakthroughs
        }
    
    def _generate_narrative(self, segment: List[Dict], analysis: Dict, archetype: Optional[ArchetypalResonancePattern]) -> str:
        """Generate human-readable narrative for the episode"""
        
        narrative_parts = []
        
        # Opening with temporal context
        narrative_parts.append(f"During {analysis['time_markers'][0]}, ")
        
        # Describe the exploration phase
        if analysis['phase'] == "exploration":
            narrative_parts.append("the architecture search explored diverse patterns including ")
            narrative_parts.append(", ".join(analysis['patterns'][:3]))  # Limit to first 3 patterns
        elif analysis['phase'] == "exploitation":
            narrative_parts.append("the search focused on refining ")
            narrative_parts.append(analysis['patterns'][0] if analysis['patterns'] else "the current architecture")
        else:
            narrative_parts.append("the search balanced exploration and exploitation across ")
            narrative_parts.append(", ".join(analysis['patterns'][:2]))
        
        # Performance trajectory
        perf = analysis['performance']
        if perf['improvement'] > 0.1:
            narrative_parts.append(f". Performance improved significantly from {perf['start_performance']:.3f} to {perf['end_performance']:.3f}")
        elif perf['improvement'] > 0.05:
            narrative_parts.append(f". Performance showed modest improvement from {perf['start_performance']:.3f} to {perf['end_performance']:.3f}")
        else:
            narrative_parts.append(f". Performance remained relatively stable around {perf['start_performance']:.3f}")
        
        # Breakthrough moments
        if analysis['breakthroughs']:
            narrative_parts.append(f". Key breakthroughs included: {'; '.join(analysis['breakthroughs'][:2])}")
        
        # Archetypal context
        if archetype:
            archetype_name = archetype.value.replace('_', ' ').title()
            narrative_parts.append(f". This episode exemplified the {archetype_name} archetypal pattern")
        
        return "".join(narrative_parts) + "."

# =============================================================================
# epLSTM Architecture Components
# =============================================================================

class MemoryOperation(Enum):
    """Operations performed on episodic memory"""
    STORE = "store"                 # Store new memory
    RETRIEVE = "retrieve"           # Retrieve existing memory
    UPDATE = "update"               # Update existing memory
    FORGET = "forget"               # Remove memory
    CONSOLIDATE = "consolidate"     # Strengthen memory

class ContextSimilarityMetric(Enum):
    """Metrics for measuring context similarity"""
    COSINE_DISTANCE = "cosine_distance"         # Standard cosine similarity
    EUCLIDEAN_DISTANCE = "euclidean_distance"   # L2 distance
    ARCHETYPAL_RESONANCE = "archetypal_resonance" # Based on archetypal patterns
    NARRATIVE_COHERENCE = "narrative_coherence"   # Based on story similarity

@dataclass
class EpisodicMemoryEntry:
    """Single entry in episodic memory"""
    
    # Core Memory Content
    context_key: np.ndarray                 # Context embedding (for retrieval)
    cell_state: np.ndarray                  # LSTM cell state (the "memory")
    hidden_state: np.ndarray                # LSTM hidden state
    
    # Metadata
    timestamp: float                        # When memory was created
    access_count: int                       # How many times accessed
    last_accessed: float                    # Last access timestamp
    consolidation_strength: float           # How "solid" the memory is
    
    # Archetypal Context
    archetypal_pattern: Optional[ArchetypalResonancePattern] # Associated archetype
    narrative_context: Optional[str]        # Story/narrative context
    
    # Task Context
    task_id: Optional[str]                  # Associated task identifier
    performance_outcome: Optional[float]    # How well the task was performed
    
    def update_access(self, current_time: float):
        """Update access statistics"""
        self.access_count += 1
        self.last_accessed = current_time
        # Strengthen memory through access (spacing effect)
        self.consolidation_strength = min(1.0, self.consolidation_strength * 1.1)

@dataclass 
class DifferentiableNeuralDictionary:
    """Differentiable Neural Dictionary for episodic memory storage"""
    
    # Architecture Parameters
    key_dim: int                            # Dimensionality of context keys
    value_dim: int                          # Dimensionality of stored values (cell states)
    max_capacity: int                       # Maximum number of memories
    
    # Memory Storage
    memory_entries: List[EpisodicMemoryEntry] = field(default_factory=list)
    
    # Retrieval Parameters
    k_neighbors: int = 1                    # Number of nearest neighbors to retrieve
    similarity_threshold: float = 0.7       # Minimum similarity for retrieval
    similarity_metric: ContextSimilarityMetric = ContextSimilarityMetric.COSINE_DISTANCE
    
    # Forgetting Parameters
    forgetting_rate: float = 0.01           # Rate of memory decay
    consolidation_threshold: float = 0.8    # Threshold for permanent storage
    
    def store_memory(self, 
                    context_key: np.ndarray,
                    cell_state: np.ndarray, 
                    hidden_state: np.ndarray,
                    archetypal_pattern: Optional[ArchetypalResonancePattern] = None,
                    narrative_context: Optional[str] = None,
                    task_id: Optional[str] = None,
                    performance_outcome: Optional[float] = None) -> int:
        """Store new episodic memory"""
        
        current_time = np.random.random()  # Placeholder for actual timestamp
        
        # Create memory entry
        memory_entry = EpisodicMemoryEntry(
            context_key=context_key,
            cell_state=cell_state,
            hidden_state=hidden_state,
            timestamp=current_time,
            access_count=1,
            last_accessed=current_time,
            consolidation_strength=0.5,  # Start with medium consolidation
            archetypal_pattern=archetypal_pattern,
            narrative_context=narrative_context,
            task_id=task_id,
            performance_outcome=performance_outcome
        )
        
        # Handle capacity limits
        if len(self.memory_entries) >= self.max_capacity:
            self._forget_weakest_memory()
        
        # Store memory
        self.memory_entries.append(memory_entry)
        
        return len(self.memory_entries) - 1  # Return memory index
    
    def retrieve_memory(self, query_context: np.ndarray) -> Tuple[Optional[EpisodicMemoryEntry], float]:
        """Retrieve most similar memory"""
        
        if not self.memory_entries:
            return None, 0.0
        
        best_memory = None
        best_similarity = -1.0
        
        current_time = np.random.random()  # Placeholder for actual timestamp
        
        # Find most similar memory
        for memory in self.memory_entries:
            similarity = self._compute_similarity(query_context, memory.context_key)
            
            if similarity > best_similarity and similarity >= self.similarity_threshold:
                best_similarity = similarity
                best_memory = memory
        
        # Update access statistics if memory found
        if best_memory is not None:
            best_memory.update_access(current_time)
        
        return best_memory, best_similarity
    
    def _compute_similarity(self, query: np.ndarray, key: np.ndarray) -> float:
        """Compute similarity between query and stored key"""
        
        if self.similarity_metric == ContextSimilarityMetric.COSINE_DISTANCE:
            # Cosine similarity
            dot_product = np.dot(query, key)
            norms = np.linalg.norm(query) * np.linalg.norm(key)
            if norms == 0:
                return 0.0
            return dot_product / norms
        
        elif self.similarity_metric == ContextSimilarityMetric.EUCLIDEAN_DISTANCE:
            # Convert Euclidean distance to similarity (0-1 range)
            distance = np.linalg.norm(query - key)
            return 1.0 / (1.0 + distance)
        
        else:
            # Default to cosine similarity
            dot_product = np.dot(query, key)
            norms = np.linalg.norm(query) * np.linalg.norm(key)
            if norms == 0:
                return 0.0
            return dot_product / norms
    
    def _forget_weakest_memory(self):
        """Remove the weakest (least consolidated) memory"""
        if not self.memory_entries:
            return
        
        # Find memory with lowest consolidation strength
        weakest_idx = 0
        weakest_strength = self.memory_entries[0].consolidation_strength
        
        for i, memory in enumerate(self.memory_entries):
            if memory.consolidation_strength < weakest_strength:
                weakest_strength = memory.consolidation_strength
                weakest_idx = i
        
        # Remove weakest memory
        del self.memory_entries[weakest_idx]
    
    def consolidate_memories(self):
        """Strengthen frequently accessed memories"""
        for memory in self.memory_entries:
            # Decay all memories slightly
            memory.consolidation_strength *= (1.0 - self.forgetting_rate)
            
            # Strengthen based on access frequency
            access_bonus = min(0.1, memory.access_count * 0.01)
            memory.consolidation_strength = min(1.0, memory.consolidation_strength + access_bonus)

@dataclass
class ReinstatementGates:
    """Reinstatement gates for coordinating memory integration"""
    
    # Gate Parameters
    input_gate_strength: float = 0.5        # Controls new input integration
    forget_gate_strength: float = 0.3       # Controls forgetting of current state
    reinstatement_gate_strength: float = 0.7 # Controls episodic memory integration
    
    # Gate Functions
    gate_functions: List[ReinstatementGateFunction] = field(
        default_factory=lambda: [ReinstatementGateFunction.COORDINATE_MEMORY_STREAMS]
    )
    
    # Adaptive Parameters
    adaptive_gating: bool = True            # Whether gates adapt based on context
    archetypal_modulation: bool = True      # Whether archetypal patterns modulate gates
    
    def compute_gate_values(self, 
                           current_input: np.ndarray,
                           current_state: np.ndarray,
                           retrieved_memory: Optional[EpisodicMemoryEntry],
                           archetypal_context: Optional[ArchetypalResonancePattern] = None) -> Tuple[float, float, float]:
        """Compute gate values for memory integration"""
        
        # Base gate values
        input_gate = self.input_gate_strength
        forget_gate = self.forget_gate_strength  
        reinstatement_gate = self.reinstatement_gate_strength
        
        # Adaptive gating based on retrieved memory quality
        if self.adaptive_gating and retrieved_memory is not None:
            # Strengthen reinstatement gate for highly consolidated memories
            consolidation_bonus = retrieved_memory.consolidation_strength * 0.3
            reinstatement_gate = min(1.0, reinstatement_gate + consolidation_bonus)
            
            # Adjust other gates to maintain balance
            total_strength = input_gate + forget_gate + reinstatement_gate
            if total_strength > 1.5:  # Normalize if too strong
                factor = 1.5 / total_strength
                input_gate *= factor
                forget_gate *= factor
                reinstatement_gate *= factor
        
        # Archetypal modulation
        if self.archetypal_modulation and archetypal_context is not None:
            if retrieved_memory and retrieved_memory.archetypal_pattern == archetypal_context:
                # Strengthen reinstatement for matching archetypal patterns
                reinstatement_gate = min(1.0, reinstatement_gate * 1.2)
        
        return input_gate, forget_gate, reinstatement_gate

@dataclass
class EpisodicLSTMCell:
    """Episodic LSTM cell with memory integration"""
    
    # Architecture Parameters
    input_dim: int                          # Input dimensionality
    hidden_dim: int                         # Hidden state dimensionality
    cell_dim: int                           # Cell state dimensionality
    
    # Memory Components
    episodic_memory: DifferentiableNeuralDictionary
    reinstatement_gates: ReinstatementGates
    
    # State
    current_hidden: Optional[np.ndarray] = None
    current_cell: Optional[np.ndarray] = None
    
    # Context
    current_context: Optional[np.ndarray] = None
    current_archetype: Optional[ArchetypalResonancePattern] = None
    
    def forward(self, 
               input_vector: np.ndarray,
               context_vector: np.ndarray,
               archetypal_pattern: Optional[ArchetypalResonancePattern] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass through episodic LSTM"""
        
        # Initialize states if needed
        if self.current_hidden is None:
            self.current_hidden = np.zeros(self.hidden_dim)
        if self.current_cell is None:
            self.current_cell = np.zeros(self.cell_dim)
        
        # Store current context
        self.current_context = context_vector
        self.current_archetype = archetypal_pattern
        
        # Retrieve relevant episodic memory
        retrieved_memory, similarity = self.episodic_memory.retrieve_memory(context_vector)
        
        # Compute gate values
        input_gate, forget_gate, reinstatement_gate = self.reinstatement_gates.compute_gate_values(
            input_vector, self.current_cell, retrieved_memory, archetypal_pattern
        )
        
        # Standard LSTM computation (simplified)
        # In a real implementation, this would involve proper weight matrices
        
        # Input contribution (simplified)
        input_contribution = input_vector[:self.cell_dim] * input_gate
        
        # Forget gate application
        forgotten_state = self.current_cell * forget_gate
        
        # Reinstatement contribution
        reinstatement_contribution = np.zeros(self.cell_dim)
        if retrieved_memory is not None:
            reinstatement_contribution = retrieved_memory.cell_state * reinstatement_gate
        
        # Update cell state
        new_cell_state = input_contribution + forgotten_state + reinstatement_contribution
        
        # Update hidden state (simplified)
        new_hidden_state = np.tanh(new_cell_state)
        
        # Store states
        self.current_cell = new_cell_state
        self.current_hidden = new_hidden_state
        
        return new_hidden_state, new_cell_state
    
    def store_episode_memory(self, 
                            task_id: Optional[str] = None,
                            performance_outcome: Optional[float] = None,
                            narrative_context: Optional[str] = None):
        """Store current state as episodic memory at episode end"""
        
        if self.current_context is not None and self.current_cell is not None:
            self.episodic_memory.store_memory(
                context_key=self.current_context,
                cell_state=self.current_cell.copy(),
                hidden_state=self.current_hidden.copy(),
                archetypal_pattern=self.current_archetype,
                narrative_context=narrative_context,
                task_id=task_id,
                performance_outcome=performance_outcome
            )

# =============================================================================
# ASI-Arch Integration
# =============================================================================

class ASIArchEpisodicMetaLearner:
    """Episodic meta-learning system for ASI-Arch neural architecture discovery"""
    
    def __init__(self, 
                 integrated_profile: IntegratedContextEngineering,
                 architecture_dim: int = 256,
                 context_dim: int = 128):
        """Initialize episodic meta-learner for ASI-Arch"""
        
        self.profile = integrated_profile
        self.architecture_dim = architecture_dim
        self.context_dim = context_dim
        
        # Create episodic memory system
        self.episodic_memory = DifferentiableNeuralDictionary(
            key_dim=context_dim,
            value_dim=architecture_dim,
            max_capacity=integrated_profile.episodic_profile.hierarchical_memory_levels * 100,
            similarity_threshold=integrated_profile.episodic_profile.task_recognition_threshold
        )
        
        # Create reinstatement gates
        self.reinstatement_gates = ReinstatementGates(
            reinstatement_gate_strength=integrated_profile.episodic_profile.reinstatement_gate_strength,
            adaptive_gating=True,
            archetypal_modulation=True
        )
        
        # Create episodic LSTM
        self.eplstm = EpisodicLSTMCell(
            input_dim=architecture_dim,
            hidden_dim=architecture_dim,
            cell_dim=architecture_dim,
            episodic_memory=self.episodic_memory,
            reinstatement_gates=self.reinstatement_gates
        )
        
        # Task tracking
        self.current_task_id: Optional[str] = None
        self.task_performance_history: Dict[str, List[float]] = {}
        
        # Logging
        self.logger = logging.getLogger(__name__)
    
    def process_architecture_candidate(self, 
                                     architecture_encoding: np.ndarray,
                                     task_context: np.ndarray,
                                     task_id: str,
                                     archetypal_pattern: Optional[ArchetypalResonancePattern] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process architecture candidate with episodic memory"""
        
        self.current_task_id = task_id
        
        # Forward pass through episodic LSTM
        hidden_output, cell_output = self.eplstm.forward(
            input_vector=architecture_encoding,
            context_vector=task_context,
            archetypal_pattern=archetypal_pattern
        )
        
        # Generate insights
        insights = self._generate_processing_insights(task_context, archetypal_pattern)
        
        return hidden_output, insights
    
    def complete_task_episode(self, 
                            performance_score: float,
                            narrative_summary: Optional[str] = None):
        """Complete current task episode and store memory"""
        
        if self.current_task_id is None:
            return
        
        # Store performance history
        if self.current_task_id not in self.task_performance_history:
            self.task_performance_history[self.current_task_id] = []
        self.task_performance_history[self.current_task_id].append(performance_score)
        
        # Store episodic memory
        self.eplstm.store_episode_memory(
            task_id=self.current_task_id,
            performance_outcome=performance_score,
            narrative_context=narrative_summary
        )
        
        # Consolidate memories periodically
        if len(self.task_performance_history[self.current_task_id]) % 10 == 0:
            self.episodic_memory.consolidate_memories()
        
        self.logger.info(f"Completed episode for task {self.current_task_id} with performance {performance_score:.3f}")
        
        # Reset for next episode
        self.current_task_id = None
    
    def _generate_processing_insights(self, 
                                    task_context: np.ndarray,
                                    archetypal_pattern: Optional[ArchetypalResonancePattern]) -> Dict[str, Any]:
        """Generate insights about the processing"""
        
        insights = {}
        
        # Check for retrieved memory
        retrieved_memory, similarity = self.episodic_memory.retrieve_memory(task_context)
        if retrieved_memory is not None:
            insights['memory_retrieved'] = True
            insights['memory_similarity'] = similarity
            insights['memory_access_count'] = retrieved_memory.access_count
            insights['memory_consolidation'] = retrieved_memory.consolidation_strength
            
            # Check for archetypal resonance
            if (archetypal_pattern is not None and 
                retrieved_memory.archetypal_pattern == archetypal_pattern):
                insights['archetypal_resonance'] = True
                insights['resonant_pattern'] = archetypal_pattern.value
        else:
            insights['memory_retrieved'] = False
            insights['novel_context'] = True
        
        # Memory system statistics
        insights['total_memories'] = len(self.episodic_memory.memory_entries)
        insights['memory_capacity_used'] = len(self.episodic_memory.memory_entries) / self.episodic_memory.max_capacity
        
        return insights
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        
        status = {
            'episodic_memory': {
                'total_memories': len(self.episodic_memory.memory_entries),
                'capacity_used': len(self.episodic_memory.memory_entries) / self.episodic_memory.max_capacity,
                'average_consolidation': np.mean([m.consolidation_strength for m in self.episodic_memory.memory_entries]) if self.episodic_memory.memory_entries else 0.0
            },
            'task_history': {
                'unique_tasks': len(self.task_performance_history),
                'total_episodes': sum(len(history) for history in self.task_performance_history.values()),
                'average_performance': np.mean([np.mean(history) for history in self.task_performance_history.values()]) if self.task_performance_history else 0.0
            },
            'integration': {
                'memory_archetype_coupling': self.profile.memory_archetype_coupling,
                'episodic_narrative_coherence': self.profile.episodic_narrative_coherence,
                'temporal_attractor_stability': self.profile.temporal_attractor_stability
            }
        }
        
        return status

# =============================================================================
# Usage Example
# =============================================================================

async def demonstrate_eplstm_architecture():
    """Demonstrate the epLSTM architecture for ASI-Arch"""
    
    print("🧠 Episodic LSTM Architecture for ASI-Arch Context Engineering")
    print("=" * 65)
    
    # Import integrated system creator
    from .theoretical_foundations import create_asi_arch_context_engineering_system
    
    # Create integrated system
    integrated_system = create_asi_arch_context_engineering_system()
    
    # Create episodic meta-learner
    meta_learner = ASIArchEpisodicMetaLearner(integrated_system)
    
    print(f"✅ Created episodic meta-learner")
    print(f"   Memory capacity: {meta_learner.episodic_memory.max_capacity}")
    print(f"   Architecture dim: {meta_learner.architecture_dim}")
    print(f"   Context dim: {meta_learner.context_dim}")
    
    # Simulate architecture discovery episodes
    print("\n🔄 Simulating architecture discovery episodes...")
    
    for episode in range(5):
        # Generate synthetic architecture and context
        architecture = np.random.randn(meta_learner.architecture_dim)
        context = np.random.randn(meta_learner.context_dim)
        task_id = f"task_{episode % 3}"  # Simulate task reoccurrence
        
        # Process with archetypal pattern
        from .theoretical_foundations import ArchetypalResonancePattern
        archetype = ArchetypalResonancePattern.HERO_DRAGON_SLAYER
        
        # Process architecture
        output, insights = meta_learner.process_architecture_candidate(
            architecture, context, task_id, archetype
        )
        
        # Simulate performance (better performance for recurring tasks)
        base_performance = np.random.uniform(0.5, 0.9)
        if insights.get('memory_retrieved', False):
            performance = min(1.0, base_performance + 0.1)  # Bonus for memory retrieval
        else:
            performance = base_performance
        
        # Complete episode
        narrative = f"Episode {episode}: Architecture evolution with {archetype.value} pattern"
        meta_learner.complete_task_episode(performance, narrative)
        
        print(f"   Episode {episode}: Task {task_id}, Performance {performance:.3f}")
        if insights.get('memory_retrieved'):
            print(f"      💭 Retrieved memory (similarity: {insights.get('memory_similarity', 0):.3f})")
        if insights.get('archetypal_resonance'):
            print(f"      🎭 Archetypal resonance detected")
    
    # Show system status
    print("\n📊 Final System Status:")
    status = meta_learner.get_system_status()
    
    print(f"   Episodic Memory: {status['episodic_memory']['total_memories']} memories")
    print(f"   Capacity Used: {status['episodic_memory']['capacity_used']:.1%}")
    print(f"   Average Consolidation: {status['episodic_memory']['average_consolidation']:.3f}")
    print(f"   Unique Tasks: {status['task_history']['unique_tasks']}")
    print(f"   Total Episodes: {status['task_history']['total_episodes']}")
    print(f"   Average Performance: {status['task_history']['average_performance']:.3f}")
    
    print("\n🎯 Integration Parameters:")
    integration = status['integration']
    print(f"   Memory-Archetype Coupling: {integration['memory_archetype_coupling']:.3f}")
    print(f"   Episodic Narrative Coherence: {integration['episodic_narrative_coherence']:.3f}")
    print(f"   Temporal Attractor Stability: {integration['temporal_attractor_stability']:.3f}")
    
    print("\n✨ epLSTM Architecture successfully demonstrated!")

if __name__ == "__main__":
    import asyncio
    asyncio.run(demonstrate_eplstm_architecture())
