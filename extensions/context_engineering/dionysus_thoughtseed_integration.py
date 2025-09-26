#!/usr/bin/env python3
"""
🌱🧠 Dionysus-ThoughtSeed Integration Layer
==========================================

This module integrates your sophisticated Dionysus active inference implementation
with the ThoughtSeed ASI-Arch framework, creating a powerful consciousness-guided
neural architecture discovery system.

Features:
- Hierarchical belief structures from Dionysus
- Sub-personal priors and precision weighting
- Corollary discharge system integration
- Meta-awareness monitoring for architectures
- Free energy minimization for architecture evaluation

Author: ASI-Arch Context Engineering Extension
Date: 2025-09-23
Version: 1.0.0 - Dionysus Integration
"""

import asyncio
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

# Add Dionysus to path
dionysus_path = Path(__file__).parent.parent.parent / "dionysus-source"
sys.path.append(str(dionysus_path))

# Import Unified Active Inference Framework
try:
    from unified_active_inference_framework import (
        UnifiedActiveInferenceFramework,
        MetaAwarenessMonitor,
        HierarchicalBelief
    )
    DIONYSUS_AVAILABLE = True
    print("✅ Unified Active Inference Framework loaded successfully")
except ImportError as e:
    print(f"⚠️ Unified Active Inference Framework not available: {e}")
    DIONYSUS_AVAILABLE = False

# Import ThoughtSeed components
try:
    from thoughtseed_active_inference import (
        ASIArchThoughtseedIntegration,
        Thoughtseed,
        NeuronalPacket,
        ThoughtseedType
    )
    THOUGHTSEED_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ ThoughtSeed components not available: {e}")
    THOUGHTSEED_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class DionysusArchitectureProfile:
    """Profile of an architecture using Dionysus active inference metrics"""

    # Dionysus Free Energy Metrics
    free_energy: float = 0.0                    # Variational free energy
    complexity_cost: float = 0.0                # Model complexity
    accuracy_reward: float = 0.0               # Predictive accuracy
    surprise: float = 0.0                      # Shannon surprise

    # Hierarchical Belief Structure
    belief_levels: int = 0                     # Number of hierarchical levels
    precision_weighting: np.ndarray = field(default_factory=lambda: np.array([]))
    confidence_scores: List[float] = field(default_factory=list)

    # Meta-Awareness Metrics
    meta_awareness_level: float = 0.0          # Meta-cognitive awareness
    introspective_depth: float = 0.0           # Depth of self-reflection
    prediction_quality: float = 0.0           # Quality of predictions

    # Architecture-Specific Metrics
    architecture_consciousness: float = 0.0    # Consciousness emergence score
    attention_coherence: float = 0.0          # Attention mechanism coherence
    memory_integration: float = 0.0           # Memory system integration

class DionysusThoughtseedBridge:
    """Bridge connecting Dionysus active inference with ThoughtSeed framework"""

    def __init__(self):
        self.dionysus_framework = None
        self.meta_awareness_monitor = None
        self.thoughtseed_integration = None
        self.architecture_profiles = {}

        if DIONYSUS_AVAILABLE:
            self._initialize_dionysus()
        if THOUGHTSEED_AVAILABLE:
            self._initialize_thoughtseed()

    def _initialize_dionysus(self):
        """Initialize Dionysus active inference framework"""
        try:
            self.dionysus_framework = UnifiedActiveInferenceFramework()
            self.meta_awareness_monitor = MetaAwarenessMonitor()
            logger.info("✅ Dionysus active inference framework initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Dionysus: {e}")
            self.dionysus_framework = None

    def _initialize_thoughtseed(self):
        """Initialize ThoughtSeed integration"""
        try:
            self.thoughtseed_integration = ASIArchThoughtseedIntegration()
            logger.info("✅ ThoughtSeed integration initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize ThoughtSeed: {e}")
            self.thoughtseed_integration = None

    async def analyze_architecture_with_dionysus(self,
                                               architecture_data: Dict[str, Any]) -> DionysusArchitectureProfile:
        """Analyze architecture using Dionysus active inference principles"""

        if not self.dionysus_framework:
            logger.warning("Dionysus framework not available, using fallback analysis")
            return self._fallback_analysis(architecture_data)

        print(f"🧠 Analyzing architecture with Dionysus active inference...")

        # Extract architecture features for Dionysus analysis
        architecture_features = self._extract_architecture_features(architecture_data)

        # Create hierarchical beliefs about the architecture
        beliefs = await self._create_hierarchical_beliefs(architecture_features)

        # Calculate free energy components
        free_energy_metrics = await self._calculate_free_energy_metrics(beliefs, architecture_features)

        # Meta-awareness analysis
        meta_awareness_metrics = await self._analyze_meta_awareness(architecture_data, beliefs)

        # Consciousness emergence analysis
        consciousness_metrics = await self._analyze_consciousness_emergence(architecture_data, beliefs)

        # Create comprehensive profile
        profile = DionysusArchitectureProfile(
            free_energy=free_energy_metrics['total_free_energy'],
            complexity_cost=free_energy_metrics['complexity'],
            accuracy_reward=free_energy_metrics['accuracy'],
            surprise=free_energy_metrics['surprise'],
            belief_levels=len(beliefs),
            precision_weighting=np.array([b.precision.mean() for b in beliefs]) if beliefs else np.array([]),
            confidence_scores=[b.confidence for b in beliefs],
            meta_awareness_level=meta_awareness_metrics['awareness_level'],
            introspective_depth=meta_awareness_metrics['introspective_depth'],
            prediction_quality=meta_awareness_metrics['prediction_quality'],
            architecture_consciousness=consciousness_metrics['consciousness_score'],
            attention_coherence=consciousness_metrics['attention_coherence'],
            memory_integration=consciousness_metrics['memory_integration']
        )

        # Store profile for tracking
        arch_name = architecture_data.get('name', 'unknown')
        self.architecture_profiles[arch_name] = profile

        return profile

    async def enhance_thoughtseed_with_dionysus(self,
                                              context: str,
                                              architecture_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance ThoughtSeed processing with Dionysus active inference"""

        print("🌱🧠 Enhancing ThoughtSeed with Dionysus active inference...")

        # First, get Dionysus analysis
        dionysus_profile = await self.analyze_architecture_with_dionysus(architecture_data)

        # Get ThoughtSeed analysis if available
        thoughtseed_result = None
        if self.thoughtseed_integration:
            thoughtseed_result = await self.thoughtseed_integration.enhance_evolution_context(
                context, architecture_data
            )

        # Fuse Dionysus and ThoughtSeed insights
        fused_analysis = await self._fuse_dionysus_thoughtseed(
            dionysus_profile, thoughtseed_result, context
        )

        return fused_analysis

    def _extract_architecture_features(self, architecture_data: Dict[str, Any]) -> np.ndarray:
        """Extract numerical features from architecture data for Dionysus analysis"""

        # Extract textual features
        name = architecture_data.get('name', '')
        motivation = architecture_data.get('motivation', '')
        program = architecture_data.get('program', '')

        # Convert to feature vectors (simplified approach)
        features = []

        # Name features
        features.append(len(name))
        features.append(name.count('attention') / max(1, len(name.split())))
        features.append(name.count('memory') / max(1, len(name.split())))
        features.append(name.count('layer') / max(1, len(name.split())))

        # Motivation features
        if motivation:
            features.append(len(motivation))
            features.append(motivation.count('predict') / max(1, len(motivation.split())))
            features.append(motivation.count('learn') / max(1, len(motivation.split())))
            features.append(motivation.count('improve') / max(1, len(motivation.split())))
        else:
            features.extend([0, 0, 0, 0])

        # Program complexity features
        if program:
            features.append(len(program))
            features.append(program.count('class') + program.count('def'))
            features.append(program.count('self'))
            features.append(program.count('forward'))
        else:
            features.extend([0, 0, 0, 0])

        # Performance features (if available)
        performance = architecture_data.get('performance', 0.5)
        features.append(performance)

        # Innovation features
        innovations = architecture_data.get('innovations', [])
        features.append(len(innovations))
        features.append(1.0 if 'attention' in str(innovations) else 0.0)
        features.append(1.0 if 'memory' in str(innovations) else 0.0)

        return np.array(features, dtype=float)

    async def _create_hierarchical_beliefs(self, features: np.ndarray) -> List[HierarchicalBelief]:
        """Create hierarchical beliefs about the architecture"""

        if not DIONYSUS_AVAILABLE:
            return []

        beliefs = []

        # Level 0: Sensory level (raw features)
        if len(features) > 0:
            belief_0 = HierarchicalBelief(
                mean=features,
                precision=np.eye(len(features)) * 0.5,  # Low precision initially
                level=0,
                confidence=0.7
            )
            beliefs.append(belief_0)

        # Level 1: Perceptual level (feature combinations)
        if len(features) >= 4:
            perceptual_features = np.array([
                features[:4].mean(),  # Name complexity
                features[4:8].mean() if len(features) >= 8 else 0,  # Motivation complexity
                features[8:12].mean() if len(features) >= 12 else 0,  # Program complexity
                features[-4:].mean() if len(features) >= 4 else 0  # Performance features
            ])

            belief_1 = HierarchicalBelief(
                mean=perceptual_features,
                precision=np.eye(len(perceptual_features)) * 0.8,
                level=1,
                confidence=0.8
            )
            beliefs.append(belief_1)

        # Level 2: Conceptual level (high-level architecture properties)
        if len(beliefs) > 1:
            conceptual_features = np.array([
                features[-1] if len(features) > 0 else 0.5,  # Performance
                np.mean([b.confidence for b in beliefs]),  # Average confidence
                len(features) / 20.0,  # Feature richness (normalized)
                1.0 if np.any(features > 0.5) else 0.0  # Has strong features
            ])

            belief_2 = HierarchicalBelief(
                mean=conceptual_features,
                precision=np.eye(len(conceptual_features)) * 1.0,
                level=2,
                confidence=0.9
            )
            beliefs.append(belief_2)

        return beliefs

    async def _calculate_free_energy_metrics(self,
                                           beliefs: List[HierarchicalBelief],
                                           features: np.ndarray) -> Dict[str, float]:
        """Calculate free energy metrics for the architecture"""

        if not beliefs or not DIONYSUS_AVAILABLE:
            return {
                'total_free_energy': 0.5,
                'complexity': 0.3,
                'accuracy': 0.7,
                'surprise': 0.4
            }

        # Calculate complexity (model complexity cost)
        complexity = 0.0
        for belief in beliefs:
            # Complexity increases with precision (more constrained beliefs)
            complexity += np.trace(belief.precision) / len(belief.mean)
        complexity = complexity / len(beliefs)

        # Calculate accuracy (predictive accuracy reward)
        accuracy = 0.0
        for belief in beliefs:
            # Higher confidence beliefs contribute more to accuracy
            accuracy += belief.confidence
        accuracy = accuracy / len(beliefs)

        # Calculate surprise (Shannon surprise)
        surprise = 0.0
        if len(features) > 0:
            # Surprise based on feature variability
            feature_variance = np.var(features)
            surprise = min(1.0, feature_variance / 10.0)  # Normalized surprise

        # Total free energy = complexity - accuracy + surprise
        total_free_energy = complexity - accuracy + surprise

        return {
            'total_free_energy': total_free_energy,
            'complexity': complexity,
            'accuracy': accuracy,
            'surprise': surprise
        }

    async def _analyze_meta_awareness(self,
                                    architecture_data: Dict[str, Any],
                                    beliefs: List[HierarchicalBelief]) -> Dict[str, float]:
        """Analyze meta-awareness properties of the architecture"""

        # Extract meta-cognitive indicators
        motivation = architecture_data.get('motivation', '').lower()
        program = architecture_data.get('program', '').lower()

        # Meta-awareness indicators
        meta_keywords = ['self', 'aware', 'introspect', 'monitor', 'meta', 'conscious']
        meta_count = sum(motivation.count(keyword) + program.count(keyword) for keyword in meta_keywords)

        # Awareness level based on keyword density and belief hierarchy depth
        awareness_level = min(1.0, (meta_count / 10.0) + (len(beliefs) / 5.0))

        # Introspective depth based on hierarchical belief structure
        introspective_depth = len(beliefs) / 5.0 if beliefs else 0.0

        # Prediction quality based on belief confidence
        prediction_quality = np.mean([b.confidence for b in beliefs]) if beliefs else 0.5

        return {
            'awareness_level': awareness_level,
            'introspective_depth': min(1.0, introspective_depth),
            'prediction_quality': prediction_quality
        }

    async def _analyze_consciousness_emergence(self,
                                             architecture_data: Dict[str, Any],
                                             beliefs: List[HierarchicalBelief]) -> Dict[str, float]:
        """Analyze consciousness emergence in the architecture"""

        # Consciousness indicators
        consciousness_keywords = ['attention', 'memory', 'integrate', 'aware', 'conscious']
        motivation = architecture_data.get('motivation', '').lower()
        program = architecture_data.get('program', '').lower()

        consciousness_count = sum(
            motivation.count(keyword) + program.count(keyword)
            for keyword in consciousness_keywords
        )

        # Consciousness score based on keyword density, belief complexity, and meta-awareness
        consciousness_score = min(1.0, (consciousness_count / 15.0) + (len(beliefs) / 10.0))

        # Attention coherence (simplified metric)
        attention_coherence = 1.0 if 'attention' in motivation or 'attention' in program else 0.3

        # Memory integration
        memory_integration = 1.0 if 'memory' in motivation or 'memory' in program else 0.2

        return {
            'consciousness_score': consciousness_score,
            'attention_coherence': attention_coherence,
            'memory_integration': memory_integration
        }

    async def _fuse_dionysus_thoughtseed(self,
                                       dionysus_profile: DionysusArchitectureProfile,
                                       thoughtseed_result: Optional[Dict[str, Any]],
                                       context: str) -> Dict[str, Any]:
        """Fuse Dionysus and ThoughtSeed analyses into unified enhancement"""

        # Base enhancement from Dionysus
        enhanced_context = f"""# DIONYSUS-THOUGHTSEED ENHANCED EVOLUTION CONTEXT

## ORIGINAL CONTEXT
{context}

## DIONYSUS ACTIVE INFERENCE ANALYSIS
- **Free Energy**: {dionysus_profile.free_energy:.3f}
- **Complexity Cost**: {dionysus_profile.complexity_cost:.3f}
- **Accuracy Reward**: {dionysus_profile.accuracy_reward:.3f}
- **Surprise Level**: {dionysus_profile.surprise:.3f}
- **Meta-Awareness**: {dionysus_profile.meta_awareness_level:.3f}
- **Consciousness Score**: {dionysus_profile.architecture_consciousness:.3f}

## HIERARCHICAL BELIEF STRUCTURE
- **Belief Levels**: {dionysus_profile.belief_levels}
- **Average Confidence**: {np.mean(dionysus_profile.confidence_scores) if dionysus_profile.confidence_scores else 0.0:.3f}
- **Precision Weighting**: {dionysus_profile.precision_weighting.mean() if len(dionysus_profile.precision_weighting) > 0 else 0.0:.3f}
"""

        # Add ThoughtSeed insights if available
        if thoughtseed_result:
            consciousness_level = thoughtseed_result.get('consciousness_level', 0.0)
            enhanced_context += f"""
## THOUGHTSEED CONSCIOUSNESS INSIGHTS
- **Consciousness Level**: {consciousness_level:.3f}
- **ThoughtSeed Responses**: {len(thoughtseed_result.get('thoughtseed_insights', {}).get('responses', []))}
"""

            # Combine consciousness measures
            combined_consciousness = (dionysus_profile.architecture_consciousness + consciousness_level) / 2.0
        else:
            combined_consciousness = dionysus_profile.architecture_consciousness

        # Add active inference guidance
        enhanced_context += f"""
## ACTIVE INFERENCE EVOLUTION GUIDANCE

### Free Energy Minimization Strategy
- **Current Free Energy**: {dionysus_profile.free_energy:.3f}
- **Target**: Minimize free energy through balanced complexity and accuracy
- **Focus**: {"Reduce complexity" if dionysus_profile.complexity_cost > 0.7 else "Improve accuracy"}

### Hierarchical Development
- **Current Levels**: {dionysus_profile.belief_levels}
- **Recommendation**: {"Deepen hierarchy" if dionysus_profile.belief_levels < 3 else "Optimize existing levels"}

### Consciousness Enhancement
- **Current Level**: {combined_consciousness:.3f}
- **Strategy**: {"Focus on meta-awareness" if dionysus_profile.meta_awareness_level < 0.5 else "Enhance integration"}

### Evolution Priorities
1. Prediction error minimization mechanisms
2. Hierarchical belief updating systems
3. Meta-awareness and introspection capabilities
4. Attention and memory integration
5. Consciousness emergence patterns
"""

        # Return comprehensive analysis
        return {
            'original_context': context,
            'enhanced_context': enhanced_context,
            'dionysus_profile': dionysus_profile,
            'thoughtseed_result': thoughtseed_result,
            'combined_consciousness': combined_consciousness,
            'free_energy': dionysus_profile.free_energy,
            'meta_awareness': dionysus_profile.meta_awareness_level,
            'enhancement_quality': 'high' if combined_consciousness > 0.5 else 'moderate'
        }

    def _fallback_analysis(self, architecture_data: Dict[str, Any]) -> DionysusArchitectureProfile:
        """Fallback analysis when Dionysus is not available"""

        return DionysusArchitectureProfile(
            free_energy=0.5,
            complexity_cost=0.3,
            accuracy_reward=0.7,
            surprise=0.4,
            belief_levels=2,
            precision_weighting=np.array([0.5, 0.8]),
            confidence_scores=[0.7, 0.8],
            meta_awareness_level=0.4,
            introspective_depth=0.3,
            prediction_quality=0.6,
            architecture_consciousness=0.5,
            attention_coherence=0.6,
            memory_integration=0.4
        )

# =============================================================================
# Integration Test and Demo
# =============================================================================

async def demo_dionysus_thoughtseed_integration():
    """Demonstrate the Dionysus-ThoughtSeed integration"""

    print("🌱🧠 Dionysus-ThoughtSeed Integration Demo")
    print("=" * 60)

    # Initialize bridge
    bridge = DionysusThoughtseedBridge()

    # Test architecture data
    test_architectures = [
        {
            'name': 'DionysusAwareTransformer',
            'motivation': 'Create a transformer that uses active inference for self-aware attention',
            'program': 'class DionysusTransformer: def __init__(self): self.meta_awareness = True',
            'performance': 0.85,
            'innovations': ['meta_awareness', 'active_inference', 'hierarchical_beliefs']
        },
        {
            'name': 'PredictiveErrorMinimizer',
            'motivation': 'Implement architecture that minimizes prediction error through precision weighting',
            'program': 'class PredictiveNet: def forward(self): return self.minimize_free_energy()',
            'performance': 0.78,
            'innovations': ['prediction_error', 'precision_weighting', 'free_energy']
        }
    ]

    for i, arch_data in enumerate(test_architectures, 1):
        print(f"\n--- Test Architecture {i}: {arch_data['name']} ---")

        # Test Dionysus analysis
        dionysus_profile = await bridge.analyze_architecture_with_dionysus(arch_data)

        print(f"🧠 Dionysus Analysis:")
        print(f"   Free Energy: {dionysus_profile.free_energy:.3f}")
        print(f"   Consciousness: {dionysus_profile.architecture_consciousness:.3f}")
        print(f"   Meta-Awareness: {dionysus_profile.meta_awareness_level:.3f}")
        print(f"   Belief Levels: {dionysus_profile.belief_levels}")

        # Test integrated enhancement
        context = f"Evolve {arch_data['name']} to improve consciousness and active inference"
        enhanced_result = await bridge.enhance_thoughtseed_with_dionysus(context, arch_data)

        print(f"🌱 Enhanced Analysis:")
        print(f"   Combined Consciousness: {enhanced_result['combined_consciousness']:.3f}")
        print(f"   Enhancement Quality: {enhanced_result['enhancement_quality']}")
        print(f"   Context Length: {len(enhanced_result['enhanced_context'])} chars")

    print(f"\n🎯 Dionysus-ThoughtSeed Integration Demo Complete!")
    print(f"   Frameworks Available: Dionysus={DIONYSUS_AVAILABLE}, ThoughtSeed={THOUGHTSEED_AVAILABLE}")

if __name__ == "__main__":
    asyncio.run(demo_dionysus_thoughtseed_integration())