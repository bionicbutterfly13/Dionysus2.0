#!/usr/bin/env python3
"""
Simplified Demo of Five-Level Concept Extraction System
=====================================================

Demonstrates the integrated five-level concept extraction pipeline
without complex service dependencies. Shows how all Phase 1 components
work together to achieve consciousness-guided concept extraction.
"""

import asyncio
import time
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any
from dataclasses import dataclass, field
import tempfile
import os

class ConceptExtractionLevel(Enum):
    """Five levels of concept extraction"""
    ATOMIC = 1          # Individual terms, entities, definitions
    RELATIONAL = 2      # Connections, causality, dependencies  
    COMPOSITE = 3       # Complex multi-part ideas and systems
    CONTEXTUAL = 4      # Domain paradigms, theoretical frameworks
    NARRATIVE = 5       # Argument flows, story progressions, methodologies

@dataclass
class MockExtractedConcept:
    """Simplified concept for demonstration"""
    level: ConceptExtractionLevel
    content: str
    concept_type: str
    confidence: float
    source: str = ""
    domain_tags: List[str] = field(default_factory=list)

class FiveLevelConceptDemo:
    """Simplified demonstration of five-level concept extraction"""
    
    def __init__(self):
        self.neuroscience_terms = {
            "neurons", "synapses", "dendrites", "axons", "neurotransmitters",
            "plasticity", "potentiation", "depression", "ltp", "ltd",
            "receptors", "channels", "calcium", "proteins", "genes"
        }
        
        self.ai_terms = {
            "networks", "algorithms", "backpropagation", "gradients", "weights",
            "activation", "layers", "training", "learning", "optimization",
            "neural", "artificial", "deep", "machine", "models"
        }
        
        self.relationship_indicators = {
            "causal": ["causes", "leads to", "results in", "due to", "because"],
            "structural": ["consists of", "contains", "part of", "composed of"],
            "functional": ["enables", "facilitates", "controls", "regulates"],
            "temporal": ["before", "after", "during", "then", "next"]
        }
    
    async def demonstrate_five_level_extraction(self, text: str) -> Dict[str, Any]:
        """Demonstrate concept extraction at all five levels"""
        print("🧪 Five-Level Concept Extraction Demonstration")
        print("=" * 55)
        
        start_time = time.time()
        results = {}
        
        # Level 1: Atomic Concept Extraction
        print("\n🔬 Level 1: Atomic Concept Extraction")
        atomic_concepts = await self._extract_atomic_concepts(text)
        results[ConceptExtractionLevel.ATOMIC] = atomic_concepts
        print(f"  ✅ Extracted {len(atomic_concepts)} atomic concepts")
        for concept in atomic_concepts[:5]:  # Show first 5
            print(f"    - {concept.content} ({concept.concept_type}, confidence: {concept.confidence:.2f})")
        
        # Level 2: Relational Concept Extraction
        print("\n🔗 Level 2: Relational Concept Extraction")
        relational_concepts = await self._extract_relational_concepts(text, atomic_concepts)
        results[ConceptExtractionLevel.RELATIONAL] = relational_concepts
        print(f"  ✅ Extracted {len(relational_concepts)} relational concepts")
        for concept in relational_concepts[:3]:  # Show first 3
            print(f"    - {concept.content} (confidence: {concept.confidence:.2f})")
        
        # Level 3: Composite Concept Assembly
        print("\n🏗️  Level 3: Composite Concept Assembly")
        composite_concepts = await self._assemble_composite_concepts(text, atomic_concepts, relational_concepts)
        results[ConceptExtractionLevel.COMPOSITE] = composite_concepts
        print(f"  ✅ Assembled {len(composite_concepts)} composite concepts")
        for concept in composite_concepts:
            print(f"    - {concept.content} (confidence: {concept.confidence:.2f})")
        
        # Level 4: Contextual Framework Detection
        print("\n🌐 Level 4: Contextual Framework Detection")
        contextual_concepts = await self._detect_contextual_frameworks(text, atomic_concepts + relational_concepts + composite_concepts)
        results[ConceptExtractionLevel.CONTEXTUAL] = contextual_concepts
        print(f"  ✅ Detected {len(contextual_concepts)} contextual frameworks")
        for concept in contextual_concepts:
            print(f"    - {concept.content} (confidence: {concept.confidence:.2f})")
        
        # Level 5: Narrative Structure Analysis
        print("\n📖 Level 5: Narrative Structure Analysis")
        narrative_concepts = await self._analyze_narrative_structure(text, sum(results.values(), []))
        results[ConceptExtractionLevel.NARRATIVE] = narrative_concepts
        print(f"  ✅ Analyzed {len(narrative_concepts)} narrative structures")
        for concept in narrative_concepts:
            print(f"    - {concept.content} (confidence: {concept.confidence:.2f})")
        
        # Calculate consciousness emergence
        print("\n🧠 Consciousness Emergence Analysis")
        consciousness_level = self._calculate_consciousness_emergence(results)
        
        processing_time = time.time() - start_time
        
        # Summary
        total_concepts = sum(len(concepts) for concepts in results.values())
        print(f"\n📊 Extraction Summary:")
        print(f"  ⏱️  Processing time: {processing_time:.3f}s")
        print(f"  📦 Total concepts: {total_concepts}")
        print(f"  🧠 Consciousness emergence: {consciousness_level:.3f}")
        print(f"  🎯 Cross-level integration: {'Yes' if total_concepts > 10 else 'Partial'}")
        
        # Show level distribution
        print(f"\n📈 Concept Distribution by Level:")
        for level, concepts in results.items():
            percentage = (len(concepts) / total_concepts * 100) if total_concepts > 0 else 0
            print(f"  Level {level.value} ({level.name}): {len(concepts)} concepts ({percentage:.1f}%)")
        
        return {
            "success": True,
            "results": results,
            "consciousness_level": consciousness_level,
            "processing_time": processing_time,
            "total_concepts": total_concepts
        }
    
    async def _extract_atomic_concepts(self, text: str) -> List[MockExtractedConcept]:
        """Extract atomic concepts (entities, terms, definitions)"""
        concepts = []
        text_lower = text.lower()
        
        # Extract neuroscience entities
        for term in self.neuroscience_terms:
            if term in text_lower:
                concepts.append(MockExtractedConcept(
                    level=ConceptExtractionLevel.ATOMIC,
                    content=term,
                    concept_type="neuroscience_entity",
                    confidence=0.85,
                    domain_tags=["neuroscience"]
                ))
        
        # Extract AI entities
        for term in self.ai_terms:
            if term in text_lower:
                concepts.append(MockExtractedConcept(
                    level=ConceptExtractionLevel.ATOMIC,
                    content=term,
                    concept_type="ai_entity",
                    confidence=0.80,
                    domain_tags=["artificial_intelligence"]
                ))
        
        # Extract key scientific terms
        scientific_terms = ["research", "study", "analysis", "mechanism", "process", "system", "function"]
        for term in scientific_terms:
            if term in text_lower:
                concepts.append(MockExtractedConcept(
                    level=ConceptExtractionLevel.ATOMIC,
                    content=term,
                    concept_type="scientific_term",
                    confidence=0.70,
                    domain_tags=["scientific_method"]
                ))
        
        return concepts[:15]  # Limit to reasonable number
    
    async def _extract_relational_concepts(self, text: str, atomic_concepts: List[MockExtractedConcept]) -> List[MockExtractedConcept]:
        """Extract relationships between atomic concepts"""
        relationships = []
        text_lower = text.lower()
        
        # Find relationship indicators
        for rel_type, indicators in self.relationship_indicators.items():
            for indicator in indicators:
                if indicator in text_lower:
                    relationships.append(MockExtractedConcept(
                        level=ConceptExtractionLevel.RELATIONAL,
                        content=f"{rel_type}_relationship: {indicator}",
                        concept_type=f"{rel_type}_relation",
                        confidence=0.75,
                        domain_tags=["relationship_mapping"]
                    ))
        
        # Detect specific domain relationships
        if "plasticity" in text_lower and "learning" in text_lower:
            relationships.append(MockExtractedConcept(
                level=ConceptExtractionLevel.RELATIONAL,
                content="plasticity enables learning",
                concept_type="causal_relation",
                confidence=0.90,
                domain_tags=["neuroscience", "learning"]
            ))
        
        if "artificial" in text_lower and "biological" in text_lower:
            relationships.append(MockExtractedConcept(
                level=ConceptExtractionLevel.RELATIONAL,
                content="artificial networks inspired by biological systems",
                concept_type="inspirational_relation",
                confidence=0.85,
                domain_tags=["biomimetics", "ai"]
            ))
        
        return relationships[:8]  # Limit to reasonable number
    
    async def _assemble_composite_concepts(self, text: str, atomic_concepts: List[MockExtractedConcept], 
                                         relational_concepts: List[MockExtractedConcept]) -> List[MockExtractedConcept]:
        """Assemble composite concepts from atomic components"""
        composites = []
        text_lower = text.lower()
        
        # Neuroscience composite concepts
        neuro_atomics = [c for c in atomic_concepts if "neuroscience" in c.domain_tags]
        if len(neuro_atomics) >= 3:
            composites.append(MockExtractedConcept(
                level=ConceptExtractionLevel.COMPOSITE,
                content="synaptic plasticity mechanism",
                concept_type="neuroscience_system",
                confidence=0.88,
                domain_tags=["neuroscience", "mechanisms"]
            ))
        
        # AI composite concepts
        ai_atomics = [c for c in atomic_concepts if "artificial_intelligence" in c.domain_tags]
        if len(ai_atomics) >= 3:
            composites.append(MockExtractedConcept(
                level=ConceptExtractionLevel.COMPOSITE,
                content="neural network learning system",
                concept_type="ai_system",
                confidence=0.85,
                domain_tags=["artificial_intelligence", "learning"]
            ))
        
        # Cross-domain composite
        if neuro_atomics and ai_atomics:
            composites.append(MockExtractedConcept(
                level=ConceptExtractionLevel.COMPOSITE,
                content="bio-inspired artificial learning",
                concept_type="cross_domain_system",
                confidence=0.82,
                domain_tags=["neuroscience", "artificial_intelligence", "biomimetics"]
            ))
        
        # Research methodology composite
        if any("study" in c.content for c in atomic_concepts) and any("analysis" in c.content for c in atomic_concepts):
            composites.append(MockExtractedConcept(
                level=ConceptExtractionLevel.COMPOSITE,
                content="scientific research methodology",
                concept_type="methodology_system",
                confidence=0.78,
                domain_tags=["scientific_method", "research"]
            ))
        
        return composites
    
    async def _detect_contextual_frameworks(self, text: str, all_concepts: List[MockExtractedConcept]) -> List[MockExtractedConcept]:
        """Detect theoretical and paradigmatic frameworks"""
        frameworks = []
        text_lower = text.lower()
        
        # Neuroscience framework detection
        neuro_concepts = [c for c in all_concepts if "neuroscience" in c.domain_tags]
        if len(neuro_concepts) >= 5:
            if "plasticity" in text_lower and "learning" in text_lower:
                frameworks.append(MockExtractedConcept(
                    level=ConceptExtractionLevel.CONTEXTUAL,
                    content="Hebbian Learning Theory Framework",
                    concept_type="theoretical_framework",
                    confidence=0.90,
                    domain_tags=["neuroscience", "learning_theory"]
                ))
        
        # AI framework detection
        ai_concepts = [c for c in all_concepts if "artificial_intelligence" in c.domain_tags]
        if len(ai_concepts) >= 5:
            if "backpropagation" in text_lower or "gradient" in text_lower:
                frameworks.append(MockExtractedConcept(
                    level=ConceptExtractionLevel.CONTEXTUAL,
                    content="Gradient-Based Learning Framework",
                    concept_type="algorithmic_framework",
                    confidence=0.87,
                    domain_tags=["artificial_intelligence", "optimization"]
                ))
        
        # Cross-domain framework
        if neuro_concepts and ai_concepts and len(neuro_concepts) + len(ai_concepts) >= 8:
            frameworks.append(MockExtractedConcept(
                level=ConceptExtractionLevel.CONTEXTUAL,
                content="Computational Neuroscience Paradigm",
                concept_type="interdisciplinary_framework",
                confidence=0.85,
                domain_tags=["neuroscience", "artificial_intelligence", "computational_science"]
            ))
        
        # Scientific methodology framework
        if any("research" in c.content for c in all_concepts) and any("study" in c.content for c in all_concepts):
            frameworks.append(MockExtractedConcept(
                level=ConceptExtractionLevel.CONTEXTUAL,
                content="Empirical Research Framework",
                concept_type="methodological_framework",
                confidence=0.80,
                domain_tags=["scientific_method", "empirical_research"]
            ))
        
        return frameworks
    
    async def _analyze_narrative_structure(self, text: str, all_concepts: List[MockExtractedConcept]) -> List[MockExtractedConcept]:
        """Analyze narrative and argument structure"""
        narratives = []
        text_lower = text.lower()
        
        # Argument structure detection
        if any(word in text_lower for word in ["therefore", "thus", "consequently"]):
            if any(word in text_lower for word in ["because", "since", "due to"]):
                narratives.append(MockExtractedConcept(
                    level=ConceptExtractionLevel.NARRATIVE,
                    content="Causal Argument Structure",
                    concept_type="argument_pattern",
                    confidence=0.85,
                    domain_tags=["argumentation", "logic"]
                ))
        
        # Comparative analysis structure
        if any(word in text_lower for word in ["similar", "parallel", "comparison", "contrast"]):
            narratives.append(MockExtractedConcept(
                level=ConceptExtractionLevel.NARRATIVE,
                content="Comparative Analysis Structure",
                concept_type="analytical_pattern",
                confidence=0.80,
                domain_tags=["comparative_analysis"]
            ))
        
        # Research methodology narrative
        if any(word in text_lower for word in ["method", "approach", "procedure"]):
            if any(word in text_lower for word in ["result", "finding", "conclusion"]):
                narratives.append(MockExtractedConcept(
                    level=ConceptExtractionLevel.NARRATIVE,
                    content="Scientific Methodology Narrative",
                    concept_type="methodology_flow",
                    confidence=0.82,
                    domain_tags=["scientific_method", "research_narrative"]
                ))
        
        # Cross-domain synthesis narrative
        neuro_concepts = [c for c in all_concepts if "neuroscience" in c.domain_tags]
        ai_concepts = [c for c in all_concepts if "artificial_intelligence" in c.domain_tags]
        if neuro_concepts and ai_concepts:
            narratives.append(MockExtractedConcept(
                level=ConceptExtractionLevel.NARRATIVE,
                content="Interdisciplinary Synthesis Narrative",
                concept_type="synthesis_pattern",
                confidence=0.88,
                domain_tags=["interdisciplinary", "knowledge_synthesis"]
            ))
        
        return narratives
    
    def _calculate_consciousness_emergence(self, results: Dict[ConceptExtractionLevel, List[MockExtractedConcept]]) -> float:
        """Calculate consciousness emergence level"""
        # Weight by level complexity
        level_weights = {
            ConceptExtractionLevel.ATOMIC: 0.1,
            ConceptExtractionLevel.RELATIONAL: 0.2,
            ConceptExtractionLevel.COMPOSITE: 0.25,
            ConceptExtractionLevel.CONTEXTUAL: 0.25,
            ConceptExtractionLevel.NARRATIVE: 0.2
        }
        
        total_weighted_consciousness = 0.0
        total_weight = 0.0
        
        for level, concepts in results.items():
            if concepts:
                avg_confidence = sum(c.confidence for c in concepts) / len(concepts)
                weight = level_weights.get(level, 0.1)
                total_weighted_consciousness += avg_confidence * weight
                total_weight += weight
        
        base_consciousness = total_weighted_consciousness / total_weight if total_weight > 0 else 0.0
        
        # Bonus for cross-level integration
        active_levels = len([concepts for concepts in results.values() if concepts])
        integration_bonus = (active_levels / 5) * 0.1  # Up to 10% bonus
        
        return min(1.0, base_consciousness + integration_bonus)

async def main():
    """Main demonstration function"""
    # Test content with rich neuroscience and AI concepts
    test_content = """
    # Neural Networks and Synaptic Plasticity Research

    ## Introduction

    Artificial neural networks are computational models inspired by biological neural networks. 
    These systems consist of interconnected processing units called neurons that communicate 
    through weighted connections analogous to synapses in biological systems.

    Synaptic plasticity represents the ability of synapses between neurons to strengthen or 
    weaken over time, in response to increases or decreases in their activity. This fundamental 
    mechanism underlies learning and memory formation in biological neural networks.

    ## Mechanisms and Analysis

    Long-term potentiation (LTP) is a persistent strengthening of synapses based on recent 
    patterns of activity. When synaptic connections are repeatedly activated, they undergo 
    structural and functional changes that enhance signal transmission. This process involves 
    NMDA receptor activation, calcium influx, and subsequent protein synthesis.

    Modern deep learning architectures implement plasticity-like mechanisms through 
    backpropagation algorithms. During training, connection weights between artificial neurons 
    are adjusted based on error signals propagated backward through the network. This process 
    mirrors biological learning principles while enabling efficient optimization of network parameters.

    ## Cross-Domain Implications

    Therefore, understanding synaptic plasticity mechanisms informs the development of more 
    biologically plausible artificial intelligence systems. The parallels between biological 
    and artificial learning continue to drive innovation in neural network architectures.

    This research demonstrates how interdisciplinary approaches combining neuroscience and 
    artificial intelligence can lead to advances in both fields. The bidirectional exchange 
    of concepts enhances our understanding of both biological and artificial learning systems.
    """
    
    demo = FiveLevelConceptDemo()
    result = await demo.demonstrate_five_level_extraction(test_content)
    
    print(f"\n🎉 Five-Level Concept Extraction Demonstration Complete!")
    print(f"   Overall Success: {result['success']}")
    print(f"   Consciousness Level: {result['consciousness_level']:.3f}")
    print(f"   Processing Efficiency: {result['total_concepts'] / result['processing_time']:.1f} concepts/second")

if __name__ == "__main__":
    asyncio.run(main())