"""
Five-Level Concept Extraction System
===================================

Integrates all Phase 1 components into a comprehensive concept extraction pipeline:
- Level 1: Atomic concept extraction using enhanced ThoughtSeed domains
- Level 2: Relationship mapping with consciousness competition
- Level 3: Composite concept assembly through cross-domain integration
- Level 4: Contextual framework detection with domain specialization
- Level 5: Narrative structure analysis with holistic understanding

Uses Ollama for processing, granular chunking for preparation, and enhanced
ThoughtSeed domains for consciousness-guided extraction.

Implements Spec-022 Task 2.1 requirements.
"""

import asyncio
import logging
import time
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
import json
import uuid

# Import our completed Phase 1 components
try:
    from .ollama_integration import OllamaModelManager, OllamaConceptExtractor, ConceptExtractionRequest
    from .document_ingestion import DocumentIngestionService, DocumentContent
    from .granular_chunking import UltraGranularChunker, ChunkingLevel, ProcessedChunk
    from .enhanced_thoughtseed_domains import (
        EnhancedThoughtSeedOrchestrator, 
        GranularProcessingDomain,
        GranularThoughtSeed
    )
except ImportError:
    # For standalone testing, import directly
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    
    from ollama_integration import OllamaModelManager, OllamaConceptExtractor, ConceptExtractionRequest
    from document_ingestion import DocumentIngestionService, DocumentContent
    from granular_chunking import UltraGranularChunker, ChunkingLevel, ProcessedChunk
    from enhanced_thoughtseed_domains import (
        EnhancedThoughtSeedOrchestrator, 
        GranularProcessingDomain,
        GranularThoughtSeed
    )

logger = logging.getLogger(__name__)

class ConceptExtractionLevel(Enum):
    """Five levels of concept extraction"""
    ATOMIC = 1          # Individual terms, entities, definitions
    RELATIONAL = 2      # Connections, causality, dependencies  
    COMPOSITE = 3       # Complex multi-part ideas and systems
    CONTEXTUAL = 4      # Domain paradigms, theoretical frameworks
    NARRATIVE = 5       # Argument flows, story progressions, methodologies

class ExtractionComplexity(Enum):
    """Complexity levels for processing optimization"""
    SIMPLE = "simple"           # Basic entity extraction
    MODERATE = "moderate"       # Relationship detection
    COMPLEX = "complex"         # Composite concept assembly
    ADVANCED = "advanced"       # Contextual framework analysis
    HOLISTIC = "holistic"       # Narrative structure analysis

@dataclass
class ExtractedConcept:
    """A concept extracted at any level"""
    concept_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    level: ConceptExtractionLevel = ConceptExtractionLevel.ATOMIC
    content: str = ""
    concept_type: str = "general"
    confidence: float = 0.0
    
    # Metadata and context
    source_chunk_id: str = ""
    extraction_method: str = ""
    domain_tags: List[str] = field(default_factory=list)
    position_info: Dict[str, Any] = field(default_factory=dict)
    
    # Relationships to other concepts
    related_concepts: List[str] = field(default_factory=list)
    parent_concepts: List[str] = field(default_factory=list)
    child_concepts: List[str] = field(default_factory=list)
    
    # Processing metadata
    extraction_time: datetime = field(default_factory=datetime.now)
    processing_duration: float = 0.0
    consciousness_score: float = 0.0
    
    # Specialized fields by level
    atomic_properties: Dict[str, Any] = field(default_factory=dict)
    relationship_data: Dict[str, Any] = field(default_factory=dict)
    composite_structure: Dict[str, Any] = field(default_factory=dict)
    contextual_framework: Dict[str, Any] = field(default_factory=dict)
    narrative_elements: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LevelExtractionResult:
    """Results from processing at a specific level"""
    level: ConceptExtractionLevel
    concepts: List[ExtractedConcept] = field(default_factory=list)
    processing_time: float = 0.0
    consciousness_emergence: float = 0.0
    success_rate: float = 0.0
    
    # Level-specific metrics
    total_chunks_processed: int = 0
    successful_extractions: int = 0
    failed_extractions: int = 0
    
    # Quality metrics
    average_confidence: float = 0.0
    concept_diversity: int = 0
    cross_level_connections: int = 0

@dataclass
class FiveLevelExtractionResult:
    """Complete five-level extraction results"""
    success: bool = True
    document_id: str = ""
    total_processing_time: float = 0.0
    
    # Results by level
    level_results: Dict[ConceptExtractionLevel, LevelExtractionResult] = field(default_factory=dict)
    
    # Integrated results
    all_concepts: List[ExtractedConcept] = field(default_factory=list)
    concept_hierarchy: Dict[str, List[str]] = field(default_factory=dict)
    cross_level_relationships: List[Dict[str, Any]] = field(default_factory=list)
    
    # Consciousness metrics
    overall_consciousness_level: float = 0.0
    emergence_indicators: List[str] = field(default_factory=list)
    meta_cognitive_insights: List[Dict[str, Any]] = field(default_factory=list)
    
    # Performance metrics
    processing_stats: Dict[str, Any] = field(default_factory=dict)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Error handling
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

class AtomicConceptExtractor:
    """Level 1: Atomic concept extraction"""
    
    def __init__(self, ollama_extractor: OllamaConceptExtractor):
        self.ollama_extractor = ollama_extractor
        self.level = ConceptExtractionLevel.ATOMIC
        self.complexity = ExtractionComplexity.SIMPLE
    
    async def extract_concepts(self, chunks: List[ProcessedChunk], context: Dict[str, Any] = None) -> LevelExtractionResult:
        """Extract atomic concepts from chunks"""
        start_time = time.time()
        concepts = []
        successful = 0
        failed = 0
        
        logger.info(f"Starting atomic concept extraction from {len(chunks)} chunks")
        
        for chunk in chunks:
            if chunk.metadata.level == ChunkingLevel.SENTENCE:  # Process sentence-level chunks
                try:
                    # Create extraction request
                    request = ConceptExtractionRequest(
                        text=chunk.content,
                        granularity_level=1,
                        domain_focus=context.get("domain_focus", ["neuroscience", "ai"]) if context else ["neuroscience", "ai"],
                        extraction_type="atomic_concepts"
                    )
                    
                    # Extract using Ollama
                    result = await self.ollama_extractor.extract_atomic_concepts(request)
                    
                    if result.success:
                        # Convert Ollama results to our concept format
                        for concept_data in result.concepts:
                            concept = ExtractedConcept(
                                level=self.level,
                                content=concept_data.get("name", ""),
                                concept_type=concept_data.get("type", "entity"),
                                confidence=concept_data.get("confidence", 0.0),
                                source_chunk_id=chunk.metadata.chunk_id,
                                extraction_method="ollama_atomic",
                                domain_tags=concept_data.get("domains", []),
                                position_info={
                                    "chunk_position": concept_data.get("position", 0),
                                    "chunk_level": chunk.metadata.level.value
                                },
                                processing_duration=result.processing_time,
                                consciousness_score=concept_data.get("consciousness_score", 0.0),
                                atomic_properties={
                                    "definition": concept_data.get("definition", ""),
                                    "frequency": concept_data.get("frequency", 1),
                                    "context": concept_data.get("context", ""),
                                    "semantic_category": concept_data.get("category", "")
                                }
                            )
                            concepts.append(concept)
                        
                        successful += 1
                    else:
                        failed += 1
                        logger.warning(f"Atomic extraction failed for chunk {chunk.metadata.chunk_id}: {result.error}")
                
                except Exception as e:
                    failed += 1
                    logger.error(f"Error extracting atomic concepts from chunk {chunk.metadata.chunk_id}: {e}")
        
        processing_time = time.time() - start_time
        
        # Calculate metrics
        avg_confidence = sum(c.confidence for c in concepts) / len(concepts) if concepts else 0.0
        concept_types = set(c.concept_type for c in concepts)
        
        result = LevelExtractionResult(
            level=self.level,
            concepts=concepts,
            processing_time=processing_time,
            consciousness_emergence=avg_confidence,  # Simple consciousness metric
            success_rate=successful / (successful + failed) if (successful + failed) > 0 else 0.0,
            total_chunks_processed=len(chunks),
            successful_extractions=successful,
            failed_extractions=failed,
            average_confidence=avg_confidence,
            concept_diversity=len(concept_types)
        )
        
        logger.info(f"Atomic extraction completed: {len(concepts)} concepts from {successful}/{len(chunks)} chunks")
        return result

class RelationalConceptExtractor:
    """Level 2: Relationship mapping between concepts"""
    
    def __init__(self, ollama_extractor: OllamaConceptExtractor):
        self.ollama_extractor = ollama_extractor
        self.level = ConceptExtractionLevel.RELATIONAL
        self.complexity = ExtractionComplexity.MODERATE
    
    async def extract_concepts(self, chunks: List[ProcessedChunk], atomic_concepts: List[ExtractedConcept], context: Dict[str, Any] = None) -> LevelExtractionResult:
        """Extract relational concepts from chunks with atomic concept context"""
        start_time = time.time()
        relationships = []
        successful = 0
        failed = 0
        
        logger.info(f"Starting relational concept extraction from {len(chunks)} chunks with {len(atomic_concepts)} atomic concepts")
        
        # Process relationship-level chunks
        relationship_chunks = [c for c in chunks if c.metadata.level == ChunkingLevel.RELATIONSHIP]
        
        for chunk in relationship_chunks:
            try:
                # Find atomic concepts in this chunk
                chunk_atomics = [ac for ac in atomic_concepts if ac.source_chunk_id == chunk.metadata.chunk_id]
                
                # Create extraction request with atomic context
                request = ConceptExtractionRequest(
                    text=chunk.content,
                    granularity_level=2,
                    domain_focus=context.get("domain_focus", ["neuroscience", "ai"]) if context else ["neuroscience", "ai"],
                    extraction_type="relationships",
                    context={
                        "atomic_concepts": [{"name": ac.content, "type": ac.concept_type} for ac in chunk_atomics]
                    }
                )
                
                # Extract using Ollama
                result = await self.ollama_extractor.extract_relationships(request)
                
                if result.success:
                    # Convert to relationship concepts
                    for rel_data in result.relationships:
                        relationship = ExtractedConcept(
                            level=self.level,
                            content=f"{rel_data.get('source_concept', '')} {rel_data.get('relationship_type', '')} {rel_data.get('target_concept', '')}",
                            concept_type=rel_data.get("relationship_type", "relation"),
                            confidence=rel_data.get("confidence", 0.0),
                            source_chunk_id=chunk.metadata.chunk_id,
                            extraction_method="ollama_relational",
                            domain_tags=rel_data.get("domains", []),
                            related_concepts=[ac.concept_id for ac in chunk_atomics],
                            processing_duration=result.processing_time,
                            consciousness_score=rel_data.get("consciousness_score", 0.0),
                            relationship_data={
                                "source_concept": rel_data.get("source_concept", ""),
                                "target_concept": rel_data.get("target_concept", ""),
                                "relationship_type": rel_data.get("relationship_type", ""),
                                "directionality": rel_data.get("directionality", "bidirectional"),
                                "strength": rel_data.get("strength", 0.5),
                                "evidence": rel_data.get("evidence", "")
                            }
                        )
                        relationships.append(relationship)
                    
                    successful += 1
                else:
                    failed += 1
                    logger.warning(f"Relational extraction failed for chunk {chunk.metadata.chunk_id}: {result.error}")
            
            except Exception as e:
                failed += 1
                logger.error(f"Error extracting relationships from chunk {chunk.metadata.chunk_id}: {e}")
        
        processing_time = time.time() - start_time
        
        # Calculate cross-level connections
        cross_connections = sum(len(r.related_concepts) for r in relationships)
        avg_confidence = sum(r.confidence for r in relationships) / len(relationships) if relationships else 0.0
        
        result = LevelExtractionResult(
            level=self.level,
            concepts=relationships,
            processing_time=processing_time,
            consciousness_emergence=avg_confidence * 1.1,  # Slightly higher for relationship complexity
            success_rate=successful / (successful + failed) if (successful + failed) > 0 else 0.0,
            total_chunks_processed=len(relationship_chunks),
            successful_extractions=successful,
            failed_extractions=failed,
            average_confidence=avg_confidence,
            concept_diversity=len(set(r.concept_type for r in relationships)),
            cross_level_connections=cross_connections
        )
        
        logger.info(f"Relational extraction completed: {len(relationships)} relationships from {successful}/{len(relationship_chunks)} chunks")
        return result

class CompositeConceptExtractor:
    """Level 3: Composite concept assembly"""
    
    def __init__(self, thoughtseed_orchestrator: EnhancedThoughtSeedOrchestrator):
        self.thoughtseed_orchestrator = thoughtseed_orchestrator
        self.level = ConceptExtractionLevel.COMPOSITE
        self.complexity = ExtractionComplexity.COMPLEX
    
    async def extract_concepts(self, chunks: List[ProcessedChunk], 
                             atomic_concepts: List[ExtractedConcept],
                             relational_concepts: List[ExtractedConcept],
                             context: Dict[str, Any] = None) -> LevelExtractionResult:
        """Extract composite concepts by assembling atomic and relational components"""
        start_time = time.time()
        composites = []
        successful = 0
        failed = 0
        
        logger.info(f"Starting composite concept assembly from {len(atomic_concepts)} atomics and {len(relational_concepts)} relationships")
        
        # Process paragraph-level chunks for composite concepts
        composite_chunks = [c for c in chunks if c.metadata.level == ChunkingLevel.PARAGRAPH]
        
        for chunk in composite_chunks:
            try:
                # Find related atomic and relational concepts for this chunk
                chunk_atomics = [ac for ac in atomic_concepts if ac.source_chunk_id == chunk.metadata.chunk_id]
                chunk_relations = [rc for rc in relational_concepts if rc.source_chunk_id == chunk.metadata.chunk_id]
                
                # Use ThoughtSeed orchestrator for composite assembly
                chunk_data = {
                    "content": chunk.content,
                    "metadata": {
                        "chunk_id": chunk.metadata.chunk_id,
                        "chunk_type": chunk.metadata.chunk_type.value,
                        "granularity_level": 3
                    }
                }
                
                # Process through composite assembly domain
                thoughtseed_result = await self.thoughtseed_orchestrator.process_granular_chunks(
                    [chunk_data],
                    domain_focus=["composite_assembly"]
                )
                
                if thoughtseed_result["success"]:
                    chunk_results = thoughtseed_result["chunk_results"].get(chunk.metadata.chunk_id, {})
                    assembly_results = chunk_results.get("composite_assembly", {})
                    
                    for pattern in assembly_results.get("extracted_patterns", []):
                        composite = ExtractedConcept(
                            level=self.level,
                            content=pattern.get("type", "unknown_composite"),
                            concept_type="composite_concept",
                            confidence=assembly_results.get("consciousness_contribution", 0.0),
                            source_chunk_id=chunk.metadata.chunk_id,
                            extraction_method="thoughtseed_assembly",
                            domain_tags=pattern.get("domains", []),
                            related_concepts=[ac.concept_id for ac in chunk_atomics] + [rc.concept_id for rc in chunk_relations],
                            parent_concepts=[ac.concept_id for ac in chunk_atomics],  # Atomics are parents
                            consciousness_score=assembly_results.get("consciousness_contribution", 0.0),
                            composite_structure={
                                "components": pattern.get("components", []),
                                "assembly_strategy": pattern.get("assembly_strategy", ""),
                                "complexity_level": len(chunk_atomics) + len(chunk_relations),
                                "semantic_coherence": pattern.get("confidence", 0.0),
                                "functional_unity": self._assess_functional_unity(chunk_atomics, chunk_relations)
                            }
                        )
                        composites.append(composite)
                    
                    successful += 1
                else:
                    failed += 1
                    logger.warning(f"Composite assembly failed for chunk {chunk.metadata.chunk_id}")
            
            except Exception as e:
                failed += 1
                logger.error(f"Error assembling composite concepts from chunk {chunk.metadata.chunk_id}: {e}")
        
        processing_time = time.time() - start_time
        
        # Calculate metrics
        avg_confidence = sum(c.confidence for c in composites) / len(composites) if composites else 0.0
        total_connections = sum(len(c.related_concepts) for c in composites)
        
        result = LevelExtractionResult(
            level=self.level,
            concepts=composites,
            processing_time=processing_time,
            consciousness_emergence=avg_confidence * 1.2,  # Higher for composite complexity
            success_rate=successful / (successful + failed) if (successful + failed) > 0 else 0.0,
            total_chunks_processed=len(composite_chunks),
            successful_extractions=successful,
            failed_extractions=failed,
            average_confidence=avg_confidence,
            concept_diversity=len(set(c.content for c in composites)),
            cross_level_connections=total_connections
        )
        
        logger.info(f"Composite assembly completed: {len(composites)} composite concepts from {successful}/{len(composite_chunks)} chunks")
        return result
    
    def _assess_functional_unity(self, atomics: List[ExtractedConcept], relations: List[ExtractedConcept]) -> float:
        """Assess how well atomic concepts form a functional unit"""
        if not atomics:
            return 0.0
        
        # Simple heuristic: more relationships between atomics = higher functional unity
        total_relations = len(relations)
        possible_relations = len(atomics) * (len(atomics) - 1) / 2  # Max possible pairwise relations
        
        if possible_relations == 0:
            return 1.0 if len(atomics) == 1 else 0.0
        
        return min(1.0, total_relations / possible_relations)

class ContextualFrameworkExtractor:
    """Level 4: Contextual framework detection"""
    
    def __init__(self, thoughtseed_orchestrator: EnhancedThoughtSeedOrchestrator):
        self.thoughtseed_orchestrator = thoughtseed_orchestrator
        self.level = ConceptExtractionLevel.CONTEXTUAL
        self.complexity = ExtractionComplexity.ADVANCED
    
    async def extract_concepts(self, chunks: List[ProcessedChunk],
                             lower_level_concepts: List[ExtractedConcept],
                             context: Dict[str, Any] = None) -> LevelExtractionResult:
        """Extract contextual frameworks from section-level content"""
        start_time = time.time()
        frameworks = []
        successful = 0
        failed = 0
        
        logger.info(f"Starting contextual framework extraction with {len(lower_level_concepts)} lower-level concepts")
        
        # Process section-level chunks for contextual frameworks
        contextual_chunks = [c for c in chunks if c.metadata.level == ChunkingLevel.SECTION]
        
        # If no section chunks, use paragraph chunks
        if not contextual_chunks:
            contextual_chunks = [c for c in chunks if c.metadata.level == ChunkingLevel.PARAGRAPH]
        
        for chunk in contextual_chunks:
            try:
                # Analyze domain-specific frameworks
                frameworks_found = await self._detect_domain_frameworks(chunk, lower_level_concepts, context)
                
                for framework_data in frameworks_found:
                    framework = ExtractedConcept(
                        level=self.level,
                        content=framework_data["name"],
                        concept_type="contextual_framework",
                        confidence=framework_data["confidence"],
                        source_chunk_id=chunk.metadata.chunk_id,
                        extraction_method="domain_framework_detection",
                        domain_tags=framework_data.get("domains", []),
                        related_concepts=[c.concept_id for c in lower_level_concepts if c.source_chunk_id == chunk.metadata.chunk_id],
                        consciousness_score=framework_data["confidence"] * 1.3,  # High consciousness for frameworks
                        contextual_framework={
                            "framework_type": framework_data["type"],
                            "theoretical_basis": framework_data.get("theory", ""),
                            "domain_scope": framework_data.get("domains", []),
                            "key_principles": framework_data.get("principles", []),
                            "supporting_concepts": framework_data.get("supporting_concepts", []),
                            "paradigm_level": framework_data.get("paradigm_level", "domain_specific")
                        }
                    )
                    frameworks.append(framework)
                
                if frameworks_found:
                    successful += 1
                else:
                    failed += 1
            
            except Exception as e:
                failed += 1
                logger.error(f"Error extracting contextual frameworks from chunk {chunk.metadata.chunk_id}: {e}")
        
        processing_time = time.time() - start_time
        
        # Calculate metrics
        avg_confidence = sum(f.confidence for f in frameworks) / len(frameworks) if frameworks else 0.0
        
        result = LevelExtractionResult(
            level=self.level,
            concepts=frameworks,
            processing_time=processing_time,
            consciousness_emergence=avg_confidence * 1.3,  # High consciousness for contextual understanding
            success_rate=successful / (successful + failed) if (successful + failed) > 0 else 0.0,
            total_chunks_processed=len(contextual_chunks),
            successful_extractions=successful,
            failed_extractions=failed,
            average_confidence=avg_confidence,
            concept_diversity=len(set(f.contextual_framework.get("framework_type", "") for f in frameworks)),
            cross_level_connections=sum(len(f.related_concepts) for f in frameworks)
        )
        
        logger.info(f"Contextual framework extraction completed: {len(frameworks)} frameworks from {successful}/{len(contextual_chunks)} chunks")
        return result
    
    async def _detect_domain_frameworks(self, chunk: ProcessedChunk, lower_concepts: List[ExtractedConcept], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect domain-specific theoretical frameworks"""
        content = chunk.content.lower()
        frameworks = []
        
        # Neuroscience frameworks
        if any(term in content for term in ["plasticity", "neural", "synaptic", "neuron"]):
            if "plasticity" in content and any(term in content for term in ["learning", "memory", "adaptation"]):
                frameworks.append({
                    "name": "Synaptic Plasticity Framework",
                    "type": "neuroscience_mechanism",
                    "confidence": 0.8,
                    "domains": ["neuroscience", "learning"],
                    "theory": "Activity-dependent modification of synaptic strength",
                    "principles": ["Hebbian learning", "LTP/LTD", "Homeostatic plasticity"],
                    "paradigm_level": "mechanistic"
                })
            
            if "network" in content and "connectivity" in content:
                frameworks.append({
                    "name": "Neural Network Connectivity Framework",
                    "type": "neuroscience_systems",
                    "confidence": 0.75,
                    "domains": ["neuroscience", "systems_biology"],
                    "theory": "Brain function emerges from network connectivity patterns",
                    "principles": ["Small-world networks", "Hub connectivity", "Modular organization"],
                    "paradigm_level": "systems"
                })
        
        # AI/ML frameworks
        if any(term in content for term in ["algorithm", "learning", "network", "training"]):
            if "backpropagation" in content or "gradient" in content:
                frameworks.append({
                    "name": "Gradient-Based Learning Framework",
                    "type": "ai_learning_paradigm",
                    "confidence": 0.85,
                    "domains": ["artificial_intelligence", "machine_learning"],
                    "theory": "Learning through iterative error minimization",
                    "principles": ["Gradient descent", "Error backpropagation", "Weight optimization"],
                    "paradigm_level": "algorithmic"
                })
            
            if "deep" in content and ("layer" in content or "hierarchical" in content):
                frameworks.append({
                    "name": "Deep Hierarchical Learning Framework",
                    "type": "ai_architecture_paradigm",
                    "confidence": 0.8,
                    "domains": ["deep_learning", "representation_learning"],
                    "theory": "Hierarchical feature learning through deep architectures",
                    "principles": ["Feature hierarchy", "Representation learning", "End-to-end training"],
                    "paradigm_level": "architectural"
                })
        
        # Cross-domain frameworks
        neuroscience_concepts = [c for c in lower_concepts if "neuroscience" in c.domain_tags]
        ai_concepts = [c for c in lower_concepts if "ai" in c.domain_tags or "artificial_intelligence" in c.domain_tags]
        
        if neuroscience_concepts and ai_concepts:
            frameworks.append({
                "name": "Bio-Inspired AI Framework",
                "type": "cross_domain_paradigm",
                "confidence": 0.7,
                "domains": ["neuroscience", "artificial_intelligence", "biomimetics"],
                "theory": "AI systems inspired by biological neural mechanisms",
                "principles": ["Biological plausibility", "Neural inspiration", "Functional mimicry"],
                "supporting_concepts": [c.content for c in neuroscience_concepts[:3]] + [c.content for c in ai_concepts[:3]],
                "paradigm_level": "interdisciplinary"
            })
        
        return frameworks

class NarrativeStructureExtractor:
    """Level 5: Narrative structure analysis"""
    
    def __init__(self, thoughtseed_orchestrator: EnhancedThoughtSeedOrchestrator):
        self.thoughtseed_orchestrator = thoughtseed_orchestrator
        self.level = ConceptExtractionLevel.NARRATIVE
        self.complexity = ExtractionComplexity.HOLISTIC
    
    async def extract_concepts(self, chunks: List[ProcessedChunk],
                             all_lower_concepts: List[ExtractedConcept],
                             context: Dict[str, Any] = None) -> LevelExtractionResult:
        """Extract narrative structures and argument flows"""
        start_time = time.time()
        narratives = []
        successful = 0
        failed = 0
        
        logger.info(f"Starting narrative structure analysis with {len(all_lower_concepts)} concepts across all levels")
        
        # Process narrative-level chunks (largest granularity)
        narrative_chunks = [c for c in chunks if c.metadata.level == ChunkingLevel.NARRATIVE]
        
        # If no narrative chunks, use section chunks
        if not narrative_chunks:
            narrative_chunks = [c for c in chunks if c.metadata.level == ChunkingLevel.SECTION]
        
        for chunk in narrative_chunks:
            try:
                # Use ThoughtSeed orchestrator for narrative flow detection
                chunk_data = {
                    "content": chunk.content,
                    "metadata": {
                        "chunk_id": chunk.metadata.chunk_id,
                        "chunk_type": chunk.metadata.chunk_type.value,
                        "granularity_level": 5
                    }
                }
                
                # Process through narrative flow domain
                thoughtseed_result = await self.thoughtseed_orchestrator.process_granular_chunks(
                    [chunk_data],
                    domain_focus=["narrative_flow"]
                )
                
                if thoughtseed_result["success"]:
                    chunk_results = thoughtseed_result["chunk_results"].get(chunk.metadata.chunk_id, {})
                    narrative_results = chunk_results.get("narrative_flow", {})
                    
                    # Analyze argument structure and methodology flow
                    narrative_analysis = await self._analyze_narrative_structure(chunk, all_lower_concepts)
                    
                    for pattern in narrative_results.get("extracted_patterns", []):
                        narrative = ExtractedConcept(
                            level=self.level,
                            content=f"{pattern.get('type', 'narrative')}: {pattern.get('indicator', '')}",
                            concept_type="narrative_structure",
                            confidence=narrative_results.get("consciousness_contribution", 0.0),
                            source_chunk_id=chunk.metadata.chunk_id,
                            extraction_method="thoughtseed_narrative",
                            domain_tags=["narrative_analysis", "argument_structure"],
                            related_concepts=[c.concept_id for c in all_lower_concepts],
                            consciousness_score=narrative_results.get("consciousness_contribution", 0.0) * 1.5,  # Highest for holistic understanding
                            narrative_elements={
                                "narrative_type": pattern.get("type", ""),
                                "flow_indicators": pattern.get("indicators", []),
                                "argument_structure": narrative_analysis.get("argument_structure", {}),
                                "methodology_flow": narrative_analysis.get("methodology_flow", {}),
                                "evidence_progression": narrative_analysis.get("evidence_progression", {}),
                                "conclusion_strength": narrative_analysis.get("conclusion_strength", 0.0),
                                "coherence_score": self._calculate_narrative_coherence(chunk, all_lower_concepts)
                            }
                        )
                        narratives.append(narrative)
                    
                    successful += 1
                else:
                    failed += 1
                    logger.warning(f"Narrative analysis failed for chunk {chunk.metadata.chunk_id}")
            
            except Exception as e:
                failed += 1
                logger.error(f"Error analyzing narrative structure from chunk {chunk.metadata.chunk_id}: {e}")
        
        processing_time = time.time() - start_time
        
        # Calculate metrics
        avg_confidence = sum(n.confidence for n in narratives) / len(narratives) if narratives else 0.0
        
        result = LevelExtractionResult(
            level=self.level,
            concepts=narratives,
            processing_time=processing_time,
            consciousness_emergence=avg_confidence * 1.5,  # Highest consciousness for narrative understanding
            success_rate=successful / (successful + failed) if (successful + failed) > 0 else 0.0,
            total_chunks_processed=len(narrative_chunks),
            successful_extractions=successful,
            failed_extractions=failed,
            average_confidence=avg_confidence,
            concept_diversity=len(set(n.narrative_elements.get("narrative_type", "") for n in narratives)),
            cross_level_connections=sum(len(n.related_concepts) for n in narratives)
        )
        
        logger.info(f"Narrative structure analysis completed: {len(narratives)} structures from {successful}/{len(narrative_chunks)} chunks")
        return result
    
    async def _analyze_narrative_structure(self, chunk: ProcessedChunk, all_concepts: List[ExtractedConcept]) -> Dict[str, Any]:
        """Analyze detailed narrative and argument structure"""
        content = chunk.content.lower()
        
        analysis = {
            "argument_structure": {},
            "methodology_flow": {},
            "evidence_progression": {},
            "conclusion_strength": 0.0
        }
        
        # Argument structure analysis
        premises = []
        conclusions = []
        
        if any(word in content for word in ["because", "since", "given that", "due to"]):
            premises.append("causal_premises_present")
        if any(word in content for word in ["therefore", "thus", "consequently", "hence"]):
            conclusions.append("logical_conclusions_present")
        if any(word in content for word in ["however", "but", "although", "despite"]):
            premises.append("counterarguments_addressed")
        
        analysis["argument_structure"] = {
            "premises": premises,
            "conclusions": conclusions,
            "argument_strength": len(premises) + len(conclusions)
        }
        
        # Methodology flow analysis
        method_indicators = []
        if any(word in content for word in ["method", "approach", "technique", "procedure"]):
            method_indicators.append("methodology_described")
        if any(word in content for word in ["first", "second", "next", "then", "finally"]):
            method_indicators.append("sequential_steps")
        if any(word in content for word in ["result", "outcome", "finding"]):
            method_indicators.append("results_reported")
        
        analysis["methodology_flow"] = {
            "indicators": method_indicators,
            "flow_strength": len(method_indicators)
        }
        
        # Evidence progression analysis
        evidence_types = []
        if any(word in content for word in ["data", "evidence", "study", "research"]):
            evidence_types.append("empirical_evidence")
        if any(word in content for word in ["theory", "model", "framework"]):
            evidence_types.append("theoretical_support")
        if any(word in content for word in ["example", "case", "instance"]):
            evidence_types.append("illustrative_examples")
        
        analysis["evidence_progression"] = {
            "types": evidence_types,
            "diversity": len(evidence_types)
        }
        
        # Overall conclusion strength
        total_indicators = len(premises) + len(conclusions) + len(method_indicators) + len(evidence_types)
        analysis["conclusion_strength"] = min(1.0, total_indicators * 0.1)
        
        return analysis
    
    def _calculate_narrative_coherence(self, chunk: ProcessedChunk, all_concepts: List[ExtractedConcept]) -> float:
        """Calculate how coherent the narrative is based on concept connections"""
        chunk_concepts = [c for c in all_concepts if c.source_chunk_id == chunk.metadata.chunk_id]
        
        if len(chunk_concepts) < 2:
            return 1.0 if len(chunk_concepts) == 1 else 0.0
        
        # Count cross-level concept connections
        total_connections = 0
        for concept in chunk_concepts:
            total_connections += len(concept.related_concepts) + len(concept.parent_concepts) + len(concept.child_concepts)
        
        # Normalize by possible connections
        possible_connections = len(chunk_concepts) * (len(chunk_concepts) - 1)
        coherence = total_connections / possible_connections if possible_connections > 0 else 0.0
        
        return min(1.0, coherence)

class FiveLevelConceptExtractionService:
    """Main service orchestrating five-level concept extraction"""
    
    def __init__(self):
        # Initialize Phase 1 components
        self.ollama_manager = OllamaModelManager()
        self.ollama_extractor = OllamaConceptExtractor(self.ollama_manager)
        self.document_ingestion = DocumentIngestionService()
        self.granular_chunker = UltraGranularChunker()
        self.thoughtseed_orchestrator = EnhancedThoughtSeedOrchestrator()
        
        # Initialize level extractors
        self.level_extractors = {
            ConceptExtractionLevel.ATOMIC: AtomicConceptExtractor(self.ollama_extractor),
            ConceptExtractionLevel.RELATIONAL: RelationalConceptExtractor(self.ollama_extractor),
            ConceptExtractionLevel.COMPOSITE: CompositeConceptExtractor(self.thoughtseed_orchestrator),
            ConceptExtractionLevel.CONTEXTUAL: ContextualFrameworkExtractor(self.thoughtseed_orchestrator),
            ConceptExtractionLevel.NARRATIVE: NarrativeStructureExtractor(self.thoughtseed_orchestrator)
        }
        
        self.processing_history = []
    
    async def extract_concepts_from_document(self, 
                                           document_source: Union[str, bytes],
                                           source_type: str = "auto",
                                           domain_focus: List[str] = None,
                                           levels: List[ConceptExtractionLevel] = None) -> FiveLevelExtractionResult:
        """
        Complete five-level concept extraction from a document
        
        Args:
            document_source: File path, URL, or content bytes
            source_type: "file", "url", "bytes", or "auto"
            domain_focus: Target domains (default: ["neuroscience", "ai"])
            levels: Extraction levels to process (default: all levels)
            
        Returns:
            Complete five-level extraction results
        """
        start_time = time.time()
        
        if levels is None:
            levels = list(ConceptExtractionLevel)
        
        if domain_focus is None:
            domain_focus = ["neuroscience", "ai"]
        
        logger.info(f"Starting five-level concept extraction for {len(levels)} levels with domains: {domain_focus}")
        
        try:
            # Step 1: Document ingestion
            logger.info("Step 1: Document ingestion")
            ingestion_result = await self.document_ingestion.ingest_document(document_source, source_type)
            
            if not ingestion_result.success:
                return FiveLevelExtractionResult(
                    success=False,
                    errors=[f"Document ingestion failed: {ingestion_result.error}"]
                )
            
            document_content = ingestion_result.content.raw_text
            document_id = ingestion_result.document_id
            
            # Step 2: Ultra-granular chunking
            logger.info("Step 2: Ultra-granular chunking")
            chunking_results = await self.granular_chunker.chunk_document(
                document_content,
                levels=[ChunkingLevel.SENTENCE, ChunkingLevel.RELATIONSHIP, ChunkingLevel.PARAGRAPH, 
                       ChunkingLevel.SECTION, ChunkingLevel.NARRATIVE]
            )
            
            # Collect all chunks for processing
            all_chunks = []
            for chunking_level, result in chunking_results.items():
                if result.success:
                    all_chunks.extend(result.chunks)
            
            if not all_chunks:
                return FiveLevelExtractionResult(
                    success=False,
                    document_id=document_id,
                    errors=["No chunks generated from document"]
                )
            
            logger.info(f"Generated {len(all_chunks)} chunks across {len(chunking_results)} levels")
            
            # Step 3: Sequential five-level concept extraction
            level_results = {}
            all_extracted_concepts = []
            
            context = {
                "domain_focus": domain_focus,
                "document_metadata": ingestion_result.content.metadata.__dict__,
                "chunking_stats": {level.value: result.level_stats for level, result in chunking_results.items()}
            }
            
            # Level 1: Atomic concepts (foundation)
            if ConceptExtractionLevel.ATOMIC in levels:
                logger.info("Step 3.1: Atomic concept extraction")
                atomic_result = await self.level_extractors[ConceptExtractionLevel.ATOMIC].extract_concepts(all_chunks, context)
                level_results[ConceptExtractionLevel.ATOMIC] = atomic_result
                all_extracted_concepts.extend(atomic_result.concepts)
                logger.info(f"Extracted {len(atomic_result.concepts)} atomic concepts")
            
            # Level 2: Relational concepts (building on atomics)
            if ConceptExtractionLevel.RELATIONAL in levels:
                logger.info("Step 3.2: Relational concept extraction")
                atomic_concepts = level_results.get(ConceptExtractionLevel.ATOMIC, LevelExtractionResult(ConceptExtractionLevel.ATOMIC)).concepts
                relational_result = await self.level_extractors[ConceptExtractionLevel.RELATIONAL].extract_concepts(
                    all_chunks, atomic_concepts, context
                )
                level_results[ConceptExtractionLevel.RELATIONAL] = relational_result
                all_extracted_concepts.extend(relational_result.concepts)
                logger.info(f"Extracted {len(relational_result.concepts)} relational concepts")
            
            # Level 3: Composite concepts (assembling atomics and relations)
            if ConceptExtractionLevel.COMPOSITE in levels:
                logger.info("Step 3.3: Composite concept assembly")
                atomic_concepts = level_results.get(ConceptExtractionLevel.ATOMIC, LevelExtractionResult(ConceptExtractionLevel.ATOMIC)).concepts
                relational_concepts = level_results.get(ConceptExtractionLevel.RELATIONAL, LevelExtractionResult(ConceptExtractionLevel.RELATIONAL)).concepts
                composite_result = await self.level_extractors[ConceptExtractionLevel.COMPOSITE].extract_concepts(
                    all_chunks, atomic_concepts, relational_concepts, context
                )
                level_results[ConceptExtractionLevel.COMPOSITE] = composite_result
                all_extracted_concepts.extend(composite_result.concepts)
                logger.info(f"Assembled {len(composite_result.concepts)} composite concepts")
            
            # Level 4: Contextual frameworks (higher-order understanding)
            if ConceptExtractionLevel.CONTEXTUAL in levels:
                logger.info("Step 3.4: Contextual framework detection")
                lower_concepts = [c for c in all_extracted_concepts if c.level != ConceptExtractionLevel.CONTEXTUAL]
                contextual_result = await self.level_extractors[ConceptExtractionLevel.CONTEXTUAL].extract_concepts(
                    all_chunks, lower_concepts, context
                )
                level_results[ConceptExtractionLevel.CONTEXTUAL] = contextual_result
                all_extracted_concepts.extend(contextual_result.concepts)
                logger.info(f"Detected {len(contextual_result.concepts)} contextual frameworks")
            
            # Level 5: Narrative structures (holistic understanding)
            if ConceptExtractionLevel.NARRATIVE in levels:
                logger.info("Step 3.5: Narrative structure analysis")
                lower_concepts = [c for c in all_extracted_concepts if c.level != ConceptExtractionLevel.NARRATIVE]
                narrative_result = await self.level_extractors[ConceptExtractionLevel.NARRATIVE].extract_concepts(
                    all_chunks, lower_concepts, context
                )
                level_results[ConceptExtractionLevel.NARRATIVE] = narrative_result
                all_extracted_concepts.extend(narrative_result.concepts)
                logger.info(f"Analyzed {len(narrative_result.concepts)} narrative structures")
            
            # Step 4: Build concept hierarchy and cross-level relationships
            logger.info("Step 4: Building concept hierarchy")
            concept_hierarchy = self._build_concept_hierarchy(all_extracted_concepts)
            cross_level_relationships = self._build_cross_level_relationships(all_extracted_concepts)
            
            # Step 5: Calculate consciousness metrics
            logger.info("Step 5: Calculating consciousness emergence")
            consciousness_metrics = self._calculate_consciousness_metrics(level_results, all_extracted_concepts)
            
            processing_time = time.time() - start_time
            
            # Build final result
            result = FiveLevelExtractionResult(
                success=True,
                document_id=document_id,
                total_processing_time=processing_time,
                level_results=level_results,
                all_concepts=all_extracted_concepts,
                concept_hierarchy=concept_hierarchy,
                cross_level_relationships=cross_level_relationships,
                overall_consciousness_level=consciousness_metrics["overall_level"],
                emergence_indicators=consciousness_metrics["emergence_indicators"],
                meta_cognitive_insights=consciousness_metrics["meta_insights"],
                processing_stats={
                    "total_chunks": len(all_chunks),
                    "total_concepts": len(all_extracted_concepts),
                    "concepts_by_level": {level.value: len(result.concepts) for level, result in level_results.items()},
                    "average_confidence": sum(c.confidence for c in all_extracted_concepts) / len(all_extracted_concepts) if all_extracted_concepts else 0.0
                },
                quality_metrics={
                    "cross_level_connectivity": len(cross_level_relationships) / len(all_extracted_concepts) if all_extracted_concepts else 0.0,
                    "concept_diversity": len(set(c.concept_type for c in all_extracted_concepts)),
                    "processing_efficiency": len(all_extracted_concepts) / processing_time if processing_time > 0 else 0.0
                }
            )
            
            logger.info(f"Five-level extraction completed successfully in {processing_time:.2f}s")
            logger.info(f"Total concepts extracted: {len(all_extracted_concepts)}")
            logger.info(f"Consciousness emergence level: {consciousness_metrics['overall_level']:.3f}")
            
            return result
        
        except Exception as e:
            logger.error(f"Five-level concept extraction failed: {e}")
            return FiveLevelExtractionResult(
                success=False,
                document_id=document_id if 'document_id' in locals() else "unknown",
                total_processing_time=time.time() - start_time,
                errors=[str(e)]
            )
    
    def _build_concept_hierarchy(self, concepts: List[ExtractedConcept]) -> Dict[str, List[str]]:
        """Build hierarchical relationships between concepts"""
        hierarchy = {}
        
        for concept in concepts:
            hierarchy[concept.concept_id] = {
                "level": concept.level.value,
                "parents": concept.parent_concepts,
                "children": concept.child_concepts,
                "related": concept.related_concepts
            }
        
        return hierarchy
    
    def _build_cross_level_relationships(self, concepts: List[ExtractedConcept]) -> List[Dict[str, Any]]:
        """Build relationships that span across extraction levels"""
        relationships = []
        
        # Group concepts by level
        concepts_by_level = {}
        for concept in concepts:
            level = concept.level
            if level not in concepts_by_level:
                concepts_by_level[level] = []
            concepts_by_level[level].append(concept)
        
        # Find cross-level connections
        for lower_level in [ConceptExtractionLevel.ATOMIC, ConceptExtractionLevel.RELATIONAL, ConceptExtractionLevel.COMPOSITE]:
            for higher_level in [ConceptExtractionLevel.COMPOSITE, ConceptExtractionLevel.CONTEXTUAL, ConceptExtractionLevel.NARRATIVE]:
                if lower_level.value >= higher_level.value:
                    continue
                
                lower_concepts = concepts_by_level.get(lower_level, [])
                higher_concepts = concepts_by_level.get(higher_level, [])
                
                for lower_concept in lower_concepts:
                    for higher_concept in higher_concepts:
                        # Check if there's a relationship
                        if (lower_concept.concept_id in higher_concept.related_concepts or
                            lower_concept.concept_id in higher_concept.parent_concepts or
                            higher_concept.concept_id in lower_concept.child_concepts):
                            
                            relationships.append({
                                "lower_concept": lower_concept.concept_id,
                                "higher_concept": higher_concept.concept_id,
                                "lower_level": lower_level.value,
                                "higher_level": higher_level.value,
                                "relationship_type": "hierarchical_composition",
                                "strength": min(lower_concept.confidence, higher_concept.confidence)
                            })
        
        return relationships
    
    def _calculate_consciousness_metrics(self, level_results: Dict[ConceptExtractionLevel, LevelExtractionResult], 
                                       all_concepts: List[ExtractedConcept]) -> Dict[str, Any]:
        """Calculate overall consciousness emergence metrics"""
        
        # Weight consciousness contributions by level complexity
        level_weights = {
            ConceptExtractionLevel.ATOMIC: 0.1,
            ConceptExtractionLevel.RELATIONAL: 0.2,
            ConceptExtractionLevel.COMPOSITE: 0.25,
            ConceptExtractionLevel.CONTEXTUAL: 0.25,
            ConceptExtractionLevel.NARRATIVE: 0.2
        }
        
        total_weighted_consciousness = 0.0
        total_weight = 0.0
        
        for level, result in level_results.items():
            weight = level_weights.get(level, 0.1)
            total_weighted_consciousness += result.consciousness_emergence * weight
            total_weight += weight
        
        overall_level = total_weighted_consciousness / total_weight if total_weight > 0 else 0.0
        
        # Emergence indicators
        emergence_indicators = []
        if len(level_results) >= 4:
            emergence_indicators.append("multi_level_integration")
        if overall_level > 0.7:
            emergence_indicators.append("high_consciousness_emergence")
        if any(len(result.concepts) > 0 for result in level_results.values()):
            emergence_indicators.append("concept_extraction_success")
        
        # Meta-cognitive insights
        meta_insights = []
        if ConceptExtractionLevel.NARRATIVE in level_results:
            narrative_concepts = level_results[ConceptExtractionLevel.NARRATIVE].concepts
            if narrative_concepts:
                meta_insights.append({
                    "type": "narrative_awareness",
                    "description": f"System demonstrates narrative understanding with {len(narrative_concepts)} structures",
                    "confidence": sum(c.confidence for c in narrative_concepts) / len(narrative_concepts)
                })
        
        if ConceptExtractionLevel.CONTEXTUAL in level_results:
            contextual_concepts = level_results[ConceptExtractionLevel.CONTEXTUAL].concepts
            if contextual_concepts:
                meta_insights.append({
                    "type": "framework_recognition",
                    "description": f"System recognizes {len(contextual_concepts)} theoretical frameworks",
                    "confidence": sum(c.confidence for c in contextual_concepts) / len(contextual_concepts)
                })
        
        return {
            "overall_level": overall_level,
            "emergence_indicators": emergence_indicators,
            "meta_insights": meta_insights,
            "level_contributions": {level.value: result.consciousness_emergence for level, result in level_results.items()}
        }

# Global service instance
five_level_extractor = FiveLevelConceptExtractionService()

# Test function
async def test_five_level_extraction():
    """Test the complete five-level concept extraction system"""
    print("🧪 Testing Five-Level Concept Extraction System")
    print("=" * 60)
    
    # Test with neuroscience/AI content
    test_content = """# Neural Networks and Synaptic Plasticity

## Introduction

Artificial neural networks are computational models inspired by biological neural networks. These systems consist of interconnected processing units called neurons that communicate through weighted connections analogous to synapses in biological systems.

Synaptic plasticity represents the ability of synapses between neurons to strengthen or weaken over time, in response to increases or decreases in their activity. This fundamental mechanism underlies learning and memory formation in biological neural networks.

## Mechanisms of Plasticity

### Long-term Potentiation

Long-term potentiation (LTP) is a persistent strengthening of synapses based on recent patterns of activity. When synaptic connections are repeatedly activated, they undergo structural and functional changes that enhance signal transmission. This process involves NMDA receptor activation, calcium influx, and subsequent protein synthesis.

The molecular cascades triggered by LTP include activation of protein kinases, CREB-mediated gene expression, synthesis of new proteins, and structural modifications at synaptic terminals. These changes result in increased synaptic efficacy that can persist for hours to days.

### Artificial Neural Network Learning

Modern deep learning architectures implement plasticity-like mechanisms through backpropagation algorithms. During training, connection weights between artificial neurons are adjusted based on error signals propagated backward through the network. This process mirrors biological learning principles while enabling efficient optimization of network parameters.

The gradient descent optimization used in artificial networks bears conceptual similarity to biological plasticity mechanisms. Both systems adjust connection strengths to minimize prediction errors and improve performance on specific tasks.

## Cross-Domain Implications

Therefore, understanding synaptic plasticity mechanisms informs the development of more biologically plausible artificial intelligence systems. The parallels between biological and artificial learning continue to drive innovation in neural network architectures and training algorithms.

This research demonstrates how interdisciplinary approaches combining neuroscience and artificial intelligence can lead to advances in both fields. The bidirectional exchange of concepts enhances our understanding of both biological and artificial learning systems.
"""
    
    # Create temporary file for testing
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(test_content)
        test_file = f.name
    
    try:
        extractor = FiveLevelConceptExtractionService()
        
        # Test complete five-level extraction
        print("🔄 Starting complete five-level concept extraction...")
        result = await extractor.extract_concepts_from_document(
            test_file,
            source_type="file",
            domain_focus=["neuroscience", "artificial_intelligence"],
            levels=list(ConceptExtractionLevel)
        )
        
        # Display results
        print(f"\n📊 Extraction Results:")
        print(f"  ✅ Success: {result.success}")
        print(f"  📄 Document ID: {result.document_id}")
        print(f"  ⏱️  Total processing time: {result.total_processing_time:.3f}s")
        print(f"  🧠 Overall consciousness level: {result.overall_consciousness_level:.3f}")
        print(f"  📦 Total concepts extracted: {len(result.all_concepts)}")
        
        # Show results by level
        print(f"\n📈 Results by Level:")
        for level, level_result in result.level_results.items():
            print(f"  Level {level.value} ({level.name}):")
            print(f"    📝 Concepts: {len(level_result.concepts)}")
            print(f"    ⏱️  Processing time: {level_result.processing_time:.3f}s")
            print(f"    🧠 Consciousness emergence: {level_result.consciousness_emergence:.3f}")
            print(f"    📊 Success rate: {level_result.success_rate:.3f}")
            print(f"    🎯 Average confidence: {level_result.average_confidence:.3f}")
            
            # Show sample concepts
            if level_result.concepts:
                print(f"    📝 Sample concepts:")
                for i, concept in enumerate(level_result.concepts[:3]):  # Show first 3
                    print(f"      {i+1}. {concept.content[:60]}... (confidence: {concept.confidence:.2f})")
        
        # Show consciousness metrics
        print(f"\n🧠 Consciousness Analysis:")
        print(f"  🌟 Emergence indicators: {result.emergence_indicators}")
        print(f"  💡 Meta-cognitive insights: {len(result.meta_cognitive_insights)}")
        for insight in result.meta_cognitive_insights:
            print(f"    - {insight['type']}: {insight['description']}")
        
        # Show concept hierarchy
        print(f"\n🏗️  Concept Hierarchy:")
        print(f"  🔗 Cross-level relationships: {len(result.cross_level_relationships)}")
        print(f"  🌐 Concept connectivity: {result.quality_metrics.get('cross_level_connectivity', 0):.3f}")
        print(f"  🎯 Concept diversity: {result.quality_metrics.get('concept_diversity', 0)}")
        
        # Show processing stats
        print(f"\n📊 Processing Statistics:")
        concepts_by_level = result.processing_stats.get("concepts_by_level", {})
        for level_name, count in concepts_by_level.items():
            print(f"  {level_name}: {count} concepts")
        
        print(f"  📈 Processing efficiency: {result.quality_metrics.get('processing_efficiency', 0):.1f} concepts/second")
        
        if result.errors:
            print(f"\n❌ Errors:")
            for error in result.errors:
                print(f"  - {error}")
        
        print("\n🎉 Five-level concept extraction test completed!")
        return result.success
    
    finally:
        # Cleanup
        os.unlink(test_file)

if __name__ == "__main__":
    asyncio.run(test_five_level_extraction())