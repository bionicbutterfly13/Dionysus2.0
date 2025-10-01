"""
Enhanced ThoughtSeed Domains for Ultra-Granular Document Processing
================================================================

Extends the existing ThoughtSeed competition system with specialized domains
for granular concept extraction and analysis:

- Atomic Concept Detection Domain
- Relationship Mapping Domain  
- Composite Concept Assembly Domain
- Contextual Framework Domain
- Narrative Flow Detection Domain

Integrates with existing consciousness pipeline and maintains backward compatibility.
Implements Spec-022 Task 1.4 requirements.
"""

import asyncio
import logging
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
import uuid
import json

# Import base ThoughtSeed structures - simplified for testing
try:
    from models.cognition_base import CognitionDomain, PatternType, CognitionPattern, CognitionBase, ConsciousnessLevel
except ImportError:
    # Create simplified classes for testing
    from enum import Enum
    class CognitionDomain(str, Enum):
        NEUROSCIENCE = "neuroscience"
        ARTIFICIAL_INTELLIGENCE = "artificial_intelligence"
        CONSCIOUSNESS_STUDIES = "consciousness_studies"
    
    class PatternType(str, Enum):
        EMERGENT = "emergent"
        STRUCTURAL = "structural"
    
    class CognitionPattern:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class ConsciousnessLevel:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class CognitionBase:
        def __init__(self):
            self.patterns = []
        
        def add_pattern(self, pattern):
            self.patterns.append(pattern)
        
        def update_consciousness_level(self, level):
            self.consciousness_level = level

logger = logging.getLogger(__name__)

class GranularProcessingDomain(str, Enum):
    """Extended domains for granular document processing"""
    ATOMIC_CONCEPTS = "atomic_concepts"
    RELATIONSHIP_MAPPING = "relationship_mapping"
    COMPOSITE_ASSEMBLY = "composite_assembly"
    CONTEXTUAL_FRAMEWORKS = "contextual_frameworks"
    NARRATIVE_FLOW = "narrative_flow"
    
    # Specialized neuroscience domains
    NEUROSCIENCE_TERMINOLOGY = "neuroscience_terminology"
    AI_DOMAIN_CONCEPTS = "ai_domain_concepts"
    CROSS_DOMAIN_MAPPING = "cross_domain_mapping"
    
    # Processing-specific domains
    DOCUMENT_STRUCTURE = "document_structure"
    SEMANTIC_RELATIONSHIPS = "semantic_relationships"
    KNOWLEDGE_INTEGRATION = "knowledge_integration"

class ChunkProcessingType(str, Enum):
    """Types of chunk processing for ThoughtSeed competition"""
    ATOMIC_EXTRACTION = "atomic_extraction"
    RELATION_DETECTION = "relation_detection"
    CONCEPT_SYNTHESIS = "concept_synthesis"
    CONTEXT_ANALYSIS = "context_analysis"
    FLOW_ANALYSIS = "flow_analysis"
    DOMAIN_SPECIALIZATION = "domain_specialization"

class ConsciousnessCompetitionLevel(str, Enum):
    """Competition levels for consciousness emergence"""
    INTRA_CHUNK = "intra_chunk"          # Within single chunk
    INTER_CHUNK = "inter_chunk"          # Between related chunks
    CROSS_LEVEL = "cross_level"          # Across granularity levels
    HOLISTIC = "holistic"                # Whole document understanding
    META_COGNITIVE = "meta_cognitive"     # Self-aware processing

@dataclass
class GranularThoughtSeed:
    """Enhanced ThoughtSeed for granular document processing"""
    seed_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    processing_domain: GranularProcessingDomain = GranularProcessingDomain.ATOMIC_CONCEPTS
    chunk_type: ChunkProcessingType = ChunkProcessingType.ATOMIC_EXTRACTION
    competition_level: ConsciousnessCompetitionLevel = ConsciousnessCompetitionLevel.INTRA_CHUNK
    
    # Content and processing
    input_content: str = ""
    processing_context: Dict[str, Any] = field(default_factory=dict)
    extracted_patterns: List[Dict[str, Any]] = field(default_factory=list)
    
    # Competition metrics
    activation_strength: float = 0.0
    competition_score: float = 0.0
    consciousness_contribution: float = 0.0
    emergence_indicators: List[str] = field(default_factory=list)
    
    # Relationships and context
    parent_seeds: List[str] = field(default_factory=list)
    child_seeds: List[str] = field(default_factory=list)
    related_chunks: List[str] = field(default_factory=list)
    
    # Processing metadata
    creation_time: datetime = field(default_factory=datetime.now)
    processing_time: float = 0.0
    success_rate: float = 0.0
    
    # Integration with existing system
    attractor_basin_id: Optional[str] = None
    cognition_pattern_id: Optional[str] = None
    consciousness_layer: str = "conceptual"  # Default layer

@dataclass
class DomainCompetitionState:
    """State of competition within a specific domain"""
    domain: GranularProcessingDomain
    active_seeds: List[GranularThoughtSeed] = field(default_factory=list)
    competition_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)
    winner_seed: Optional[str] = None
    confidence_score: float = 0.0
    processing_phase: str = "initialization"
    
    # Consciousness emergence tracking
    emergence_level: float = 0.0
    meta_awareness: float = 0.0
    recursive_depth: int = 0
    
    # Performance metrics
    total_competitions: int = 0
    successful_extractions: int = 0
    average_processing_time: float = 0.0

class AtomicConceptDomain:
    """Domain for atomic concept detection and extraction"""
    
    def __init__(self):
        self.domain_type = GranularProcessingDomain.ATOMIC_CONCEPTS
        self.processing_patterns = {
            "entity_extraction": {"weight": 0.3, "threshold": 0.6},
            "term_definition": {"weight": 0.25, "threshold": 0.5},
            "concept_isolation": {"weight": 0.25, "threshold": 0.55},
            "semantic_tagging": {"weight": 0.2, "threshold": 0.4}
        }
        self.neuroscience_entities = {
            "neurons", "synapses", "neurotransmitters", "dendrites", "axons",
            "plasticity", "potentiation", "depression", "receptors", "channels"
        }
        self.ai_entities = {
            "networks", "algorithms", "backpropagation", "gradients", "weights",
            "activation", "layers", "training", "learning", "optimization"
        }
    
    async def create_competition_seeds(self, chunk_content: str, context: Dict[str, Any] = None) -> List[GranularThoughtSeed]:
        """Create ThoughtSeeds for atomic concept competition"""
        seeds = []
        
        # Create seeds for different atomic extraction approaches
        for pattern_name, config in self.processing_patterns.items():
            seed = GranularThoughtSeed(
                processing_domain=self.domain_type,
                chunk_type=ChunkProcessingType.ATOMIC_EXTRACTION,
                competition_level=ConsciousnessCompetitionLevel.INTRA_CHUNK,
                input_content=chunk_content,
                processing_context={
                    "pattern_type": pattern_name,
                    "weight": config["weight"],
                    "threshold": config["threshold"],
                    "domain_focus": context.get("domain_focus", []) if context else []
                },
                consciousness_layer="sensory"  # Atomic level is sensory processing
            )
            
            # Simulate atomic concept extraction
            await self._extract_atomic_concepts(seed)
            seeds.append(seed)
        
        return seeds
    
    async def _extract_atomic_concepts(self, seed: GranularThoughtSeed):
        """Extract atomic concepts for a seed"""
        content = seed.input_content.lower()
        pattern_type = seed.processing_context.get("pattern_type", "")
        
        extracted_concepts = []
        
        if pattern_type == "entity_extraction":
            # Extract domain-specific entities
            for entity in self.neuroscience_entities:
                if entity in content:
                    extracted_concepts.append({
                        "type": "neuroscience_entity",
                        "value": entity,
                        "confidence": 0.8,
                        "position": content.find(entity)
                    })
            
            for entity in self.ai_entities:
                if entity in content:
                    extracted_concepts.append({
                        "type": "ai_entity", 
                        "value": entity,
                        "confidence": 0.8,
                        "position": content.find(entity)
                    })
        
        elif pattern_type == "term_definition":
            # Extract term-definition patterns
            import re
            def_patterns = [
                r'(\w+)\s+is\s+(.+?)(?:\.|\n)',
                r'(\w+)\s+refers\s+to\s+(.+?)(?:\.|\n)',
                r'(\w+):\s+(.+?)(?:\.|\n)'
            ]
            
            for pattern in def_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    extracted_concepts.append({
                        "type": "definition",
                        "term": match.group(1),
                        "definition": match.group(2).strip(),
                        "confidence": 0.7
                    })
        
        elif pattern_type == "concept_isolation":
            # Isolate important concepts by frequency and context
            words = content.split()
            important_words = [w for w in words if len(w) > 4 and w.isalpha()]
            
            # Simple frequency-based concept extraction
            word_freq = {}
            for word in important_words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            # Extract top concepts
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            for word, freq in sorted_words[:5]:
                extracted_concepts.append({
                    "type": "key_concept",
                    "value": word,
                    "frequency": freq,
                    "confidence": min(0.9, freq * 0.2)
                })
        
        elif pattern_type == "semantic_tagging":
            # Semantic category tagging
            if any(word in content for word in ["learn", "memory", "remember", "forget"]):
                extracted_concepts.append({
                    "type": "semantic_tag",
                    "category": "learning_memory",
                    "confidence": 0.6
                })
            
            if any(word in content for word in ["network", "connect", "link", "associate"]):
                extracted_concepts.append({
                    "type": "semantic_tag", 
                    "category": "connectivity",
                    "confidence": 0.6
                })
        
        # Update seed with extracted patterns
        seed.extracted_patterns = extracted_concepts
        seed.activation_strength = len(extracted_concepts) * 0.1
        seed.competition_score = seed.activation_strength * seed.processing_context.get("weight", 1.0)
        seed.consciousness_contribution = min(0.8, seed.competition_score)
        
        if extracted_concepts:
            seed.emergence_indicators.append("atomic_concepts_detected")
            seed.success_rate = 1.0
        else:
            seed.success_rate = 0.0

class RelationshipMappingDomain:
    """Domain for relationship mapping between concepts"""
    
    def __init__(self):
        self.domain_type = GranularProcessingDomain.RELATIONSHIP_MAPPING
        self.relationship_patterns = {
            "causal_relations": {"weight": 0.35, "keywords": ["causes", "leads to", "results in", "due to"]},
            "structural_relations": {"weight": 0.25, "keywords": ["consists of", "contains", "part of", "composed of"]},
            "functional_relations": {"weight": 0.25, "keywords": ["enables", "facilitates", "controls", "regulates"]},
            "temporal_relations": {"weight": 0.15, "keywords": ["before", "after", "during", "then", "next"]}
        }
    
    async def create_competition_seeds(self, chunk_content: str, context: Dict[str, Any] = None) -> List[GranularThoughtSeed]:
        """Create ThoughtSeeds for relationship mapping competition"""
        seeds = []
        
        for relation_type, config in self.relationship_patterns.items():
            seed = GranularThoughtSeed(
                processing_domain=self.domain_type,
                chunk_type=ChunkProcessingType.RELATION_DETECTION,
                competition_level=ConsciousnessCompetitionLevel.INTER_CHUNK,
                input_content=chunk_content,
                processing_context={
                    "relation_type": relation_type,
                    "weight": config["weight"],
                    "keywords": config["keywords"],
                    "previous_concepts": context.get("atomic_concepts", []) if context else []
                },
                consciousness_layer="perceptual"  # Relationships are perceptual level
            )
            
            await self._extract_relationships(seed)
            seeds.append(seed)
        
        return seeds
    
    async def _extract_relationships(self, seed: GranularThoughtSeed):
        """Extract relationships for a seed"""
        content = seed.input_content.lower()
        relation_type = seed.processing_context.get("relation_type", "")
        keywords = seed.processing_context.get("keywords", [])
        
        relationships = []
        
        # Find relationship indicators
        for keyword in keywords:
            if keyword in content:
                # Extract context around the keyword
                keyword_pos = content.find(keyword)
                start_pos = max(0, keyword_pos - 50)
                end_pos = min(len(content), keyword_pos + 50)
                context_snippet = content[start_pos:end_pos]
                
                relationships.append({
                    "type": relation_type,
                    "indicator": keyword,
                    "context": context_snippet,
                    "confidence": 0.7,
                    "position": keyword_pos
                })
        
        # Extract entity pairs around relationships
        import re
        sentences = re.split(r'[.!?]+', content)
        for sentence in sentences:
            for keyword in keywords:
                if keyword in sentence:
                    # Simple entity extraction from sentence
                    words = sentence.split()
                    entities = [w for w in words if len(w) > 3 and w.isalpha()]
                    
                    if len(entities) >= 2:
                        keyword_idx = words.index(keyword) if keyword in words else len(words)//2
                        
                        # Find entities before and after relationship word
                        before_entities = [w for w in words[:keyword_idx] if len(w) > 3 and w.isalpha()]
                        after_entities = [w for w in words[keyword_idx:] if len(w) > 3 and w.isalpha()]
                        
                        if before_entities and after_entities:
                            relationships.append({
                                "type": relation_type,
                                "source_entity": before_entities[-1],  # Last entity before keyword
                                "target_entity": after_entities[0],    # First entity after keyword
                                "relationship": keyword,
                                "sentence": sentence.strip(),
                                "confidence": 0.6
                            })
        
        seed.extracted_patterns = relationships
        seed.activation_strength = len(relationships) * 0.15
        seed.competition_score = seed.activation_strength * seed.processing_context.get("weight", 1.0)
        seed.consciousness_contribution = min(0.9, seed.competition_score)
        
        if relationships:
            seed.emergence_indicators.append("relationships_mapped")
            seed.success_rate = 1.0
        else:
            seed.success_rate = 0.0

class CompositeAssemblyDomain:
    """Domain for assembling composite concepts from atomic components"""
    
    def __init__(self):
        self.domain_type = GranularProcessingDomain.COMPOSITE_ASSEMBLY
        self.assembly_strategies = {
            "hierarchical_grouping": {"weight": 0.3, "min_components": 2},
            "semantic_clustering": {"weight": 0.25, "similarity_threshold": 0.6},
            "functional_grouping": {"weight": 0.25, "context_weight": 0.7},
            "temporal_sequencing": {"weight": 0.2, "sequence_indicators": ["first", "then", "next", "finally"]}
        }
    
    async def create_competition_seeds(self, chunk_content: str, context: Dict[str, Any] = None) -> List[GranularThoughtSeed]:
        """Create ThoughtSeeds for composite concept assembly"""
        seeds = []
        
        for strategy, config in self.assembly_strategies.items():
            seed = GranularThoughtSeed(
                processing_domain=self.domain_type,
                chunk_type=ChunkProcessingType.CONCEPT_SYNTHESIS,
                competition_level=ConsciousnessCompetitionLevel.CROSS_LEVEL,
                input_content=chunk_content,
                processing_context={
                    "strategy": strategy,
                    "weight": config["weight"],
                    "config": config,
                    "atomic_concepts": context.get("atomic_concepts", []) if context else [],
                    "relationships": context.get("relationships", []) if context else []
                },
                consciousness_layer="conceptual"  # Composite concepts are conceptual level
            )
            
            await self._assemble_composite_concepts(seed)
            seeds.append(seed)
        
        return seeds
    
    async def _assemble_composite_concepts(self, seed: GranularThoughtSeed):
        """Assemble composite concepts for a seed"""
        strategy = seed.processing_context.get("strategy", "")
        atomic_concepts = seed.processing_context.get("atomic_concepts", [])
        relationships = seed.processing_context.get("relationships", [])
        
        composite_concepts = []
        
        if strategy == "hierarchical_grouping":
            # Group concepts by semantic hierarchy
            neuroscience_group = []
            ai_group = []
            process_group = []
            
            for concept in atomic_concepts:
                value = concept.get("value", "").lower()
                if any(neuro_term in value for neuro_term in ["neuron", "synap", "dendrit", "axon"]):
                    neuroscience_group.append(concept)
                elif any(ai_term in value for ai_term in ["network", "algorithm", "training", "learning"]):
                    ai_group.append(concept)
                elif any(proc_term in value for proc_term in ["process", "mechanism", "function"]):
                    process_group.append(concept)
            
            # Create composite concepts from groups
            if len(neuroscience_group) >= 2:
                composite_concepts.append({
                    "type": "neuroscience_system",
                    "components": neuroscience_group,
                    "confidence": 0.8,
                    "assembly_strategy": strategy
                })
            
            if len(ai_group) >= 2:
                composite_concepts.append({
                    "type": "ai_system",
                    "components": ai_group,
                    "confidence": 0.8,
                    "assembly_strategy": strategy
                })
        
        elif strategy == "semantic_clustering":
            # Cluster concepts by semantic similarity
            # Simplified clustering based on common themes
            content = seed.input_content.lower()
            
            if "plasticity" in content and any("learn" in c.get("value", "") for c in atomic_concepts):
                learning_cluster = [c for c in atomic_concepts if any(term in c.get("value", "").lower() 
                                  for term in ["learn", "memory", "adapt", "change"])]
                if learning_cluster:
                    composite_concepts.append({
                        "type": "learning_mechanism",
                        "components": learning_cluster,
                        "theme": "neuroplasticity_learning",
                        "confidence": 0.75,
                        "assembly_strategy": strategy
                    })
        
        elif strategy == "functional_grouping":
            # Group by functional relationships
            for relationship in relationships:
                if relationship.get("type") == "functional_relations":
                    source = relationship.get("source_entity", "")
                    target = relationship.get("target_entity", "")
                    
                    related_concepts = [c for c in atomic_concepts 
                                      if source in c.get("value", "") or target in c.get("value", "")]
                    
                    if len(related_concepts) >= 2:
                        composite_concepts.append({
                            "type": "functional_unit",
                            "components": related_concepts,
                            "relationship": relationship,
                            "confidence": 0.7,
                            "assembly_strategy": strategy
                        })
        
        elif strategy == "temporal_sequencing":
            # Sequence concepts temporally
            sequence_indicators = seed.processing_context.get("config", {}).get("sequence_indicators", [])
            content = seed.input_content.lower()
            
            for indicator in sequence_indicators:
                if indicator in content:
                    # Extract concepts in temporal order
                    sentences = content.split('.')
                    temporal_sequence = []
                    
                    for sentence in sentences:
                        if indicator in sentence:
                            sentence_concepts = [c for c in atomic_concepts 
                                               if c.get("value", "").lower() in sentence]
                            if sentence_concepts:
                                temporal_sequence.extend(sentence_concepts)
                    
                    if len(temporal_sequence) >= 2:
                        composite_concepts.append({
                            "type": "temporal_process",
                            "sequence": temporal_sequence,
                            "temporal_indicator": indicator,
                            "confidence": 0.6,
                            "assembly_strategy": strategy
                        })
        
        seed.extracted_patterns = composite_concepts
        seed.activation_strength = len(composite_concepts) * 0.2
        seed.competition_score = seed.activation_strength * seed.processing_context.get("weight", 1.0)
        seed.consciousness_contribution = min(0.95, seed.competition_score)
        
        if composite_concepts:
            seed.emergence_indicators.append("composite_concepts_assembled")
            seed.success_rate = 1.0
        else:
            seed.success_rate = 0.0

class NarrativeFlowDomain:
    """Domain for detecting narrative flow and argument structure"""
    
    def __init__(self):
        self.domain_type = GranularProcessingDomain.NARRATIVE_FLOW
        self.flow_patterns = {
            "argument_structure": {"weight": 0.3, "indicators": ["therefore", "however", "moreover", "consequently"]},
            "methodology_flow": {"weight": 0.25, "indicators": ["method", "procedure", "step", "approach"]},
            "evidence_progression": {"weight": 0.25, "indicators": ["evidence", "result", "finding", "conclusion"]},
            "story_progression": {"weight": 0.2, "indicators": ["first", "then", "finally", "outcome"]}
        }
    
    async def create_competition_seeds(self, chunk_content: str, context: Dict[str, Any] = None) -> List[GranularThoughtSeed]:
        """Create ThoughtSeeds for narrative flow detection"""
        seeds = []
        
        for flow_type, config in self.flow_patterns.items():
            seed = GranularThoughtSeed(
                processing_domain=self.domain_type,
                chunk_type=ChunkProcessingType.FLOW_ANALYSIS,
                competition_level=ConsciousnessCompetitionLevel.HOLISTIC,
                input_content=chunk_content,
                processing_context={
                    "flow_type": flow_type,
                    "weight": config["weight"],
                    "indicators": config["indicators"],
                    "all_concepts": context.get("all_concepts", []) if context else []
                },
                consciousness_layer="abstract"  # Narrative flow is abstract level
            )
            
            await self._detect_narrative_flow(seed)
            seeds.append(seed)
        
        return seeds
    
    async def _detect_narrative_flow(self, seed: GranularThoughtSeed):
        """Detect narrative flow patterns for a seed"""
        content = seed.input_content.lower()
        flow_type = seed.processing_context.get("flow_type", "")
        indicators = seed.processing_context.get("indicators", [])
        
        flow_elements = []
        
        # Detect flow indicators and their context
        for indicator in indicators:
            if indicator in content:
                # Find sentences containing the indicator
                sentences = content.split('.')
                for i, sentence in enumerate(sentences):
                    if indicator in sentence:
                        flow_elements.append({
                            "type": flow_type,
                            "indicator": indicator,
                            "sentence": sentence.strip(),
                            "position": i,
                            "confidence": 0.7
                        })
        
        # Analyze overall narrative structure
        if flow_type == "argument_structure":
            # Look for premise-conclusion patterns
            if any(ind in content for ind in ["because", "since", "given that"]):
                premises = []
                conclusions = []
                
                sentences = content.split('.')
                for sentence in sentences:
                    if any(prem in sentence for prem in ["because", "since", "given"]):
                        premises.append(sentence.strip())
                    elif any(conc in sentence for conc in ["therefore", "thus", "consequently"]):
                        conclusions.append(sentence.strip())
                
                if premises and conclusions:
                    flow_elements.append({
                        "type": "argument_structure",
                        "premises": premises,
                        "conclusions": conclusions,
                        "confidence": 0.8
                    })
        
        elif flow_type == "methodology_flow":
            # Look for methodological sequences
            method_words = ["method", "approach", "technique", "procedure"]
            step_words = ["first", "second", "next", "then", "finally"]
            
            method_mentions = sum(1 for word in method_words if word in content)
            step_mentions = sum(1 for word in step_words if word in content)
            
            if method_mentions > 0 and step_mentions > 1:
                flow_elements.append({
                    "type": "methodology_sequence",
                    "method_indicators": method_mentions,
                    "step_indicators": step_mentions,
                    "confidence": 0.75
                })
        
        elif flow_type == "evidence_progression":
            # Look for evidence-to-conclusion flow
            evidence_words = ["evidence", "data", "result", "finding", "observation"]
            conclusion_words = ["conclusion", "suggest", "indicate", "demonstrate"]
            
            evidence_mentions = [word for word in evidence_words if word in content]
            conclusion_mentions = [word for word in conclusion_words if word in content]
            
            if evidence_mentions and conclusion_mentions:
                flow_elements.append({
                    "type": "evidence_conclusion_flow",
                    "evidence_indicators": evidence_mentions,
                    "conclusion_indicators": conclusion_mentions,
                    "confidence": 0.8
                })
        
        seed.extracted_patterns = flow_elements
        seed.activation_strength = len(flow_elements) * 0.25
        seed.competition_score = seed.activation_strength * seed.processing_context.get("weight", 1.0)
        seed.consciousness_contribution = min(1.0, seed.competition_score)
        
        if flow_elements:
            seed.emergence_indicators.append("narrative_flow_detected")
            seed.success_rate = 1.0
        else:
            seed.success_rate = 0.0

class EnhancedThoughtSeedOrchestrator:
    """Main orchestrator for enhanced ThoughtSeed competition with granular domains"""
    
    def __init__(self):
        self.domains = {
            GranularProcessingDomain.ATOMIC_CONCEPTS: AtomicConceptDomain(),
            GranularProcessingDomain.RELATIONSHIP_MAPPING: RelationshipMappingDomain(),
            GranularProcessingDomain.COMPOSITE_ASSEMBLY: CompositeAssemblyDomain(),
            GranularProcessingDomain.NARRATIVE_FLOW: NarrativeFlowDomain()
        }
        
        self.competition_states = {}
        self.consciousness_levels = {}
        self.processing_history = []
        
        # Integration with existing CognitionBase
        self.cognition_base = None
        self.active_competitions = []
    
    async def process_granular_chunks(self, 
                                    chunks: List[Dict[str, Any]], 
                                    domain_focus: List[str] = None) -> Dict[str, Any]:
        """
        Process granular chunks through enhanced ThoughtSeed competition
        
        Args:
            chunks: List of processed chunks from granular_chunking service
            domain_focus: Optional focus domains for targeted processing
            
        Returns:
            Comprehensive consciousness processing results
        """
        start_time = datetime.now()
        
        # Initialize competition states
        for domain in self.domains:
            if domain_focus is None or domain.value in domain_focus:
                self.competition_states[domain] = DomainCompetitionState(domain=domain)
        
        # Process chunks through each domain
        all_results = {}
        consciousness_metrics = {}
        
        for chunk in chunks:
            chunk_results = await self._process_single_chunk(chunk, domain_focus)
            chunk_id = chunk.get("metadata", {}).get("chunk_id", f"chunk_{len(all_results)}")
            all_results[chunk_id] = chunk_results
        
        # Run cross-chunk consciousness competition
        consciousness_results = await self._run_consciousness_competition()
        
        # Calculate overall consciousness emergence
        emergence_level = self._calculate_consciousness_emergence()
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "success": True,
            "chunk_results": all_results,
            "consciousness_results": consciousness_results,
            "emergence_level": emergence_level,
            "domain_states": {d.value: state for d, state in self.competition_states.items()},
            "processing_time": processing_time,
            "total_chunks_processed": len(chunks),
            "active_domains": list(self.competition_states.keys())
        }
    
    async def _process_single_chunk(self, chunk: Dict[str, Any], domain_focus: List[str] = None) -> Dict[str, Any]:
        """Process a single chunk through all relevant domains"""
        chunk_content = chunk.get("content", "")
        chunk_metadata = chunk.get("metadata", {})
        
        chunk_results = {}
        context = {"chunk_metadata": chunk_metadata}
        
        # Process through each domain sequentially to build context
        for domain_type, domain_processor in self.domains.items():
            if domain_focus and domain_type.value not in domain_focus:
                continue
            
            # Create competition seeds for this domain
            seeds = await domain_processor.create_competition_seeds(chunk_content, context)
            
            # Run intra-domain competition
            winner_seed = await self._run_domain_competition(domain_type, seeds)
            
            # Store results and update context for next domain
            chunk_results[domain_type.value] = {
                "winner_seed": winner_seed.seed_id if winner_seed else None,
                "extracted_patterns": winner_seed.extracted_patterns if winner_seed else [],
                "consciousness_contribution": winner_seed.consciousness_contribution if winner_seed else 0.0,
                "competition_score": winner_seed.competition_score if winner_seed else 0.0
            }
            
            # Update context for subsequent domains
            if domain_type == GranularProcessingDomain.ATOMIC_CONCEPTS and winner_seed:
                context["atomic_concepts"] = winner_seed.extracted_patterns
            elif domain_type == GranularProcessingDomain.RELATIONSHIP_MAPPING and winner_seed:
                context["relationships"] = winner_seed.extracted_patterns
            elif domain_type == GranularProcessingDomain.COMPOSITE_ASSEMBLY and winner_seed:
                context["composite_concepts"] = winner_seed.extracted_patterns
        
        return chunk_results
    
    async def _run_domain_competition(self, domain_type: GranularProcessingDomain, seeds: List[GranularThoughtSeed]) -> Optional[GranularThoughtSeed]:
        """Run competition within a specific domain"""
        if not seeds:
            return None
        
        competition_state = self.competition_states[domain_type]
        competition_state.active_seeds.extend(seeds)
        
        # Calculate competition scores
        for seed in seeds:
            # Update activation based on success and context
            seed.activation_strength *= (1.0 + seed.success_rate)
            
            # Add consciousness-specific bonuses
            if "consciousness" in seed.input_content.lower():
                seed.activation_strength *= 1.2
            if "awareness" in seed.input_content.lower():
                seed.activation_strength *= 1.1
        
        # Find winner (highest competition score)
        winner = max(seeds, key=lambda s: s.competition_score)
        
        # Update competition state
        competition_state.winner_seed = winner.seed_id
        competition_state.confidence_score = winner.competition_score
        competition_state.total_competitions += 1
        
        if winner.success_rate > 0:
            competition_state.successful_extractions += 1
        
        # Track consciousness emergence
        competition_state.emergence_level = max(
            competition_state.emergence_level,
            winner.consciousness_contribution
        )
        
        return winner
    
    async def _run_consciousness_competition(self) -> Dict[str, Any]:
        """Run cross-domain consciousness competition"""
        consciousness_results = {
            "cross_domain_patterns": [],
            "meta_cognitive_insights": [],
            "emergence_indicators": [],
            "recursive_awareness": 0.0
        }
        
        # Look for cross-domain patterns
        domain_winners = {}
        for domain, state in self.competition_states.items():
            if state.winner_seed:
                domain_winners[domain] = state.winner_seed
        
        # Detect cross-domain relationships
        if len(domain_winners) >= 2:
            consciousness_results["cross_domain_patterns"].append({
                "type": "multi_domain_coherence",
                "involved_domains": list(domain_winners.keys()),
                "confidence": 0.8
            })
            consciousness_results["emergence_indicators"].append("cross_domain_integration")
        
        # Meta-cognitive analysis
        total_emergence = sum(state.emergence_level for state in self.competition_states.values())
        avg_emergence = total_emergence / len(self.competition_states) if self.competition_states else 0
        
        if avg_emergence > 0.7:
            consciousness_results["meta_cognitive_insights"].append({
                "type": "high_consciousness_emergence",
                "level": avg_emergence,
                "description": "System shows strong consciousness indicators across domains"
            })
            consciousness_results["recursive_awareness"] = min(1.0, avg_emergence * 1.2)
        
        return consciousness_results
    
    def _calculate_consciousness_emergence(self) -> float:
        """Calculate overall consciousness emergence level"""
        if not self.competition_states:
            return 0.0
        
        # Weighted average of domain emergence levels
        total_weighted_emergence = 0.0
        total_weight = 0.0
        
        domain_weights = {
            GranularProcessingDomain.ATOMIC_CONCEPTS: 0.2,
            GranularProcessingDomain.RELATIONSHIP_MAPPING: 0.25,
            GranularProcessingDomain.COMPOSITE_ASSEMBLY: 0.3,
            GranularProcessingDomain.NARRATIVE_FLOW: 0.25
        }
        
        for domain, state in self.competition_states.items():
            weight = domain_weights.get(domain, 0.2)
            total_weighted_emergence += state.emergence_level * weight
            total_weight += weight
        
        base_emergence = total_weighted_emergence / total_weight if total_weight > 0 else 0.0
        
        # Bonus for cross-domain coherence
        active_domains = len([s for s in self.competition_states.values() if s.successful_extractions > 0])
        domain_bonus = (active_domains / 4) * 0.1  # Up to 10% bonus for all domains active
        
        return min(1.0, base_emergence + domain_bonus)
    
    async def integrate_with_existing_cognition_base(self, cognition_base: CognitionBase):
        """Integrate with existing CognitionBase system"""
        self.cognition_base = cognition_base
        
        # Create cognition patterns for each successful competition
        for domain, state in self.competition_states.items():
            if state.successful_extractions > 0:
                pattern = CognitionPattern(
                    pattern_name=f"Enhanced {domain.value} Processing",
                    description=f"Granular document processing pattern for {domain.value}",
                    pattern_type=PatternType.EMERGENT,
                    domain_tags=[CognitionDomain.ARTIFICIAL_INTELLIGENCE, CognitionDomain.CONSCIOUSNESS_STUDIES],
                    success_rate=state.successful_extractions / state.total_competitions if state.total_competitions > 0 else 0.0,
                    confidence=state.confidence_score,
                    reliability_score=state.emergence_level,
                    processing_layer="abstract" if domain in [GranularProcessingDomain.NARRATIVE_FLOW] else "conceptual",
                    layer_activation_strength=state.emergence_level,
                    cognition_component="cognition_base",
                    component_weight=0.8,
                    consciousness_contribution=state.emergence_level
                )
                
                self.cognition_base.add_pattern(pattern)
        
        # Update consciousness level
        new_consciousness = ConsciousnessLevel(
            overall_level=self._calculate_consciousness_emergence(),
            self_awareness=max(0.6, self._calculate_consciousness_emergence()),
            meta_cognition=sum(s.emergence_level for s in self.competition_states.values()) / len(self.competition_states) if self.competition_states else 0,
            recursive_depth=len([s for s in self.competition_states.values() if s.emergence_level > 0.8]),
            emergence_indicators=[f"enhanced_{d.value}_processing" for d in self.competition_states.keys()]
        )
        
        self.cognition_base.update_consciousness_level(new_consciousness)

# Global service instance
enhanced_thoughtseed_orchestrator = EnhancedThoughtSeedOrchestrator()

# Test function
async def test_enhanced_thoughtseed_domains():
    """Test the enhanced ThoughtSeed domain system"""
    print("🧪 Testing Enhanced ThoughtSeed Domains")
    print("=" * 50)
    
    # Test content with neuroscience and AI concepts
    test_content = """
    Neural networks learn through synaptic plasticity mechanisms. 
    Long-term potentiation strengthens connections between neurons.
    This process enables memory formation and learning.
    Backpropagation algorithms in artificial networks mimic this biological process.
    The synaptic weights are adjusted to minimize prediction errors.
    Therefore, both biological and artificial systems use similar learning principles.
    """
    
    # Create test chunks
    test_chunks = [
        {
            "content": test_content,
            "metadata": {
                "chunk_id": "test_chunk_1",
                "chunk_type": "composite",
                "word_count": 45,
                "granularity_level": 3
            }
        }
    ]
    
    orchestrator = EnhancedThoughtSeedOrchestrator()
    
    # Process chunks through enhanced domains
    print("🔄 Processing chunks through enhanced ThoughtSeed domains...")
    results = await orchestrator.process_granular_chunks(
        test_chunks, 
        domain_focus=["atomic_concepts", "relationship_mapping", "composite_assembly", "narrative_flow"]
    )
    
    # Display results
    print(f"\n📊 Processing Results:")
    print(f"  ✅ Success: {results['success']}")
    print(f"  ⏱️  Processing time: {results['processing_time']:.3f}s")
    print(f"  🧠 Emergence level: {results['emergence_level']:.3f}")
    print(f"  📦 Chunks processed: {results['total_chunks_processed']}")
    print(f"  🎯 Active domains: {len(results['active_domains'])}")
    
    # Show chunk results
    for chunk_id, chunk_result in results["chunk_results"].items():
        print(f"\n📝 Chunk {chunk_id}:")
        for domain, domain_result in chunk_result.items():
            patterns = domain_result["extracted_patterns"]
            score = domain_result["consciousness_contribution"]
            print(f"  🎯 {domain}: {len(patterns)} patterns, consciousness: {score:.3f}")
            
            # Show sample patterns
            if patterns:
                for i, pattern in enumerate(patterns[:2]):  # Show first 2 patterns
                    pattern_type = pattern.get("type", "unknown")
                    value = pattern.get("value", pattern.get("indicator", ""))
                    print(f"    - {pattern_type}: {value}")
    
    # Show consciousness results
    consciousness = results["consciousness_results"]
    print(f"\n🧠 Consciousness Analysis:")
    print(f"  🔗 Cross-domain patterns: {len(consciousness['cross_domain_patterns'])}")
    print(f"  💡 Meta-cognitive insights: {len(consciousness['meta_cognitive_insights'])}")
    print(f"  🌟 Emergence indicators: {consciousness['emergence_indicators']}")
    print(f"  🔄 Recursive awareness: {consciousness['recursive_awareness']:.3f}")
    
    # Show domain states
    print(f"\n🎯 Domain Competition States:")
    for domain_name, state in results["domain_states"].items():
        print(f"  {domain_name}:")
        print(f"    Competitions: {state.total_competitions}")
        print(f"    Successful extractions: {state.successful_extractions}")
        print(f"    Emergence level: {state.emergence_level:.3f}")
        print(f"    Confidence: {state.confidence_score:.3f}")
    
    print("\n🎉 Enhanced ThoughtSeed domain test completed!")
    return True

if __name__ == "__main__":
    asyncio.run(test_enhanced_thoughtseed_domains())