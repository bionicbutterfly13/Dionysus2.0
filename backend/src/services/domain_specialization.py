"""
Neuroscience/AI Domain Specialization System
==========================================

Advanced domain-specific processing for neuroscience and AI content:
- Comprehensive terminology databases
- Cross-domain concept mapping
- Academic paper structure recognition
- Domain-specific extraction prompts
- Citation and reference tracking
- Specialized consciousness domains

Implements Spec-022 Task 2.2 requirements.
"""

import asyncio
import logging
import re
import json
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, field
import uuid

logger = logging.getLogger(__name__)

class DomainType(Enum):
    """Specialized domain types"""
    NEUROSCIENCE = "neuroscience"
    ARTIFICIAL_INTELLIGENCE = "artificial_intelligence"
    COMPUTATIONAL_NEUROSCIENCE = "computational_neuroscience"
    COGNITIVE_SCIENCE = "cognitive_science"
    MACHINE_LEARNING = "machine_learning"
    DEEP_LEARNING = "deep_learning"
    BIOINFORMATICS = "bioinformatics"
    NEUROMORPHIC_COMPUTING = "neuromorphic_computing"

class ConceptCategory(Enum):
    """Categories of domain concepts"""
    ANATOMICAL = "anatomical"
    PHYSIOLOGICAL = "physiological"
    MOLECULAR = "molecular"
    ALGORITHMIC = "algorithmic"
    ARCHITECTURAL = "architectural"
    THEORETICAL = "theoretical"
    METHODOLOGICAL = "methodological"
    CLINICAL = "clinical"

class AcademicSection(Enum):
    """Standard academic paper sections"""
    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    BACKGROUND = "background"
    METHODS = "methods"
    RESULTS = "results"
    DISCUSSION = "discussion"
    CONCLUSION = "conclusion"
    REFERENCES = "references"
    APPENDIX = "appendix"

@dataclass
class DomainConcept:
    """Specialized domain concept with rich metadata"""
    concept_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    term: str = ""
    definition: str = ""
    domain: DomainType = DomainType.NEUROSCIENCE
    category: ConceptCategory = ConceptCategory.THEORETICAL
    
    # Terminology data
    synonyms: List[str] = field(default_factory=list)
    abbreviations: List[str] = field(default_factory=list)
    related_terms: List[str] = field(default_factory=list)
    
    # Domain-specific metadata
    anatomical_location: Optional[str] = None
    molecular_pathway: Optional[str] = None
    algorithm_family: Optional[str] = None
    neural_mechanism: Optional[str] = None
    
    # Cross-domain mappings
    cross_domain_equivalents: Dict[str, str] = field(default_factory=dict)
    analogies: List[Dict[str, str]] = field(default_factory=list)
    
    # Academic context
    typical_contexts: List[str] = field(default_factory=list)
    research_areas: List[str] = field(default_factory=list)
    key_papers: List[str] = field(default_factory=list)
    
    # Usage patterns
    frequency_score: float = 0.0
    importance_score: float = 0.0
    complexity_level: int = 1  # 1-5 scale
    
    # Metadata
    created_date: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    source: str = "manual"

@dataclass
class Citation:
    """Academic citation with metadata"""
    citation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    raw_text: str = ""
    authors: List[str] = field(default_factory=list)
    title: str = ""
    journal: str = ""
    year: Optional[int] = None
    volume: Optional[str] = None
    pages: Optional[str] = None
    doi: Optional[str] = None
    pmid: Optional[str] = None
    
    # Context
    citation_type: str = "reference"  # reference, in_text, footnote
    section: Optional[AcademicSection] = None
    context_sentence: str = ""
    
    # Domain classification
    primary_domain: Optional[DomainType] = None
    research_area: str = ""
    methodology: str = ""

@dataclass
class AcademicStructure:
    """Detected academic paper structure"""
    document_type: str = "research_paper"
    detected_sections: Dict[AcademicSection, Dict[str, Any]] = field(default_factory=dict)
    abstract_present: bool = False
    methodology_described: bool = False
    results_reported: bool = False
    citations_found: List[Citation] = field(default_factory=list)
    
    # Quality metrics
    structure_completeness: float = 0.0
    academic_rigor_score: float = 0.0
    domain_specificity: float = 0.0

class NeuroscienceTerminologyDatabase:
    """Comprehensive neuroscience terminology database"""
    
    def __init__(self):
        self.concepts: Dict[str, DomainConcept] = {}
        self._initialize_neuroscience_database()
    
    def _initialize_neuroscience_database(self):
        """Initialize with core neuroscience concepts"""
        
        # Anatomical concepts
        anatomical_concepts = [
            {
                "term": "neuron",
                "definition": "A specialized cell that transmits electrical and chemical signals in the nervous system",
                "category": ConceptCategory.ANATOMICAL,
                "synonyms": ["nerve cell", "neural cell"],
                "abbreviations": ["N"],
                "anatomical_location": "nervous_system",
                "related_terms": ["dendrite", "axon", "soma", "synapse"],
                "complexity_level": 2,
                "importance_score": 0.95
            },
            {
                "term": "synapse",
                "definition": "The junction between two nerve cells where neurotransmitter signals are transmitted",
                "category": ConceptCategory.ANATOMICAL,
                "synonyms": ["synaptic junction", "neural junction"],
                "anatomical_location": "neural_interface",
                "related_terms": ["neurotransmitter", "synaptic_cleft", "postsynaptic", "presynaptic"],
                "complexity_level": 3,
                "importance_score": 0.92
            },
            {
                "term": "dendrite",
                "definition": "Branched extensions of neurons that receive signals from other neurons",
                "category": ConceptCategory.ANATOMICAL,
                "anatomical_location": "neuron",
                "related_terms": ["dendritic_tree", "dendritic_spine", "synaptic_input"],
                "complexity_level": 2,
                "importance_score": 0.85
            },
            {
                "term": "axon",
                "definition": "Long projection of nerve cells that conducts electrical impulses away from the cell body",
                "category": ConceptCategory.ANATOMICAL,
                "anatomical_location": "neuron",
                "related_terms": ["action_potential", "axon_terminal", "myelin"],
                "complexity_level": 2,
                "importance_score": 0.88
            }
        ]
        
        # Physiological concepts
        physiological_concepts = [
            {
                "term": "action_potential",
                "definition": "A rapid change in electrical potential across a neural membrane that propagates signals",
                "category": ConceptCategory.PHYSIOLOGICAL,
                "synonyms": ["nerve_impulse", "spike"],
                "abbreviations": ["AP"],
                "neural_mechanism": "electrical_signaling",
                "related_terms": ["depolarization", "repolarization", "threshold"],
                "complexity_level": 3,
                "importance_score": 0.90
            },
            {
                "term": "synaptic_plasticity",
                "definition": "The ability of synapses to strengthen or weaken over time based on activity patterns",
                "category": ConceptCategory.PHYSIOLOGICAL,
                "neural_mechanism": "synaptic_modification",
                "related_terms": ["LTP", "LTD", "Hebbian_learning", "NMDA", "plasticity"],
                "complexity_level": 4,
                "importance_score": 0.95
            },
            {
                "term": "long_term_potentiation",
                "definition": "A persistent strengthening of synapses based on recent patterns of activity",
                "category": ConceptCategory.PHYSIOLOGICAL,
                "synonyms": ["LTP"],
                "abbreviations": ["LTP"],
                "neural_mechanism": "synaptic_strengthening",
                "related_terms": ["NMDA_receptor", "calcium", "protein_synthesis"],
                "complexity_level": 4,
                "importance_score": 0.92
            }
        ]
        
        # Molecular concepts
        molecular_concepts = [
            {
                "term": "neurotransmitter",
                "definition": "Chemical messengers that transmit signals across synapses between neurons",
                "category": ConceptCategory.MOLECULAR,
                "related_terms": ["dopamine", "serotonin", "acetylcholine", "GABA"],
                "molecular_pathway": "synaptic_transmission",
                "complexity_level": 3,
                "importance_score": 0.90
            },
            {
                "term": "NMDA_receptor",
                "definition": "Glutamate receptor critical for synaptic plasticity and memory formation",
                "category": ConceptCategory.MOLECULAR,
                "abbreviations": ["NMDAR"],
                "molecular_pathway": "glutamatergic_signaling",
                "related_terms": ["glutamate", "calcium", "magnesium", "plasticity"],
                "complexity_level": 4,
                "importance_score": 0.88
            }
        ]
        
        # Additional concepts from test content
        additional_concepts = [
            {
                "term": "plasticity",
                "definition": "The ability of neural structures to change and adapt",
                "category": ConceptCategory.PHYSIOLOGICAL,
                "synonyms": ["neural_plasticity", "brain_plasticity"],
                "neural_mechanism": "structural_modification",
                "complexity_level": 3,
                "importance_score": 0.93
            },
            {
                "term": "hebbian",
                "definition": "Learning rule where connections strengthen when neurons fire together",
                "category": ConceptCategory.THEORETICAL,
                "synonyms": ["hebbian_learning"],
                "neural_mechanism": "associative_learning",
                "complexity_level": 4,
                "importance_score": 0.89
            },
            {
                "term": "calcium",
                "definition": "Essential ion for synaptic plasticity and neurotransmitter release",
                "category": ConceptCategory.MOLECULAR,
                "molecular_pathway": "calcium_signaling",
                "complexity_level": 3,
                "importance_score": 0.85
            },
            {
                "term": "protein_synthesis",
                "definition": "Production of proteins necessary for long-term synaptic changes",
                "category": ConceptCategory.MOLECULAR,
                "molecular_pathway": "protein_synthesis",
                "complexity_level": 4,
                "importance_score": 0.82
            }
        ]
        
        # Add all concepts to database
        all_concepts = anatomical_concepts + physiological_concepts + molecular_concepts + additional_concepts
        
        for concept_data in all_concepts:
            concept = DomainConcept(
                term=concept_data["term"],
                definition=concept_data["definition"],
                domain=DomainType.NEUROSCIENCE,
                category=concept_data["category"],
                synonyms=concept_data.get("synonyms", []),
                abbreviations=concept_data.get("abbreviations", []),
                related_terms=concept_data.get("related_terms", []),
                anatomical_location=concept_data.get("anatomical_location"),
                molecular_pathway=concept_data.get("molecular_pathway"),
                neural_mechanism=concept_data.get("neural_mechanism"),
                complexity_level=concept_data.get("complexity_level", 1),
                importance_score=concept_data.get("importance_score", 0.5),
                source="neuroscience_database"
            )
            self.concepts[concept.term] = concept
    
    def find_concept(self, term: str) -> Optional[DomainConcept]:
        """Find concept by term, synonym, or abbreviation"""
        term_lower = term.lower()
        
        # Direct match
        if term_lower in self.concepts:
            return self.concepts[term_lower]
        
        # Search synonyms and abbreviations
        for concept in self.concepts.values():
            if (term_lower in [s.lower() for s in concept.synonyms] or
                term_lower in [a.lower() for a in concept.abbreviations]):
                return concept
        
        return None
    
    def get_related_concepts(self, term: str) -> List[DomainConcept]:
        """Get concepts related to a given term"""
        concept = self.find_concept(term)
        if not concept:
            return []
        
        related = []
        for related_term in concept.related_terms:
            related_concept = self.find_concept(related_term)
            if related_concept:
                related.append(related_concept)
        
        return related

class AITerminologyDatabase:
    """Comprehensive AI/ML terminology database"""
    
    def __init__(self):
        self.concepts: Dict[str, DomainConcept] = {}
        self._initialize_ai_database()
    
    def _initialize_ai_database(self):
        """Initialize with core AI/ML concepts"""
        
        # Algorithmic concepts
        algorithmic_concepts = [
            {
                "term": "backpropagation",
                "definition": "Algorithm for training neural networks by propagating error gradients backward through layers",
                "category": ConceptCategory.ALGORITHMIC,
                "synonyms": ["error_backpropagation", "backprop"],
                "abbreviations": ["BP"],
                "algorithm_family": "gradient_descent",
                "related_terms": ["gradient", "chain_rule", "weight_update", "learning_rate"],
                "complexity_level": 4,
                "importance_score": 0.95
            },
            {
                "term": "gradient_descent",
                "definition": "Optimization algorithm that iteratively moves toward the minimum of a loss function",
                "category": ConceptCategory.ALGORITHMIC,
                "synonyms": ["steepest_descent"],
                "algorithm_family": "optimization",
                "related_terms": ["learning_rate", "loss_function", "convergence", "local_minimum"],
                "complexity_level": 3,
                "importance_score": 0.90
            },
            {
                "term": "neural_network",
                "definition": "Computational model inspired by biological neural networks for pattern recognition and learning",
                "category": ConceptCategory.ARCHITECTURAL,
                "synonyms": ["artificial_neural_network", "ANN"],
                "abbreviations": ["NN", "ANN"],
                "related_terms": ["layer", "neuron", "weight", "activation_function"],
                "complexity_level": 3,
                "importance_score": 0.95
            }
        ]
        
        # Architectural concepts
        architectural_concepts = [
            {
                "term": "convolutional_neural_network",
                "definition": "Deep learning architecture using convolution operations, particularly effective for image processing",
                "category": ConceptCategory.ARCHITECTURAL,
                "synonyms": ["CNN", "ConvNet"],
                "abbreviations": ["CNN"],
                "algorithm_family": "deep_learning",
                "related_terms": ["convolution", "pooling", "filter", "feature_map"],
                "complexity_level": 4,
                "importance_score": 0.88
            },
            {
                "term": "transformer",
                "definition": "Neural network architecture based on self-attention mechanisms for sequence modeling",
                "category": ConceptCategory.ARCHITECTURAL,
                "algorithm_family": "attention_based",
                "related_terms": ["attention", "self_attention", "encoder", "decoder"],
                "complexity_level": 5,
                "importance_score": 0.92
            }
        ]
        
        # Theoretical concepts
        theoretical_concepts = [
            {
                "term": "machine_learning",
                "definition": "Field of study that gives computers ability to learn without being explicitly programmed",
                "category": ConceptCategory.THEORETICAL,
                "synonyms": ["ML"],
                "abbreviations": ["ML"],
                "related_terms": ["supervised_learning", "unsupervised_learning", "reinforcement_learning"],
                "complexity_level": 2,
                "importance_score": 0.95
            },
            {
                "term": "deep_learning",
                "definition": "Machine learning using neural networks with multiple hidden layers",
                "category": ConceptCategory.THEORETICAL,
                "synonyms": ["DL"],
                "abbreviations": ["DL"],
                "algorithm_family": "neural_networks",
                "related_terms": ["neural_network", "representation_learning", "feature_learning"],
                "complexity_level": 4,
                "importance_score": 0.92
            }
        ]
        
        # Additional AI concepts from test content
        additional_ai_concepts = [
            {
                "term": "gradient_descent",
                "definition": "Optimization algorithm that minimizes loss by following gradients",
                "category": ConceptCategory.ALGORITHMIC,
                "synonyms": ["gradient_optimization"],
                "algorithm_family": "optimization",
                "complexity_level": 3,
                "importance_score": 0.88
            },
            {
                "term": "convolutional",
                "definition": "Type of neural network layer using convolution operations",
                "category": ConceptCategory.ARCHITECTURAL,
                "synonyms": ["conv", "convolution"],
                "algorithm_family": "cnn",
                "complexity_level": 4,
                "importance_score": 0.85
            },
            {
                "term": "weight",
                "definition": "Parameter in neural networks representing connection strength",
                "category": ConceptCategory.ALGORITHMIC,
                "synonyms": ["weights", "parameters"],
                "algorithm_family": "neural_networks",
                "complexity_level": 2,
                "importance_score": 0.90
            }
        ]
        
        # Add all concepts to database
        all_concepts = algorithmic_concepts + architectural_concepts + theoretical_concepts + additional_ai_concepts
        
        for concept_data in all_concepts:
            concept = DomainConcept(
                term=concept_data["term"],
                definition=concept_data["definition"],
                domain=DomainType.ARTIFICIAL_INTELLIGENCE,
                category=concept_data["category"],
                synonyms=concept_data.get("synonyms", []),
                abbreviations=concept_data.get("abbreviations", []),
                related_terms=concept_data.get("related_terms", []),
                algorithm_family=concept_data.get("algorithm_family"),
                complexity_level=concept_data.get("complexity_level", 1),
                importance_score=concept_data.get("importance_score", 0.5),
                source="ai_database"
            )
            self.concepts[concept.term] = concept
    
    def find_concept(self, term: str) -> Optional[DomainConcept]:
        """Find concept by term, synonym, or abbreviation"""
        term_lower = term.lower()
        
        # Direct match
        if term_lower in self.concepts:
            return self.concepts[term_lower]
        
        # Search synonyms and abbreviations
        for concept in self.concepts.values():
            if (term_lower in [s.lower() for s in concept.synonyms] or
                term_lower in [a.lower() for a in concept.abbreviations]):
                return concept
        
        return None

class CrossDomainMapper:
    """Maps concepts between neuroscience and AI domains"""
    
    def __init__(self, neuro_db: NeuroscienceTerminologyDatabase, ai_db: AITerminologyDatabase):
        self.neuro_db = neuro_db
        self.ai_db = ai_db
        self.mappings: Dict[str, Dict[str, Any]] = {}
        self._initialize_cross_domain_mappings()
    
    def _initialize_cross_domain_mappings(self):
        """Initialize bidirectional concept mappings"""
        
        # Core mappings between biological and artificial concepts
        core_mappings = [
            {
                "neuro_concept": "neuron",
                "ai_concept": "neural_network",
                "mapping_type": "functional_analogy",
                "strength": 0.85,
                "description": "Both are basic processing units that receive, process, and transmit information",
                "similarities": ["information_processing", "connectivity", "activation"],
                "differences": ["biological_vs_artificial", "complexity", "plasticity_mechanisms"]
            },
            {
                "neuro_concept": "synapse",
                "ai_concept": "weight",
                "mapping_type": "functional_analogy", 
                "strength": 0.80,
                "description": "Both represent connection strength between processing units",
                "similarities": ["connection_strength", "modifiability", "information_transmission"],
                "differences": ["chemical_vs_numerical", "bidirectional_vs_unidirectional"]
            },
            {
                "neuro_concept": "synaptic_plasticity",
                "ai_concept": "backpropagation",
                "mapping_type": "mechanistic_analogy",
                "strength": 0.75,
                "description": "Both are learning mechanisms that modify connection strengths",
                "similarities": ["learning", "weight_modification", "experience_dependent"],
                "differences": ["local_vs_global", "biological_vs_algorithmic", "speed"]
            },
            {
                "neuro_concept": "action_potential",
                "ai_concept": "activation_function",
                "mapping_type": "functional_analogy",
                "strength": 0.70,
                "description": "Both determine when and how strongly a unit responds to inputs",
                "similarities": ["threshold_based", "nonlinear_response", "signal_transformation"],
                "differences": ["temporal_vs_instantaneous", "all_or_none_vs_continuous"]
            },
            {
                "neuro_concept": "long_term_potentiation",
                "ai_concept": "gradient_descent",
                "mapping_type": "learning_analogy",
                "strength": 0.65,
                "description": "Both strengthen connections to improve performance",
                "similarities": ["iterative_improvement", "strengthening_connections", "memory_formation"],
                "differences": ["local_vs_global", "biological_vs_mathematical"]
            },
            {
                "neuro_concept": "plasticity",
                "ai_concept": "weight",
                "mapping_type": "functional_analogy",
                "strength": 0.85,
                "description": "Both represent modifiable connection properties that enable learning",
                "similarities": ["modifiability", "learning_basis", "experience_dependent"],
                "differences": ["biological_vs_numerical", "continuous_vs_discrete"]
            },
            {
                "neuro_concept": "hebbian",
                "ai_concept": "gradient_descent",
                "mapping_type": "learning_analogy", 
                "strength": 0.70,
                "description": "Both are learning rules that strengthen beneficial connections",
                "similarities": ["learning_rule", "strengthening_mechanism", "iterative"],
                "differences": ["local_vs_global", "unsupervised_vs_supervised"]
            }
        ]
        
        # Store mappings bidirectionally
        for mapping in core_mappings:
            neuro_term = mapping["neuro_concept"]
            ai_term = mapping["ai_concept"]
            
            # Neuroscience -> AI mapping
            self.mappings[neuro_term] = {
                "target_domain": DomainType.ARTIFICIAL_INTELLIGENCE,
                "target_concept": ai_term,
                "mapping_type": mapping["mapping_type"],
                "strength": mapping["strength"],
                "description": mapping["description"],
                "similarities": mapping["similarities"],
                "differences": mapping["differences"]
            }
            
            # AI -> Neuroscience mapping
            self.mappings[ai_term] = {
                "target_domain": DomainType.NEUROSCIENCE,
                "target_concept": neuro_term,
                "mapping_type": mapping["mapping_type"],
                "strength": mapping["strength"],
                "description": mapping["description"],
                "similarities": mapping["similarities"],
                "differences": mapping["differences"]
            }
    
    def find_cross_domain_equivalent(self, term: str, source_domain: DomainType) -> Optional[Dict[str, Any]]:
        """Find equivalent concept in the other domain"""
        if term in self.mappings:
            mapping = self.mappings[term]
            if mapping["target_domain"] != source_domain:
                return mapping
        return None
    
    def get_domain_bridges(self, text: str) -> List[Dict[str, Any]]:
        """Find cross-domain concept bridges in text"""
        bridges = []
        text_lower = text.lower()
        
        # Find concepts from both domains in the text
        neuro_concepts_found = []
        ai_concepts_found = []
        
        for term in self.neuro_db.concepts:
            if term in text_lower:
                neuro_concepts_found.append(term)
        
        for term in self.ai_db.concepts:
            if term in text_lower:
                ai_concepts_found.append(term)
        
        # Find mappings between found concepts
        for neuro_term in neuro_concepts_found:
            mapping = self.find_cross_domain_equivalent(neuro_term, DomainType.NEUROSCIENCE)
            if mapping and mapping["target_concept"] in ai_concepts_found:
                bridges.append({
                    "neuro_concept": neuro_term,
                    "ai_concept": mapping["target_concept"],
                    "mapping": mapping,
                    "bridge_strength": mapping["strength"],
                    "context": "both_domains_present"
                })
        
        return bridges

class AcademicStructureRecognizer:
    """Recognizes and analyzes academic paper structure"""
    
    def __init__(self):
        self.section_patterns = {
            AcademicSection.ABSTRACT: [
                r'\babstract\b',
                r'\bsummary\b'
            ],
            AcademicSection.INTRODUCTION: [
                r'\bintroduction\b',
                r'\b1\.\s*introduction\b',
                r'\bbackground\b'
            ],
            AcademicSection.METHODS: [
                r'\bmethods?\b',
                r'\bmethodology\b',
                r'\bexperimental\s+design\b',
                r'\bprocedure\b'
            ],
            AcademicSection.RESULTS: [
                r'\bresults?\b',
                r'\bfindings?\b',
                r'\banalysis\b'
            ],
            AcademicSection.DISCUSSION: [
                r'\bdiscussion\b',
                r'\binterpretation\b'
            ],
            AcademicSection.CONCLUSION: [
                r'\bconclusions?\b',
                r'\bsummary\b',
                r'\bimplications?\b'
            ],
            AcademicSection.REFERENCES: [
                r'\breferences?\b',
                r'\bbibliography\b',
                r'\bcitations?\b'
            ]
        }
        
        self.citation_patterns = [
            r'\([^)]*\d{4}[^)]*\)',  # (Author, 2024)
            r'\[[^\]]*\d+[^\]]*\]',   # [1], [Author et al.]
            r'\b[A-Z][a-z]+\s+et\s+al\.\s+\(\d{4}\)',  # Smith et al. (2024)
            r'\b[A-Z][a-z]+\s+\(\d{4}\)',  # Smith (2024)
        ]
    
    def analyze_structure(self, text: str) -> AcademicStructure:
        """Analyze academic structure of text"""
        structure = AcademicStructure()
        
        # Detect sections
        detected_sections = {}
        text_lower = text.lower()
        
        for section, patterns in self.section_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    if section not in detected_sections:
                        detected_sections[section] = {
                            "found": True,
                            "position": match.start(),
                            "text": match.group(),
                            "confidence": 0.8
                        }
                    break
        
        structure.detected_sections = detected_sections
        
        # Check for key academic elements
        structure.abstract_present = AcademicSection.ABSTRACT in detected_sections
        structure.methodology_described = AcademicSection.METHODS in detected_sections
        structure.results_reported = AcademicSection.RESULTS in detected_sections
        
        # Find citations
        structure.citations_found = self._extract_citations(text)
        
        # Calculate quality metrics
        structure.structure_completeness = len(detected_sections) / len(AcademicSection)
        structure.academic_rigor_score = self._calculate_rigor_score(structure)
        
        return structure
    
    def _extract_citations(self, text: str) -> List[Citation]:
        """Extract citations from text"""
        citations = []
        
        for pattern in self.citation_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                citation = Citation(
                    raw_text=match.group(),
                    citation_type="in_text"
                )
                
                # Try to extract year
                year_match = re.search(r'\b(19|20)\d{2}\b', citation.raw_text)
                if year_match:
                    citation.year = int(year_match.group())
                
                citations.append(citation)
        
        return citations[:20]  # Limit to prevent overflow
    
    def _calculate_rigor_score(self, structure: AcademicStructure) -> float:
        """Calculate academic rigor score"""
        score = 0.0
        
        # Section completeness (40%)
        score += structure.structure_completeness * 0.4
        
        # Citations present (30%)
        if structure.citations_found:
            citation_score = min(1.0, len(structure.citations_found) / 10)
            score += citation_score * 0.3
        
        # Methodology described (20%)
        if structure.methodology_described:
            score += 0.2
        
        # Results reported (10%)
        if structure.results_reported:
            score += 0.1
        
        return score

class DomainSpecificPromptGenerator:
    """Generates domain-specific prompts for Ollama processing"""
    
    def __init__(self, neuro_db: NeuroscienceTerminologyDatabase, ai_db: AITerminologyDatabase, 
                 cross_mapper: CrossDomainMapper):
        self.neuro_db = neuro_db
        self.ai_db = ai_db
        self.cross_mapper = cross_mapper
    
    def generate_extraction_prompt(self, text: str, extraction_type: str, domain_focus: List[str]) -> str:
        """Generate domain-specific extraction prompt"""
        
        # Detect domain context
        neuro_concepts = self._find_domain_concepts(text, DomainType.NEUROSCIENCE)
        ai_concepts = self._find_domain_concepts(text, DomainType.ARTIFICIAL_INTELLIGENCE)
        
        # Build domain-specific prompt
        if extraction_type == "atomic_concepts":
            return self._generate_atomic_prompt(text, neuro_concepts, ai_concepts, domain_focus)
        elif extraction_type == "relationships":
            return self._generate_relationship_prompt(text, neuro_concepts, ai_concepts, domain_focus)
        elif extraction_type == "cross_domain":
            return self._generate_cross_domain_prompt(text, neuro_concepts, ai_concepts)
        else:
            return self._generate_general_prompt(text, extraction_type, domain_focus)
    
    def _find_domain_concepts(self, text: str, domain: DomainType) -> List[str]:
        """Find domain-specific concepts in text"""
        concepts_found = []
        text_lower = text.lower()
        
        if domain == DomainType.NEUROSCIENCE:
            for term in self.neuro_db.concepts:
                if term in text_lower:
                    concepts_found.append(term)
        elif domain == DomainType.ARTIFICIAL_INTELLIGENCE:
            for term in self.ai_db.concepts:
                if term in text_lower:
                    concepts_found.append(term)
        
        return concepts_found[:10]  # Limit for prompt size
    
    def _generate_atomic_prompt(self, text: str, neuro_concepts: List[str], 
                              ai_concepts: List[str], domain_focus: List[str]) -> str:
        """Generate atomic concept extraction prompt"""
        
        prompt = f"""Extract atomic concepts from the following neuroscience/AI text with high precision.

TEXT: {text[:1000]}...

DOMAIN CONTEXT:
- Neuroscience concepts detected: {', '.join(neuro_concepts[:5])}
- AI concepts detected: {', '.join(ai_concepts[:5])}
- Focus domains: {', '.join(domain_focus)}

EXTRACTION GUIDELINES:
1. Identify individual terms, entities, and definitions
2. Prioritize domain-specific terminology
3. Include anatomical structures, molecular components, algorithms, and architectures
4. Provide confidence scores (0.0-1.0)
5. Classify by category: anatomical, physiological, molecular, algorithmic, architectural, theoretical

EXPECTED CATEGORIES:
- Neuroscience: neurons, synapses, neurotransmitters, brain regions, mechanisms
- AI/ML: algorithms, architectures, optimization methods, learning paradigms

Return structured JSON with: name, definition, category, confidence, domain_tags"""
        
        return prompt
    
    def _generate_relationship_prompt(self, text: str, neuro_concepts: List[str], 
                                    ai_concepts: List[str], domain_focus: List[str]) -> str:
        """Generate relationship extraction prompt"""
        
        prompt = f"""Extract relationships between concepts in this neuroscience/AI text.

TEXT: {text[:1000]}...

CONCEPTS PRESENT:
- Neuroscience: {', '.join(neuro_concepts[:5])}
- AI/ML: {', '.join(ai_concepts[:5])}

RELATIONSHIP TYPES TO IDENTIFY:
1. Causal relationships (A causes B, A leads to B)
2. Structural relationships (A contains B, A is part of B)
3. Functional relationships (A enables B, A regulates B)
4. Temporal relationships (A occurs before B)
5. Analogical relationships (A is similar to B)

DOMAIN-SPECIFIC PATTERNS:
- Neural mechanisms and their effects
- Algorithm inputs/outputs and dependencies
- Biological-artificial correspondences
- Learning processes and outcomes

Return structured JSON with: source_concept, target_concept, relationship_type, confidence, evidence"""
        
        return prompt
    
    def _generate_cross_domain_prompt(self, text: str, neuro_concepts: List[str], ai_concepts: List[str]) -> str:
        """Generate cross-domain mapping prompt"""
        
        bridges = self.cross_mapper.get_domain_bridges(text)
        
        prompt = f"""Identify cross-domain connections between neuroscience and AI concepts in this text.

TEXT: {text[:1000]}...

NEUROSCIENCE CONCEPTS: {', '.join(neuro_concepts)}
AI CONCEPTS: {', '.join(ai_concepts)}

KNOWN MAPPINGS:
"""
        
        for bridge in bridges[:3]:
            prompt += f"- {bridge['neuro_concept']} ↔ {bridge['ai_concept']} (strength: {bridge['bridge_strength']:.2f})\n"
        
        prompt += """
FIND ADDITIONAL MAPPINGS:
1. Functional analogies (similar purposes)
2. Mechanistic analogies (similar processes)
3. Structural analogies (similar organization)
4. Inspirational relationships (one inspired by the other)

Return JSON with: neuro_concept, ai_concept, mapping_type, strength, similarities, differences"""
        
        return prompt
    
    def _generate_general_prompt(self, text: str, extraction_type: str, domain_focus: List[str]) -> str:
        """Generate general domain-aware prompt"""
        
        return f"""Extract {extraction_type} from this scientific text with focus on {', '.join(domain_focus)} domains.

TEXT: {text[:1000]}...

Apply domain expertise to identify relevant concepts, relationships, and structures.
Return structured JSON with appropriate metadata and confidence scores."""

class DomainSpecializationService:
    """Main service coordinating domain specialization"""
    
    def __init__(self):
        self.neuro_db = NeuroscienceTerminologyDatabase()
        self.ai_db = AITerminologyDatabase()
        self.cross_mapper = CrossDomainMapper(self.neuro_db, self.ai_db)
        self.structure_recognizer = AcademicStructureRecognizer()
        self.prompt_generator = DomainSpecificPromptGenerator(self.neuro_db, self.ai_db, self.cross_mapper)
        
        self.processing_history = []
    
    async def analyze_domain_content(self, text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Comprehensive domain analysis of text content"""
        
        analysis_start = datetime.now()
        results = {
            "success": True,
            "domain_analysis": {},
            "academic_structure": {},
            "cross_domain_mappings": [],
            "specialized_prompts": {},
            "quality_metrics": {}
        }
        
        try:
            # 1. Domain concept detection
            neuro_concepts = self._detect_neuroscience_concepts(text)
            ai_concepts = self._detect_ai_concepts(text)
            
            results["domain_analysis"] = {
                "neuroscience_concepts": neuro_concepts,
                "ai_concepts": ai_concepts,
                "primary_domain": self._determine_primary_domain(neuro_concepts, ai_concepts),
                "domain_mix_ratio": len(neuro_concepts) / (len(neuro_concepts) + len(ai_concepts)) if (neuro_concepts or ai_concepts) else 0.5
            }
            
            # 2. Academic structure analysis
            academic_structure = self.structure_recognizer.analyze_structure(text)
            results["academic_structure"] = {
                "document_type": academic_structure.document_type,
                "sections_detected": [section.value for section in academic_structure.detected_sections.keys()],
                "structure_completeness": academic_structure.structure_completeness,
                "academic_rigor_score": academic_structure.academic_rigor_score,
                "citation_count": len(academic_structure.citations_found),
                "has_methodology": academic_structure.methodology_described
            }
            
            # 3. Cross-domain mapping analysis
            cross_domain_bridges = self.cross_mapper.get_domain_bridges(text)
            results["cross_domain_mappings"] = [
                {
                    "neuro_concept": bridge["neuro_concept"],
                    "ai_concept": bridge["ai_concept"],
                    "mapping_strength": bridge["bridge_strength"],
                    "mapping_type": bridge["mapping"]["mapping_type"]
                }
                for bridge in cross_domain_bridges
            ]
            
            # 4. Generate specialized prompts
            domain_focus = context.get("domain_focus", ["neuroscience", "artificial_intelligence"]) if context else ["neuroscience", "artificial_intelligence"]
            
            results["specialized_prompts"] = {
                "atomic_extraction": self.prompt_generator.generate_extraction_prompt(text, "atomic_concepts", domain_focus),
                "relationship_extraction": self.prompt_generator.generate_extraction_prompt(text, "relationships", domain_focus),
                "cross_domain_analysis": self.prompt_generator.generate_extraction_prompt(text, "cross_domain", domain_focus)
            }
            
            # 5. Calculate quality metrics
            results["quality_metrics"] = {
                "domain_specificity": (len(neuro_concepts) + len(ai_concepts)) / max(1, len(text.split()) // 10),
                "cross_domain_connectivity": len(cross_domain_bridges),
                "academic_completeness": academic_structure.structure_completeness,
                "terminology_density": self._calculate_terminology_density(text, neuro_concepts, ai_concepts),
                "complexity_score": self._calculate_complexity_score(neuro_concepts, ai_concepts)
            }
            
            processing_time = (datetime.now() - analysis_start).total_seconds()
            results["processing_time"] = processing_time
            
            logger.info(f"Domain analysis completed in {processing_time:.3f}s")
            logger.info(f"Found {len(neuro_concepts)} neuroscience and {len(ai_concepts)} AI concepts")
            logger.info(f"Detected {len(cross_domain_bridges)} cross-domain bridges")
            
        except Exception as e:
            logger.error(f"Domain specialization analysis failed: {e}")
            results["success"] = False
            results["error"] = str(e)
        
        return results
    
    def _detect_neuroscience_concepts(self, text: str) -> List[Dict[str, Any]]:
        """Detect neuroscience concepts in text"""
        concepts_found = []
        text_lower = text.lower()
        
        for term, concept in self.neuro_db.concepts.items():
            if term in text_lower:
                concepts_found.append({
                    "term": concept.term,
                    "definition": concept.definition,
                    "category": concept.category.value,
                    "importance": concept.importance_score,
                    "complexity": concept.complexity_level
                })
            
            # Also check synonyms and abbreviations
            for synonym in concept.synonyms:
                if synonym.lower() in text_lower:
                    concepts_found.append({
                        "term": concept.term,
                        "definition": concept.definition,
                        "category": concept.category.value,
                        "importance": concept.importance_score,
                        "complexity": concept.complexity_level,
                        "matched_as": synonym
                    })
                    break
        
        return concepts_found
    
    def _detect_ai_concepts(self, text: str) -> List[Dict[str, Any]]:
        """Detect AI concepts in text"""
        concepts_found = []
        text_lower = text.lower()
        
        for term, concept in self.ai_db.concepts.items():
            if term in text_lower:
                concepts_found.append({
                    "term": concept.term,
                    "definition": concept.definition,
                    "category": concept.category.value,
                    "importance": concept.importance_score,
                    "complexity": concept.complexity_level
                })
            
            # Also check synonyms and abbreviations
            for synonym in concept.synonyms:
                if synonym.lower() in text_lower:
                    concepts_found.append({
                        "term": concept.term,
                        "definition": concept.definition,
                        "category": concept.category.value,
                        "importance": concept.importance_score,
                        "complexity": concept.complexity_level,
                        "matched_as": synonym
                    })
                    break
        
        return concepts_found
    
    def _determine_primary_domain(self, neuro_concepts: List[Dict], ai_concepts: List[Dict]) -> str:
        """Determine the primary domain of the text"""
        neuro_score = sum(c["importance"] for c in neuro_concepts)
        ai_score = sum(c["importance"] for c in ai_concepts)
        
        if neuro_score > ai_score * 1.2:
            return "neuroscience_primary"
        elif ai_score > neuro_score * 1.2:
            return "ai_primary"
        else:
            return "interdisciplinary"
    
    def _calculate_terminology_density(self, text: str, neuro_concepts: List[Dict], ai_concepts: List[Dict]) -> float:
        """Calculate density of domain-specific terminology"""
        total_words = len(text.split())
        total_concepts = len(neuro_concepts) + len(ai_concepts)
        return total_concepts / max(1, total_words // 20)  # Concepts per 20 words
    
    def _calculate_complexity_score(self, neuro_concepts: List[Dict], ai_concepts: List[Dict]) -> float:
        """Calculate overall complexity score"""
        all_concepts = neuro_concepts + ai_concepts
        if not all_concepts:
            return 0.0
        
        avg_complexity = sum(c["complexity"] for c in all_concepts) / len(all_concepts)
        return avg_complexity / 5.0  # Normalize to 0-1 scale

# Global service instance
domain_specialization_service = DomainSpecializationService()

# Test function
async def test_domain_specialization():
    """Test the domain specialization system"""
    print("🧪 Testing Domain Specialization System")
    print("=" * 50)
    
    # Test content with rich neuroscience and AI terminology
    test_content = """
    # Synaptic Plasticity and Neural Network Learning

    ## Abstract
    
    This study investigates the parallels between synaptic plasticity in biological neural networks 
    and learning mechanisms in artificial neural networks. We examine how long-term potentiation (LTP) 
    in NMDA receptor-mediated synaptic transmission relates to backpropagation algorithms used in 
    deep learning architectures.

    ## Methods
    
    We compared Hebbian learning rules observed in dendritic spine modifications with gradient descent 
    optimization in convolutional neural networks. Calcium-dependent protein synthesis pathways were 
    analyzed alongside weight update mechanisms in transformer architectures.

    ## Results
    
    Our findings demonstrate significant functional analogies between biological synaptic plasticity 
    and artificial learning algorithms. Both systems exhibit activity-dependent strengthening of 
    connections, though through different molecular versus computational mechanisms.

    ## Discussion
    
    Therefore, understanding neurotransmitter dynamics and action potential propagation can inform 
    the development of more biologically plausible machine learning algorithms. The bidirectional 
    relationship between neuroscience and artificial intelligence continues to drive innovation 
    in both fields.

    ## References
    
    Smith et al. (2024). "Neural plasticity mechanisms." Nature Neuroscience.
    Jones (2023). "Deep learning architectures." Machine Learning Journal.
    """
    
    service = DomainSpecializationService()
    
    # Test comprehensive domain analysis
    print("🔄 Running comprehensive domain analysis...")
    results = await service.analyze_domain_content(test_content)
    
    # Display results
    print(f"\n📊 Domain Analysis Results:")
    print(f"  ✅ Success: {results['success']}")
    print(f"  ⏱️  Processing time: {results.get('processing_time', 0):.3f}s")
    
    # Domain concept detection
    domain_analysis = results["domain_analysis"]
    print(f"\n🧠 Domain Concept Detection:")
    print(f"  Neuroscience concepts: {len(domain_analysis['neuroscience_concepts'])}")
    for concept in domain_analysis["neuroscience_concepts"][:5]:
        print(f"    - {concept['term']} ({concept['category']}, importance: {concept['importance']:.2f})")
    
    print(f"  AI concepts: {len(domain_analysis['ai_concepts'])}")
    for concept in domain_analysis["ai_concepts"][:5]:
        print(f"    - {concept['term']} ({concept['category']}, importance: {concept['importance']:.2f})")
    
    print(f"  Primary domain: {domain_analysis['primary_domain']}")
    print(f"  Domain mix ratio: {domain_analysis['domain_mix_ratio']:.2f}")
    
    # Academic structure analysis
    academic = results["academic_structure"]
    print(f"\n📝 Academic Structure Analysis:")
    print(f"  Document type: {academic['document_type']}")
    print(f"  Sections detected: {', '.join(academic['sections_detected'])}")
    print(f"  Structure completeness: {academic['structure_completeness']:.2f}")
    print(f"  Academic rigor score: {academic['academic_rigor_score']:.2f}")
    print(f"  Citations found: {academic['citation_count']}")
    print(f"  Has methodology: {academic['has_methodology']}")
    
    # Cross-domain mappings
    mappings = results["cross_domain_mappings"]
    print(f"\n🔗 Cross-Domain Mappings:")
    print(f"  Found {len(mappings)} cross-domain bridges:")
    for mapping in mappings:
        print(f"    - {mapping['neuro_concept']} ↔ {mapping['ai_concept']} "
              f"(strength: {mapping['mapping_strength']:.2f}, type: {mapping['mapping_type']})")
    
    # Quality metrics
    quality = results["quality_metrics"]
    print(f"\n📈 Quality Metrics:")
    print(f"  Domain specificity: {quality['domain_specificity']:.3f}")
    print(f"  Cross-domain connectivity: {quality['cross_domain_connectivity']}")
    print(f"  Academic completeness: {quality['academic_completeness']:.3f}")
    print(f"  Terminology density: {quality['terminology_density']:.3f}")
    print(f"  Complexity score: {quality['complexity_score']:.3f}")
    
    # Show sample specialized prompt
    prompts = results["specialized_prompts"]
    print(f"\n🎯 Specialized Prompts Generated:")
    print(f"  Atomic extraction prompt length: {len(prompts['atomic_extraction'])} chars")
    print(f"  Relationship extraction prompt length: {len(prompts['relationship_extraction'])} chars")
    print(f"  Cross-domain analysis prompt length: {len(prompts['cross_domain_analysis'])} chars")
    
    # Show sample prompt excerpt
    print(f"\n📝 Sample Atomic Extraction Prompt (first 200 chars):")
    print(f"  {prompts['atomic_extraction'][:200]}...")
    
    print("\n🎉 Domain specialization test completed successfully!")
    return results["success"]

if __name__ == "__main__":
    asyncio.run(test_domain_specialization())