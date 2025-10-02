# Data Model: Ultra-Granular Document Processing System

## Core Data Structures

### Document Processing Entities

#### ProcessedDocument
```python
@dataclass
class ProcessedDocument:
    """Complete processed document with all extraction levels"""
    id: str                              # Unique document identifier
    title: str                           # Document title/name
    source_type: DocumentType            # PDF, text, web, audio, etc.
    source_path: str                     # Original file path or URL
    upload_timestamp: datetime           # When document was uploaded
    processing_status: ProcessingStatus  # Processing state
    metadata: DocumentMetadata           # File-specific metadata
    
    # Processing Results
    atomic_concepts: List[AtomicConcept]          # Level 1 extractions
    concept_relationships: List[ConceptRelationship]  # Level 2 extractions
    composite_concepts: List[CompositeConcept]    # Level 3 extractions
    contextual_frameworks: List[ContextualFramework]  # Level 4 extractions
    narrative_structures: List[NarrativeStructure]    # Level 5 extractions
    
    # Knowledge Graph Integration
    knowledge_graph_id: Optional[str]    # Neo4j graph identifier
    autoschema_triples: List[KnowledgeTriple]  # AutoSchemaKG extractions
    
    # Memory Integration
    episodic_memory_id: str              # Memory of processing this document
    consciousness_cycles: List[ConsciousnessCycle]  # Processing cycles
    
    # Performance Metrics
    processing_time: float               # Total processing time
    chunk_count: int                    # Number of chunks processed
    consciousness_level_achieved: float  # Highest consciousness during processing
```

#### DocumentMetadata
```python
@dataclass
class DocumentMetadata:
    """Metadata specific to document type"""
    file_size: int                      # Size in bytes
    page_count: Optional[int]           # For PDFs, Word docs
    language: str                       # Detected language
    author: Optional[str]               # Document author
    creation_date: Optional[datetime]   # Document creation date
    domain_classification: List[str]    # Neuroscience, AI, etc.
    
    # Format-specific metadata
    pdf_metadata: Optional[PDFMetadata]
    web_metadata: Optional[WebMetadata]
    audio_metadata: Optional[AudioMetadata]
    office_metadata: Optional[OfficeMetadata]
```

### Five-Level Concept Extraction

#### AtomicConcept (Level 1)
```python
@dataclass
class AtomicConcept:
    """Individual concept extracted from text"""
    id: str                             # Unique concept identifier
    name: str                           # Concept name/term
    definition: Optional[str]           # Extracted or inferred definition
    domain: str                         # neuroscience, ai, general
    confidence_score: float             # Extraction confidence (0-1)
    
    # Source Information
    source_document_id: str             # Document this came from
    source_chunk_id: str                # Specific chunk
    source_sentence: str                # Original sentence
    character_offset: Tuple[int, int]   # Position in document
    
    # Semantic Properties
    concept_type: ConceptType           # entity, process, property, relation
    synonyms: List[str]                 # Alternative terms
    embeddings: np.ndarray              # Vector representation
    
    # Neuroscience/AI Domain Properties
    neuroscience_category: Optional[str]  # neuron, synapse, network, etc.
    ai_category: Optional[str]            # algorithm, model, training, etc.
    cross_domain_mappings: List[str]      # Related concepts in other domains
    
    # Temporal Information
    discovered_at: datetime             # When concept was extracted
    last_reinforced: datetime           # Last time seen in processing
    reinforcement_count: int            # Times encountered across documents
```

#### ConceptRelationship (Level 2)
```python
@dataclass
class ConceptRelationship:
    """Relationship between two or more concepts"""
    id: str                             # Unique relationship identifier
    relationship_type: RelationshipType # causes, enables, is_part_of, etc.
    strength: float                     # Relationship strength (0-1)
    confidence: float                   # Extraction confidence (0-1)
    
    # Relationship Participants
    source_concept_id: str              # Primary concept
    target_concept_id: str              # Related concept
    additional_concepts: List[str]      # Multi-way relationships
    
    # Context Information
    context_sentence: str               # Sentence expressing relationship
    source_document_id: str             # Document source
    source_chunk_id: str                # Chunk source
    
    # Semantic Properties
    directionality: RelationshipDirection  # bidirectional, source_to_target, etc.
    temporal_aspect: Optional[str]       # temporal ordering if relevant
    causal_strength: Optional[float]     # For causal relationships
    
    # Domain-Specific Properties
    neuroscience_relation_type: Optional[str]  # synaptic, neural, cognitive
    ai_relation_type: Optional[str]             # computational, architectural
    
    # Temporal Information
    discovered_at: datetime             # When relationship was extracted
    last_reinforced: datetime           # Last time observed
    evidence_count: int                 # Supporting evidence instances
```

#### CompositeConcept (Level 3)
```python
@dataclass
class CompositeConcept:
    """Complex concept composed of multiple atomic concepts"""
    id: str                             # Unique composite concept identifier
    name: str                           # Composite concept name
    description: str                    # Comprehensive description
    complexity_score: float             # Measure of conceptual complexity (0-1)
    
    # Composition Structure
    component_concepts: List[str]        # Atomic concept IDs
    composition_pattern: CompositionType # hierarchical, networked, sequential
    integration_relationships: List[str] # How components relate
    
    # Emergent Properties
    emergent_properties: List[str]       # Properties not in components
    system_level_behavior: Optional[str] # Emergent behaviors
    
    # Source Information
    source_document_ids: List[str]       # Documents contributing to concept
    supporting_chunks: List[str]         # Evidence chunks
    
    # Domain Integration
    neuroscience_system_type: Optional[str]  # neural_circuit, brain_region
    ai_system_type: Optional[str]             # architecture, algorithm_class
    cross_domain_analogies: List[str]         # Analogies between domains
    
    # Knowledge Integration
    prerequisite_concepts: List[str]     # Concepts needed to understand this
    derived_concepts: List[str]          # Concepts that build on this
    
    # Temporal Evolution
    concept_evolution: List[ConceptEvolutionEvent]  # How understanding evolved
    stability_score: float               # How stable this concept is
```

#### ContextualFramework (Level 4)
```python
@dataclass
class ContextualFramework:
    """Broader theoretical or paradigmatic context"""
    id: str                             # Unique framework identifier
    name: str                           # Framework name
    description: str                    # Framework description
    scope: FrameworkScope               # local, domain, paradigm, universal
    
    # Framework Structure
    core_principles: List[str]           # Fundamental principles
    key_assumptions: List[str]           # Underlying assumptions
    theoretical_foundations: List[str]   # Supporting theories
    
    # Conceptual Organization
    encompassed_concepts: List[str]      # Concepts within this framework
    framework_relationships: List[str]   # Relations to other frameworks
    conceptual_hierarchy: Dict[str, List[str]]  # Hierarchical organization
    
    # Domain Specification
    neuroscience_paradigm: Optional[str]     # computational, biological, etc.
    ai_paradigm: Optional[str]               # symbolic, connectionist, etc.
    interdisciplinary_connections: List[str] # Cross-domain connections
    
    # Historical and Evolutionary Context
    historical_development: List[str]    # Evolution of framework
    current_status: FrameworkStatus      # active, emerging, declining
    future_directions: List[str]         # Predicted developments
    
    # Source and Evidence
    source_documents: List[str]          # Supporting documents
    evidence_strength: float             # Evidence supporting framework
    
    # Application and Impact
    practical_applications: List[str]    # Real-world applications
    research_implications: List[str]     # Research directions
    technological_impact: Optional[str]  # Technology implications
```

#### NarrativeStructure (Level 5)
```python
@dataclass
class NarrativeStructure:
    """Narrative or argumentative structure within content"""
    id: str                             # Unique narrative identifier
    narrative_type: NarrativeType       # research_methodology, argument, story
    title: str                          # Narrative title/theme
    
    # Structural Components
    sequence_elements: List[SequenceElement]  # Ordered narrative elements
    argument_structure: Optional[ArgumentStructure]  # For argumentative texts
    methodology_structure: Optional[MethodologyStructure]  # For research
    
    # Narrative Flow
    progression_type: ProgressionType   # linear, circular, branching
    temporal_markers: List[TemporalMarker]  # Time-based progression
    logical_flow: List[LogicalConnection]    # Logical progressions
    
    # Content Integration
    supporting_concepts: List[str]       # Concepts supporting narrative
    evidence_chain: List[EvidenceElement]  # Evidence supporting argument
    counterarguments: List[str]          # Opposing viewpoints
    
    # Domain-Specific Narratives
    research_narrative: Optional[ResearchNarrative]      # Scientific methodology
    technical_narrative: Optional[TechnicalNarrative]    # Technical explanations
    historical_narrative: Optional[HistoricalNarrative]  # Historical development
    
    # Narrative Quality and Impact
    coherence_score: float               # Narrative coherence (0-1)
    persuasiveness_score: float          # Argument effectiveness (0-1)
    novelty_score: float                 # Originality of narrative (0-1)
    
    # Source and Context
    source_documents: List[str]          # Documents containing narrative
    narrative_context: str               # Broader context of narrative
    
    # Temporal and Evolution
    narrative_evolution: List[NarrativeEvolutionEvent]  # How narrative developed
    cross_document_continuity: List[str]  # Continuation across documents
```

### Processing and Memory Integration

#### ProcessingChunk
```python
@dataclass
class ProcessingChunk:
    """Individual chunk processed by consciousness system"""
    id: str                             # Unique chunk identifier
    document_id: str                    # Parent document
    chunk_level: int                    # Processing level (1-5)
    content: str                        # Chunk content
    
    # Position Information
    start_offset: int                   # Character start in document
    end_offset: int                     # Character end in document
    sequence_number: int                # Order within document
    
    # Processing Context
    preprocessing_context: Dict[str, Any]  # Context before processing
    postprocessing_context: Dict[str, Any] # Context after processing
    overlap_chunks: List[str]           # Overlapping chunks
    
    # Consciousness Processing
    consciousness_cycles: List[str]      # Cycles that processed this chunk
    thoughtseed_winners: List[str]       # Winning thoughtseeds per cycle
    processing_quality: float           # Quality of extraction (0-1)
    
    # Extracted Knowledge References
    extracted_concepts: List[str]        # Concept IDs extracted
    extracted_relationships: List[str]   # Relationship IDs extracted
    composite_contributions: List[str]   # Composite concept contributions
    
    # Performance Metrics
    processing_time: float              # Time to process chunk
    consciousness_level: float          # Peak consciousness during processing
    memory_formation_success: bool      # Whether memories were formed
```

#### ConsciousnessCycle
```python
@dataclass
class ConsciousnessCycle:
    """Individual consciousness processing cycle"""
    id: str                             # Unique cycle identifier
    document_id: str                    # Document being processed
    chunk_id: str                       # Chunk being processed
    cycle_number: int                   # Sequential cycle number
    
    # Consciousness State
    pre_cycle_state: ConsciousnessState  # State before processing
    post_cycle_state: ConsciousnessState # State after processing
    consciousness_level: float          # Achieved consciousness level
    
    # ThoughtSeed Competition
    competing_thoughtseeds: List[ThoughtSeedSnapshot]  # All competing seeds
    winning_thoughtseed: ThoughtSeedSnapshot           # Winner
    competition_dynamics: CompetitionMetrics           # Competition analysis
    
    # MAC Analysis
    mac_analysis: List[MACAnalysisResult]  # Actor-critic analysis
    metacognitive_assessment: MetacognitiveAssessment  # Self-assessment
    
    # IWMT Consciousness Metrics
    iwmt_metrics: IWMTMetrics           # Consciousness measurements
    consciousness_threshold_met: bool   # Whether consciousness achieved
    
    # Context Engineering
    river_flow_state: RiverFlowState    # Information flow dynamics
    semantic_affordances: List[SemanticAffordance]  # Extracted affordances
    attractor_activations: Dict[str, float]  # Basin activations
    
    # Knowledge Formation
    concepts_formed: List[str]           # New concepts discovered
    relationships_formed: List[str]      # New relationships discovered
    memory_consolidations: List[str]     # Memories formed
    
    # Performance and Quality
    cycle_duration: float               # Processing time
    processing_quality: float           # Quality of results
    error_detected: bool                # Whether errors occurred
    error_description: Optional[str]    # Error details if any
```

### Memory and Persistence

#### MemoryFormation
```python
@dataclass
class MemoryFormation:
    """Formation of different types of memories"""
    id: str                             # Unique memory formation identifier
    memory_type: MemoryType             # episodic, semantic, procedural
    formation_trigger: str              # What triggered memory formation
    
    # Source Information
    source_document_id: str             # Document that created memory
    source_consciousness_cycle: str     # Cycle that formed memory
    formation_context: Dict[str, Any]   # Context during formation
    
    # Memory Content
    episodic_content: Optional[EpisodicMemory]      # Episodic memory details
    semantic_content: Optional[SemanticMemory]      # Semantic memory details
    procedural_content: Optional[ProceduralMemory]  # Procedural memory details
    
    # Memory Properties
    strength: float                     # Memory strength (0-1)
    stability: float                    # Memory stability over time
    accessibility: float                # How easily retrieved
    
    # Integration with Knowledge
    related_concepts: List[str]         # Related concept IDs
    related_memories: List[str]         # Associated memory IDs
    cross_document_connections: List[str]  # Connections to other documents
    
    # Temporal Dynamics
    formation_timestamp: datetime       # When memory was formed
    last_accessed: datetime             # Last retrieval
    access_count: int                   # Times accessed
    decay_rate: float                   # Memory decay parameters
    
    # Quality and Validation
    formation_quality: float            # Quality of memory formation
    verification_status: MemoryVerificationStatus  # Verified, suspected, false
    confidence_level: float             # Confidence in memory accuracy
```

#### KnowledgeGraph
```python
@dataclass
class KnowledgeGraph:
    """AutoSchemaKG generated knowledge graph"""
    id: str                             # Unique graph identifier
    name: str                           # Graph name/title
    domain: str                         # Primary domain (neuroscience, ai)
    
    # Graph Structure
    nodes: List[KnowledgeNode]          # All nodes in graph
    edges: List[KnowledgeEdge]          # All edges/relationships
    subgraphs: List[SubGraph]           # Identified subgraphs
    
    # AutoSchemaKG Integration
    extracted_triples: List[KnowledgeTriple]     # Raw extracted triples
    schema_evolution: List[SchemaEvolutionEvent] # Schema development
    confidence_distribution: Dict[str, float]    # Confidence by node/edge
    
    # Source Tracking
    source_documents: List[str]         # Documents contributing to graph
    extraction_sessions: List[str]      # Extraction session IDs
    
    # Graph Quality and Metrics
    graph_density: float                # Connectivity density
    clustering_coefficient: float       # Network clustering
    centrality_measures: Dict[str, float]  # Node centrality scores
    semantic_coherence: float           # Overall semantic consistency
    
    # Evolution and Updates
    version: int                        # Graph version number
    last_updated: datetime              # Last modification
    update_history: List[GraphUpdateEvent]  # Change history
    
    # Integration Points
    neo4j_graph_id: Optional[str]       # Neo4j storage identifier
    vector_embeddings: Dict[str, np.ndarray]  # Node embeddings
    memory_integrations: List[str]       # Associated memory formations
```

### Database Schema Mappings

#### Neo4j Graph Schema
```cypher
# Core Concepts
(:Concept {id, name, definition, domain, confidence_score, neuroscience_category, ai_category})
(:CompositeConcept {id, name, description, complexity_score, system_type})
(:ContextualFramework {id, name, description, scope, paradigm})
(:NarrativeStructure {id, narrative_type, title, coherence_score})

# Relationships
(:Concept)-[:RELATES_TO {strength, confidence, relation_type}]->(:Concept)
(:Concept)-[:PART_OF]->(:CompositeConcept)
(:CompositeConcept)-[:WITHIN_FRAMEWORK]->(:ContextualFramework)
(:NarrativeStructure)-[:INCORPORATES]->(:Concept)

# Documents and Processing
(:Document {id, title, source_type, upload_timestamp, processing_status})
(:ProcessingChunk {id, chunk_level, content, sequence_number})
(:ConsciousnessCycle {id, cycle_number, consciousness_level, processing_time})

# Memory
(:EpisodicMemory {id, formation_timestamp, strength, accessibility})
(:SemanticMemory {id, knowledge_content, integration_level})
(:Memory)-[:FORMED_FROM]->(:ConsciousnessCycle)
(:Memory)-[:RELATES_TO]->(:Concept)
```

#### Redis Hot Memory Schema
```json
{
  "active_session": {
    "session_id": "string",
    "active_documents": ["doc_ids"],
    "current_processing": {
      "document_id": "string",
      "chunk_id": "string", 
      "consciousness_cycle": "cycle_id"
    },
    "recent_concepts": ["concept_ids"],
    "active_context": "context_object"
  },
  "processing_queue": {
    "pending_documents": ["doc_ids"],
    "processing_priority": "priority_object",
    "estimated_completion": "timestamp"
  },
  "real_time_updates": {
    "websocket_connections": ["connection_ids"],
    "update_queue": ["update_objects"],
    "visualization_state": "state_object"
  }
}
```

#### Vector Database Schema
```python
# Qdrant Collections
COLLECTIONS = {
    "atomic_concepts": {
        "vector_size": 768,  # nomic-embed-text dimensions
        "payload": {
            "concept_id": "string",
            "name": "string",
            "domain": "string",
            "document_id": "string",
            "confidence": "float"
        }
    },
    "document_chunks": {
        "vector_size": 768,
        "payload": {
            "chunk_id": "string",
            "document_id": "string",
            "chunk_level": "integer",
            "content": "string",
            "processing_quality": "float"
        }
    },
    "composite_concepts": {
        "vector_size": 768,
        "payload": {
            "composite_id": "string",
            "component_concepts": ["concept_ids"],
            "complexity_score": "float",
            "domain": "string"
        }
    }
}
```

### API Response Models

#### DocumentProcessingResponse
```python
@dataclass
class DocumentProcessingResponse:
    """Response from document processing request"""
    document_id: str
    processing_status: ProcessingStatus
    
    # Progress Information
    total_chunks: int
    processed_chunks: int
    current_processing_level: int
    estimated_completion: datetime
    
    # Real-time Results
    concepts_discovered: int
    relationships_formed: int
    consciousness_cycles_completed: int
    current_consciousness_level: float
    
    # Quality Metrics
    processing_quality: float
    extraction_confidence: float
    memory_formation_success_rate: float
    
    # Errors and Warnings
    errors: List[ProcessingError]
    warnings: List[ProcessingWarning]
    
    # WebSocket Connection
    websocket_endpoint: str
    real_time_updates_available: bool
```

#### ConceptExplorationResponse
```python
@dataclass
class ConceptExplorationResponse:
    """Response for concept exploration requests"""
    concept: AtomicConcept
    
    # Related Concepts
    direct_relationships: List[ConceptRelationship]
    composite_memberships: List[CompositeConcept] 
    framework_contexts: List[ContextualFramework]
    narrative_appearances: List[NarrativeStructure]
    
    # Cross-Document Analysis
    document_appearances: List[DocumentAppearance]
    concept_evolution: List[ConceptEvolutionEvent]
    reinforcement_history: List[ReinforcementEvent]
    
    # Domain Analysis
    neuroscience_connections: List[str]
    ai_connections: List[str]
    cross_domain_analogies: List[str]
    
    # Memory Integration
    associated_memories: List[MemoryFormation]
    episodic_contexts: List[EpisodicMemory]
    
    # Visualization Data
    graph_visualization: GraphVisualizationData
    timeline_visualization: TimelineVisualizationData
```

---

*This data model provides comprehensive structure for ultra-granular document processing with consciousness-guided extraction, multi-tier memory persistence, and real-time visualization capabilities.*