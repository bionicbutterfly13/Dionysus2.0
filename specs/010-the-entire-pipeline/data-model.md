# Data Model: Complete ThoughtSeed Pipeline Implementation

## Core Entities

### Document
**Purpose**: Represents uploaded research documents for processing
**Storage**: Primary in Neo4j, metadata cached in Redis
```python
Document {
    id: str (UUID)
    filename: str
    content_type: str  # "application/pdf", "text/plain", etc.
    file_size: int     # bytes, max 500MB
    upload_timestamp: datetime
    processing_status: DocumentStatus
    extracted_text: str
    batch_id: str      # Reference to ProcessingBatch

    # Validation Rules
    file_size <= 500_000_000  # 500MB limit
    content_type in ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "text/plain", "text/markdown"]
}

enum DocumentStatus {
    UPLOADED,
    VALIDATING,
    QUEUED,
    PROCESSING,
    COMPLETED,
    FAILED,
    EXPIRED
}
```

### ProcessingBatch
**Purpose**: Groups documents for batch processing operations
**Storage**: Redis with 30-day TTL, summary in Neo4j
```python
ProcessingBatch {
    batch_id: str (UUID)
    document_count: int
    total_size_bytes: int
    status: BatchStatus
    created_timestamp: datetime
    started_timestamp: datetime?
    completed_timestamp: datetime?
    progress_percentage: float  # 0.0 to 100.0

    # Validation Rules
    document_count <= 1000      # Max 1000 files per batch
    total_size_bytes <= 500_000_000_000  # 500GB total batch limit
}

enum BatchStatus {
    CREATED,
    QUEUED,
    PROCESSING,
    COMPLETED,
    FAILED,
    CAPACITY_LIMITED
}
```

### ThoughtSeed
**Purpose**: Autonomous cognitive units processing through hierarchical layers
**Storage**: Redis with 24-hour TTL for active states, permanent records in Neo4j
```python
ThoughtSeed {
    thoughtseed_id: str (UUID)
    document_id: str
    type: ThoughtSeedType
    layer: int          # 1-5 for hierarchical processing
    activation_level: float  # 0.0 to 1.0
    consciousness_score: float  # 0.0 to 1.0

    # Neuronal packets
    neuronal_packets: List[NeuronalPacket]

    # Processing timestamps
    created_timestamp: datetime
    last_updated: datetime

    # Hierarchical relationships
    parent_thoughtseed_id: str?
    child_thoughtseed_ids: List[str]
}

enum ThoughtSeedType {
    SENSORIMOTOR,    # Layer 1: Raw sensory processing
    PERCEPTUAL,      # Layer 2: Pattern recognition
    CONCEPTUAL,      # Layer 3: Concept extraction
    ABSTRACT,        # Layer 4: Abstract reasoning
    METACOGNITIVE    # Layer 5: Meta-awareness
}
```

### NeuronalPacket
**Purpose**: Discrete processing units within ThoughtSeeds
**Storage**: Redis with 24-hour TTL
```python
NeuronalPacket {
    packet_id: str (UUID)
    thoughtseed_id: str
    content: str
    activation_level: float      # 0.0 to 1.0
    prediction_error: float      # Bayesian surprise
    surprise_value: float        # Information theoretic surprise
    target_thoughtseeds: List[str]

    # Processing metadata
    processing_timestamp: datetime
    layer_context: dict         # Layer-specific processing context
}
```

### AttractorBasin
**Purpose**: Cognitive landscape regions that capture similar concepts
**Storage**: Redis (7-day TTL) for active basins, Neo4j for persistent structure
```python
AttractorBasin {
    basin_id: str (UUID)
    center_concept: str
    center_vector: List[float]   # 384-dimensional semantic vector
    strength: float              # 0.0 to 1.0
    radius: float               # Influence radius in vector space
    influence_type: BasinInfluenceType

    # Associated entities
    thoughtseed_ids: Set[str]
    related_concepts: Dict[str, float]  # concept -> similarity score

    # Temporal tracking
    formation_timestamp: datetime
    last_modification: datetime
    activation_history: List[float]

    # Mathematical foundation
    phi_function_params: dict    # Parameters for φ_i(x) = σ_i · exp(-||x - c_i||² / (2r_i²))
}

enum BasinInfluenceType {
    REINFORCEMENT,  # Strengthens existing basin (similarity > 0.8)
    COMPETITION,    # Competes with existing basin (0.5 < similarity < 0.8, strong basin)
    SYNTHESIS,      # Merges with existing basin (0.5 < similarity < 0.8, weak basin)
    EMERGENCE       # Creates new basin (similarity < 0.5)
}
```

### NeuralField
**Purpose**: Continuous mathematical fields governing cognitive dynamics
**Storage**: Redis for field states, Neo4j for field topology
```python
NeuralField {
    field_id: str (UUID)
    field_type: NeuralFieldType
    state_vectors: List[complex]  # Complex-valued field state

    # PDE parameters for ∂ψ/∂t = i(∇²ψ + α|ψ|²ψ)
    alpha_parameter: float
    diffusion_coefficient: float

    # Pullback attractors
    pullback_attractors: List[PullbackAttractor]

    # Temporal evolution
    evolution_timestamp: datetime
    field_coherence: float       # 0.0 to 1.0
    energy_level: float
}

enum NeuralFieldType {
    COGNITIVE_LANDSCAPE,
    SEMANTIC_FIELD,
    CONSCIOUSNESS_FIELD,
    MEMORY_CONSOLIDATION_FIELD
}

PullbackAttractor {
    attractor_id: str
    position: List[float]        # Position in field space
    attraction_strength: float
    basin_of_attraction: List[float]  # Geometric boundaries
}
```

### ConsciousnessState
**Purpose**: Measured consciousness levels and detection patterns
**Storage**: Neo4j for analysis, Redis for real-time states
```python
ConsciousnessState {
    state_id: str (UUID)
    thoughtseed_id: str
    consciousness_level: float   # 0.0 to 1.0
    detection_confidence: float  # 0.0 to 1.0

    # Meta-cognitive indicators
    meta_awareness_score: float
    introspective_depth: float
    self_reflection_capacity: float

    # Active inference metrics
    free_energy: float
    complexity_cost: float
    accuracy_reward: float
    prediction_quality: float

    # Measurement context
    measurement_timestamp: datetime
    measurement_method: str      # "active_inference", "meta_tot", etc.
}
```

### MemoryFormation
**Purpose**: Different types of memory created during processing
**Storage**: Multi-timescale storage strategy
```python
MemoryFormation {
    memory_id: str (UUID)
    memory_type: MemoryType
    content: str
    formation_timestamp: datetime
    retrieval_count: int
    last_accessed: datetime

    # Memory-specific attributes
    consolidation_level: float   # 0.0 to 1.0
    forgetting_curve_params: dict
    associative_links: List[str] # Links to other memories

    # Context information
    source_document_id: str?
    thoughtseed_context: List[str]
    attractor_influences: List[str]
}

enum MemoryType {
    WORKING,      # Seconds to minutes, high volatility
    EPISODIC,     # Hours to days, event-based
    SEMANTIC,     # Persistent, fact-based
    PROCEDURAL    # Persistent, skill-based
}
```

### KnowledgeTriple
**Purpose**: Subject-predicate-object relationships from AutoSchemaKG
**Storage**: Neo4j as graph relationships
```python
KnowledgeTriple {
    triple_id: str (UUID)
    subject: str
    predicate: str
    object: str
    confidence_score: float      # 0.0 to 1.0

    # Extraction metadata
    source_document_id: str
    extraction_method: str       # "autoschema_kg", "manual", etc.
    extraction_timestamp: datetime

    # Graph context
    subject_entity_id: str?      # Link to Neo4j entity
    object_entity_id: str?       # Link to Neo4j entity
    relationship_weight: float   # Graph edge weight
}
```

## Specialized Entities

### EvolutionaryPrior
**Purpose**: Hierarchical priors for Bayesian processing
**Storage**: Redis with procedural memory TTL
```python
EvolutionaryPrior {
    prior_id: str (UUID)
    prior_type: PriorType
    activation_threshold: float
    context_relevance: float
    learning_rate: float

    # Hierarchical structure
    parent_prior_id: str?
    confidence_level: float
    update_frequency: timedelta
}

enum PriorType {
    BASAL,              # Basic survival/processing priors
    LINEAGE_SPECIFIC,   # Domain-specific learned patterns
    DISPOSITIONAL,      # Individual processing preferences
    LEARNED            # Recently acquired patterns
}
```

### AttractorType
**Purpose**: Specialized attractor implementations for research integration
**Storage**: Configuration in Neo4j, active states in Redis
```python
AttractorType {
    type_id: str
    name: str           # concept_extractor, semantic_analyzer, etc.
    research_origin: str # "MIT_MEM1", "IBM_Zurich", "Shanghai_AI_Lab"

    # Processing capabilities
    processing_function: str     # Function signature for processing
    input_requirements: dict     # Required input data structure
    output_format: dict         # Expected output data structure

    # Integration parameters
    cross_resonance_enabled: bool
    harmonic_interaction_params: dict
}
```

## Data Relationships

### Document → Processing Flow
```
Document (1) → (1) ProcessingBatch
Document (1) → (1..*) ThoughtSeed
ThoughtSeed (1) → (1..*) NeuronalPacket
ThoughtSeed (1..*) → (1..*) AttractorBasin
AttractorBasin (1..*) → (1..*) NeuralField
```

### Memory Integration Flow
```
ThoughtSeed (1) → (0..1) ConsciousnessState
ThoughtSeed (1..*) → (1..*) MemoryFormation
MemoryFormation (1..*) → (1..*) KnowledgeTriple
```

### Temporal Evolution
```
AttractorBasin → NeuralField (dynamic transformation)
NeuralField → PullbackAttractor (field dynamics)
ConsciousnessState → MemoryFormation (consolidation)
```

## Validation Rules

### Batch Processing Constraints
- Max 1000 documents per batch (clarified)
- Max 500MB per document (clarified)
- Total batch size should not exceed reasonable processing capacity
- Queue-based processing when capacity reached (clarified)

### TTL Management (Redis)
- NeuronalPackets: 24 hours (clarified)
- AttractorBasins: 7 days (clarified)
- ProcessingResults: 30 days (clarified)

### Vector Dimensions
- All semantic vectors: 384 dimensions (consistent with research)
- Attractor center vectors: 384 dimensions
- Neural field state vectors: variable based on field resolution

### Consciousness Thresholds
- Consciousness detection threshold: 0.3 (based on research)
- Meta-awareness minimum: 0.2
- Basin influence calculation thresholds as specified in mathematical foundation

## State Transitions

### Document Processing States
```
UPLOADED → VALIDATING → QUEUED → PROCESSING → COMPLETED
                      ↓
                   FAILED (validation/processing errors)
                      ↓
                   EXPIRED (TTL exceeded)
```

### ThoughtSeed Layer Progression
```
SENSORIMOTOR → PERCEPTUAL → CONCEPTUAL → ABSTRACT → METACOGNITIVE
(Layer 1)      (Layer 2)     (Layer 3)    (Layer 4)   (Layer 5)
```

### Basin Influence Evolution
```
New ThoughtSeed → Similarity Calculation → Influence Type Determination
                                        ↓
REINFORCEMENT (>0.8) | COMPETITION (0.5-0.8, strong) | SYNTHESIS (0.5-0.8, weak) | EMERGENCE (<0.5)
                                        ↓
                              Basin State Update → Neural Field Modification
```

This data model supports the complete ThoughtSeed pipeline from document upload through consciousness processing to knowledge graph storage, with proper validation rules and state management for research-grade processing at scale.