# Data Model: ASI-GO-2 Research Intelligence System

**Date**: 2025-09-26
**Feature**: Remove ASI-Arch and Integrate ASI-GO-2

## Core Entities

### CognitionBase
**Purpose**: Stores accumulated problem-solving patterns and research strategies
**Fields**:
- `pattern_id`: UUID, unique identifier
- `pattern_name`: String, human-readable pattern name
- `description`: Text, detailed pattern description
- `success_rate`: Float (0.0-1.0), effectiveness metric
- `confidence`: Float (0.0-1.0), reliability metric
- `domain_tags`: Array[String], knowledge domains where pattern applies
- `thoughtseed_layer`: Enum (sensory, perceptual, conceptual, abstract, metacognitive)
- `attractor_basin_id`: String, Context Engineering basin association
- `creation_timestamp`: DateTime
- `last_used`: DateTime
- `usage_count`: Integer

**Relationships**:
- One-to-many with ResearchPattern
- Many-to-many with DocumentSource (patterns extracted from documents)
- One-to-many with ThoughtseedWorkspace (pattern competition results)

**State Transitions**:
- Created → Active → Refined → Archived
- Validation Rules: success_rate and confidence must be 0.0-1.0

### ResearchPattern
**Purpose**: Specific instantiation of a cognition pattern for research tasks
**Fields**:
- `research_pattern_id`: UUID, unique identifier
- `cognition_pattern_id`: UUID, foreign key to CognitionBase
- `research_query`: Text, original research question
- `applied_context`: JSON, specific context where pattern was applied
- `outcome_quality`: Float (0.0-1.0), quality of research synthesis result
- `adaptation_notes`: Text, how pattern was modified for this context
- `active_inference_trace`: JSON, prediction error and adaptation history
- `narrative_elements`: Array[String], detected narrative/motif patterns
- `timestamp`: DateTime

**Relationships**:
- Many-to-one with CognitionBase
- One-to-many with ThoughtseedWorkspace
- Many-to-many with DocumentSource

### ThoughtseedWorkspace
**Purpose**: Inner workspace where pattern competition and thoughtseed processing occurs
**Fields**:
- `workspace_id`: UUID, unique identifier
- `research_query`: Text, question being processed
- `competing_patterns`: Array[UUID], CognitionBase pattern IDs in competition
- `winning_pattern`: UUID, selected optimal pattern
- `competition_trace`: JSON, detailed competition dynamics and scores
- `consciousness_level`: Float (0.0-1.0), measured consciousness emergence
- `neural_field_state`: JSON, Context Engineering neural field configuration
- `attractor_modifications`: Array[String], basin changes triggered
- `thoughtseed_layers_activated`: Array[Enum], which 5 layers were engaged
- `start_time`: DateTime
- `completion_time`: DateTime
- `energy_levels`: JSON, per-thought energy states during competition

**Relationships**:
- Many-to-one with ResearchPattern
- One-to-many with ThoughtseedTrace
- Many-to-one with AttractorBasin

### DocumentSource
**Purpose**: Documents processed by the system that contribute to pattern learning
**Fields**:
- `document_id`: UUID, unique identifier
- `filename`: String, original document name
- `content_hash`: String, SHA-256 of document content
- `processing_timestamp`: DateTime
- `extraction_quality`: Float (0.0-1.0), quality of pattern extraction
- `patterns_extracted`: Array[UUID], CognitionBase patterns derived
- `narrative_elements`: JSON, detected stories, motifs, themes
- `thoughtseed_traces`: Array[UUID], processing trace IDs
- `metadata`: JSON, additional document properties

**Relationships**:
- Many-to-many with CognitionBase (patterns extracted)
- One-to-many with ThoughtseedTrace
- Many-to-many with ResearchPattern

### ThoughtseedTrace
**Purpose**: Detailed trace of 5-layer thoughtseed processing for analysis and learning
**Fields**:
- `trace_id`: UUID, unique identifier
- `layer`: Enum (sensory, perceptual, conceptual, abstract, metacognitive)
- `input_content`: Text, content processed at this layer
- `processing_result`: JSON, layer-specific processing outcome
- `attention_focus`: String, what the layer focused on
- `surprise_level`: Float (0.0-1.0), unexpected pattern detection
- `energy_level`: Float (0.0-1.0), processing intensity
- `connections_formed`: Array[String], links to other traces/patterns
- `timestamp`: DateTime
- `parent_workspace_id`: UUID, foreign key to ThoughtseedWorkspace

**Relationships**:
- Many-to-one with ThoughtseedWorkspace
- Many-to-one with DocumentSource

### AttractorBasin
**Purpose**: Context Engineering stable states for knowledge clustering
**Fields**:
- `basin_id`: UUID, unique identifier
- `basin_name`: String, descriptive name
- `knowledge_domain`: String, subject area (neuroscience, AI, psychology, etc.)
- `stability_level`: Float (0.0-1.0), resistance to perturbation
- `pattern_cluster`: Array[UUID], CognitionBase patterns in this basin
- `neural_field_parameters`: JSON, field configuration
- `emergence_conditions`: JSON, conditions that activate this basin
- `modification_history`: Array[JSON], changes over time
- `last_activation`: DateTime

**Relationships**:
- One-to-many with CognitionBase (patterns clustered in basin)
- One-to-many with ThoughtseedWorkspace (workspaces that modify basin)

### ResearchQuery
**Purpose**: User research questions and system responses
**Fields**:
- `query_id`: UUID, unique identifier
- `user_query`: Text, original question
- `processed_query`: Text, normalized/enhanced version
- `selected_patterns`: Array[UUID], patterns used for response
- `synthesis_response`: Text, generated research answer
- `confidence_score`: Float (0.0-1.0), system confidence in response
- `thoughtseed_workspace_id`: UUID, workspace used for processing
- `processing_time_ms`: Integer, response time
- `user_feedback`: Enum (helpful, not_helpful, partially_helpful)
- `timestamp`: DateTime

**Relationships**:
- One-to-one with ThoughtseedWorkspace
- Many-to-many with CognitionBase (patterns used)

## Schema Validation Rules

### CognitionBase Validation
- `success_rate` and `confidence` must be between 0.0 and 1.0
- `pattern_name` must be unique within the system
- `thoughtseed_layer` must be valid enum value
- `domain_tags` must contain at least one tag

### ThoughtseedWorkspace Validation
- `completion_time` must be after `start_time`
- `winning_pattern` must be in `competing_patterns` array
- `consciousness_level` must be between 0.0 and 1.0
- `thoughtseed_layers_activated` must contain at least one layer

### ResearchQuery Validation
- `confidence_score` must be between 0.0 and 1.0
- `processing_time_ms` must be positive integer
- `selected_patterns` must reference existing CognitionBase entries

## Data Relationships Graph

```
DocumentSource ---> ThoughtseedTrace
      |                    |
      |                    v
      +----------> ThoughtseedWorkspace
                          |
ResearchQuery ----------->+
      |                  |
      |                  v
      +----------> CognitionBase <---> AttractorBasin
                          |
                          v
                  ResearchPattern
```

## Hybrid Database Architecture

**AutoSchemaKG Integration**:
- Auto-generated knowledge graph schema based on document content
- Dynamic schema evolution as new document types are processed
- Entity and relationship extraction using local OLLAMA models

**Vector + Graph Hybrid Storage**:
- **Vector Database**: Semantic similarity search using embeddings
- **Graph Database**: Entity relationships and knowledge connections
- **Linking Strategy**: Vector similarity results linked to graph relationships
- **Query Flow**: Vector search finds semantically similar content → Graph traversal finds related entities and relationships

**Port Assignments**:
- Neo4j (Graph): 7474 (HTTP), 7687 (Bolt)
- Qdrant (Vector): 6333 (HTTP API), 6334 (gRPC)
- Redis (Cache): 6379
- ASI-GO-2 API: 8001
- OLLAMA Server: 11434
- Frontend (if needed): 3000

**Local OLLAMA Configuration**:
- All LLM processing uses local OLLAMA instance
- No external API calls for privacy and cost control
- Models: llama3.1, codellama, mistral for different tasks
- Embedding generation through OLLAMA's embedding models

## Migration from ASI-Arch

**Strategy**: Complete replacement - no data migration
- ASI-Arch database schemas will be dropped
- ASI-GO-2 starts with clean slate using AutoSchemaKG
- Pattern accumulation begins with first document processing
- Hybrid vector+graph storage enables rich semantic and relational queries
- Local OLLAMA ensures privacy-preserving processing