# Spec-022: Ultra-Granular Document Processing System
**Version**: 1.0.0  
**Date**: 2025-09-30  
**Status**: Draft  
**Priority**: High  

## Overview

Implementation of an ultra-granular document processing system that uses consciousness-guided analysis to extract atomic concepts, relationships, composite concepts, contextual frameworks, and narrative structures from multi-format documents. The system processes neuroscience and AI domain documents with real-time visualization and decades-long memory persistence.

## Functional Requirements

### FR-022-001: Multi-Format Document Ingestion
**Priority**: Must Have  
**Description**: System must ingest and process documents in multiple formats
**Acceptance Criteria**:
- Support PDF, text, Word documents, web links, audio, YouTube, PowerPoint
- Phase 1: PDF, text files, web links (simple content)
- Phase 2: Word documents, PowerPoint slides, complex PDFs
- Phase 3: Audio files, YouTube videos, multimedia content
- Auto-detection of document type and routing to appropriate processor
- Error handling for unsupported formats with clear user feedback

### FR-022-002: Ultra-Granular Content Chunking
**Priority**: Must Have  
**Description**: Break documents into consciousness-digestible chunks at multiple granularity levels
**Acceptance Criteria**:
- Level 1: Sentence-level chunks for atomic concept extraction
- Level 2: 2-3 sentence chunks for relationship mapping
- Level 3: Paragraph-level chunks for composite concepts
- Level 4: Section-level chunks for contextual frameworks
- Level 5: Multi-section chunks for narrative analysis
- Configurable overlap between chunks (default: 100 tokens)
- Preserve document structure and metadata in chunks

### FR-022-003: Ollama-Powered Consciousness Processing
**Priority**: Must Have  
**Description**: Use local Ollama models for concept extraction and analysis
**Acceptance Criteria**:
- Primary model: qwen2.5:14b for concept analysis and reasoning
- Embedding model: nomic-embed-text for local vector generation
- Backup model: llama3.1:8b for faster simple extractions
- Auto-fallback between models based on task complexity
- Real-time processing with visible progress indicators
- No external API dependencies for core functionality

### FR-022-004: Neuroscience/AI Domain Specialization
**Priority**: Must Have  
**Description**: Optimized processing for neuroscience and AI domain knowledge
**Acceptance Criteria**:
- Pre-loaded neuroscience and AI terminology databases
- Domain-specific concept extraction prompts
- Cross-domain mapping between neuroscience ↔ AI concepts
- Specialized ThoughtSeed domains for scientific content
- Academic paper structure recognition and processing
- Citation and reference relationship tracking

### FR-022-005: Five-Level Concept Extraction
**Priority**: Must Have  
**Description**: Extract knowledge at five distinct granularity levels
**Acceptance Criteria**:
- **Atomic Concepts**: Individual terms, definitions, entities
- **Concept Relationships**: Connections, causality, dependencies
- **Composite Concepts**: Complex multi-part ideas and systems
- **Contextual Frameworks**: Domain paradigms, theoretical contexts
- **Narrative Structures**: Argument flows, story progressions, methodologies
- Each level triggers separate consciousness processing cycles
- Progressive building from atomic to narrative levels

### FR-022-006: Real-Time Visualization
**Priority**: Must Have  
**Description**: Live visualization of concept extraction and consciousness processing
**Acceptance Criteria**:
- Real-time concept discovery animation in ThoughtSeed Monitor
- Live graph updates showing relationship formation
- Progress indicators for each granularity level
- Visual representation of consciousness cycles per chunk
- Interactive exploration of extracted knowledge structures
- Document processing timeline with concept emergence events

### FR-022-007: Context-Aware Processing
**Priority**: Must Have  
**Description**: Maintain context between documents and processing sessions
**Acceptance Criteria**:
- Preserve active context from recently processed documents
- Cross-document concept relationship detection
- Cumulative domain knowledge building
- Session continuity across document uploads
- Context inheritance for related documents
- Smart context pruning to prevent information overload

### FR-022-008: Decades-Long Memory Persistence
**Priority**: Must Have  
**Description**: Store and maintain extracted knowledge for decades
**Acceptance Criteria**:
- **Hot Memory**: Redis for current session + 24 hours
- **Warm Memory**: Neo4j knowledge graph for months/years
- **Cold Memory**: Vector database + file system for decades
- Hierarchical memory compression every 6 months
- Memory retrieval by time, domain, or concept
- Graceful degradation for memory system failures

### FR-022-009: AutoSchemaKG Knowledge Graph Integration
**Priority**: Should Have  
**Description**: Automatic knowledge graph construction from extracted concepts
**Acceptance Criteria**:
- Automatic triple extraction from processed content
- Dynamic schema generation for neuroscience/AI domains
- Integration with Neo4j database backend
- Knowledge graph evolution over multiple documents
- Concept hierarchy and relationship visualization
- Export capabilities for knowledge graphs

### FR-022-010: Selective OpenAI Enhancement
**Priority**: Could Have  
**Description**: Optional OpenAI integration for complex specialized tasks
**Acceptance Criteria**:
- Configurable OpenAI usage for 5% of processing tasks
- Selective enhancement for complex neuroscience terminology
- Cost optimization: ~$20/month budget instead of $975
- Quality validation of Ollama extractions
- Academic paper deep analysis mode
- Fallback to Ollama if OpenAI unavailable

## Technical Requirements

### TR-022-001: Ollama Model Management
**Description**: Efficient management of local Ollama models
**Implementation**:
- Automatic model downloads and updates
- Model health monitoring and restart capabilities
- Dynamic model selection based on task requirements
- Memory usage optimization for concurrent model access
- Model performance monitoring and fallback logic

### TR-022-002: Multi-Level ThoughtSeed Processing
**Description**: Enhanced ThoughtSeed system for granular concept analysis
**Implementation**:
- Extended ThoughtSeed domains for concept extraction types
- Granular river flow states for different processing phases
- Consciousness cycle management for multi-level processing
- Memory integration for each processing level
- Performance optimization for high-frequency processing

### TR-022-003: Hybrid Database Architecture
**Description**: Multi-tier storage system for different memory types
**Implementation**:
- Redis: Hot memory for active processing state
- Neo4j: Knowledge graph for semantic relationships
- Vector database: Embeddings for similarity search
- File system: Raw document storage and archival
- Backup and recovery for all storage tiers

### TR-022-004: Document Processing Pipeline
**Description**: Scalable pipeline for multi-format document processing
**Implementation**:
- Async processing with queue management
- Document type detection and routing
- Content extraction with format-specific processors
- Error handling and retry logic
- Progress tracking and user notifications

### TR-022-005: Real-Time WebSocket Updates
**Description**: Live updates for visualization components
**Implementation**:
- WebSocket connections for real-time data streaming
- Event-driven updates for concept discovery
- Efficient data serialization for large knowledge graphs
- Client-side state management for visualization
- Connection resilience and reconnection logic

### TR-022-006: Extraction Adapter Implementations
**Description**: Implement production-grade adapters for every document extraction path (text, structured data, algorithm detection, knowledge graph triples).
**Implementation**:
- `_extract_text_pymupdf` MUST use PyMuPDF to return raw text, metadata, and page spans for PDFs; unit tests MUST assert non-empty output for fixture documents.
- `_extract_structured_langextract` MUST invoke LangExtract pipelines (or documented local equivalent) to return structured sections, tables, and figures; tests MUST fail if placeholder payloads (e.g., `{ "status": "placeholder" }`) are returned.
- `_extract_algorithms` MUST identify algorithm blocks/code listings using the configured extractor and emit typed records (name, complexity, pseudocode snippet); regression tests cover known algorithm fixtures.
- `_extract_knowledge_graph` MUST call KGGen (or the specified auto-schema builder) to emit entities and relationships compatible with AutoSchemaKG; contract tests MUST validate schema conformity before ingestion.
- All adapters MUST surface actionable error messages when upstream tools are unavailable and provide deterministic fallbacks that still satisfy TDD assertions (no silent passes).

## Performance Requirements

### PR-022-001: Processing Speed
- Sentence-level processing: < 2 seconds per sentence
- Document ingestion: < 30 seconds per page
- Real-time visualization updates: < 100ms latency
- Memory query response: < 500ms for recent documents
- Knowledge graph queries: < 1 second for complex traversals

### PR-022-002: Memory Usage
- Ollama models: < 16GB RAM total
- Redis hot memory: < 4GB for active session
- Vector database: Scalable to 100GB+ for decades storage
- Neo4j knowledge graph: < 8GB for active domain knowledge
- Frontend visualization: < 1GB for complex graph rendering

### PR-022-003: Scalability
- Process documents up to 100 pages efficiently
- Support 1000+ documents in knowledge base
- Handle 10+ concurrent processing sessions
- Scale to decades of accumulated knowledge
- Maintain performance with growing knowledge graphs

## Security Requirements

### SR-022-001: Data Privacy
- All processing performed locally (Ollama-based)
- No document content sent to external APIs by default
- Optional OpenAI usage with explicit user consent
- Secure storage of processed knowledge and documents
- User control over data retention and deletion

### SR-022-002: System Security
- Secure file upload handling with validation
- Protection against malicious document content
- Access control for knowledge base and memories
- Audit logging for document processing activities
- Backup encryption for long-term storage

## Integration Requirements

### IR-022-001: Consciousness System Integration
- Full integration with existing consciousness pipeline
- ThoughtSeed competition for each processing level
- Context engineering for document flow analysis
- MAC theory application for concept evaluation
- IWMT consciousness detection for knowledge emergence

### IR-022-002: Frontend Integration
- Enhanced ThoughtSeed Monitor for document processing
- New document upload interface with progress tracking
- Knowledge graph visualization components
- Memory exploration and search interfaces
- Real-time processing status and analytics

### IR-022-003: Database Integration
- Unified database system compatibility
- AutoSchemaKG integration for knowledge graphs
- Vector database for embedding storage
- Redis for session and cache management
- Neo4j for semantic knowledge representation

## Success Criteria

1. **Successful multi-format document processing** with 95% accuracy for supported formats
2. **Real-time concept extraction** visible in consciousness interface
3. **Knowledge persistence** for decades with efficient retrieval
4. **Domain expertise** demonstrable in neuroscience/AI content processing
5. **Cost efficiency** with 95% local processing, 5% optional cloud enhancement
6. **User experience** enabling intuitive exploration of extracted knowledge

## Implementation Phases

### Phase 1: Foundation (Weeks 1-2)
- Basic document ingestion (PDF, text, web)
- Ollama model integration and management
- Ultra-granular chunking implementation
- Basic concept extraction with consciousness

### Phase 2: Knowledge Systems (Weeks 3-4)
- Five-level concept extraction implementation
- AutoSchemaKG integration and knowledge graphs
- Multi-tier memory persistence system
- Real-time visualization enhancements

### Phase 3: Domain Specialization (Weeks 5-6)
- Neuroscience/AI domain optimization
- Selective OpenAI integration
- Advanced narrative structure analysis
- Performance optimization and scaling

### Phase 4: Advanced Features (Weeks 7-8)
- Multimedia document support (audio, video)
- Cross-document relationship analysis
- Memory compression and archival systems
- User interface refinements and analytics

## Risk Assessment

### High Risk
- **Ollama model performance** for complex neuroscience content
- **Memory system complexity** for decades-long persistence
- **Real-time processing** performance with ultra-granular chunks

### Medium Risk
- **AutoSchemaKG integration** complexity and reliability
- **Knowledge graph scalability** with growing document corpus
- **Frontend performance** with large visualization datasets

### Low Risk
- **Document format support** with established libraries
- **Database integration** with existing infrastructure
- **Basic concept extraction** with proven NLP techniques

## Dependencies

- **Ollama**: qwen2.5:14b, llama3.1:8b, nomic-embed-text models
- **AutoSchemaKG**: Knowledge graph construction framework
- **Neo4j**: Graph database for semantic relationships
- **Redis**: In-memory storage for hot memory
- **Vector Database**: Qdrant or similar for embeddings
- **LangChain**: Document processing and text splitting
- **Existing Consciousness System**: ThoughtSeed competition and processing

---

*This specification defines the ultra-granular document processing system that enables consciousness-guided knowledge extraction with real-time visualization and decades-long memory persistence for neuroscience and AI domain expertise.*
