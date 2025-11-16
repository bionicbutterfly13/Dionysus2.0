# Document Processing

## Overview

The Document Processing capability enables Dionysus to receive, analyze, and persistently store documents with consciousness-enhanced processing. Documents flow through a multi-stage pipeline: perceptual gateway → LangGraph workflow → consciousness processing → Neo4j persistence. The system extracts five-level concepts, generates attractor basins, creates thoughtseeds, and stores all artifacts with relationships for later retrieval and analysis.

**Key Value**: Transform uploaded documents into a searchable, interconnected knowledge graph with consciousness-enhanced insights, enabling researchers to discover deep conceptual relationships and explore questions generated during processing.

---

## Purpose

Enable consciousness-enhanced document processing by transforming uploaded content into interconnected knowledge graphs with extracted concepts, attractor basins, and research questions, providing researchers with deep semantic relationships and curiosity-driven exploration capabilities.

---

## Requirements

### Requirement: Perceptual Gateway

The system MUST provide a clean, single-responsibility gateway for receiving perceptual information from external sources (uploaded files, URLs, APIs).

#### Scenario: File Upload Reception
**Given** a user uploads a PDF document to the system
**When** the upload completes
**Then** the Daedalus perceptual gateway receives the file data without modification or transformation

#### Scenario: Multiple File Types
**Given** users can upload various file formats (PDF, text, markdown, HTML)
**When** any supported file type is uploaded
**Then** Daedalus accepts and processes it through the LangGraph workflow

#### Scenario: Gateway Simplicity
**Given** the Daedalus class exists
**When** inspecting its public interface
**Then** it exposes exactly one primary method: `receive_perceptual_information()`

---

### Requirement: LangGraph Processing Workflow

The system MUST process documents through a structured 6-node LangGraph workflow integrating SurfSense extraction, ASI-GO-2 refinement, and consciousness processing.

#### Scenario: Extract and Process
**Given** a document has been received by Daedalus
**When** the LangGraph workflow begins
**Then** the system extracts text, computes content hash, converts to markdown, and performs initial concept extraction

#### Scenario: Research Question Generation
**Given** initial concepts have been extracted
**When** the research planning node executes
**Then** the system generates curiosity-driven questions using ASI-GO-2 Researcher and R-Zero patterns

#### Scenario: Consciousness Processing
**Given** research questions have been generated
**When** the consciousness processing node executes
**Then** the system creates attractor basins, generates thoughtseeds, and tracks neural field resonance patterns

#### Scenario: Quality Analysis
**Given** consciousness processing has completed
**When** the analysis node executes
**Then** the system computes quality scores (coherence, novelty, depth) and determines if refinement is needed

#### Scenario: Iterative Refinement
**Given** quality scores are below the threshold (default: 0.7) and iterations remain
**When** the decision node evaluates completion criteria
**Then** the workflow loops back to consciousness processing with refined parameters

#### Scenario: Finalization
**Given** quality threshold is met or max iterations (default: 3) are reached
**When** the finalization node executes
**Then** the system packages complete results including concepts, basins, thoughtseeds, quality metrics, and research insights

---

### Requirement: Five-Level Concept Extraction

The system MUST extract and classify concepts across five hierarchical levels, from atomic units to narrative structures.

#### Scenario: Atomic Concepts
**Given** a document contains fundamental terms and entities
**When** concept extraction runs
**Then** the system identifies atomic concepts (single words, named entities) with salience scores

#### Scenario: Relationship Concepts
**Given** atomic concepts exist in the document
**When** relationship extraction runs
**Then** the system identifies connections between concepts (e.g., "climate" → "affects" → "agriculture")

#### Scenario: Composite Concepts
**Given** multiple atomic concepts co-occur
**When** composite extraction runs
**Then** the system creates higher-level concepts from atomic components (e.g., "machine learning" from "machine" + "learning")

#### Scenario: Context Concepts
**Given** concepts exist within a domain
**When** context extraction runs
**Then** the system assigns domain/era context (e.g., "neural networks" → domain: "artificial intelligence")

#### Scenario: Narrative Concepts
**Given** the document has a cohesive structure
**When** narrative extraction runs
**Then** the system captures storyline and thematic flow across the entire document

---

### Requirement: Document Persistence

The system MUST persistently store all document processing artifacts in Neo4j via the constitutional Graph Channel interface, with performance targets and duplicate detection.

#### Scenario: Document Storage
**Given** the LangGraph workflow has completed
**When** persistence begins
**Then** the system creates a Document node with metadata (filename, content_hash, upload_timestamp, quality scores, tags, tier classification)

#### Scenario: Content Hash Computation
**Given** a document is ready for persistence
**When** no content_hash is provided
**Then** the system computes SHA-256 hash from document_body + namespace and validates the 64-character hex format

#### Scenario: Duplicate Detection
**Given** a document with a specific content_hash
**When** checking for duplicates before persistence
**Then** the system queries existing documents by content_hash and rejects duplicates with a structured error pointing to the canonical record

#### Scenario: LLM Summary Generation
**Given** a document is being persisted
**When** the DocumentSummarizer is available
**Then** the system generates a token-aware LLM summary (max 150 tokens, temperature 0.3) and stores it with the document

#### Scenario: Concept Persistence
**Given** five-level concepts have been extracted
**When** persistence runs
**Then** the system creates Concept nodes for all levels and links them to the Document via EXTRACTED_FROM relationships

#### Scenario: Basin Persistence
**Given** attractor basins have been created during consciousness processing
**When** persistence runs
**Then** the system creates/updates AttractorBasin nodes with depth, stability, strength and links them via ATTRACTED_TO relationships with activation strength

#### Scenario: ThoughtSeed Persistence
**Given** thoughtseeds have been generated
**When** persistence runs
**Then** the system creates ThoughtSeed nodes with content, germination_potential, resonance_score, and field_resonance data, linked via GERMINATED_FROM relationships

#### Scenario: Performance Target - Persistence
**Given** a typical document (1-5 MB PDF)
**When** full persistence completes
**Then** the operation takes less than 2000 milliseconds

#### Scenario: Atomic Transactions
**Given** multiple nodes and relationships are being created
**When** any operation fails mid-persistence
**Then** the system rolls back all changes to prevent partial writes

---

### Requirement: Document Retrieval

The system MUST provide fast, paginated, filterable document listing and detailed retrieval with all linked processing artifacts.

#### Scenario: Document Listing
**Given** 150 documents have been persisted
**When** a user requests the document list
**Then** the system returns 50 documents per page with filename, upload_timestamp, quality_overall, tags, tier, and artifact counts

#### Scenario: Filtering by Tags
**Given** documents are tagged with categories
**When** filtering by tag "neuroscience"
**Then** the system returns only documents where "neuroscience" exists in the tags array

#### Scenario: Filtering by Quality
**Given** documents have quality scores
**When** filtering with quality_min=0.8
**Then** the system returns only documents with quality_overall >= 0.8

#### Scenario: Filtering by Date Range
**Given** documents have upload timestamps
**When** filtering with date_from and date_to
**Then** the system returns only documents within that date range (inclusive)

#### Scenario: Sorting Options
**Given** documents exist with various metadata
**When** requesting sorted results
**Then** the system supports sorting by upload_date, quality, or curiosity_triggers in ascending or descending order

#### Scenario: Performance Target - Listing
**Given** a database with 10,000 documents
**When** a listing query executes
**Then** the response returns within 500 milliseconds

#### Scenario: Document Detail Retrieval
**Given** a user selects a specific document from the list
**When** requesting document detail by document_id
**Then** the system returns complete metadata, quality scores, all concepts organized by level, all linked basins, all linked thoughtseeds, and summary

#### Scenario: Access Tracking
**Given** a document is retrieved
**When** the detail query completes
**Then** the system increments access_count and updates last_accessed timestamp for tier management

---

### Requirement: URL Ingestion

The system MUST support document ingestion from HTTPS URLs with download, conversion, and chunking for PDF and HTML sources.

#### Scenario: URL Download
**Given** a valid HTTPS URL to a PDF or HTML page
**When** URL ingestion is requested
**Then** the system downloads the content with retry/backoff and validates the MIME type

#### Scenario: PDF Conversion
**Given** a downloaded PDF from a URL
**When** text extraction runs
**Then** the system extracts text from all pages and feeds it to the processing pipeline

#### Scenario: HTML Conversion
**Given** a downloaded HTML page from a URL
**When** text extraction runs
**Then** the system removes script/style tags and extracts clean text content

#### Scenario: Document Chunking
**Given** extracted text from a URL source
**When** chunking runs
**Then** the system splits text using RecursiveCharacterTextSplitter with configurable chunk_size (default: 1000) and overlap (default: 200)

#### Scenario: Chunk Persistence
**Given** chunks have been created
**When** persistence runs
**Then** the system creates Chunk nodes with chunk_id, content, position, start_char, end_char and links them via PART_OF relationships to the Document

#### Scenario: Source Metadata
**Given** a document ingested from a URL
**When** stored in Neo4j
**Then** the system marks source_type="url", stores original_url, infers connector_icon from MIME type, and preserves download_metadata (status_code, redirected_url, download_duration_ms)

---

### Requirement: Tier Management

The system MUST classify documents into warm/cool/cold tiers based on age and access patterns for efficient storage management.

#### Scenario: Initial Tier Assignment
**Given** a new document is persisted
**When** the Document node is created
**Then** the system assigns tier="warm" by default

#### Scenario: Access-Based Tier Adjustment
**Given** a document has been accessed frequently
**When** tier evaluation runs
**Then** frequently accessed documents remain in warm tier regardless of age

#### Scenario: Age-Based Tier Migration
**Given** a document has not been accessed recently
**When** tier evaluation runs
**Then** older rarely-accessed documents migrate to cool tier, and very old unaccessed documents migrate to cold tier

#### Scenario: Cold Tier Archival
**Given** a document has been marked as cold tier
**When** archival runs
**Then** the system moves document content to separate cheaper storage (S3/filesystem), retains metadata and document_id in Neo4j, and accepts slower retrieval times

---

### Requirement: Constitutional Compliance

The system MUST access Neo4j exclusively through the DaedalusGraphChannel interface to maintain architectural boundaries and audit trails.

#### Scenario: Graph Channel Access
**Given** any component needs to read or write to Neo4j
**When** database operations execute
**Then** all operations go through `get_graph_channel()` with no direct neo4j driver imports

#### Scenario: Audit Trail
**Given** a database operation is executed
**When** calling the Graph Channel
**Then** the system includes caller_service and caller_function parameters for audit tracking

#### Scenario: Import Restrictions
**Given** a new module needs database access
**When** reviewing imports
**Then** only `from daedalus_gateway import get_graph_channel` is allowed, no direct `from neo4j import` statements

---

### Requirement: Error Handling and Resilience

The system MUST handle failures gracefully with retries, fallbacks, and structured error responses.

#### Scenario: Corrupted Upload
**Given** an upload contains corrupted or unreadable data
**When** Daedalus processes it
**Then** the system returns a structured error response with descriptive message and does not create incomplete documents

#### Scenario: Database Transient Failures
**Given** Neo4j returns a transient connection error
**When** persistence is attempted
**Then** the system retries up to 3 times with exponential backoff before failing

#### Scenario: LLM Summarizer Unavailable
**Given** the DocumentSummarizer cannot connect to the local LLM
**When** summary generation is attempted
**Then** the system falls back to extractive summarization and continues persistence with summary_metadata indicating the fallback method

#### Scenario: Concurrent Upload Race Condition
**Given** 100+ concurrent upload requests
**When** all are processed
**Then** each completes without data corruption, and duplicate content_hash uploads are rejected cleanly

---

## Key Entities

### Document
Represents an uploaded or URL-ingested file with:
- **Metadata**: filename, content_hash (SHA-256), upload_timestamp, file_size, mime_type, tags
- **Source**: source_type (uploaded_file, url, api), original_url (if applicable), connector_icon, download_metadata
- **Processing Results**: quality scores (overall, coherence, novelty, depth), processing_duration_ms, processing_status
- **Tier Classification**: tier (warm/cool/cold), last_accessed, access_count, tier_changed_at
- **Summary**: LLM-generated or extractive summary with summary_metadata (method, model, tokens_used)
- **Archival**: archive_location, archived_at (for cold tier)

### Concept
Represents an extracted idea or term classified by level:
- **Identity**: concept_id, name, level (atomic/relationship/composite/context/narrative)
- **Scoring**: salience (importance score 0-1)
- **Level-Specific Data**:
  - Atomic: definition
  - Relationship: source_concept, target_concept
  - Composite: components (array of atomic concept_ids)
  - Context: domain, era
  - Narrative: storyline
- **Provenance**: Links to Document via EXTRACTED_FROM with confidence and extraction_method

### AttractorBasin
Represents a stable conceptual domain in consciousness processing:
- **Identity**: basin_id, name
- **Dynamics**: depth (attraction strength), stability (resistance to change), strength (cumulative reinforcement)
- **Evolution**: modification_count, last_modified, associated_concepts
- **Document Links**: ATTRACTED_TO relationships with activation_strength, influence_type (reinforcement/competition/synthesis/emergence), strength_delta

### ThoughtSeed
Represents a question or insight generated during processing:
- **Identity**: seed_id, content (the question or insight text)
- **Potential**: germination_potential (likelihood of leading to new insights), resonance_score (connection to existing knowledge)
- **Neural Field**: field_resonance_energy, field_resonance_phase, field_resonance_pattern (interference patterns)
- **Provenance**: source_stage (which processing node created it), generated_at
- **Document Links**: GERMINATED_FROM relationships with potential and generation_stage

### Chunk
Represents a text segment from URL-ingested documents:
- **Identity**: chunk_id (stable identifier for citation)
- **Content**: content (text of the chunk)
- **Position**: position (sequence index), start_char, end_char (character offsets in original document)
- **Document Links**: PART_OF relationships with chunk_index

---

## Non-Functional Requirements

- **Performance**:
  - Document persistence: <2000ms for 1-5 MB files
  - Document listing: <500ms for 10,000 documents
  - Concurrent uploads: Support 100+ simultaneous requests without corruption

- **Reliability**:
  - Atomic transactions prevent partial writes
  - Retry logic (3 attempts, exponential backoff) for transient failures
  - Graceful degradation when optional services (Redis, LLM) unavailable

- **Data Integrity**:
  - Content hash ensures duplicate detection
  - Referential integrity between documents and linked artifacts
  - Required field validation before persistence

- **Observability**:
  - Structured logging at info/warning/error levels
  - Performance metrics (duration_ms) in all responses
  - Constitutional audit trail (caller_service, caller_function)

---

## Dependencies

- **Neo4j**: Graph database for all persistence (graph + vector + full-text)
- **DaedalusGraphChannel**: Constitutional interface for Neo4j access (Spec 040)
- **LangGraph**: Workflow orchestration framework
- **Redis**: Optional basin evolution tracking
- **Local LLM**: Optional Ollama/qwen2.5:14b for summary generation
- **PyPDF2**: PDF text extraction
- **BeautifulSoup**: HTML text extraction
- **RecursiveCharacterTextSplitter**: Text chunking

---

## Acceptance Criteria Summary

1. ✅ Daedalus class has single responsibility: `receive_perceptual_information()`
2. ✅ LangGraph workflow executes all 6 nodes: extract → research → consciousness → analyze → decide → finalize
3. ✅ Five-level concepts extracted and persisted with relationships
4. ✅ Attractor basins and thoughtseeds created and linked to documents
5. ✅ Content hash computed, validated, and used for duplicate detection
6. ✅ LLM summary generated and stored with fallback to extractive method
7. ✅ Document listing supports pagination, filtering (tags, quality, date, tier), and sorting
8. ✅ Document retrieval returns complete data with access tracking
9. ✅ URL ingestion downloads, converts, chunks, and persists with source metadata
10. ✅ All Neo4j access goes through Graph Channel (no direct imports)
11. ✅ Performance targets met: persistence <2s, listing <500ms
12. ✅ Error handling includes retries, fallbacks, and structured responses
