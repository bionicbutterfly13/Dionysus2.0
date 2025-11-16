# Document Processing - Design

## Architecture Overview

The Document Processing capability implements a multi-stage pipeline architecture:

```
┌─────────────────┐
│  Upload/URL     │
│  (External)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Daedalus      │  Perceptual Gateway (Spec 021)
│   Gateway       │  - Single responsibility: receive_perceptual_information()
└────────┬────────┘  - File validation and routing
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│              DocumentProcessingGraph                         │
│              (LangGraph 6-Node Workflow)                     │
│                                                              │
│  1. Extract & Process    → SurfSense patterns               │
│  2. Research Planning    → ASI-GO-2 Researcher + R-Zero     │
│  3. Consciousness        → Basins + ThoughtSeeds            │
│  4. Analysis             → Quality scoring                  │
│  5. Refine (conditional) → Iterative improvement            │
│  6. Finalize             → Package results + AutoSchemaKG   │
└────────┬────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│ DocumentRepo    │  Persistence Layer (Spec 054/055)
│                 │  - Content hash computation
│                 │  - Duplicate detection
│                 │  - LLM summarization
│                 │  - Neo4j persistence via Graph Channel
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│     Neo4j       │  Unified Storage (Spec 040)
│  (Graph Channel)│  - Document nodes
│                 │  - Concept hierarchy (5 levels)
│                 │  - Attractor basins
│                 │  - ThoughtSeeds
│                 │  - Chunks (URL ingestion)
└─────────────────┘
```

**Design Principles**:
1. **Single Responsibility**: Each component has one clear purpose
2. **Constitutional Compliance**: All database access through Graph Channel
3. **Observable**: Structured logging and performance metrics throughout
4. **Resilient**: Graceful degradation when optional services unavailable
5. **Spec-Driven**: Implementation follows formal specifications (021, 054, 055)

---

## Key Components

### 1. Daedalus Perceptual Gateway

**File**: `backend/src/services/daedalus.py`

**Purpose**: Single-responsibility gateway for receiving perceptual information from external sources.

**Public Interface**:
```python
class Daedalus:
    def receive_perceptual_information(
        self,
        data: BinaryIO,
        tags: List[str] = None,
        max_iterations: int = 3,
        quality_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Receive and process perceptual information through LangGraph workflow.

        Returns:
            {
                'status': 'received' | 'error',
                'document': {...},
                'extraction': {...},
                'consciousness': {...},
                'research': {...},
                'quality': {...},
                'meta_cognitive': {...},
                'workflow': {...},
                'timestamp': float
            }
        """
```

**Implementation Pattern**:
- Validates input data
- Reads file content and resets file pointer
- Delegates to `DocumentProcessingGraph.process_document()`
- Packages results with error handling

**Spec Reference**: Spec 021 (remove-all-that) - simplified to single method

---

### 2. DocumentProcessingGraph (LangGraph Workflow)

**File**: `backend/src/services/document_processing_graph.py`

**Purpose**: Orchestrate consciousness-enhanced document processing through a 6-node LangGraph state machine.

**State Object**:
```python
class DocumentProcessingState(TypedDict):
    # Input
    content: bytes
    filename: str
    tags: List[str]

    # Processing artifacts
    processing_result: DocumentProcessingResult
    research_plan: Dict[str, Any]
    analysis: Dict[str, Any]

    # Iteration control
    iteration: int
    max_iterations: int
    quality_threshold: float

    # Output
    final_output: Dict[str, Any]
    messages: List[str]
```

**Node Implementations**:

#### Node 1: Extract & Process
- Detects file type (PDF vs text)
- Processes via `ConsciousnessDocumentProcessor`
- Extracts initial concepts
- Updates state with `processing_result`

#### Node 2: Generate Research Plan
- Receives extracted concepts
- Calls `DocumentResearcher.generate_research_questions()`
- Generates curiosity-driven questions (ASI-GO-2 + R-Zero)
- Updates state with `research_plan`

#### Node 3: Consciousness Processing
- Creates attractor basins from concepts
- Generates thoughtseeds (questions/insights)
- Tracks neural field resonance patterns
- (In current implementation, this happens in Node 1 - can be enhanced)

#### Node 4: Analyze Results
- Calls `DocumentAnalyst.analyze_processing_result()`
- Computes quality scores (overall, coherence, novelty, depth)
- Extracts insights and recommendations
- Updates state with `analysis`

#### Node 5: Decide (Conditional)
```python
def _should_refine(state) -> str:
    if iteration >= max_iterations:
        return "complete"
    if quality >= quality_threshold:
        return "complete"
    return "refine"
```
- Returns "refine" → loop to Node 5 (refine processing) → Node 3
- Returns "complete" → proceed to Node 6

#### Node 6: Finalize Output
- Extracts five-level concepts
- Processes concepts through AutoSchemaKG (async)
- Creates knowledge graph (nodes + relationships)
- Packages final_output with all artifacts
- Returns to Daedalus

**Integration Points**:
- `ConsciousnessDocumentProcessor`: PDF/text processing
- `DocumentCognitionBase`: Learned patterns storage
- `DocumentResearcher`: Question generation
- `DocumentAnalyst`: Quality analysis
- `AutoSchemaKGService`: Knowledge graph construction
- `FiveLevelConceptExtractionService`: Concept classification
- `MultiTierMemorySystem`: Warm/cool/cold tier management

---

### 3. DocumentRepository (Persistence Layer)

**File**: `backend/src/services/document_repository.py`

**Purpose**: Persist and retrieve documents with all processing artifacts via Graph Channel.

**Key Methods**:

#### `persist_document(final_output, metadata) -> Dict`
**Flow**:
1. Compute content_hash if not provided (`compute_content_hash(document_body, namespace)`)
2. Validate content_hash format (64 hex characters)
3. Check for duplicates (`find_duplicate_by_hash()`)
4. Create Document node with metadata + quality scores + summary
5. Persist 5-level concepts (`_persist_concepts()`)
6. Persist attractor basins (`_persist_basins()`)
7. Persist thoughtseeds (`_persist_thoughtseeds()`)
8. Return result with performance metrics

**Performance**: Target <2000ms for 1-5 MB files

#### `get_document(document_id) -> Dict`
**Flow**:
1. Execute single comprehensive Cypher query:
   - MATCH Document
   - OPTIONAL MATCH Concepts by level
   - OPTIONAL MATCH Basins
   - OPTIONAL MATCH ThoughtSeeds
   - SET access tracking (last_accessed, access_count)
2. Organize concepts by level (atomic/relationship/composite/context/narrative)
3. Return complete document with all artifacts

**Performance**: Access tracking updates happen in same query (write operation)

#### `list_documents(page, limit, filters) -> Dict`
**Flow**:
1. Build WHERE clause from filters (tags, quality_min, date_from/to, tier, source_type)
2. Build ORDER BY clause (upload_date, quality, curiosity)
3. Calculate pagination (skip, limit)
4. Execute two parallel queries:
   - Main query with SKIP/LIMIT for documents
   - Count query for total
5. Return documents + pagination metadata

**Performance**: Target <500ms for 10,000 documents, uses Neo4j indexes

#### `persist_document_from_url(url, metadata) -> Dict`
**Flow** (Spec 056):
1. Download URL via `URLDownloader.download_url()`
2. Convert to text based on MIME type (PDF/HTML/plain)
3. Compute content_hash and check duplicates
4. Chunk text via `DocumentChunker.chunk_document()`
5. Process through Daedalus LangGraph workflow
6. Persist document with source_type="url", original_url, connector_icon
7. Persist chunks via `_persist_chunks()`
8. Return result with chunks_created count

---

### 4. Content Hash System (Spec 055)

**Utility Functions** (in `document_repository.py`):

#### `compute_content_hash(document_body: str, namespace: str) -> str`
```python
combined = document_body + namespace
hash_bytes = hashlib.sha256(combined.encode('utf-8')).digest()
return hash_bytes.hex()  # 64 lowercase hex characters
```
**Purpose**: Deterministic duplicate detection
**Namespace**: Allows same content in different contexts (default: "default")

#### `validate_content_hash(content_hash: str) -> bool`
```python
return re.match(r'^[0-9a-f]{64}$', content_hash.lower()) is not None
```
**Purpose**: Ensure hash format before persistence

#### `find_duplicate_by_hash(content_hash: str) -> Optional[Dict]`
**Purpose**: Return canonical document metadata for 409 Conflict responses
**Query**: `MATCH (d:Document {content_hash: $content_hash})`

---

### 5. LLM Summarization System (Spec 055 Agent 3)

**Component**: `DocumentSummarizer` (in `document_summarizer.py`)

**Configuration**:
```python
SummarizerConfig(
    model="qwen2.5:14b",    # Local Ollama model
    max_tokens=150,          # Summary length limit
    temperature=0.3          # Low variance for consistency
)
```

**Fallback Strategy**:
1. Try local LLM via Ollama
2. If unavailable, use extractive summarization (first N sentences)
3. Store summary_metadata indicating method used

**Integration** (in `_create_document_node()`):
```python
summary_result = await self.summarizer.generate_summary(document_text, max_tokens=150)
summary = summary_result.get("summary")
summary_metadata = {
    "method": "llm" | "extractive",
    "model": "qwen2.5:14b" | None,
    "tokens_used": int,
    "generated_at": timestamp,
    "error": str | None
}
```

---

### 6. URL Ingestion Pipeline (Spec 056)

**Components**:
- `URLDownloader`: HTTPS download with retry/backoff
- `DocumentChunker`: RecursiveCharacterTextSplitter wrapper
- Integration in `DocumentRepository.persist_document_from_url()`

**Text Extraction**:
```python
if mime_type == "application/pdf":
    # PyPDF2.PdfReader
    for page in reader.pages:
        text += page.extract_text()

elif mime_type == "text/html":
    # BeautifulSoup
    soup = BeautifulSoup(content_bytes, 'html.parser')
    for script in soup(["script", "style"]):
        script.decompose()
    text = soup.get_text(separator="\n", strip=True)

elif mime_type == "text/plain":
    text = content_bytes.decode('utf-8', errors='ignore')
```

**Chunking** (default config):
- chunk_size: 1000 characters
- overlap: 200 characters
- Stable chunk_id generation: `f"{document_id}_chunk_{position}"`

**Chunk Schema**:
```cypher
CREATE (c:Chunk {
    chunk_id: string,
    content: string,
    position: int,
    start_char: int,
    end_char: int,
    created_at: datetime
})
CREATE (c)-[:PART_OF {position: int}]->(d:Document)
```

---

### 7. Five-Level Concept Hierarchy

**Levels**:
1. **Atomic**: Single words, named entities, fundamental terms
   - Properties: concept_id, name, salience, definition

2. **Relationship**: Connections between atomic concepts
   - Properties: source_concept, target_concept (references to atomic concept_ids)

3. **Composite**: Higher-level concepts from atomic components
   - Properties: components (array of atomic concept_ids)

4. **Context**: Domain/era classification
   - Properties: domain, era

5. **Narrative**: Storyline and thematic flow
   - Properties: storyline

**Schema**:
```cypher
CREATE (c:Concept {
    concept_id: string,
    name: string,
    level: "atomic" | "relationship" | "composite" | "context" | "narrative",
    salience: float,
    // Level-specific properties
    definition: string,           // atomic
    source_concept: string,       // relationship
    target_concept: string,       // relationship
    components: [string],         // composite
    domain: string,               // context
    era: string,                  // context
    storyline: string,            // narrative
    created_at: datetime
})
CREATE (c)-[:EXTRACTED_FROM {
    confidence: float,
    extraction_method: "AutoSchemaKG",
    timestamp: datetime
}]->(d:Document)
```

**Current Limitation**: AutoSchemaKG integration in `_finalize_output_node()` only extracts atomic concepts. Relationship/composite/context/narrative extraction is marked TODO for future enhancement.

---

### 8. Attractor Basin System

**Purpose**: Model stable conceptual domains in consciousness processing landscape.

**Schema**:
```cypher
CREATE (b:AttractorBasin {
    basin_id: string,
    name: string,
    depth: float,              // Strength of attraction
    stability: float,          // Resistance to change
    strength: float,           // Cumulative reinforcement
    associated_concepts: [string],
    created_at: datetime,
    last_modified: datetime,
    modification_count: int
})
CREATE (b)-[:ATTRACTED_TO {
    activation_strength: float,
    influence_type: "reinforcement" | "competition" | "synthesis" | "emergence",
    strength_delta: float,     // Change in basin strength from this document
    timestamp: datetime
}]->(d:Document)
```

**Evolution Tracking** (Redis):
- Key: `basin:evolution:{basin_id}`
- Value: List of influence events (document_id, influence_type, strength_delta, timestamp)
- TTL: 90 days

**Update Pattern** (in `_persist_basins()`):
```cypher
MERGE (b:AttractorBasin {basin_id: $basin_id})
ON CREATE SET b.strength = $strength, b.modification_count = 0
ON MATCH SET b.strength = b.strength + $strength_delta,
             b.modification_count = b.modification_count + 1
```

---

### 9. ThoughtSeed System

**Purpose**: Capture questions and insights generated during processing for future exploration.

**Schema**:
```cypher
CREATE (t:ThoughtSeed {
    seed_id: string,
    content: string,           // Question or insight text
    germination_potential: float,
    resonance_score: float,
    field_resonance_energy: float,
    field_resonance_phase: float,
    field_resonance_pattern: string,
    generated_at: datetime,
    source_stage: "consciousness_processing" | "research_planning" | "analysis"
})
CREATE (t)-[:GERMINATED_FROM {
    potential: float,
    generation_stage: string,
    timestamp: datetime
}]->(d:Document)
```

**Neural Field Resonance**: Tracks interference patterns between concepts for cross-domain insights (from Context Engineering Foundation).

---

### 10. Tier Management System

**Tier Classification**:
- **Warm**: Recently uploaded or frequently accessed documents
- **Cool**: Older documents with moderate access
- **Cold**: Very old rarely-accessed documents (archived to S3/filesystem)

**Initial Assignment**: All documents start in "warm" tier

**Migration Criteria** (Spec 054/055):
- **Hybrid approach**: Age + access patterns
- Frequently accessed documents stay warm regardless of age
- Rarely accessed documents age to cool/cold faster

**Cold Tier Archival**:
- Move document content to archive_location (S3/filesystem)
- Retain metadata and document_id in Neo4j
- Set archived_at timestamp
- Accept slower retrieval times

**Access Tracking** (in `get_document()`):
```cypher
SET d.last_accessed = datetime(),
    d.access_count = d.access_count + 1
```

**Future Enhancement**: Automated tier migration based on access patterns (currently manual via `update_tier()`)

---

## Data Flow Patterns

### Upload Flow (Spec 021 + 054)
```
User Upload
  → FastAPI endpoint (multipart/form-data)
  → Daedalus.receive_perceptual_information(data, tags)
    → DocumentProcessingGraph.process_document(content, filename, tags)
      → [6-node LangGraph workflow]
      → Returns final_output
    → Returns formatted response
  → DocumentRepository.persist_document(final_output, metadata)
    → Compute content_hash
    → Check duplicates
    → Create Document node
    → Persist concepts/basins/seeds
    → Returns persistence result
  → Response to user with document_id
```

### URL Ingestion Flow (Spec 056)
```
URL Request
  → FastAPI endpoint (JSON with url field)
  → DocumentRepository.persist_document_from_url(url, metadata)
    → URLDownloader.download_url(url)
    → Convert to text (PDF/HTML/plain)
    → Compute content_hash, check duplicates
    → DocumentChunker.chunk_document(text)
    → Daedalus.receive_perceptual_information(content, tags)
    → persist_document(final_output, metadata)
    → _persist_chunks(chunks, document_id)
    → Returns result with chunks_created
  → Response to user
```

### Listing Flow (Spec 054)
```
GET /api/documents?page=1&tags=ai&quality_min=0.8
  → DocumentRepository.list_documents(page, filters)
    → Build WHERE clause from filters
    → Execute parallel queries (documents + count)
    → Organize results with pagination metadata
    → Returns {documents, pagination, performance}
  → Response to user
```

### Detail Retrieval Flow (Spec 054)
```
GET /api/documents/{document_id}
  → DocumentRepository.get_document(document_id)
    → Execute comprehensive Cypher query
      → MATCH Document
      → OPTIONAL MATCH Concepts/Basins/Seeds
      → SET last_accessed, access_count++
    → Organize concepts by level
    → Returns complete document data
  → Response to user
```

---

## Storage Patterns

### Document Storage (Neo4j)

**Node Labels**:
- `Document`: Core document metadata
- `Concept`: Five-level concept hierarchy
- `AttractorBasin`: Consciousness processing artifacts
- `ThoughtSeed`: Generated questions/insights
- `Chunk`: Text segments (URL ingestion)

**Relationship Types**:
- `EXTRACTED_FROM`: Concept → Document (with confidence, extraction_method)
- `ATTRACTED_TO`: AttractorBasin → Document (with activation_strength, influence_type)
- `GERMINATED_FROM`: ThoughtSeed → Document (with potential, generation_stage)
- `PART_OF`: Chunk → Document (with position, chunk_index)

**Indexes** (from plan.md T020):
```cypher
CREATE INDEX idx_document_content_hash ON :Document(content_hash);
CREATE INDEX idx_document_tags ON :Document(tags);
CREATE INDEX idx_document_upload ON :Document(upload_timestamp);
CREATE INDEX idx_document_quality ON :Document(quality_overall);
CREATE INDEX idx_concept_level ON :Concept(level);
```

**Vector Storage** (Neo4j native):
- 512-dim embeddings for concepts
- Cosine similarity search
- Hybrid queries (graph + vector + full-text in single Cypher query)

**Full-Text Search** (Neo4j native):
- Built-in indexing on Document.filename, Concept.name
- Supports fuzzy matching and relevance scoring

---

### Redis Storage (Optional)

**Basin Evolution Tracking**:
- Key: `basin:evolution:{basin_id}`
- Structure: List of JSON strings
- TTL: 90 days
- Purpose: Temporary event history for basin strength analysis

**Fallback**: If Redis unavailable, basin persistence continues without evolution tracking (logged warning).

---

## Constitutional Compliance (Spec 040)

**Graph Channel Pattern**:
```python
from daedalus_gateway import get_graph_channel

graph_channel = get_graph_channel()

# Read operation
result = await graph_channel.execute_read(
    query="MATCH (d:Document) RETURN d",
    parameters={},
    caller_service="document_repository",
    caller_function="list_documents"
)

# Write operation
result = await graph_channel.execute_write(
    query="CREATE (d:Document {document_id: $id})",
    parameters={"id": "doc_123"},
    caller_service="document_repository",
    caller_function="persist_document"
)
```

**Prohibited Imports**:
```python
# ❌ NEVER do this:
from neo4j import GraphDatabase, Driver
import neo4j

# ✅ ALWAYS do this:
from daedalus_gateway import get_graph_channel
```

**Audit Trail**: Every database operation includes `caller_service` and `caller_function` for traceability.

---

## Error Handling Patterns

### Validation Errors
```python
# Missing required fields
if "document_id" not in metadata:
    raise ValueError("Missing required field: document_id")

# Invalid content_hash format
if not validate_content_hash(content_hash):
    raise ValueError(f"Invalid content_hash format: {content_hash}")
```

### Duplicate Detection
```python
duplicate = await find_duplicate_by_hash(content_hash)
if duplicate:
    raise ValueError(
        f"Duplicate document detected. Content hash {content_hash} "
        f"already exists as document {duplicate['document_id']}"
    )
```
**API Response**: 409 Conflict with canonical document metadata

### Transient Failures
```python
# Retry logic (Graph Channel handles this internally)
for attempt in range(3):
    try:
        result = await graph_channel.execute_write(...)
        break
    except TransientError as e:
        if attempt == 2:
            raise
        await asyncio.sleep(2 ** attempt)  # Exponential backoff
```

### Optional Service Failures
```python
# LLM Summarizer unavailable
try:
    summary_result = await self.summarizer.generate_summary(text)
except Exception as e:
    logger.warning(f"Summary generation failed: {e}")
    # Continue without summary - not a critical failure
    summary = None
    summary_metadata = {"error": str(e)}
```

### Structured Error Responses
```python
return {
    "status": "error",
    "error_message": "Corrupted data: unable to read file",
    "timestamp": time.time()
}
```

---

## Performance Optimization

### Query Optimization
1. **Single comprehensive query**: Get document + concepts + basins + seeds in one Cypher query
2. **Parallel queries**: Run document list + count queries concurrently
3. **Indexes**: Use Neo4j indexes for WHERE clauses (content_hash, tags, upload_timestamp, quality)
4. **Pattern comprehension**: Use `[(c:Concept)-[:EXTRACTED_FROM]->(d) | c]` for efficient counting

### Pagination Strategy
```python
skip = (page - 1) * limit
# Separate count query to avoid counting on every page
# Cache total count for repeated requests (future enhancement)
```

### Concurrent Upload Handling
- Content hash locking prevents duplicate processing
- Atomic transactions prevent race conditions
- Target: 100+ concurrent requests without corruption

### Performance Targets
- Document persistence: <2000ms (1-5 MB files)
- Document listing: <500ms (10,000 documents)
- Document retrieval: <200ms (single document)

**Monitoring**: All methods return `performance` object with `duration_ms` and `met_target` boolean.

---

## Testing Patterns

### Contract Tests (Spec 054/055)
```python
# Test duplicate detection
@pytest.mark.contract
async def test_persist_document_duplicate_rejection():
    # First upload succeeds
    result1 = await repo.persist_document(output, metadata)

    # Second upload with same content_hash fails
    with pytest.raises(ValueError, match="Duplicate document detected"):
        await repo.persist_document(output, metadata)
```

### Integration Tests
```python
# Test LangGraph workflow end-to-end
@pytest.mark.integration
def test_document_processing_workflow():
    daedalus = Daedalus()
    result = daedalus.receive_perceptual_information(file_data, tags=["test"])

    assert result["status"] == "received"
    assert len(result["extraction"]["concepts"]) > 0
    assert result["quality"]["scores"]["overall"] > 0
```

### Unit Tests
```python
# Test content hash computation
def test_compute_content_hash():
    hash1 = compute_content_hash("Test content", "default")
    assert len(hash1) == 64
    assert hash1.islower()

    # Same content, same namespace → same hash
    hash2 = compute_content_hash("Test content", "default")
    assert hash1 == hash2

    # Same content, different namespace → different hash
    hash3 = compute_content_hash("Test content", "research")
    assert hash1 != hash3
```

---

## Migration from Legacy Specs

This design consolidates three legacy specs:

**Spec 021 (remove-all-that)**:
- ✅ Implemented: Daedalus simplified to `receive_perceptual_information()`
- ✅ Implemented: LangGraph workflow integration
- ✅ Implemented: Clean single-responsibility architecture

**Spec 054 (document-persistence-repository)**:
- ✅ Implemented: DocumentRepository with persist/get/list methods
- ✅ Implemented: Five-level concept persistence
- ✅ Implemented: Attractor basin and thoughtseed persistence
- ✅ Implemented: Constitutional compliance (Graph Channel only)
- ✅ Implemented: Performance targets (<2s persistence, <500ms listing)
- ⚠️ Partial: Tier migration (manual only, automated migration TODO)

**Spec 055 (document-persistence-baseline)**:
- ✅ Implemented: Content hash computation (SHA-256)
- ✅ Implemented: Duplicate detection via content_hash
- ✅ Implemented: LLM summary generation with fallback
- ✅ Implemented: Contract tests updated

**Spec 056 (url-and-chunk-ingestion-pipeline)**:
- ✅ Implemented: URL download with retry/backoff
- ✅ Implemented: PDF/HTML text extraction
- ✅ Implemented: RecursiveCharacterTextSplitter chunking
- ✅ Implemented: Chunk persistence with stable IDs
- ✅ Implemented: Source metadata (source_type, original_url, connector_icon)

**Spec 057 (source-metadata-and-external-access)**:
- ✅ Implemented: source_type, original_url, connector_icon fields
- ✅ Implemented: Connector icon inference from MIME type
- ✅ Implemented: Download metadata persistence
- ⚠️ Partial: UI affordance for "open original" (backend ready, UI TODO)

---

## Future Enhancements

1. **AutoSchemaKG Full Integration**: Extract all five concept levels (currently only atomic)
2. **Automated Tier Migration**: Background worker to move documents between tiers based on access patterns
3. **Cold Tier Archival**: Implement S3/filesystem archival for cold tier documents
4. **Chunk Highlighting**: UI support for citation side-sheet with auto-scroll (Spec 058)
5. **LangGraph Transformations**: Post-ingestion transformations (summaries, key points, questions) (Spec 059)
6. **Notebook Insights**: Group insights and thoughtseeds into explorable notebooks (Spec 059)
7. **Redis Caching**: Cache document list total counts for repeated requests
8. **Vector Search**: Expose concept embedding search API for similarity queries
9. **Cross-Document Linking**: Detect and link related documents via shared concepts/basins
10. **Provenance Tracking**: Full audit trail of document modifications and re-processing

---

## References

- **Spec 021**: `/Volumes/Asylum/dev/Dionysus-2.0/specs/021-remove-all-that/spec.md`
- **Spec 054**: `/Volumes/Asylum/dev/Dionysus-2.0/specs/054-document-persistence-repository/spec.md`
- **Spec 055**: `/Volumes/Asylum/dev/Dionysus-2.0/specs/055-document-persistence-baseline/spec.md`
- **Spec 040**: Constitutional compliance (Graph Channel)
- **Implementation Files**:
  - `backend/src/services/daedalus.py` (118 lines)
  - `backend/src/services/document_processing_graph.py` (429 lines)
  - `backend/src/services/document_repository.py` (1449 lines)
