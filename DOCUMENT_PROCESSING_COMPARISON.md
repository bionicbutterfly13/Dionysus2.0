# Document Processing Systems Comparison

**Date**: 2025-10-01
**Systems Analyzed**: SurfSense, Dionysus, Perplexica, OpenNotebook

## Executive Summary

Four distinct approaches to document processing, each with unique strengths:

| System | Core Strength | Storage | Best Feature |
|--------|---------------|---------|--------------|
| **SurfSense** | Production RAG with duplicate detection | PostgreSQL + pgvector | Content hash + LLM summary |
| **Dionysus** | Consciousness-guided learning | Neo4j + Redis | Attractor basins + pattern learning |
| **Perplexica** | Simple file upload + embeddings | Filesystem JSON | Clean RecursiveCharacterTextSplitter |
| **OpenNotebook** | LangGraph workflow + transformations | SQLite + vector DB | Content extraction + insights |

## Detailed Comparison

### 1. SurfSense (Production RAG System)

**Repository**: /Volumes/Asylum/dev/Flux/surfsense_backend

**Processing Pipeline**:
```python
Upload → Markdown Conversion → Content Hash →
Duplicate Check → LLM Summary → Chunks →
Embeddings → PostgreSQL + pgvector
```

**Key Files**:
- `app/tasks/document_processors/file_processors.py`
- `app/utils/document_converters.py`

**Storage Model**:
```python
class Document:
    id: int
    title: str
    content: Text  # LLM summary with metadata
    content_hash: str (unique)  # SHA-256 duplicate detection
    embedding: Vector  # pgvector
    chunks: List[Chunk]  # One-to-many
    document_metadata: JSON

class Chunk:
    id: int
    content: Text
    embedding: Vector
    document_id: int (FK)
```

**Best Practices**:
1. **Content Hash for Duplicate Detection**
   ```python
   content_hash = hashlib.sha256(f"{search_space_id}:{content}".encode()).hexdigest()
   existing = db.query(Document).filter_by(content_hash=content_hash).first()
   if existing:
       return existing  # Skip re-processing
   ```

2. **LLM-Generated Summary with Metadata**
   ```python
   summary = await user_llm.ainvoke({
       "document": f"<METADATA>{metadata}</METADATA><CONTENT>{content}</CONTENT>"
   })
   enhanced_summary = f"# METADATA\n{metadata}\n\n# SUMMARY\n{summary}"
   ```

3. **Token-Aware Content Optimization**
   ```python
   context_window = get_model_context_window(model_name)
   available_tokens = context_window - reserved_tokens
   # Binary search to find optimal content length
   optimized_content = content[:optimal_length]
   ```

4. **Hybrid Search (Vector + Full-Text)**
   - Document-level: Summary embedding for broad search
   - Chunk-level: Precise embeddings for detailed retrieval

**Strengths**:
- ✅ Production-ready duplicate detection
- ✅ Token-aware content optimization
- ✅ LLM summary generation
- ✅ Hybrid search strategy
- ✅ Supports multiple ETL services (Unstructured, LlamaCloud, Docling)

**Limitations**:
- ❌ No consciousness tracking
- ❌ No pattern learning
- ❌ Relational database (less flexible for concept graphs)

---

### 2. Dionysus (Consciousness-Guided Learning)

**Repository**: /Volumes/Asylum/dev/Dionysus-2.0

**Processing Pipeline**:
```python
Upload → Daedalus Gateway → Concept Extraction →
AttractorBasin Creation → ThoughtSeed Generation →
Pattern Learning (4 types) → Neo4j + Redis
```

**Key Files**:
- `backend/src/services/consciousness_document_processor.py`
- `backend/src/services/daedalus.py`
- `extensions/context_engineering/attractor_basin_dynamics.py`

**Storage Model**:
```python
class Document(Pydantic):
    id: str (UUID)
    filename: str
    file_hash: str  # SHA-256
    extracted_text: str

    # Consciousness tracking
    thoughtseed_ids: List[str]
    attractor_basin_ids: List[str]
    neural_field_ids: List[str]
    memory_formation_ids: List[str]

# Stored in Redis
class AttractorBasin:
    basin_id: str
    center_concept: str
    strength: float (0.0-2.0)  # Basin depth
    thoughtseeds: Set[str]
    formation_timestamp: datetime
    last_modification: datetime
    activation_history: List[float]
```

**Pattern Learning (Unique Feature)**:
1. **REINFORCEMENT**: New concept strengthens existing basin
   - Similarity > 0.8, strong basin (strength > 1.5)
   - `basin.strength = min(2.0, basin.strength + 0.2)`

2. **COMPETITION**: New concept creates competing basin
   - Similarity > 0.5, reduces original strength
   - Creates new basin with moderate strength (0.8)

3. **SYNTHESIS**: New concept merges with existing basin
   - Similarity > 0.7, expands basin concept
   - Merges word sets, increases radius

4. **EMERGENCE**: New concept creates entirely new basin
   - Similarity < 0.5
   - New basin with neutral strength (1.0)

**Memory Decay**:
```python
# Basins unused for 7+ days decay
decay_rate = min(0.1, days_inactive * 0.01)
basin.strength = max(0.1, basin.strength - decay_rate)

# Weak basins (< 0.2) are removed
if basin.strength < 0.2:
    del basins[basin_id]
```

**Strengths**:
- ✅ Simulates how mind learns new information
- ✅ Pattern evolution tracking
- ✅ Memory decay (forgetting)
- ✅ Graph database for concept relationships
- ✅ Real-time learning feedback

**Limitations**:
- ❌ No duplicate detection (now added from SurfSense)
- ❌ Simple concept extraction (now improved)
- ❌ No LLM summary (now added)

---

### 3. Perplexica (Clean Simplicity)

**Repository**: /tmp/Perplexica

**Processing Pipeline**:
```typescript
Upload → PDF/DOCX/TXT Loader → RecursiveCharacterTextSplitter →
Embeddings → Filesystem JSON
```

**Key Files**:
- `src/app/api/uploads/route.ts`
- `src/lib/utils/documents.ts`

**Storage Model**:
```typescript
// Filesystem storage (uploads/)
{uniqueId}.pdf
{uniqueId}-extracted.json  // { title, contents: [...chunks] }
{uniqueId}-embeddings.json  // { title, embeddings: [...] }
```

**Processing Code**:
```typescript
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 500,
  chunkOverlap: 100
});

// PDF
const loader = new PDFLoader(filePath);
const docs = await loader.load();

// Split
const splitted = await splitter.splitDocuments(docs);

// Embed
const embeddings = await embeddingsModel.embedDocuments(
  splitted.map(doc => doc.pageContent)
);

// Save
fs.writeFileSync('extracted.json', JSON.stringify({
  title: fileName,
  contents: splitted.map(doc => doc.pageContent)
}));

fs.writeFileSync('embeddings.json', JSON.stringify({
  title: fileName,
  embeddings
}));
```

**Link Processing** (Unique Feature):
```typescript
// Handles both PDFs and web pages from URLs
const res = await axios.get(link, { responseType: 'arraybuffer' });

if (isPdf) {
  const pdfText = await pdfParse(res.data);
  // Process PDF from URL
} else {
  const parsedText = htmlToText(res.data);
  // Process web page
}
```

**Strengths**:
- ✅ Dead simple implementation
- ✅ Handles URLs directly (downloads and processes)
- ✅ Clean RecursiveCharacterTextSplitter usage
- ✅ Async parallel processing
- ✅ No database setup needed

**Limitations**:
- ❌ Filesystem storage (not scalable)
- ❌ No duplicate detection
- ❌ No summary generation
- ❌ No metadata tracking

---

### 4. OpenNotebook (LangGraph Workflows)

**Repository**: /tmp/open-notebook

**Processing Pipeline**:
```python
Upload → content_core.extract_content →
LangGraph Workflow → Save Source → Vectorize →
Apply Transformations → Generate Insights
```

**Key Files**:
- `open_notebook/graphs/source.py`
- `open_notebook/domain/notebook.py`

**Storage Model**:
```python
class Source:
    asset: Asset (url or file_path)
    full_text: str  # Extracted content
    title: str

    async def save(self):
        # Save to database

    async def add_to_notebook(self, notebook_id):
        # Associate with notebook

    async def vectorize(self):
        # Create embeddings

    async def add_insight(self, title, content):
        # Store transformation result
```

**LangGraph Workflow**:
```python
workflow = StateGraph(SourceState)

# Pipeline
workflow.add_edge(START, "content_process")
workflow.add_edge("content_process", "save_source")
workflow.add_conditional_edges(
    "save_source", trigger_transformations, ["transform_content"]
)
workflow.add_edge("transform_content", END)

# Execute
result = await source_graph.ainvoke({
    "content_state": {...},
    "notebook_id": "...",
    "apply_transformations": [...],
    "embed": True
})
```

**Content Processing** (Unique Feature):
```python
# Uses content_core library for extraction
content_state["url_engine"] = "auto"  # Or "jina", "firecrawl", etc.
content_state["document_engine"] = "auto"  # Or "docling", etc.
content_state["output_format"] = "markdown"

processed = await extract_content(content_state)
# Returns: { url, file_path, content, title }
```

**Transformations** (Unique Feature):
```python
# Apply LLM transformations to content
transformations = [
    Transformation(name="summarize", prompt="Summarize this..."),
    Transformation(name="extract_key_points", prompt="Extract..."),
    Transformation(name="generate_questions", prompt="Generate...")
]

# Each transformation creates an "insight" stored with the source
await source.add_insight("summary", summary_text)
await source.add_insight("key_points", points_text)
```

**Strengths**:
- ✅ LangGraph workflow orchestration
- ✅ Pluggable content extraction engines
- ✅ Transformation system (insights)
- ✅ Notebook organization
- ✅ Clean async architecture

**Limitations**:
- ❌ No duplicate detection
- ❌ No chunking strategy
- ❌ Simple storage model

---

## Synthesis: Best of All Systems

### Recommended Hybrid Approach

Combine strengths from all 4 systems:

```python
class UnifiedDocumentProcessor:
    """
    Best practices from:
    - SurfSense: Duplicate detection, LLM summary, token optimization
    - Dionysus: Consciousness processing, pattern learning, memory decay
    - Perplexica: Clean chunking, URL processing
    - OpenNotebook: LangGraph workflows, transformations
    """

    async def process_document(self, file_or_url: str) -> ProcessingResult:
        # 1. PERPLEXICA: Handle URLs directly
        if is_url(file_or_url):
            content = await download_and_extract(file_or_url)
        else:
            content = await load_file(file_or_url)

        # 2. SURFSENSE: Duplicate detection
        content_hash = generate_content_hash(content)
        existing = await check_duplicate(content_hash)
        if existing:
            return existing

        # 3. OPENNOTEBOOK: Extract with content_core
        extracted = await extract_content({
            "content": content,
            "output_format": "markdown",
            "engine": "auto"
        })

        # 4. SURFSENSE: Token-aware optimization
        optimized = optimize_for_context_window(
            extracted.content,
            model_name
        )

        # 5. PERPLEXICA: Clean chunking
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        chunks = await splitter.split_text(optimized)

        # 6. DIONYSUS: Concept extraction
        concepts = extract_concepts(optimized)

        # 7. DIONYSUS: Consciousness processing
        basin_results = await process_through_basins(concepts)

        # 8. SURFSENSE: LLM summary with metadata
        summary = await generate_llm_summary(
            optimized,
            metadata={
                "filename": filename,
                "concepts": concepts[:10],
                "basins_created": basin_results.basins_created
            }
        )

        # 9. PERPLEXICA: Generate embeddings
        embeddings = await embed_model.embed_documents(chunks)

        # 10. OPENNOTEBOOK: Apply transformations
        insights = {}
        for transformation in transformations:
            result = await apply_transformation(
                content=optimized,
                transformation=transformation
            )
            insights[transformation.name] = result

        # 11. Storage
        document = Document(
            content_hash=content_hash,
            markdown=optimized,
            summary=summary,
            chunks=chunks,
            embeddings=embeddings,
            concepts=concepts,
            basins=basin_results.basins,
            thoughtseeds=basin_results.thoughtseeds,
            patterns=basin_results.patterns,
            insights=insights
        )

        await document.save()
        return document
```

## Feature Comparison Matrix

| Feature | SurfSense | Dionysus | Perplexica | OpenNotebook | Recommended |
|---------|-----------|----------|------------|--------------|-------------|
| **Duplicate Detection** | ✅ Content hash | ❌ | ❌ | ❌ | SurfSense |
| **LLM Summary** | ✅ With metadata | ❌ (now added) | ❌ | ❌ | SurfSense |
| **Token Optimization** | ✅ Binary search | ❌ | ❌ | ❌ | SurfSense |
| **Chunking** | ✅ LangChain | ✅ Granular | ✅ Clean | ❌ | Perplexica (simple) |
| **Embeddings** | ✅ Summary + chunks | ❌ | ✅ | ✅ | SurfSense (hybrid) |
| **URL Processing** | ❌ | ❌ | ✅ Direct download | ✅ content_core | Perplexica |
| **Consciousness** | ❌ | ✅ Basins + patterns | ❌ | ❌ | Dionysus |
| **Pattern Learning** | ❌ | ✅ 4 types | ❌ | ❌ | Dionysus |
| **Memory Decay** | ❌ | ✅ Time-based | ❌ | ❌ | Dionysus |
| **Workflows** | ❌ | ❌ | ❌ | ✅ LangGraph | OpenNotebook |
| **Transformations** | ❌ | ❌ | ❌ | ✅ Insights | OpenNotebook |
| **Storage** | PostgreSQL | Neo4j | Filesystem | SQLite | Depends on use case |

## Implementation Priority for Dionysus

### Already Integrated ✅
1. Content hash duplicate detection (from SurfSense)
2. Markdown conversion (from SurfSense)
3. Document chunking (from SurfSense)
4. Consciousness processing (native)
5. Pattern learning (native)
6. Memory decay (native)

### Next Steps (Priority Order)

**High Priority**:
1. **LLM Summary Generation** (SurfSense pattern)
   - Integrate with Ollama for local processing
   - Add metadata to summaries
   - Token-aware content optimization

2. **URL Processing** (Perplexica pattern)
   - Download and process URLs directly
   - Handle both PDFs and web pages
   - Async parallel processing

3. **Embedding Generation** (All systems)
   - Generate embeddings for summaries
   - Generate embeddings for chunks
   - Store in Qdrant

**Medium Priority**:
4. **LangGraph Workflows** (OpenNotebook pattern)
   - Orchestrate processing pipeline
   - Conditional transformations
   - Error handling and retries

5. **Transformation System** (OpenNotebook pattern)
   - Apply insights to documents
   - Store insights with sources
   - Track which transformations were applied

6. **Content Extraction Engines** (OpenNotebook pattern)
   - Integrate content_core library
   - Support multiple engines (Jina, Firecrawl, Docling)
   - Auto-select best engine for content type

**Low Priority**:
7. **Hybrid Search** (SurfSense pattern)
   - Combine vector + full-text search
   - Document-level + chunk-level retrieval
   - Ranking and fusion

## Code Examples

### 1. SurfSense Content Hash (Already Integrated)

```python
from consciousness_document_processor import ConsciousnessDocumentProcessor

processor = ConsciousnessDocumentProcessor()
result = processor.process_pdf(pdf_bytes, "paper.pdf")

print(f"Content Hash: {result.content_hash}")  # SHA-256
# Check for duplicates before processing
```

### 2. Perplexica URL Processing (TODO)

```python
# Add to ConsciousnessDocumentProcessor
async def process_url(self, url: str) -> DocumentProcessingResult:
    res = await httpx.get(url, follow_redirects=True)

    if res.headers.get('content-type') == 'application/pdf':
        return self.process_pdf(res.content, url)
    else:
        html = res.text
        markdown = html_to_markdown(html)
        return self.process_text(markdown.encode(), url)
```

### 3. OpenNotebook LangGraph Workflow (TODO)

```python
from langgraph.graph import StateGraph, END

workflow = StateGraph(DocumentState)

workflow.add_node("extract", extract_content)
workflow.add_node("duplicate_check", check_duplicate)
workflow.add_node("consciousness", process_consciousness)
workflow.add_node("summarize", generate_summary)
workflow.add_node("embed", generate_embeddings)
workflow.add_node("save", save_document)

workflow.add_edge(START, "extract")
workflow.add_edge("extract", "duplicate_check")
workflow.add_conditional_edges(
    "duplicate_check",
    lambda s: "skip" if s["duplicate"] else "continue",
    {"skip": END, "continue": "consciousness"}
)
workflow.add_edge("consciousness", "summarize")
workflow.add_edge("summarize", "embed")
workflow.add_edge("embed", "save")
workflow.add_edge("save", END)

document_graph = workflow.compile()
```

### 4. SurfSense Token Optimization (TODO)

```python
def optimize_content_for_context_window(
    content: str,
    model_name: str,
    reserved_tokens: int = 2000
) -> str:
    context_window = get_model_context_window(model_name)
    available_tokens = context_window - reserved_tokens

    # Binary search for optimal length
    left, right = 0, len(content)
    optimal_length = 0

    while left <= right:
        mid = (left + right) // 2
        test_content = content[:mid]
        test_tokens = count_tokens(test_content, model_name)

        if test_tokens <= available_tokens:
            optimal_length = mid
            left = mid + 1
        else:
            right = mid - 1

    return content[:optimal_length]
```

## Conclusion

**Dionysus Strengths**:
- Unique consciousness-guided learning
- Pattern evolution tracking
- Memory decay simulation
- Graph-based concept relationships

**Added from SurfSense**:
- Production-ready duplicate detection
- Structured document processing
- Clean chunking strategy

**To Add from Perplexica**:
- Direct URL processing
- Simple async architecture

**To Add from OpenNotebook**:
- LangGraph workflow orchestration
- Transformation/insights system
- Pluggable content extraction

**Result**: A hybrid system that combines production RAG (SurfSense) with consciousness-guided learning (Dionysus), orchestrated through LangGraph workflows (OpenNotebook), with clean URL handling (Perplexica).
