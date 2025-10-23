# Upload Pipeline Debugging Guide

## Complete Data Flow: File Upload to Storage

This document shows you **exactly** where your data goes when you upload a file.

---

## 🔄 Step-by-Step Pipeline

### Step 1: Frontend Upload (DocumentUpload.tsx)
**Location**: `flux-desktop/src/pages/DocumentUpload.tsx:173`

```typescript
// User uploads file → Creates FormData
const formData = new FormData()
formData.append('file', file)
formData.append('tags', JSON.stringify(['desktop']))

// POST to backend
const response = await fetch('/api/v1/documents?mode=local', {
  method: 'POST',
  body: formData
})
```

**Data Sent**:
- File binary content
- Tags: `['desktop']`
- Mode: `local`

**API Endpoint**: `POST http://127.0.0.1:9127/api/v1/documents?mode=local`

---

### Step 2: Backend API Route (document_persistence.py)
**Location**: `backend/src/api/routes/document_persistence.py`

```python
@router.post("/api/v1/documents")
async def upload_document(
    file: UploadFile,
    tags: List[str] = [],
    mode: str = "local"
):
    # Step 2.1: Save file to disk
    file_path = save_file_to_temp(file)

    # Step 2.2: Send to Daedalus Gateway
    result = await daedalus.receive_perceptual_information(
        file_path, tags
    )

    return result
```

**What Happens**:
1. File saved to `/tmp/flux_uploads/[uuid].pdf`
2. Passed to Daedalus Gateway for processing

---

### Step 3: Daedalus Gateway (daedalus.py)
**Location**: `backend/src/services/daedalus.py`

```python
def receive_perceptual_information(self, file_path, tags):
    # Single responsibility: Receive input
    # Pass to LangGraph workflow
    return self.document_processor.process(file_path, tags)
```

**What Happens**:
- File received
- Passed to LangGraph workflow for processing

---

### Step 4: LangGraph Workflow (document_processing_graph.py)
**Location**: `backend/src/services/document_processing_graph.py`

**6 Processing Nodes**:

#### Node 1: Extract & Process
```python
def extract_and_process(state):
    # Extract text from PDF/file
    text = extract_text(file_path)
    chunks = chunk_text(text)
    summary = summarize(chunks)

    return {
        "text": text,
        "chunks": chunks,
        "summary": summary
    }
```

**Outputs**:
- Full text content
- Text chunks (for processing)
- Summary

#### Node 2: Generate Research Plan
```python
def generate_research_plan(state):
    # ASI-GO-2 + R-Zero curiosity
    concepts = extract_concepts(state['text'])
    curiosity_triggers = identify_gaps(concepts)

    return {
        "concepts": concepts,
        "curiosity_triggers": curiosity_triggers
    }
```

**Outputs**:
- Extracted concepts
- Curiosity triggers (what to explore next)

#### Node 3: Consciousness Processing
```python
def process_consciousness(state):
    # Create attractor basins
    basins = create_basins(state['concepts'])

    # Generate thoughtseeds
    thoughtseeds = generate_thoughtseeds(
        state['text'],
        basins
    )

    return {
        "basins": basins,
        "thoughtseeds": thoughtseeds
    }
```

**Outputs**:
- Attractor basins (concept clusters)
- ThoughtSeeds (conceptual units)

#### Node 4: Analyze Results
```python
def analyze_results(state):
    quality_scores = calculate_quality(state)
    insights = extract_insights(state)

    return {
        "quality": quality_scores,
        "insights": insights
    }
```

**Outputs**:
- Quality scores
- Meta-cognitive insights

#### Node 5: Refine Processing (Optional)
```python
def refine_processing(state):
    # Iterative improvement if needed
    if quality_score < threshold:
        return improved_state
    return state
```

#### Node 6: Finalize Output
```python
def finalize_output(state):
    return {
        "extraction": {
            "concepts": state['concepts'],
            "chunks": len(state['chunks']),
            "summary": state['summary']
        },
        "consciousness": {
            "basins_created": len(state['basins']),
            "thoughtseeds_generated": len(state['thoughtseeds'])
        },
        "research": {
            "curiosity_triggers": state['curiosity_triggers']
        },
        "quality": state['quality']
    }
```

---

### Step 5: Storage (Neo4j + AutoSchemaKG)
**Location**: `backend/src/services/document_repository.py`

```python
def store_document(doc_data):
    # Step 5.1: Store in Neo4j
    CREATE (d:Document {
        id: doc_id,
        title: title,
        content: text,
        summary: summary,
        uploaded_at: timestamp,
        embedding: vector_embedding
    })

    # Step 5.2: Store concepts
    for concept in concepts:
        CREATE (c:Concept {name: concept})
        CREATE (d)-[:EXTRACTED]->(c)

    # Step 5.3: Store basins
    for basin in basins:
        CREATE (b:Basin {
            name: basin.name,
            stability: basin.stability
        })
        CREATE (d)-[:CREATES]->(b)

    # Step 5.4: Store thoughtseeds
    for seed in thoughtseeds:
        CREATE (ts:ThoughtSeed {
            content: seed.content,
            resonance: seed.resonance
        })
        CREATE (d)-[:GENERATES]->(ts)
```

**Data Stored**:
1. **Document node**: Full text, metadata
2. **Concept nodes**: Extracted concepts with relationships
3. **Basin nodes**: Attractor basins with stability scores
4. **ThoughtSeed nodes**: Conceptual units with resonance

---

### Step 6: Frontend Response
**Location**: `flux-desktop/src/pages/DocumentUpload.tsx:203-232`

```typescript
// Response received from backend
const uploadData = await response.json()

// Step 6.1: Update UI
setUploadedFiles(prev =>
  prev.map(f => f.id === fileEntry.id ? {
    ...f,
    status: 'completed',
    extraction: uploadData.extraction,        // ← Concepts, chunks, summary
    consciousness: uploadData.consciousness,  // ← Basins, thoughtseeds
    research: uploadData.research,            // ← Curiosity triggers
    quality: uploadData.quality               // ← Quality scores
  } : f)
)

// Step 6.2: Cache in localStorage
localStorage.setItem('flux:recent-documents', JSON.stringify({
  id: doc.id,
  title: doc.title,
  extraction: uploadData.extraction,
  quality: uploadData.quality
}))

// Step 6.3: Trigger UI refresh
window.dispatchEvent(new CustomEvent('flux:documents-updated'))
```

---

## 🔍 How to Debug Each Step

### Debug Frontend (Step 1)
**Open Browser DevTools** in the Tauri app:
1. Click app window
2. Press `Cmd+Option+I` (macOS) or `Ctrl+Shift+I` (Windows/Linux)
3. Go to Network tab
4. Upload a file
5. Watch for `POST /api/v1/documents`

**What to look for**:
- Request payload (FormData with file)
- Response status (should be 200)
- Response body (extraction, consciousness, research, quality)

### Debug Backend API (Step 2)
**Terminal 1: Start backend with logging**
```bash
cd /Volumes/Asylum/dev/Dionysus-2.0/backend
python3 -m uvicorn src.app_factory:app --host 127.0.0.1 --port 9127 --reload --log-level debug
```

**Watch for**:
```
INFO:     POST /api/v1/documents?mode=local
INFO:     File received: example.pdf (1.2MB)
INFO:     Saved to: /tmp/flux_uploads/abc123.pdf
INFO:     Sending to Daedalus...
```

### Debug Daedalus Gateway (Step 3)
**Add logging** to `backend/src/services/daedalus.py`:
```python
def receive_perceptual_information(self, file_path, tags):
    print(f"[DAEDALUS] Received file: {file_path}")
    print(f"[DAEDALUS] Tags: {tags}")

    result = self.document_processor.process(file_path, tags)

    print(f"[DAEDALUS] Processing complete!")
    print(f"[DAEDALUS] Concepts: {len(result['extraction']['concepts'])}")
    print(f"[DAEDALUS] Basins: {result['consciousness']['basins_created']}")

    return result
```

### Debug LangGraph Workflow (Step 4)
**Add node logging** to `backend/src/services/document_processing_graph.py`:
```python
def extract_and_process(state):
    print("[NODE 1] Extract & Process START")
    text = extract_text(state['file_path'])
    print(f"[NODE 1] Extracted {len(text)} characters")
    chunks = chunk_text(text)
    print(f"[NODE 1] Created {len(chunks)} chunks")
    print("[NODE 1] Extract & Process COMPLETE")
    return ...
```

### Debug Neo4j Storage (Step 5)
**Query Neo4j directly**:
```bash
# Open Neo4j Browser: http://localhost:7474

# Check documents
MATCH (d:Document)
RETURN d.id, d.title, d.uploaded_at
ORDER BY d.uploaded_at DESC
LIMIT 10

# Check concepts for a document
MATCH (d:Document {id: 'your-doc-id'})-[:EXTRACTED]->(c:Concept)
RETURN d.title, collect(c.name) as concepts

# Check basins
MATCH (d:Document {id: 'your-doc-id'})-[:CREATES]->(b:Basin)
RETURN b.name, b.stability

# Check thoughtseeds
MATCH (d:Document {id: 'your-doc-id'})-[:GENERATES]->(ts:ThoughtSeed)
RETURN ts.content, ts.resonance
LIMIT 5
```

### Debug Frontend Cache (Step 6)
**Open Browser Console** in Tauri app:
```javascript
// Check localStorage cache
JSON.parse(localStorage.getItem('flux:recent-documents'))

// Should show array of documents:
[
  {
    "id": "abc-123",
    "title": "example.pdf",
    "extraction": {
      "concepts": ["AI", "consciousness", "neural networks"],
      "chunks": 45
    },
    "quality": {
      "scores": {"overall": 0.87}
    }
  }
]
```

---

## 🎯 Quick Test: See Your Data Flow

### 1. Start Backend
```bash
cd /Volumes/Asylum/dev/Dionysus-2.0/backend
python3 -m uvicorn src.app_factory:app --host 127.0.0.1 --port 9127 --reload
```

### 2. Open Tauri App (already running)
The app is already running at PID 65665

### 3. Open DevTools
Press `Cmd+Option+I`

### 4. Upload a Test File
1. Go to "Upload" page in app
2. Drag & drop a PDF or markdown file
3. Watch Network tab in DevTools

### 5. See the Data!

**In Network Tab**:
- Request to `POST /api/v1/documents`
- Response with:
  - `extraction`: {"concepts": [...], "summary": "..."}
  - `consciousness`: {"basins_created": 3, "thoughtseeds_generated": 12}
  - `research`: {"curiosity_triggers": [...]}
  - `quality`: {"scores": {"overall": 0.85}}

**In Console Tab**:
```javascript
// Check cached data
localStorage.getItem('flux:recent-documents')
```

**In Neo4j Browser** (http://localhost:7474):
```cypher
MATCH (d:Document)
WHERE d.uploaded_at > datetime() - duration('PT1H')  // Last hour
RETURN d
```

---

## 🐛 Common Issues & Solutions

### Issue 1: Backend Not Starting
**Error**: `ModuleNotFoundError: No module named 'models.query'`

**Fix**: Import error in backend code
```bash
cd backend
grep -r "from models.query" src/
# Fix the import path
```

### Issue 2: No Data in Neo4j
**Check**: Is Neo4j running?
```bash
brew services list | grep neo4j
# or
neo4j status
```

**Start Neo4j**:
```bash
brew services start neo4j
# Wait 30 seconds, then check http://localhost:7474
```

### Issue 3: Upload Hangs
**Check**: Backend status indicator in app sidebar
- 🟢 Green: Backend connected
- 🔴 Red: Backend offline

**Solution**: Start backend (see Step 1 above)

---

## 📊 Data Persistence Locations

### 1. Neo4j Database
**Location**: `/usr/local/var/neo4j/data/`
**Type**: Graph database
**Contains**: Documents, Concepts, Basins, ThoughtSeeds, Relationships
**Persistent**: YES ✅

### 2. LocalStorage (Browser)
**Location**: Tauri app's browser storage
**Type**: JSON cache
**Contains**: Recent documents metadata only (not full content)
**Persistent**: YES ✅ (until cleared)

### 3. Temporary Files
**Location**: `/tmp/flux_uploads/`
**Type**: Original uploaded files
**Contains**: PDF, markdown, etc.
**Persistent**: NO ❌ (cleared on reboot)

---

## ✅ Verification Checklist

After uploading a file, verify data reached each step:

- [ ] **Step 1**: Browser DevTools shows POST request succeeded (200 OK)
- [ ] **Step 2**: Backend logs show "File received"
- [ ] **Step 3**: Backend logs show "Processing complete"
- [ ] **Step 4**: Response includes `extraction`, `consciousness`, `research`, `quality`
- [ ] **Step 5**: Neo4j contains Document node with your file
- [ ] **Step 6**: localStorage has document in cache
- [ ] **Step 6**: Sidebar shows document in recent files

---

**That's it!** Now you can see exactly where every piece of your data goes! 🎉
