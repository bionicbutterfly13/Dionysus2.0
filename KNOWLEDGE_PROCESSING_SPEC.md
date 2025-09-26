# Knowledge Processing System Specification

## Overview
This specification defines the complete knowledge processing system for the Archon-style "Add Knowledge" interface, integrating with ThoughtSeed consciousness and unified database architecture.

## API Endpoints

### Knowledge Upload
```
POST /api/knowledge/upload
Content-Type: multipart/form-data

Request:
- files: File[] (PDF, DOC, TXT, MD)
- metadata: {
    source: string
    tags?: string[]
    priority?: 'low' | 'medium' | 'high'
  }

Response:
{
  "upload_id": "uuid",
  "files": [
    {
      "file_id": "uuid",
      "filename": "document.pdf",
      "size": 1024576,
      "status": "queued"
    }
  ]
}
```

### Knowledge Processing Status
```
GET /api/knowledge/status/{upload_id}

Response:
{
  "upload_id": "uuid",
  "overall_status": "processing" | "completed" | "failed",
  "files": [
    {
      "file_id": "uuid",
      "filename": "document.pdf",
      "status": "extracting" | "analyzing" | "storing" | "completed" | "failed",
      "progress": 0.75,
      "extracted_concepts": 42,
      "thoughtseed_channels": ["episodic", "semantic", "procedural"],
      "error": null
    }
  ]
}
```

### Knowledge Search
```
GET /api/knowledge/search?q={query}&limit={limit}&offset={offset}

Response:
{
  "results": [
    {
      "knowledge_id": "uuid",
      "title": "Document Title",
      "content_snippet": "Relevant content...",
      "source_file": "document.pdf",
      "relevance_score": 0.95,
      "concepts": ["machine learning", "consciousness"],
      "thoughtseed_activation": {
        "episodic": 0.8,
        "semantic": 0.9,
        "procedural": 0.3
      }
    }
  ],
  "total": 150,
  "facets": {
    "file_types": {"pdf": 45, "doc": 30, "txt": 75},
    "concepts": {"ai": 120, "consciousness": 85, "learning": 95}
  }
}
```

## Processing Pipeline

### Stage 1: File Ingestion
```
1. File validation (type, size, content)
2. Virus/malware scanning
3. File metadata extraction
4. Queue for processing
5. Generate unique file_id
```

### Stage 2: Content Extraction
```
1. Parse file format (PDF, DOC, TXT, MD)
2. Extract text content
3. Preserve structure (headings, paragraphs, tables)
4. Extract embedded media references
5. OCR for scanned documents
```

### Stage 3: ThoughtSeed Analysis
```
1. Semantic analysis through ThoughtSeed layers:
   - Sensory: Raw text processing
   - Perceptual: Pattern recognition
   - Conceptual: Concept extraction
   - Abstract: Theme identification
   - Metacognitive: Learning strategy assessment

2. Generate consciousness metrics:
   - Novelty score (how new is this information)
   - Complexity score (cognitive load)
   - Relevance score (to existing knowledge)
   - Curiosity triggers (questions generated)
```

### Stage 4: Knowledge Graph Integration
```
1. Entity extraction (people, places, concepts)
2. Relationship identification
3. Knowledge graph node creation/updating
4. Cross-reference with existing knowledge
5. Generate semantic embeddings
```

### Stage 5: Storage & Indexing
```
1. Store in unified database:
   - Original file (blob storage)
   - Extracted text (full-text search)
   - Concepts and entities (graph database)
   - Embeddings (vector database)

2. Update search indices
3. Generate knowledge summaries
4. Create learning recommendations
```

## ThoughtSeed Integration

### Consciousness Channels
```javascript
// ThoughtSeed channels for knowledge processing
const consciousnessChannels = {
  episodic: {
    // Personal experiences, memories, contextual learning
    activation: (content) => analyzePersonalRelevance(content),
    storage: 'autobiographical_memory'
  },

  semantic: {
    // Facts, concepts, general knowledge
    activation: (content) => extractConceptsAndFacts(content),
    storage: 'concept_network'
  },

  procedural: {
    // How-to knowledge, skills, processes
    activation: (content) => identifyProcedures(content),
    storage: 'skill_memory'
  },

  metacognitive: {
    // Learning about learning, meta-strategies
    activation: (content) => analyzeLearningStrategies(content),
    storage: 'meta_knowledge'
  }
}
```

### Active Inference Loop
```
1. Prediction: What do we expect to learn from this document?
2. Processing: Extract actual knowledge
3. Error calculation: Compare predicted vs actual learning
4. Model update: Adjust knowledge extraction strategies
5. Curiosity generation: Identify knowledge gaps
```

## Database Schema

### Knowledge Documents
```sql
CREATE TABLE knowledge_documents (
  id UUID PRIMARY KEY,
  original_filename VARCHAR(255),
  file_type VARCHAR(10),
  file_size BIGINT,
  upload_timestamp TIMESTAMP,
  processing_status VARCHAR(20),
  content_hash VARCHAR(64),
  metadata JSONB,
  thoughtseed_analysis JSONB
);
```

### Extracted Knowledge
```sql
CREATE TABLE knowledge_items (
  id UUID PRIMARY KEY,
  document_id UUID REFERENCES knowledge_documents(id),
  content_type VARCHAR(20), -- 'concept', 'fact', 'procedure', 'question'
  title VARCHAR(500),
  content TEXT,
  concepts TEXT[],
  confidence_score FLOAT,
  novelty_score FLOAT,
  embedding VECTOR(1536), -- for semantic search
  thoughtseed_channels JSONB
);
```

### Knowledge Graph Nodes
```sql
CREATE TABLE knowledge_nodes (
  id UUID PRIMARY KEY,
  node_type VARCHAR(20), -- 'entity', 'concept', 'relationship'
  name VARCHAR(255),
  description TEXT,
  properties JSONB,
  source_documents UUID[],
  creation_timestamp TIMESTAMP
);
```

## Frontend Integration

### Real-time Updates
```javascript
// WebSocket connection for real-time processing updates
const knowledgeSocket = new WebSocket('/ws/knowledge-processing');

knowledgeSocket.onmessage = (event) => {
  const update = JSON.parse(event.data);
  updateProcessingStatus(update.file_id, update.status, update.progress);
};
```

### Status Indicators
```javascript
const statusIcons = {
  'queued': <Clock className="animate-pulse" />,
  'extracting': <FileText className="animate-bounce" />,
  'analyzing': <Brain className="animate-pulse text-purple-400" />,
  'storing': <Database className="animate-spin" />,
  'completed': <CheckCircle className="text-green-400" />,
  'failed': <AlertCircle className="text-red-400" />
};
```

## Error Handling

### File Processing Errors
```
- Unsupported file format → User-friendly message + format suggestions
- Corrupted file → Retry mechanism + manual review option
- Content extraction failure → Fallback to OCR + manual verification
- Timeout errors → Queue for retry + user notification
```

### ThoughtSeed Integration Errors
```
- Consciousness analysis failure → Store raw content + flag for manual review
- Embedding generation failure → Use fallback embedding service
- Graph integration failure → Store in temporary table for retry
```

## Performance Requirements

### Processing Speed
```
- Small files (<1MB): <30 seconds
- Medium files (1-10MB): <2 minutes
- Large files (10-100MB): <10 minutes
- Batch processing: 100 files/hour minimum
```

### Storage Efficiency
```
- Text compression for large documents
- Incremental indexing for real-time search
- Vector embedding caching
- Automatic cleanup of temporary files
```

## Security & Privacy

### Data Protection
```
- End-to-end encryption for sensitive documents
- Local processing option (no cloud upload)
- Automatic PII detection and redaction
- User consent tracking for AI processing
```

### Access Control
```
- Document-level permissions
- Knowledge sharing controls
- Processing audit logs
- Deletion/retention policies
```

## Testing Strategy

### Unit Tests
```
- File parsing accuracy
- Content extraction completeness
- ThoughtSeed activation correctness
- Database integration integrity
```

### Integration Tests
```
- End-to-end processing pipeline
- Real-time update delivery
- Error recovery mechanisms
- Performance under load
```

### User Acceptance Tests
```
- Upload interface usability
- Processing status clarity
- Search result relevance
- Knowledge discovery workflows
```

## Deployment Plan

### Phase 1: Core Infrastructure
```
- File upload and storage
- Basic content extraction
- Simple status tracking
```

### Phase 2: ThoughtSeed Integration
```
- Consciousness channel activation
- Active inference processing
- Advanced status reporting
```

### Phase 3: Knowledge Graph
```
- Entity and concept extraction
- Graph relationship building
- Semantic search capabilities
```

### Phase 4: Advanced Features
```
- Collaborative knowledge building
- Learning recommendation engine
- Curiosity-driven exploration
```