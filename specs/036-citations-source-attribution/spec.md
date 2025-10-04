# Feature Specification: Citations & Source Attribution

**Spec ID**: 036
**Feature Name**: Citations & Source Attribution System
**Status**: Draft
**Created**: 2025-10-03
**Dependencies**: Document Processing (Spec 021), Knowledge Graph (Spec 001)
**Source**: SurfSense project analysis

## Overview

Implement a comprehensive citation and source attribution system that provides transparent, verifiable links between AI-generated insights and source documents. When the system generates research questions, analysis, or summaries from documents, it must track and display which specific document chunks informed each statement, with clickable citations that navigate users to the exact source material.

## Motivation

**Trust & Transparency**: Users need to verify AI-generated insights by examining source material directly.

**Knowledge Provenance**: Consciousness-enhanced document processing generates rich insights (basins, thoughtseeds, curiosity triggers) - users must understand which documents contributed to these insights.

**Navigation**: Citations become interactive pathways through the knowledge graph, enabling exploration from insights back to source documents and forward to related concepts.

**Regulatory Compliance**: Source attribution supports academic/research use cases requiring proper citations.

## Goals

### Primary Goals
- Track document chunk provenance for all generated insights
- Display inline citations in research questions, summaries, and analysis
- Implement clickable citations that navigate to source documents
- Show confidence scores for each citation
- Support multiple citation formats (inline, footnotes, bibliography)
- Integrate with existing DocumentDetail page for seamless navigation

### Success Metrics
- **Citation Accuracy**: 100% of generated insights have traceable sources
- **Navigation Latency**: <100ms to navigate from citation to source
- **UI Clarity**: Users can identify cited vs uncited content at a glance
- **Confidence Visibility**: All citations display confidence scores
- **Coverage**: ≥80% of document chunks get cited in at least one insight

## Non-Goals

- **Academic Citation Formats**: APA, MLA, Chicago style formatting (deferred to future)
- **Citation Export**: Export citations to reference managers (deferred)
- **Cross-Document Citation Networks**: Graph visualization of citation relationships (deferred to Spec 030)
- **Citation Editing**: Manual citation management by users (out of scope)

## Functional Requirements

### FR-001: Citation Data Model
**Priority**: P0 (Critical)
**Description**: Define citation schema in Neo4j and backend

**Acceptance Criteria**:
- Citation nodes store: `source_doc_id`, `chunk_id`, `chunk_text`, `confidence_score`, `position_start`, `position_end`
- Relationships: `(:Insight)-[:CITED_FROM]->(:DocumentChunk)`
- Insights include: research questions, summaries, basin descriptions, thoughtseed content
- Each citation includes metadata: `created_at`, `citation_type`, `relevance_score`

**Test Cases**:
```python
def test_citation_creation():
    citation = create_citation(
        insight_id="insight_123",
        chunk_id="chunk_456",
        confidence=0.92,
        chunk_text="Neural networks exhibit emergent properties..."
    )
    assert citation.confidence == 0.92
    assert citation.chunk_text is not None
    assert citation.relationships["CITED_FROM"] is not None
```

### FR-002: Citation Extraction During Processing
**Priority**: P0 (Critical)
**Description**: Extract citations during document processing workflow

**Acceptance Criteria**:
- LangGraph document processing nodes track source chunks for all generated content
- Research plan generation (ASI-GO-2 Researcher) records which chunks inspired each question
- Consciousness processing (basins, thoughtseeds) links to originating document chunks
- Quality analysis cites specific passages supporting quality scores
- Citations stored in Neo4j during final workflow node

**Test Cases**:
```python
def test_research_question_citations():
    result = daedalus.receive_perceptual_information(file_obj, tags=["test"])
    questions = result["research"]["curiosity_triggers"]
    for question in questions:
        assert "citations" in question
        assert len(question["citations"]) > 0
        assert all(c["chunk_id"] for c in question["citations"])
```

### FR-003: Citation Display in Document Detail
**Priority**: P0 (Critical)
**Description**: Show citations in DocumentDetail page with visual indicators

**Acceptance Criteria**:
- Research questions display superscript citation numbers `[1][2]`
- Clicking citation number opens inline preview of source chunk
- Preview shows: chunk text, confidence score, "View in document" button
- Cited text highlighted in document content preview
- Citations grouped by section (research questions, quality metrics, etc.)

**UI Mockup**:
```
Research Questions
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. How do emergent properties arise in neural networks? [1][2]
   ↳ [1] "Neural networks exhibit emergent..." (92% confidence)
   ↳ [2] "Consciousness may emerge from..." (87% confidence)

2. What role does attention play in consciousness? [3]
   ↳ [3] "Attention mechanisms focus..." (95% confidence)
```

**Test Cases**:
```typescript
test('citation display in research questions', () => {
  render(<DocumentDetail id="doc_123" />)
  const question = screen.getByText(/emergent properties/i)
  const citation = within(question).getByText('[1]')

  fireEvent.click(citation)
  expect(screen.getByText(/Neural networks exhibit/i)).toBeVisible()
  expect(screen.getByText(/92% confidence/i)).toBeVisible()
})
```

### FR-004: Citation Navigation
**Priority**: P1 (High)
**Description**: Navigate from citation to source document chunk

**Acceptance Criteria**:
- "View in document" button navigates to source document
- URL includes fragment identifier for chunk position: `/document/doc_123#chunk_456`
- Document content scrolls to cited chunk automatically
- Cited chunk highlighted with subtle background color
- Breadcrumb shows: Document > Section > Cited Chunk

**Test Cases**:
```typescript
test('citation navigation to source', async () => {
  render(<CitationPreview citation={{chunk_id: "chunk_456", doc_id: "doc_123"}} />)
  const viewButton = screen.getByText(/view in document/i)

  fireEvent.click(viewButton)
  await waitFor(() => {
    expect(window.location.pathname).toBe('/document/doc_123')
    expect(window.location.hash).toBe('#chunk_456')
  })
})
```

### FR-005: Citations in Chat (Future Integration)
**Priority**: P2 (Medium - deferred to Chat spec)
**Description**: Display citations in future chat interface

**Acceptance Criteria**:
- Chat responses include inline citations `[1][2][3]`
- Citations appear as the response streams
- Hover shows citation preview tooltip
- Click opens citation panel on right side

**Note**: Full implementation deferred until Chat Interface spec (Spec 039)

### FR-006: Citation Confidence Scores
**Priority**: P1 (High)
**Description**: Calculate and display citation relevance/confidence

**Acceptance Criteria**:
- Confidence based on: semantic similarity, chunk length, position in document
- Formula: `confidence = 0.6 * similarity + 0.3 * (1 / position_penalty) + 0.1 * length_bonus`
- Display confidence as percentage with color coding:
  - ≥90%: Green (high confidence)
  - 70-89%: Blue (medium confidence)
  - <70%: Yellow (low confidence)
- Sort citations by confidence (highest first)

**Test Cases**:
```python
def test_citation_confidence_calculation():
    citation = calculate_citation_confidence(
        similarity=0.92,
        chunk_position=2,  # Early in document
        chunk_length=150   # Good length
    )
    assert 0.85 <= citation.confidence <= 0.95
    assert citation.color_class == "text-green-400"
```

### FR-007: Backend API Endpoints
**Priority**: P0 (Critical)
**Description**: REST endpoints for citation management

**Acceptance Criteria**:
- `GET /api/v1/documents/{doc_id}/citations` - List all citations for a document
- `GET /api/v1/citations/{citation_id}` - Get citation details with chunk text
- `GET /api/v1/insights/{insight_id}/citations` - Get citations for specific insight
- Response includes: citation metadata, chunk text, confidence, navigation URL

**API Response Example**:
```json
{
  "citations": [
    {
      "id": "cite_789",
      "chunk_id": "chunk_456",
      "chunk_text": "Neural networks exhibit emergent properties...",
      "confidence": 0.92,
      "position": {"start": 1200, "end": 1450},
      "document": {
        "id": "doc_123",
        "title": "Computational Consciousness.pdf"
      },
      "navigation_url": "/document/doc_123#chunk_456"
    }
  ],
  "total": 15
}
```

## Technical Requirements

### TR-001: Neo4j Schema Extension
Add citation nodes and relationships:
```cypher
CREATE (c:Citation {
  id: "cite_789",
  chunk_id: "chunk_456",
  chunk_text: "Neural networks exhibit...",
  confidence: 0.92,
  position_start: 1200,
  position_end: 1450,
  created_at: timestamp()
})

CREATE (q:ResearchQuestion {content: "How do emergent properties arise?"})-[:CITED_FROM]->(c)
CREATE (c)-[:REFERENCES_CHUNK]->(chunk:DocumentChunk {id: "chunk_456"})
CREATE (chunk)-[:PART_OF]->(doc:Document {id: "doc_123"})
```

### TR-002: LangGraph Citation Tracking
Modify document processing nodes to track sources:
```python
# In document_researcher.py
async def generate_research_questions(state: ResearchState) -> dict:
    questions = []
    for concept in state.concepts:
        # Generate question using ASI-GO-2
        question_text = await researcher.generate_question(concept)

        # Track which chunks inspired this question
        citations = [
            Citation(
                chunk_id=chunk.id,
                chunk_text=chunk.text[:200],
                confidence=calculate_confidence(chunk, question_text)
            )
            for chunk in state.relevant_chunks
            if is_relevant(chunk, concept)
        ]

        questions.append({
            "question": question_text,
            "citations": citations
        })

    return {"research_questions": questions}
```

### TR-003: Frontend Citation Component
React component for citation display:
```typescript
interface CitationProps {
  citation: {
    id: string
    chunk_text: string
    confidence: number
    doc_id: string
    chunk_id: string
  }
  inline?: boolean  // Inline vs. expanded view
}

function Citation({ citation, inline = true }: CitationProps) {
  const [isExpanded, setIsExpanded] = useState(false)

  if (inline) {
    return (
      <sup className="citation-number" onClick={() => setIsExpanded(true)}>
        [{citation.number}]
      </sup>
    )
  }

  return (
    <div className="citation-preview">
      <p className="chunk-text">{citation.chunk_text}</p>
      <div className="citation-metadata">
        <Badge confidence={citation.confidence} />
        <Button onClick={() => navigate(`/document/${citation.doc_id}#${citation.chunk_id}`)}>
          View in document
        </Button>
      </div>
    </div>
  )
}
```

## UI/UX Requirements

### UX-001: Citation Visual Language
- **Inline citations**: Superscript numbers `[1]` in blue-400 color
- **Confidence badges**: Color-coded percentage (green/blue/yellow)
- **Hover preview**: 300ms delay before showing citation tooltip
- **Click behavior**: Opens inline expansion (not new page)
- **Keyboard navigation**: Tab through citations, Enter to expand

### UX-002: Citation Panel Layout
```
┌─────────────────────────────────────────────┐
│ Research Questions                          │
├─────────────────────────────────────────────┤
│ 1. How do emergent properties arise? [1][2]│
│    ┌──────────────────────────────────┐    │
│    │ [1] Neural networks exhibit... 92%│    │
│    │     [View in document →]          │    │
│    └──────────────────────────────────┘    │
│    ┌──────────────────────────────────┐    │
│    │ [2] Consciousness may emerge... 87%│   │
│    │     [View in document →]          │    │
│    └──────────────────────────────────┘    │
└─────────────────────────────────────────────┘
```

## Data Model

### Citation Schema
```typescript
interface Citation {
  id: string
  chunk_id: string
  chunk_text: string           // First 200 chars of chunk
  full_chunk_text?: string     // Optional full text
  confidence: number           // 0.0-1.0
  position: {
    start: number              // Character offset in document
    end: number
  }
  document: {
    id: string
    title: string
  }
  created_at: string
  citation_type: 'inline' | 'footnote' | 'bibliography'
  relevance_score: number
}

interface InsightWithCitations {
  id: string
  type: 'research_question' | 'summary' | 'basin' | 'thoughtseed'
  content: string
  citations: Citation[]
  confidence: number
}
```

## Implementation Plan

### Phase 1: Backend Foundation (Week 1)
1. Extend Neo4j schema with Citation nodes
2. Modify LangGraph nodes to track source chunks
3. Implement citation extraction in document_researcher.py
4. Add citation confidence calculation
5. Create REST API endpoints

### Phase 2: Frontend Display (Week 2)
1. Create Citation React component
2. Update DocumentDetail page with citation display
3. Implement inline citation numbers
4. Add citation preview expansion
5. Style confidence badges

### Phase 3: Navigation (Week 3)
1. Implement chunk-level navigation with URL fragments
2. Add auto-scroll to cited chunks
3. Implement chunk highlighting
4. Add breadcrumb navigation
5. Test cross-document navigation

### Phase 4: Polish & Testing (Week 4)
1. End-to-end testing of citation flow
2. Performance optimization (citation queries)
3. Accessibility testing (screen readers, keyboard nav)
4. Documentation and examples
5. User testing and refinement

## Dependencies

### Upstream Dependencies
- **Spec 021**: Document processing via Daedalus/LangGraph (complete)
- **Spec 001**: Neo4j knowledge graph (complete)
- Current DocumentDetail page (complete)

### Downstream Dependencies
- **Spec 039**: Chat Interface (will use citations)
- **Spec 030**: Visual Testing Interface (citation visualization)
- **Spec 037**: Markdown Viewer (enhanced chunk display)

## Testing Strategy

### Unit Tests
- Citation confidence calculation
- Citation extraction from chunks
- Citation data model validation
- API endpoint responses

### Integration Tests
- End-to-end document processing with citation tracking
- Citation storage in Neo4j
- Citation retrieval via API
- Frontend citation display

### E2E Tests
- Upload document → process → view citations
- Click citation → navigate to source
- Expand/collapse citation previews
- Keyboard navigation through citations

## Success Criteria

**Definition of Done**:
- ✅ All research questions display inline citations
- ✅ Citations clickable and navigate to source chunks
- ✅ Confidence scores visible for all citations
- ✅ <100ms navigation latency
- ✅ 100% citation accuracy (all insights traceable)
- ✅ API documentation complete
- ✅ Unit test coverage ≥80%
- ✅ E2E tests passing

## Open Questions

1. **Citation Deduplication**: If multiple insights cite the same chunk, how do we handle display? (Answer: Show same chunk multiple times with context)

2. **Citation Versioning**: If document is re-processed, do old citations remain valid? (Answer: Defer to Spec 031 conflict resolution)

3. **Cross-Document Citations**: How do we show when an insight combines chunks from multiple documents? (Answer: Display multiple citations, group by document)

4. **Citation Limits**: Should we limit citations per insight to avoid UI clutter? (Answer: Show top 5, "+ N more" link)

## References

- **SurfSense**: `/Volumes/Asylum/dev/Flux/surfsense_web/components/chat/ChatCitation.tsx`
- **CLAUSE Provenance**: Spec 032 (emerges pattern detection has provenance requirements)
- **Knowledge Graph**: Spec 001 (Neo4j unified architecture)
