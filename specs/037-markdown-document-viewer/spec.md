# Spec 037: Markdown Document Viewer with Consciousness Annotations

**Feature**: Enhanced markdown viewer page with consciousness-enhanced concept navigation
**Type**: New Page (preserves existing DocumentDetail)
**Priority**: P1 (High)
**Complexity**: Medium
**Timeline**: 2 weeks

---

## Problem Statement

### Current State
- **DocumentDetail page** (`/document/:id`) shows raw document data in JSON/text format
- No syntax highlighting or formatted markdown rendering
- Concepts are extracted in backend but not highlighted in frontend
- Basins and ThoughtSeeds are created but not visually linked to content
- No way to navigate from document text to knowledge graph

### User Pain Points
1. **Poor Readability**: Raw text is hard to read, no formatting preservation
2. **Lost Context**: Can't see which parts of text created which basins/concepts
3. **No Navigation**: Extracted concepts not clickable to explore graph
4. **Hidden Intelligence**: Consciousness processing results not visible in document view

### What We're NOT Changing
✅ Keep existing DocumentDetail page (`/document/:id`)
✅ Keep existing navigation structure
✅ Keep existing upload/processing workflow
✅ Keep existing data structures

---

## Solution Overview

### New Route: `/reader/:id`
**Purpose**: Enhanced markdown reading experience with consciousness annotations

**What It Adds**:
1. Beautifully rendered markdown (syntax highlighting, tables, images)
2. Inline concept highlighting (clickable → navigate to graph)
3. Basin markers showing which text segments created basins
4. ThoughtSeed connections visible inline
5. Collapsible metadata sections (quality scores, processing stages)

**User Flow**:
```
DocumentDetail page (/document/:id)
    ↓
User clicks "Open in Reader" button
    ↓
Reader page (/reader/:id)
    ↓
Rendered markdown + consciousness annotations
    ↓
Click concept → Navigate to /knowledge-graph?concept=X
```

---

## Functional Requirements

### FR-001: Markdown Rendering
**Description**: Render document markdown with syntax highlighting

**Acceptance Criteria**:
- [ ] Display document content as formatted markdown
- [ ] Code blocks render with syntax highlighting (language auto-detect)
- [ ] Tables render with proper formatting
- [ ] Images display responsively (if document contains images)
- [ ] Heading hierarchy preserved (h1, h2, h3)
- [ ] Blockquotes, lists, links render correctly

**Implementation**:
- Use `react-markdown` + `rehype-highlight` for rendering
- Extract markdown from `document.extraction.markdown` (already in backend)
- Fallback to raw content if markdown not available

### FR-002: Concept Highlighting
**Description**: Highlight extracted concepts inline with navigation

**Acceptance Criteria**:
- [ ] Concepts from `document.extraction.concepts` highlighted in text
- [ ] Distinct visual style (subtle bg color, not intrusive)
- [ ] Hover shows concept metadata (basin link, confidence)
- [ ] Click navigates to `/knowledge-graph?concept={concept_name}`
- [ ] Multiple occurrences of same concept all highlighted

**Implementation**:
```tsx
// Pseudo-code
<ReactMarkdown
  components={{
    text: ({ node }) => {
      const concept = findConceptInText(node.value, document.extraction.concepts)
      if (concept) {
        return (
          <ConceptHighlight
            concept={concept}
            onClick={() => navigate(`/knowledge-graph?concept=${concept.name}`)}
          >
            {node.value}
          </ConceptHighlight>
        )
      }
      return node.value
    }
  }}
>
  {document.extraction.markdown}
</ReactMarkdown>
```

### FR-003: Basin Context Markers
**Description**: Show which text segments created attractor basins

**Acceptance Criteria**:
- [ ] Basin markers appear next to relevant paragraphs
- [ ] Marker shows basin name + strength
- [ ] Hover reveals basin details (layer influences, emergence pattern)
- [ ] Click navigates to basin detail view (if exists)
- [ ] Visual indicator distinguishes high-strength vs low-strength basins

**Data Source**: `document.consciousness.basins` (array of basins created during processing)

### FR-004: ThoughtSeed Connections
**Description**: Display thoughtseed links inline

**Acceptance Criteria**:
- [ ] ThoughtSeed markers where concepts triggered thoughtseed creation
- [ ] Shows linked documents (cross-document connections)
- [ ] Hover reveals thoughtseed insight/question
- [ ] Click opens thoughtseed detail panel
- [ ] Badge shows number of linked documents

**Data Source**: `document.consciousness.thoughtseeds`

### FR-005: Collapsible Metadata
**Description**: Quality scores and processing info available but not intrusive

**Acceptance Criteria**:
- [ ] Collapsible "Processing Details" section at top
- [ ] Shows quality scores (overall, readability, coherence)
- [ ] Shows processing stages completed (6 LangGraph nodes)
- [ ] Shows timing metrics (processing duration)
- [ ] Defaults to collapsed (user expands if interested)

**Data Source**: `document.quality`, `document.processing_metadata`

### FR-006: Reading Mode Toggle
**Description**: Switch between annotated and clean reading modes

**Acceptance Criteria**:
- [ ] Toggle button: "Annotated" / "Clean" reading mode
- [ ] Clean mode: Just markdown, no highlights/markers
- [ ] Annotated mode: Full consciousness annotations
- [ ] Preference saved in localStorage
- [ ] Smooth transition between modes (no page reload)

---

## Non-Functional Requirements

### NFR-001: Performance
- Markdown rendering: <100ms for 10,000 word documents
- Concept highlighting: <50ms for 100 concepts
- Smooth scrolling with 100+ annotations

### NFR-002: Accessibility
- Keyboard navigation for concept highlights
- Screen reader support for annotations
- ARIA labels for all interactive elements
- Sufficient color contrast (WCAG AA)

### NFR-003: Responsive Design
- Works on desktop, tablet, mobile
- Reading width optimized (60-80 characters per line)
- Sidebar for metadata on desktop, accordion on mobile

---

## Technical Design

### Component Architecture
```
ReaderPage (/reader/:id)
├── ReaderHeader
│   ├── Back to Detail button
│   ├── Annotated/Clean toggle
│   └── Share/Export options
├── ReaderSidebar (desktop only)
│   ├── Table of Contents (from headings)
│   ├── Quality Metrics (collapsible)
│   └── Processing Stages (collapsible)
├── ReaderContent
│   ├── MarkdownRenderer
│   │   ├── ConceptHighlight (inline)
│   │   ├── BasinMarker (margin)
│   │   └── ThoughtSeedBadge (inline)
│   └── CodeBlock (syntax highlighted)
└── ReaderFooter
    └── Related Documents (thoughtseed links)
```

### Data Flow
```
1. User navigates to /reader/:id
2. Fetch document from /api/documents/:id (already exists)
3. Extract rendering data:
   - markdown: document.extraction.markdown
   - concepts: document.extraction.concepts
   - basins: document.consciousness.basins
   - thoughtseeds: document.consciousness.thoughtseeds
4. Render with annotations
5. User interactions:
   - Click concept → navigate to graph
   - Click basin → show basin detail
   - Click thoughtseed → open related docs
```

### Styling Approach
```css
/* Concept Highlighting */
.concept-highlight {
  background: rgba(59, 130, 246, 0.1); /* blue-500 at 10% */
  border-bottom: 1px dashed rgba(59, 130, 246, 0.3);
  cursor: pointer;
  transition: background 0.2s;
}

.concept-highlight:hover {
  background: rgba(59, 130, 246, 0.2);
}

/* Basin Markers */
.basin-marker {
  position: absolute;
  left: -2rem;
  width: 1.5rem;
  height: 1.5rem;
  border-radius: 50%;
  background: var(--basin-color);
  opacity: 0.6;
}

/* ThoughtSeed Badges */
.thoughtseed-badge {
  display: inline-flex;
  align-items: center;
  padding: 0.125rem 0.5rem;
  background: rgba(168, 85, 247, 0.1); /* purple-500 */
  border-radius: 9999px;
  font-size: 0.75rem;
  margin-left: 0.25rem;
}
```

---

## Integration Points

### Backend API (No Changes Required)
- ✅ `/api/documents/:id` already returns all needed data
- ✅ Markdown in `extraction.markdown`
- ✅ Concepts in `extraction.concepts`
- ✅ Basins in `consciousness.basins`
- ✅ ThoughtSeeds in `consciousness.thoughtseeds`

### Frontend Routes (Add New)
```tsx
// In App.tsx - ADD this route
import Reader from './pages/Reader'

<Route path="/reader/:id" element={<Reader />} />
```

### Existing Pages (Minor Update)
```tsx
// In DocumentDetail.tsx - ADD button
<Button
  onClick={() => navigate(`/reader/${documentId}`)}
  variant="secondary"
>
  📖 Open in Reader
</Button>
```

---

## User Stories

### Story 1: Research Reading
**As a** researcher
**I want to** read documents with concept highlighting
**So that** I can quickly navigate to related knowledge graph nodes

**Acceptance**: Click highlighted concept → knowledge graph opens with that concept

### Story 2: Understanding Processing
**As a** user
**I want to** see which text created which basins
**So that** I understand how consciousness processing worked

**Acceptance**: Basin markers show exact text segments that created basins

### Story 3: Cross-Document Discovery
**As a** user
**I want to** see thoughtseed connections while reading
**So that** I can discover related documents

**Acceptance**: ThoughtSeed badges clickable → opens related document list

### Story 4: Distraction-Free Reading
**As a** user
**I want to** toggle annotations off
**So that** I can read without distractions

**Acceptance**: Clean mode shows pure markdown, no highlights

---

## Testing Strategy

### Unit Tests
- [ ] Concept highlighting logic (match text to concepts)
- [ ] Basin marker positioning (paragraph association)
- [ ] ThoughtSeed badge rendering (count, links)
- [ ] Mode toggle state management

### Integration Tests
- [ ] Document fetch → render pipeline
- [ ] Concept click → graph navigation
- [ ] Basin marker → detail panel
- [ ] ThoughtSeed → related docs

### E2E Tests
- [ ] Full reading experience (load → interact → navigate)
- [ ] Performance (large document rendering)
- [ ] Mobile responsiveness

---

## Success Metrics

### User Engagement
- **Target**: 60% of users open Reader view at least once
- **Measure**: Track `/reader/:id` page views vs `/document/:id`

### Navigation Efficiency
- **Target**: 40% of concept highlights clicked
- **Measure**: Click-through rate on highlighted concepts

### User Satisfaction
- **Target**: 4.5/5 rating for reading experience
- **Measure**: In-app feedback survey

### Performance
- **Target**: <200ms p95 rendering time
- **Measure**: Web Vitals LCP for Reader page

---

## Future Enhancements (Out of Scope)

### Phase 2 Additions
- [ ] Inline annotations (user comments on text)
- [ ] Highlight multiple concepts by category (toggle filters)
- [ ] Export annotated markdown (with concept links)
- [ ] Side-by-side comparison (two documents)
- [ ] Reading progress tracking (resume where left off)

### Phase 3 Innovations
- [ ] AI-powered reading assistant (ask questions while reading)
- [ ] Dynamic concept discovery (LLM suggests new concepts)
- [ ] Collaborative annotations (multi-user comments)

---

## Dependencies

### External Libraries
- `react-markdown`: Markdown rendering (already used?)
- `rehype-highlight`: Syntax highlighting
- `lucide-react`: Icons (already in project)

### Internal Dependencies
- Document data structure (stable)
- Knowledge graph navigation (already exists)
- ThoughtSeed system (already exists)
- Basin tracking (already exists)

---

## Open Questions

1. **Concept Matching**: How to handle partial word matches? (e.g., "neural" in "neural networks")
   - **Proposal**: Exact phrase match only, case-insensitive

2. **Basin Association**: How to determine which paragraph created which basin?
   - **Proposal**: Use basin metadata `source_text` field (if available), else show at document level

3. **Mobile Experience**: How to show basin markers on small screens?
   - **Proposal**: Inline badges instead of margin markers on mobile

4. **Performance**: Will 1000+ concept highlights slow down rendering?
   - **Proposal**: Virtualized rendering for large documents, lazy highlight on scroll

---

## Appendix: SurfSense Comparison

### What We Adopt from SurfSense
✅ Markdown rendering with syntax highlighting
✅ Collapsible metadata sections
✅ Responsive reading layout
✅ Clean reading mode toggle

### What We Add Beyond SurfSense
🎯 Consciousness annotations (basins, thoughtseeds)
🎯 Interactive concept navigation
🎯 Cross-document discovery inline
🎯 Processing transparency (show LangGraph stages)

### Why This is Better
- **SurfSense**: Just renders content
- **Dionysus**: Renders content + reveals intelligence
- **Hybrid**: Beautiful UX + consciousness context
