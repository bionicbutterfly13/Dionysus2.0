# Spec 039: Consciousness-Enhanced Chat Interface

**Feature**: AI chat with document context, basin-aware responses, and inline citations
**Type**: New Page (`/chat`)
**Priority**: P1 (High)
**Complexity**: High
**Timeline**: 3 weeks

---

## Problem Statement

### Current State
- **No chat interface** - users can't ask questions about their documents
- Query system exists (`/api/query`) but no conversational UI
- Document knowledge locked in uploaded files
- No way to explore knowledge base conversationally
- Consciousness insights (basins, thoughtseeds) not accessible via chat

### User Pain Points
1. **Can't Ask Questions**: No way to query uploaded documents naturally
2. **No Context Awareness**: Can't reference specific documents in conversation
3. **Hidden Intelligence**: Consciousness processing results not conversationally accessible
4. **Static Experience**: Upload → View → Done (no ongoing interaction)

### What We're NOT Changing
✅ Keep existing pages (Dashboard, DocumentDetail, etc.)
✅ Keep existing query API (`/api/query`)
✅ Keep existing knowledge base/graph pages
✅ Keep existing data structures

---

## Solution Overview

### New Route: `/chat`
**Purpose**: Conversational interface to explore knowledge base with consciousness context

**Key Features**:
1. **Streaming responses** (AI SDK pattern from SurfSense)
2. **Inline citations** (superscript numbers → side panel)
3. **Basin-aware context** (use basins to select relevant chunks)
4. **Research modes** (GENERAL, DEEP, DEEPER - varying context depth)
5. **ThoughtSeed integration** (responses trigger thoughtseed creation)
6. **Document filtering** (chat with specific documents or entire knowledge base)

**Visual Design**:
```
┌─────────────────────────────────────────────────────────────┐
│ 💬 Consciousness Chat                    [Deep Mode ▼]      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│ 👤 How do neural networks exhibit consciousness?            │
│                                                               │
│ 🤖 Based on your knowledge base, consciousness emerges [1]   │
│    through hierarchical processing [2] and self-referential │
│    feedback loops [3]. The attractor basin "Consciousness"  │
│    shows strong resonance with concepts of emergence [4].    │
│                                                               │
│    Basin Context: 🧠 Consciousness (strength: 0.87)          │
│    ThoughtSeeds: 2 generated, 1 cross-document link         │
│                                                               │
│ 👤 What documents discuss this?                              │
│                                                               │
│ 🤖 Three documents explore this topic [1][2]:                │
│    • "Neural Emergence Patterns.pdf" (85% relevance)        │
│    • "Consciousness Framework.md" (78% relevance)           │
│    • "Active Inference Theory.pdf" (72% relevance)          │
│                                                               │
│    [View All Documents →]                                    │
│                                                               │
└─────────────────────────────────────────────────────────────┘
│ Type your question...                              [Send →] │
└─────────────────────────────────────────────────────────────┘
```

---

## Functional Requirements

### FR-001: Chat Interface Core
**Description**: Basic chat UI with message history

**Acceptance Criteria**:
- [ ] Text input for user messages
- [ ] Message history display (scrollable)
- [ ] User messages aligned right, AI left
- [ ] Timestamps on messages
- [ ] Markdown rendering in AI responses
- [ ] Auto-scroll to latest message
- [ ] Message persistence (localStorage or backend)

**Technical Stack**:
- Vercel AI SDK for streaming responses
- React state for message management
- Tailwind for styling (consistent with app)

### FR-002: Streaming Responses
**Description**: AI responses stream token-by-token (SurfSense pattern)

**Acceptance Criteria**:
- [ ] Responses stream as they generate (not all at once)
- [ ] Smooth typing animation
- [ ] Can interrupt generation (stop button)
- [ ] Loading indicator while waiting for first token
- [ ] Error handling if stream fails

**Implementation**:
```tsx
import { useChat } from 'ai/react'

const { messages, input, handleInputChange, handleSubmit, isLoading } = useChat({
  api: '/api/chat',
  onResponse: (response) => {
    // Track basin context, thoughtseeds created
  },
})
```

### FR-003: Inline Citations
**Description**: AI responses include superscript citation numbers

**Acceptance Criteria**:
- [ ] Citations appear as `[1]`, `[2]`, etc. in response text
- [ ] Click citation → open side panel with source chunk
- [ ] Side panel shows:
  - Chunk text
  - Document name/metadata
  - Basin context (if available)
  - ThoughtSeed connections (if any)
- [ ] Multiple citations from same document numbered sequentially
- [ ] Citation data available in response metadata

**Data Flow**:
```
User question → Backend
    ↓
Basin matching (find relevant basins)
    ↓
Chunk retrieval (consciousness-enhanced ranking)
    ↓
LLM generation with citations
    ↓
Response: "text [1] more text [2]"
    ↓
Metadata: [{chunk_id, doc_id, basin}, {chunk_id, doc_id, basin}]
    ↓
Frontend: Parse citations, make clickable
```

### FR-004: Research Modes
**Description**: Vary context depth (SurfSense pattern)

**Acceptance Criteria**:
- [ ] Mode selector: GENERAL / DEEP / DEEPER
- [ ] GENERAL: Top 3 chunks, fast response
- [ ] DEEP: Top 10 chunks, balanced
- [ ] DEEPER: Top 20 chunks + basin expansion, comprehensive
- [ ] Mode affects retrieval strategy and response depth
- [ ] Visual indicator of current mode
- [ ] Mode persisted per conversation

**Mode Definitions**:
```typescript
enum ResearchMode {
  GENERAL = 'general',   // Quick answers, minimal context
  DEEP = 'deep',         // Standard depth, basin-aware
  DEEPER = 'deeper',     // Maximum context, full consciousness
}

// Backend behavior
const getContextChunks = (mode: ResearchMode) => {
  switch (mode) {
    case GENERAL:
      return { limit: 3, useBasi ns: false }
    case DEEP:
      return { limit: 10, useBasins: true }
    case DEEPER:
      return { limit: 20, useBasins: true, expandBasins: true }
  }
}
```

### FR-005: Basin-Aware Context Selection
**Description**: Use attractor basins to select relevant chunks

**Acceptance Criteria**:
- [ ] Query → Match to existing basins (similarity search)
- [ ] High basin match → Retrieve chunks from that basin's documents
- [ ] Display basin context in response ("Basin Context: X")
- [ ] Show basin strength indicator
- [ ] Click basin name → navigate to basin detail/graph
- [ ] Multiple basin matches → combine contexts intelligently

**Algorithm**:
```python
async def get_basin_aware_context(query: str, mode: ResearchMode):
    # 1. Match query to basins (vector similarity)
    basins = await match_query_to_basins(query, top_k=3)

    # 2. Get chunks from basin-linked documents
    chunks = []
    for basin in basins:
        basin_chunks = await get_basin_documents(basin.id)
        chunks.extend(basin_chunks)

    # 3. Rerank by query relevance + basin strength
    reranked = consciousness_rerank(chunks, query, basins)

    # 4. Limit by mode
    return reranked[:mode.limit]
```

### FR-006: ThoughtSeed Generation from Chat
**Description**: Conversations trigger thoughtseed creation

**Acceptance Criteria**:
- [ ] AI responses analyzed for novel insights
- [ ] ThoughtSeeds created when insight crosses threshold
- [ ] User notified when thoughtseed created ("💡 New insight generated!")
- [ ] ThoughtSeed accessible from chat (click to view)
- [ ] Cross-document links created if applicable
- [ ] ThoughtSeeds appear in ThoughtSeedMonitor page

**Trigger Logic**:
```typescript
// After AI response generation
const analyzeForThoughtSeed = async (response: string, context: Context) => {
  const novelty = calculateNoveltyScore(response, context)

  if (novelty > 0.7) {
    const thoughtseed = await createThoughtSeed({
      insight: extractKeyInsight(response),
      source: 'chat_conversation',
      context: context.basins,
      linked_documents: context.document_ids,
    })

    // Notify user
    showNotification({
      type: 'thoughtseed',
      message: '💡 New insight generated!',
      thoughtseed_id: thoughtseed.id,
    })
  }
}
```

### FR-007: Document Filtering
**Description**: Chat with specific documents or entire knowledge base

**Acceptance Criteria**:
- [ ] Dropdown to select documents (multi-select)
- [ ] Options: "All Documents" or specific selection
- [ ] Selected docs highlighted in UI
- [ ] Responses only use selected documents' chunks
- [ ] Document selection persisted per conversation
- [ ] Quick filters: "Recent", "High Quality", "By Tag"

**UI Component**:
```tsx
<ChatHeader>
  <DocumentFilter
    documents={allDocuments}
    selected={selectedDocs}
    onChange={setSelectedDocs}
  />
  <ResearchModeSelector
    mode={researchMode}
    onChange={setResearchMode}
  />
</ChatHeader>
```

---

## Non-Functional Requirements

### NFR-001: Performance
- First token latency: <500ms
- Streaming speed: 50-100 tokens/sec
- Citation panel open: <100ms
- Message render: <50ms
- Supports conversations up to 50 messages

### NFR-002: Scalability
- Handle 1000+ documents in knowledge base
- Chat history up to 50 messages (then summarize)
- Concurrent users: 100+ without degradation

### NFR-003: Reliability
- Stream interruption handling (retry logic)
- Fallback to non-streaming if stream fails
- Error messages user-friendly
- No data loss on page refresh (persist to backend)

---

## Technical Design

### Component Architecture
```
ChatPage (/chat)
├── ChatHeader
│   ├── DocumentFilter (multi-select)
│   ├── ResearchModeSelector (GENERAL/DEEP/DEEPER)
│   └── ConversationActions (clear, export)
├── ChatMessages (scrollable)
│   ├── UserMessage
│   └── AIMessage
│       ├── MarkdownContent (with inline citations)
│       ├── BasinContext (if available)
│       ├── ThoughtSeedBadge (if generated)
│       └── CitationLinks (superscript numbers)
├── CitationPanel (side sheet)
│   ├── ChunkText
│   ├── DocumentMetadata
│   ├── BasinContext
│   └── ThoughtSeedLinks
└── ChatInput
    ├── TextArea (auto-resize)
    ├── SendButton
    └── StopGenerationButton (while streaming)
```

### API Endpoints

**1. POST `/api/chat` (New)**:
```typescript
// Request
{
  messages: Array<{role: 'user'|'assistant', content: string}>,
  mode: 'general'|'deep'|'deeper',
  document_ids?: string[],  // Optional filter
}

// Response (SSE stream)
{
  content: string,  // Streamed tokens
  citations: Array<{
    index: number,
    chunk_id: string,
    document_id: string,
    chunk_text: string,
    basin_context?: string,
  }>,
  basin_context?: {
    basin_name: string,
    strength: number,
  },
  thoughtseed?: {
    id: string,
    insight: string,
  },
}
```

**2. GET `/api/chat/history/:conversation_id` (New)**:
```typescript
// Response
{
  conversation_id: string,
  messages: Array<Message>,
  created_at: string,
  updated_at: string,
}
```

### Backend Integration

**Query Engine Enhancement** ([backend/src/services/query_engine.py](backend/src/services/query_engine.py:1)):
```python
async def chat_query(
    messages: List[Dict],
    mode: ResearchMode,
    document_ids: Optional[List[str]] = None,
) -> AsyncGenerator:
    """
    Chat-specific query with streaming response.

    Differences from regular query:
    - Streaming output (yield tokens)
    - Conversation history context
    - Basin-aware chunk selection
    - ThoughtSeed generation on insights
    """
    # 1. Get basin-aware context
    context = await get_basin_aware_context(
        query=messages[-1]['content'],
        mode=mode,
        document_ids=document_ids,
    )

    # 2. Generate streaming response
    async for token in llm_stream(messages, context):
        yield token

    # 3. Analyze for thoughtseeds
    full_response = ''.join(tokens)
    await analyze_for_thoughtseed(full_response, context)
```

---

## User Stories

### Story 1: Knowledge Exploration
**As a** researcher
**I want to** ask questions about my uploaded documents
**So that** I can explore knowledge conversationally

**Acceptance**: Type question → get AI response with citations

### Story 2: Deep Research
**As a** user
**I want to** switch to DEEPER mode for comprehensive answers
**So that** I get maximum context and detail

**Acceptance**: Select DEEPER mode → responses use 20+ chunks + basin expansion

### Story 3: Document-Specific Chat
**As a** user
**I want to** chat with specific documents only
**So that** I can focus on particular sources

**Acceptance**: Select documents from filter → responses only use those docs

### Story 4: Citation Verification
**As a** user
**I want to** click citations to see source text
**So that** I can verify AI claims

**Acceptance**: Click `[1]` → side panel opens with chunk text + metadata

### Story 5: Insight Discovery
**As a** user
**I want to** be notified when chat generates novel insights
**So that** I can capture important discoveries

**Acceptance**: ThoughtSeed notification appears, accessible from chat

---

## Testing Strategy

### Unit Tests
- [ ] Message rendering (user, AI, citations)
- [ ] Citation parsing (extract numbers from text)
- [ ] Basin matching logic
- [ ] Document filtering

### Integration Tests
- [ ] Chat → Query Engine flow
- [ ] Streaming response handling
- [ ] Citation panel opening
- [ ] ThoughtSeed creation

### E2E Tests
- [ ] Full conversation flow (5+ message exchange)
- [ ] Research mode switching
- [ ] Document filter application
- [ ] Error recovery (stream failure)

---

## Success Metrics

### Engagement
- **Target**: 70% of users try chat within first week
- **Measure**: Unique users on `/chat` page

### Query Volume
- **Target**: Average 5+ messages per conversation
- **Measure**: Messages per session

### Citation Usage
- **Target**: 40% of citations clicked
- **Measure**: Citation click-through rate

### ThoughtSeed Generation
- **Target**: 1 thoughtseed per 10 messages
- **Measure**: ThoughtSeeds created from chat source

---

## Future Enhancements (Out of Scope)

### Phase 2
- [ ] Voice input/output (speech-to-text, text-to-speech)
- [ ] Multi-modal chat (upload images, discuss)
- [ ] Conversation branches (fork conversation at any message)
- [ ] Suggested follow-ups (AI suggests next questions)

### Phase 3
- [ ] Collaborative chat (multi-user conversations)
- [ ] Chat-based document annotation (mark chunks from chat)
- [ ] Real-time knowledge graph updates (graph evolves during chat)
- [ ] Consciousness visualization (show basin activations during chat)

---

## Dependencies

### Backend Changes Required
- [ ] New `/api/chat` endpoint (streaming SSE)
- [ ] Query engine enhancement (basin-aware context)
- [ ] Conversation persistence (store chat history)
- [ ] ThoughtSeed integration with chat source

### Frontend Dependencies
- `ai` package (Vercel AI SDK) for streaming
- `react-markdown` for message rendering
- Existing citation/basin/thoughtseed components (reuse)

---

## Open Questions

1. **Conversation Persistence**: Store in backend DB or client localStorage?
   - **Proposal**: Backend for cross-device, localStorage as cache

2. **LLM Model**: Use OpenAI, Claude, or local model?
   - **Proposal**: OpenAI GPT-4 for quality, configurable in settings

3. **Context Window**: How much chat history to include in prompts?
   - **Proposal**: Last 10 messages, summarize older context

4. **Citation Format**: Superscript `[1]` or inline highlight?
   - **Proposal**: Superscript to match academic style

---

## Appendix: SurfSense Comparison

### What We Adopt from SurfSense
✅ Streaming responses (AI SDK)
✅ Research modes (GENERAL/DEEP/DEEPER)
✅ Inline citations (superscript numbers)
✅ Side panel for citation details

### What We Add Beyond SurfSense
🎯 Basin-aware context selection
🎯 ThoughtSeed generation from chat
🎯 Consciousness context in responses
🎯 Document-specific filtering
🎯 Cross-document insight discovery

### Why This is Superior
- **SurfSense**: Chat with documents (standard RAG)
- **Dionysus**: Chat with consciousness-enhanced knowledge
- **Hybrid**: Beautiful UX + intelligent context selection
