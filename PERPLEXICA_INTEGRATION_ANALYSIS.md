# Perplexica Integration Analysis for Dionysus-2.0

**Analysis Date**: 2025-10-04
**Perplexica Repository**: https://github.com/ItzCrazyKns/Perplexica
**License**: MIT
**Tech Stack**: Next.js, TypeScript, LangChain, SearxNG, Hugging Face Transformers

---

## Executive Summary

Perplexica is an open-source AI-powered search engine that serves as an alternative to Perplexity AI. After comprehensive analysis of the codebase, **12 high-value features** have been identified that would significantly enhance Dionysus-2.0's consciousness-enhanced document processing system. The most impactful integrations focus on:

1. **Streaming Response Architecture** - Real-time UI updates during processing
2. **Multi-Mode Search System** - Specialized processing modes (academic, technical, etc.)
3. **Embedding-Based Reranking** - Relevance-based result prioritization
4. **Citation Management** - Inline source attribution with markdown integration
5. **Query Rewriting Pipeline** - Context-aware query optimization

**Estimated Total Integration Effort**: 4-6 weeks
**High-Priority Features (Phase 1)**: 2 features, ~1 week
**Medium-Priority Features (Phase 2)**: 5 features, ~2 weeks
**Advanced Features (Phase 3)**: 5 features, ~2-3 weeks

**ROI Assessment**: HIGH - These features directly address current gaps in Dionysus-2.0's user interface, search capabilities, and multi-agent coordination, particularly for CLAUSE Phase 2.

---

## 1. Feature Inventory & Benefit Analysis

### 1.1 Streaming Response System ⭐⭐⭐⭐⭐

**Description**: Real-time event-driven response streaming from backend to frontend with typed event handling.

**Implementation Approach**:
- Backend: Server-Sent Events (SSE) using Next.js streaming
- Frontend: React hooks with `ReadableStream` API
- Event Types: `init`, `sources`, `message`, `messageEnd`, `error`
- Progressive message building with incremental UI updates

**Key Technical Details**:
```typescript
// Event Types
type StreamEvent =
  | { type: 'init', messageId: string }
  | { type: 'sources', data: Source[] }
  | { type: 'message', messageId: string, data: string }
  | { type: 'messageEnd', messageId: string }
  | { type: 'error', data: string }

// Frontend handling
const reader = res.body?.getReader();
const decoder = new TextDecoder('utf-8');
while (true) {
  const { value, done } = await reader.read();
  if (done) break;
  const messages = decoder.decode(value).split('\n');
  for (const msg of messages) {
    const event = JSON.parse(msg);
    await messageHandler(event);
  }
}
```

**Benefits for Dionysus-2.0**:
- ✅ **Real-time consciousness processing feedback** - Users see attractor basin formation, thoughtseed generation, and research steps as they happen
- ✅ **Enhanced UX for LangGraph workflows** - Visualize each of the 6 processing nodes in real-time
- ✅ **Multi-agent coordination visibility** - Stream PathNavigator decisions, ContextCurator selections, and LC-MAPPO coordination
- ✅ **Reduced perceived latency** - Users engage with partial results while processing continues
- ✅ **Better debugging** - Real-time insight into processing pipeline stages

**Integration Complexity**: Medium
- Requires FastAPI streaming endpoint modifications
- Frontend React hooks for event handling
- State management for progressive updates

**Priority**: **MUST-HAVE (Phase 1)**

---

### 1.2 Focus Modes / Specialized Search System ⭐⭐⭐⭐⭐

**Description**: Multiple specialized search/processing modes with mode-specific prompts, result filtering, and optimizations.

**Implementation Approach**:
- Mode Registry: `searchHandlers[focusMode]` pattern
- Available Modes in Perplexica:
  - `webSearch` - General web search
  - `academicSearch` - Research papers and academic content
  - `youtubeSearch` - Video content
  - `wolframAlpha` - Computational queries
  - `redditSearch` - Discussion-based content
  - `writingAssistant` - Content generation
- Each mode has custom:
  - Query rewriting prompts
  - Search engine configurations
  - Result filtering logic
  - Response generation templates

**Proposed Modes for Dionysus-2.0**:
```typescript
type DionysusFocusMode =
  | 'consciousness_analysis'    // Deep attractor basin analysis
  | 'technical_research'        // Code and technical documentation
  | 'conceptual_exploration'    // Thoughtseed generation and concept mapping
  | 'academic_research'         // Research papers with citation tracking
  | 'multi_document_synthesis'  // Cross-document pattern recognition
  | 'active_inference'          // Prediction error minimization mode
```

**Benefits for Dionysus-2.0**:
- ✅ **Mode-specific consciousness processing** - Different attractor basin strategies per mode
- ✅ **CLAUSE Phase 2 integration** - PathNavigator can select optimal mode per query
- ✅ **Specialized thoughtseed generation** - Mode-specific seed patterns
- ✅ **Optimized Neo4j queries** - Different graph patterns per mode
- ✅ **Better user control** - Explicit processing strategy selection

**Integration Complexity**: Medium-High
- Requires mode-specific prompt engineering
- Custom Neo4j query patterns per mode
- Integration with existing LangGraph workflow

**Priority**: **MUST-HAVE (Phase 1)**

---

### 1.3 Embedding-Based Result Reranking ⭐⭐⭐⭐⭐

**Description**: Cosine similarity-based reranking of search results using embeddings with configurable optimization modes.

**Implementation Approach**:
- Embedding Model: Hugging Face Transformers (`all-MiniLM-L6-v2` or `nomic-embed-text`)
- Optimization Modes: `speed`, `balanced`, `quality`
- Reranking Algorithm:
  ```typescript
  // Convert query and documents to embeddings
  const queryEmbedding = await embeddingModel.embedQuery(query);
  const docEmbeddings = await embeddingModel.embedDocuments(docs.map(d => d.content));

  // Calculate cosine similarity
  const similarities = docEmbeddings.map(docEmb =>
    cosineSimilarity(queryEmbedding, docEmb)
  );

  // Filter by threshold and sort
  const reranked = docs
    .map((doc, i) => ({ doc, score: similarities[i] }))
    .filter(item => item.score >= relevanceThreshold)
    .sort((a, b) => b.score - a.score)
    .slice(0, maxResults);
  ```

**Optimization Mode Configurations**:
- `speed`: Lower threshold (0.5), fewer results (5), faster model
- `balanced`: Medium threshold (0.6), moderate results (10), balanced model
- `quality`: Higher threshold (0.7), more results (15), best model

**Benefits for Dionysus-2.0**:
- ✅ **Enhanced Neo4j vector search** - Complement existing 512-dim embeddings
- ✅ **Multi-source result fusion** - Rank results from Neo4j + Redis + AutoSchemaKG
- ✅ **Consciousness-aware ranking** - Factor in attractor basin strength
- ✅ **ContextCurator optimization** - Rerank context selections for LC-MAPPO
- ✅ **Performance flexibility** - Users choose speed vs quality tradeoff

**Integration Complexity**: Low-Medium
- Integrate with existing Neo4j vector search
- Add reranking layer to `neo4j_searcher.py`
- FastAPI endpoint for configurable optimization modes

**Priority**: **MUST-HAVE (Phase 2)**

---

### 1.4 Citation Management System ⭐⭐⭐⭐

**Description**: Inline citation tracking with numbered references, source metadata, and markdown rendering.

**Implementation Approach**:
- Citation Format: `[1]`, `[2]`, etc. embedded in response text
- Source Structure:
  ```typescript
  interface Source {
    title: string;
    url: string;
    content: string;  // Excerpt
    favicon?: string;
    metadata?: Record<string, any>;
  }
  ```
- Markdown Integration:
  ```markdown
  The consciousness framework [1] utilizes attractor basins [2] to model
  cognitive states, with thoughtseeds [3] serving as nucleation points.
  ```
- UI Components:
  - `<Citation />` - Inline citation link component
  - `<MessageSources />` - Expandable source list with favicons
  - Modal with full source details

**Benefits for Dionysus-2.0**:
- ✅ **Source attribution for AI outputs** - Track which documents informed each insight
- ✅ **Neo4j provenance tracking** - Link citations to knowledge graph nodes
- ✅ **Academic research mode** - Proper citation formatting for research
- ✅ **Explainable AI** - Users see evidence for consciousness analysis
- ✅ **ASI-GO-2 Analyst integration** - Quality scoring includes citation coverage

**Integration Complexity**: Low-Medium
- Backend: Modify `response_synthesizer.py` to track sources
- Frontend: React components for citation display
- Citation extraction from LLM outputs

**Priority**: **NICE-TO-HAVE (Phase 2)**

---

### 1.5 Query Rewriting Pipeline ⭐⭐⭐⭐

**Description**: LLM-powered query transformation from conversational input to optimized search queries.

**Implementation Approach**:
- Uses LangChain `RunnableSequence`
- Prompt Template:
  ```
  Given conversation history and a follow-up question, rephrase it into
  a standalone question optimized for searching.

  Chat History:
  {chat_history}

  Follow-up Question: {question}

  Output the rephrased question in <question> tags.
  ```
- Handles multiple scenarios:
  - Simple questions → Direct reformulation
  - Context-dependent questions → Incorporate chat history
  - URL-based questions → Extract and include links in `<links>` tag
  - Summarization requests → Reformulate as informative queries

**Benefits for Dionysus-2.0**:
- ✅ **Better Neo4j search queries** - Optimized for graph traversal
- ✅ **PathNavigator enhancement** - Query planning step before search
- ✅ **Conversational interface** - Support follow-up questions
- ✅ **Multi-turn consciousness analysis** - Build on previous processing
- ✅ **R-Zero curiosity integration** - Reformulate curiosity triggers as queries

**Integration Complexity**: Low
- Integrate into `query_engine.py`
- Add conversation history tracking
- LangChain integration already in place

**Priority**: **NICE-TO-HAVE (Phase 2)**

---

### 1.6 Suggestion Generation Agent ⭐⭐⭐⭐

**Description**: AI-powered generation of follow-up query suggestions based on conversation context.

**Implementation Approach**:
- LangChain agent with custom prompt
- Prompt Template:
  ```
  Generate 4-5 medium-length follow-up questions based on this conversation
  that help the user explore the topic further.

  Conversation:
  {chat_history}

  Output each suggestion in a <suggestions> tag, one per line.
  ```
- Output Parsing: `ListLineOutputParser` extracts suggestions from XML tags
- Temperature: 0 (consistent, focused suggestions)

**Benefits for Dionysus-2.0**:
- ✅ **Guided consciousness exploration** - Suggest related attractor basins
- ✅ **Thoughtseed expansion** - Propose related concepts to explore
- ✅ **R-Zero curiosity suggestions** - Surface high-value curiosity triggers
- ✅ **Learning path guidance** - ASI-GO-2 Researcher suggests next steps
- ✅ **User engagement** - Reduce friction in exploration

**Integration Complexity**: Low
- Standalone FastAPI endpoint
- React component for suggestion buttons
- Integration with chat history

**Priority**: **NICE-TO-HAVE (Phase 2)**

---

### 1.7 Multi-Provider LLM Abstraction ⭐⭐⭐

**Description**: Unified interface supporting multiple LLM providers (OpenAI, Anthropic, Ollama, Groq, Gemini, etc.).

**Implementation Approach**:
- Provider Registry: `/src/lib/providers/`
- Configuration-based provider selection:
  ```typescript
  interface ChatModelConfig {
    provider: 'openai' | 'anthropic' | 'ollama' | 'groq' | 'gemini';
    model: string;
    temperature?: number;
    maxTokens?: number;
  }
  ```
- LangChain integration for provider abstraction
- Runtime provider switching per request

**Benefits for Dionysus-2.0**:
- ✅ **Cost optimization** - Use cheaper models for reranking, premium for synthesis
- ✅ **Model experimentation** - Easy A/B testing of consciousness processing
- ✅ **Local LLM support** - Ollama for privacy-sensitive processing
- ✅ **Fallback strategies** - Switch providers on failure
- ✅ **Multi-agent diversity** - Different models per agent (PathNavigator, ContextCurator)

**Integration Complexity**: Medium
- Already using OpenAI, can extend to other providers
- Configuration management updates
- Testing across providers

**Priority**: **NICE-TO-HAVE (Phase 3)**

---

### 1.8 "Think Box" Reasoning Visualization ⭐⭐⭐⭐

**Description**: Collapsible UI component displaying the AI's step-by-step reasoning process during search and analysis.

**Implementation Approach**:
- Special markdown tag: `<think>reasoning content</think>`
- React Component: `<ThinkBox />` with expand/collapse
- Features:
  - Brain circuit icon
  - Auto-expands during thinking
  - Auto-collapses when complete
  - Preserves whitespace and formatting
  - Dark/light mode styling

**Example Output**:
```markdown
<think>
Step 1: Analyzing query for consciousness-related concepts...
Step 2: Searching knowledge graph for attractor basin patterns...
Step 3: Found 3 related basins: "attention", "metacognition", "memory"
Step 4: Generating thoughtseed for "attention-metacognition" relationship...
Step 5: Synthesizing response with citations...
</think>

Based on the analysis of consciousness patterns [1], the relationship
between attention and metacognition [2] shows...
```

**Benefits for Dionysus-2.0**:
- ✅ **LangGraph transparency** - Show each node's processing
- ✅ **Consciousness insights** - Explain attractor basin formation
- ✅ **Debugging aid** - Understand why certain patterns emerged
- ✅ **User trust** - Transparent AI reasoning
- ✅ **Educational value** - Learn how consciousness processing works

**Integration Complexity**: Low
- Backend: Emit `<think>` tags during processing
- Frontend: React component for rendering
- Integration with streaming system

**Priority**: **NICE-TO-HAVE (Phase 2)**

---

### 1.9 Image/Video Search Specialization ⭐⭐

**Description**: Specialized agents for searching and presenting image and video content.

**Implementation Approach**:
- Image Search Agent: Searches Bing Images, Google Images
- Video Search Agent: Searches YouTube, other video platforms
- Query rewriting specific to visual content
- Structured result format:
  ```typescript
  interface ImageResult {
    img_src: string;
    url: string;
    title: string;
    thumbnail?: string;
  }
  ```

**Benefits for Dionysus-2.0**:
- ⚠️ **Limited relevance** - Current focus is text-based document processing
- ✅ **Potential for visualization** - Future attractor basin visualizations
- ✅ **Diagram extraction** - Extract figures from academic papers
- ✅ **Multi-modal documents** - Handle PDFs with images

**Integration Complexity**: Medium
- Requires image search API integration
- Frontend components for image grids
- Storage for image metadata

**Priority**: **SKIP (Low priority for current roadmap)**

---

### 1.10 SearxNG Meta-Search Integration ⭐⭐⭐

**Description**: Privacy-preserving meta-search engine aggregating results from multiple sources.

**Implementation Approach**:
- SearxNG: Self-hosted search engine
- Aggregates: Google, Bing, DuckDuckGo, academic databases
- Configuration:
  ```typescript
  interface SearxngSearchOptions {
    categories?: string[];
    engines?: string[];
    language?: string;
    pageno?: number;
  }
  ```
- Returns: Results + Suggestions

**Benefits for Dionysus-2.0**:
- ⚠️ **Moderate relevance** - Current focus is internal document knowledge
- ✅ **External research** - ASI-GO-2 Researcher could search web sources
- ✅ **Academic search** - Integrate arXiv, PubMed, Google Scholar
- ✅ **Privacy** - Self-hosted search for sensitive topics

**Integration Complexity**: High
- Requires SearxNG Docker deployment
- API integration layer
- Result normalization across sources

**Priority**: **SKIP (Phase 3 - External research only)**

---

### 1.11 Conversation History Management ⭐⭐⭐⭐

**Description**: Persistent chat history with context windows for multi-turn interactions.

**Implementation Approach**:
- Database: Drizzle ORM with SQLite/PostgreSQL
- Structure:
  ```typescript
  interface Message {
    id: string;
    chatId: string;
    role: 'human' | 'assistant' | 'source';
    content: string;
    sources?: Source[];
    timestamp: Date;
  }
  ```
- API Endpoints: `/api/chats`, `/api/chat/:id`
- Context Window: Last N messages passed to LLM

**Benefits for Dionysus-2.0**:
- ✅ **Multi-turn consciousness analysis** - Build on previous insights
- ✅ **Session persistence** - Resume document exploration
- ✅ **User history tracking** - Learn from user interests
- ✅ **Context for PathNavigator** - Use history to guide decisions
- ✅ **ASI-GO-2 learning** - Evolve processing based on interactions

**Integration Complexity**: Medium
- Already have PostgreSQL, can extend schema
- API endpoints for chat management
- Frontend UI for chat history

**Priority**: **NICE-TO-HAVE (Phase 3)**

---

### 1.12 Configurable System Instructions ⭐⭐⭐⭐

**Description**: User-customizable system prompts to guide AI behavior and response style.

**Implementation Approach**:
- Per-request or per-chat system instructions
- Override default prompts
- Examples:
  - "Focus on technical details and code examples"
  - "Provide academic-style responses with citations"
  - "Emphasize consciousness framework terminology"
- Stored in chat settings or user preferences

**Benefits for Dionysus-2.0**:
- ✅ **User-guided consciousness processing** - Customize analysis style
- ✅ **Domain-specific modes** - Technical vs conceptual emphasis
- ✅ **Researcher customization** - Tailor ASI-GO-2 Researcher behavior
- ✅ **Response quality** - Fine-tune outputs to user preferences
- ✅ **Experimentation** - Users explore different processing strategies

**Integration Complexity**: Low
- Modify prompt templates to include system instructions
- UI for instruction input
- Storage in chat metadata

**Priority**: **NICE-TO-HAVE (Phase 3)**

---

## 2. Prioritized Integration Roadmap

### Phase 1: Quick Wins (1 Week) - CRITICAL PATH

**Goal**: Immediate UX improvements for consciousness processing visibility

#### T1.1: Streaming Response System (3 days)
- **Backend**:
  - Implement FastAPI streaming endpoint
  - Emit events from `document_processing_graph.py` nodes
  - Add `StreamEvent` types
- **Frontend**:
  - Create `useStreamingSearch` hook
  - Update `DocumentUpload.tsx` to show real-time progress
  - Display LangGraph node transitions
- **Testing**:
  - Test all 6 LangGraph nodes emit events
  - Verify UI updates in real-time
  - Error handling for stream interruptions
- **Deliverable**: Users see live consciousness processing updates

#### T1.2: Focus Modes System (2 days)
- **Backend**:
  - Define `DionysusFocusMode` enum
  - Create mode-specific prompt templates
  - Integrate with `query_engine.py`
- **Frontend**:
  - Add focus mode selector to upload UI
  - Display active mode during processing
- **Testing**:
  - Test each mode with sample documents
  - Verify mode-specific processing behavior
- **Deliverable**: Users select processing strategy (consciousness_analysis, technical_research, etc.)

**Phase 1 Success Metrics**:
- ✅ Real-time processing visible in UI
- ✅ 5+ focus modes implemented
- ✅ User satisfaction with transparency

---

### Phase 2: Core Search Enhancements (2 Weeks) - HIGH VALUE

**Goal**: Improve search quality and user guidance

#### T2.1: Embedding-Based Reranking (3 days)
- **Backend**:
  - Integrate Hugging Face Transformers
  - Implement cosine similarity reranking
  - Add optimization modes (speed/balanced/quality)
  - Integrate with `neo4j_searcher.py`
- **Testing**:
  - Benchmark reranking quality vs current search
  - Measure latency impact
  - Validate optimization modes
- **Deliverable**: 20%+ improvement in search relevance

#### T2.2: Citation Management (3 days)
- **Backend**:
  - Modify `response_synthesizer.py` to track sources
  - Emit citation events in stream
  - Store citations in Neo4j
- **Frontend**:
  - `<Citation />` component
  - `<MessageSources />` expandable list
  - Inline citation rendering
- **Testing**:
  - Verify citations link to correct sources
  - Test modal source display
- **Deliverable**: All AI responses include source attribution

#### T2.3: Query Rewriting Pipeline (2 days)
- **Backend**:
  - Implement query rewriting agent
  - Integrate with `query_engine.py`
  - Add conversation history context
- **Testing**:
  - Test conversational follow-ups
  - Verify query optimization
- **Deliverable**: Improved search for conversational queries

#### T2.4: Suggestion Generation (2 days)
- **Backend**:
  - Implement suggestion generation agent
  - Create `/api/suggestions` endpoint
- **Frontend**:
  - Suggestion buttons in chat interface
  - Click-to-search suggestions
- **Testing**:
  - Validate suggestion relevance
  - Test across different conversation contexts
- **Deliverable**: Users get guided exploration suggestions

#### T2.5: Think Box Visualization (2 days)
- **Backend**:
  - Emit `<think>` tags during processing
  - Include reasoning steps from each LangGraph node
- **Frontend**:
  - `<ThinkBox />` component with expand/collapse
  - Parse and render thinking content
- **Testing**:
  - Verify all processing steps captured
  - Test auto-expand/collapse behavior
- **Deliverable**: Transparent reasoning visualization

**Phase 2 Success Metrics**:
- ✅ 20%+ improvement in search relevance (reranking)
- ✅ 100% citation coverage for AI responses
- ✅ 80%+ user satisfaction with suggestions
- ✅ Reasoning transparency improves user trust

---

### Phase 3: Advanced Features (2-3 Weeks) - STRATEGIC

**Goal**: Multi-provider support, persistence, customization

#### T3.1: Multi-Provider LLM Support (4 days)
- **Backend**:
  - Extend provider registry (`/providers/`)
  - Add Anthropic, Ollama, Groq providers
  - Implement provider switching logic
  - Configuration for model selection
- **Testing**:
  - Test each provider integration
  - Validate fallback strategies
  - Benchmark cost vs quality
- **Deliverable**: Support for 5+ LLM providers

#### T3.2: Conversation History (3 days)
- **Backend**:
  - Extend PostgreSQL schema for chat history
  - Create `/api/chats` endpoints
  - Implement context window management
- **Frontend**:
  - Chat history sidebar
  - Resume chat sessions
  - Search chat history
- **Testing**:
  - Test multi-turn conversations
  - Verify context persistence
- **Deliverable**: Persistent multi-turn interactions

#### T3.3: Configurable System Instructions (2 days)
- **Backend**:
  - Add system instruction parameter to prompts
  - Store instructions in chat metadata
- **Frontend**:
  - System instruction input field
  - Preset instruction templates
- **Testing**:
  - Test custom instructions across modes
  - Verify instruction persistence
- **Deliverable**: User-customizable processing behavior

#### T3.4: SearxNG Integration (Optional - 4 days)
- **Infrastructure**:
  - Deploy SearxNG via Docker
  - Configure search engines
- **Backend**:
  - Create SearxNG client
  - Integrate with ASI-GO-2 Researcher
  - Add external research mode
- **Testing**:
  - Test multi-source aggregation
  - Validate result quality
- **Deliverable**: External web research capability

**Phase 3 Success Metrics**:
- ✅ Cost reduction through provider optimization
- ✅ Improved user engagement with chat history
- ✅ Higher customization satisfaction
- ✅ (Optional) External research enriches knowledge base

---

## 3. Technical Implementation Notes

### 3.1 Integration with Existing Architecture

#### Dionysus-2.0 Current State
- **Backend**: FastAPI + Neo4j + Redis + PostgreSQL
- **Frontend**: React + TypeScript + Three.js
- **Processing**: LangGraph 6-node workflow
- **Knowledge**: AutoSchemaKG + Consciousness processing
- **Multi-Agent**: CLAUSE Phase 2 (PathNavigator, ContextCurator, LC-MAPPO)

#### Integration Points

**Streaming System**:
- Modify `document_processing_graph.py` to emit events at each node
- Create new `/api/stream-process` endpoint in FastAPI
- Update React components to use `useStreamingSearch` hook

**Focus Modes**:
- Extend `query_engine.py` with mode registry
- Create mode-specific prompt files in `/prompts/`
- PathNavigator selects optimal mode based on query analysis

**Reranking**:
- Add reranking layer to `neo4j_searcher.py` after vector search
- ContextCurator uses reranking for context selection
- Integrate with existing 512-dim embeddings

**Citations**:
- Modify `response_synthesizer.py` to track source documents
- Store citation links in Neo4j relationships
- Frontend components render citations inline

---

### 3.2 Consciousness Framework Integration

#### Perplexica Patterns → Consciousness Enhancements

| Perplexica Feature | Consciousness Integration |
|--------------------|--------------------------|
| Query Rewriting | PathNavigator query planning |
| Reranking | Attractor basin strength weighting |
| Focus Modes | Consciousness processing strategies |
| Streaming | Real-time basin formation visualization |
| Citations | ThoughtSeed provenance tracking |
| Suggestions | R-Zero curiosity trigger suggestions |
| Think Box | Meta-cognitive awareness display |

#### Example: Consciousness-Aware Reranking
```python
# In neo4j_searcher.py
def rerank_with_consciousness(results, query, mode='balanced'):
    """Rerank results using embeddings + consciousness metrics"""

    # Standard embedding similarity
    embedding_scores = calculate_embedding_similarity(query, results)

    # Consciousness factors
    basin_strengths = get_attractor_basin_strengths(results)
    thoughtseed_resonance = calculate_thoughtseed_resonance(query, results)

    # Weighted combination
    final_scores = (
        0.5 * embedding_scores +
        0.3 * basin_strengths +
        0.2 * thoughtseed_resonance
    )

    return sort_by_scores(results, final_scores)
```

---

### 3.3 CLAUSE Phase 2 Multi-Agent Integration

#### PathNavigator Enhancement
- **Query Rewriting**: PathNavigator uses query rewriting to plan optimal search paths
- **Focus Mode Selection**: PathNavigator chooses focus mode based on query type
- **Stream Coordination**: PathNavigator emits decision events in stream

#### ContextCurator Enhancement
- **Reranking**: ContextCurator uses reranking to select most relevant context
- **Citation Tracking**: Track which sources informed each context selection
- **Suggestion Generation**: Generate suggestions for context expansion

#### LC-MAPPO Coordinator Enhancement
- **Multi-Provider**: Coordinator assigns different LLM providers to agents
- **Performance Tracking**: Use conversation history to learn agent performance
- **Streaming Feedback**: Real-time coordination decisions visible to users

---

### 3.4 Technical Risks & Mitigation

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Streaming latency increases | Medium | Low | Implement buffering, optimize event frequency |
| Embedding reranking slows queries | High | Medium | Use faster models for speed mode, cache embeddings |
| LangChain version conflicts | Medium | Low | Pin LangChain versions, test thoroughly |
| Citation extraction errors | Low | Medium | Fallback to no citations, improve prompt engineering |
| Multi-provider complexity | Medium | Medium | Start with 2 providers, expand gradually |
| Storage overhead (history) | Low | Low | Implement retention policies, archive old chats |

---

### 3.5 Performance Considerations

#### Expected Performance Impact

| Feature | Latency Impact | Storage Impact | Compute Impact |
|---------|---------------|----------------|----------------|
| Streaming | -20% (perceived) | None | +5% (overhead) |
| Reranking | +50-200ms | None | +15% (embeddings) |
| Citations | +10ms | +10% (metadata) | None |
| Query Rewriting | +100-300ms | None | +10% (LLM call) |
| Suggestions | +200-500ms | None | +15% (LLM call) |
| History | None | +50% (messages) | None |

#### Optimization Strategies
- **Caching**: Cache embeddings, query rewrites, suggestions
- **Async Processing**: Run reranking and suggestions in parallel
- **Lazy Loading**: Load chat history on-demand
- **Model Selection**: Use smaller models for speed-critical features
- **Batching**: Batch embedding generation

---

## 4. Code Examples & Patterns

### 4.1 Streaming Implementation

**Backend (FastAPI)**:
```python
# In backend/src/api/routes/search.py
from fastapi.responses import StreamingResponse
import json

@router.post("/stream-search")
async def stream_search(request: SearchRequest):
    async def event_generator():
        # Initialize
        yield json.dumps({
            "type": "init",
            "messageId": request.message_id
        }) + "\n"

        # Process through LangGraph
        async for node_result in process_document_graph(request):
            # Emit sources
            if node_result.sources:
                yield json.dumps({
                    "type": "sources",
                    "data": [s.dict() for s in node_result.sources]
                }) + "\n"

            # Emit thinking
            if node_result.reasoning:
                yield json.dumps({
                    "type": "message",
                    "messageId": request.message_id,
                    "data": f"<think>{node_result.reasoning}</think>\n"
                }) + "\n"

            # Emit partial response
            if node_result.partial_response:
                yield json.dumps({
                    "type": "message",
                    "messageId": request.message_id,
                    "data": node_result.partial_response
                }) + "\n"

        # Finalize
        yield json.dumps({
            "type": "messageEnd",
            "messageId": request.message_id
        }) + "\n"

    return StreamingResponse(
        event_generator(),
        media_type="application/x-ndjson"
    )
```

**Frontend (React)**:
```typescript
// In frontend/src/hooks/useStreamingSearch.ts
import { useState, useCallback } from 'react';

interface Message {
  role: 'user' | 'assistant' | 'source';
  content: string;
  sources?: Source[];
}

export function useStreamingSearch() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const sendQuery = useCallback(async (query: string, mode: FocusMode) => {
    setIsLoading(true);

    // Add user message
    setMessages(prev => [...prev, { role: 'user', content: query }]);

    // Add placeholder for assistant message
    const assistantId = Date.now().toString();
    setMessages(prev => [...prev, {
      role: 'assistant',
      content: '',
      messageId: assistantId
    }]);

    // Fetch streaming response
    const response = await fetch('/api/stream-search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, mode, message_id: assistantId })
    });

    const reader = response.body?.getReader();
    const decoder = new TextDecoder('utf-8');

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      // Parse events
      const lines = decoder.decode(value).split('\n');
      for (const line of lines) {
        if (!line.trim()) continue;

        const event = JSON.parse(line);

        switch (event.type) {
          case 'sources':
            setMessages(prev => prev.map(msg =>
              msg.messageId === assistantId
                ? { ...msg, sources: event.data }
                : msg
            ));
            break;

          case 'message':
            setMessages(prev => prev.map(msg =>
              msg.messageId === event.messageId
                ? { ...msg, content: msg.content + event.data }
                : msg
            ));
            break;

          case 'messageEnd':
            setIsLoading(false);
            break;
        }
      }
    }
  }, []);

  return { messages, isLoading, sendQuery };
}
```

---

### 4.2 Focus Mode Implementation

**Backend (Mode Registry)**:
```python
# In backend/src/services/focus_modes.py
from enum import Enum
from typing import Dict, Callable

class FocusMode(str, Enum):
    CONSCIOUSNESS_ANALYSIS = "consciousness_analysis"
    TECHNICAL_RESEARCH = "technical_research"
    CONCEPTUAL_EXPLORATION = "conceptual_exploration"
    ACADEMIC_RESEARCH = "academic_research"
    MULTI_DOCUMENT_SYNTHESIS = "multi_document_synthesis"

class FocusModeConfig:
    def __init__(
        self,
        prompt_template: str,
        neo4j_query_strategy: str,
        max_basin_depth: int,
        thoughtseed_threshold: float
    ):
        self.prompt_template = prompt_template
        self.neo4j_query_strategy = neo4j_query_strategy
        self.max_basin_depth = max_basin_depth
        self.thoughtseed_threshold = thoughtseed_threshold

FOCUS_MODE_CONFIGS: Dict[FocusMode, FocusModeConfig] = {
    FocusMode.CONSCIOUSNESS_ANALYSIS: FocusModeConfig(
        prompt_template="""
        Analyze this document through the lens of consciousness frameworks.
        Focus on:
        - Attractor basin patterns
        - Meta-cognitive structures
        - Thoughtseed generation opportunities
        - Consciousness emergence indicators

        Document: {content}
        """,
        neo4j_query_strategy="deep_basin_traversal",
        max_basin_depth=5,
        thoughtseed_threshold=0.7
    ),

    FocusMode.TECHNICAL_RESEARCH: FocusModeConfig(
        prompt_template="""
        Extract technical details, code patterns, and implementation specifics.
        Focus on:
        - APIs and interfaces
        - Algorithms and data structures
        - Performance characteristics
        - Integration patterns

        Document: {content}
        """,
        neo4j_query_strategy="technical_concept_graph",
        max_basin_depth=3,
        thoughtseed_threshold=0.5
    ),

    # ... other modes
}

def get_mode_handler(mode: FocusMode):
    """Get processing handler for focus mode"""
    config = FOCUS_MODE_CONFIGS[mode]

    async def handler(query: str, documents: List[Document]):
        # Apply mode-specific processing
        prompt = config.prompt_template.format(content=documents[0].content)

        # Use mode-specific Neo4j strategy
        graph_results = await neo4j_searcher.search(
            query,
            strategy=config.neo4j_query_strategy
        )

        # Filter by mode-specific thresholds
        thoughtseeds = generate_thoughtseeds(
            documents,
            threshold=config.thoughtseed_threshold
        )

        return {
            "prompt": prompt,
            "graph_results": graph_results,
            "thoughtseeds": thoughtseeds
        }

    return handler
```

**Frontend (Mode Selector)**:
```typescript
// In frontend/src/components/FocusModeSelector.tsx
import React from 'react';

const FOCUS_MODES = [
  {
    id: 'consciousness_analysis',
    label: 'Consciousness Analysis',
    description: 'Deep attractor basin and meta-cognitive analysis',
    icon: '🧠'
  },
  {
    id: 'technical_research',
    label: 'Technical Research',
    description: 'Code, APIs, and implementation patterns',
    icon: '⚙️'
  },
  {
    id: 'conceptual_exploration',
    label: 'Conceptual Exploration',
    description: 'ThoughtSeed generation and concept mapping',
    icon: '🌱'
  },
  {
    id: 'academic_research',
    label: 'Academic Research',
    description: 'Research papers with citation tracking',
    icon: '📚'
  },
  {
    id: 'multi_document_synthesis',
    label: 'Multi-Document Synthesis',
    description: 'Cross-document pattern recognition',
    icon: '🔗'
  },
];

export function FocusModeSelector({ value, onChange }) {
  return (
    <div className="focus-mode-selector">
      <label>Processing Mode:</label>
      <div className="mode-grid">
        {FOCUS_MODES.map(mode => (
          <button
            key={mode.id}
            className={`mode-button ${value === mode.id ? 'active' : ''}`}
            onClick={() => onChange(mode.id)}
          >
            <span className="mode-icon">{mode.icon}</span>
            <span className="mode-label">{mode.label}</span>
            <span className="mode-description">{mode.description}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
```

---

### 4.3 Embedding Reranking Implementation

```python
# In backend/src/services/reranker.py
from typing import List, Dict, Literal
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

class ConsciousnessAwareReranker:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Mean pooling
        embeddings = outputs.last_hidden_state.mean(dim=1)
        # Normalize
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings.numpy()[0]

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    async def rerank(
        self,
        query: str,
        results: List[Dict],
        mode: Literal['speed', 'balanced', 'quality'] = 'balanced',
        consciousness_weight: float = 0.3
    ) -> List[Dict]:
        """Rerank results using embeddings + consciousness metrics"""

        # Configuration based on mode
        configs = {
            'speed': {'threshold': 0.5, 'max_results': 5},
            'balanced': {'threshold': 0.6, 'max_results': 10},
            'quality': {'threshold': 0.7, 'max_results': 15}
        }
        config = configs[mode]

        # Generate embeddings
        query_emb = self.embed_text(query)
        result_embs = [self.embed_text(r['content']) for r in results]

        # Calculate embedding similarities
        emb_scores = [
            self.cosine_similarity(query_emb, result_emb)
            for result_emb in result_embs
        ]

        # Get consciousness metrics from Neo4j
        consciousness_scores = await self._get_consciousness_scores(results)

        # Combine scores
        final_scores = []
        for i, result in enumerate(results):
            combined_score = (
                (1 - consciousness_weight) * emb_scores[i] +
                consciousness_weight * consciousness_scores[i]
            )
            final_scores.append({
                'result': result,
                'score': combined_score,
                'emb_score': emb_scores[i],
                'consciousness_score': consciousness_scores[i]
            })

        # Filter and sort
        filtered = [s for s in final_scores if s['score'] >= config['threshold']]
        sorted_results = sorted(filtered, key=lambda x: x['score'], reverse=True)

        return sorted_results[:config['max_results']]

    async def _get_consciousness_scores(self, results: List[Dict]) -> List[float]:
        """Get consciousness-based relevance scores from Neo4j"""
        scores = []
        for result in results:
            # Query Neo4j for attractor basin strength
            basin_strength = await neo4j_searcher.get_basin_strength(result['id'])
            # Query for thoughtseed resonance
            thoughtseed_resonance = await neo4j_searcher.get_thoughtseed_resonance(result['id'])
            # Combine
            consciousness_score = 0.6 * basin_strength + 0.4 * thoughtseed_resonance
            scores.append(consciousness_score)
        return scores
```

---

### 4.4 Citation Management Implementation

**Backend (Citation Tracking)**:
```python
# In backend/src/services/response_synthesizer.py
from typing import List, Dict
import re

class CitationTracker:
    def __init__(self):
        self.sources: List[Dict] = []
        self.citation_map: Dict[int, str] = {}

    def add_source(self, source: Dict) -> int:
        """Add source and return citation number"""
        citation_num = len(self.sources) + 1
        self.sources.append(source)
        self.citation_map[citation_num] = source['id']
        return citation_num

    def format_response_with_citations(self, response: str, source_docs: List[Dict]) -> str:
        """Add inline citations to response"""
        # Track which sources are referenced
        cited_sources = []

        # For each sentence, find relevant sources
        sentences = response.split('. ')
        formatted_sentences = []

        for sentence in sentences:
            # Find matching sources using keyword overlap
            matching_sources = self._find_matching_sources(sentence, source_docs)

            if matching_sources:
                # Add citations
                citations = []
                for source in matching_sources:
                    if source not in cited_sources:
                        cited_sources.append(source)
                    citation_num = cited_sources.index(source) + 1
                    citations.append(citation_num)

                # Format: "Sentence content [1][2]."
                citation_str = ''.join([f'[{c}]' for c in citations])
                formatted_sentences.append(f"{sentence}{citation_str}")
            else:
                formatted_sentences.append(sentence)

        formatted_response = '. '.join(formatted_sentences)

        # Store sources for later retrieval
        self.sources = cited_sources

        return formatted_response

    def _find_matching_sources(self, text: str, sources: List[Dict]) -> List[Dict]:
        """Find sources that match text content"""
        matches = []
        text_lower = text.lower()

        for source in sources:
            # Check keyword overlap
            source_keywords = set(source['content'].lower().split())
            text_keywords = set(text_lower.split())
            overlap = len(source_keywords & text_keywords)

            if overlap >= 3:  # Threshold for match
                matches.append(source)

        return matches[:2]  # Max 2 citations per sentence

    def get_sources(self) -> List[Dict]:
        """Get all cited sources"""
        return self.sources
```

**Frontend (Citation Component)**:
```typescript
// In frontend/src/components/Citation.tsx
import React from 'react';

interface CitationProps {
  number: number;
  source: {
    title: string;
    url: string;
    favicon?: string;
  };
}

export function Citation({ number, source }: CitationProps) {
  return (
    <a
      href={source.url}
      target="_blank"
      rel="noopener noreferrer"
      className="citation"
      title={source.title}
    >
      [{number}]
    </a>
  );
}

// In frontend/src/components/MessageSources.tsx
import React, { useState } from 'react';
import { Dialog } from '@headlessui/react';

export function MessageSources({ sources }) {
  const [showAll, setShowAll] = useState(false);
  const displayedSources = showAll ? sources : sources.slice(0, 3);

  return (
    <div className="message-sources">
      <div className="sources-grid">
        {displayedSources.map((source, i) => (
          <a
            key={i}
            href={source.url}
            target="_blank"
            rel="noopener noreferrer"
            className="source-card"
          >
            <div className="source-number">{i + 1}</div>
            {source.favicon && (
              <img src={source.favicon} alt="" className="source-favicon" />
            )}
            <div className="source-info">
              <div className="source-title">{source.title}</div>
              <div className="source-domain">{new URL(source.url).hostname}</div>
            </div>
          </a>
        ))}
      </div>

      {sources.length > 3 && !showAll && (
        <button onClick={() => setShowAll(true)} className="show-more-btn">
          View {sources.length - 3} more sources
        </button>
      )}

      {showAll && (
        <Dialog open={showAll} onClose={() => setShowAll(false)}>
          <div className="sources-modal">
            <Dialog.Title>All Sources</Dialog.Title>
            <div className="sources-grid">
              {sources.map((source, i) => (
                // ... same source card
              ))}
            </div>
          </div>
        </Dialog>
      )}
    </div>
  );
}
```

---

## 5. Recommendations

### 5.1 Must-Have Features (Implement Immediately)

1. **Streaming Response System** ⭐⭐⭐⭐⭐
   - **Why**: Dramatically improves UX for long-running consciousness processing
   - **Impact**: Users stay engaged, perceive faster performance
   - **Effort**: 3 days
   - **ROI**: Extremely High

2. **Focus Modes System** ⭐⭐⭐⭐⭐
   - **Why**: Enables specialized consciousness processing strategies
   - **Impact**: Better results, user control, CLAUSE Phase 2 integration
   - **Effort**: 2 days
   - **ROI**: Extremely High

### 5.2 High-Value Features (Implement in Phase 2)

3. **Embedding-Based Reranking** ⭐⭐⭐⭐⭐
   - **Why**: Improves search relevance significantly
   - **Impact**: 20%+ improvement in result quality
   - **Effort**: 3 days
   - **ROI**: Very High

4. **Citation Management** ⭐⭐⭐⭐
   - **Why**: Source attribution critical for AI transparency
   - **Impact**: User trust, academic credibility
   - **Effort**: 3 days
   - **ROI**: High

5. **Think Box Visualization** ⭐⭐⭐⭐
   - **Why**: Shows consciousness processing reasoning
   - **Impact**: Transparency, debugging, user education
   - **Effort**: 2 days
   - **ROI**: High

### 5.3 Strategic Features (Implement in Phase 3)

6. **Multi-Provider LLM Support** ⭐⭐⭐
   - **Why**: Cost optimization, experimentation
   - **Impact**: 30-50% cost reduction potential
   - **Effort**: 4 days
   - **ROI**: Medium-High

7. **Conversation History** ⭐⭐⭐⭐
   - **Why**: Multi-turn consciousness exploration
   - **Impact**: Better user engagement, learning
   - **Effort**: 3 days
   - **ROI**: Medium

### 5.4 Skip or Defer

8. **Image/Video Search** - Not aligned with current text-focused roadmap
9. **SearxNG Integration** - Only if external web research becomes priority

---

### 5.5 Integration Strategy

**Week 1**: Phase 1 Features
- Day 1-3: Streaming Response System
- Day 4-5: Focus Modes System
- **Deliverable**: Live consciousness processing with mode selection

**Week 2-3**: Phase 2 Core Features
- Day 6-8: Embedding Reranking
- Day 9-11: Citation Management
- Day 12-13: Query Rewriting + Suggestions
- Day 14-15: Think Box Visualization
- **Deliverable**: Complete search enhancement suite

**Week 4-6**: Phase 3 Strategic Features
- Day 16-19: Multi-Provider LLM
- Day 20-22: Conversation History
- Day 23-24: Configurable Instructions
- Day 25-28: (Optional) SearxNG Integration
- **Deliverable**: Advanced customization and persistence

---

### 5.6 Success Metrics

**Phase 1 (Week 1)**:
- ✅ 100% of LangGraph nodes emit stream events
- ✅ Users see real-time progress updates
- ✅ 5+ focus modes implemented
- ✅ User satisfaction score > 8/10

**Phase 2 (Week 2-3)**:
- ✅ 20%+ improvement in search relevance (A/B test)
- ✅ 100% citation coverage on AI responses
- ✅ 80%+ users find suggestions helpful
- ✅ Think Box used by 60%+ users
- ✅ User engagement time increases 30%+

**Phase 3 (Week 4-6)**:
- ✅ LLM costs reduced 30%+ through provider optimization
- ✅ Multi-turn conversation sessions increase 50%+
- ✅ Custom instructions used by 40%+ users
- ✅ Overall user satisfaction > 9/10

---

## 6. Conclusion

Perplexica provides a **mature, battle-tested architecture** for AI-powered search that aligns remarkably well with Dionysus-2.0's consciousness-enhanced document processing needs. The **streaming response system** and **focus modes** are particularly valuable for visualizing and controlling LangGraph consciousness processing workflows.

**Key Takeaways**:

1. **Immediate Value**: Streaming + Focus Modes can be implemented in 1 week with massive UX impact
2. **Search Quality**: Embedding reranking addresses current search limitations
3. **Transparency**: Citations and Think Box provide explainable AI
4. **CLAUSE Phase 2 Synergy**: Focus modes integrate perfectly with PathNavigator/ContextCurator
5. **Proven Patterns**: Perplexica's architecture has been validated by thousands of users

**Recommended Approach**:
- **Start with Phase 1** (Streaming + Focus Modes) - 1 week, extremely high ROI
- **Evaluate results** before committing to Phase 2
- **Iterate based on user feedback** from early features
- **Phase 3 features** can be added incrementally based on demand

**Total Estimated Effort**: 4-6 weeks
**Expected ROI**: Very High (70%+ features are high-value)
**Risk Level**: Low (proven patterns, MIT licensed, active community)

---

## Appendix: Additional Resources

**Perplexica Repository**: https://github.com/ItzCrazyKns/Perplexica
**Architecture Docs**: https://github.com/ItzCrazyKns/Perplexica/tree/master/docs/architecture
**API Docs**: https://github.com/ItzCrazyKns/Perplexica/blob/master/docs/API/SEARCH.md

**Key Files to Study**:
- `/src/lib/chains/suggestionGeneratorAgent.ts` - Suggestion generation
- `/src/lib/search/metaSearchAgent.ts` - Core search orchestration
- `/src/lib/huggingfaceTransformer.ts` - Embedding generation
- `/src/lib/hooks/useChat.tsx` - Streaming frontend implementation
- `/src/app/api/search/route.ts` - Search API endpoint
- `/src/components/MessageBox.tsx` - Message rendering with citations
- `/src/components/ThinkBox.tsx` - Reasoning visualization

**Related Dionysus-2.0 Files**:
- `/backend/src/services/document_processing_graph.py` - LangGraph workflow (add streaming)
- `/backend/src/services/query_engine.py` - Query processing (add rewriting, modes)
- `/backend/src/services/neo4j_searcher.py` - Search implementation (add reranking)
- `/backend/src/services/response_synthesizer.py` - Response generation (add citations)
- `/frontend/src/pages/DocumentUpload.tsx` - Upload UI (add streaming, mode selector)

---

**Document Version**: 1.0
**Last Updated**: 2025-10-04
**Next Review**: After Phase 1 implementation
