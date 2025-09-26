# Nemori-Inspired Episodic Memory Integration Specification

**Version**: 1.0.0  
**Status**: ACTIVE DEVELOPMENT  
**Last Updated**: 2025-09-22  
**Specification Type**: Integration Enhancement Specification  
**Related Projects**: [Nemori AI](https://github.com/nemori-ai/nemori.git), Ritter et al. (2018)

## 🎯 Executive Summary

This specification integrates insights from the Nemori project into our ASI-Arch Context Flow episodic meta-learning architecture. Nemori demonstrates that **aligning AI memory with human episodic memory granularity** enables simple methods to achieve excellent performance. We adapt their approach for neural architecture discovery and evolution.

## 🧠 Core Nemori Insights

### Insight 1: Episodic Granularity Alignment
- **Principle**: Align memory granularity with human episodic memory patterns
- **Benefit**: Reduces distributional shift, improves retrieval precision
- **Application**: Structure architecture evolution as coherent "episodes" rather than individual evaluations

### Insight 2: Minimalist Pipeline Effectiveness
- **Nemori Approach**: 2 prompts + BM25 retrieval
- **Our Adaptation**: Episode boundary detection + architecture episode summarization + BM25 retrieval
- **Benefit**: Simplicity without sacrificing performance

### Insight 3: Training Distribution Alignment
- **Principle**: Episodic descriptions match "natural world event granularity"
- **Benefit**: Better token prediction probabilities, enhanced semantic matching
- **Application**: Architecture episodes described in natural, human-understandable terms

## 📋 Integration Requirements

### 🔥 Critical Integration Requirements (MUST HAVE)

#### CIR-001: Episodic Boundary Detection for Architecture Evolution
- **Requirement**: System MUST detect natural episode boundaries in architecture evolution sequences
- **Acceptance Criteria**: 
  - Detect boundaries based on performance shifts, architectural changes, or exploration phases
  - Use minimal prompting approach (≤2 LLM calls per boundary detection)
  - Achieve >85% accuracy in identifying meaningful episode boundaries
- **Nemori Inspiration**: "Detect episode boundaries along natural topic shifts"
- **Test Cases**: 
  - Evolution sequence with 3 distinct phases → 2 boundaries detected
  - Gradual improvement sequence → No false boundaries
  - Sudden performance jump → Boundary detected

#### CIR-002: Architecture Episode Generation
- **Requirement**: System MUST generate coherent episodic summaries of architecture evolution segments
- **Acceptance Criteria**:
  - Summarize each evolution segment into human-readable episodes
  - Include architectural patterns, performance outcomes, and narrative context
  - Episodes align with archetypal patterns when applicable
- **Nemori Inspiration**: "Summarize each segment into an episodic memory"
- **Test Cases**:
  - 10-evaluation segment → Coherent episode summary
  - Episode includes performance trend and key architectural changes
  - Archetypal pattern identified and included

#### CIR-003: BM25-Based Episode Retrieval
- **Requirement**: System MUST implement BM25 retrieval for architecture episodes
- **Acceptance Criteria**:
  - Build BM25 index from architecture episodes
  - Retrieve top-k relevant episodes for queries
  - No embedding computation required for basic retrieval
- **Nemori Inspiration**: "Build BM25 Index — No extra LLM calls, no embeddings required"
- **Test Cases**:
  - Query "transformer architectures" → Retrieves relevant transformer episodes
  - BM25 scoring works correctly for architecture terminology
  - Retrieval speed <10ms per query

#### CIR-004: Human-Scale Architecture Narratives
- **Requirement**: System MUST generate architecture descriptions at human episodic memory scale
- **Acceptance Criteria**:
  - Episodes span meaningful architectural developments (not individual evaluations)
  - Descriptions use natural language accessible to human researchers
  - Temporal context included (relative time, sequence information)
- **Nemori Inspiration**: "Aligning with the granularity of human memory event episodes"
- **Test Cases**:
  - Episode spans 5-20 architecture evaluations
  - Natural language description readable by humans
  - Temporal markers included ("after initial exploration", "during optimization phase")

### ⚡ Important Integration Requirements (SHOULD HAVE)

#### IIR-001: Hybrid Retrieval Enhancement
- **Requirement**: System SHOULD support hybrid BM25 + vector retrieval
- **Acceptance Criteria**:
  - Optional dense retrieval for semantic similarity
  - Configurable weighting between sparse and dense methods
  - Reranking strategies for different use cases
- **Nemori Inspiration**: "Hybrid retrieval strategy combining sparse (BM25) and dense (vector retrieval) methods"

#### IIR-002: Semantic Memory Integration
- **Requirement**: System SHOULD preserve important details through semantic memory
- **Acceptance Criteria**:
  - Extract and preserve key architectural components, names, and parameters
  - Link semantic details to episodic memories
  - Prevent information loss during episode summarization
- **Nemori Inspiration**: "Add 'semantic memory' capability to address the issue of episodic memory losing information"

#### IIR-003: Hierarchical Episode Aggregation
- **Requirement**: System SHOULD aggregate related episodes into higher-level patterns
- **Acceptance Criteria**:
  - Group similar episodes by architectural patterns or performance characteristics
  - Create meta-episodes representing longer-term trends
  - Support multi-level episode hierarchy
- **Nemori Inspiration**: "Aggregate episodes through similarity measures to form longer-term and more general high-level episodes"

## 🏗️ Architecture Integration Design

### Nemori-Inspired Pipeline for ASI-Arch

```
Architecture Evolution Sequence
    ↓
1. Episode Boundary Detection (LLM Prompt)
    ↓
2. Episode Generation (LLM Prompt)  
    ↓
3. BM25 Index Construction (No LLM)
    ↓
4. Episode Retrieval & Architecture Guidance (BM25 Only)
```

### Episode Structure for Architecture Evolution

```python
@dataclass
class ArchitectureEpisode:
    """Nemori-inspired episode for architecture evolution"""
    
    # Core Episode Content
    episode_id: str                     # Unique identifier
    title: str                          # Human-readable episode title
    narrative_summary: str              # Natural language description
    
    # Temporal Context
    start_evaluation: int               # Starting evaluation number
    end_evaluation: int                 # Ending evaluation number
    relative_time_markers: List[str]    # "after initial exploration", etc.
    
    # Architectural Content
    key_architectural_patterns: List[str]    # Main architectural features
    performance_trajectory: Dict[str, float] # Performance metrics over episode
    breakthrough_moments: List[str]          # Key discoveries or improvements
    
    # Semantic Details (Preserved Information)
    architectural_parameters: Dict[str, Any] # Specific parameters and values
    evaluation_details: List[Dict]           # Detailed evaluation results
    
    # Archetypal Context
    dominant_archetype: Optional[ArchetypalResonancePattern]
    narrative_coherence_score: float
    
    # Retrieval Metadata
    bm25_tokens: List[str]              # Tokenized content for BM25
    semantic_keywords: List[str]        # Key terms for retrieval
```

### BM25 Index Structure

```python
class ArchitectureEpisodeBM25Index:
    """BM25 index for architecture episodes"""
    
    def __init__(self):
        self.episodes: List[ArchitectureEpisode] = []
        self.bm25_index = None  # BM25 index object
        self.term_frequencies: Dict[str, Dict[str, int]] = {}
        
    def add_episode(self, episode: ArchitectureEpisode):
        """Add episode to index"""
        # Tokenize episode content
        tokens = self._tokenize_episode(episode)
        episode.bm25_tokens = tokens
        
        # Update BM25 index
        self._update_bm25_index(episode, tokens)
        
    def retrieve_episodes(self, query: str, top_k: int = 10) -> List[Tuple[ArchitectureEpisode, float]]:
        """Retrieve most relevant episodes using BM25"""
        query_tokens = self._tokenize_query(query)
        scores = self._compute_bm25_scores(query_tokens)
        
        # Return top-k episodes with scores
        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
```

## 🔌 Implementation Strategy

### Phase 1: Episode Boundary Detection (Week 1)
Following Nemori's minimalist approach:

```python
def detect_episode_boundaries(evolution_sequence: List[ArchitectureEvaluation]) -> List[int]:
    """Detect episode boundaries in architecture evolution"""
    
    # Single LLM prompt for boundary detection
    prompt = f"""
    Analyze this architecture evolution sequence and identify natural episode boundaries.
    Look for:
    - Significant performance shifts
    - Major architectural pattern changes  
    - Transition between exploration and exploitation phases
    
    Evolution sequence: {evolution_sequence}
    
    Return boundary positions as a list of evaluation indices.
    """
    
    boundaries = llm_call(prompt)  # Single LLM call
    return parse_boundaries(boundaries)
```

### Phase 2: Episode Generation (Week 1-2)
Generate human-scale architecture narratives:

```python
def generate_architecture_episode(segment: List[ArchitectureEvaluation]) -> ArchitectureEpisode:
    """Generate episodic summary of architecture evolution segment"""
    
    # Single LLM prompt for episode generation
    prompt = f"""
    Create a coherent episodic summary of this architecture evolution segment.
    Focus on:
    - Natural language description of architectural developments
    - Key breakthrough moments and discoveries
    - Performance trajectory and outcomes
    - Temporal context and narrative flow
    
    Segment: {segment}
    
    Generate a human-readable episode that captures the essence of this architectural journey.
    """
    
    episode_content = llm_call(prompt)  # Single LLM call
    return parse_episode(episode_content)
```

### Phase 3: BM25 Retrieval System (Week 2)
Implement efficient episode retrieval:

```python
from rank_bm25 import BM25Okapi

class ArchitectureEpisodeRetriever:
    """Nemori-inspired BM25 retrieval for architecture episodes"""
    
    def __init__(self):
        self.episodes: List[ArchitectureEpisode] = []
        self.bm25 = None
        
    def build_index(self, episodes: List[ArchitectureEpisode]):
        """Build BM25 index from episodes"""
        self.episodes = episodes
        
        # Tokenize all episodes
        tokenized_episodes = []
        for episode in episodes:
            tokens = self._tokenize_episode_content(episode)
            tokenized_episodes.append(tokens)
        
        # Build BM25 index
        self.bm25 = BM25Okapi(tokenized_episodes)
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[ArchitectureEpisode, float]]:
        """Retrieve relevant episodes"""
        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)
        
        # Return top-k with episodes and scores
        results = [(self.episodes[i], scores[i]) for i in range(len(scores))]
        return sorted(results, key=lambda x: x[1], reverse=True)[:top_k]
```

## 📊 Performance Expectations

Based on Nemori's results, we expect:

### Retrieval Performance
- **Latency**: <10ms per retrieval query (BM25 only)
- **Accuracy**: >85% relevant episode retrieval
- **Scalability**: Support 10,000+ episodes efficiently

### Memory Efficiency
- **Token Usage**: Reduced by 50% compared to raw evaluation storage
- **Storage**: Compact episode representation
- **Indexing**: Fast BM25 index construction and updates

### Narrative Quality
- **Human Readability**: Episodes understandable by researchers
- **Coherence**: Logical narrative flow within episodes
- **Information Preservation**: Key details retained through semantic memory

## 🧪 Testing Strategy

### Episode Quality Tests
```python
def test_episode_narrative_quality():
    """Test human readability and coherence of generated episodes"""
    episode = generate_test_episode()
    
    # Human evaluation metrics
    assert episode.narrative_coherence_score > 0.7
    assert len(episode.narrative_summary.split()) > 50  # Substantial content
    assert contains_architectural_terminology(episode.narrative_summary)
```

### Retrieval Accuracy Tests
```python
def test_bm25_retrieval_accuracy():
    """Test BM25 retrieval accuracy for architecture queries"""
    retriever = setup_test_retriever()
    
    # Test queries
    results = retriever.retrieve("transformer attention mechanisms")
    
    # Verify relevant episodes retrieved
    assert len(results) > 0
    assert any("attention" in episode.narrative_summary.lower() 
              for episode, score in results[:5])
```

### Boundary Detection Tests
```python
def test_episode_boundary_detection():
    """Test accuracy of episode boundary detection"""
    evolution_sequence = create_test_evolution_sequence()
    boundaries = detect_episode_boundaries(evolution_sequence)
    
    # Verify meaningful boundaries detected
    expected_boundaries = [10, 25, 40]  # Known ground truth
    boundary_accuracy = calculate_boundary_accuracy(boundaries, expected_boundaries)
    assert boundary_accuracy > 0.85
```

## 🔄 Integration with Existing Architecture

### Connecting to epLSTM Architecture
```python
class NemoriInspiredEpisodicLSTM(EpisodicLSTMCell):
    """Enhanced epLSTM with Nemori-inspired episode processing"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.episode_detector = EpisodeBoundaryDetector()
        self.episode_generator = ArchitectureEpisodeGenerator()
        self.episode_retriever = ArchitectureEpisodeRetriever()
    
    def process_evolution_sequence(self, sequence: List[ArchitectureEvaluation]):
        """Process architecture evolution using Nemori approach"""
        
        # 1. Detect episode boundaries
        boundaries = self.episode_detector.detect_boundaries(sequence)
        
        # 2. Generate episodes
        episodes = []
        for start, end in zip([0] + boundaries, boundaries + [len(sequence)]):
            segment = sequence[start:end]
            episode = self.episode_generator.generate_episode(segment)
            episodes.append(episode)
        
        # 3. Update episode index
        self.episode_retriever.add_episodes(episodes)
        
        return episodes
    
    def retrieve_relevant_episodes(self, query: str) -> List[ArchitectureEpisode]:
        """Retrieve episodes relevant to current architecture challenge"""
        results = self.episode_retriever.retrieve(query, top_k=5)
        return [episode for episode, score in results]
```

## 📈 Success Metrics

### Quantitative Metrics
- **Episode Boundary Accuracy**: >85% correct boundary detection
- **Retrieval Precision@5**: >80% relevant episodes in top 5
- **Narrative Coherence Score**: >0.7 average across episodes
- **Token Efficiency**: 50% reduction vs. raw evaluation storage
- **Retrieval Latency**: <10ms per query

### Qualitative Metrics
- **Human Readability**: Episodes understandable by architecture researchers
- **Information Preservation**: Key architectural insights retained
- **Narrative Flow**: Logical progression within episodes
- **Archetypal Alignment**: Episodes align with architectural development patterns

---

**Integration Status**: ✅ READY FOR IMPLEMENTATION  
**Nemori Reference**: [https://github.com/nemori-ai/nemori.git](https://github.com/nemori-ai/nemori.git)  
**Next Review Date**: 2025-09-29  
**Integration Owner**: ASI-Arch Context Engineering Team
