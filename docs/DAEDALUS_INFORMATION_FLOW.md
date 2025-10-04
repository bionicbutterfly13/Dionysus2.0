# Daedalus Information Flow Architecture

**Document Version**: 1.0
**Date**: 2025-10-02
**Constitutional Framework**: CONST_ARCH_2025

---

## 🎯 Overview

This document maps the complete flow of information from user upload through Daedalus perceptual gateway to Neo4j knowledge graph storage, including all data transformations, filters, and relationship extraction mechanisms.

---

## 📊 High-Level Architecture

```
USER UPLOAD (PDF/TXT/etc.)
    ↓
┌─────────────────────────────────────────────────────────┐
│ LEVEL 1: DAEDALUS PERCEPTUAL GATEWAY                    │
│ File: backend/src/services/daedalus.py                  │
│ Method: receive_perceptual_information()                │
│ Markov Blanket: Security/Constitutional Compliance      │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ LEVEL 2: LANGGRAPH WORKFLOW                             │
│ File: backend/src/services/document_processing_graph.py │
│ State Machine: 6 Nodes with Iterative Refinement        │
│ Markov Blanket: ASI-GO-2 Processing Coordination        │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ LEVEL 3: NEO4J KNOWLEDGE GRAPH                          │
│ Storage: Document + Concept + Basin + Relationship Nodes│
│ AutoSchemaKG: Dynamic relationship type creation via LLM │
│ Markov Blanket: Memory Formation & Retrieval            │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ LEVEL 4: COGNITION BASE LEARNING                        │
│ File: backend/src/services/document_cognition_base.py   │
│ Learning: Successful patterns → Future strategy updates │
│ Markov Blanket: Meta-cognitive Optimization             │
└─────────────────────────────────────────────────────────┘
```

---

## 🔄 Detailed Information Flow

### **LEVEL 1: Daedalus Perceptual Gateway**

**Location**: `backend/src/services/daedalus.py:41`

#### Entry Point
```python
def receive_perceptual_information(
    data: BinaryIO,              # File binary content
    tags: List[str] = None,      # User-provided tags
    max_iterations: int = 3,     # Max refinement loops
    quality_threshold: float = 0.7  # Quality target
) -> Dict[str, Any]
```

#### Input Data Structure
```python
{
    "data": <BinaryIO>,           # Raw file bytes
    "tags": ["Research", "AI"],   # Optional categorization
    "max_iterations": 3,          # Refinement limit
    "quality_threshold": 0.7      # Quality target (0.0-1.0)
}
```

#### Processing
1. **Validate**: Check data exists
2. **Read**: Extract binary content
3. **Prepare**: Create file-like object with metadata
4. **Dispatch**: Route to DocumentProcessingGraph

#### Output Data Structure
```python
{
    "status": "received",
    "document": {...},           # Document metadata
    "extraction": {...},         # Extracted concepts/chunks
    "consciousness": {...},      # Basin/ThoughtSeed data
    "research": {...},           # Research questions
    "quality": {...},            # Quality metrics
    "meta_cognitive": {...},     # Meta-awareness data
    "workflow": {
        "iterations": 2,
        "messages": ["Extracted 42 concepts...", ...]
    },
    "timestamp": 1696234567.89,
    "source": "langgraph_workflow"
}
```

---

### **LEVEL 2: LangGraph Workflow State Machine**

**Location**: `backend/src/services/document_processing_graph.py:58`

#### State Machine Graph
```
START → [Node 1] → [Node 2] → [Node 3] → [Node 4] → [Node 5] → END
           ↓         ↓          ↓          ↓          ↓ ↑
        Extract   Research   Conscious   Analyze   Decision
                                                      ↓
                                                   [Refine] → Loop to Node 3
```

---

#### **Node 1: Extract & Process** (Line 148)

**Function**: `_extract_and_process_node()`

##### Input
```python
{
    "content": b"PDF binary...",
    "filename": "markov_blankets.pdf",
    "tags": ["Research", "Neuroscience"]
}
```

##### Processing Steps
1. **Content Extraction**
   - PDF → Text via PyPDF2
   - Text → UTF-8 decoding
   - Markdown conversion

2. **Semantic Chunking**
   - Text → 512-character chunks
   - Overlap: 50 characters
   - Preserves sentence boundaries

3. **Concept Extraction**
   ```python
   # Pattern 1: Capitalized phrases
   r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
   → "Markov Blanket", "Active Inference"

   # Pattern 2: Multi-word technical terms
   r'\b[A-Z][a-z]+\s+[a-z]+(?:\s+[a-z]+)?\b'
   → "Free Energy Principle", "Prediction Error"

   # Pattern 3: Frequent domain terms (3+ occurrences)
   word_frequency_analysis()
   → "inference", "perception", "action"
   ```

4. **Basin Creation**
   - Concepts → Attractor basins
   - Co-occurring concepts → Basin clusters
   - Basin strength calculation

##### Output Data Structure
```python
DocumentProcessingResult {
    content_hash: "abc123def456...",        # MD5 of content
    filename: "markov_blankets.pdf",
    concepts: [                             # Top concepts extracted
        "Markov Blanket",
        "Active Inference",
        "Free Energy Principle",
        "Prediction Error",
        ...                                 # Up to 50 concepts
    ],
    chunks: [                               # Text chunks
        "Markov blankets are...",
        "Active inference minimizes...",
        ...
    ],
    summary: "Markov blankets are statistical boundaries...",  # First 200 chars
    basins_created: 8,                      # Number of attractor basins
    thoughtseeds_generated: [               # Generated ThoughtSeeds
        "ts_markov_blanket_001",
        "ts_free_energy_002",
        ...
    ],
    patterns_learned: [                     # Discovered patterns
        {
            "pattern_type": "hierarchical_structure",
            "confidence": 0.85,
            "description": "Nested Markov blankets..."
        },
        ...
    ]
}
```

**Filters Applied**:
- Concepts: Top 50 by frequency/importance
- Chunks: All preserved
- Basins: Automatically created for concept clusters

---

#### **Node 2: Generate Research Plan** (Line 173)

**Function**: `_generate_research_plan_node()`

##### Input
```python
{
    "concepts": ["Markov Blanket", "Active Inference", ...],
    "prediction_errors": None  # TODO: Active inference integration
}
```

##### Processing Steps
1. **ASI-GO-2 Researcher Analysis**
   - Review existing cognition base
   - Identify knowledge gaps
   - Generate exploration strategies

2. **R-Zero Challenging Questions**
   ```python
   # Question types:
   - "How does X relate to Y?"
   - "What happens if X fails?"
   - "Can we extend X to domain Y?"
   - "What evidence supports X?"
   ```

3. **Curiosity Trigger Identification**
   - High prediction error concepts
   - Underspecified relationships
   - Novel term combinations

##### Output Data Structure
```python
research_plan {
    challenging_questions: [
        "How do nested Markov blankets enable hierarchical inference?",
        "What role does prediction error play in boundary formation?",
        "Can Markov blankets explain consciousness emergence?",
        ...                                 # 5-10 questions
    ],
    curiosity_triggers: [
        {
            "concept": "Prediction Error",
            "prediction_error": 0.75,       # High uncertainty
            "priority": "high",
            "reason": "Key to understanding active inference"
        },
        ...
    ],
    exploration_plan: {
        "immediate": ["Investigate prediction error minimization"],
        "short_term": ["Map hierarchical relationships"],
        "long_term": ["Connect to consciousness theories"]
    },
    cognitive_strategy: "deep_exploration"  # vs "breadth_first"
}
```

**Filters Applied**:
- Questions: Top 10 most challenging
- Curiosity triggers: Prediction error > 0.6
- Exploration plan: 3 levels (immediate/short/long)

---

#### **Node 3: Consciousness Processing** (Line 199)

**Function**: `_consciousness_processing_node()`

##### Processing
- Refine attractor basins
- Generate ThoughtSeeds
- Track basin transitions
- Measure consciousness emergence

##### Output
```python
consciousness_data {
    basins_refined: 8,
    thoughtseeds_active: 5,
    emergence_detected: True,
    consciousness_score: 0.72
}
```

---

#### **Node 4: Analyze Results** (Line 218)

**Function**: `_analyze_results_node()`

##### Input
```python
{
    "processing_result": DocumentProcessingResult,
    "research_plan": research_plan
}
```

##### Processing Steps
1. **Quality Assessment**
   ```python
   quality_scores = {
       "completeness": calculate_completeness(concepts, chunks),
       "coherence": calculate_coherence(concepts, relationships),
       "depth": calculate_depth(basins, patterns),
       "novelty": calculate_novelty(concepts, cognition_base)
   }
   overall_quality = weighted_average(quality_scores)
   ```

2. **Insight Extraction**
   - Pattern recognition
   - Relationship discovery
   - Meta-cognitive tracking

3. **Recommendation Generation**
   - "Extract more relationships between X and Y"
   - "Increase chunk overlap for better context"
   - "Focus on hierarchical structures"

##### Output Data Structure
```python
analysis {
    quality_scores: {
        "overall": 0.85,
        "completeness": 0.90,
        "coherence": 0.85,
        "depth": 0.80,
        "novelty": 0.75
    },
    insights: [
        "Document heavily focuses on hierarchical inference",
        "Strong causal relationships detected",
        "Novel application of Markov blankets to consciousness"
    ],
    recommendations: [
        "Extract additional hierarchical relationships",
        "Investigate temporal dynamics",
        "Connect to existing consciousness theories"
    ],
    meta_cognitive: {
        "processing_efficiency": 0.88,
        "strategy_effectiveness": 0.82,
        "learning_rate": 0.15
    }
}
```

**Filters Applied**:
- Quality threshold: 0.7 minimum
- Insights: Top 5-10 most significant
- Recommendations: Actionable only

---

#### **Node 5: Decision Point** (Line 238)

**Function**: `_should_refine()`

##### Decision Logic
```python
if iteration >= max_iterations:
    return "complete"  # Stop iterating

if overall_quality >= quality_threshold:
    return "complete"  # Quality target met

return "refine"  # Loop back to Node 3
```

##### Example Flow
```
Iteration 1: quality=0.65 → REFINE → Loop to Node 3
Iteration 2: quality=0.78 → COMPLETE → Proceed to Node 6
```

---

#### **Node 6: Finalize & Persist to Neo4j** (Line 283)

**Function**: `_finalize_output_node()` + `_store_to_neo4j()`

This is where **AutoSchemaKG** happens.

---

### **LEVEL 3: Neo4j Knowledge Graph Storage**

**Location**: `backend/src/services/document_processing_graph.py:349`

---

#### **Step 1: Create Document Node** (Line 425)

##### Cypher Query
```cypher
CREATE (d:Document {
    id: $content_hash,              # MD5 hash (unique identifier)
    filename: $filename,
    content_hash: $content_hash,
    tags: $tags,                    # ["Research", "Neuroscience"]
    extracted_text: $summary,       # First 200 chars
    upload_timestamp: $timestamp,
    processing_status: 'completed',
    chunks_count: $chunks_count,
    concepts_count: $concepts_count,
    basins_created: $basins_count,
    quality_score: $quality_score,
    iterations: $iterations
})
RETURN d.id as document_id
```

##### Data Structure
```python
{
    "id": "abc123def456",
    "filename": "markov_blankets.pdf",
    "content_hash": "abc123def456",
    "tags": ["Research", "Neuroscience"],
    "extracted_text": "Markov blankets are statistical boundaries that separate internal states from external states...",
    "upload_timestamp": "2025-10-02T14:32:15.123Z",
    "processing_status": "completed",
    "chunks_count": 15,
    "concepts_count": 42,
    "basins_created": 8,
    "quality_score": 0.85,
    "iterations": 2
}
```

**Filter**: NONE - All documents are stored

---

#### **Step 2: Create Concept Nodes** (Line 459)

##### Cypher Query
```cypher
MATCH (d:Document {id: $document_id})
MERGE (c:Concept {id: $concept_id})
ON CREATE SET
    c.text = $concept_text,
    c.created_at = $timestamp
MERGE (d)-[r:HAS_CONCEPT]->(c)
ON CREATE SET
    r.extraction_index = $index,
    r.created_at = $timestamp
```

##### Data Structure
```python
# Concept node
{
    "id": "abc123_markov",          # md5(document_id:concept_text)[:16]
    "text": "Markov Blanket",
    "created_at": "2025-10-02T14:32:16.456Z"
}

# Relationship
(Document)-[HAS_CONCEPT {
    extraction_index: 0,            # Order extracted
    created_at: "2025-10-02T14:32:16.456Z"
}]->(Concept)
```

**Filter**: Top 20 concepts per document (Line 371)

**Reason**: Avoid graph bloat while preserving most important concepts

---

#### **Step 3: Extract Concept Relationships (AutoSchemaKG)** (Line 610)

This is the **critical AutoSchemaKG step** - dynamic relationship type creation.

##### Method A: LLM-based Relationship Extraction (Line 682)

**LLM**: Ollama qwen2.5:14b (local)

**Prompt Template**:
```
You are analyzing research concepts to build a knowledge graph.

Concepts:
[
  "Markov Blanket",
  "Active Inference",
  "Free Energy Principle",
  "Prediction Error",
  "Perception",
  "Action"
]

Extract ALL semantic relationships between these concepts. Be thorough and precise.

For EACH pair of related concepts, identify:
1. The SOURCE concept
2. The TARGET concept
3. The RELATIONSHIP TYPE (use precise verbs: CAUSES, ENABLES, REQUIRES, EXTENDS, VALIDATES, CONTRADICTS, REPLACES, MODIFIES, REGULATES, etc.)
4. CONFIDENCE (0.0-1.0)

Rules:
- Use UPPERCASE_WITH_UNDERSCORES for relationship types (e.g., THEORETICALLY_EXTENDS, EMPIRICALLY_VALIDATES)
- Be specific: prefer "EMPIRICALLY_VALIDATES" over generic "RELATES_TO"
- Only include relationships you're confident about (>0.5 confidence)
- For research papers: look for theoretical extensions, empirical validations, contradictions, refinements

Return ONLY a JSON array, no other text:
[
  {"source": "concept A", "target": "concept B", "type": "RELATIONSHIP_TYPE", "confidence": 0.85},
  ...
]
```

**LLM Output Example**:
```json
[
  {
    "source": "Active Inference",
    "target": "Free Energy Principle",
    "type": "MINIMIZES",
    "confidence": 0.95
  },
  {
    "source": "Markov Blanket",
    "target": "Active Inference",
    "type": "ENABLES",
    "confidence": 0.90
  },
  {
    "source": "Prediction Error",
    "target": "Perception",
    "type": "UPDATES",
    "confidence": 0.85
  },
  {
    "source": "Free Energy Principle",
    "target": "Consciousness",
    "type": "THEORETICALLY_EXPLAINS",
    "confidence": 0.78
  },
  {
    "source": "Hierarchical Model",
    "target": "Free Energy Principle",
    "type": "EXTENDS",
    "confidence": 0.82
  }
]
```

**Relationship Types Created** (examples):
- `MINIMIZES` - Optimization relationships
- `ENABLES` - Necessary conditions
- `UPDATES` - Dynamic modifications
- `THEORETICALLY_EXPLAINS` - Explanatory power
- `EXTENDS` - Theoretical extensions
- `EMPIRICALLY_VALIDATES` - Experimental support
- `CONTRADICTS` - Conflicting theories
- `REFINES` - Incremental improvements
- `REQUIRES` - Dependencies
- `PRODUCES` - Causal outcomes
- `REGULATES` - Control mechanisms

**Key Point**: The LLM **creates new relationship types dynamically** based on semantic analysis. Neo4j accepts any relationship type, so the schema grows organically.

##### Method B: Heuristic Fallback (Line 757)

If LLM fails (timeout, error, etc.), uses pattern matching:

```python
verbs_map = {
    'cause': 'CAUSES',
    'enable': 'ENABLES',
    'require': 'REQUIRES',
    'modify': 'MODIFIES',
    'regulate': 'REGULATES',
    'control': 'CONTROLS',
    'produce': 'PRODUCES',
    'influence': 'INFLUENCES',
    'affect': 'AFFECTS',
    'improve': 'IMPROVES',
    'enhance': 'ENHANCES',
    'replace': 'REPLACES',
    'supersede': 'SUPERSEDES',
    'neutralize': 'NEUTRALIZES',
    'negate': 'NEGATES',
    'create': 'CREATES',
    'generate': 'GENERATES',
    'advance': 'ADVANCES',
    'develop': 'DEVELOPS',
    'deprecate': 'DEPRECATES'
}

# Scan concept text for verbs
if 'cause' in concept1_text or 'cause' in concept2_text:
    relationships.append(('CAUSES', 0.6))
```

**Confidence scores**:
- Verb detection: 0.6
- Semantic similarity: 0.7
- Opposition: 0.6
- Hierarchy: 0.7
- Generic co-occurrence: 0.3

**Filter**: Only relationships with **confidence ≥ 0.5** are created

---

#### **Step 4: Create Relationships in Neo4j** (Line 634)

##### Cypher Query (Dynamic)
```cypher
MATCH (c1:Concept), (c2:Concept)
WHERE c1.text = $source AND c2.text = $target
MERGE (c1)-[r:MINIMIZES]->(c2)  # Relationship type from LLM
SET r.confidence = $confidence,
    r.document_id = $document_id,
    r.created_at = $timestamp
```

##### Data Structure
```python
# Relationship
(Concept {text: "Active Inference"})-[
    MINIMIZES {
        confidence: 0.95,
        document_id: "abc123",
        created_at: "2025-10-02T14:32:18.789Z"
    }
]->(Concept {text: "Free Energy"})
```

**Filter**: Confidence ≥ 0.5

**Example Neo4j Graph**:
```
(Markov Blanket)-[ENABLES {confidence: 0.90}]->(Active Inference)
                  ↓
            [MINIMIZES {confidence: 0.95}]
                  ↓
          (Free Energy Principle)
                  ↓
    [THEORETICALLY_EXPLAINS {confidence: 0.78}]
                  ↓
            (Consciousness)
```

---

#### **Step 5: Create AttractorBasin Nodes** (Line 483)

##### Cypher Query
```cypher
MATCH (d:Document {id: $document_id})
CREATE (b:AttractorBasin {
    id: $basin_id,
    center_concept: $center_concept,
    strength: $strength,
    stability: $stability,
    created_at: $timestamp
})
CREATE (d)-[:CREATED_BASIN]->(b)

# Link basin to concepts
MATCH (b:AttractorBasin {id: $basin_id})
MATCH (c:Concept) WHERE c.text IN $related_concepts
MERGE (b)-[:ATTRACTS]->(c)
```

##### Data Structure
```python
# Basin node
{
    "id": "basin_abc123_markov",
    "center_concept": "Markov Blanket",
    "strength": 1.0,                # Basin attraction strength
    "stability": 0.8,               # Temporal stability
    "created_at": "2025-10-02T14:32:19.123Z"
}

# Relationships
(Document)-[CREATED_BASIN]->(AttractorBasin)
(AttractorBasin)-[ATTRACTS]->(Concept)
```

**Filter**: Top 10 patterns (Line 400)

**Reason**: Focus on most significant attractor dynamics

---

#### **Step 6: Link Curiosity Triggers** (Line 523)

##### Cypher Query
```cypher
MATCH (d:Document {id: $document_id})
MATCH (c:Concept) WHERE c.text = $concept
MERGE (d)-[r:CURIOSITY_TRIGGER]->(c)
SET r.prediction_error = $prediction_error,
    r.priority = $priority,
    r.created_at = $timestamp
```

##### Data Structure
```python
# Relationship
(Document)-[
    CURIOSITY_TRIGGER {
        prediction_error: 0.75,     # Uncertainty measure
        priority: "high",           # Investigation priority
        created_at: "2025-10-02T14:32:19.456Z"
    }
]->(Concept {text: "Prediction Error"})
```

**Filter**: Top 10 curiosity triggers per document

**Reason**: Focus research on highest uncertainty areas

---

### **LEVEL 4: Cognition Base Learning Loop**

**Location**: `backend/src/services/document_processing_graph.py:560`

#### **Function**: `_update_cognition_from_results()`

##### Input
```python
{
    "concepts": ["Markov Blanket", ...],
    "relationships": [{"source": "A", "target": "B", "type": "CAUSES", ...}],
    "quality_score": 0.85,
    "basin_context": {...}
}
```

##### Learning Logic

**1. Record Successful Patterns** (quality ≥ 0.7)
```python
if quality_score > 0.7:
    pattern = {
        "pattern_type": "concept_relationship_extraction",
        "concepts_count": 42,
        "relationships_extracted": 25,
        "relationship_types": ["CAUSES", "ENABLES", "MINIMIZES", ...],
        "quality_score": 0.85,
        "success": True,
        "timestamp": "2025-10-02T14:32:20.000Z"
    }

    cognition_base.record_successful_pattern(
        category="relationship_extraction",
        pattern=pattern
    )
```

**2. Boost Frequent Relationship Types**
```python
# If relationship type appears 3+ times in one document
for rel_type, count in relationship_type_counts.items():
    if count >= 3:
        cognition_base.boost_strategy_priority(
            category="relationship_types",
            strategy_name=rel_type,
            boost_amount=0.1  # 10% priority increase
        )
```

##### Effect on Future Processing

**Example**: After processing 10 neuroscience papers:

```python
cognition_base.strategies = {
    "relationship_types": {
        "ENABLES": {
            "priority": 0.85,       # Boosted from 0.50 (common in neuro)
            "success_rate": 0.92,
            "usage_count": 47
        },
        "THEORETICALLY_EXPLAINS": {
            "priority": 0.78,       # Boosted from 0.50
            "success_rate": 0.88,
            "usage_count": 32
        },
        "EMPIRICALLY_VALIDATES": {
            "priority": 0.72,       # Boosted from 0.50
            "success_rate": 0.85,
            "usage_count": 28
        },
        "UNRELATED_TO": {
            "priority": 0.20,       # Decreased (rarely useful)
            "success_rate": 0.35,
            "usage_count": 5
        }
    }
}
```

**Future documents** will prioritize extracting high-priority relationship types.

---

## 📋 Complete Filter Summary

| **Level** | **Component** | **Filter Criteria** | **Threshold** | **Reason** |
|-----------|---------------|---------------------|---------------|-----------|
| **Level 1** | Documents | ALL stored | None | Complete audit trail |
| **Level 2 Node 1** | Concepts | Frequency + importance | Top 50 | Focus on key terms |
| **Level 2 Node 1** | Chunks | ALL preserved | None | Context preservation |
| **Level 2 Node 1** | Basins | Automatic clustering | All clusters | Attractor dynamics |
| **Level 2 Node 2** | Research questions | Challenging + novel | Top 10 | Prioritize exploration |
| **Level 2 Node 2** | Curiosity triggers | Prediction error | ≥ 0.6 | Focus on uncertainty |
| **Level 2 Node 4** | Quality check | Overall quality | ≥ 0.7 | Iteration decision |
| **Level 3 Step 2** | Concept storage | Top concepts | Top 20 | Avoid graph bloat |
| **Level 3 Step 3** | LLM relationships | Confidence | ≥ 0.5 | High-confidence only |
| **Level 3 Step 3** | Heuristic relationships | Confidence | ≥ 0.5 | Minimum reliability |
| **Level 3 Step 5** | AttractorBasin storage | Pattern significance | Top 10 | Most important dynamics |
| **Level 3 Step 6** | Curiosity links | Prediction error | Top 10 | Highest uncertainty |
| **Level 4** | Learning patterns | Processing quality | ≥ 0.7 | Learn from success |
| **Level 4** | Relationship boosting | Frequency | ≥ 3 occurrences | Common patterns |

---

## 🗂️ Data Structure Summary

### **In-Memory Python Objects**

```python
# Level 1: Daedalus Input
{
    "data": BinaryIO,
    "tags": List[str],
    "max_iterations": int,
    "quality_threshold": float
}

# Level 2: Processing Result
DocumentProcessingResult {
    content_hash: str,
    filename: str,
    concepts: List[str],
    chunks: List[str],
    summary: str,
    basins_created: int,
    thoughtseeds_generated: List[str],
    patterns_learned: List[Dict[str, Any]]
}

# Level 2: Research Plan
{
    "challenging_questions": List[str],
    "curiosity_triggers": List[Dict],
    "exploration_plan": Dict,
    "cognitive_strategy": str
}

# Level 2: Analysis
{
    "quality_scores": Dict[str, float],
    "insights": List[str],
    "recommendations": List[str],
    "meta_cognitive": Dict
}
```

### **Neo4j Graph Structures**

```cypher
// Document Node
(:Document {
    id: string,                 // MD5 hash
    filename: string,
    content_hash: string,
    tags: [string],
    extracted_text: string,
    upload_timestamp: datetime,
    processing_status: string,
    chunks_count: int,
    concepts_count: int,
    basins_created: int,
    quality_score: float,
    iterations: int
})

// Concept Node
(:Concept {
    id: string,                 // md5(doc_id:concept)[:16]
    text: string,
    created_at: datetime
})

// AttractorBasin Node
(:AttractorBasin {
    id: string,
    center_concept: string,
    strength: float,
    stability: float,
    created_at: datetime
})

// Relationships
(Document)-[HAS_CONCEPT {
    extraction_index: int,
    created_at: datetime
}]->(Concept)

(Concept)-[DYNAMIC_TYPE {       // Type created by LLM
    confidence: float,
    document_id: string,
    created_at: datetime
}]->(Concept)

(Document)-[CREATED_BASIN]->(AttractorBasin)

(AttractorBasin)-[ATTRACTS]->(Concept)

(Document)-[CURIOSITY_TRIGGER {
    prediction_error: float,
    priority: string,
    created_at: datetime
}]->(Concept)
```

---

## 🔍 AutoSchemaKG: Dynamic Relationship Type Creation

**Key Innovation**: Neo4j schema grows **organically** based on LLM semantic analysis.

### Traditional Approach (Predefined Schema)
```cypher
// Fixed relationship types
(A)-[RELATES_TO]->(B)
(A)-[PART_OF]->(B)
(A)-[CAUSES]->(B)

// Problem: Limited expressiveness
// Can't represent: "EMPIRICALLY_VALIDATES", "THEORETICALLY_EXTENDS", etc.
```

### AutoSchemaKG Approach (Dynamic Schema)
```cypher
// LLM creates relationship types on-the-fly
(Active_Inference)-[MINIMIZES]->(Free_Energy)
(Markov_Blanket)-[ENABLES]->(Active_Inference)
(Hierarchical_Model)-[THEORETICALLY_EXTENDS]->(Free_Energy_Principle)
(Experiment_123)-[EMPIRICALLY_VALIDATES]->(Prediction_Error_Theory)
(Old_Framework)-[SUPERSEDED_BY]->(New_Framework)

// Advantage: Rich, precise semantic relationships
```

### Example: Processing Your Markov Blanket PDF

**LLM Discovers**:
```json
[
  {"source": "Markov Blanket", "target": "Active Inference", "type": "ENABLES", "confidence": 0.90},
  {"source": "Active Inference", "target": "Free Energy", "type": "MINIMIZES", "confidence": 0.95},
  {"source": "Prediction Error", "target": "Perception", "type": "UPDATES", "confidence": 0.85},
  {"source": "Sensory States", "target": "Internal States", "type": "INFLUENCES_VIA_BLANKET", "confidence": 0.82},
  {"source": "Active States", "target": "External States", "type": "MODIFIES_VIA_ACTION", "confidence": 0.88},
  {"source": "Hierarchical Inference", "target": "Free Energy Principle", "type": "EXTENDS", "confidence": 0.80}
]
```

**Neo4j Creates**:
```cypher
CREATE (mb:Concept {text: "Markov Blanket"})
CREATE (ai:Concept {text: "Active Inference"})
CREATE (fe:Concept {text: "Free Energy"})
CREATE (pe:Concept {text: "Prediction Error"})
CREATE (p:Concept {text: "Perception"})

MERGE (mb)-[r1:ENABLES {confidence: 0.90}]->(ai)
MERGE (ai)-[r2:MINIMIZES {confidence: 0.95}]->(fe)
MERGE (pe)-[r3:UPDATES {confidence: 0.85}]->(p)
// ... and so on
```

**Result**: Rich semantic graph that captures precise theoretical relationships from the paper.

---

## 🎯 Summary: From Upload to Knowledge Graph

1. **User uploads** PDF → Daedalus gateway
2. **Daedalus** validates → LangGraph workflow
3. **LangGraph Node 1** extracts → 50 concepts, 15 chunks, 8 basins
4. **LangGraph Node 2** generates → 10 research questions, 5 curiosity triggers
5. **LangGraph Node 3** refines → Consciousness processing
6. **LangGraph Node 4** analyzes → Quality = 0.85
7. **LangGraph Node 5** decides → COMPLETE (quality ≥ 0.7)
8. **LangGraph Node 6** persists → Neo4j
9. **Neo4j Step 1** creates → Document node
10. **Neo4j Step 2** creates → 20 Concept nodes (filtered from 50)
11. **Neo4j Step 3** LLM extracts → 25 relationships with dynamic types
12. **Neo4j Step 4** creates → 25 relationships (confidence ≥ 0.5)
13. **Neo4j Step 5** creates → 10 AttractorBasin nodes
14. **Neo4j Step 6** creates → 10 CURIOSITY_TRIGGER relationships
15. **Cognition Base** learns → Pattern successful, boost priorities

**Final Neo4j Graph**:
- 1 Document node
- 20 Concept nodes
- 10 AttractorBasin nodes
- 20 HAS_CONCEPT relationships
- 25 semantic relationships (dynamic types from LLM)
- 10 ATTRACTS relationships
- 10 CURIOSITY_TRIGGER relationships

**Total**: 31 nodes, 65 relationships

---

## 🔗 Key Files Reference

| **Component** | **File** | **Key Functions** |
|---------------|----------|-------------------|
| Daedalus Gateway | `backend/src/services/daedalus.py` | `receive_perceptual_information()` |
| LangGraph Workflow | `backend/src/services/document_processing_graph.py` | `process_document()`, `_extract_and_process_node()` |
| Consciousness Processing | `backend/src/services/consciousness_document_processor.py` | `process_pdf()`, `process_text()` |
| ASI-GO-2 Researcher | `backend/src/services/document_researcher.py` | `generate_research_questions()` |
| ASI-GO-2 Analyst | `backend/src/services/document_analyst.py` | `analyze_processing_result()` |
| Cognition Base | `backend/src/services/document_cognition_base.py` | `record_successful_pattern()` |
| Neo4j Schema | `extensions/context_engineering/neo4j_unified_schema.py` | `connect()`, Database driver |
| API Routes | `backend/src/api/routes/documents.py` | `POST /api/documents` |
| Constitutional Rules | `dionysus-source/CONSTITUTIONAL_DOCUMENT_PROCESSING_RULE.md` | Markov blanket principles |

---

**End of Document**
