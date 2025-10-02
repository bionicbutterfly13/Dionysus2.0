# Dionysus Document Processing Architecture

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Upload                              │
│                    (PDF, Text, Markdown)                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Daedalus Gateway                             │
│                  (Perceptual Information)                        │
│                  receive_perceptual_information()                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              DocumentProcessingGraph (LangGraph)                 │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Node 1: Extract & Process                                │  │
│  │ - Content hash (SurfSense)                               │  │
│  │ - Markdown conversion (SurfSense)                        │  │
│  │ - Concept extraction (Dionysus)                          │  │
│  │ - Semantic chunking (SurfSense)                          │  │
│  └──────────────────┬───────────────────────────────────────┘  │
│                     │                                           │
│                     ▼                                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Node 2: Generate Research Plan                           │  │
│  │ - Active Inference prediction errors                     │  │
│  │ - Identify curiosity triggers                            │  │
│  │ - Generate challenging questions (R-Zero)                │  │
│  │ - Create exploration plan (ASI-GO-2 Researcher)          │  │
│  └──────────────────┬───────────────────────────────────────┘  │
│                     │                                           │
│                     ▼                                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Node 3: Consciousness Processing                         │  │
│  │ - Create attractor basins                                │  │
│  │ - Generate thoughtseeds                                  │  │
│  │ - Update hierarchical beliefs                            │  │
│  │ - Track pattern emergence                                │  │
│  └──────────────────┬───────────────────────────────────────┘  │
│                     │                                           │
│                     ▼                                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Node 4: Analyze Results                                  │  │
│  │ - Quality assessment (ASI-GO-2 Analyst)                  │  │
│  │ - Extract insights                                       │  │
│  │ - Meta-cognitive analysis                                │  │
│  │ - Generate recommendations                               │  │
│  └──────────────────┬───────────────────────────────────────┘  │
│                     │                                           │
│                     ▼                                           │
│              ┌──────────────┐                                   │
│              │ Quality OK?  │                                   │
│              └──────┬───────┘                                   │
│                     │                                           │
│         ┌───────────┴───────────┐                               │
│         │ No (< threshold)      │ Yes (>= threshold)            │
│         ▼                       ▼                               │
│  ┌────────────────┐      ┌──────────────────────────────────┐  │
│  │ Node 5: Refine │      │ Node 6: Finalize Output          │  │
│  │ - Adjust params│      │ - Package results                │  │
│  │ - Loop back    │      │ - Return to gateway              │  │
│  └────────┬───────┘      └──────────────────────────────────┘  │
│           │                                    │                │
│           └─────(iteration++)──────────────────┘                │
│                  (if < max_iterations)                          │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Output to User / Storage                      │
│                                                                  │
│  - Document metadata (hash, chunks, summary)                     │
│  - Concepts extracted                                            │
│  - Consciousness artifacts (basins, thoughtseeds)                │
│  - Research questions (curiosity triggers)                       │
│  - Quality scores and insights                                   │
│  - Meta-cognitive analysis                                       │
└─────────────────────────────────────────────────────────────────┘
```

## Component Interactions

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DocumentProcessingGraph                           │
│                                                                          │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐  │
│  │ DocumentCognition│  │ DocumentResearcher│  │ DocumentAnalyst      │  │
│  │ Base             │  │                   │  │                      │  │
│  │                  │  │                   │  │                      │  │
│  │ Strategy         │◄─┤ Get strategies    │  │ Store insights   ────┼──┤
│  │ repository       │  │ for questions     │  │                      │  │
│  │                  │  │                   │  │ Quality assessment   │  │
│  └────────┬─────────┘  └──────────┬────────┘  └──────────┬───────────┘  │
│           │                       │                       │              │
│           │ Success rates         │ Research plan         │ Analysis     │
│           │                       │                       │              │
│           ▼                       ▼                       ▼              │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │              ConsciousnessDocumentProcessor                       │  │
│  │                                                                   │  │
│  │  ┌────────────────┐  ┌────────────────┐  ┌───────────────────┐  │  │
│  │  │ SurfSense      │  │ Dionysus       │  │ Active Inference  │  │  │
│  │  │ Patterns       │  │ Consciousness  │  │ Engine            │  │  │
│  │  │                │  │                │  │                   │  │  │
│  │  │ - Hash         │  │ - Basins       │  │ - Prediction err  │  │  │
│  │  │ - Markdown     │  │ - ThoughtSeeds │  │ - Free energy     │  │  │
│  │  │ - Chunks       │  │ - Patterns     │  │ - Beliefs         │  │  │
│  │  └────────────────┘  └────────────────┘  └───────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

## Data Flow Through System

```
Input: PDF document "Introduction to BERT"
  │
  ├─► [Extract & Process]
  │     └─► Concepts: ["BERT", "transformer", "attention", "pretraining", ...]
  │         Content Hash: "a3f2b8..."
  │         Chunks: 15 chunks, ~1000 tokens each
  │         Summary: "BERT is a bidirectional transformer..."
  │
  ├─► [Generate Research Plan]
  │     └─► Prediction Errors:
  │           - "BERT": 0.85 (HIGH - curiosity trigger!)
  │           - "transformer": 0.2 (known)
  │           - "attention": 0.15 (known)
  │         Questions:
  │           1. "What are fundamental principles of BERT?" (high difficulty)
  │           2. "How does BERT differ from GPT?" (high difficulty)
  │           3. "What tasks is BERT optimized for?" (medium difficulty)
  │         Exploration Plan:
  │           Phase 1: Foundational understanding (academic papers)
  │           Phase 2: Relational mapping (survey papers)
  │           Phase 3: Applications (case studies)
  │
  ├─► [Consciousness Processing]
  │     └─► Basins created: 42
  │         ThoughtSeeds generated: 42
  │         Patterns learned:
  │           - "Bidirectional transformers for language understanding"
  │           - "Masked language model pretraining"
  │           - "Fine-tuning for downstream tasks"
  │
  ├─► [Analyze Results]
  │     └─► Quality Scores:
  │           - Concept extraction: 0.85
  │           - Chunking: 0.92
  │           - Consciousness integration: 0.88
  │           - Overall: 0.87
  │         Insights:
  │           - "High concept density suggests complex document"
  │           - "Strong pattern recognition (basin efficiency 0.88)"
  │           - "5 high-priority curiosity triggers detected"
  │         Meta-cognitive:
  │           - Learning effectiveness: 0.90
  │           - Curiosity alignment: 0.85
  │           - Pattern trend: "improving"
  │
  └─► [Finalize] (quality 0.87 > threshold 0.7, complete)
        └─► Output:
              Document: filename, hash, tags
              Extraction: 45 concepts, 15 chunks
              Consciousness: 42 basins, 42 thoughtseeds
              Research: 3 high-priority questions, 4-phase plan
              Quality: 0.87 overall, 4 insights, 2 recommendations
              Workflow: 1 iteration, completed in 2.3s
```

## Integration with External Systems

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          External Influences                             │
└─────────────────────────────────────────────────────────────────────────┘
           │          │          │          │          │          │
     SurfSense   ASI-GO-2    R-Zero    Context    OpenNotebook  Dionysus
           │          │          │      Engineer      │         Original
           │          │          │          │          │          │
           ▼          ▼          ▼          ▼          ▼          ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                  DocumentProcessingGraph (Integration Layer)              │
│                                                                           │
│  Hash, Markdown,  Cognition Base,  Question Gen,  Token       LangGraph  │
│  Chunks, Summary  Researcher,       Co-evolution  Efficiency  Workflow   │
│  (SurfSense)      Analyst,          (R-Zero)      (Context)  (Notebook)  │
│                   Refinement                      Engineer               │
│                   (ASI-GO-2)                                             │
│                                                                           │
│                         Active Inference, Basins, ThoughtSeeds           │
│                         (Dionysus Consciousness System)                  │
└──────────────────────────────────────────────────────────────────────────┘
```

## Active Inference Loop

```
┌─────────────────────────────────────────────────────────────────┐
│                  Active Inference Engine                         │
│                                                                  │
│  New Concept: "BERT"                                            │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────────────────────────────────────────┐           │
│  │ Hierarchical Beliefs (Prior Knowledge)          │           │
│  │                                                  │           │
│  │ Sensory: ["transformer", "attention", ...]      │           │
│  │ Perceptual: "Transformers use attention"        │           │
│  │ Cognitive: "Seq2seq models for NLP"             │           │
│  │ Meta: "System knows NLP basics"                 │           │
│  └───────────────────┬─────────────────────────────┘           │
│                      │                                          │
│                      ▼                                          │
│  ┌─────────────────────────────────────────────────┐           │
│  │ Prediction: BERT = "transformer variant"        │           │
│  │ Observation: BERT = "bidirectional pretrained"  │           │
│  │                                                  │           │
│  │ Prediction Error = 0.85 (HIGH!)                 │           │
│  └───────────────────┬─────────────────────────────┘           │
│                      │                                          │
│                      ▼                                          │
│  ┌─────────────────────────────────────────────────┐           │
│  │ Free Energy Minimization                        │           │
│  │                                                  │           │
│  │ Option 1: Update beliefs (assimilate)           │           │
│  │ Option 2: Gather info (explore) ← CHOSEN!       │           │
│  └───────────────────┬─────────────────────────────┘           │
│                      │                                          │
│                      ▼                                          │
│  ┌─────────────────────────────────────────────────┐           │
│  │ Curiosity Triggered                             │           │
│  │                                                  │           │
│  │ Generate Questions:                             │           │
│  │ - "What is BERT's architecture?"                │           │
│  │ - "How does BERT differ from GPT?"              │           │
│  │ - "What is bidirectional pretraining?"          │           │
│  └───────────────────┬─────────────────────────────┘           │
│                      │                                          │
│                      ▼                                          │
│  ┌─────────────────────────────────────────────────┐           │
│  │ Exploration Plan Created                        │           │
│  │ (Future: Web crawl, retrieve papers)            │           │
│  │                                                  │           │
│  │ Expected Free Energy Reduction: 0.75            │           │
│  └─────────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

## State Persistence

```
┌─────────────────────────────────────────────────────────────────┐
│                    DocumentCognitionBase                         │
│                                                                  │
│  File: document_cognition_knowledge.json                        │
│                                                                  │
│  {                                                               │
│    "extraction_strategies": [                                   │
│      {                                                           │
│        "name": "Content Hash Deduplication",                    │
│        "source": "SurfSense",                                   │
│        "success_rate": 1.0,                                     │
│        "use_case": "Prevent re-processing"                      │
│      },                                                          │
│      ...                                                         │
│    ],                                                            │
│    "learned_patterns": [                                        │
│      {                                                           │
│        "name": "BERT Understanding",                            │
│        "source": "Session Learning",                            │
│        "significance": 0.85,                                    │
│        "timestamp": "2025-10-01T10:30:00"                       │
│      }                                                           │
│    ]                                                             │
│  }                                                               │
│                                                                  │
│  Updates: After each high-significance insight (> 0.75)         │
│  Growth: Continuously learns from each document                 │
└─────────────────────────────────────────────────────────────────┘
```

## Summary

**6 Nodes** → **4 Components** → **6 External Sources** → **1 Clean System**

- **Daedalus**: Gateway (receive)
- **DocumentProcessingGraph**: LangGraph orchestration (6 nodes)
- **DocumentCognitionBase**: Strategy repository (learns)
- **DocumentResearcher**: Question generation (curiosity)
- **DocumentAnalyst**: Quality & insights (meta-cognitive)
- **ConsciousnessDocumentProcessor**: Hybrid processing (SurfSense + Dionysus)

All integrated via **LangGraph workflow** with **Active Inference** driving curiosity and pattern learning.

---

**Last Updated**: 2025-10-01
