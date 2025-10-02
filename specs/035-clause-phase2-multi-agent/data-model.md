# Data Model - CLAUSE Phase 2 Multi-Agent

**Date**: 2025-10-02
**Feature**: CLAUSE Phase 2 - Path Navigator & Context Curator
**Status**: ✅ Complete - All models defined

## Overview

This document defines Pydantic V2 models for CLAUSE Phase 2 agents, extending Phase 1's SubgraphArchitect with PathNavigator and ContextCurator. Models support three new API endpoints (/navigate, /curate, /coordinate) and four intelligence integrations (ThoughtSeeds, Curiosity, Causal, Provenance).

## Model Categories

1. **Path Navigator Models** - Navigation requests/responses
2. **Context Curator Models** - Curation requests/responses with provenance
3. **LC-MAPPO Coordinator Models** - Multi-agent orchestration
4. **Intelligence Integration Models** - ThoughtSeed, Curiosity, Causal, Provenance
5. **Shared Models** - Common types used across agents

---

## 1. Path Navigator Models

### PathNavigationRequest
**Purpose**: Input for POST /api/clause/navigate

```python
from pydantic import BaseModel, Field
from typing import Optional

class PathNavigationRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Natural language query")
    start_node: str = Field(..., description="Starting concept node ID")
    step_budget: int = Field(default=10, ge=1, le=20, description="Maximum navigation steps")
    enable_thoughtseeds: bool = Field(default=True, description="Generate ThoughtSeeds during exploration")
    enable_curiosity: bool = Field(default=True, description="Trigger curiosity agents on prediction errors")
    enable_causal: bool = Field(default=True, description="Use causal reasoning for path selection")
    curiosity_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Prediction error threshold for curiosity")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What causes climate change?",
                "start_node": "climate_change",
                "step_budget": 10,
                "enable_thoughtseeds": True,
                "enable_curiosity": True,
                "enable_causal": True,
                "curiosity_threshold": 0.7
            }
        }
```

### PathStep
**Purpose**: Single step in navigation path

```python
class PathStep(BaseModel):
    step: int = Field(..., ge=1, description="Step number (1-indexed)")
    from_node: str = Field(..., description="Source node")
    to_node: str = Field(..., description="Target node")
    relation: str = Field(..., description="Edge relation type")
    action: str = Field(..., pattern="^(CONTINUE|BACKTRACK|STOP)$", description="Navigator action")
    causal_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Causal intervention score")
    thoughtseed_id: Optional[str] = Field(None, description="Generated ThoughtSeed ID")
```

### PathNavigationResponse
**Purpose**: Output from POST /api/clause/navigate

```python
from typing import List, Dict, Any

class PathNavigationResponse(BaseModel):
    path: Dict[str, Any] = Field(..., description="Navigation path details")
    metadata: Dict[str, Any] = Field(..., description="Path metadata (budgets, triggers)")
    performance: Dict[str, float] = Field(..., description="Latency breakdown")

    class Config:
        json_schema_extra = {
            "example": {
                "path": {
                    "nodes": ["climate_change", "greenhouse_gases", "CO2_emissions"],
                    "edges": [
                        {"from": "climate_change", "relation": "caused_by", "to": "greenhouse_gases"}
                    ],
                    "steps": [
                        {"step": 1, "from": "climate_change", "to": "greenhouse_gases", "action": "CONTINUE", "causal_score": 0.85}
                    ]
                },
                "metadata": {
                    "budget_used": 3,
                    "budget_total": 10,
                    "final_action": "STOP",
                    "thoughtseeds_generated": 12,
                    "curiosity_triggers_spawned": 2
                },
                "performance": {
                    "latency_ms": 145,
                    "thoughtseed_gen_ms": 23,
                    "causal_pred_ms": 87
                }
            }
        }
```

---

## 2. Context Curator Models

### ContextCurationRequest
**Purpose**: Input for POST /api/clause/curate

```python
class ContextCurationRequest(BaseModel):
    evidence_pool: List[str] = Field(..., min_length=1, description="Pool of evidence snippets to curate")
    token_budget: int = Field(default=2048, ge=100, le=8192, description="Maximum tokens for selected evidence")
    enable_provenance: bool = Field(default=True, description="Add full provenance metadata")
    lambda_tok: float = Field(default=0.01, ge=0.0, le=1.0, description="Token cost multiplier for shaped utility")

    class Config:
        json_schema_extra = {
            "example": {
                "evidence_pool": [
                    "Greenhouse gases trap heat in the atmosphere...",
                    "CO2 is the primary greenhouse gas from human activity..."
                ],
                "token_budget": 2048,
                "enable_provenance": True,
                "lambda_tok": 0.01
            }
        }
```

### ProvenanceMetadata
**Purpose**: Full provenance tracking per Spec 032

```python
from datetime import datetime

class TrustSignals(BaseModel):
    reputation_score: float = Field(..., ge=0.0, le=1.0, description="Source reputation (0-1)")
    recency_score: float = Field(..., ge=0.0, le=1.0, description="Information recency (0-1)")
    semantic_consistency: float = Field(..., ge=0.0, le=1.0, description="Consistency with query (0-1)")

class ProvenanceMetadata(BaseModel):
    source_uri: str = Field(..., description="Neo4j URI or document source")
    extraction_timestamp: datetime = Field(..., description="When evidence was extracted")
    extractor_identity: str = Field(..., description="Agent/service that extracted evidence")
    supporting_evidence: str = Field(..., max_length=200, description="Evidence snippet (200 chars)")
    verification_status: str = Field(..., pattern="^(verified|pending_review|unverified)$")
    corroboration_count: int = Field(..., ge=0, description="Number of corroborating sources")
    trust_signals: TrustSignals = Field(..., description="Trust signal scores")

    class Config:
        json_schema_extra = {
            "example": {
                "source_uri": "neo4j://concept/greenhouse_gases",
                "extraction_timestamp": "2025-10-02T10:30:15Z",
                "extractor_identity": "ContextCurator-v2.0",
                "supporting_evidence": "Greenhouse gases trap heat...",
                "verification_status": "verified",
                "corroboration_count": 5,
                "trust_signals": {
                    "reputation_score": 0.95,
                    "recency_score": 0.88,
                    "semantic_consistency": 0.91
                }
            }
        }
```

### SelectedEvidence
**Purpose**: Evidence snippet with provenance

```python
class SelectedEvidence(BaseModel):
    text: str = Field(..., description="Evidence text")
    tokens: int = Field(..., ge=1, description="Token count (tiktoken)")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    shaped_utility: float = Field(..., description="Score minus token cost")
    provenance: ProvenanceMetadata = Field(..., description="Full provenance metadata")
```

### ContextCurationResponse
**Purpose**: Output from POST /api/clause/curate

```python
class ContextCurationResponse(BaseModel):
    selected_evidence: List[SelectedEvidence] = Field(..., description="Curated evidence with provenance")
    metadata: Dict[str, Any] = Field(..., description="Curation metadata")
    performance: Dict[str, float] = Field(..., description="Latency breakdown")

    class Config:
        json_schema_extra = {
            "example": {
                "selected_evidence": [
                    {
                        "text": "Greenhouse gases trap heat...",
                        "tokens": 156,
                        "score": 0.92,
                        "shaped_utility": 0.904,
                        "provenance": {"source_uri": "neo4j://concept/greenhouse_gases", "...": "..."}
                    }
                ],
                "metadata": {
                    "tokens_used": 428,
                    "tokens_total": 2048,
                    "learned_stop_triggered": True
                },
                "performance": {
                    "latency_ms": 78,
                    "provenance_overhead_ms": 12
                }
            }
        }
```

---

## 3. LC-MAPPO Coordinator Models

### CoordinationRequest
**Purpose**: Input for POST /api/clause/coordinate

```python
class BudgetAllocation(BaseModel):
    edge_budget: int = Field(default=50, ge=10, le=200, description="Subgraph edge budget")
    step_budget: int = Field(default=10, ge=1, le=20, description="Path navigation step budget")
    token_budget: int = Field(default=2048, ge=100, le=8192, description="Evidence curation token budget")

class LambdaParameters(BaseModel):
    edge: float = Field(default=0.01, ge=0.0, le=1.0, description="Edge cost multiplier")
    latency: float = Field(default=0.01, ge=0.0, le=1.0, description="Latency cost multiplier")
    token: float = Field(default=0.01, ge=0.0, le=1.0, description="Token cost multiplier")

class CoordinationRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Natural language query")
    budgets: BudgetAllocation = Field(..., description="Budget allocation across agents")
    lambdas: LambdaParameters = Field(default_factory=LambdaParameters, description="Cost multipliers")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What causes climate change?",
                "budgets": {
                    "edge_budget": 50,
                    "step_budget": 10,
                    "token_budget": 2048
                },
                "lambdas": {
                    "edge": 0.01,
                    "latency": 0.01,
                    "token": 0.01
                }
            }
        }
```

### AgentHandoff
**Purpose**: Track agent execution in coordination

```python
class AgentHandoff(BaseModel):
    step: int = Field(..., ge=1, description="Execution order (1=Architect, 2=Navigator, 3=Curator)")
    agent: str = Field(..., pattern="^(SubgraphArchitect|PathNavigator|ContextCurator)$")
    action: str = Field(..., description="Action taken by agent")
    budget_used: Dict[str, int] = Field(..., description="Budgets consumed")
    latency_ms: float = Field(..., ge=0.0, description="Agent execution latency")
```

### CoordinationResponse
**Purpose**: Output from POST /api/clause/coordinate

```python
class CoordinationResponse(BaseModel):
    result: Dict[str, Any] = Field(..., description="Combined results from all agents")
    agent_handoffs: List[AgentHandoff] = Field(..., description="Agent execution timeline")
    conflicts_detected: int = Field(..., ge=0, description="Number of write conflicts")
    conflicts_resolved: int = Field(..., ge=0, description="Number of conflicts resolved")
    performance: Dict[str, float] = Field(..., description="Total latency breakdown")

    class Config:
        json_schema_extra = {
            "example": {
                "result": {
                    "subgraph": {"nodes": [...], "edges": [...]},
                    "path": {"nodes": [...], "edges": [...]},
                    "evidence": [{"text": "...", "provenance": {...}}]
                },
                "agent_handoffs": [
                    {"step": 1, "agent": "SubgraphArchitect", "action": "built_subgraph", "budget_used": {"edges": 35}, "latency_ms": 287}
                ],
                "conflicts_detected": 0,
                "conflicts_resolved": 0,
                "performance": {
                    "total_latency_ms": 542,
                    "architect_ms": 287,
                    "navigator_ms": 145,
                    "curator_ms": 78
                }
            }
        }
```

---

## 4. Intelligence Integration Models

### ThoughtSeed Models (Spec 028)

```python
class BasinContext(BaseModel):
    strength: float = Field(..., ge=1.0, le=2.0, description="Basin strength from Phase 1")
    activation_count: int = Field(..., ge=0, description="Activation count")
    co_occurring: Dict[str, int] = Field(..., description="Co-occurrence counts")

class ThoughtSeed(BaseModel):
    id: str = Field(..., description="Unique ThoughtSeed ID")
    concept: str = Field(..., description="Concept node")
    source_doc: str = Field(..., description="Source document/query")
    basin_context: BasinContext = Field(..., description="Basin context from Phase 1")
    similarity_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    linked_documents: List[str] = Field(default_factory=list, description="Cross-document links")
    created_at: datetime = Field(default_factory=datetime.now)
```

### Curiosity Models (Spec 029)

```python
class CuriosityTrigger(BaseModel):
    trigger_type: str = Field(..., pattern="^prediction_error$", description="Trigger type")
    concept: str = Field(..., description="Concept that triggered curiosity")
    error_magnitude: float = Field(..., ge=0.0, le=1.0, description="Prediction error magnitude")
    timestamp: datetime = Field(default_factory=datetime.now)
    investigation_status: str = Field(default="queued", pattern="^(queued|investigating|completed)$")
```

### Causal Models (Spec 033)

```python
class CausalIntervention(BaseModel):
    intervention_node: str = Field(..., description="Node to intervene on")
    target_node: str = Field(..., description="Target outcome node")
    intervention_score: float = Field(..., ge=0.0, le=1.0, description="P(target | do(intervention))")
    computation_time_ms: float = Field(..., ge=0.0, description="Inference latency")
```

---

## 5. Shared Models

### StateEncoding
**Purpose**: Navigator state representation

```python
import numpy as np
from typing import Union

class StateEncoding(BaseModel):
    query_embedding: List[float] = Field(..., min_length=384, max_length=384, description="Query embedding (384-dim)")
    node_embedding: List[float] = Field(..., min_length=384, max_length=384, description="Current node embedding")
    node_degree: int = Field(..., ge=0, description="Node degree in graph")
    basin_strength: float = Field(..., ge=1.0, le=2.0, description="Basin strength")
    neighborhood_mean: List[float] = Field(..., min_length=384, max_length=384, description="Neighborhood mean embedding")
    budget_remaining: float = Field(..., ge=0.0, le=1.0, description="Normalized budget remaining")

    def to_numpy(self) -> np.ndarray:
        """Convert to NumPy array for ML processing"""
        return np.concatenate([
            self.query_embedding,
            self.node_embedding,
            [self.node_degree, self.basin_strength],
            self.neighborhood_mean,
            [self.budget_remaining]
        ])

    class Config:
        arbitrary_types_allowed = True
```

### BudgetUsage
**Purpose**: Track budget consumption across agents

```python
class BudgetUsage(BaseModel):
    edge_used: int = Field(default=0, ge=0, description="Edges used by Architect")
    step_used: int = Field(default=0, ge=0, description="Steps used by Navigator")
    token_used: int = Field(default=0, ge=0, description="Tokens used by Curator")

    edge_total: int = Field(..., ge=1, description="Total edge budget")
    step_total: int = Field(..., ge=1, description="Total step budget")
    token_total: int = Field(..., ge=1, description="Total token budget")

    @property
    def edge_remaining(self) -> int:
        return max(0, self.edge_total - self.edge_used)

    @property
    def step_remaining(self) -> int:
        return max(0, self.step_total - self.step_used)

    @property
    def token_remaining(self) -> int:
        return max(0, self.token_total - self.token_used)
```

---

## Model Relationships

```
CoordinationRequest
├── BudgetAllocation (edge, step, token budgets)
└── LambdaParameters (cost multipliers)
         ↓
    Coordinator orchestrates 3 agents:
         ↓
    ┌────────────────────────────────────┐
    │ 1. SubgraphArchitect (Phase 1)     │
    │    - Uses edge_budget              │
    │    - Returns subgraph              │
    └────────────────────────────────────┘
         ↓
    ┌────────────────────────────────────┐
    │ 2. PathNavigator (Phase 2)         │
    │    - Uses step_budget              │
    │    - Generates ThoughtSeeds        │
    │    - Triggers Curiosity            │
    │    - Uses CausalReasoning          │
    │    - Returns path                  │
    └────────────────────────────────────┘
         ↓
    ┌────────────────────────────────────┐
    │ 3. ContextCurator (Phase 2)        │
    │    - Uses token_budget             │
    │    - Adds Provenance               │
    │    - Returns selected evidence     │
    └────────────────────────────────────┘
         ↓
    CoordinationResponse
    ├── Combined results
    ├── AgentHandoffs (timeline)
    └── Performance metrics
```

---

## Validation Rules

### Request Validation
- **PathNavigationRequest**: `step_budget` must be 1-20
- **ContextCurationRequest**: `evidence_pool` must have ≥1 snippet
- **CoordinationRequest**: All budgets must be positive

### Response Validation
- **PathNavigationResponse**: `path.steps` length ≤ request `step_budget`
- **ContextCurationResponse**: `metadata.tokens_used` ≤ request `token_budget`
- **CoordinationResponse**: `agent_handoffs` must have exactly 3 entries

### Provenance Validation
- `verification_status` must be one of: verified, pending_review, unverified
- `trust_signals` scores must be in [0.0, 1.0]
- `supporting_evidence` max length 200 chars

---

## Usage Examples

### Path Navigation
```python
request = PathNavigationRequest(
    query="What causes climate change?",
    start_node="climate_change",
    step_budget=10,
    enable_thoughtseeds=True,
    enable_curiosity=True,
    enable_causal=True
)

response = path_navigator.navigate(request)
assert len(response.path["steps"]) <= request.step_budget
```

### Context Curation
```python
request = ContextCurationRequest(
    evidence_pool=["Greenhouse gases trap heat...", "CO2 is the primary..."],
    token_budget=2048,
    enable_provenance=True
)

response = context_curator.curate(request)
assert response.metadata["tokens_used"] <= request.token_budget
```

### Full Coordination
```python
request = CoordinationRequest(
    query="What causes climate change?",
    budgets=BudgetAllocation(edge_budget=50, step_budget=10, token_budget=2048),
    lambdas=LambdaParameters(edge=0.01, latency=0.01, token=0.01)
)

response = coordinator.coordinate(request)
assert len(response.agent_handoffs) == 3  # Architect, Navigator, Curator
```

---

## Neo4j Schema Extensions

### Provenance Nodes
```cypher
// Create Provenance node type
CREATE CONSTRAINT provenance_unique IF NOT EXISTS
FOR (p:Provenance) REQUIRE p.id IS UNIQUE;

// Store provenance with Evidence
MERGE (e:Evidence {text: $text})
MERGE (p:Provenance {id: $provenance_id})
SET p.source_uri = $source_uri,
    p.extraction_timestamp = $timestamp,
    p.extractor_identity = $extractor,
    p.verification_status = $status,
    p.corroboration_count = $count,
    p.trust_signals = $trust_signals
MERGE (e)-[:HAS_PROVENANCE]->(p)
```

### ThoughtSeed Nodes
```cypher
// Create ThoughtSeed node type
CREATE CONSTRAINT thoughtseed_unique IF NOT EXISTS
FOR (ts:ThoughtSeed) REQUIRE ts.id IS UNIQUE;

// Store ThoughtSeeds with basin context
MERGE (ts:ThoughtSeed {id: $thoughtseed_id})
SET ts.concept = $concept,
    ts.source_doc = $source_doc,
    ts.basin_context = $basin_context,
    ts.created_at = datetime()
```

---

## Status

**Data Model**: ✅ Complete
**Total Models**: 22 Pydantic models
**Neo4j Extensions**: 2 new node types (Provenance, ThoughtSeed)
**Validation Rules**: Comprehensive request/response validation

**Next Phase**: Generate API contracts from data models

---
*Data model complete: 2025-10-02*
