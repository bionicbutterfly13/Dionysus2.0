# Data Model: Self-Teaching Consciousness Emulator (Flux)

**Branch**: `002-i-m-building`  
**Source Spec**: [spec.md](spec.md)  
**Context**: Flux (our implementation) integrates Dionysus and SurfSense capabilities while enforcing knowledge graph single-source-of-truth and context-engineering best practices.

## 1. Storage Overview
- **Knowledge Graph (Neo4j / AutoSchema KG)**: Canonical record for all long-lived entities and relationships.
- **Vector Store (Qdrant)**: Embedding indexes for `DocumentArtifact`, `ConceptNode`, `ThoughtSeedTrace`, `CuriosityMission`, `AutobiographicalJourney` snapshots.
- **Metadata Store (SQLite)**: Lightweight caches (processing queues, UI state) synced with Neo4j IDs.
- **Redis Streams**: Transient curiosity/mind-wandering signals (durable persistence mirrored into Neo4j per Redis decisions).

## 2. Core Entities

### 2.1 `UserProfile`
| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `user_id` | UUID | PK, immutable | Unique user identifier |
| `display_name` | string | required, <= 100 chars | Preferred name |
| `learning_style` | enum(`reflective`,`experimental`,`balanced`) | optional | Persona derived from usage |
| `curiosity_bias` | float | default 0.5 (0-1) | Baseline curiosity modulation |
| `created_at` | datetime | required | Profile creation timestamp |
| `updated_at` | datetime | required | Last profile update |

Relationships:
- `UserProfile` `OWNS` → `AutobiographicalJourney`
- `UserProfile` `CREATED` → `DocumentArtifact`
- `UserProfile` `INTERACTED_WITH` → `ThoughtSeedTrace`

### 2.2 `AutobiographicalJourney`
| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `journey_id` | UUID | PK | Story arc identifier |
| `user_id` | UUID | FK → `UserProfile.user_id` | Owner |
| `title` | string | required, <= 200 chars | User-labeled journey (e.g., project) |
| `summary` | text | optional | Generated synopsis |
| `timeline_vector_id` | UUID | optional | Qdrant vector reference |
| `created_at` | datetime | required | |
| `updated_at` | datetime | required | |

Relationships:
- `AutobiographicalJourney` `CONTAINS` → `EventNode`
- `AutobiographicalJourney` `LINKS_TO` → `ConceptNode` (persistent themes)

### 2.3 `EventNode`
Represents episodic memory entries (working-memory snapshots, event cache).

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `event_id` | UUID | PK |
| `journey_id` | UUID | FK |
| `source` | enum(`document_ingest`,`curiosity_replay`,`dreaming`,`user_note`,`conversation`) | required |
| `timestamp` | datetime | required | Event occurrence |
| `context_window` | json | optional | Token/segment metadata |
| `affective_state` | json | optional | Mosaic observation snapshot |
| `thoughtseed_trace_id` | UUID | optional | FK → `ThoughtSeedTrace.trace_id` |
| `curiosity_signal` | float | optional (0-1) | Latest curiosity energy |

Relationships:
- `EventNode` `REFERENCES` → `DocumentArtifact`
- `EventNode` `DERIVES_FROM` → `ThoughtSeedTrace`
- `EventNode` `PROMPTS` → `CuriosityMission`

### 2.4 `DocumentArtifact`
| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `document_id` | UUID | PK |
| `user_id` | UUID | FK |
| `title` | string | required |
| `source_type` | enum(`pdf`,`markdown`,`website`,`audio`,`video`) | required |
| `source_path` | string | required | Local absolute path/URI |
| `ingested_at` | datetime | required |
| `hash` | string | required, unique | SHA256 for dedupe |
| `embedding_id` | UUID | optional | Qdrant vector reference |
| `status` | enum(`pending`,`processing`,`processed`,`error`) | required |
| `mock_data` | boolean | default false | Flag for mock data transparency |

Relationships:
- `DocumentArtifact` `EMITS` → `ConceptNode`
- `DocumentArtifact` `UPDATES` → `ThoughtSeedTrace`
- `DocumentArtifact` `ATTACHES_TO` → `AutobiographicalJourney`

### 2.5 `ConceptNode`
| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `concept_id` | UUID | PK |
| `label` | string | required, unique per graph |
| `definition` | text | optional |
| `embedding_id` | UUID | optional |
| `salience_score` | float | default 0 | Aggregated salience |
| `last_observed_at` | datetime | required |

Relationships:
- `ConceptNode` `RELATES_TO` → `ConceptNode` (typed edges: `supports`, `contradicts`, `extends`)
- `ConceptNode` `ORIGINATED_FROM` → `DocumentArtifact`
- `ConceptNode` `FEATURES_IN` → `ThoughtSeedTrace`

### 2.6 `ThoughtSeedTrace`
| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `trace_id` | UUID | PK |
| `seed_type` | enum(`concept`,`strategy`,`question`,`affordance`) | required |
| `consciousness_state` | enum(`dormant`,`emerging`,`active`,`self-aware`,`meta-aware`,`dreaming`) | required |
| `affordance_quality` | float | 0-1 | Calculated via context-engineering |
| `trace_vector_id` | UUID | optional | Embedding |
| `created_at` | datetime | required |
| `updated_at` | datetime | required |
| `explanation` | text | optional | Narrative explanation for transparency |

Relationships:
- `ThoughtSeedTrace` `INFORMS` → `CuriosityMission`
- `ThoughtSeedTrace` `LINKS_TO` → `ConceptNode`
- `ThoughtSeedTrace` `LOGS` → `EvaluationFrame`

### 2.7 `CuriosityMission`
| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `mission_id` | UUID | PK |
| `trigger_event_id` | UUID | FK → `EventNode.event_id` |
| `status` | enum(`queued`,`searching`,`retrieved`,`integrated`,`dismissed`) | required |
| `query` | text | required | Generated query string |
| `retrieved_sources` | json | default [] | Candidate source metadata |
| `trust_score` | float | default 0 | Aggregated credibility |
| `curiosity_vector_id` | UUID | optional |
| `replay_priority` | enum(`immediate`,`scheduled`,`nightly`) | default `scheduled` |
| `created_at` | datetime | required |
| `completed_at` | datetime | optional |

Relationships:
- `CuriosityMission` `RESULTED_IN` → `DocumentArtifact`
- `CuriosityMission` `EVALUATED_BY` → `EvaluationFrame`

### 2.8 `EvaluationFrame`
Captures evaluative feedback required by constitution (what’s good/broken/etc.).

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `frame_id` | UUID | PK |
| `context_type` | enum(`ingestion`,`curiosity`,`reasoning`,`visualization`) | required |
| `context_id` | UUID | required | References entity under evaluation |
| `whats_good` | text | required |
| `whats_broken` | text | required |
| `works_but_shouldnt` | text | required |
| `pretends_but_doesnt` | text | required |
| `created_at` | datetime | required |

Relationships:
- `EvaluationFrame` `ANNOTATES` → [context entity]

### 2.9 `VisualizationState`
Stores front-end state derived from knowledge graph.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `viz_id` | UUID | PK |
| `user_id` | UUID | FK |
| `view_type` | enum(`graph`,`card_stack`,`timeline`,`dashboard`) | required |
| `active_nodes` | json | required |
| `layout_state` | json | optional |
| `updated_at` | datetime | required |

## 3. Relationships Graph Summary
- `UserProfile` `OWNS` `AutobiographicalJourney`
- `AutobiographicalJourney` `CONTAINS` `EventNode`
- `EventNode` `REFERENCES` `DocumentArtifact`
- `DocumentArtifact` `EMITS` `ConceptNode`
- `ConceptNode` `FEATURES_IN` `ThoughtSeedTrace`
- `ThoughtSeedTrace` `INFORMS` `CuriosityMission`
- `CuriosityMission` `RESULTED_IN` `DocumentArtifact` (new or updated)
- `ThoughtSeedTrace` `LOGS` `EvaluationFrame`
- `EvaluationFrame` `ANNOTATES` any entity (polymorphic relationship)

## 4. State Transitions
### 4.1 Document Processing
1. `DocumentArtifact.status = pending`
2. Ingestion pipeline processes through ThoughtSeed basins → `processing`
3. Concepts, thought seeds, event nodes created; evaluations logged
4. On success → `processed`; on failure → `error` (EvaluationFrame required)

### 4.2 Curiosity Mission
1. Triggered by `EventNode` with knowledge gap
2. Mission queued (`queued`) → proactive search (`searching`)
3. Candidate sources scored → `retrieved`
4. User/system integration → `integrated` or `dismissed`
5. Replay metadata stored for nightly runs

### 4.3 Evaluation Feedback
- Every processing cycle must instantiate `EvaluationFrame` capturing four constitutional questions.

## 5. Validation Rules
- All UUIDs must be globally unique; enforce through database constraints.
- `trust_score` derived from source provenance; fail ingestion if below threshold (configurable).
- `mock_data` true requires `EvaluationFrame` explanation and is disallowed in production readiness checks.
- `consciousness_state` enumeration must include `dreaming` (nightly replays).
- `affective_state` uses Mosaic schema: `{senses, actions, emotions, impulses, cognitions}` each 0-1 scale.
- `curiosity_signal` decay modeled per NEMORI guidance; nightly job reduces values unless reinforced.

## 6. External Integrations Mapping
| Requirement | Entity / Field | Notes |
|-------------|----------------|-------|
| ThoughtSeed attractor basins | `ThoughtSeedTrace`, `EventNode` | Store attractor IDs for visualization |
| Meta Tree-of-Thought | `ThoughtSeedTrace`, `CuriosityMission` | Hypothesis trees serialized in `explanation` / `retrieved_sources` |
| Mosaic observation | `EventNode.affective_state` | Schema enforced per Mosaic Systems LLC model |
| Archetypal reconsolidation | `ConceptNode` + `AutobiographicalJourney` | Archetype nodes linked to journeys |
| Redis replay | `CuriosityMission.replay_priority` + mirrored EventNodes | Replay scheduler uses these flags |

## 7. Data Residency & Privacy
- All `source_path` references remain local unless user explicitly configures remote storage.
- No external embeddings or tokens transmitted without opt-in.
- Sensitive fields (`source_path`, `affective_state`) encrypted at rest when persisted outside memory.

## 8. Mock Data Handling
- `mock_data = true` on `DocumentArtifact` or derived entities requires:
  - `EvaluationFrame` entry explaining mock usage
  - Flag propagation to VisualizationState to display disclosure banner
  - Exclusion from production readiness checks until real data validated

## 9. Service Layer Architecture (Implementation Requirements)

### 9.1 Repository Pattern Specifications
Based on implementation testing, the following repository interfaces are required:

#### **GraphRepository** (Neo4j Integration)
```python
class GraphRepository:
    """Neo4j repository for all graph operations"""

    async def create_node(self, node_type: str, properties: Dict[str, Any]) -> str
    async def create_relationship(self, from_id: str, to_id: str, rel_type: str, properties: Dict[str, Any]) -> str
    async def find_by_id(self, node_id: str) -> Optional[Dict[str, Any]]
    async def find_by_properties(self, node_type: str, properties: Dict[str, Any]) -> List[Dict[str, Any]]
    async def update_node(self, node_id: str, properties: Dict[str, Any]) -> bool
    async def delete_node(self, node_id: str) -> bool

    # Constitutional compliance queries
    async def find_with_evaluation_frames(self, context_type: str) -> List[Dict[str, Any]]
    async def find_mock_data_entities(self) -> List[Dict[str, Any]]
```

#### **VectorRepository** (Qdrant Integration)
```python
class VectorRepository:
    """Qdrant repository for vector operations"""

    async def store_embedding(self, doc_id: str, embedding: List[float], metadata: Dict[str, Any]) -> str
    async def search_similar(self, query_embedding: List[float], limit: int = 10) -> List[Dict[str, Any]]
    async def delete_embedding(self, embedding_id: str) -> bool
    async def update_metadata(self, embedding_id: str, metadata: Dict[str, Any]) -> bool
```

#### **CacheRepository** (Redis Integration)
```python
class CacheRepository:
    """Redis repository for caching and streams"""

    async def set_cache(self, key: str, value: Any, ttl: int = 3600) -> bool
    async def get_cache(self, key: str) -> Optional[Any]
    async def publish_curiosity_signal(self, mission_data: Dict[str, Any]) -> bool
    async def subscribe_curiosity_stream(self) -> AsyncIterator[Dict[str, Any]]
    async def store_replay_schedule(self, schedule: Dict[str, Any]) -> bool
```

### 9.2 Service Layer Specifications
Required services identified from implementation gaps:

#### **DocumentIngestionService**
```python
class DocumentIngestionService:
    """Handles document upload, processing, and ThoughtSeed integration"""

    async def ingest_document(self, file: UploadFile, metadata: Dict[str, Any]) -> DocumentArtifact
    async def process_with_thoughtseed(self, doc_id: str) -> List[ThoughtSeedTrace]
    async def extract_concepts(self, doc_id: str) -> List[ConceptNode]
    async def create_evaluation_frame(self, doc_id: str, context: str) -> EvaluationFrame
```

#### **CuriosityEngineService**
```python
class CuriosityEngineService:
    """Manages curiosity missions, trust scoring, and replay scheduling"""

    async def create_mission(self, trigger_event_id: str, query: str) -> CuriosityMission
    async def search_sources(self, query: str) -> List[Dict[str, Any]]
    async def calculate_trust_score(self, sources: List[Dict[str, Any]]) -> float
    async def schedule_replay(self, mission_id: str, priority: str) -> bool
```

#### **ThoughtSeedPipelineService**
```python
class ThoughtSeedPipelineService:
    """Integrates with existing ASI-Arch ThoughtSeed modules"""

    async def activate_attractor_basin(self, content: str) -> Dict[str, Any]
    async def detect_consciousness_level(self, traces: List[ThoughtSeedTrace]) -> str
    async def process_mosaic_observation(self, event: EventNode) -> Dict[str, float]
    async def generate_evaluation_insights(self, context: str) -> EvaluationFrame
```

### 9.3 Database Connection Management
Required connection pooling and configuration:

#### **Neo4j Connection Specifications**
- **Driver**: `neo4j==5.15.0`
- **Connection Pool**: 10-20 connections
- **Authentication**: Local username/password
- **Database**: `flux` (dedicated database)
- **Constraints**: UUID uniqueness, relationship validation
- **Indexes**: Node types, frequently queried properties

#### **Redis Connection Specifications**
- **Client**: `aioredis==2.0.1`
- **Connection Pool**: 20 connections
- **Persistence**: AOF + RDB for durability
- **Streams**: `curiosity:missions`, `consciousness:events`
- **TTL**: Session data (1 hour), replay schedules (24 hours)

#### **Qdrant Connection Specifications**
- **Client**: `qdrant-client>=1.7.1`
- **Collection**: `flux_embeddings`
- **Vector Size**: 384 (sentence-transformers/all-MiniLM-L6-v2)
- **Distance**: Cosine similarity
- **Indexing**: HNSW with ef_construct=128

### 9.4 Configuration Integration
Service configuration requirements for `configs/flux.yaml`:

```yaml
# Database connection pools
database:
  neo4j:
    max_pool_size: 15
    connection_timeout: 10
    retry_attempts: 3

  redis:
    max_connections: 20
    retry_on_timeout: true
    socket_keepalive: true

  qdrant:
    timeout: 30
    grpc_port: 6334
    prefer_grpc: true

# Service-specific settings
services:
  document_ingestion:
    max_file_size: 50_000_000  # 50MB
    supported_formats: ["pdf", "txt", "md", "docx"]
    chunk_size: 1000

  curiosity_engine:
    max_concurrent_missions: 5
    trust_score_threshold: 0.3
    replay_batch_size: 10

  thoughtseed_pipeline:
    consciousness_threshold: 0.6
    basin_activation_timeout: 30
    evaluation_mandatory: true
```

## 10. Implementation Validation Requirements

### 10.1 Database Schema Validation
- All entities must have corresponding Neo4j node types
- Relationships must enforce referential integrity
- Constitutional compliance fields must be indexed
- Mock data entities must be clearly flagged

### 10.2 Service Integration Testing
- Repository layer must handle connection failures gracefully
- Services must implement circuit breaker patterns
- All operations must generate evaluation frames
- Mock data transparency must be enforced

### 10.3 Performance Requirements
- Document ingestion: < 5 seconds per document
- Graph queries: < 100ms for simple lookups
- Vector searches: < 500ms for similarity queries
- Curiosity mission creation: < 2 seconds

---
**Next**: Use this enhanced data model with service specifications to implement the complete repository and service layers. All implementations must maintain Neo4j as the authoritative source of truth and synchronize embeddings/Redis mirrors accordingly.
