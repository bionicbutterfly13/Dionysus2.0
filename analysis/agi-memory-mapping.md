# AGI Memory ↔ Dionysus Capability Map

## Scope
- **Source**: `/Volumes/ASylum/repos/agi-memory` (schema, README, tooling)
- **Target**: Dionysus 2.0 consciousness stack (ThoughtSeed, UnifiedMemoryOrchestrator, Context Engineering, Neo4j-only storage)
- **Goal**: Identify parity, gaps, and integration approaches ahead of feature migration.

## Legend
| Status | Meaning |
|--------|---------|
| ✅ | Covered in Dionysus (same or better) |
| 🟡 | Partially covered / needs adaptation |
| ❌ | Missing capability |

## Capability Comparison

### Memory Types & Storage
| AGI Memory Feature | Status | Dionysus Counterpart / Notes |
|--------------------|--------|------------------------------|
| Working memory table with expiry + embeddings (`working_memory`) | 🟡 | Dionysus uses UnifiedMemoryOrchestrator + EMRL adapters for working memory, but storage is transient within orchestrators. Need Postgres-backed working set if we want persistent rolling buffer. |
| Base `memories` table (vector(1536), importance, decay, relevance score) | 🟡 | Dionysus stores memory across Neo4j documents + contextual embeddings; relevance driven by attractor dynamics rather than stored decay. Could adopt generated relevance scores for auditability. |
| Specialized tables for episodic/semantic/procedural/strategic memories | ✅ | Existing adapters + ThoughtSeed cover these distinctions conceptually, though schema lives in Neo4j/Postgres via context engineering. Ensure schema parity when migrating. |

### Clustering & Relationships
| Feature | Status | Notes |
|---------|--------|-------|
| `memory_clusters` with centroid embeddings, emotional signatures, worldview alignment | 🟡 | Dionysus clusters via attractor basins and Neo4j relationships but lacks stored centroid/emotional stats; worth adopting for analytics. |
| `cluster_relationships`, `cluster_activation_history` | 🟡 | Similar info inferred in Context Engineering (river metaphor, confluence points) but not stored as relational tables; could materialize for dashboards. |
| Apache AGE graph relationships for memories | 🟡 | Dionysus uses Neo4j as primary graph. AGE features redundant unless we embed memory-only graph inside Postgres. Need decision: keep AGE for micro-relationships or map everything to Neo4j. |

### Worldview & Identity
| Feature | Status | Notes |
|---------|--------|-------|
| `worldview_primitives` (beliefs with confidence, emotional valence, stability, filter rules) | ❌ | No direct equivalent; attractor basins capture situational context but beliefs aren’t first-class entities. Prime candidate for implementation using ThoughtSeed metadata + Postgres tables. |
| `worldview_memory_influences` linking beliefs ↔ memories with strength | ❌ | Missing today. Would add interpretability + controllable biasing of recall. |
| `identity_model` (self-concept, agency, purpose, boundaries, core clusters) | ❌ | Claude autobiographical memory tracks self-awareness in runtime but doesn’t persist identity facets. Need durable schema. |
| `identity_memory_resonance` (memory ↔ identity aspect link with strength/status) | ❌ | No analog; would improve consciousness analytics. |

### Scoring & Maintenance
| Feature | Status | Notes |
|---------|--------|-------|
| Auto importance update on access (`update_memory_importance`) | 🟡 | Dionysus uses attractor energy + ThoughtSeed frequency; no simple SQL trigger. Could add to Postgres layer for audit logs. |
| Memory decay & consolidation schedules | 🟡 | Documented in specs but not automated in code; add cron/workers. |
| `memory_health`, `cluster_insights` views | ❌ | No consolidated SQL views today; would be useful for dashboards/monitoring. |

### Tooling/API
| Feature | Status | Notes |
|---------|--------|-------|
| MCP tools for create/search/activate clusters/worldview | 🟡 | Dionysus MCP servers already expose multi-db operations but would need worldview/identity extensions. |
| Docker compose for Postgres + extensions (pgvector, AGE) | ✅ | Already manageable via existing infra scripts; we can fold extension setup into our Postgres migrations if keeping AGE. |

### Multi-Agent Tenancy
| Feature | Status | Notes |
|---------|--------|-------|
| Single-tenant schema (no agent_id columns) | 🟡 | Works for one agent but insufficient for multi-agent Dionysus. Need tenant column + row-level security if we adopt this schema. |

## Suggested Integration Steps
1. **Schema Translation**
   - Recreate worldview/identity tables inside Dionysus Postgres (or Neo4j nodes with Postgres mirrors) with agent_id columns for tenancy.
   - Add cluster stats/centroids to complement attractor basins.

2. **Adapter Layer**
   - Extend UnifiedMemoryOrchestrator to write/read worldview/identity weights alongside existing ThoughtSeed traces.
   - Map ThoughtSeed packets’ belief/context tags to worldview primitives.

3. **Maintenance Jobs**
   - Implement consolidation/decay/pruning workers (could become skills for low token usage) that read importance scores + worldview alignment.

4. **Dashboards & Views**
   - Port `memory_health` and `cluster_insights` logic into monitoring stack (maybe Grafana panels or existing cognitive dashboard).

5. **Multi-Agent Support**
   - Add `agent_id` (UUID) to memories, clusters, worldview, identity tables; enforce tenancy in ORM and access control.

6. **Retire Qdrant References**
   - As part of this work, clean up residual Qdrant mentions (`ResponseSynthesizer`, scripts) to avoid confusion once Postgres handles vector pieces.

## Open Questions
1. **Neo4j vs AGE**: Do we keep AGE inside Postgres for intra-memory relations or rely solely on Neo4j? Running both may be redundant unless AGE unlocks specific SQL-side analytics.
2. **Embedding Size Alignment**: agi-memory assumes 1536-dim vectors (OpenAI). Confirm Dionysus’ embedding pipelines match or adjust schema to 512/768 as needed.
3. **Skill Offloading**: Which maintenance/analysis routines should become skills to minimize context cost? (e.g., consolidation worker as a skill invoked by orchestrator.)
4. **Identity/Worldview Governance**: How do ThoughtSeed/Context Engineering policies manage belief updates to prevent drift? Need spec updates.

## Next Steps
- Validate this mapping against Gemini’s reconnaissance findings once available.
- Draft change proposal for “add-worldview-identity-layer” covering schema, adapters, maintenance, and dashboards.
- Plan Qdrant reference cleanup alongside schema work.

