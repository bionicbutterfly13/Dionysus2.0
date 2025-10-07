# Task Breakdown: Daedalus Graph Hardening

## Phase Alignment
- **Phase M1 – Facade Delivery**
- **Phase M2 – Backend Cutover**
- **Phase M3 – Compliance Enforcement**

## Detailed Tasks

### T1 – Graph Channel Scaffold (M1)
- Create `DaedalusGraphChannel` interface within `daedalus-gateway`.
- Implement LangGraph River flow definitions for schema, read, write, and health ops.
- Document usage patterns for Dionysus services.

### T2 – Channel Telemetry & Resilience (M1)
- Add structured logging + metrics for each graph operation.
- Define timeout/retry policy and queueing guidance when Neo4j is unreachable.
- Provide developer cookbook for common troubleshooting scenarios.

### T3 – Backend Client Adapters (M2)
- Extend `backend/src/services/` wrappers to instantiate a Daedalus Graph client.
- Refactor Multi-Tier Memory, AutoSchemaKG, DatabaseManager, and health checks to consume the client.
- Update integration tests to assert zero direct `neo4j` imports.

### T4 – Legacy Module Containment (M2)
- Audit `extensions/context_engineering/` and `dionysus-source/` scripts.
- Migrate critical flows onto the facade; relocate non-critical scripts to an archival namespace with warnings.
- Record adapter plans in the legacy registry.

### T5 – Governance Automation (M3)
- Configure pre-commit + CI checks preventing new `neo4j` imports outside `daedalus-gateway`.
- Publish compliance checklist and rollout timeline.
- Add regression tests ensuring the ban remains enforced.

### T6 – Change Management & Enablement (M3)
- Broadcast adoption milestones to all agents.
- Update AGENT_CONSTITUTION references with final enforcement date.
- Run post-migration review capturing lessons learned.

