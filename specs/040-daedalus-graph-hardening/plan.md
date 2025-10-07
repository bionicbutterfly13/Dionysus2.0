# Technical Implementation Plan: Daedalus Graph Hardening

**Derived From**: `specs/040-daedalus-graph-hardening/spec.md`

## Guiding Principles
- Respect the constitution: Daedalus is the sole ingress and LangGraph River owns long-term knowledge orchestration.
- Preserve service boundaries: backend modules interact through compatibility shims; Daedalus owns infrastructure connections.
- Deliver incremental wins: facade first, migrations second, governance third.

## Architecture Decisions

1. **Facade Ownership**
   - Implement the Graph Channel inside `daedalus-gateway` so all Neo4j drivers reside there.
   - Channel exposes async tasks for schema, read, write, health, and ingestion audit.

2. **Backend Integration**
   - Replace direct Neo4j imports with a `DaedalusGraphClient` obtained from existing service wrappers.
   - Extend compatibility modules in `backend/src/services` to call the channel.

3. **Governance & Tooling**
   - Introduce lint checks (pre-commit + CI) that block new `neo4j` imports outside the gateway.
   - Maintain a registry describing temporary adapters and deprecation timelines.

## Milestones

1. **M1 – Facade Delivery**
   - Ship Graph Channel API, unit coverage, LangGraph River flows, and developer guide.

2. **M2 – Backend Cutover**
   - Multi-Tier Memory, AutoSchemaKG, DatabaseManager, and health endpoints consume the facade.
   - Legacy extension modules either refactored or moved to archival namespace.

3. **M3 – Compliance Enforcement**
   - Guardrails active in CI, documentation updated, rollout checklist completed.

## Dependencies
- Coordination with Daedalus/LangGraph maintainers.
- Access to `daedalus-gateway` repository for facade implementation.
- Agreement on deprecation handling for legacy scripts.

## Open Questions
- Do we need a synchronous facade alongside async for CLI tools?
- What telemetry should be attached to each graph operation for auditing?
- How do we surface graceful degradation when LangGraph River is offline?

