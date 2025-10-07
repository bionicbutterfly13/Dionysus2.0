# Feature Specification: Daedalus Graph Hardening Phase

**Feature Branch**: `040-daedalus-graph-hardening`  
**Created**: 2025-10-02  
**Status**: Draft  
**Input**: Align all agents so Neo4j access flows exclusively through Daedalus + LangGraph River

## User Scenarios & Testing

### Primary User Story
Internal engineers and automation agents need to evolve Dionysus without violating constitutional rules. When they ingest documents or update knowledge graphs, they rely on Daedalus to orchestrate LangGraph River operations, never calling Neo4j drivers directly.

### Acceptance Scenarios
1. **Given** a backend service requires knowledge graph reads, **When** it executes through the Daedalus Graph Channel facade, **Then** no module in `Dionysus-2.0/` imports `neo4j` and the request succeeds via LangGraph River tasks.
2. **Given** a migration agent is onboarding, **When** they read the hardening specification, **Then** they can identify the facade entry points, allowed APIs, and the governance rules that block any direct driver usage.

### Edge Cases
- What happens when Daedalus or LangGraph River is unavailable? -> Fallback guidance must specify queueing, not reconnecting via raw Neo4j drivers.
- How does the system handle legacy scripts that still depend on direct driver access? -> Specification must define archival or adapter strategy so they cannot regress runtime compliance.

## Requirements

### Functional Requirements
- **FR-001**: System MUST expose a Daedalus Graph Channel API that encapsulates all Neo4j interactions required by Dionysus services.
- **FR-002**: System MUST provide migration guidance so existing services (Multi-Tier Memory, AutoSchemaKG, DatabaseManager, health tooling) consume the facade exclusively.
- **FR-003**: System MUST define guardrails (code checks + CI) that block any new direct imports of `neo4j` within the Dionysus workspace.
- **FR-004**: System MUST document operational playbooks for failure scenarios without instructing users to bypass Daedalus.
- **FR-005**: System MUST communicate adoption checkpoints so other agents know when the prohibition becomes enforceable.

### Key Entities
- **Daedalus Graph Channel**: High-level operations (schema, read, write, health) implemented on top of LangGraph River tasks.
- **Graph Access Policy**: Rules describing which Dionysus modules may request which channel operations and the allowed payloads.
- **Compliance Gate**: Automated checks (pre-commit + CI) that scan for unauthorized `neo4j` imports or direct driver usage.
- **Legacy Adapter Registry**: Catalog of scripts/tools that cannot be refactored immediately and the shims that keep them within the facade boundary.

## Review & Acceptance Checklist

### Content Quality
- [x] No implementation details (languages, frameworks, APIs)
- [x] Focused on user value and business needs
- [x] Written for non-technical stakeholders
- [x] All mandatory sections completed

### Requirement Completeness
- [x] No [NEEDS CLARIFICATION] markers remain
- [x] Requirements are testable and unambiguous
- [x] Success criteria are measurable
- [x] Scope is clearly bounded
- [x] Dependencies and assumptions identified

## Execution Status

- [x] User description parsed
- [x] Key concepts extracted
- [ ] Ambiguities marked
- [x] User scenarios defined
- [x] Requirements generated
- [x] Entities identified
- [x] Review checklist passed
