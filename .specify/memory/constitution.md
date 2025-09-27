# Dionysus 2.0 ThoughtSeed Constitution

## Core Principles

### I. Code-Review-First (NON-NEGOTIABLE)
**MANDATORY**: Before writing ANY code, model, feature, or implementation, MUST thoroughly read and review ALL existing related code to understand:
- Existing model patterns and structures
- Current implementations that might overlap
- Established coding conventions and standards
- Existing functionality that can be extended vs recreated
- Database schemas and entity relationships already defined

**ENFORCEMENT**: Any code submission without evidence of prior code review will be automatically rejected. Developers must document what existing code was reviewed before implementation.

### II. Test-First Development (TDD)
TDD mandatory: Tests written → User approved → Tests fail → Then implement. Red-Green-Refactor cycle strictly enforced. All contract tests must exist and fail before implementation begins.

### III. Constitutional Document Processing
ALL document processing must follow the mandatory delegation pipeline: Dionysus Agent → Daedalus → Specialist Agent. No direct processing allowed. All four attractors must be activated: concept_extractor, semantic_analyzer, episodic_encoder, procedural_integrator.

### IV. Hybrid Database Architecture
All data must be stored according to constitutional requirements: Redis (TTL cache), Neo4j (graph relationships), Vector DB (embeddings). Triple storage system is mandatory for memory formation compliance.

### [PRINCIPLE_5_NAME]
<!-- Example: V. Observability, VI. Versioning & Breaking Changes, VII. Simplicity -->
[PRINCIPLE_5_DESCRIPTION]
<!-- Example: Text I/O ensures debuggability; Structured logging required; Or: MAJOR.MINOR.BUILD format; Or: Start simple, YAGNI principles -->

## [SECTION_2_NAME]
<!-- Example: Additional Constraints, Security Requirements, Performance Standards, etc. -->

[SECTION_2_CONTENT]
<!-- Example: Technology stack requirements, compliance standards, deployment policies, etc. -->

## [SECTION_3_NAME]
<!-- Example: Development Workflow, Review Process, Quality Gates, etc. -->

[SECTION_3_CONTENT]
<!-- Example: Code review requirements, testing gates, deployment approval process, etc. -->

## Governance
<!-- Example: Constitution supersedes all other practices; Amendments require documentation, approval, migration plan -->

[GOVERNANCE_RULES]
<!-- Example: All PRs/reviews must verify compliance; Complexity must be justified; Use [GUIDANCE_FILE] for runtime development guidance -->

**Version**: [CONSTITUTION_VERSION] | **Ratified**: [RATIFICATION_DATE] | **Last Amended**: [LAST_AMENDED_DATE]
<!-- Example: Version: 2.1.1 | Ratified: 2025-06-13 | Last Amended: 2025-07-16 -->