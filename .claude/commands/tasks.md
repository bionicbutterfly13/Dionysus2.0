---
description: Generate an actionable, dependency-ordered tasks.md for the feature based on available design artifacts.
---

The user input to you can be provided directly by the agent or as a command argument - you **MUST** consider it before proceeding with the prompt (if not empty).

User input:

$ARGUMENTS

1. **FIRST**: Validate Context Engineering Components
   - Verify Attractor Basin Manager is accessible
   - Verify Neural Field System is available
   - Add mandatory Context Engineering validation tasks to task list
   - Ensure every feature includes basin/field integration tests

2. Run `.specify/scripts/bash/check-prerequisites.sh --json` from repo root and parse FEATURE_DIR and AVAILABLE_DOCS list. All paths must be absolute.

3. Load and analyze available design documents:
   - Always read plan.md for tech stack and libraries
   - IF EXISTS: Read data-model.md for entities
   - IF EXISTS: Read contracts/ for API endpoints
   - IF EXISTS: Read research.md for technical decisions
   - IF EXISTS: Read quickstart.md for test scenarios

   Note: Not all projects have all documents. For example:
   - CLI tools might not have contracts/
   - Simple libraries might not need data-model.md
   - Generate tasks based on what's available

4. Generate tasks following the template:
   - Use `.specify/templates/tasks-template.md` as the base
   - Replace example tasks with actual tasks based on:
     * **Setup tasks**: Project init, dependencies, linting
     * **Context Engineering tasks [MANDATORY]**:
       - Basin integration test (verify AttractorBasinManager integration)
       - Field resonance test (verify Neural Field System integration)
       - Redis persistence test (verify basin state storage)
     * **Test tasks [P]**: One per contract, one per integration scenario
     * **Core tasks**: One per entity, service, CLI command, endpoint
     * **Integration tasks**: DB connections, middleware, logging
     * **Polish tasks [P]**: Unit tests, performance, docs

5. Task generation rules:
   - **MANDATORY**: Include Context Engineering validation tasks FIRST
   - Each contract file → contract test task marked [P]
   - Each entity in data-model → model creation task marked [P]
   - Each endpoint → implementation task (not parallel if shared files)
   - Each user story → integration test marked [P]
   - Different files = can be parallel [P]
   - Same file = sequential (no [P])
   - Context Engineering tests → must pass before core implementation

6. Order tasks by dependencies:
   - Setup before everything
   - **Context Engineering validation before core work**
   - Tests before implementation (TDD)
   - Models before services
   - Services before endpoints
   - Core before integration
   - Everything before polish

7. Include parallel execution examples:
   - Group [P] tasks that can run together
   - Show actual Task agent commands
   - Include Context Engineering validation examples

8. Create FEATURE_DIR/tasks.md with:
   - Correct feature name from implementation plan
   - **Context Engineering validation tasks (T001-T003 typically)**
   - Numbered tasks (T001, T002, etc.)
   - Clear file paths for each task
   - Dependency notes showing Context Engineering prerequisites
   - Parallel execution guidance

Context for task generation: $ARGUMENTS

The tasks.md should be immediately executable - each task must be specific enough that an LLM can complete it without additional context.
