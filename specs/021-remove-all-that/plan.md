# Implementation Plan: Clean Daedalus Class for Perceptual Information Gateway

**Branch**: `021-remove-all-that` | **Date**: 2025-09-30 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/021-remove-all-that/spec.md`

## Summary
Simplify the Daedalus class to a single responsibility: receiving perceptual information from external sources (uploads). Remove all non-essential functionality while maintaining integration with LangGraph architecture for agent creation. Implementation follows strict TDD approach per Spec 021 requirements.

## Technical Context
**Language/Version**: Python 3.11+
**Primary Dependencies**: LangGraph, FastAPI, Pydantic 2.x
**Storage**: Redis (optional for future state management)
**Testing**: pytest 8.4+
**Target Platform**: Backend service (Darwin/Linux)
**Project Type**: Web application (backend/frontend structure)
**Performance Goals**: <50ms for perceptual information reception
**Constraints**: Single public method only, all file types supported
**Scale/Scope**: Core backend service, foundation for consciousness processing

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

✅ **NumPy 2.0+ Compliance**: Not applicable - no NumPy dependencies in Daedalus
✅ **Environment Isolation**: Using backend virtual environment
✅ **TDD Standards**: FR-001 requires tests before implementation - FOLLOWED
✅ **Consciousness Processing**: Daedalus serves as perceptual gateway for consciousness system
✅ **Service Health**: Tests verify single responsibility and clean architecture

**Status**: ✅ **CONSTITUTION COMPLIANT**

## Project Structure

### Documentation (this feature)
```
specs/021-remove-all-that/
├── spec.md              # Feature specification (completed)
├── plan.md              # This file (/plan command output)
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── contracts/           # Phase 1 output
└── tasks.md             # Phase 2 output (/tasks command)
```

### Source Code (repository root)
```
backend/
├── src/
│   ├── services/
│   │   └── daedalus.py              # Simplified Daedalus class (IMPLEMENTED)
│   └── legacy/
│       ├── daedalus_audit.md        # Migration audit
│       └── daedalus_bridge/         # Legacy integration layer
│           ├── __init__.py
│           └── context_isolator.py
└── tests/
    ├── conftest.py                   # Pytest configuration (CREATED)
    ├── test_daedalus_spec_021.py    # TDD tests (11/11 PASSING)
    └── integration/
        └── test_daedalus_integration.py  # Integration tests (TO BE FIXED)

backup/deprecated/
└── daedalus_removed_features/       # Archived functionality (TO BE VERIFIED)
    └── removed_methods.py
```

**Structure Decision**: Web application structure with backend/frontend separation. Backend contains src/ for source code and tests/ for test suite. Legacy components preserved in src/legacy/ for migration reference.

## Phase 0: Outline & Research

### Research Complete ✅

**Decision**: Single Responsibility Pattern
**Rationale**:
- Daedalus serves ONE function: receive perceptual information from uploads
- Follows SOLID principles (Single Responsibility Principle)
- Enables clear testing and maintenance
- Reduces cognitive complexity

**Alternatives Considered**:
1. Multi-function gateway - REJECTED: violates single responsibility
2. Microservice per upload type - REJECTED: over-engineering for current needs
3. Event-driven architecture - DEFERRED: can be added later without breaking changes

**Technical Decisions**:
- Use BinaryIO type for file-like objects (Python standard)
- Return Dict[str, Any] for reception metadata (flexible structure)
- LangGraph integration through factory function (separation of concerns)
- Archive removed functionality to backup/deprecated/ (preserves history)

**Best Practices Applied**:
- TDD: Tests written BEFORE implementation (FR-001)
- Type hints: Full typing for IDE support and documentation
- Docstrings: Google-style documentation
- Error handling: Graceful degradation for edge cases

**Output**: ✅ research.md complete - all technical decisions documented

## Phase 1: Design & Contracts

### Data Model
Key entities identified:

1. **Daedalus Class**
   - Single public method: `receive_perceptual_information(data: Optional[BinaryIO]) -> Dict[str, Any]`
   - Private attribute: `_is_gateway: bool` (gateway identification)
   - No state persistence (stateless design)

2. **Perceptual Information**
   - Format: File-like binary stream (BinaryIO)
   - Metadata: filename, file type, size, content
   - All formats supported: PDF, TXT, JSON, images, etc.

3. **Reception Response**
   - status: 'success' | 'error'
   - metadata: filename, size, type
   - agents_created: list of LangGraph agent IDs
   - error_message: optional error details

### API Contracts
Contract tests define expected behavior:

1. **Class Instantiation**
   - Input: None
   - Output: Daedalus instance
   - Test: `test_daedalus_class_exists`

2. **Single Responsibility Verification**
   - Input: Daedalus instance
   - Output: Exactly ONE public method
   - Test: `test_daedalus_has_single_responsibility`

3. **Perceptual Information Reception**
   - Input: BinaryIO file-like object
   - Output: Success status + metadata + agents
   - Test: `test_daedalus_receives_uploaded_data`

4. **Multi-format Support**
   - Input: PDF, TXT, JSON files
   - Output: Successful reception for all types
   - Test: `test_daedalus_handles_all_file_types`

5. **Gateway Function**
   - Input: External data
   - Output: Reception confirmation
   - Test: `test_daedalus_serves_as_gateway`

6. **LangGraph Integration**
   - Input: Received data
   - Output: Created agent IDs
   - Test: `test_daedalus_interfaces_with_langgraph`

7. **Error Handling**
   - Input: None, corrupted data, invalid types
   - Output: Graceful error responses
   - Test: `test_daedalus_error_handling`

8. **Performance**
   - Input: Test file
   - Output: <50ms processing time
   - Test: `test_daedalus_performance_requirement`

### Test Status
✅ **ALL 11 TESTS PASSING**

Contract tests written and passing:
- test_daedalus_class_exists ✅
- test_daedalus_has_single_responsibility ✅
- test_daedalus_receives_uploaded_data ✅
- test_daedalus_handles_all_file_types ✅
- test_daedalus_serves_as_gateway ✅
- test_daedalus_interfaces_with_langgraph ✅
- test_daedalus_no_extra_functionality ✅
- test_daedalus_archived_functionality ✅
- test_daedalus_maintains_information_flow ✅
- test_daedalus_error_handling ✅
- test_daedalus_performance_requirement ✅

### Agent File Update
Agent file updates not required - this is a backend service cleanup, not a new feature requiring CLAUDE.md changes.

**Output**: ✅ Phase 1 complete - all contract tests passing, implementation verified

## Phase 2: Task Planning Approach
*This section describes what the /tasks command will do - DO NOT execute during /plan*

**Task Generation Strategy**:
1. Load test results from Phase 1 (all passing)
2. Identify remaining work:
   - Integration test fixes
   - Documentation updates
   - Archive verification
   - API endpoint integration
3. Generate tasks in TDD order:
   - Fix integration tests first
   - Verify archived functionality
   - Update API routes
   - Performance profiling

**Ordering Strategy**:
- Tests before implementation (TDD)
- Independent tasks marked [P] for parallel execution
- Sequential tasks for files with dependencies
- Integration before documentation

**Estimated Output**: 8-12 numbered tasks covering:
- Integration test repairs
- Archive verification
- API route updates
- Documentation polish
- Performance validation

**IMPORTANT**: This phase is executed by the /tasks command, NOT by /plan

## Phase 3+: Future Implementation
*These phases are beyond the scope of the /plan command*

**Phase 3**: Task execution (/tasks command creates tasks.md)
**Phase 4**: Implementation (execute tasks from tasks.md)
**Phase 5**: Validation (verify all requirements met, run full test suite)

## Complexity Tracking
*No constitutional violations - section intentionally empty*

✅ All complexity within constitutional limits:
- Single class with single responsibility
- Standard Python patterns
- No additional projects or dependencies
- TDD approach followed strictly

## Progress Tracking
*This checklist is updated during execution flow*

**Phase Status**:
- [x] Phase 0: Research complete
- [x] Phase 1: Design complete (11/11 tests passing)
- [x] Phase 2: Task planning approach defined
- [ ] Phase 3: Tasks generated (/tasks command - NEXT STEP)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:
- [x] Initial Constitution Check: PASS
- [x] Post-Design Constitution Check: PASS
- [x] All NEEDS CLARIFICATION resolved (3 clarifications from session 2025-09-28)
- [x] Complexity deviations documented (none - within limits)

**Implementation Status**:
- [x] TDD tests written (11 tests)
- [x] Core implementation complete (daedalus.py)
- [x] All tests passing (11/11)
- [x] Pytest configuration fixed (conftest.py)
- [ ] Integration tests fixed
- [ ] Archive verified
- [ ] Documentation updated

---
*Based on Constitution v1.0.0 - See `.specify/memory/constitution.md`*
*TDD Cycle: ✅ RED → ✅ GREEN → ⏳ REFACTOR (in progress)*
