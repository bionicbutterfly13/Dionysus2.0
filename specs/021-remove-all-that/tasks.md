# Tasks: Clean Daedalus Class for Perceptual Information Gateway

**Input**: Design documents from `/specs/021-remove-all-that/`
**Prerequisites**: plan.md (✅ complete), spec.md (✅ complete), 11/11 tests passing

## Path Conventions
- **Web app structure**: `backend/src/`, `backend/tests/`
- All paths relative to repository root: `/Volumes/Asylum/dev/Dionysus-2.0/`

## Phase 3.1: Setup ✅ COMPLETE
- [x] T001 ✅ Project structure verified (backend/src/services/, backend/tests/)
- [x] T002 ✅ Python 3.11 environment with dependencies installed
- [x] T003 ✅ Pytest configuration working (conftest.py created)

## Phase 3.2: Tests First (TDD) ✅ COMPLETE
**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**
- [x] T004 [P] ✅ Contract test: Class instantiation (test_daedalus_class_exists)
- [x] T005 [P] ✅ Contract test: Single responsibility (test_daedalus_has_single_responsibility)
- [x] T006 [P] ✅ Contract test: Upload reception (test_daedalus_receives_uploaded_data)
- [x] T007 [P] ✅ Contract test: Multi-format support (test_daedalus_handles_all_file_types)
- [x] T008 [P] ✅ Contract test: Gateway function (test_daedalus_serves_as_gateway)
- [x] T009 [P] ✅ Contract test: LangGraph integration (test_daedalus_interfaces_with_langgraph)
- [x] T010 [P] ✅ Contract test: No extra functionality (test_daedalus_no_extra_functionality)
- [x] T011 [P] ✅ Contract test: Archive verification (test_daedalus_archived_functionality)
- [x] T012 [P] ✅ Contract test: Information flow (test_daedalus_maintains_information_flow)
- [x] T013 [P] ✅ Contract test: Error handling (test_daedalus_error_handling)
- [x] T014 [P] ✅ Contract test: Performance <50ms (test_daedalus_performance_requirement)

## Phase 3.3: Core Implementation ✅ COMPLETE
- [x] T015 ✅ Implement Daedalus class in backend/src/services/daedalus.py
- [x] T016 ✅ Implement receive_perceptual_information method
- [x] T017 ✅ Add LangGraph agent creation integration
- [x] T018 ✅ Implement error handling for edge cases
- [x] T019 ✅ Add type hints and documentation
- [x] T020 ✅ Verify all 11 tests passing

## Phase 3.4: Integration & Cleanup ⚠️ IN PROGRESS
- [ ] T021 Fix integration test imports in backend/tests/integration/test_daedalus_integration.py
- [ ] T022 [P] Verify archived functionality in backup/deprecated/daedalus_removed_features/
- [ ] T023 [P] Create data-model.md documenting Daedalus entities
- [ ] T024 Update backend API routes to use simplified Daedalus gateway
- [ ] T025 Test upload flow: Frontend → API → Daedalus → LangGraph

## Phase 3.5: Documentation & Validation
- [ ] T026 [P] Create contracts/ directory with API specifications
- [ ] T027 [P] Document LangGraph integration patterns
- [ ] T028 [P] Update CLAUDE.md with Daedalus cleanup completion
- [ ] T029 Run full backend test suite to verify no regressions
- [ ] T030 Performance profiling: Verify <50ms requirement under load

## Dependencies
```
Setup (T001-T003) → Tests (T004-T014) → Implementation (T015-T020) → Integration (T021-T025) → Documentation (T026-T030)

Parallel Tasks:
- T004-T014: All contract tests (independent files)
- T022-T023: Archive verification + documentation (independent)
- T026-T028: Documentation tasks (independent files)

Sequential Tasks:
- T021 → T024 → T025: Integration tests must pass before API integration
- T029 → T030: Full test suite before performance profiling
```

## Parallel Execution Examples
```bash
# Contract Tests (Already Complete)
pytest backend/tests/test_daedalus_spec_021.py::TestDaedalusSpecification::test_daedalus_class_exists -v
pytest backend/tests/test_daedalus_spec_021.py::TestDaedalusSpecification::test_daedalus_has_single_responsibility -v
# ... (all 11 tests can run in parallel)

# Documentation Tasks (T026-T028)
# Task 1: Create contracts/daedalus_gateway_api.yaml
# Task 2: Document LangGraph integration in contracts/langgraph_integration.md
# Task 3: Update CLAUDE.md with cleanup status
```

## Current Status Summary
✅ **Phases 3.1-3.3 COMPLETE**: 20/30 tasks done (66.7%)
⚠️ **Phase 3.4 IN PROGRESS**: 5 integration tasks remaining
📋 **Phase 3.5 PENDING**: 5 documentation tasks

**Next Actions**:
1. Fix integration test imports (T021)
2. Verify archived functionality (T022)
3. Create data model documentation (T023)
4. Update API routes for Daedalus integration (T024)
5. Test end-to-end upload flow (T025)

## Notes
- ✅ TDD cycle complete: RED → GREEN achieved
- ✅ Constitution compliance verified (NumPy not required, TDD followed)
- ✅ All 11 contract tests passing
- ⚠️ Integration tests need import path fixes
- 📋 Archive verification pending
- 📋 Documentation tasks remain

## Validation Checklist
*GATE: Checked before marking feature complete*

- [x] All contract tests passing (11/11)
- [x] Core implementation complete
- [x] Single responsibility verified
- [ ] Integration tests passing
- [ ] Archive verified and documented
- [ ] API routes updated
- [ ] End-to-end flow tested
- [ ] Performance requirements met (<50ms)
- [ ] No regressions in full test suite
- [ ] Documentation complete
