# Tasks: ThoughtSeed State Watching

**Input**: Design documents from `/specs/005-a-thought-seed/`
**Prerequisites**: plan.md (✅ complete), spec.md (✅ complete)

## Phase 3.1: Setup
- [ ] T001 Install thoughtseed-active-inference package (already extracted)
- [ ] T002 Configure Redis connection for state logging
- [ ] T003 [P] Setup pytest with asyncio support

## Phase 3.2: Tests First (TDD)
- [ ] T004 [P] Contract test: ThoughtSeed watch toggle (tests/contract/test_thoughtseed_watch.py)
- [ ] T005 [P] Contract test: State capture (tests/contract/test_state_capture.py)
- [ ] T006 [P] Contract test: Log retrieval with TTL (tests/contract/test_log_retrieval.py)
- [ ] T007 [P] Integration test: Watch 5 concurrent instances (tests/integration/test_concurrent_watching.py)
- [ ] T008 [P] Integration test: 10-minute retention (tests/integration/test_log_retention.py)

## Phase 3.3: Core Implementation
- [ ] T009 Implement ThoughtSeed watch manager (backend/src/services/thoughtseed_watch_manager.py)
- [ ] T010 Add watch toggle to ThoughtSeed model (integrate with extracted package)
- [ ] T011 Implement state capture with <10ms overhead
- [ ] T012 Implement Redis state logger with TTL
- [ ] T013 Implement log view integration
- [ ] T014 Add relationship explanation tracking

## Phase 3.4: Integration
- [ ] T015 Integrate with existing attractor basins
- [ ] T016 Connect to context engineering pipeline
- [ ] T017 Add API endpoints for watch control
- [ ] T018 Test with active inference framework

## Phase 3.5: Polish
- [ ] T019 Performance validation (<10ms overhead)
- [ ] T020 Create data-model.md documentation
- [ ] T021 Run full test suite

## Dependencies
```
Setup (T001-T003) → Tests (T004-T008) → Implementation (T009-T014) → Integration (T015-T018) → Polish (T019-T021)
```

## Notes
- ThoughtSeed package already extracted to /Volumes/Asylum/dev/thoughtseeds
- Integration must preserve existing active inference functionality
- Performance critical: <10ms logging overhead
