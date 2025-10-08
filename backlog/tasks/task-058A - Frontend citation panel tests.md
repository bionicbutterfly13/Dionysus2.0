---
id: task-058A
title: Add frontend tests for Citation Trust panel
status: To Do
assignee: []
created_date: '2025-10-07 14:30'
labels:
  - spec-058
  - frontend
  - testing
dependencies: []
priority: high
milestone: spec-058
---

## Description

Write the initial RED test suite for the Spec 058 citation side-panel before implementation. The tests should target a new `CitationPanel` React component that will live under `frontend/src/components/CitationPanel/` and verify the shadcn-based side sheet renders correctly, highlights the selected chunk, and surfaces basin / thoughtseed context.

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Unit test mounts the panel when `isOpen` is true and asserts the headline plus close control are rendered.
- [ ] #2 Test feeds a highlighted chunk snippet and expects it to appear with the `data-testid="citation-highlight"` marker.
- [ ] #3 Basin and thoughtseed metadata blocks render when provided in props, each with their section headers.
- [ ] #4 Close button triggers the provided `onClose` callback exactly once per click.
- [ ] #5 Snapshot or DOM assertions cover loading state when `chunk` data is undefined.
- [ ] #6 Tests run via `npm test -- CitationPanel` and fail prior to implementation (RED).
<!-- AC:END -->

## Notes

- Coordinate with Agent 058C on the final prop contract as the implementation plan evolves.
- Prefer React Testing Library utilities already used in the project (see `frontend/tests`).
