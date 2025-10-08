---
id: task-058B
title: Add contract tests for /api/documents/{id}/citations
status: To Do
assignee: []
created_date: '2025-10-07 14:35'
labels:
  - spec-058
  - backend
  - testing
dependencies: []
priority: high
milestone: spec-058
---

## Description

Introduce a new contract test module that exercises the forthcoming citation trust API endpoint. The suite should define RED cases for the happy path payload (chunks + basins + thoughtseeds), 404 when the document does not exist, and 400 when the requested chunk is missing.

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Create `backend/tests/contract/test_documents_get_citations.py` (or update existing) with pytest-asyncio tests using the project test client.
- [ ] #2 Happy-path test seeds a document fixture with at least one chunk/basin/thoughtseed and asserts the endpoint returns `200` plus normalized metadata keys (`chunks`, `basins`, `thoughtseeds`).
- [ ] #3 Error test covers `404` when the document ID is unknown.
- [ ] #4 Error test covers `400` when the chunk identifier is invalid for the given document.
- [ ] #5 Contract suite remains RED until the API is implemented (document failures allowed temporarily).
- [ ] #6 Tests documented in the task notes with exact command (`pytest backend/tests/contract/test_documents_get_citations.py`).
<!-- AC:END -->

## Notes

- Coordinate with backend schema owners to reuse existing repositories for chunk retrieval.
- Use `async_client` fixtures similar to other contract suites for authentication/headers.
