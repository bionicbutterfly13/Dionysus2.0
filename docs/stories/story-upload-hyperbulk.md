# Story: Flux Hyper-Bulk Upload UI

- **Status**: Draft
- **Goal**: Allow users to drop entire directories (hundreds of files) into Flux, process locally by default, and surface concept summaries + related documents instantly.

## Acceptance Criteria
1. Drag/drop accepts folders at scale (hundreds of files).
2. Local processing toggle is ON by default; switching to cloud warns the user.
3. Concept summary and related-docs panel populate after each batch.
4. Bulk uploads append to the knowledge graph (concept nodes, links).
5. API hooks documented for backend batch processing.

## Tasks
- [ ] Implement folder drag/drop UI with progress indicator.
- [ ] Add processing mode control (local vs. cloud) with warnings.
- [ ] Render concept summary widget (top concepts, count).
- [ ] Render related documents column with highlights.
- [ ] Emit batch-processing events to backend.
- [ ] Tests: component renders, bulk upload handling, toggle behavior.

## Notes
- See brainstorming doc: `docs/brainstorming/flux-ui-integrations-brainstorm.md`
- Coordinates with backend batch extraction service.
