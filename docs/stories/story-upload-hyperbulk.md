# Story: Flux Hyper-Bulk Upload UI

- **Status**: Approved
- **Goal**: Allow users to drop entire directories (hundreds of files) into Flux, process locally by default, and surface concept summaries + related documents instantly.

## Acceptance Criteria
1. Drag/drop accepts folders (up to **10 files per batch** for the initial release). Additional files are queued for the next batch with clear messaging.
2. Local processing toggle defaults to **Local ON**. If the local Daedalus pipeline is unavailable, immediately notify the user (“Local processing is offline. Please try again later.”) and abort the batch without calling cloud APIs.
3. Upload view displays each file with a live progress bar and status icon: spinner while processing, ✅ on success, ❌ on validation/upload failure (process continues with remaining files).
4. Successful files flow through the existing Daedalus auto-schema pipeline to append concepts and links to the knowledge graph; failures are logged and surfaced in the UI.
5. After each batch completes, the sidebar refreshes concept summaries and related documents using the existing local cache (no external API/token usage).
6. User-facing copy is terse and actionable for successes, failures, and processing outages.

## Tasks
- [ ] Implement folder drag/drop UI with per-file progress bars and status icons.
- [ ] Enforce 10-file batch limit with overflow queued messaging.
- [ ] Add processing mode control (local only for now) and “local offline” blocker message.
- [ ] Stream files into Daedalus ingest pipeline sequentially; skip invalid files, continue with next.
- [ ] Refresh concept summary + related docs sidebar after batch using local cache hooks.
- [ ] Emit local Daedalus ingest events (no external API calls) and log failures.
- [ ] Tests: component rendering, batch limit enforcement, failure-handling, concept summary refresh triggers.
- [ ] UX copy review for notifications and status labels.

## Notes
- See brainstorming doc: `docs/brainstorming/flux-ui-integrations-brainstorm.md`
- Coordinates with existing Daedalus batch extraction service (local pipeline only for this release).
