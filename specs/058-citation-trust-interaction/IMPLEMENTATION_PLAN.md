# Citation Trust UI Implementation Plan

**Related milestone:** `spec-058`
**Related backlog tasks:** task-058A, task-058B, task-058C
**Last updated:** 2025-10-07

## Component Contract

- `CitationPanel` (new) lives in `frontend/src/components/CitationPanel/`.
  - **Props:**
    - `isOpen: boolean` – controls sheet visibility.
    - `citation: { chunkId: string; chunkText: string; sequence: number } | null` – highlighted chunk payload.
    - `basins: Array<{ id: string; name: string; stability: number; resonance: number }>` – context extracted from Spec 057 metadata.
    - `thoughtseeds: Array<{ id: string; content: string; resonance: number }>` – downstream context for trust scoring.
    - `onClose(): void` – invoked by close affordance and Esc key press.
    - `onNavigate?(targetChunkId: string): void` – optional callback to request scroll synchronisation in the document view.
  - **State handling:** the component remains stateless; it receives fully-resolved data and surfaces loading placeholders when `citation` is null while `isOpen` is true.
  - **Accessibility:** use shadcn `Sheet` primitives, ensure focus trap + aria labelling.

## API/Data Flow

1. `DocumentDetail` page intercepts citation link clicks and calls `/api/documents/{id}/citations?chunk_id=<chunk>`.
2. Backend endpoint returns payload:
   ```json
   {
     "chunk": { "id": "doc_123_chunk_4", "text": "...", "position": 4 },
     "basins": [ { "id": "basin_1", "name": "Alignment", "stability": 0.82, "resonance": 0.67 } ],
     "thoughtseeds": [ { "id": "seed_9", "content": "Revisit data provenance", "resonance": 0.74 } ]
   }
   ```
3. Frontend stores the result in local component state (`useState` within DocumentDetail) and passes props to `CitationPanel`.
4. When the panel closes, state resets to null; repeated clicks re-fetch only if the requested chunk differs or cache is stale.

## UX Flow

1. User clicks a numbered citation badge inside the summary or chunk list.
2. `DocumentDetail` prevents default anchor behaviour, stores `selectedChunkId`, and opens `CitationPanel`.
3. Panel displays loading skeleton until fetch completes; upon success it renders chunk text, basin metadata, and thoughtseed list.
4. If `onNavigate` is provided, clicking "Scroll to chunk" triggers highlight scroll in the base page using existing Spec 056 chunk IDs.
5. Close button or overlay click invokes `onClose`, dismissing the panel and clearing highlight state.

## Testing Strategy

- **Frontend:** `frontend/tests/unit/CitationPanel.test.tsx` (task-058A) verifies rendering, highlight markup, metadata sections, and close behaviour.
- **Backend:** `backend/tests/contract/test_documents_get_citations.py` (task-058B) covers success/404/400 responses.
- **Integration:** follow-up to assert DocumentDetail opens the panel and scrolls to chunks once API is implemented.

## Open Questions / Dependencies

- Should the backend support batching multiple citation IDs in one request?
- Do we need optimistic updates when navigating between citations rapidly?
- How does the panel behave on very small viewports (mobile)? May require responsive tweak by UI team.
- Confirm chunk text length limits to avoid rendering extremely long passages.

## Next Steps

- Finalise contract tests and merge once backend endpoint skeleton exists.
- Build `CitationPanel` with shadcn components after tests are RED.
- Wire `DocumentDetail` click handlers to open the panel.
- Extend analytics/logging if trust interactions need telemetry.
