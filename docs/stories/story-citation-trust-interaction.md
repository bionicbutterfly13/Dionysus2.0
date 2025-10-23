# Story: Spec 058 · Citation Trust Interaction (Frontend)

- **Status**: Ready for SR Re-review
- **Goal**: Deliver the CitationPanel side-sheet UI that presents chunk excerpts, basin telemetry, and thoughtseed context for Spec 058 users inspecting trust signals.
- **Spec**: `specs/058-citation-trust-interaction/IMPLEMENTATION_PLAN.md`

## Acceptance Criteria Snapshot
1. Panel opens/closes via `isOpen`, Escape key, and backdrop click.
2. Highlighted chunk text plus metadata (index, offsets) render when provided.
3. Basin details surface stability and attractor strength with per-layer breakdown.
4. Thoughtseed metadata shows resonance, emergence timestamp, and concept labels.
5. Component remains accessible (dialog semantics, focus handling) and passes Jest suite.

## Implementation Notes (GREEN handoff)
- Component mounted at `frontend/src/components/CitationPanel.tsx`.
- Hooks: `useEffect` for Escape handling, `useRef` for default focus.
- Styling: Tailwind utility classes for layout/sections; placeholders for missing data.
- Tests: `frontend/src/components/__tests__/CitationPanel.test.tsx` (25 assertions) all GREEN via `npm run test -- --runTestsByPath src/components/__tests__/CitationPanel.test.tsx`.

## Senior Review · Agent SR (2025-10-10)
1. ❌ **Layer influence keys were hard-coded** – Component expected `layerInfluences.layer1|layer2|layer3` (`frontend/src/components/CitationPanel.tsx:181-187`), but backend returns dynamic keys (`backend/src/services/citation_service.py:138-149`). Result: real data rendered `undefined` values. **Action**: Iterate over `Object.entries(basinData.layerInfluences)` and render label/value pairs without assuming fixed keys. Update TypeScript interface to `Record<string, number>` and adjust tests accordingly. ✅ **Resolved 2025-10-10**
2. ⚠️ **Optional** – Current focus management only auto-focuses close button; there is no actual focus trap. If UX expects full trap per plan, consider integrating shadcn `Sheet` primitives or focus-lock. (Not blocking if scope limited to MVP.)

## Remediation (2025-10-10)
- Updated `CitationPanel` to treat `layerInfluences` as `Record<string, number>` and render entries dynamically, including humanised labels and placeholder text when empty (`frontend/src/components/CitationPanel.tsx`).
- Expanded Jest coverage to assert dynamic key rendering + empty state, maintaining original RED coverage (now 25 assertions) (`frontend/src/components/__tests__/CitationPanel.test.tsx`).
- Verified with `npm run test -- --runTestsByPath src/components/__tests__/CitationPanel.test.tsx` → ✅ 25/25 passing.

**Next Step**: SR agent to re-run review-story workflow; if no additional findings, promote Spec 058 to BLUE phase.
