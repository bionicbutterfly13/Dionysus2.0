# Story: Flux Concept Explorer & Basin Visualizer

- **Status**: Draft
- **Goal**: Interactive concept map that reacts to user selections, reruns the basin, and shows cluster insights in real time.

## Acceptance Criteria
1. Selecting text or topics triggers concept map update.
2. Real-time basin cluster displayed (concepts, documents, links).
3. Optional daydream panel shows layered thoughtseeds and emergent priors.
4. Narrative extraction feed (archetypes/sentiment/isomorphs) updates live.
5. Concept nodes link to “best explanations” saved in knowledge manager.

## Tasks
- [ ] Build concept map component with data streaming.
- [ ] Integrate basin API for cluster activation.
- [ ] Render optional “Inner Screen” panel (daydream feed).
- [ ] Hook up narrative extraction service for chat-style reflections.
- [ ] Tests: concept selection, basin update, feed rendering.

## Notes
- Draw from SurfSense visualization patterns + new narrative feed requirements.
- Requires backend events for basin activation and narrative extraction.
