# Flux UI Integrations Brainstorm – 2025-10-09

## Context
- Unify Perplexica, SurfSense, and OpenNotebook experiences inside Flux.
- Use existing data sources, keep processing local by default, and ensure everything flows through the knowledge graph.
- Deliver Flux-native UI; external repos are reference only.

---

## Stage Summary (Experience Mapping)
1. **Discover** – Bulk document upload (hundreds at once), concept extraction, basin activation, related-doc panel.
2. **Explore** – Select text or topics → real-time basin cluster, concept map, narrative archetype/sentiment feedback.
3. **Synthesize** – Narrow/deep/wide generation, dual-concept graphing, writing workspace, notebook integration.
4. **Reflect** – Knowledge manager with bidirectional links, Zettelkasten distillations, concept timeline.
5. **Inner Daydream** – Optional “machine consciousness” feed showing layered thoughtseeds and emergent priors.

---

## Feature / UI Ideas
- **Bulk Upload Panel**
  - Supports entire directories (hundreds of files).
  - Local processing toggle default; warning when switching to cloud.
  - Shows concept extraction summary + related docs column.

- **Interactive Concept Explorer**
  - Concept map updates when user hovers or selects text.
  - Consciousness feed highlights the active cluster; optional “daydream window” shows ongoing thoughtseeds.
  - Narrative extraction (archetypes/sentiment/isomorphs) updates live, displayed chat-style.

- **Synthesis Workspace**
  - Buttons for narrow / deeper / wider.
  - Dual-concept analysis returns flowcharts/graphs showing relationships.
  - Users can pin the resulting summary to the notebook; Flux auto-tags relevant concepts.
  - Zettelkasten card view for distillations and cross-context semantic emergence.

- **Knowledge Manager**
  - Saved artifacts list with date, tags, linked concepts.
  - Bidirectional linking: each concept points to the best explanation; each artifact lists the concepts it enriches.
  - Timeline view showing the evolution of a concept cluster over time.
  - Search across all saved distillations.

- **Notebook / Persona Chat**
  - Main writing space with real-time knowledge prompts (the “bubble to mind” effect).
  - Narrative reflection feed: archetypes, dominant themes, sentiment.
  - “Request metaphor” button triggers isomorphic metaphors.

- **Curiosity Engine Deep Search**
  - Accepts topic clusters, runs deep traversal through linked docs/citations.
  - Returns a coherent model, navigable via the knowledge graph.

---

## Next Story Seeds
1. `story-upload-hyperbulk.md` – Implement enhanced bulk upload UI (local default, concept summary, related docs).
2. `story-concept-explorer.md` – Interactive concept map + basin activation + daydream feed.
3. `story-synthesis-workspace.md` – Generation controls and notebook integration (narrow/deep/wide, dual-concept graphs).
4. `story-knowledge-manager.md` – Bidirectional links, Zettelkasten distillation, timeline view.
5. `story-notebook-narrative.md` – Narrative extraction feed, archetype detection, metaphor requests.

---

## Open Questions
- Exact data formats needed from legacy components (Perplexica, SurfSense) to seed Flux features.
- CPU / GPU requirements to keep all processing local at scale.
- How to visualize the “daydream feed” without overwhelming the user.
- Permission model for saved distillations and knowledge graph entries.

