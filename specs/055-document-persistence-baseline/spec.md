  # Spec 055 - Document Persistence Baseline

  ## Background
  Contract POST flows now pass, but we still need SurfSense-level guardrails on duplicates
  and summaries before the endpoint carries more load.

  ## Goals
  - Persist a SHA-256 content hash with every document.
  - Reject duplicates with a structured 409 payload that points to the canonical record.
  - Generate and store a token-aware LLM summary for each accepted document.

  ## Non-Goals
  - No UI changes.
  - No LangGraph or transformation work (covered later in Spec 059).

  ## Functional Requirements
  1. Compute a deterministic content hash from document body plus namespace.
  2. Short-circuit persistence on hash collision; respond with 409 including reuse guidance.
  3. Produce an LLM summary that respects model token limits and store it in the document
  record.
  4. Update API and contract tests to cover the new fields and duplicate path.

  ## Technical Notes
  - Extend Neo4j persistence models with  and .
  - Keep Redis and tier metadata in sync.
  - DaedalusGraphChannel remains the only Neo4j access point.

  ## Dependencies and Sequencing
  - Must ship before Specs 056, 057, 058, and 059.

  ## Parallelization
  - None; treat as the foundation.

  ## Acceptance Criteria
  - Duplicate upload returns 409 with canonical document info.
  - Successful uploads expose  and .
  - Contract suite updated and passing.
  EOF

  mkdir -p specs/056-url-and-chunk-ingestion-pipeline
  cat > specs/056-url-and-chunk-ingestion-pipeline/spec.md <<EOF
  # Spec 056 - URL and Chunk Ingestion Pipeline

  ## Background
  Ingestion currently covers only raw uploads. Perplexica demonstrates a minimal, reliable
  pattern for URLs and clean chunking that we want to adopt.

  ## Goals
  - Support ingesting documents from HTTPS URLs (PDF and HTML).
  - Standardize chunking via a shared RecursiveCharacterTextSplitter wrapper.
  - Emit ingestion metadata flagging source type and chunk identifiers.

  ## Non-Goals
  - No UI changes (handled in later specs).
  - No duplicate detection logic changes (owned by Spec 055).

  ## Functional Requirements
  1. Add a downloader that fetches PDFs or HTML pages with retry/backoff and MIME validation.
  2. Convert downloads to text using existing converters with graceful fallback.
  3. Run text through a shared splitter with repository defaults (chunk size and overlap).
  4. Emit stable chunk identifiers for downstream highlighting.
  5. Extend tests to cover direct URL ingestion.

  ## Technical Notes
  - Reuse DocumentRepository flow; hook into persistence layer.
  - Coordinate metadata naming with Spec 057.

  ## Dependencies and Sequencing
  - Requires Spec 055 schema updates.
  - Blocks Spec 058 (needs chunk IDs) and Spec 059.

  ## Parallelization
  - Can run in parallel with Spec 057 once Spec 055 is merged.

  ## Acceptance Criteria
  - URL ingestion succeeds for representative PDF and HTML samples.
  - Chunk metadata stored with documents.
  - Tests cover both upload and URL flows.
  EOF

  mkdir -p specs/057-source-metadata-and-external-access
  cat > specs/057-source-metadata-and-external-access/spec.md <<EOF
  # Spec 057 - Source Metadata and External Access

  ## Background
  We capture files but lose provenance. SurfSense keeps source typing plus an "open original"
  affordance that users expect.

  ## Goals
  - Persist  and  when applicable.
  - Surface an external link action for clients.
  - Migrate existing data without breaking consumers.

  ## Non-Goals
  - No duplicate logic (Spec 055).
  - No citation UI redesign (Spec 058).

  ## Functional Requirements
  1. Extend document schema with , , and connector icon hints.
  2. Backfill existing records with sensible defaults ( with null URL).
  3. Update REST responses and Pydantic models to include the new fields.
  4. Provide a thin UI affordance to open the original source.
  5. Add tests covering upload versus URL records.

  ## Technical Notes
  - Keep Neo4j and Redis data aligned.
  - Coordinate field names with Spec 056.

  ## Dependencies and Sequencing
  - Depends on Spec 055.
  - Runs in parallel with Spec 056.
  - Required before Spec 058.

  ## Parallelization
  - Yes; can proceed alongside Spec 056 after Spec 055 lands.

  ## Acceptance Criteria
  - API responses include source metadata.
  - Existing documents migrated without data loss.
  - UI opens original source when available.
  EOF

  mkdir -p specs/058-citation-trust-interaction
  cat > specs/058-citation-trust-interaction/spec.md <<EOF
  # Spec 058 - Citation Trust Interaction

  ## Background
  SurfSense demonstrates a side-sheet citation panel with chunk highlighting and auto-scroll
  that dramatically improves trust. We want the same while keeping consciousness context
  visible.

  ## Goals
  - Build a side-sheet citation panel using shadcn components.
  - Auto-scroll and highlight the referenced chunk.
  - Expose basin and thoughtseed context, plus collapsible summaries.

  ## Non-Goals
  - No ingestion or schema work (handled by Specs 055 through 057).
  - No LangGraph workflow changes (deferred to Spec 059).

  ## Functional Requirements
  1. Create a responsive side-sheet panel triggered from existing citation links.
  2. Load chunk content by ID and scroll into view on open.
  3. Apply SurfSense-style highlight visuals (background, border, badge).
  4. Include a collapsible summary section and quick filters when available.
  5. Display attractor basin and thoughtseed metadata inline.
  6. Add UI regression coverage or snapshots where feasible.

  ## Technical Notes
  - Keep the legacy document detail page accessible.
  - Reuse existing API payloads when possible.

  ## Dependencies and Sequencing
  - Requires Specs 055, 056, and 057.
  - Unblocks Spec 059.

  ## Parallelization
  - Limited. UI can stage while backend finalizes chunk metadata, but integration waits for
  prior specs.

  ## Acceptance Criteria
  - Citation click opens side-sheet without page navigation.
  - Highlighted chunk is scrolled into view and visually distinct.
  - Consciousness metadata visible in the panel.
  EOF

  mkdir -p specs/059-langgraph-transformations-and-notebook-insights
  cat > specs/059-langgraph-transformations-and-notebook-insights/spec.md <<EOF
  # Spec 059 - LangGraph Transformations and Notebook Insights

  ## Background
  OpenNotebook shows transformational workflows and notebook surfacing we want after the core
  UX lands.

  ## Goals
  - Add a LangGraph-based transformation workflow (summaries, key points, questions).
  - Persist insights and notebook groupings linked to documents and tier state.
  - Surface insights in the UI with toggles to manage runtime cost.

  ## Non-Goals
  - No basic ingestion or citation tweaks (handled earlier).
  - No vector infrastructure overhaul; stay on existing stacks.

  ## Functional Requirements
  1. Implement a LangGraph workflow that orchestrates configurable transformations post-
  ingestion.
  2. Store insights referencing documents, chunks, basins, and thoughtseeds.
  3. Expose notebook-style grouping in the UI with opt-in execution controls.
  4. Provide performance guardrails (max docs per batch, async queue if needed).
  5. Add tests validating workflow execution and data persistence.

  ## Technical Notes
  - Reuse DaedalusGraphChannel and capture telemetry for transformations.
  - Consider a background worker for heavy jobs.

  ## Dependencies and Sequencing
  - Requires Specs 055 through 058 to be complete.

  ## Parallelization
  - After dependencies land, can split into workflow and UI sub-tasks.

  ## Acceptance Criteria
  - Transformations run through LangGraph and persist insights.
  - UI surfaces notebooks and insights with enable/disable controls.
  - End-to-end tests verify insight creation and display.
  EOF

  echo "Specs 055-059 created under specs/"
  
