# Placeholder Remediation Plan (Agent-CX)

**Owner**: Agent-CX (Spec 043)  
**Created**: 2025-10-07  
**Branch Workflow**: Each slice executed via dedicated feature branch → TDD → merge to `main`

---

## 1. Response Synthesizer (Spec 006 · FR-013)
- **Gap**: `backend/src/services/response_synthesizer.py` returns template strings / TODO for LLM integration.
- **Target Outcome**:
  - Streaming or batched generation via local Ollama (default) with graceful opt-in for remote models.
  - Inline provenance + confidence per FR-013 acceptance criteria.
- **Test Strategy**:
  - Add failing integration test in `backend/tests/integration/test_query_response.py` asserting answers >200 chars and include citation markers when sources available.
  - Unit tests covering fallback messaging when LLM unavailable.
- **Planned Branch**: `feature/response-synthesizer-llm`.
- **Dependencies**: Confirm Ollama availability; align with `SPEC_DRIVEN_DEVELOPMENT_PROTOCOL.md` + TDD rules.

## 2. Unified Document Processor Adapters (Spec 022 · TR-022-006)
- **Gap**: `_extract_text_pymupdf`, `_extract_structured_langextract`, `_extract_algorithms`, `_extract_knowledge_graph` return placeholder dicts.
- **Target Outcome**:
  - Implement real adapter calls (PyMuPDF, LangExtract, algorithm extractor, KGGen) with structured results + error handling.
  - Provide deterministic fallback stubs that raise actionable exceptions.
- **Test Strategy**:
  - Fixture-based tests in `backend/tests/integration/test_unified_document_processor.py` covering PDF/text sample and asserting non-placeholder payloads.
  - Contract tests verifying schema compatibility before KG ingestion.
- **Planned Branch**: `feature/unified-doc-processor-adapters`.
- **Dependencies**: Confirm availability of extractor binaries; document install steps if needed.

## 3. Backend API Placeholders (Specs 029, 025, 035)
- **Routes Impacted**:
  - `backend/src/api/routes/curiosity.py` (Spec 029).
  - `backend/src/api/routes/visualization.py` (Spec 025).
  - CLAUSE endpoints in `backend/src/api/routes/clause.py` (Spec 035).
- **Target Outcome**:
  - Replace placeholder responses with real service calls (curiosity queue, visualization stream, CLAUSE services).
  - Ensure error handling and dependency injection align with existing service layers.
- **Test Strategy**:
  - API contract tests using FastAPI test client verifying JSON schema + real data flows (`backend/tests/interface/`).
  - Mock underlying services where necessary for TDD (start with failing tests hitting TODO responses).
- **Planned Branches**:
  - `feature/curiosity-api-implementation`
  - `feature/visualization-websocket`
  - `feature/clause-endpoints`
- **Dependencies**: Confirm Redis/Neo4j services running for CLAUSE and curiosity flows.

## 4. Frontend Mock Data Removal (Specs 029, 038, 025)
- **Components**:
  - `frontend/src/pages/KnowledgeGraph.tsx`
  - `frontend/src/pages/CuriosityMissions.tsx`
  - `frontend/src/components/VisualizationStream.tsx`
- **Target Outcome**:
  - Replace `setTimeout` mock loaders with real API/WebSocket integration.
  - Align UI with backend schema + Spec 038 chips for curiosity triggers.
- **Test Strategy**:
  - Cypress or Playwright E2E flows once backend endpoints live.
  - React component tests verifying data rendering from fetch mocks.
- **Planned Branch**: `feature/frontend-live-data` (may split per page if scope grows).
- **Dependencies**: Backend endpoints above; WebSocket connection stability.

## 5. Clause Service Placeholders (Spec 035)
- **Gap**: `backend/src/services/clause/edge_scorer.py`, `lc_mappo_coordinator.py`, and others contain `pass`/TODO blocks.
- **Target Outcome**:
  - Implement signal calculations per Spec 035 (edge scoring, LC-MAPPO coordination).
  - Ensure coordination service integrates with curiosity triggers and causal reasoning.
- **Test Strategy**:
  - Unit tests for scoring functions with fixture graphs.
  - Integration tests simulating navigation requests.
- **Planned Branch**: `feature/clause-services`.
- **Dependencies**: Review Spec 035 data models + existing partial implementations.

## 6. Sentence Transformer Fallback Transparency (Spec 022 & Constitution)
- **Gap**: `extensions/context_engineering/unified_database.py` fallback raises NotImplementedError without user guidance beyond install message.
- **Target Outcome**:
  - Provide CLI-friendly warnings, optional deterministic vector stub for development mode.
  - Ensure constitution compliance checks for embeddings before operations.
- **Test Strategy**:
  - Unit tests mocking ImportError path verifying warning + exception content.
- **Planned Branch**: `feature/sentence-transformer-fallback`.

---

## Execution Cadence
1. Finalize and circulate this plan via coordination docs (done).
2. Prioritize Response Synthesizer + Document Processor adapters (highest external impact).
3. Schedule branch creation + failing tests within next coordination cycle.
4. Maintain status updates every 15 minutes while actively modifying files.

---

## References
- AGENT_CONSTITUTION.md
- SPEC_DRIVEN_DEVELOPMENT_PROTOCOL.md
- TDD_RULES.md
- Specs: 006, 022, 025, 029, 035, 038, 043

