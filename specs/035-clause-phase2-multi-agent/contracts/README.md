# CLAUSE Phase 2 API Contracts

**Date**: 2025-10-02
**Status**: ✅ Complete - 3 OpenAPI 3.0 specifications

## Contracts

### 1. Navigator API (`navigator_api.yaml`)
- **Endpoint**: POST /api/clause/navigate
- **Purpose**: Budget-aware path exploration with ThoughtSeeds, Curiosity, Causal reasoning
- **Request**: PathNavigationRequest (query, start_node, step_budget, flags)
- **Response**: PathNavigationResponse (path, metadata, performance)

### 2. Curator API (`curator_api.yaml`)
- **Endpoint**: POST /api/clause/curate
- **Purpose**: Listwise evidence selection with token budget and provenance
- **Request**: ContextCurationRequest (evidence_pool, token_budget, lambda_tok)
- **Response**: ContextCurationResponse (selected_evidence with provenance, metadata)

### 3. Coordinator API (`coordinator_api.yaml`)
- **Endpoint**: POST /api/clause/coordinate
- **Purpose**: Multi-agent orchestration (Architect → Navigator → Curator)
- **Request**: CoordinationRequest (query, budgets, lambdas)
- **Response**: CoordinationResponse (combined results, agent_handoffs, conflicts)

## OpenAPI Specification

All contracts follow OpenAPI 3.0.3 standard with:
- ✅ Request/response schemas
- ✅ Validation rules (min/max, patterns)
- ✅ Example payloads
- ✅ Error responses (400, 422, 503)

## Contract Testing

Contract tests validate request/response schemas:

```python
# backend/tests/contract/test_navigator_contract.py
def test_navigator_request_schema():
    request = {
        "query": "What causes climate change?",
        "start_node": "climate_change",
        "step_budget": 10
    }
    # Validate against PathNavigationRequest schema
    assert PathNavigationRequest(**request)

def test_navigator_response_schema():
    response = {...}  # From API
    # Validate against PathNavigationResponse schema
    assert PathNavigationResponse(**response)
```

---
*Contracts complete: 2025-10-02*
