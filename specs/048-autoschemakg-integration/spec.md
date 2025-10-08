# Spec 048: Fix AutoSchemaKG Knowledge Graph Integration

**Status**: CRITICAL - Integration Missing
**Created**: 2025-10-07
**Priority**: P0 - Blocks all document processing

## Problem Statement

**DISCOVERED BUG**: The document processing pipeline bypasses AutoSchemaKG and writes directly to Neo4j, violating system architecture and months of integration work.

### What Should Happen
```
Document Upload → Daedalus Gateway → DocumentProcessingGraph
  → FiveLevelConceptExtraction (5 concept levels)
  → AutoSchemaKG (automatic knowledge graph construction)
  → MultiTierMemory (warm/cool/cold storage)
  → Neo4j (via unified schema)
```

### What Actually Happens
```
Document Upload → Daedalus Gateway → DocumentProcessingGraph
  → ConsciousnessDocumentProcessor (basic concepts)
  → Neo4jUnifiedSchema (DIRECT WRITE) ❌
```

### Evidence
- `document_processing_graph.py:27` imports `Neo4jUnifiedSchema` directly
- NO imports of `AutoSchemaKGService`, `FiveLevelConceptExtractionService`, or `MultiTierMemorySystem`
- `autoschemakg_integration.py` exists (1000+ lines) but is NOT imported anywhere
- Tests confirm: All 9 AutoSchemaKG integration tests FAIL

## Functional Requirements

### FR1: Use FiveLevelConceptExtraction
**MUST** extract concepts at five levels, not flat list:
- Level 1: Atomic concepts (individual entities)
- Level 2: Relationships (connections between concepts)
- Level 3: Composite concepts (conceptual groups)
- Level 4: Context (domain/topic framing)
- Level 5: Narrative (high-level story/theme)

**Success Criteria**:
- `DocumentProcessingGraph.__init__` creates `FiveLevelConceptExtractionService` instance
- Processing result contains `concept_hierarchy` with all 5 levels
- Test `test_concept_extraction_five_levels` PASSES

### FR2: Use AutoSchemaKG for Graph Construction
**MUST** use AutoSchemaKG to automatically construct knowledge graph from extracted concepts:
- Create nodes for each concept level (atomic, composite, context, narrative)
- Infer relationships between concepts (RELATES_TO, CONTAINS, PART_OF, etc.)
- Assign confidence scores to nodes and relationships
- Generate embeddings for semantic similarity

**Success Criteria**:
- `DocumentProcessingGraph.__init__` creates `AutoSchemaKGService` instance
- Processing result contains `knowledge_graph` with nodes and relationships
- Test `test_autoschemakg_service_exists_and_works` PASSES
- Test `test_pdf_processing_creates_knowledge_graph_nodes` PASSES

### FR3: Use MultiTierMemorySystem for Storage
**MUST** store across three memory tiers:
- **Warm Tier** (Neo4j): Active knowledge graph, frequently accessed
- **Cool Tier** (Vector DB): Semantic embeddings for similarity search
- **Cold Tier** (Archive): Long-term storage, infrequently accessed

**Success Criteria**:
- `DocumentProcessingGraph.__init__` creates `MultiTierMemorySystem` instance
- Processing result contains `memory_storage` with tier information
- Test `test_memory_tiers_used_for_storage` PASSES

### FR4: Never Bypass Daedalus Gateway
**CONSTITUTIONAL REQUIREMENT**: All external data MUST enter through Daedalus gateway.

**Success Criteria**:
- No direct database writes from routes/controllers
- All document processing goes through `Daedalus.receive_perceptual_information()`
- Constitution updated with this rule

### FR5: Never Write Directly to Databases
**CONSTITUTIONAL REQUIREMENT**: Services MUST use abstraction layers, NEVER direct database connections.

**Success Criteria**:
- Use `AutoSchemaKGService` instead of direct Neo4j writes
- Use `MultiTierMemorySystem` instead of direct Qdrant/PostgreSQL writes
- Constitution updated with this rule

## Success Metrics

- ✅ All 9 AutoSchemaKG tests pass
- ✅ Document processing uses 5-level extraction
- ✅ Knowledge graph nodes created with proper types
- ✅ Memory tiers populated correctly
- ✅ Constitution updated and enforced
- ✅ Zero direct database writes in codebase

## References

- Implementation: `backend/src/services/autoschemakg_integration.py`
- Tests: `backend/tests/integration/test_autoschemakg_document_processing.py`
- Constitution: `AGENT_CONSTITUTION.md`
